import functools
import logging
import os

import models.registry as registry
import numpy as np
import torch
import wandb
from datasets.loaders import get_dataloaders
from models.ema import ExponentialMovingAverage
from optim import get_diagnsotic_fn, get_step_fn, optimization_manager
from torch.utils import tensorboard

from .sampling import get_sampling_fn
from .utils import plot_slices, restore_pretrained_weights, save_checkpoint

makedirs = functools.partial(os.makedirs, exist_ok=True)


def finetuner(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize main model.
    fast_model = registry.create_model(config, print_summary=True)
    sde = registry.create_sde(config)

    # Always keep the latest updated weights
    ema = ExponentialMovingAverage(fast_model.parameters(), decay=0)

    state = dict(
        model=fast_model,
        ema=ema,
        step=0,
    )

    # Initialize optimization state
    optimize_fn = optimization_manager(state, config.finetuning)
    assert "optimizer" in state, "Optimizer not found in state!"

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    makedirs(checkpoint_dir)

    # Resume training from a checkpoint
    pretrain_dir = os.path.join(config.training.pretrain_dir, "checkpoint.pth")
    state = restore_pretrained_weights(pretrain_dir, state, config.device)

    # Build data iterators
    dataloaders, datasets = get_dataloaders(
        config,
        evaluation=False,
        num_workers=4,
        infinite_sampler=True,
    )

    train_dl, eval_dl, _ = dataloaders
    train_iter = iter(train_dl)
    eval_iter = iter(eval_dl)

    # Build one-step training and evaluation functions
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting

    train_step_fn = get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        likelihood_weighting=likelihood_weighting,
        use_fp16=config.fp16,
    )

    eval_step_fn = get_step_fn(
        sde,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        likelihood_weighting=likelihood_weighting,
        use_fp16=config.fp16,
    )

    diagnsotic_step_fn = get_diagnsotic_fn(
        sde,
        reduce_mean=reduce_mean,
        likelihood_weighting=likelihood_weighting,
        use_fp16=config.fp16,
    )

    sampling_shape = (
        config.eval.sample_size,
        config.data.num_channels,
        *config.data.image_size,
    )
    sampling_fn = get_sampling_fn(config, sde, sampling_shape)

    # These will be the 'global' model weights that will be updated slowly
    slow_model = ExponentialMovingAverage(
        state["model"].parameters(), decay=1 - config.finetuning.outer_step_size
    )

    num_finetune_steps = config.finetuning.n_iters
    num_fast_steps = config.finetuning.n_fast_steps

    logging.info(f"Starting finetuning loop for {num_finetune_steps:d} iters...")

    for step in range(num_finetune_steps + 1):
        # Execute fast weight updates
        loss = 0.0
        for _ in range(num_fast_steps):
            batch = next(train_iter)["image"].to(config.device)
            loss += train_step_fn(state, batch).item()
        loss /= num_fast_steps

        # Execute slow weight updates
        slow_model.update(state["model"].parameters())

        # Update main model with slow weights
        slow_model.copy_to(state["model"].parameters())
        state["ema"] = ExponentialMovingAverage(state["model"].parameters(), decay=0)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss))
            writer.add_scalar("training_loss", loss, step)
            wandb.log({"loss": loss}, step=step)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_loss = 0.0
            sigma_losses = {}

            eval_batch = next(eval_iter)["image"].to(config.device)
            eval_loss = eval_step_fn(state, eval_batch).item()

            per_sigma_loss = diagnsotic_step_fn(state, eval_batch)
            for sigma, (loss, norms) in per_sigma_loss.items():
                if sigma not in sigma_losses:
                    sigma_losses[sigma] = (loss, norms)
                else:
                    l, n = sigma_losses[sigma]
                    l += loss
                    n = torch.cat((n, norms))
                    sigma_losses[sigma] = (l, n)

            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
            writer.add_scalar("eval_loss", eval_loss, step)
            wandb.log({"val_loss": eval_loss}, step=step)

            for t, (sigma_loss, sigma_norms) in sigma_losses.items():
                logging.info(f"\t\t\t t: {t}, eval_loss:{ sigma_loss:.5f}")
                writer.add_scalar(f"eval_loss/{t}", sigma_loss, step)

                wandb.log({f"val_loss/{t}": sigma_loss}, step=step)
                wandb.log(
                    {f"score_dist/{t}": wandb.Histogram(sigma_norms.numpy())},
                    step=step,
                )

        # Save a checkpoint periodically
        if step % config.training.snapshot_freq == 0:
            # Save the checkpoint.
            logging.info(f"step: {step}, saving checkpoint...")
            save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint-meta.pth"), state)

        # Generate samples periodically
        if (step + 1) % (config.training.sampling_freq) == 0:
            logging.info("step: %d, generating samples..." % (step))
            sample, n = sampling_fn(state["model"])
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            makedirs(this_sample_dir)
            sample = sample.permute(0, 2, 3, 4, 1).cpu().numpy()
            logging.info("step: %d, done!" % (step))

            with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                np.save(fout, sample)

            fname = os.path.join(this_sample_dir, "sample.png")

            try:
                plot_slices(sample, fname)
                wandb.log({"sample": wandb.Image(fname)})
            except:
                logging.warning("Plotting failed!")

    # Save the final checkpoint.
    save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{step}.pth"), state)

    # Generate and save samples

    logging.info("step: %d, generating samples..." % (step))
    sample, n = sampling_fn(state["model"])
    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
    makedirs(this_sample_dir)
    sample = sample.permute(0, 2, 3, 4, 1).cpu().numpy()
    logging.info("step: %d, done!" % (step))

    with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
        np.save(fout, sample)

    fname = os.path.join(this_sample_dir, "sample.png")

    try:
        plot_slices(sample, fname)
        wandb.log({"sample": wandb.Image(fname)})
    except:
        logging.warning("Plotting failed!")
