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
from .utils import (
    plot_slices,
    restore_checkpoint,
    restore_pretrained_weights,
    save_checkpoint,
)

makedirs = functools.partial(os.makedirs, exist_ok=True)


def trainer(config, workdir):
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

    # Initialize model.
    score_model = registry.create_model(config, print_summary=True)
    sde = registry.create_sde(config)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

    state = dict(
        model=score_model,
        ema=ema,
        step=0,
    )

    # Initialize optimization state
    optimize_fn = optimization_manager(state, config)
    assert "optimizer" in state, "Optimizer not found in state!"

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    makedirs(checkpoint_dir)
    makedirs(os.path.dirname(checkpoint_meta_dir))

    # # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    step = initial_step = int(state["step"])

    # Used for accumulating gradients
    state["train-step"] = 0

    if initial_step == 0 and config.training.load_pretrain:
        state = restore_pretrained_weights(
            config.training.pretrained_checkpoint, state, config.device
        )

    # Build data iterators
    dataloaders, datasets = get_dataloaders(
        config,
        evaluation=False,
        ood_eval=False,
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
        gradient_accumulation_factor=config.training.grad_accum_factor,
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

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.eval.sample_size,
            config.data.num_channels,
            *config.data.image_size,
        )
        print(f"Sampling shape: {sampling_shape}")
        sampling_fn = get_sampling_fn(config, sde, sampling_shape)

    num_train_steps = config.training.n_iters
    logging.info("Starting training loop at step %d." % (initial_step,))

    while step < num_train_steps:
        batch = next(train_iter)["image"].to(config.device)

        # Execute one training step
        step = state["step"]
        loss = train_step_fn(state, batch)
        loss = loss.item()

        # If still grad accumulating, step will not change
        if step == state["step"]:
            continue

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss))
            writer.add_scalar("training_loss", loss, step)
            wandb.log({"loss": loss}, step=step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())

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

            ema.restore(score_model.parameters())

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

            if config.optim.scheduler != "skip":
                wandb.log(
                    {
                        "lr": state["optimizer"].param_groups[0]["lr"],
                    },
                    step=step,
                )

        # Save a checkpoint periodically and generate samples if needed
        if (
            step != 0
            and step % config.training.snapshot_freq == 0
            or step == num_train_steps
        ):
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"), state
            )

        # Generate and save samples
        if (
            step != 0
            and config.training.snapshot_sampling
            and step % config.training.sampling_freq == 0
        ):
            logging.info("step: %d, generating samples..." % (step))
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model)
            ema.restore(score_model.parameters())
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
