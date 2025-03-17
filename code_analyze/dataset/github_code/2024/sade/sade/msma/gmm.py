import logging
import os
import sys
from datetime import datetime

import torch
from torch.utils import tensorboard
from torchinfo import summary
from tqdm import tqdm

import sade.models.registry as registry
from sade.datasets.loaders import get_dataloaders
from sade.models.distributions import GMM
from sade.models.ema import ExponentialMovingAverage
from sade.run.utils import restore_pretrained_weights


def train(config, workdir, n_components=10, kimg=100, lr=3e-4):
    log_tensorboard = config.flow.log_tensorboard
    log_interval = config.training.log_freq
    device = config.device

    # Forcing the number of timesteps to be 10
    # Otherwise, the model will be too large for GradCam
    config.msma.n_timesteps = 10

    # Initialize score model
    score_model = registry.create_model(config, log_grads=False)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    # Get the score model checkpoint from pretrained run
    state = restore_pretrained_weights(
        config.training.pretrained_checkpoint, state, config.device
    )
    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(config, score_model, return_norm=True)

    # Initialize GMM model
    gmm = GMM(n_components=n_components, n_features=config.msma.n_timesteps).to(device)
    summary(gmm, depth=0, verbose=2)

    # Build data iterators
    dataloaders, _ = get_dataloaders(
        config,
        evaluation=False,
        num_workers=4,
        infinite_sampler=True,
    )

    train_dl, eval_dl, _ = dataloaders
    train_iter = iter(train_dl)
    eval_iter = iter(eval_dl)

    run_dir = os.path.join(workdir, "msma")
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Saving checkpoints to {run_dir}")
    gmm_checkpoint_path = f"{run_dir}/gmm_checkpoint.pth"

    # Set logger so that it outputs to both console and file
    gfile_stream = open(os.path.join(run_dir, "stdout.txt"), "w")
    file_handler = logging.StreamHandler(gfile_stream)
    stdout_handler = logging.StreamHandler(sys.stdout)

    # Override root handler
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
        handlers=[file_handler, stdout_handler],
    )
    if log_tensorboard:
        t = datetime.now().strftime("%Y-%m-%d_%H:%M")
        writer = tensorboard.SummaryWriter(log_dir=f"{run_dir}/logs/{t}")

    # Defining optimizer
    opt = torch.optim.AdamW(gmm.parameters(), lr=lr, weight_decay=1e-5)

    losses = []
    batch_sz = config.training.batch_size
    total_iters = kimg * 1000 // batch_sz + 1
    logging.info("Starting training for iters: %d", total_iters)
    niter = 0
    imgcount = 0
    best_val_loss = torch.inf
    checkpoint_interval = 10
    loss_dict = {}
    gmm.train()
    progbar = tqdm(range(total_iters))

    for niter in progbar:
        x_batch = next(train_iter)["image"].to(device)
        scores = scorer(x_batch)
        loss = -gmm(scores).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_dict["train_loss"] = loss.item()
        imgcount += x_batch.shape[0]

        if niter % log_interval == 0:
            gmm.eval()

            with torch.no_grad():
                x = next(eval_iter)["image"].to(device)
                x = scorer(x)
                val_loss = -gmm(x).mean()
                loss_dict["val_loss"] = val_loss.item()

            progbar.set_description(f"Val Loss: {val_loss:.4f}")
            losses.append(val_loss)
            gmm.train()

        if log_tensorboard:
            for loss_type in loss_dict:
                writer.add_scalar(f"loss/{loss_type}", loss_dict[loss_type], niter)

        progbar.set_postfix(batch=f"{imgcount}/{kimg}K")

        if niter % checkpoint_interval == 0 and val_loss < best_val_loss:
            # if the current validation loss is the best one
            best_val_loss = val_loss  # Update the best validation loss

            torch.save(
                {
                    "kimg": niter,
                    "model_state_dict": gmm.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": val_loss,
                },
                gmm_checkpoint_path,
            )

    # Recall that Val set does not use augmentations
    progbar = tqdm(range(1000))
    for refine_iter in progbar:
        x_batch = next(eval_iter)["image"].to(device)
        scores = scorer(x_batch)
        loss = -gmm(scores).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_dict["val_loss"] = loss.item()
        if log_tensorboard:
            for loss_type in loss_dict:
                writer.add_scalar(
                    f"loss/{loss_type}", loss_dict[loss_type], total_iters + niter
                )

    torch.save(
        {
            "kimg": niter + refine_iter,
            "model_state_dict": gmm.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_loss": loss,
        },
        gmm_checkpoint_path,
    )

    if log_tensorboard:
        writer.close()

    return losses


if __name__ == "__main__":
    from sade.configs.ve import biggan_config

    workdir = sys.argv[1]

    config = biggan_config.get_config()
    config.data.cache_rate = 1.0
    config.fp16 = False
    config.training.batch_size = 32
    config.training.log_freq = 2
    config.eval.batch_size = 64

    train(config, workdir, kimg=200, lr=1e-3)
