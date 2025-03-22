import logging
import os
import sys
from datetime import datetime

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal
from torch.utils import tensorboard
from torchinfo import summary
from tqdm import tqdm

import sade.models.registry as registry
from sade.datasets.loaders import get_dataloaders
from sade.models.ema import ExponentialMovingAverage
from sade.models.flows import gaussian_logprob
from sade.run.utils import restore_pretrained_weights


@torch.jit.script
def gaussian_logprob(z, ldj):
    _GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))
    return _GCONST_ - 0.5 * torch.sum(z**2, dim=-1) + ldj


def subnet_fc(c_in, c_out, ndim=256):
    return nn.Sequential(
        nn.Linear(c_in, ndim),
        nn.LayerNorm(ndim),
        nn.LeakyReLU(0.2),
        nn.Linear(ndim, ndim),
        nn.LayerNorm(ndim),
        nn.LeakyReLU(0.2),
        nn.Linear(ndim, c_out),
    )


def build_nd_flow(input_dim, num_blocks=20):
    inn = Ff.SequenceINN(input_dim)
    for k in range(num_blocks):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    return inn


def nll(zs, log_jac_dets):
    return -torch.mean(gaussian_logprob(zs, log_jac_dets))


class MultivariateNormal(nn.Module):
    def __init__(self, input_dims, low_rank_dims=128):
        super().__init__()

        self.cov_factor = nn.Parameter(torch.randn(input_dims, low_rank_dims), requires_grad=True)
        self.cov_diag = nn.Parameter(torch.ones(input_dims), requires_grad=True)
        self.mean = nn.Parameter(torch.zeros(input_dims), requires_grad=False)

    def forward(self, x):
        return LowRankMultivariateNormal(
            loc=self.mean, cov_factor=self.cov_factor, cov_diag=self.diag
        ).log_prob(x)
    
    @property
    def diag(self):
        return torch.nn.functional.softplus(self.cov_diag) + 1e-5

    @property
    def covariance_matrix(self):
        return LowRankMultivariateNormal(
            loc=self.mean, cov_factor=self.cov_factor, cov_diag=self.diag
        ).covariance_matrix
    
class FlowModel(nn.Module):
    def __init__(self, n_timesteps, num_blocks=20, device="cpu"):
        super().__init__()
        self.flow = build_nd_flow(n_timesteps, num_blocks)
        self.base_distribution = MultivariateNormal(n_timesteps)
        self.init_weights()
        self.to(device)

    def init_weights(self):
        # Initialize weights with Xavier
        linear_modules = list(filter(lambda m: isinstance(m, nn.Linear), self.flow.modules()))
        total = len(linear_modules)
        # pdb.set_trace()
        for idx, m in enumerate(linear_modules):
            
            # Last layer gets init w/ zeros
            if idx == total - 1:
                nn.init.zeros_(m.weight.data)
            else:
                nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        z, ldj = self.flow(x)
        return self.base_distribution(z) + ldj

    @torch.inference_mode()
    def score(self, x):
        return -self.forward(x)
    



def train(config, workdir):
    kimg = config.flow.training_kimg
    log_tensorboard = config.flow.log_tensorboard
    lr = config.flow.lr
    log_interval = 10  # config.training.log_freq
    device = config.device

    # Forcing the number of timesteps to be 10
    # Otherwise, the model will be too large for GradCam
    config.msma.n_timesteps = 10

    # Initialize score model
    score_model = registry.create_model(config, log_grads=False)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    # Get the score model checkpoint from pretrained run
    checkpoint_paths = os.path.join(config.training.pretrain_dir, "checkpoint.pth")
    state = restore_pretrained_weights(checkpoint_paths, state, config.device)
    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(config, score_model, return_norm=True)

    # Initialize flow model
    flownet = FlowModel(config.msma.n_timesteps, device=device)

    summary(flownet, depth=0, verbose=2)

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
    flow_checkpoint_path = f"{run_dir}/checkpoint.pth"

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

    # Main copy to be used for "fast" weight updates
    flownet = flownet.to(device)

    # Model will be updated with EMA weights
    ema = ExponentialMovingAverage(flownet.parameters(), decay=0.99)
    # Defining optimizer
    opt = torch.optim.AdamW(flownet.parameters(), lr=lr, weight_decay=1e-5)

    losses = []
    batch_sz = config.training.batch_size
    total_iters = kimg * 1000 // batch_sz + 1
    logging.info("Starting training for iters: %d", total_iters)
    niter = 0
    imgcount = 0
    best_val_loss = np.inf
    checkpoint_interval = 100
    loss_dict = {}
    flownet.train()
    progbar = tqdm(range(total_iters))
    for niter in progbar:
        x_batch = next(train_iter)["image"].to(device)
        scores = scorer(x_batch)
        loss = -flownet(scores).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 5.0)
        opt.step()

        loss_dict["train_loss"] = loss.item()
        imgcount += x_batch.shape[0]

        # Update EMA
        ema.update(flownet.parameters())

        if niter % log_interval == 0:
            flownet.eval()
            ema.store(flownet.parameters())
            ema.copy_to(flownet.parameters())
            with torch.no_grad():
                x = next(eval_iter)["image"].to(device)
                x = scorer(x)
                val_loss = -flownet(scores).mean()
                loss_dict["val_loss"] = val_loss.item()

            progbar.set_description(f"Val Loss: {val_loss:.4f}")
            losses.append(val_loss)
            ema.restore(flownet.parameters())
            flownet.train()

        if log_tensorboard:
            for loss_type in loss_dict:
                writer.add_scalar(f"loss/{loss_type}", loss_dict[loss_type], niter)

        progbar.set_postfix(batch=f"{imgcount}/{kimg}K")

        if niter % checkpoint_interval == 0 and val_loss < best_val_loss:
            # if the current validation loss is the best one
            best_val_loss = val_loss  # Update the best validation loss

            ema.store(flownet.parameters())
            ema.copy_to(flownet.parameters())

            torch.save(
                {
                    "kimg": niter,
                    "model_state_dict": flownet.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": val_loss,
                },
                flow_checkpoint_path,
            )

            ema.restore(flownet.parameters())

    # TODO: A few iterations on the val set to get the best model
    # Recall that Val set does not use augmentations
    progbar = tqdm(range(100))
    for niter in progbar:
        x_batch = next(eval_iter)["image"].to(device)
        scores = scorer(x_batch)
        loss = -flownet(scores).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 2.0)
        opt.step()

        ema.update(flownet.parameters())

        loss_dict["val_loss"] = loss.item()
        if log_tensorboard:
            for loss_type in loss_dict:
                writer.add_scalar(
                    f"loss/{loss_type}", loss_dict[loss_type], total_iters + niter
                )

    ema.copy_to(flownet.parameters())
    torch.save(
        {
            "kimg": niter,
            "model_state_dict": flownet.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_loss": loss,
        },
        flow_checkpoint_path,
    )

    if log_tensorboard:
        writer.close()

    return losses


if __name__ == "__main__":
    from sade.configs.ve import biggan_config

    workdir = sys.argv[1]

    config = biggan_config.get_config()
    config.fp16 = True
    config.training.batch_size = 32
    config.eval.batch_size = 32
    config.flow.training_kimg = 50

    train(config, workdir)
