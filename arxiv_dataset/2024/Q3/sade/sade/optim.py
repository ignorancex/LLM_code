import logging

import numpy as np
import torch
import torch.optim as optim

from sade.losses import get_sde_loss_fn
from sade.models.registry import get_score_fn

avail_optimizers = {
    "Adam": optim.Adam,
    "Adamax": optim.Adamax,
    "AdamW": optim.AdamW,
    "RAdam": optim.RAdam,
}


def get_optimizer(config, params):
    """Returns an optimizer object based on `config`."""
    if config.optim.optimizer in avail_optimizers:
        opt = avail_optimizers[config.optim.optimizer]

        optimizer = opt(
            params,
            lr=config.optim.lr,
            betas=(
                config.optim.beta1,
                0.999,
            ),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not supported yet!")

    return optimizer


def get_scheduler(config, optimizer):
    """Returns a scheduler object based on `config`."""

    if config.optim.scheduler == "skip":
        return None

    if config.optim.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(0.4 * config.training.n_iters),
            gamma=0.3,
            verbose=False,
        )

    if config.optim.scheduler == "cosine":
        # Assumes LR in opt is initial learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.n_iters,
            eta_min=1e-6,
        )

    logging.info(f"Using scheduler: {scheduler.__class__}")
    return scheduler


def optimization_manager(state_dict, config):
    """
    Populates the state with optimization related objects based on `config`.
    Returns an optimize_fn.
    """
    assert "model" in state_dict, "state_dict must contain a model"
    optimizer = get_optimizer(config, state_dict["model"].parameters())
    scheduler = get_scheduler(config, optimizer)
    grad_scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
    optimizer.zero_grad()

    state_dict["optimizer"] = optimizer
    if scheduler is not None:
        state_dict["scheduler"] = scheduler
    if grad_scaler is not None:
        state_dict["grad_scaler"] = grad_scaler

    def optimize_fn(
        params,
        step,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
    ):
        """
        Optimizes with warmup and gradient clipping (disabled if negative).
        Scheduler and mixed precision are handled here as well.
        """
        if step <= warmup:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=grad_clip,
            )

        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            # grad_scaler.update(
            #     new_scale=2.0**8 if grad_scaler.get_scale() > 2.0**10 else None
            # )
            grad_scaler.update()
        else:
            optimizer.step()

        if step > warmup and scheduler is not None:
            scheduler.step()
        
        optimizer.zero_grad()

    return optimize_fn


def get_step_fn(
    sde,
    train,
    optimize_fn=None,
    reduce_mean=False,
    likelihood_weighting=False,
    use_fp16=False,
    gradient_accumulation_factor=1,
):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """

    loss_fn = get_sde_loss_fn(
        sde,
        train,
        reduce_mean=reduce_mean,
        likelihood_weighting=likelihood_weighting,
        amp=use_fp16,
    )

    if use_fp16:
        print(f"Using AMP for {'training' if train else 'evaluation'}.")

        def step_fn(state, batch):
            """Running one step of training or evaluation with AMP"""
            model = state["model"]
            if train:
                optimizer = state["optimizer"]
                loss_scaler = state["grad_scaler"]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch)

                loss_scaler.scale(loss).backward()
                optimize_fn(
                    model.parameters(),
                    step=state["step"],
                )
                state["step"] += 1
                if not torch.isnan(loss):
                    state["ema"].update(model.parameters())
            else:
                with torch.inference_mode():
                    loss = loss_fn(model, batch)

            return loss

    else:

        def step_fn(state, batch):
            """Running one step of training or evaluation.

            Args:
            state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data.

            Returns:
            loss: The average loss value of this state.
            """
            model = state["model"]
            if train:
                loss = loss_fn(model, batch)
                loss.backward()

                if state['train-step'] % gradient_accumulation_factor == 0:
                    optimize_fn(
                        model.parameters(),
                        step=state["step"],
                    )
                    state["ema"].update(model.parameters())
                    state["step"] += 1
                
                state['train-step'] += 1
            else:
                with torch.no_grad():
                    loss = loss_fn(model, batch)

            return loss

    return step_fn


def get_diagnsotic_fn(
    sde,
    reduce_mean=False,
    likelihood_weighting=False,
    eps=1e-5,
    steps=5,
    use_fp16=False,
):
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def sde_loss_fn(model, batch, t):
        """Compute the per-sigma loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = get_score_fn(sde, model, train=False, amp=use_fp16)
        _t = torch.ones(batch.shape[0], device=batch.device) * t * (sde.T - eps) + eps

        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, _t)
        perturbed_data = mean + sde._unsqueeze(std) * z

        score = score_fn(perturbed_data, _t)
        score_norms = torch.linalg.norm(score.reshape((score.shape[0], -1)), dim=-1)
        score_norms = score_norms * std

        if not likelihood_weighting:
            losses = torch.square(score * sde._unsqueeze(std) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), _t)[1] ** 2
            losses = torch.square(score + z / sde._unsqueeze(std))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)

        return loss, score_norms

    final_timepoint = 1.0
    loss_fn = sde_loss_fn

    def step_fn(state, batch):
        model = state["model"]
        with torch.no_grad():
            # ema = state["ema"]
            # ema.store(model.parameters())
            # ema.copy_to(model.parameters())

            losses = {}

            for t in torch.linspace(0.0, final_timepoint, steps, dtype=torch.float32):
                loss, norms = loss_fn(model, batch, t)
                losses[f"{t:.3f}"] = (loss.item(), norms.cpu())

            # ema.restore(model.parameters())

        return losses

    return step_fn
