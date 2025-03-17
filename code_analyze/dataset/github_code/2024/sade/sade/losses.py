"""All functions related to loss computation.
"""
import torch
from sade.models.registry import get_score_fn


def get_sde_loss_fn(
    sde,
    train,
    reduce_mean=True,
    likelihood_weighting=False,
    amp=False,
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.

    Returns:
      A loss function.
    """

    # For numerical stability - the smallest time step to sample from.
    eps = 1e-5
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch, log_alpha=False):
        """Compute the loss function.

        Args:
        model: A score model.
        batch: A mini-batch of training data.

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = get_score_fn(sde, model, train=train, amp=amp)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + sde._unsqueeze(std) * z
        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
            losses = torch.square(score * sde._unsqueeze(std) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / sde._unsqueeze(std))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_scorer(sde, eps=1e-5):
    def scorer(model, batch):
        """Compute the weighted scores function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          score: A tensor that represents the weighted scores for each sample in the mini-batch.
        """
        score_fn = get_score_fn(sde, model, train=False)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        score = score_fn(batch, t)
        return score

    return scorer


def score_step_fn(sde, eps=1e-5):
    scorer = get_scorer(
        sde,
    )

    def step_fn(state, batch):
        """Running one step of scoring

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          score: The average loss value of this state.
        """
        model = state["model"]
        ema = state["ema"]
        ema.copy_to(model.parameters())
        with torch.no_grad():
            score = scorer(model, batch)
        return score

    return step_fn
