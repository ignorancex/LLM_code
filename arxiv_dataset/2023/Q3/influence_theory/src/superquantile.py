"""From the SQwash API (https://krishnap25.github.io/sqwash/)
    Used to calculate most influential subset (MIS), run_MIS_exp.py
"""

import torch


def reduce_superquantile(batch: torch.Tensor, superquantile_tail_fraction: float = 0.5) -> torch.Tensor:
    """Reduce a batch of values by their superquantile (a.k.a. Conditional Value at Risk, CVaR).

    See the class :class:`SuperquantileReducer` for details.

    :param batch: Tensor of values to reduce. Reshaped into a single dimension.
    :type batch: torch.Tensor
    :param superquantile_tail_fraction: What fraction of the tail to average over for the superquantile. 
        If 1.0, the function returns `batch.mean()` and if 0.0, the function returns `batch.max()`.
    :type superquantile_tail_fraction: float, default is 0.5

    Examples::

        for x, y in dataloader: 
            y_hat = model(x)
            batch_loss = torch.nn.functional.cross_entropy(y_hat, y, reduction='none')  # must set `reduction='none'`
            loss = reduce_superquantile(batch_loss, superquantile_tail_fraction=0.6) 
            loss.backward()  # Proceed as usual from here
            ...
    """
    if not (0 <= superquantile_tail_fraction <= 1):
        raise ValueError(
            f"superquantile_tail_fraction must be between 0 and 1 (got {superquantile_tail_fraction}.")
    batch = batch.view(-1)
    n = batch.shape[0]
    sup = torch.max(batch)
    quantile = torch.quantile(batch, 1 - superquantile_tail_fraction)
    if n == 1 or superquantile_tail_fraction == 1.0:
        return torch.mean(batch)
    elif superquantile_tail_fraction < 1 / n or quantile == sup:
        return sup
    else:
        tail_mean = torch.mean(batch[batch > quantile])
        w = (batch > quantile).sum() / (n * superquantile_tail_fraction)
        return w * tail_mean + (1 - w) * quantile
