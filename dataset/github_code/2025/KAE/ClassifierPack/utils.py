from ExpToolKit.models import BaseAE
import torch
import numpy as np


def dimension_reduction(
    model: BaseAE | None = None,
    x: torch.Tensor = None,
) -> np.ndarray:
    """
    Reduce the dimension of the given data.

    Parameters
    ----------
    model : BaseAE | None
        The model to be used for dimension reduction.
        If None, the raw data is used.
    x : torch.Tensor
        The data to be reduced.

    Returns
    -------
    np.ndarray
        The reduced data.

    Raises
    ------
    ValueError
        If x is None or model is not supported.
    """
    if x is None:
        raise ValueError("x is None")

    if isinstance(model, BaseAE):
        model.eval()
        model.to("cpu")
        return model.encode(x).detach().numpy()
    elif model is None:
        x = x.reshape(x.shape[0], -1)
        return x.detach().numpy()
    else:
        raise ValueError("Unknown model type")
