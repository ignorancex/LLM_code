import torch
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np

from ExpToolKit.models import BaseAE


def dimension_reduction(
    model: BaseAE | None = None,
    x: np.ndarray | torch.Tensor = None,
) -> np.ndarray:
    """
    Reduce the dimension of the given data.

    Parameters
    ----------
    model : BaseAE | Reducer | None
        The model to be used for dimension reduction.
        If None, the raw data is used.
    x : np.ndarray | torch.Tensor
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


def get_x(
    model: BaseAE | None, dataloader: DataLoader, label_num: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the data and latent data from the given dataloader.

    Parameters
    ----------
    model : BaseAE | Reducer | None
        The model to be used for dimension reduction.
        If None, the raw data is used.
    dataloader : DataLoader
        The DataLoader of the dataset to be processed.
    label_num : int, optional
        The number of data points to be processed for each label, by default 100.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The data and latent data, respectively.
    """
    x_list = []
    x_latent_list = []
    y_list = []
    for x_, y_ in dataloader:
        x_: torch.Tensor
        x_ = x_.reshape(x_.shape[0], -1)
        x_list.append(x_.detach().numpy())
        x_latent_list.append(dimension_reduction(model=model, x=x_))
        y_list.append(y_)
    x = np.concatenate(x_list, axis=0)
    x_latent = np.concatenate(x_latent_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    x_dict: dict = {}
    x_latent_dict: dict = {}
    for i, label in enumerate(y):
        if label not in x_dict:
            x_dict[label] = [x[i]]
            x_latent_dict[label] = [x_latent[i]]
        else:
            x_dict[label].append(x[i])
            x_latent_dict[label].append(x_latent[i])

    x_list = []
    x_latent_list = []
    for label in x_dict:
        x_list.append(np.array(x_dict[label][:label_num]))
        x_latent_list.append(np.array(x_latent_dict[label][:label_num]))
    x = np.concatenate(x_list, axis=0)
    x_latent = np.concatenate(x_latent_list, axis=0)

    return x, x_latent
