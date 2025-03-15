"""This module implements utility functions for the `networks.uscg` subpackage."""
from typing import Tuple
from tqdm.auto import tqdm
from math import ceil
import torch
from collections import OrderedDict
from ..abstract_net import AbstractNet


def make_dict_compatible(state_dict: OrderedDict):
    """Transforms `state_dict` keys to match new GcnnNet attributes format.

    *Changed in version 2.0.0:*

    - Remove ".module" in front of the saved weights name.
    - Remove extra attributes due to deprecated normalization layer.
    - Transform layers names to lowercase.

    Parameters
    ----------
    state_dict: OrderedDict
        The original dictionary containing network saved weights.

    Returns
    -------
    new_state_dict: OrderedDict
        The dictionary updated version.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.lower().replace("module.", "")
        new_state_dict[new_k] = v

    return new_state_dict


def time_windows(
    planes: torch.Tensor, w: int, stride: int
) -> Tuple[torch.Tensor, list[torch.Tensor], list[list[int]]]:
    """Takes time windows of given width and stride from a planes.

    Parameters
    ----------
    planes: torch.Tensor
        Planes of shape=(N,C,H,W).
    w: int
        Width of the time windows.
    stride: int
        Steps between time windows.

    Returns
    -------
    divisions: torch.Tensor
        Number of times each pixel should be processed by denoising network, of
        shape=(N,C,H,W).
    windows: list[torch.Tensor]
        Time windows to be processed, each of shape=(N,C,H,w)
    idxs: list[list[int]]
        Start-end time indices for the correspondent window. Each elements is
        [start idx, end idx].
    """
    _, _, _, width = planes.size()
    n = ceil((width - w) / stride) + 1
    base = torch.arange(n).unsqueeze(1) * stride
    idxs = torch.Tensor([[0, w]]).long() + base
    windows = []
    divisions = torch.zeros_like(planes)
    for start, end in idxs:
        window = planes[..., start:end]
        windows.append(window)
        divisions[..., start:end] += 1
    return divisions, windows, idxs


def uscg_inference_pass(
    test_loader: torch.utils.data.DataLoader,
    network: AbstractNet,
    dev: str,
    verbose: int = 1,
) -> torch.Tensor:
    """Consumes data through USCG network and gives outputs.

    Parameters
    ----------
    test_loader: torch.utils.data.Dataloader
        The inference dataset generator.
    network: AbstractNet
        The denoising network.
    dev: str
        The device hosting the computation.
    verbose: int
        Switch to log information. Defaults to 1. Available options:

        - 0: no logs.
        - 1: display progress bar.

    Returns
    -------
    output: torch.Tensor
        Denoised data, of shape=(N,1,H,W).
    """
    w = network.w
    network.eval()
    network.to(dev)
    outs = []
    wrap = tqdm(test_loader, desc="uscg.predict") if verbose else test_loader
    for noisy, _ in wrap:
        div, nwindows, idxs = time_windows(noisy, w, network.stride)
        out = torch.zeros_like(noisy)
        for nwindow, (start, end) in zip(nwindows, idxs):
            out[..., start:end] = network(nwindow.to(dev)).detach().cpu()
        outs.append(out / div)
    output = torch.cat(outs)
    network.to("cpu")
    return output
