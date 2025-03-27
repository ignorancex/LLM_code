"""This module implements utility functions for the `networks.gcnn` subpackage."""
from collections import OrderedDict
from dunedn.networks.utils import BatchProfiler
from tqdm.auto import tqdm
import torch
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
    extras = ["module.norm_fn.med", "module.norm_fn.Min", "module.norm_fn.Max"]
    for extra in extras:
        if state_dict.get(extra, None) is not None:
            state_dict.move_to_end(extra)
            state_dict.popitem()

    changes_from = [
        "PreProcessBlocks",
        "PostProcessBlocks",
        "PreProcessBlock",
        "PostProcessBlock",
    ]
    changes_to = [
        "pre_process_blocks",
        "post_process_blocks",
        "pre_process_block",
        "post_process_block",
    ]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        for change_from, change_to in zip(changes_from, changes_to):
            if change_from in k:
                k = k.replace(change_from, change_to)
        new_k = k.lower().replace("module.", "")
        new_state_dict[new_k] = v

    return new_state_dict


def gcnn_inference_pass(
    test_loader: torch.utils.data.DataLoader,
    network: AbstractNet,
    dev: str,
    verbose: int = 1,
    profiler: BatchProfiler = None,
) -> torch.Tensor:
    """Consumes data through CNN or GCNN network and gives outputs.

    Parameters
    ----------
    test_loader: torch.utils.data.DataLoader
        The inference dataset generator.
    network: AbstractNet
        The denoising network.
    dev: str
        The device hosting the computation.
    verbose: int
        Switch to log information. Defaults to 1. Available options:

        - 0: no logs.
        - 1: display progress bar.

    profiler: BatchProfiler
            The profiler object to record batch inference time.

    Returns
    -------
    output: torch.Tensor
        Denoised data, of shape=(N,1,H,W).
    """
    network.eval()
    network.to(dev)
    outs = []
    wrap = tqdm(test_loader, desc="gcnn.predict") if verbose else test_loader
    if profiler is not None:
        wrap = profiler.set_iterable(wrap)
    for noisy, _ in wrap:
        out = network(noisy.to(dev)).detach().cpu()
        outs.append(out)
    output = torch.cat(outs)
    network.to("cpu")
    return output
