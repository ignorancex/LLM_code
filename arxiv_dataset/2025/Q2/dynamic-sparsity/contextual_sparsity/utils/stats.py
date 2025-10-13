# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import contextlib
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

MEAN = "mean"
SQ_MEAN = "sq_mean"
MIN = "min"
MAX = "max"
STD = "std"


def _compute_stats(v: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute statistics for a tensor along the 0th dimension.
    """

    assert torch.is_tensor(v), "v must be a tensor"
    # If one-dimensional add a dimension at the end
    if v.ndim == 1:
        v = v.unsqueeze(-1)
    v = v.type(torch.float64)

    # If multidimensional flatten all but the last
    v = v.view(-1, v.shape[-1])

    assert v.ndim == 2
    n = v.shape[0]

    stats = {
        MEAN: [v.sum(0), n],
        SQ_MEAN: [(v**2).sum(0), n],
        MIN: v.min(0).values,
        MAX: v.max(0).values,
    }

    return stats


def _mean_aggregation(v_old, v_new):
    """
    Aggregate statistics the sum and element count for mean computation
    """
    # If no count is passed, we assume one value
    if not isinstance(v_old, list):
        v_old = [v_old, 1]
    if not isinstance(v_new, list):
        v_new = [v_new, 1]
    # Aggregate sum and count
    return [v_old[0] + v_new[0], v_old[1] + v_new[1]]


def _min_aggregation(v_old, v_new):
    """
    Aggregate the minimum value
    """
    return torch.cat([v_old.unsqueeze(0), v_new.unsqueeze(0)], dim=0).min(dim=0).values


def _max_aggregation(v_old, v_new):
    """
    Aggregate the maximum value
    """
    return torch.cat([v_old.unsqueeze(0), v_new.unsqueeze(0)], dim=0).max(dim=0).values


def _mean_summarization(v):
    """
    Summarize sum and element count into one mean value
    """
    assert isinstance(v, list)
    assert len(v) == 2
    s, count = v[0], v[1]
    return s / float(count)


def _accumulate_stats(stats: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accumulate a dictionary of statistics along the 0th dimension, adding values to the specified stats
    """
    if torch.is_tensor(values):
        # Update the cumulative statistics
        if stats is None:
            stats = _compute_stats(values)
        else:
            for name, new_v in _compute_stats(values).items():
                if name == MEAN or name == SQ_MEAN:
                    agg_function = _mean_aggregation
                elif name == MIN:
                    agg_function = _min_aggregation
                elif name == MAX:
                    agg_function = _max_aggregation
                else:
                    raise NotImplementedError()
                stats[name] = agg_function(stats[name], new_v)
        return stats
    else:
        new_stats = {}
        for k, v in values.items():
            if stats is None:
                sub_stats = _accumulate_stats(None, v)
            else:
                sub_stats = _accumulate_stats(stats[k], v)
            new_stats[k] = sub_stats
        return new_stats


def _summary_stats(stats: Dict[str, Any]) -> Union[Dict[str, Any], torch.Tensor]:
    """
    wrapper function to summarize the statistics
    """
    if MEAN in stats:
        summary_stats = {}

        for name, stat in stats.items():
            if name == MEAN or name == SQ_MEAN:
                summarization_function = _mean_summarization
            elif name == MIN or name == MAX:
                summarization_function = lambda x: x
            else:
                raise NotImplementedError()

            summary_stats[name] = summarization_function(stat)

        if SQ_MEAN in summary_stats:
            var = summary_stats[SQ_MEAN] - summary_stats[MEAN] ** 2
            # Numerical instability can make the variance negative when close to 0
            var = torch.clamp(var, min=0.0)
            summary_stats[STD] = var**0.5

        return summary_stats
    else:
        new_stats = {}
        for k, v in stats.items():
            new_stats[k] = _summary_stats(v)
        return new_stats


class StopComputation(Exception):
    """
    Exception raised when computation should be stopped. This is used to jump out of the forward model call early.
    """

    pass


def compute_func_stats(
    dataloader: DataLoader,
    func: Callable,
) -> Dict[str, Any]:
    """
    Compute the statistic of a given function on all the element provided by a dataloader.
    """

    # Compute statistics
    stats = None

    with torch.no_grad():
        for data in tqdm(dataloader):
            values = func(data)
            # Nested update
            stats = _accumulate_stats(stats, values)

    stats = _summary_stats(stats)
    if stats is None:
        stats = {}

    return stats


@contextlib.contextmanager
def collect_activations(
    collection_funcs: Dict[str, Callable],
    dense_model: torch.nn.Module,
    preprocess_batch: Optional[Callable] = None,
) -> Iterator[Callable[[Any], Dict[str, Any]]]:
    """
    Context manager that provides a function that returns all specified activations of a dense model for a given batch.
    The hooks are removed on context exit.
    """
    data: Dict[str, Any] = {}

    handles = []
    for layer_name, collect_data in collection_funcs.items():
        layer = dense_model.get_submodule(layer_name)
        handles.append(
            layer.register_forward_hook(
                partial(collect_data, layer_name=layer_name, data=data),
                with_kwargs=True,
            )
        )

    def collect_all_data(batch: Any) -> Dict[str, Any]:
        """
        Collect all the activations for a given batch
        """
        nonlocal data, dense_model
        if preprocess_batch is not None:
            batch = preprocess_batch(batch)
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if isinstance(batch, dict):
            kwargs = batch
        elif isinstance(batch, list) or isinstance(batch, tuple):
            args = batch
        else:
            args = [batch]
        try:
            dense_model(*args, **kwargs)
        except StopComputation:
            # Skip computation by catching the exception
            pass

        return data

    yield collect_all_data

    # On context exit remove the hooks
    for handle in handles:
        handle.remove()


def compute_layers_stats(
    dataloader: DataLoader,
    layer_ids: Union[str, List[str]],
    model: torch.nn.Module,
    preprocess_batch: Optional[Callable] = None,
    collect_data: Optional[Callable] = None,
    output: bool = True,
):
    """
    Compute the statistics of given layer_ids on all the element provided by a dataloader.
    """
    if isinstance(layer_ids, str):
        layer_ids = [layer_ids]

    if collect_data is None:
        if output:

            def collect_data(module, args, kwargs, out, layer_name, data):
                data[layer_name] = out

        else:

            def collect_data(module, args, kwargs, out, layer_name, data):
                data[layer_name] = args[0]

    # Define a function that returns the input/output of all the listed layers for one batch
    with collect_activations(
        collection_funcs={layer_name: collect_data for layer_name in layer_ids},
        dense_model=model,
        preprocess_batch=preprocess_batch,
    ) as func:
        stats = compute_func_stats(dataloader=dataloader, func=func)

    return stats


def get_stats_from_array(x: Union[torch.Tensor, np.ndarray], axis=-1):
    """
    Get statistics of a tensor along the specified axis as a list.
    """
    assert isinstance(x, np.ndarray), x
    return (
        x.mean(axis=axis).item(),
        x.std(axis=axis).item(),
        x.min(axis=axis).item(),
        x.max(axis=axis).item(),
    )


def get_stats_dict_from_array(x, axis=-1):
    """
    Get statistics of a tensor along the specified axis as a dictionary.
    """
    stats_names = [MEAN, STD, MIN, MAX]
    stats_values = get_stats_from_array(x, axis=axis)
    return {name: value for name, value in zip(stats_names, stats_values)}
