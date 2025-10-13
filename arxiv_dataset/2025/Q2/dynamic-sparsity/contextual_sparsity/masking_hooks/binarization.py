# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import tqdm
from torch import nn
from torch.utils.data import DataLoader

from contextual_sparsity.nn import ThresholdMask, TopKMask
from contextual_sparsity.utils.stats import collect_activations

log = logging.getLogger(__name__)


class BinarizationType(Enum):
    topk = "topk"
    threshold = "threshold"


def build_topk_binarization(
    dense_model: nn.Module,
    activation_ids: List[str],
    k: Optional[Union[int, List[int]]] = None,
    keep: Optional[Union[float, List[float]]] = None,
) -> Dict[str, nn.Module]:
    """
    Create the binarization function that preserve the k (or keep%) largest absolute activations
    for a list of given activations_ids.
    """
    if (keep is None and k is None) or (keep is not None and k is not None):
        raise ValueError("Either keep or k must be specified")

    if isinstance(k, int):
        k = [k] * len(activation_ids)

    if k is None:
        if isinstance(keep, float):
            keep = [keep] * len(activation_ids)

        assert len(keep) == len(
            activation_ids
        ), "`keep` should be a float or a list of floats of the same length as `activation_ids`"

        k = []
        # Determine the values of k
        for i, activation_id in enumerate(activation_ids):
            kp = keep[i]
            layer_id = activation_id.rsplit(".", 1)[0]
            output = activation_id.rsplit(".", 1)[1] == "output"

            if kp < 0 or kp > 1:
                raise ValueError("keep must be between 0 and 1")

            layer = dense_model.get_submodule(layer_id)
            if hasattr(layer, "weight"):
                dim = 0 if output else 1
                activation_size = layer.weight.shape[dim]
            else:
                # The activation we are considering is a non-linearity (e.g. for gate pruning)
                # So the activation size is the intermediate size
                activation_size = dense_model.config.intermediate_size
            k_ = int(kp * activation_size)
            k.append(k_)

    assert len(k) == len(activation_ids)

    activation_binarization = {
        activation_id: TopKMask(k=k[i]) for i, activation_id in enumerate(activation_ids)
    }
    return activation_binarization


def compute_percentile_abs_activations(
    model_id: str,
    data_id: str,
    data: DataLoader,
    dense_model: nn.Module,
    activation_ids: List[str],
    percentile: List[float],
    preprocess_batch: Optional[Callable] = None,
) -> List[float]:
    """
    Compute the percentile absolute activations for a list of activations_ids.
    """

    log.info(f"Computing the  activation percentiles for {model_id} on {data_id}")
    for activation_id in activation_ids:
        if not activation_id.endswith(".input") and not activation_id.endswith(".output"):
            raise ValueError(
                "Activation ids must end with .input or .output to specify if we consider input or output of the layers"
                f".\n{activation_id} found."
            )
        activation_id = activation_id.rsplit(".", 1)[0]
        if dense_model.get_submodule(activation_id) is None:
            raise ValueError(f"The model {model_id} does not have a layer {activation_id}.")

    def collect_data(layer, args, kwargs, output, layer_name, data, collect_output):
        if collect_output:
            x = output
        else:
            x = args[0]

        data[layer_name] = torch.abs(x)

    all_activations = {}
    with collect_activations(
        collection_funcs={
            activation_id.rsplit(".", 1)[0]: partial(
                collect_data, collect_output=activation_id.endswith(".output")
            )
            for activation_id in activation_ids
        },
        dense_model=dense_model,
        preprocess_batch=preprocess_batch,
    ) as func:
        with torch.no_grad():
            for batch in tqdm(data):
                batch = preprocess_batch(batch)
                activations = func(batch)
                if len(all_activations) == 0:
                    all_activations = {
                        k: v.detach().cpu().numpy().astype(np.float16).reshape(-1)
                        for k, v in activations.items()
                    }
                else:
                    for k, v in activations.items():
                        all_activations[k] = np.concatenate(
                            [
                                all_activations[k],
                                v.detach().cpu().numpy().astype(np.float16).reshape(-1),
                            ],
                            axis=0,
                        )

    thresholds = []
    for i, activation_id in enumerate(activation_ids):
        layer_id = activation_id.rsplit(".", 1)[0]
        p = np.percentile(all_activations[layer_id], (1 - percentile[i]) * 100, axis=0)
        thresholds.append(p)

    return thresholds


def build_threshold_binarization(
    model_id: str,
    data_id: str,
    calibration_data: DataLoader,
    dense_model: nn.Module,
    activation_ids: List[str],
    threshold: Optional[Union[float, List[float]]] = None,
    keep: Optional[Union[float, List[float]]] = None,
    preprocess_batch: Optional[Callable] = None,
) -> Dict[str, nn.Module]:
    """
    Make masking functions for every specified layer based on a given threshold (thredhold=value), or computing it
    based on the percentile of the absolute activations on the given calibration data.
    """
    if (keep is None and threshold is None) or (keep is not None and threshold is not None):
        raise ValueError("Either keep or threshold must be specified")

    if isinstance(keep, float):
        keep = [keep] * len(activation_ids)
    if isinstance(threshold, float):
        threshold = [threshold] * len(activation_ids)

    if threshold is None:
        assert isinstance(keep, list)

        threshold = compute_percentile_abs_activations(
            model_id=model_id,
            data_id=data_id,
            data=calibration_data,
            dense_model=dense_model,
            preprocess_batch=preprocess_batch,
            activation_ids=activation_ids,
            percentile=keep,
        )

    activation_binarization = {
        activation_id: ThresholdMask(threshold=threshold[i])
        for i, activation_id in enumerate(activation_ids)
    }
    return activation_binarization


def build_binarization(
    model_id: str,
    dense_model: nn.Module,
    activation_ids: List[str],
    data_id: Optional[str] = None,
    calibration_data: Optional[DataLoader] = None,
    binarization_type: str = BinarizationType.topk.value,
    keep: Optional[Union[int, List[int]]] = None,
    k: Optional[Union[int, List[int]]] = None,
    threshold: Optional[Union[int, List[int]]] = None,
    preprocess_batch: Optional[Callable] = None,
) -> Dict[str, nn.Module]:
    """
    Utility function that produces binarization functions given the specified binarization type, their parameters,
     and a list of activations to binarize.
    """

    binarization_type = BinarizationType[binarization_type]

    if binarization_type == BinarizationType.topk:
        activation_binarization = build_topk_binarization(
            dense_model=dense_model, activation_ids=activation_ids, keep=keep, k=k
        )
    else:
        activation_binarization = build_threshold_binarization(
            model_id=model_id,
            data_id=data_id,
            activation_ids=activation_ids,
            calibration_data=calibration_data,
            dense_model=dense_model,
            threshold=threshold,
            preprocess_batch=preprocess_batch,
            keep=keep,
        )

    assert len(activation_binarization) == len(activation_ids)

    return activation_binarization
