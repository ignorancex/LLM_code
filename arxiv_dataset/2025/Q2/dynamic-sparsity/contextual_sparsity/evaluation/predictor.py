# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from contextual_sparsity.utils.misc import move_to_device
from contextual_sparsity.utils.stats import compute_func_stats

MASK_OVERLAP_PERC = "mask_overlap"
PRESERVED_NORM_PERC = "preserved_norm"
MSE = "mse"
RSE = "rse"
MSE_DIFF = "mse_diff"
RSE_DIFF = "rse_diff"
ACT_NORM = "act_norm"


def evaluate_predictor(
    predictor: nn.Module,
    dataloader: DataLoader,
    down_layer: nn.Module,
    up_layer: nn.Module,
    act_fn: Callable,
    gate_layer: Optional[nn.Module] = None,
    k_spacing: int = 64,
) -> pd.DataFrame:
    """
    Evaluate a (sparsity) predictor on a given dataloader by comparing the predictions with gt activations.
    """
    device = next(predictor.parameters()).device

    # Set the correct devices and evaluation mode
    predictor.eval()
    down_layer.eval()
    predictor = predictor.to(device)
    down_layer = down_layer.to(device)
    up_layer = up_layer.to(device)
    if gate_layer is not None:
        gate_layer = gate_layer.to(device)

    # Determine the activation size
    first_batch = next(iter(dataloader))["x"][:2].to(device)
    K = predictor(first_batch).shape[-1]

    all_ks = np.arange(0, K)[::k_spacing]
    dtype = down_layer.weight.dtype

    # Function that is called every batch to compute the relevant metrics
    def compute_metrics(batch):
        nonlocal predictor, down_layer, all_ks

        batch = move_to_device(batch, device)

        logits = predictor(batch["x"].to(torch.float32))

        # Compute the true activations
        # Get the input to the layer to sparsify
        if "x_last" in batch:
            x_last = batch["x_last"]
        else:
            x_last = batch["x"]
        up = up_layer(x_last.to(up_layer.weight.dtype))
        if gate_layer is not None:
            gate = act_fn(gate_layer(x_last))
            act = up * gate
        else:
            act = act_fn(up)

        ordinal_targets = torch.argsort(torch.abs(act), dim=-1, descending=True)
        ordinal_predicted = torch.argsort(logits, dim=-1, descending=True)

        results = {
            MASK_OVERLAP_PERC: [],
            PRESERVED_NORM_PERC: [],
            MSE: [],
            MSE_DIFF: [],
        }
        # Start with all-zeros masks
        target_mask = (act * 0).type(dtype)
        predicted_mask = (act * 0).type(dtype)

        prev_k = 0
        for k in all_ks:
            # Set the k-th largest entry to 1
            target_mask.scatter_(-1, ordinal_targets[:, prev_k:k], 1)
            predicted_mask.scatter_(-1, ordinal_predicted[:, prev_k:k], 1)

            # Mask overlap
            accuracy = (target_mask * predicted_mask).float().sum(-1) / (k + 1)
            results[MASK_OVERLAP_PERC].append(accuracy.unsqueeze(-1))

            # Preserved Norm Percentage
            masked_target = (target_mask * act).type(dtype)
            masked_predicted = (predicted_mask * act).type(dtype)

            target_norm = torch.norm(masked_target, dim=-1).unsqueeze(-1)
            predicted_norm = torch.norm(masked_predicted, dim=-1).unsqueeze(-1)
            results[PRESERVED_NORM_PERC].append(predicted_norm / target_norm)

            # MSE
            topk_out = down_layer(masked_target)
            predictor_out = down_layer(masked_predicted)
            out = down_layer(act)

            topk_mse = (out - topk_out).pow(2).mean(-1)
            predictor_mse = (out - predictor_out).pow(2).mean(-1)
            results[MSE].append(predictor_mse.unsqueeze(-1))
            results[MSE_DIFF].append(predictor_mse.unsqueeze(-1) - topk_mse.unsqueeze(-1))

            prev_k = k

        results = {ky: torch.cat(v, -1) for ky, v in results.items()}
        results[ACT_NORM] = torch.norm(act, 2, -1).unsqueeze(-1)

        return results

    # Compute the function statistics over the dataloader
    results = compute_func_stats(
        dataloader=dataloader,
        func=compute_metrics,
    )

    # convert to a dataframe for convenience
    pd_results = []
    for metric in [MASK_OVERLAP_PERC, PRESERVED_NORM_PERC, MSE, MSE_DIFF]:
        for i, k in enumerate(all_ks):
            pd_results.append(
                {
                    "k": k,
                    "metric": metric,
                    "mean": results[metric]["mean"][i].item(),
                    "std": results[metric]["std"][i].item(),
                }
            )

    # Relative squared error computation
    act_var = results[ACT_NORM]["std"].item() ** 2
    for i, k in enumerate(all_ks):
        pd_results.append(
            {
                "k": k,
                "metric": RSE,
                "mean": results[MSE]["mean"][i].item() / act_var,
                "std": results[MSE]["std"][i].item() / act_var,
            }
        )
        pd_results.append(
            {
                "k": k,
                "metric": RSE_DIFF,
                "mean": results[MSE_DIFF]["mean"][i].item() / act_var,
                "std": results[MSE_DIFF]["std"][i].item() / act_var,
            }
        )

    return pd.DataFrame(pd_results)
