# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from logging import getLogger
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from contextual_sparsity.evaluation.hooks import (
    MLP_DENSITY,
    PERPLEXITY,
    CollectHooksOutput,
    EvaluationHook,
)
from contextual_sparsity.hw_simulator.simulator import HardwareSimulator
from contextual_sparsity.utils.stats import MEAN, compute_func_stats

log = getLogger(__name__)
MAX_PERPLEXITY = 1000


def get_perplexity(results: pd.DataFrame) -> float:
    """
    Extract perplexity from a result dataframe.
    """
    perplexity = results[
        (results["quantity"] == PERPLEXITY)
        & (results["stat"] == MEAN)
        & (results["computed_at"] == ".")
    ]["value"].values
    assert len(perplexity) == 1
    perplexity = perplexity[0]
    if perplexity != perplexity or perplexity > MAX_PERPLEXITY:
        perplexity = MAX_PERPLEXITY
    return perplexity


def get_mlp_density(results: pd.DataFrame) -> float:
    """
    Extract mlp density from a result dataframe.
    """
    weight_density = results[
        (results["quantity"] == MLP_DENSITY)
        & (results["stat"] == MEAN)
        & (results["computed_at"] == ".")
    ]["value"].values
    assert len(weight_density) == 1
    weight_density = weight_density[0]
    return weight_density


def evaluate_sparse_perplexity(
    model: nn.Module,
    test_data: DataLoader,
    evaluation_hooks: List[EvaluationHook],
    preprocess_batch: Optional[Callable] = None,
    hw_simulator: Optional[HardwareSimulator] = None,
) -> Dict[str, float]:
    """
    Evaluate the perplexity of a given sparse model
    """

    # Attach all the evaluation hooks to the model
    for evaluation_hook in evaluation_hooks:
        evaluation_hook.attach_to(model)

    # Define a global output collection hook.
    # This is essentially a function that given a model input returns the output stored by all the hooks
    collect_outputs = CollectHooksOutput(
        model=model, hooks=evaluation_hooks, preprocess_batch=preprocess_batch
    )

    model.eval()

    # Compute the statistics for all the values returned by the output hooks over the whole test_data
    stats = compute_func_stats(
        dataloader=test_data,
        func=collect_outputs,
    )

    # Remove all the hooks
    for evaluation_hook in evaluation_hooks:
        stats = evaluation_hook.finalize(stats)
        evaluation_hook.remove()

    # The statistics are in a nested dictionary [computed_at][quantity][stat_name] = value
    # Convert it to a pandas dataframe for convenience
    results = []
    for computed_at, quantity_dict in stats.items():
        for quantity, stat_dict in quantity_dict.items():
            for stat_name, stat_val in stat_dict.items():
                if torch.is_tensor(stat_val):
                    stat_val = stat_val.to("cpu").numpy()
                    if stat_val.ndim == 1:
                        stat_val = stat_val[0]

                results.append(
                    {
                        "computed_at": computed_at,
                        "quantity": quantity,
                        "stat": stat_name,
                        "value": stat_val,
                    }
                )
    results = pd.DataFrame(results)

    # Store the results
    results.to_csv("results.csv", index=False)
    log.info(f"Results saved in {os.path.join(os.getcwd(), 'results.csv')}")

    # Get and store results from HW simulator
    if hw_simulator is not None:
        results_hwsim = hw_simulator.get_stats_df()
        log.info(results_hwsim)
        results_hwsim.to_csv("results_hwsim.csv", index=False)
        log.info(
            f"Results for HW simulator saved in {os.path.join(os.getcwd(), 'results_hwsim.csv')}"
        )

    # Summarize the results into scalars for Optuna by reading the average weight density and perplexity
    summary = {
        MLP_DENSITY: get_mlp_density(results),
        "perplexity": get_perplexity(results),
    }

    return summary
