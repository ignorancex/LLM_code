#!/usr/bin/env python3

r"""
Get the utility values for the hypersphere example with varying number of objectives.
"""

import sys
import time
from typing import Optional, Tuple

import torch
from botorch.utils.sampling import sample_simplex
from scalarize.utils.sampling import sample_unit_vector
from scalarize.utils.scalarization_functions import ChebyshevScalarization
from torch import Tensor


def greedy_algorithm(
    all_scalarized_objective: Tensor,
    num_iterations: int,
    num_weights: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Greedy algorithm.

    Args:
        all_scalarized_objective: A `num_points x num_weights x 1`-dim Tensor
            containing the scalarized objective values.
        num_iterations: The number of iterations.
        num_weights: The number of weights used to compute the greedy policy.

    Returns:
        all_utility: A `num_iterations x 1`-dim Tensor containing the utility values.
    """
    utility_list = []
    total_num_weights = all_scalarized_objective.shape[-2]
    max_values = -torch.inf

    for i in range(num_iterations):
        if i == 0:
            if num_weights is None:
                utility = all_scalarized_objective.mean(dim=(-2, -1))
            else:
                rnd_indices = torch.randperm(total_num_weights)[:num_weights]
                utility = all_scalarized_objective[:, rnd_indices, :].mean(dim=(-2, -1))
        else:
            if num_weights is None:
                utility = torch.maximum(max_values, all_scalarized_objective).mean(
                    dim=(-1, -2)
                )
            else:
                rnd_indices = torch.randperm(total_num_weights)[:num_weights]
                utility = torch.maximum(
                    max_values[rnd_indices, :],
                    all_scalarized_objective[:, rnd_indices, :],
                ).mean(dim=(-1, -2))

        index_i = torch.argmax(utility)
        scalarized_obj = all_scalarized_objective[index_i]

        if i == 0:
            max_values = scalarized_obj
        else:
            max_values = torch.maximum(max_values, scalarized_obj)

        utility = max_values.mean(dim=(-2, -1))
        utility_list = utility_list + [utility]

    all_utility = torch.stack(utility_list)

    return all_utility


if __name__ == "__main__":
    M = int(float(sys.argv[1]))
    num_weights = 100000
    num_points = 100000
    num_iterations = 1000
    num_repeats = 100

    bounds = torch.zeros(2, M - 1)
    bounds[1] = 1.0

    torch.manual_seed(123)
    pf = sample_unit_vector(d=M, n=num_points)

    w = sample_simplex(d=M, n=num_weights, qmc=True, seed=123)
    r = torch.ones(1, M)
    chb = ChebyshevScalarization(weights=w, ref_points=r)
    s_pf = chb(pf)
    s_opt_pf = s_pf.max(dim=-1).values.max(dim=-2).values

    utilities = dict()

    for num_weights in [1, 5, 100]:
        wall_times = torch.zeros(num_repeats)

        util_g_all = []
        for j in range(num_repeats):
            start_time = time.time()
            torch.manual_seed(j)
            util_g = greedy_algorithm(
                all_scalarized_objective=s_pf,
                num_iterations=num_iterations,
                num_weights=num_weights,
            )
            util_g_all = util_g_all + [util_g]
            wall_times[j] = time.time() - start_time

        util_g_all = torch.stack(util_g_all)
        utilities[f"util_g_{num_weights}"] = util_g_all
        utilities[f"wall_time_g_{num_weights}"] = wall_times

        torch.save(utilities, f"data/utility_M{M}.pt")

    util_opt = torch.mean(s_opt_pf)
    utilities["optimal"] = util_opt
    utilities["num_objectives"] = M
    utilities["num_weights"] = num_weights
    utilities["num_points"] = num_points
    utilities["num_repeats"] = num_repeats
    utilities["num_iterations"] = num_iterations

    torch.save(utilities, f"data/utility_M{M}.pt")
