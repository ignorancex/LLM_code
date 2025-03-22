#!/usr/bin/env python3

r"""
Plotting methods for the segment plots on the hypersphere problem in two-dimensions.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import Tensor


def greedy_algorithm(
    all_scalarized_objective: Tensor,
    num_iterations: int,
    num_weights: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Run the approximate greedy algorithm.

    Args:
        all_scalarized_objective: A `num_points x num_weights`-dim Tensor
            containing the scalarized objective values.
        num_iterations: The number of iterations.
        num_weights: The number of weights used to compute the greedy policy.

    Returns:
        scalarized_obj: A `num_iterations x num_weights`-dim Tensor containing
            the selected scalarized objective values.
        indices: A `num_iterations `-dim Tensor containing the selected indices.
    """
    scalarized_obj_list = []
    indices = []
    total_num_weights = all_scalarized_objective.shape[-1]
    max_values = -torch.inf

    for i in range(num_iterations):
        if i == 0:
            if num_weights is None:
                utility = all_scalarized_objective.mean(dim=-1)
            else:
                rnd_indices = torch.randperm(total_num_weights)[:num_weights]
                utility = all_scalarized_objective[:, rnd_indices].mean(dim=-1)
        else:
            if num_weights is None:
                utility = torch.maximum(max_values, all_scalarized_objective).mean(
                    dim=-1
                )
            else:
                rnd_indices = torch.randperm(total_num_weights)[:num_weights]
                utility = torch.maximum(
                    max_values[rnd_indices], all_scalarized_objective[:, rnd_indices]
                ).mean(dim=-1)

        index_i = torch.argmax(utility)
        indices = indices + [index_i]

        scalarized_obj_list = scalarized_obj_list + [all_scalarized_objective[index_i]]

        if i == 0:
            max_values = scalarized_obj_list[-1]
        else:
            max_values = torch.maximum(max_values, scalarized_obj_list[-1])

    scalarized_obj = torch.stack(scalarized_obj_list)

    return scalarized_obj, torch.stack(indices)


def plot_segments(
    num_iterations: int,
    weights: Tensor,
    all_scalarized_objective: Tensor,
    scalarized_objectives: Tensor,
    title: str = "Segment plot",
    lower_bound: float = 0.01,
    upper_bound: float = 0.01,
) -> None:
    r"""Segment plot. This assumes one-dimensional weights.

    Args:
        num_iterations: The number of iterations.
        weights: A `num_weights`-dim Tensor containing the weights.
        all_scalarized_objective: A `num_points x num_weights`-dim Tensor containing
            all the scalarized objective values.
        num_iterations: The number of iterations.
        scalarized_objectives: A `num_iterations x num_weights`-dim Tensor
            containing the selected scalarized objective values.
        title: The title of the plot.
        lower_bound: The lower bound of the scalarization function used to determine
            the limits of the plot.
        upper_bound: The upper bound of the scalarization function used to determine
            the limits of the plot.

    Returns:
        None.
    """
    colors = pl.cm.viridis(np.linspace(0, 1, num_iterations))
    s_max = (
        r"$\text{max}_{\mathbf{x} \in \mathbb{X}} "
        r"s_{\boldsymbol{\theta}}(f(\mathbf{x}))$"
    )

    low_lim = torch.min(all_scalarized_objective) - lower_bound * torch.max(
        abs(all_scalarized_objective)
    )

    up_lim = torch.max(all_scalarized_objective) + upper_bound * torch.max(
        abs(all_scalarized_objective)
    )

    for i in range(num_iterations):
        if i == 0:
            mask = torch.ones(weights.shape, dtype=torch.bool)
            lb = low_lim * torch.ones(len(weights))
            max_values = scalarized_objectives[i]

        else:
            max_values_new = torch.maximum(max_values, scalarized_objectives[i])
            mask = max_values_new > max_values
            lb = max_values.squeeze(-1)

            max_values = max_values_new

        change_mask = mask[:-1] != mask[1:]
        # This deals with the case where there are many intersections between the
        # curves. When this happens we split the mask up into many slices.
        if torch.sum(change_mask) > 2:
            mask_list = []
            num_masks = torch.sum(change_mask) + 1
            change_index = torch.where(change_mask)[0]
            for m in range(num_masks):
                mask_m = torch.zeros(weights.shape, dtype=bool)
                if m == 0:
                    initial_idx = 0
                else:
                    initial_idx = change_index[m - 1] + 1

                if m == num_masks - 1:
                    final_idx = len(weights)
                else:
                    final_idx = change_index[m] + 1

                mask_m[initial_idx:final_idx] = mask[initial_idx]
                mask_list = mask_list + [mask_m]
        else:
            mask_list = [mask]

        for mask in mask_list:
            plt.plot(
                weights[mask],
                scalarized_objectives[i][mask],
                linewidth=3,
                zorder=num_iterations - i,
                color=colors[i],
            )

            plt.fill_between(
                weights[mask],
                scalarized_objectives[i, ...].squeeze(-1)[mask],
                lb[mask],
                alpha=0.75,
                zorder=num_iterations - i,
                color=colors[i],
            )

    optimal_front = all_scalarized_objective.max(dim=0).values

    plt.plot(
        weights,
        optimal_front,
        linewidth=3,
        linestyle="-",
        color="k",
        zorder=num_iterations + 1,
        label=s_max,
    )
    plt.xlim(0, 1)
    plt.ylim(low_lim, up_lim)
    plt.yticks([], fontsize=20)
    plt.xticks([], fontsize=20)
    # plt.xlabel(r"$\boldsymbol{\theta}$", fontsize=25)
    # plt.ylabel(r"$s_{\boldsymbol{\theta}}(f(\mathbf{x}))$", fontsize=25)
    plt.title(title, fontsize=30)
    return None


def plot_all_segments_mc(
    num_iterations: int,
    weights: Tensor,
    all_scalarized_objective: Tensor,
    scalarized_objectives_list: List[Tensor],
    num_samples: List[int],
    lower_bound: float,
    upper_bound: float,
) -> plt.Figure:
    r"""Segment plot for the approximate greedy algorithm using different number of
    samples.

    Args:
        num_iterations: The number of iterations.
        weights: A `num_weights`-dim Tensor containing the weights.
        all_scalarized_objective: A `num_points x num_weights`-dim Tensor
            containing all the scalarized objective values.
        num_iterations: The number of iterations.
        scalarized_objectives_list: A list of `num_iterations x num_weights`-dim
            Tensor containing the selected scalarized objective values.
        num_samples: A `N`-dim Tensor containing the number of samples for each
            segment plot.
        lower_bound: The lower bound of the scalarization function used to determine
            the limits of the plot.
        upper_bound: The upper bound of the scalarization function used to determine
            the limits of the plot.

    Returns:
        The figure.
    """
    num_rows = 2
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 9.5))
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.2
    )
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom_cmap", cmaplist, cmap.N
    )
    # define the bins and normalize
    bounds = np.linspace(1, num_iterations + 1, num_iterations + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    for j in range(len(num_samples)):
        scalarized_objectives = scalarized_objectives_list[j]
        plt.subplot(num_rows, num_cols, j + 1)

        plot_segments(
            num_iterations=num_iterations,
            weights=weights,
            all_scalarized_objective=all_scalarized_objective,
            scalarized_objectives=scalarized_objectives,
            title=f"$J={int(num_samples[j])}$",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        if (j + 1) > (num_rows - 1) * num_cols:
            plt.xlabel(r"$\boldsymbol{\theta}$", fontsize=25)
        if j % num_cols == 0:
            plt.ylabel(r"$s_{\boldsymbol{\theta}}(f(\mathbf{x}))$", fontsize=25)

    axes[0, 0].legend(
        fontsize=22,
        facecolor="w",
        loc="lower center",
        framealpha=0.9,
    ).set_zorder(2 * num_iterations)

    ax_bar = fig.add_axes([0.125, -0.00, 0.775, 0.05])
    cbar = matplotlib.colorbar.ColorbarBase(
        ax_bar,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=bounds + 0.5,
        boundaries=bounds,
        format="%1i",
        orientation="horizontal",
    )
    ax_bar.set_xlabel(r"$n$", fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.show()
    return fig


def plot_all_segments_sfn(
    num_iterations: int,
    weights: Tensor,
    all_scalarized_objective_list: List[Tensor],
    scalarized_objectives_list: List[Tensor],
    title_list: List[str],
    lower_bound: List[float],
    upper_bound: List[float],
) -> plt.Figure:
    r"""Segment plot for different scalarization functions.

    Args:
        num_iterations: The number of iterations.
        weights: A `num_weights`-dim Tensor containing the weights.
        all_scalarized_objective_list: A list of `num_points x num_weights`-dim
            Tensor containing all the scalarized objective values.
        scalarized_objectives_list: A list of `num_iterations x num_weights`-dim
            Tensor containing the selected scalarized objective values.
        title_list: The titles for the scalarization functions.
        lower_bound: The lower bound of the scalarization function used to determine
            the limits of the plot.
        upper_bound: The upper bound of the scalarization function used to determine
            the limits of the plot.

    Returns:
        The figure.
    """
    num_rows = 4
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.3
    )
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom_cmap", cmaplist, cmap.N
    )
    # define the bins and normalize
    bounds = np.linspace(1, num_iterations + 1, num_iterations + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    for j in range(len(title_list)):
        scalarized_objectives = scalarized_objectives_list[j]
        all_scalarized_objective = all_scalarized_objective_list[j]
        plt.subplot(num_rows, num_cols, j + 1)

        plot_segments(
            num_iterations=num_iterations,
            weights=weights,
            all_scalarized_objective=all_scalarized_objective,
            scalarized_objectives=scalarized_objectives,
            title=title_list[j],
            lower_bound=lower_bound[j],
            upper_bound=upper_bound[j],
        )

    axes[0, 0].legend(
        fontsize=20, facecolor="white", loc="lower center", framealpha=1
    ).set_zorder(2 * num_iterations)

    ax_bar = fig.add_axes([0.135, 0.05, 0.75, 0.02])
    cbar = matplotlib.colorbar.ColorbarBase(
        ax_bar,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=bounds + 0.5,
        boundaries=bounds,
        format="%1i",
        orientation="horizontal",
    )
    ax_bar.set_xlabel("N", fontsize=25)
    cbar.ax.tick_params(labelsize=20)
    plt.show()
    return fig


def plot_all_pareto_fronts(
    pareto_front: Tensor,
    num_iterations: int,
    index_list: List[int],
    title_list: List[str],
    ref_dict: Dict[str, List[Tensor]],
) -> plt.Figure:
    r"""The Pareto front plot for the different scalarization functions.

    Args:
        pareto_front: A `num_pareto_points`-dim Tensor containing the weights.
        num_iterations: The number of iterations.
        index_list: List of index of the greedily chosen points.
        title_list: The titles for the scalarization functions.
        ref_dict: The dictionary of the reference points.

    Returns:
        The figure.
    """
    num_rows = 4
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.3
    )
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom_cmap", cmaplist, cmap.N
    )
    # define the bins and normalize
    bounds = np.linspace(1, num_iterations + 1, num_iterations + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    colors = pl.cm.viridis(np.linspace(0, 1, num_iterations))

    for j in range(len(title_list)):
        indices = index_list[j]
        plt.subplot(num_rows, num_cols, j + 1)
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color="k", s=10)
        plt.scatter(
            pareto_front[indices, 0],
            pareto_front[indices, 1],
            c=colors,
            s=200,
            zorder=5,
            linewidth=2,
            edgecolors="k",
        )
        plt.title(title_list[j], fontsize=20)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xlabel(r"$y^{(1)}$", fontsize=16)
        plt.ylabel(r"$y^{(2)}$", fontsize=16)

        utopia_set = ref_dict[title_list[j]][0]
        nadir_set = ref_dict[title_list[j]][1]
        if utopia_set is not None:
            if len(utopia_set.shape) == 1:
                utopia_set = utopia_set.unsqueeze(0)

            plt.scatter(
                utopia_set[:, 0],
                utopia_set[:, 1],
                color="white",
                marker="*",
                edgecolors="k",
                linewidths=2,
                s=200,
                zorder=5,
            )

        if nadir_set is not None:
            if len(nadir_set.shape) == 1:
                nadir_set = nadir_set.unsqueeze(0)

            plt.scatter(
                nadir_set[:, 0],
                nadir_set[:, 1],
                color="white",
                marker="s",
                edgecolors="k",
                linewidths=2,
                s=100,
                zorder=5,
            )

    ax_bar = fig.add_axes([0.135, 0.05, 0.75, 0.02])
    cbar = matplotlib.colorbar.ColorbarBase(
        ax_bar,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=bounds + 0.5,
        boundaries=bounds,
        format="%1i",
        orientation="horizontal",
    )
    ax_bar.set_xlabel("N", fontsize=25)
    cbar.ax.tick_params(labelsize=20)
    plt.show()
    return fig
