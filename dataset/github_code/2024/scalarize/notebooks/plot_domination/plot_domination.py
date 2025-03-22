#!/usr/bin/env python3

r"""
Plotting functions for the contours of the scalarization functions in two-dimensions.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def pareto_domination(X: Tensor, reference_set: Tensor, upper: bool = True) -> Tensor:
    r"""Computes the points in an array that Pareto dominates the reference point for
    a maximization problem.

    Args:
        X: An `N x 2`-dim array containing the 2D points.
        reference_set: A `num_ref x 2`-dim array containing the reference points.
        upper: If true we consider the region that dominates the reference point.

    Returns:
        An `N`-dim boolean array.
    """
    weight = 1.0 if upper else -1.0
    if len(reference_set.shape) == 1:
        return torch.all(weight * (X - reference_set) > 0, axis=-1)
    else:
        num_ref = reference_set.shape[0]
        mask = []
        for n in range(num_ref):
            mask = mask + [torch.all(weight * (X - reference_set[n, :]) > 0, axis=-1)]
        mask = torch.column_stack(mask)
        return torch.any(mask, axis=-1)


def plot_pareto_domination(
    Z: Tensor,
    reference_set: Tensor,
    title: str = "Pareto domination plot",
    fontsize: int = 20,
) -> None:
    r"""Plot the Pareto domination regions.

    Args:
        Z: An `(N * N) x 2`-dim Tensor containing the xy-coordinates.
        reference_set: A `num_ref x 2`-dim Tensor containing the reference points.
        title: The title of the plot.
        fontsize: The title fontsize.

    Returns:
        None.
    """
    upper = pareto_domination(Z, reference_set, upper=True)
    lower = pareto_domination(Z, reference_set, upper=False)
    up = Z[upper]
    low = Z[lower]
    empty = torch.all(~torch.column_stack([upper, lower]), axis=-1)
    other = Z[empty]
    plt.scatter(up[:, 0], up[:, 1], color="crimson", alpha=1, s=3, marker="s")
    plt.scatter(low[:, 0], low[:, 1], color="dodgerblue", alpha=1, s=3, marker="s")
    plt.scatter(other[:, 0], other[:, 1], color="#E0E0E0", alpha=1, s=3, marker="s")

    if len(reference_set.shape) == 1:
        reference_set = reference_set.unsqueeze(0)

    plt.scatter(
        reference_set[:, 0],
        reference_set[:, 1],
        color="w",
        s=75,
        edgecolors="k",
        linewidths=2,
        zorder=5,
    )

    plt.xlabel(r"$y^{(1)}$", fontsize=fontsize)
    plt.ylabel(r"$y^{(2)}$", fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title(title, fontsize=fontsize)
    return None


def plot_scalarized_domination(
    Z: Tensor,
    ZY: Tensor,
    Zr: float,
    reference_set: Tensor,
    contours: bool = False,
    X: Optional[Tensor] = None,
    Y: Optional[Tensor] = None,
    utopia_set: Optional[Tensor] = None,
    nadir_set: Optional[Tensor] = None,
    unit_vector: Optional[Tensor] = None,
    title: str = "Domination plot",
    levels: Optional[Tuple[int, Tensor]] = None,
    fontsize: int = 20,
) -> None:
    r"""Plot the domination regions of the scalarization function.

    Args:
        Z: An `(N * N) x 2`-dim Tensor containing the xy-coordinates.
        ZY: An `(N * N) x 1`-dim Tensor scalarized objectives at `Y`.
        Zr: The scalarized objective at the reference point.
        reference_set: A `num_ref x 2`-dim Tensor containing the reference point.
        contours: If true, then we plot the contours.
        X: An `N x 1`-dim array containing the x-coordinates.
        Y: An `N x 1`-dim array containing the y-coordinates.
        utopia_set: An `num_utopia x 2`-dim array containing the utopia points.
        nadir_set: An `num_nadir x 2`-dim array containing the nadir points.
        unit_vector: A `2`-dim array containing the unit vector to draw the penalty
            boundary line.
        title: The title of the plot.
        levels: An optional value or Tensor containing the levels for the contours.
        fontsize: The title fontsize.

    Returns:
        None.
    """

    upper_mask = (ZY > Zr).squeeze(-1)
    up = Z[upper_mask]
    low = Z[~upper_mask]
    plt.scatter(up[:, 0], up[:, 1], color="crimson", alpha=1, s=3, marker="s")
    plt.scatter(low[:, 0], low[:, 1], color="dodgerblue", alpha=1, s=3, marker="s")
    if len(reference_set.shape) == 1:
        reference_set = reference_set.unsqueeze(0)

    plt.scatter(
        reference_set[:, 0],
        reference_set[:, 1],
        color="w",
        edgecolors="k",
        linewidths=2,
        s=75,
        zorder=5,
    )

    if levels is None:
        levels = 20

    if contours:
        N = X.shape[0]
        Zf = (ZY - Zr).reshape(N, N)
        plt.contour(X, Y, Zf, levels, colors="k", alpha=0.4, linewidths=2)
        plt.contour(
            X, Y, Zf, [0], colors="k", alpha=1, linewidths=2.5, linestyles="solid"
        )

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
    if unit_vector is not None:
        lines = []
        t = torch.linspace(-10, 10, 100).unsqueeze(-1)
        if utopia_set is not None:
            for u in unit_vector:
                lines = lines + [utopia_set[0, :] + t * u]
        if nadir_set is not None:
            for u in unit_vector:
                lines = lines + [nadir_set[0, :] + t * u]

        for line in lines:
            plt.plot(
                line[:, 0],
                line[:, 1],
                color="w",
                linestyle="--",
                linewidth=2.5,
                zorder=4,
            )

    plt.xlabel(r"$y^{(1)}$", fontsize=fontsize)
    plt.ylabel(r"$y^{(2)}$", fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.title(title, fontsize=fontsize)
    return None
