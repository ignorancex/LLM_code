# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import pandas as pd

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec

from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import block_diag
import networkx as nx

from typing import Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import to_rgb, to_rgba, ListedColormap
from matplotlib.patheffects import withStroke

import dgsp

PALETTE = ["#FFADAD", "#A0C4FF", "#CAFFBF", "#FFC6FF"]


def plot_palette():
    palette_rgb = [to_rgb(color) for color in PALETTE]

    plt.scatter([0, 1, 2, 3], [0] * 4, c=palette_rgb, s=400, edgecolors="k", lw=2)


def get_custom_cmap() -> ListedColormap:

    palette_rgb = [to_rgb(color) for color in PALETTE]

    custom_cmap = ListedColormap([(0, 0, 0), (1, 1, 1)] + palette_rgb)

    return custom_cmap


def add_cbar(fig: Figure, ax: Axes, **kwargs) -> Axes:
    """Add a colorbar to an existing figure/axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to get the colorbar from
    ax : matplotlib.axes.Axes
        axes to add the colorbar to

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        tuple of figure and axes with the colorbar added
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical", **kwargs)
    return fig, ax


def draw_self_loop(
    axes: Axes,
    posx: float,
    posy: float,
    size: float,
    rad: float = 0.4,
    offset: float = 0.01,
    mutation_scale: int = 20,
    onearrow: bool = True,
) -> Axes:

    all_arrows_pos = [
        (posx, posy, posx + size, posy, rad),
        (posx + size - offset, posy - offset, posx + size - offset, posy + size, rad),
        (posx + size, posy + size - offset, posx, posy + size - offset, rad),
        (posx + offset, posy + size, posx + offset, posy - offset, rad),
    ]

    for pos_i, arr_pos in enumerate(all_arrows_pos):
        px, py, sx, sy, rad = arr_pos

        style = "-"
        if onearrow:
            if pos_i == len(all_arrows_pos) - 1:
                style = "-|>"
        elif pos_i % 2:
            style = "-|>"

        arrow = FancyArrowPatch(
            (px, py),
            (sx, sy),
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=2,
            color="k",
            connectionstyle=f"arc3,rad={rad}",
        )

        axes.add_patch(arrow)

    return axes


def draw_smaller_self_loop(
    axes: Axes,
    posx: float,
    posy: float,
    size: float,
    rad: float = 0.4,
    offset: float = 0.01,
    mutation_scale: int = 20,
    onearrow: bool = True,
) -> Axes:

    all_arrows_pos = [
        (posx, posy, posx, posy - size, rad),
        (posx - offset, posy - size + offset, posx + size, posy - size, rad),
        (posx + size - offset, posy - size - offset, posx + size, posy, rad),
    ]

    all_arrows_pos = [
        (posx, posy, posx, posy - size, rad),
        (
            posx - offset,
            posy - size + offset,
            posx + size + offset,
            posy - size + offset,
            rad,
        ),
        (posx + size, posy - size, posx + size, posy, rad),
    ]

    for pos_i, arr_pos in enumerate(all_arrows_pos):
        px, py, sx, sy, rad = arr_pos

        style = "-"
        if onearrow:
            if pos_i == len(all_arrows_pos) - 1:
                style = "-|>"
        elif pos_i % 2:
            style = "-|>"

        arrow = FancyArrowPatch(
            (px, py),
            (sx, sy),
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=2,
            color="k",
            connectionstyle=f"arc3,rad={rad}",
        )

        axes.add_patch(arrow)

    return axes


def plot_community_scheme(
    ax: Optional[Axes] = None,
    use_cmap: bool = True,
    title_letter: str = "",
    fontscale: float = 1.2,
    com_names: Optional[np.ndarray] = None,
    com_names_yoffset: float = 0,
    x_names: [str, str] = ["Sending 1", "Sending 2"],
    y_names: [str, str] = ["Receiving 1", "Receiving 2"],
    arrow_colors: Optional[np.ndarray] = None,
    plot_cycle: bool = False,
    override_title: Optional[str] = None,
) -> Optional[Axes]:
    # First plot (A)
    xy_pos = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    pos = {i: xy_pos[i] for i in range(4)}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    colors = "none"
    if use_cmap:
        palette_indices = [3, 2, 1, 0]
        colors = [to_rgba(PALETTE[color_i], alpha=0.5) for color_i in palette_indices]
        # colors = [to_rgba(color, alpha=0.5) for color in PALETTE]

    ax.scatter(
        xy_pos[:, 0],
        xy_pos[:, 1],
        marker="s",
        s=1.5e4,
        color=colors,
        edgecolor="black",
        linewidth=2,
        zorder=1,
    )

    # Definition of arrows
    mid_point = 0.5

    starting_points = [
        (-0.2, -mid_point),
        (mid_point, -0.2),
        (0.2, mid_point),
        (-mid_point, 0.2),
        (0.3, -0.2),
        (-0.3, 0.2),
    ]
    ending_points = [
        (0.2, -mid_point),
        (mid_point, 0.2),
        (-0.2, mid_point),
        (-mid_point, -0.2),
        (-0.2, 0.3),
        (0.2, -0.3),
    ]

    if plot_cycle:
        starting_points = starting_points[:-2]
        ending_points = ending_points[:-2]

    if arrow_colors is None:
        arrow_colors = ["black"] * len(starting_points)

    for start, end, col in zip(starting_points, ending_points, arrow_colors):
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=30,
            linewidth=2,
            facecolor=col,
            edgecolor=col,
        )
        ax.add_patch(arrow)

    # Drawing the rectangles (communities)
    if com_names is None:
        com_names = ["$S_4$", "$S_3$", "$S_2$", "$S_1$"]

    for pos, com_name in zip(xy_pos, com_names):
        ax.text(
            pos[0],
            pos[1] + com_names_yoffset,
            com_name,
            fontsize=24 * fontscale,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Ticks parameters
    ax.set_title(
        title_letter,
        loc="left",
        fontsize=22 * fontscale,
        fontdict={"fontweight": "bold"},
    )

    if override_title is None:
        override_title = "Community structure scheme"
    ax.set_title(override_title, fontsize=20 * fontscale)

    ax.set_xticks(
        np.linspace(-0.5, 0.5, 2),
        labels=x_names,
        fontsize=18 * fontscale,
    )
    ax.set_yticks(
        np.linspace(-0.5, 0.5, 2),
        labels=y_names,
        fontsize=18 * fontscale,
        rotation=90,
        va="center",
    )
    ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Sending communities", fontsize=18 * fontscale)
    ax.set_ylabel("Receiving communities", fontsize=18 * fontscale)
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)

    return ax


def plot_adjacency(
    matrix: np.ndarray,
    ax: Optional[Axes] = None,
    use_cmap: bool = True,
    fontscale: float = 1.2,
    title_letter: str = "",
    override_title: Optional[str] = None,
) -> Optional[Axes]:

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    plot_adj = matrix.copy()

    n_per_com = matrix.shape[0] // 4

    cmap = "binary_r"
    if use_cmap:
        cmap = get_custom_cmap()
        for i in range(4):
            plot_adj[
                i * n_per_com : (i + 1) * n_per_com, i * n_per_com : (i + 1) * n_per_com
            ] = matrix[
                i * n_per_com : (i + 1) * n_per_com, i * n_per_com : (i + 1) * n_per_com
            ] * (
                i + 2
            )

    ax.imshow(plot_adj, cmap=cmap, interpolation="none")
    # ax.pcolormesh(np.zeros_like(plot_adj), cmap=cmap, vmin=0, vmax=1, edgecolors="face")
    # ax.pcolormesh(plot_adj, cmap=cmap, edgecolors="face", linewidth=0)
    # ax.set_ylim(len(plot_adj), 0)
    # axes[1].imshow(graph, cmap="binary_r")

    # Parameters B
    ax.set_title(
        title_letter,
        loc="left",
        fontsize=22 * fontscale,
        fontdict={"fontweight": "bold"},
    )
    if override_title is None:
        override_title = "Graph adjacency matrix"
    ax.set_title(override_title, fontsize=20 * fontscale)

    ax.set_xticks(
        np.arange(n_per_com // 2, matrix.shape[0] + 1, n_per_com),
        labels=["$S_1$", "$S_2$", "$S_3$", "$S_4$"],
        fontsize=18 * fontscale,
    )
    ax.set_yticks(
        np.arange(n_per_com // 2, matrix.shape[0] + 1, n_per_com),
        labels=["$S_1$", "$S_2$", "$S_3$", "$S_4$"],
        fontsize=18 * fontscale,
    )
    return ax


def plot_spectrum(
    matrix: np.ndarray,
    vector_id: int = 0,
    show_n_eig: int = 10,
    write_s: bool = False,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    split_ax: bool = False,
    fix_negative: bool = True,
    fontscale: float = 1.2,
    title_letter: str = "",
    override_title: Optional[str] = None,
    normalize_s: bool = False,
    **kwargs,
) -> Axes:

    # Building the modularity matrix
    modmat = dgsp.modularity_matrix(matrix, null_model="outin")

    U, S, Vh = dgsp.sorted_SVD(modmat, fix_negative=fix_negative)
    V = Vh.T

    r_squared = S[vector_id] ** 2 / (S**2).sum()
    print(f"s (norm) = {S[vector_id] / (2 * matrix.sum())}")
    print(f"R^2 = {r_squared}")

    n_nodes = matrix.shape[0]

    # Starting the plot
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if split_ax:
        gs1 = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=ax.get_subplotspec(), wspace=0.04
        )
        axes = [fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])]
    else:
        axes = [ax] * 2

    node_colors = "k"
    if not fix_negative:
        node_colors = []
        for i in range(show_n_eig):
            # angle = V[:, i] @ U[:, i].T
            angle = U[:, i] @ V[:, i]
            # node_colors.append(angle)
            col_id = 1 - (np.sign(angle) + 1) // 2
            node_colors.append(PALETTE[col_id.astype(int)])

    # Figure A (eigenvalues)

    # Clear the 1st subplot (in the background)

    if split_ax:
        ax.set_xticks([0], labels=[0], fontsize=16 * fontscale, fontdict={"color": "w"})
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=True,
            labelsize=16 * fontscale,
            color="w",
        )

    if split_ax:
        for _, s in ax.spines.items():
            s.set_visible(False)

    if normalize_s:
        S = S / (2 * matrix.sum())

    # axes[0].plot(S, marker="o", lw=4, ms=10, color="k")
    axes[0].plot(S, marker="o", lw=4, ms=20, markeredgewidth=1, color="k")

    if split_ax:
        axes[1].plot(S, marker="o", lw=4, ms=10, color="k")

    if not fix_negative:
        axes[0].scatter(
            np.arange(show_n_eig),
            S[:show_n_eig],
            color=node_colors,
            # color=np.sign(node_colors),
            # cmap="coolwarm",
            edgecolors="k",
            lw=2,
            s=220,
            zorder=2,
        )

    x_id = np.arange(n_nodes)[vector_id]
    if (vector_id >= 0) and (vector_id < show_n_eig):
        if write_s:
            axes[0].plot(
                x_id,
                S[vector_id],
                marker="s",
                lw=4,
                ms=22,
                # color=PALETTE[0],
                color=node_colors[vector_id],
                markeredgecolor="k",
                markeredgewidth=4,
            )
            angle = (V.T @ U)[vector_id, vector_id]
            axes[0].text(
                x_id + 1,
                S[vector_id],
                # f"$\\mu_{{{x_id+1}}}={S[vector_id]:2.2f}, R^2={r_squared:1.2f}$",
                # f"$\\mu_{{{x_id+1}}}={S[vector_id]:2.2f},\,\\mathbf{{v}}_{{{vector_id+1}}}^{{T}}\\mathbf{{u}}_{{{vector_id+1}}}={angle:1.2f}$",
                f"$\\mu_{{{x_id+1}}}={S[vector_id]:2.2f}$\n$\\mathbf{{v}}_{{{vector_id+1}}}^{{T}}\\mathbf{{u}}_{{{vector_id+1}}}={angle:1.2f}$",
                # f"$s_{{{x_id}}}={S[vector_id]:2.2f}, R^2={r_squared:1.2f}$",
                fontsize=18 * fontscale,
                ha="left",
                va="center",
                path_effects=[
                    withStroke(
                        linewidth=3 * fontscale,
                        foreground="w",
                        alpha=0.8,
                    )
                ],
            )
    else:
        axes[1].plot(
            x_id,
            S[vector_id],
            marker="s",
            lw=4,
            ms=14,
            # color=PALETTE[0],
            color=node_colors[vector_id],
            markeredgecolor="k",
            markeredgewidth=2,
        )

    # Plotting the horizontal line at y=0
    ax.set_ylim(axes[0].get_ylim())
    ax.plot([-10, 10], [0] * 2, color="k", ls="--", alpha=0.8, zorder=5)

    if split_ax:
        axes[0].plot([-2, show_n_eig], [0] * 2, color="k", ls="--", alpha=0.8)
        axes[1].plot(
            [len(S) - show_n_eig, len(S) + 2], [0] * 2, color="k", ls="--", alpha=0.8
        )
    else:
        # ax.set_ylim(S[:show_n_eig].min() - 1, S[:show_n_eig].max() + 1)
        ax.set_ylim(0.9 * S[:show_n_eig].min(), 1.1 * S[:show_n_eig].max())

    # From matplotlib examples
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )

    if split_ax:
        axes[0].plot([1], [0], transform=axes[0].transAxes, **kwargs)
        axes[1].plot([0], [0], transform=axes[1].transAxes, **kwargs)

    # Parameters
    if override_title is None:
        override_title = "Eigenspectrum"
    ax.set_title(override_title, fontsize=20 * fontscale)
    axes[0].set_title(
        title_letter,
        loc="left",
        fontsize=22 * fontscale,
        fontdict={"fontweight": "bold"},
    )

    axes[0].set_xlim(-2, show_n_eig - 0.5)

    n_ticks = show_n_eig // 3
    axes[0].set_xticks(
        np.arange(0, show_n_eig, n_ticks), labels=np.arange(0, show_n_eig, n_ticks) + 1
    )

    if split_ax:
        axes[1].set_xlim(len(S) - show_n_eig + 0.5, len(S) + 2)
        axes[1].set_xticks(
            np.arange(len(S) - show_n_eig + n_ticks, len(S) + 2, n_ticks)
        )

    axes[0].spines.right.set_visible(False)

    if split_ax:
        axes[1].spines.left.set_visible(False)

    ax.set_xlabel("Indices $n$", fontsize=18 * fontscale)
    axes[0].set_ylabel("Singular Values $\\mu_n$", fontsize=18 * fontscale)
    # axes[0].set_ylabel("Singular values $s_i$", fontsize=18 * fontscale)

    for ax in axes[:2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[:].set_linewidth(2)

        # ax.set_xticks(np.linspace(-.1, .1, 3), labels=np.linspace(-.1, .1, 3), fontsize=18*fontscale)
        # ax.set_yticks(np.linspace(-.1, .1, 3), labels=np.linspace(-.1, .1, 3), fontsize=18*fontscale)

    axes[0].tick_params(
        left=True,
        bottom=True,
        labelleft=True,
        labelbottom=True,
        labelsize=16 * fontscale,
        width=2,
    )

    if split_ax:
        axes[1].tick_params(
            left=False,
            bottom=True,
            labelleft=False,
            labelbottom=True,
            labelsize=16 * fontscale,
        )

    return ax


def plot_graph_embedding(
    matrix: np.ndarray,
    vector_id: int = 0,
    n_com: int = 4,
    ax: Optional[Axes] = None,
    use_cmap: bool = True,
    cmap: str = "plasma",
    static_color: str = "tab:blue",
    write_label: bool = False,
    write_var: bool = False,
    label_lw: int = 3,
    directed_edges: bool = True,
    edge_alpha: float = 0.02,
    fontscale: float = 1.2,
    title_letter: str = "",
    override_title: Optional[str] = None,
    node_clusers: Optional[np.ndarray] = None,
    **kwargs,
) -> Axes:

    # Building the modularity matrix
    modmat = dgsp.modularity_matrix(matrix, null_model="outin")

    U, S, Vh = dgsp.sorted_SVD(modmat, **kwargs)
    V = Vh.T

    n_nodes = matrix.shape[0]
    n_per_com = n_nodes // n_com

    graph_pos = {i: (U[i, vector_id], V[i, vector_id]) for i in range(n_nodes)}
    labels = {i: "" for i in range(n_nodes)}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if directed_edges:
        graph = nx.DiGraph(matrix)
    else:
        graph = nx.Graph(matrix)

    nx.draw_networkx_edges(graph, pos=graph_pos, alpha=edge_alpha, ax=ax)

    if use_cmap:
        if node_clusers is None:
            palette_rgb = [to_rgb(color) for color in PALETTE]
            colors = [palette_rgb[i // n_per_com] for i in np.arange(n_nodes)]
        else:
            try:
                cmap = plt.get_cmap(cmap, int(node_clusers.max() + 1))
                colors = [cmap(int(i)) for i in node_clusers]
            except ValueError:
                colors = [cmap] * n_nodes
    else:
        colors = static_color

    ax.scatter(
        U[:, vector_id],
        V[:, vector_id],
        s=200,
        color=colors,
        edgecolor="k",
        linewidth=2,
        zorder=2,
    )

    if write_var:
        expl_var = S[vector_id] ** 2 / np.sum(S**2)
        ax.text(
            0.05,
            0.90,
            f"$R^2={expl_var:1.3f}$",
            transform=ax.transAxes,
            fontsize=16 * fontscale,
            path_effects=[
                withStroke(
                    linewidth=label_lw,
                    foreground="w",
                    alpha=0.8,
                )
            ],
        )

        angle = (V.T @ U)[vector_id, vector_id]
        ax.text(
            0.05,
            0.85,
            f"$\\mathbf{{v}}^T\\mathbf{{u}}={angle:1.2f}$",
            transform=ax.transAxes,
            fontsize=16 * fontscale,
            path_effects=[
                withStroke(
                    linewidth=label_lw,
                    foreground="w",
                    alpha=0.8,
                )
            ],
        )

    if write_label:
        for com_i in range(n_com):
            mean_u = np.mean(U[com_i * n_per_com : (com_i + 1) * n_per_com, vector_id])
            mean_v = np.mean(V[com_i * n_per_com : (com_i + 1) * n_per_com, vector_id])
            ax.text(
                mean_u,
                mean_v,
                f"$S_{com_i+1}$",
                fontsize=26 * fontscale,
                color="k",
                # color="w",
                # color=PALETTE[com_i],
                fontweight="bold",
                path_effects=[
                    withStroke(
                        linewidth=label_lw,
                        foreground="w",
                        alpha=0.8,
                        # foreground="k",
                        # foreground=PALETTE[com_i],
                    )
                ],
                ha="center",
                va="center",
                zorder=4,
            )
            # ax.scatter(
            #    mean_u, mean_v, s=1000, marker="o", color="w", alpha=0.8, zorder=3
            # )

    # Parameters
    ax.set_title(
        title_letter,
        loc="left",
        fontsize=22 * fontscale,
        fontdict={"fontweight": "bold"},
    )
    if override_title is None:
        override_title = f"Bimodularity Embedding $(n={{{vector_id+1}}})$"
    ax.set_title(override_title, fontsize=20 * fontscale)
    ax.tick_params(
        left=True,
        bottom=True,
        labelleft=True,
        labelbottom=True,
        labelsize=16 * fontscale,
        width=2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[:].set_linewidth(2)

    ax.set_xticks(
        np.linspace(-0.1, 0.1, 3),
        labels=np.linspace(-0.1, 0.1, 3),
        fontsize=18 * fontscale,
    )
    ax.set_yticks(
        np.linspace(-0.1, 0.1, 3),
        labels=np.linspace(-0.1, 0.1, 3),
        fontsize=18 * fontscale,
    )

    ax.set_xlabel(
        f"Left Singular Vector $\\mathbf{{u}}_{{{vector_id+1}}}$",
        fontsize=18 * fontscale,
    )
    ax.set_ylabel(
        f"Right Singular Vector $\\mathbf{{v}}_{{{vector_id+1}}}$",
        fontsize=18 * fontscale,
    )

    return ax


def plot_bicommunity(
    adjacency,
    send_com,
    receive_com,
    fig=None,
    axes=None,
    graph_pos=None,
    lw=1,
    edge_alpha=0.05,
    draw_arrows=False,
    cmap=None,
    s=80,
    s_scale=5 / 8,
):
    if fig is None:
        fig, axes = plt.subplots(figsize=(10, 10))

    if graph_pos is None:
        graph_pos = nx.spring_layout(nx.DiGraph(adjacency))

    node_pos = np.array(list(graph_pos.values())).T

    if draw_arrows:
        nx.draw_networkx_edges(
            nx.DiGraph(adjacency), pos=graph_pos, alpha=edge_alpha, ax=axes
        )
    else:
        nx.draw_networkx_edges(
            nx.Graph(adjacency), pos=graph_pos, alpha=edge_alpha, ax=axes
        )

    is_in_none = np.logical_and(send_com == 0, receive_com == 0)
    is_in_both = np.logical_and(send_com > 0, receive_com > 0)
    send_only = np.logical_and(send_com > 0, receive_com == 0)
    receive_only = np.logical_and(send_com == 0, receive_com > 0)

    if cmap is None:
        cmap = "RdBu_r"

    axes.scatter(
        node_pos[0, is_in_none],
        node_pos[1, is_in_none],
        s=s * s_scale,
        c="tab:gray",
        edgecolor="k",
        linewidth=lw / 2,
        zorder=2,
    )
    axes.scatter(
        node_pos[0, send_only],
        node_pos[1, send_only],
        s=s,
        c=send_com[send_only],
        # cmap="RdBu_r",
        cmap=cmap,
        edgecolor="k",
        marker="s",
        linewidth=lw,
        zorder=2,
        vmin=-1,
        vmax=1,
    )
    axes.scatter(
        node_pos[0, receive_only],
        node_pos[1, receive_only],
        s=s,
        c=-receive_com[receive_only],
        # cmap="RdBu_r",
        cmap=cmap,
        edgecolor="k",
        marker="D",
        linewidth=lw,
        zorder=2,
        vmin=-1,
        vmax=1,
    )
    axes.scatter(
        node_pos[0, is_in_both],
        node_pos[1, is_in_both],
        s=s,
        c=send_com[is_in_both] - receive_com[is_in_both],
        cmap=cmap,
        # c="w",
        edgecolor="k",
        marker="o",
        linewidth=lw,
        zorder=2,
        vmin=-1,
        vmax=1,
    )

    return fig, axes


def plot_all_bicommunity(
    adjacency,
    send_com,
    receive_com,
    fig=None,
    axes=None,
    layout="embedding",
    scatter_only=False,
    titles=None,
    nrows=1,
    ncols=None,
    draw_legend=True,
    legend_on_ax=False,
    cmap=None,
    gspec_wspace=0,
    gspec_hspace=0.05,
    right_pad=None,
    **kwargs,
):
    if ncols is None:
        ncols = len(send_com) // nrows + (len(send_com) % nrows)

    if right_pad is not None:
        com_gs = GridSpecFromSubplotSpec(
            nrows=nrows,
            ncols=ncols + 1,
            # ncols=len(send_com) // nrows,
            subplot_spec=axes.get_subplotspec(),
            wspace=gspec_wspace,
            hspace=gspec_hspace,
            width_ratios=[1] * ncols + [right_pad],
        )
        com_axes = [
            fig.add_subplot(gs)
            for gi, gs in enumerate(com_gs)
            if (gi + 1) % (ncols + 1)
        ]
    else:
        com_gs = GridSpecFromSubplotSpec(
            nrows=nrows,
            ncols=ncols,
            # ncols=len(send_com) // nrows,
            subplot_spec=axes.get_subplotspec(),
            wspace=gspec_wspace,
            hspace=gspec_hspace,
        )
        com_axes = [fig.add_subplot(gs) for gs in com_gs]
    # axes.set_visible(False)
    axes.axis("off")

    if isinstance(layout, dict):
        graph_pos = layout.copy()
    elif layout == "embedding":
        U, _, Vh = dgsp.sorted_SVD(dgsp.modularity_matrix(adjacency))
        V = Vh.T

        if np.allclose(adjacency, adjacency.T):
            graph_pos = {i: (U[i, 0], V[i, 1]) for i, _ in enumerate(adjacency)}
        else:
            graph_pos = {i: (U[i, 0], V[i, 0]) for i, _ in enumerate(adjacency)}
    else:
        graph_pos = nx.spring_layout(nx.DiGraph(adjacency))

    if titles is None:
        titles = [f"Com {i+1}" for i in range(len(send_com))]

    for i, title in enumerate(titles):
        com_axes[i].set_title(title, fontsize=20)

    if cmap is None:
        cmap = plt.get_cmap("RdBu")
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap, 5)

    for i, (send, receive) in enumerate(zip(send_com, receive_com)):

        com_axes[i].set_facecolor("none")

        if scatter_only:
            com_axes[i].scatter(send, receive)
        else:
            plot_bicommunity(
                adjacency,
                send,
                receive,
                fig=fig,
                axes=com_axes[i],
                graph_pos=graph_pos,
                cmap=cmap,
                **kwargs,
            )

        com_axes[i].spines[:].set_visible(False)
    com_axes[-1].spines[:].set_visible(False)
    com_axes[-1].tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False
    )

    leg_titles = ["Send", "Both", "Receive"]
    markers = ["s", "o", "D"]

    custom_legend = [
        Line2D(
            [0],
            [0],
            color="w",
            marker=markers[i],
            markeredgecolor="k",
            markerfacecolor=cmap(1 + i),
            markersize=10,
        )
        for i, _ in enumerate(leg_titles)
    ]

    if draw_legend:
        if legend_on_ax:
            axes.legend(
                custom_legend,
                leg_titles,
                loc="lower center",
                ncol=3,
                fontsize=20,
                bbox_to_anchor=(0.5, -0.12),
            )
        else:
            fig.legend(
                custom_legend, leg_titles, loc="lower center", ncol=3, fontsize=20
            )

    return fig, axes


def plot_bicommunity_types(
    sending_communities,
    receiving_communities,
    types,
    type_colors=None,
    titles=None,
    fontsize=14,
    fig=None,
    axes=None,
):

    if type_colors is None:
        cmap = plt.get_cmap("Set1")
        type_colors = {t: cmap(i) for i, t in enumerate(types.unique())}

    if titles is None:
        titles = [f"Community {com_i+1}" for com_i in range(len(sending_communities))]

    if axes is None:
        fig, axes = plt.subplots(figsize=(5 * len(sending_communities), 5))

    types_series = pd.Series(types)
    # print(types_series)

    for com_i, (s, r) in enumerate(zip(sending_communities, receiving_communities)):
        types_send = types_series[s > 0].value_counts()
        types_receive = types_series[r > 0].value_counts()

        # print(types_send, types_receive)

        axes[com_i].set_visible(False)
        gs_com = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=axes[com_i].get_subplotspec(), hspace=0.05, wspace=0
        )
        axes_com = [fig.add_subplot(gs_com[i]) for i in range(2)]

        # axes_com[0].set_title(f"Community {com_i+1}", fontsize=20)
        # axes_com[0].set_title(titles[com_i], fontsize=20)

        # axes_com[0].set_title("Sending", fontsize=14)
        # axes_com[1].set_title("Receiving", fontsize=14)

        # axes_com[0].set_ylabel("Sending\n", fontsize=fontsize + 2, labelpad=-30)
        # axes_com[1].set_ylabel("Receiving\n", fontsize=fontsize + 2, labelpad=-30)

        axes_com[0].set_ylabel("Sending\n", fontsize=fontsize + 2, labelpad=0)
        axes_com[1].set_ylabel("Receiving\n", fontsize=fontsize + 2, labelpad=0)

        axes_com[0].yaxis.set_label_position("right")
        axes_com[1].yaxis.set_label_position("right")

        axes_com[0].pie(
            types_send,
            colors=[type_colors[t] for t in types_send.index],
            labels=types_send.index,
            labeldistance=0.6,
            # rotatelabels=True,
            wedgeprops={"edgecolor": "w", "linewidth": 2},
            textprops={"fontsize": fontsize, "ha": "center"},
            # textprops={"size": "smaller"},
        )
        axes_com[1].pie(
            types_receive,
            colors=[type_colors[t] for t in types_receive.index],
            labels=types_receive.index,
            labeldistance=0.6,
            # rotatelabels=True,
            wedgeprops={"edgecolor": "w", "linewidth": 2},
            textprops={"fontsize": fontsize, "ha": "center"},
            # textprops={"size": "smaller"},
        )

    return axes
