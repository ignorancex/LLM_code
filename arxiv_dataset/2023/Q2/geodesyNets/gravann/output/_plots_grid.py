import pathlib

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


def plot_grid_2d(eval_fn1, eval_fn2, fn_names, title, filename, it, limits=(-2, 2), plane='XY'):
    """Plots the potential in 2D for a given cross-section.

    Args:
        eval_fn1: ground truth function taking points to calculate the potential
        eval_fn2: model function taking point to calculate the potential
        fn_names: model function taking point to calculate the potential
        title (str): the title of the plot
        filename (str): the filename/ path
        it (int): iteration number
        limits (float, float): the limits of the axes
        plane (str): the plane cross-section to plot

    Returns:
        None

    """
    left_limit, right_limit = limits
    name_fn1, name_fn2 = fn_names
    values = np.arange(left_limit, right_limit + 0.01, 0.1)
    computation_points = np.array(np.meshgrid(values, values, [0])).T.reshape(-1, 3)
    computation_points = torch.tensor(computation_points)

    eval_res1 = eval_fn1(computation_points)
    eval_res2 = eval_fn2(computation_points)

    difference = (eval_res1 - eval_res2.flatten()).reshape(shape=(len(values), len(values))).cpu().detach().numpy()
    eval_res2 = eval_res2.flatten().reshape(shape=(len(values), len(values))).cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    surf = ax.contourf(values, values, difference, cmap=cm.viridis)
    fig.colorbar(surf, aspect=5, orientation='vertical', shrink=0.5)
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(left_limit, right_limit)
    ax.axis('equal')
    ax.set_title(title)
    fig.savefig(pathlib.Path(filename, f"pot_difference_iter{it}.png"), dpi=300)

    fig, ax = plt.subplots(figsize=(5, 5))
    surf = ax.contourf(values, values, eval_res2, cmap=cm.viridis)
    fig.colorbar(surf, aspect=5, orientation='vertical', shrink=0.5)
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(left_limit, right_limit)
    ax.axis('equal')
    ax.set_title(title)
    fig.savefig(pathlib.Path(filename, f"pot_{name_fn2}_iter{it}.png"), dpi=300)


def plot_quiver(X, Y, xy, title, filename, labels=("$x$", "$y$"), limits=(-2, 2), plot_rectangle=False, vertices=None,
                coordinate=None):
    """Plots a quiver plot, given the accelerations.

    Args:
        X: first coordinate array
        Y: second coordinate array
        xy: acceleration array
        title (str): the title of the plot
        filename (str): the filename/ path
        labels (str, str): the axes label names
        limits (float, float): the limits of the axes
        plot_rectangle (bool, optional): plot a rectangle centered on the origin
        vertices: the vertices of the polyhedron, if given, plot the polyhedron inside
        coordinate: the coordinate of the cross-section (lazy parameter, could be avoided)

    Returns:
        None

    """
    print("Plotting Quiver")
    fig, ax = plt.subplots(figsize=(5, 5))

    U = np.reshape(xy[:, 0], (len(X), -1))
    V = np.reshape(xy[:, 1], (len(Y), -1))

    ax.quiver(X, Y, U, V, angles='xy', linewidth=0.1, color='b', pivot='mid', units='xy')

    if plot_rectangle:
        rect = patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    left, right = limits
    ax.set_xlim(left, right)
    ax.set_ylim(left, right)

    ax.axis('equal')

    xl, yl = labels
    ax.set_title(title)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

    fig.savefig(filename, dpi=300)


