import matplotlib.colors as colors
import numpy as np
import torch
from matplotlib import pyplot as plt

from gravann.util import get_target_point_sampler


def _plot_compare_acceleration(sample, label1, label2, **kwargs):
    """Plots the relative error of the computed acceleration between a model/ ground truth and a model/ ground truth

    Args:
        sample (str): the sample body's name
        label1: label function one
        label2: label function two

    Keyword Args:
        plane (str): Either "XY","XZ" or "YZ". Defines the cross-section. Defaults to "XY".
        altitude (float): Altitude to compute error at. Defaults to 0.1.
        save_path (str): Pass to store plot, if none will display. Defaults to None.
        N (int): Number of points to sample. Defaults to 5000.
        logscale (bool): Logscale errors. Defaults to False.


    Raises:
        ValueError: On wrong input

    Returns:
        plt.Figure: created plot
    """
    N = kwargs.get('N', 5000)
    altitude = kwargs.get('altitude', 0.1)
    plane = kwargs.get('plane', 'XY')
    logscale = kwargs.get('logscale', False)
    save_path = kwargs.get('save_path', None)

    points = get_target_point_sampler(N, method="altitude", bounds=[altitude],
                                      limit_shape_to_asteroid=f"./3dmeshes/{sample}.pk",
                                      replace=False)()
    if plane == "XY":
        cut_dim = 2
        cut_dim_name = "z"
        x_dim = 0
        y_dim = 1
    elif plane == "XZ":
        cut_dim = 1
        cut_dim_name = "y"
        x_dim = 0
        y_dim = 2
    elif plane == "YZ":
        cut_dim = 0
        cut_dim_name = "x"
        x_dim = 1
        y_dim = 2
    else:
        raise ValueError("Plane has to be either XY, XZ or YZ")

    # Left and Right refer to values < 0 and > 0 in the non-crosssection dimension

    print("Splitting in left / right hemisphere")
    points_left = points[points[:, cut_dim] < 0]
    points_right = points[points[:, cut_dim] > 0]

    print("Left: ", len(points_left), " points.")
    print("Right: ", len(points_right), " points.")

    label2_values_left, label1_values_left, relative_error_left = [], [], []
    label2_values_right, label1_values_right, relative_error_right = [], [], []

    # Compute accelerations in left points, then right points
    # for both network and mascon model
    batch_size = 100

    for idx in range((len(points_left) // batch_size) + 1):
        indices = list(range(idx * batch_size, np.minimum((idx + 1) * batch_size, len(points_left))))
        label1_values_left.append(label1(points_left[indices]).detach())
        label2_values_left.append(label2(points_left[indices]).detach())
        torch.cuda.empty_cache()

    for idx in range((len(points_right) // batch_size) + 1):
        indices = list(range(idx * batch_size, np.minimum((idx + 1) * batch_size, len(points_right))))
        label1_values_right.append(label1(points_right[indices]).detach())
        label2_values_right.append(label2(points_right[indices]).detach())
        torch.cuda.empty_cache()

    # Accumulate all results
    label1_values_left = torch.cat(label1_values_left)
    label2_values_left = torch.cat(label2_values_left)
    label1_values_right = torch.cat(label1_values_right)
    label2_values_right = torch.cat(label2_values_right)

    # Compute relative errors for each hemisphere (left, right)
    relative_error_left = (torch.sum(torch.abs(label2_values_left - label1_values_left), dim=1) /
                           torch.sum(torch.abs(label1_values_left + 1e-8), dim=1)).cpu().numpy()
    relative_error_right = (torch.sum(torch.abs(label2_values_right - label1_values_right), dim=1) /
                            torch.sum(torch.abs(label1_values_right + 1e-8), dim=1)).cpu().numpy()

    min_err = np.minimum(np.min(relative_error_left),
                         np.min(relative_error_right))
    max_err = np.maximum(np.max(relative_error_left),
                         np.max(relative_error_right))

    norm = colors.Normalize(vmin=min_err, vmax=max_err)

    if logscale:
        relative_error_left = np.log(relative_error_left)
        relative_error_right = np.log(relative_error_right)

    # Get X,Y coordinates of analyzed points
    X_left = points_left[:, x_dim].cpu().numpy()
    Y_left = points_left[:, y_dim].cpu().numpy()

    X_right = points_right[:, x_dim].cpu().numpy()
    Y_right = points_right[:, y_dim].cpu().numpy()

    # Plot left side stuff
    fig = plt.figure(figsize=(8, 4), dpi=100, facecolor='white')
    fig.suptitle("Relative acceleration error in " +
                 plane + " cross section", fontsize=12)
    ax = fig.add_subplot(121, facecolor="black")

    p = ax.scatter(X_left, Y_left, c=relative_error_left,
                   cmap="plasma", alpha=1.0, s=int(N * 0.0005 + 0.5), norm=norm)

    cb = plt.colorbar(p, ax=ax)
    cb.ax.tick_params(labelsize=7)
    if logscale:
        cb.set_label('Log(Relative Error)', rotation=270, labelpad=15)
    else:
        cb.set_label('Relative Error', rotation=270, labelpad=15)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel(plane[0], fontsize=8)
    ax.set_ylabel(plane[1], fontsize=8)
    ax.set_title(cut_dim_name + " < 0")
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', 'box')
    ax.annotate("Label-1 Acc. Mag=" + str(torch.mean(torch.sum(torch.abs(label1_values_left), dim=1)).cpu().numpy()) +
                "\n" + "Label-2 Acc. Mag=" +
                str(torch.mean(
                    torch.sum(torch.abs(label2_values_left), dim=1)).cpu().numpy()),
                (-0.95, 0.8), fontsize=8, color="white")

    # Plot right side stuff
    ax = fig.add_subplot(122, facecolor="black")

    p = ax.scatter(X_right, Y_right, c=relative_error_right,
                   cmap="plasma", alpha=1.0, s=int(N * 0.0005 + 0.5), norm=norm)

    cb = plt.colorbar(p, ax=ax)
    cb.ax.tick_params(labelsize=7)
    if logscale:
        cb.set_label('Log(Relative Error)', rotation=270, labelpad=15)
    else:
        cb.set_label('Relative Error', rotation=270, labelpad=15)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel(plane[0], fontsize=8)
    ax.set_ylabel(plane[1], fontsize=8)
    ax.set_title(cut_dim_name + " > 0")
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', 'box')
    ax.annotate("Label-1 Acc. Mag=" + str(torch.mean(torch.sum(torch.abs(label1_values_right), dim=1)).cpu().numpy()) +
                "\n" + "Label-2 Acc. Mag=" +
                str(torch.mean(
                    torch.sum(torch.abs(label2_values_right), dim=1)).cpu().numpy()),
                (-0.95, 0.8), fontsize=8, color="white")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()

    return ax, label1_values_right
