# based on TSInterpret Implementation
import os

import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colorbar as cbar
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas
from utils.data_utils import convert_to_label_if_one_hot
from utils.utils import create_path_if_not_exists


def _save_or_show(save_path):
    if save_path is not None:
        path_only = os.path.split(save_path)[0]
        if path_only and not os.path.isdir(path_only):
            os.makedirs(path_only, exist_ok=True)
        plt.savefig(save_path, transparent=True)
        plt.close()
    else:
        plt.show()
    plt.close()


def plot_attribution(item, exp, figsize=(6.4, 4.8), normalize_attribution=True, save_path=None, title=""):
    """
    Plots explanation on the explained Sample.

    Arguments:
        item np.array: instance to be explained,if `mode = time`->`(1,time,feat)`  or `mode = feat`->`(1,feat,time)`.
        exp np.array: explanation ,if `mode = time`->`(time,feat)`  or `mode = feat`->`(feat,time)`.
        figsize (int,int): desired size of plot.
        heatmap bool: 'True' if only heatmap, otherwise 'False'.
        save str: Path to save figure.
    """
    if len(item[0]) == 1:
        test = item[0]
        # if only one-dimensional input
        fig, (axn, cbar_ax) = plt.subplots(
            len(item[0]), 2, sharex=False, sharey=False, figsize=figsize, gridspec_kw={'width_ratios': [40, 1]},
        )

        # Shahbaz: Set color pallete such that negative is red and positive is blue
        my_cmap = sns.diverging_palette(260, 10, as_cmap=True)

        # set min and max to have same absolute values
        extremum = np.max(abs(exp))

        # cbar_ax = fig.add_axes([.91, .3, .03, .4])
        axn012 = axn.twinx()
        if normalize_attribution:
            sns.heatmap(
                exp.reshape(1, -1),
                fmt="g",
                cmap=my_cmap,
                ax=axn,
                yticklabels=False,
                vmin=-1 * extremum,
                vmax=extremum,
                cbar_ax=cbar_ax,
                # cbar_kws={"orientation": "vertical"},
            )
        else:
            sns.heatmap(
                exp.reshape(1, -1),
                fmt="g",
                cmap="viridis",
                ax=axn,
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
        sns.lineplot(
            x=np.arange(0, len(item[0][0].reshape(-1))) + 0.5,
            y=item[0][0].flatten(),
            ax=axn012,
            color="black",
        )
        # plt.subplots_adjust(wspace=0, hspace=0, left=0.02, right=0.95, top=0.95, bottom=0.05)
        cbar_ax.tick_params(labelsize=10)
        plt.title(title)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    else:
        ax011 = []

        fig, axn = plt.subplots(
            len(item[0]), 1, sharex=True, sharey=True, figsize=figsize
        )
        cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

        for channel in item[0]:
            # print(item.shape)
            # ax011.append(plt.subplot(len(item[0]),1,i+1))
            # ax012.append(ax011[i].twinx())
            # ax011[i].set_facecolor("#440154FF")
            axn012 = axn[i].twinx()
            if normalize_attribution:

                sns.heatmap(
                    exp[i].reshape(1, -1),
                    fmt="g",
                    cmap="viridis",
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    ax=axn[i],
                    yticklabels=False,
                    vmin=np.min(exp),
                    vmax=np.max(exp),
                )
            else:
                sns.heatmap(
                    exp[i].reshape(1, -1),
                    fmt="g",
                    cmap="viridis",
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    ax=axn[i],
                    yticklabels=False,
                    vmin=0,
                    vmax=1,
                )

            sns.lineplot(
                x=range(0, len(channel.reshape(-1))),
                y=channel.flatten(),
                ax=axn012,
                color="white",
            )
            plt.xlabel("Time", fontweight="bold", fontsize="large")
            plt.ylabel(f"Feature {i}", fontweight="bold", fontsize="large")
            i = i + 1
        fig.tight_layout(rect=[0, 0, 0.9, 1])
    _save_or_show(save_path)


def plot_multiple_images(data, one_hot, n, shape, figsize=(12, 6)):
    fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
    axes = axes.flatten()

    colors = ['b', 'orange', 'r', 'c', 'm', 'y', 'k']  # Customize as needed

    label_to_color = {}
    labels = convert_to_label_if_one_hot(one_hot)
    for i in range(n):
        if i >= len(axes):
            break  # Stop if n exceeds the available subplots

        label = labels[i]
        if label not in label_to_color:
            label_to_color[label] = colors[len(label_to_color) % len(colors)]

        color = label_to_color[label]

        a = data[i].copy().reshape(1, 1, -1).flatten()
        axes[i].plot(range(len(a)), a, color=color)
        axes[i].set_title(f"Sample {i + 1}, Label: {label}")

    for ax in axes[n:]:
        ax.axis('off')

    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in label_to_color.values()]
    labels = [f'Label: {l}' for l in label_to_color.keys()]
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

    # Or alternatively, to display the legend outside the grid:
    # fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_multiple_images_with_attribution(test_x, pred_y, n, figsize=(12, 6), use_attribution=False,
                                          attributions=None, normalize_attribution=True, title="", save_path=None,
                                          test_y=None, attr_extremum=None, startings=None, shapelet_length=None,
                                          is_title=False):
    """
    Plots multiple subplots with an option to include attribution heatmaps.

    Arguments:
        test_x np.array: Input data for the images.
        test_y np.array: Labels for the images.
        n int: Number of images to display.
        shape tuple: Tuple indicating the number of rows and columns (rows, columns).
        use_attribution bool: Whether to plot attribution heatmaps.
        attributions np.array: The attributions if use_attribution is True (same shape as test_x).
        normalize_attribution bool: Whether to normalize attributions to a common scale.
        title str: Title of the figure.
        save_path: Save the image if given the path
        :param test_y the GT label for the input:
    """
    num_rol_col = math.ceil(math.sqrt(n))
    shape = (num_rol_col, math.ceil(n / num_rol_col))
    # shape = (math.ceil(n / num_rol_col), num_rol_col)
    if figsize is None:
        figsize = (num_rol_col * 3, num_rol_col * 8)

    fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
    if n > 1:
        axes = axes.flatten()  # Flatten axes for easy iteration
    else:
        axes = [axes]
    # Define different colors for different labels
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Customize as needed

    # Create a color palette for the heatmap if attributions are used
    my_cmap = sns.diverging_palette(260, 10, as_cmap=True)
    if test_y is not None:
        test_y = convert_to_label_if_one_hot(test_y)
    pred_y_labels = convert_to_label_if_one_hot(pred_y)
    # print(pred_y_labels, pred_y, test_y)
    label_to_color = {}
    _unique_labels = np.unique(pred_y_labels)
    for _unique_label in _unique_labels:
        if _unique_label not in label_to_color:
            label_to_color[_unique_label] = colors[len(label_to_color) % len(colors)]
    for i in range(n):
        if i >= len(axes):
            break
        label = pred_y_labels[i]

        color = label_to_color[label]
        length = test_x.shape[-1]

        # Plot the data
        channel_data = test_x[i].copy().reshape(1, 1, -1).flatten()
        axes[i].plot(np.arange(len(channel_data)) + 0.5, channel_data, color=color)

        if startings is not None:
            for starting in startings[i]:
                axes[i].axvline(starting + 0.5, color='r', linestyle='--', alpha=0.75)
                axes[i].axvline(starting + shapelet_length - 1 + 0.5, color='r', linestyle='--', alpha=0.75)
        # axes[i].set_xticks([])
        axes[i].tick_params(axis='x', rotation=90)
        # Add attribution heatmap if use_attribution is True
        if use_attribution and attributions is not None:
            axn = axes[i].twinx()  # Create a secondary axis for the heatmap
            extremum = np.max(abs(attributions[i])) if normalize_attribution else None
            if attr_extremum is not None:
                extremum = attr_extremum
            sns.heatmap(
                attributions[i].reshape(1, -1),
                fmt="g",
                cmap=my_cmap if normalize_attribution else "viridis",
                ax=axn,
                yticklabels=False,
                vmin=-extremum if extremum else 0,
                vmax=extremum if extremum else 1,
                cbar=True,  # Suppress color bar in individual subplots
                alpha=0.5  # Add transparency to the heatmap
            )
            # axn.set_yticks([])
            # axn.set_xticks(np.arange(0, 100, 10))
        if is_title:
            if test_y is not None:
                axes[i].set_title(f"Image {i + 1}, Predicted: {label}|GT:{test_y[i]}")
            else:
                axes[i].set_title(f"Image {i + 1}, Predicted: {label}")
    # Add a global color bar if attributions are used
    # if use_attribution:
    #     cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    #     sm = plt.cm.ScalarMappable(cmap=my_cmap if normalize_attribution else "viridis")
    # sm.set_array([])
    # fig.colorbar(sm, cax=cbar_ax)

    for i in range(n, shape[0] * shape[1]):
        axes[i].axis('off')

    # Create a custom legend for labels
    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in label_to_color.values()]
    labels = [f'Label: {l}' for l in label_to_color.keys()]
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.suptitle(title)
    _save_or_show(save_path)


def plot_implet_clusters(implets, cluster_indices, centroids,
                         figsize=(6, 8), save_path=None):
    """
    :param implets: list of array of shape (seq_len, 2), where the first dim is
    features, second dim is importance
    :param cluster_indices: list of list of int, the indices of implets in each
    cluster
    :param centroids: list of array. Shape (seq_len,) or (seq_len, 2).
    """

    k = len(cluster_indices)
    if figsize is None:
        figsize = (6, 2 * k + 2)
    fig, axs = plt.subplots(k, 1, figsize=figsize)
    if k == 1:
        axs = [axs]
    # normalize importance coloring based on max(abs(importance))

    # y-scale limits
    ymin = min([np.min(imp[0]) for imp in implets])
    ymax = max([np.max(imp[0]) for imp in implets])
    lim_y_min = ymin - (ymax - ymin) * 0.1
    lim_y_max = ymax + (ymax - ymin) * 0.1

    def plot(y, v, norm, ax, alpha=1.0, lw=1.0):
        x = np.arange(len(y))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(v)
        lc.set_alpha(alpha)
        lc.set_linewidths(lw)
        ax.add_collection(lc)

    for j in range(k):
        attrs = np.concatenate([np.abs(implets[i][:, 1]).flatten() for i in cluster_indices[j]])
        extremum = np.mean(attrs) + 3 * np.std(attrs)
        norm = mcolors.Normalize(vmin=-extremum, vmax=extremum)
        # plot members
        for t, i in enumerate(cluster_indices[j]):
            plot(implets[i][:, 0] + t / len(cluster_indices[j]) * (ymax - ymin), implets[i][:, 1], norm, axs[j],
                 alpha=0.75)
        # plot centroid
        if len(centroids[0].shape) == 1:  # 1d clustering
            axs[j].plot(centroids[j], lw=3, c='gray')
        else:  # 2d clustering
            plot(centroids[j][:, 0], centroids[j][:, 1], norm, axs[j], lw=3)

        # add colorbar
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])  # Set an empty array for ScalarMappable
        # Add the color bar to the figure
        cbar = fig.colorbar(sm, ax=axs[j])

        axs[j].set_title(f'members: {str(cluster_indices[j])}')
        axs[j].autoscale()
        # axs[j].set_ylim(lim_y_min, lim_y_max)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_implet_clusters_with_instances(implets, instances, centroid=None, lw=1.0, bg_alpha=1.0, implet_alpha = 1.0,
                                        figsize=None, save_path=None, title=None, norm=None):
    """
    :param implets: list of array of shape (seq_len, 2), where the first dim is
    features, second dim is importance
    :param instances list of instances that the implets from: each shape as (len)

    cluster
    """
    # print(implets, instances)
    if figsize is None:
        figsize = (6.0, 4.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    for instance in instances:
        plt.plot(instance.flatten(), color='gray', alpha=bg_alpha, lw=lw)
    # normalize importance coloring based on max(abs(importance))
    if implets:
        attrs = np.concatenate([np.abs(imp[2]).flatten() for imp in implets])
        extremum = np.mean(attrs) + 3 * np.std(attrs)
    else:
        attrs = []
        extremum = 0

    
    norm = mcolors.Normalize(vmin=-extremum, vmax=extremum) if norm is None else norm

    def plot(y, v, norm, start, alpha=1.0, lw=1.0):
        x = np.arange(len(y)) + start
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(v)
        lc.set_alpha(alpha)
        lc.set_linewidths(lw)
        ax.add_collection(lc)
        ax.autoscale()

    for implet in implets:
        inst_num, sub_inst, sub_attr, max_score, best_start, best_end = implet
        # print(result)

        # plt.plot(np.arange(best_start, best_end + 1), sub_inst.flatten(), color='orange')

        # plot members
        # print(sub_attr)
        plot(sub_inst.flatten(), sub_attr.flatten(), norm, best_start, lw=lw, alpha=implet_alpha)

        # if centroid.shape[0] == len(sub_attr) and np.allclose(np.array([sub_inst, sub_attr]).T, centroid):
        #     plot(sub_inst.flatten(), sub_attr.flatten(), norm, best_start, lw=5, alpha=implet_alpha)

    if centroid is not None:
        x_line = (int(instances.shape[-1] * 1.05) + instances.shape[-1] - 1) / 2
        plt.axvline(x_line, color='gray', alpha=0.5)
        plot(centroid[:, 0], centroid[:, 1], norm, int(instances.shape[-1] * 1.05), lw=7)

    # y-scale limits
    # ymin = min([np.min(imp[0]) for imp in implets])
    # ymax = max([np.max(imp[0]) for imp in implets])
    # lim_y_min = ymin - (ymax - ymin) * 0.1
    # lim_y_max = ymax + (ymax - ymin) * 0.1
    #

    #
    # for j in range(k):
    #     attrs = np.concatenate([np.abs(implets[i][:, 1]).flatten() for i in cluster_indices[j]])
    #     extremum = np.mean(attrs) + 3 * np.std(attrs)
    #     norm = mcolors.Normalize(vmin=-extremum, vmax=extremum)
    #     # plot members
    #     for t, i in enumerate(cluster_indices[j]):
    #         plot(implets[i][:, 0] + t/len(cluster_indices[j]) * (ymax - ymin), implets[i][:, 1], norm, axs[j], alpha=0.75)
    #     # plot centroid
    #
    #     # add colorbar
    #     sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    #     sm.set_array([])  # Set an empty array for ScalarMappable
    #     # Add the color bar to the figure
    #     cbar = fig.colorbar(sm, ax=axs[j])
    #
    #     axs[j].set_title(f'members: {str(cluster_indices[j])}')
    #     axs[j].autoscale()l
    #     # axs[j].set_ylim(lim_y_min, lim_y_max)
    plt.axis("off")
    plt.title(title, fontsize=20)
    plt.tight_layout()

    _save_or_show(save_path)

    return norm


def visual_attribution(inst, attr, norm=None, alpha=1.0, lw=1.0, save_path=None, figsize=None, title=None):
    if figsize is None:
        figsize = (6.0, 4.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    extremum = np.mean(attr) + 3 * np.std(attr)
    norm = mcolors.Normalize(vmin=-extremum, vmax=extremum) if norm is None else norm

    # norm = mcolors.Normalize(vmin=0, vmax=1, clip=True)
    def plot(y, v, norm, alpha=1.0, lw=1.0):
        x = np.arange(len(y))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)

        lc.set_array(v)
        lc.set_alpha(alpha)
        lc.set_linewidths(lw)
        ax.add_collection(lc)
        ax.autoscale()

        # sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm, orientation='horizontal', ax=ax)
        # cbar.set_label('Value Range')

    plot(inst.flatten(), attr.flatten(), norm, alpha=alpha, lw=lw)

    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.tight_layout()

    _save_or_show(save_path)

    return norm, extremum


def visual_inst(inst, save_path=None, figsize=None, lw=1.0, title=None):
    if figsize is None:
        figsize = (6.0, 4.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    plt.plot(inst.flatten(), color='gray', alpha=0.60, lw=lw)

    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.tight_layout()

    _save_or_show(save_path)
