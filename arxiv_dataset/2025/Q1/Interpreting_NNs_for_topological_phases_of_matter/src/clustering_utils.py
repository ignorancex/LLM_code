import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def recreate_fig_4(load_path, save_path):
    smart_model_folder = load_path.joinpath("well_generalizing_misleading_CAM")
    dumb_model_folder = load_path.joinpath("poorly_generalizing_good_CAM")
    model_paths = {
        "smart": smart_model_folder,
        "dumb": dumb_model_folder,
    }
    # Fig. 4
    # Setting the disorders and layers to plot
    disorders_to_plot = ["0", "0.15", "0.5", "1.0", "2.0"]
    layer_to_plot = "".join(
        [
            # 'clear',
            # 'conv2d1',
            # 'conv2d2',
            # 'cnn_seq_1',
            # 'conv2d3',
            # 'cnn_seq_2',
            "avg_pool",
        ]
    )
    pca_data = {}
    for model_type, model_path in model_paths.items():
        pca_dict = {}
        for W in disorders_to_plot:
            pca_p_tmp = model_path.joinpath(f"PCA/W={W}")
            df_pca = pd.read_csv(pca_p_tmp.joinpath(f"{layer_to_plot}_PCA_W={W}.csv"))
            df_pca.drop(
                columns=[
                    "val_x",
                    "val_y",
                    "val_class",
                    "val_pred",
                    "trn_x",
                    "trn_y",
                    "trn_class",
                    "trn_pred",
                ],
                inplace=True,
            )
            pca_dict[W] = df_pca
        pca_data[model_type] = pca_dict
    umap_data = {}
    for model_type, model_path in model_paths.items():
        df_umap = pd.read_csv(model_path.joinpath(f"UMAP/{layer_to_plot}_UMAP.csv"))
        umap_dict = {}
        for W in disorders_to_plot:
            df_filtered = df_umap[df_umap["class"].str.endswith(f"W={W}")].copy()
            df_filtered.loc[:, "class"] = df_filtered["class"].str.replace(
                f"_W={W}", "", regex=True
            )
            df_filtered.loc[:, "pred"] = df_filtered["pred"].str.replace(
                f"_W={W}", "", regex=True
            )
            umap_dict[W] = df_filtered.copy()
        umap_data[model_type] = umap_dict
    data_joined = {
        "PCA": pca_data,
        # "UMAP": umap_data
    }
    clean_system_data = {}
    for tech_type, _ in data_joined.items():
        clean_system_data[tech_type] = {}
        for model_type, model_path in model_paths.items():
            if tech_type == "PCA":
                df = pd.read_csv(
                    model_path.joinpath(
                        f"{tech_type}/W=0/{layer_to_plot}_{tech_type}_W=0.csv"
                    )
                )
                df.drop(
                    columns=[
                        "val_x",
                        "val_y",
                        "val_class",
                        "val_pred",
                        "trn_x",
                        "trn_y",
                        "trn_class",
                        "trn_pred",
                    ],
                    inplace=True,
                )
            else:
                df = umap_data[model_type]["0"]
            clean_system_data[tech_type][model_type] = df
    # Plotting
    # Toggle use of latex serif font
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fontsize = 24
    ticksize = 24
    ticks_num = 5
    lw = 2.5
    cmap = plt.cm.get_cmap("tab20c")
    left_labels = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
    ]
    # left_labels = [
    #     "(a)\nbadly-generalizing\nmodel",
    #     "(b)\nwell-generalizing\nmodel",
    #     "(c)\nbadly-generalizing\nmodel",
    #     "(d)\nwell-generalizing\nmodel",
    # ]

    subfigs = plt.figure(layout="constrained", figsize=(20, 7.5))

    colors_shift = 0
    mult = 4
    pl = 3
    s_s = 10

    for num_data, (data_type, data_dict) in enumerate(data_joined.items()):
        subfig = subfigs
        axs = subfig.subplots(2, len(disorders_to_plot), sharex=True, sharey=True)
        for num_model, (model_type, model_data) in enumerate(data_dict.items()):
            for num_disorder, (disorder, disorder_data) in enumerate(
                model_data.items()
            ):
                ax = axs[num_model, num_disorder]

                clean_data = clean_system_data[data_type][model_type]
                levels_p_cl, categories_p_cl = pd.factorize(clean_data["pred"])
                categories_p_cl = categories_p_cl.tolist()
                categories_p_cl = [c.lower() + " w/o disorder" for c in categories_p_cl]
                levels_p_cl *= mult
                levels_p_cl += pl + colors_shift * mult
                colors_p_cl = [cmap(i) for i in levels_p_cl]
                handles_p_cl = [
                    mpl.patches.Patch(color=cmap(i), label=c)
                    for i, c in zip([3, 7], categories_p_cl)
                ]

                levels_tr_cl, categories_tr_cl = pd.factorize(clean_data["class"])
                levels_tr_cl *= mult
                levels_tr_cl += pl + colors_shift * mult
                colors_tr_cl = [cmap(i) for i in levels_tr_cl]

                levels_p, categories_p = pd.factorize(disorder_data["pred"])
                categories_p = categories_p.tolist()
                categories_p = [c.lower() + " w/ disorder" for c in categories_p]
                levels_p *= mult
                levels_p += colors_shift * mult
                colors_p = [cmap(i) for i in levels_p]
                handles_p = [
                    mpl.patches.Patch(color=cmap(i), label=c)
                    for i, c in zip(np.unique(levels_p), categories_p)
                ]

                levels_tr, categories_tr = pd.factorize(disorder_data["class"])
                levels_tr *= mult
                levels_tr += colors_shift * mult
                colors_tr = [cmap(i) for i in levels_tr]

                categories_tr = categories_tr.tolist()
                categories_tr = [c.lower() + " w/ disorder" for c in categories_tr]
                handles_tr = [
                    mpl.patches.Patch(color=cmap(i), label=c)
                    for i, c in zip([0, 4], categories_tr)
                ]

                ax.scatter(
                    clean_data["x"],
                    clean_data["y"],
                    c=colors_p_cl,
                    cmap=cmap,
                    s=s_s,
                    marker="o",
                    alpha=0.5,
                )
                if num_disorder > 0:
                    ax.scatter(
                        disorder_data["x"],
                        disorder_data["y"],
                        c=colors_p,
                        cmap=cmap,
                        s=s_s,
                        marker="o",
                        alpha=0.5,
                    )

                if num_model == 0:
                    ax.text(
                        0.5,
                        1,
                        "Predictions",
                        size=1.1 * fontsize,
                        ha="center",
                        va="bottom",
                        transform=ax.transAxes,
                    )

                if num_data == 0:
                    # ax.set_ylim(-0.3, 1.2)
                    ax.set_ylim(-0.3, 1.5)

                    if num_model == 0:
                        axin = ax.inset_axes([0.57, 0.42, 0.4, 0.45])
                        axin.set_ylim(-0.12, 0.6)

                    elif num_model == 1:
                        axin = ax.inset_axes([0.09, 0.42, 0.4875, 0.45])
                        axin.yaxis.set_major_locator(plt.FixedLocator([0, 1]))
                        # axin.set_ylim(-0.3, 1.2)
                        axin.set_ylim(-0.3, 1.5)

                    axin.text(
                        0.5,
                        1,
                        "Labels",
                        size=1.1 * fontsize,
                        ha="center",
                        va="bottom",
                        transform=axin.transAxes,
                    )

                else:
                    # ax.set_ylim(-25, 23)
                    ax.set_ylim(-20, 28.5)
                    # ax.set_xlim(-8, 17.5)
                    ax.set_xlim(-8.5, 17.5)

                    if num_model == 0:
                        axin = ax.inset_axes([0.32, 0.09, 0.4, 0.4])
                        axin.text(
                            0.5,
                            1,
                            "Labels",
                            size=1.1 * fontsize,
                            ha="center",
                            va="bottom",
                            transform=axin.transAxes,
                        )

                    elif num_model == 1:
                        # axin = ax.inset_axes([0.33, 0.08, 0.365, 0.3875])
                        axin = ax.inset_axes([0.49, 0.38, 0.33, 0.33])
                        axin.xaxis.set_major_locator(plt.FixedLocator([0, 10]))
                        axin.set_xticklabels([0, 10], size=0.5 * ticksize)
                        if num_disorder in [7, 8]:
                            axin.text(
                                0.45,
                                1,
                                "Labels",
                                size=1 * ticksize,
                                ha="center",
                                va="bottom",
                                transform=axin.transAxes,
                            )
                        else:
                            axin.text(
                                0.5,
                                1,
                                "Labels",
                                size=1 * ticksize,
                                ha="center",
                                va="bottom",
                                transform=axin.transAxes,
                            )

                    axin.set_ylim(-19, 28.5)

                axin.scatter(
                    clean_data["x"],
                    clean_data["y"],
                    c=colors_tr_cl,
                    cmap=cmap,
                    s=s_s,
                    marker=".",
                    alpha=0.75,
                )
                if num_disorder > 0:
                    axin.scatter(
                        disorder_data["x"],
                        disorder_data["y"],
                        c=colors_tr,
                        cmap=cmap,
                        s=s_s,
                        marker=".",
                        alpha=0.75,
                    )
                axin.tick_params(
                    axis="both", which="both", direction="in", labelsize=0.8 * ticksize
                )
                # ax.set_xticks([])
                # ax.set_yticks([])
                if num_data == 0:
                    ax.set_xlabel(f"PC1", size=ticksize)
                    ax.set_ylabel(f"PC2", size=ticksize, labelpad=5)
                else:
                    ax.set_xlabel(f"UMAP1", size=ticksize)
                    ax.set_ylabel(f"UMAP2", size=ticksize, labelpad=-5)
                ax.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    labelsize=ticksize,
                    size=0.3 * ticksize,
                )
                if num_model == 0 and num_data == 0:
                    if num_disorder < len(disorders_to_plot) - 1:
                        ax.set_title(
                            r"$W \,/\,\textit{{{}}} = {}$".format("w", disorder),
                            size=1.2 * fontsize,
                            pad=35,
                        )
                    else:
                        ax.set_title(
                            r"$W \,/\,\textit{{{}}} \geq {}$".format("w", disorder),
                            size=1.2 * fontsize,
                            pad=35,
                        )
                if num_disorder == 0:
                    ax.text(
                        -0.3,
                        0.5,
                        left_labels[num_model + num_data * 2],
                        size=1.2 * fontsize,
                        ha="center",
                        va="center",
                        rotation=0,
                        transform=ax.transAxes,
                    )

        for ax in axs.flatten():
            ax.label_outer()

    subfigs.legend(
        handles=handles_p_cl + handles_tr,
        labels=categories_p_cl + categories_tr,
        loc="center",
        ncol=4,
        fontsize=fontsize,
        title="",
        title_fontsize=0.8 * fontsize,
        bbox_to_anchor=(0.5, 1.1),
    )
    subfigs.savefig(save_path.joinpath("Fig_4.pdf"), bbox_inches="tight")
    save_path.joinpath("PNG").mkdir(exist_ok=True, parents=True)
    subfigs.savefig(
        save_path.joinpath("PNG").joinpath("Fig_4.png"), bbox_inches="tight"
    )


def recreate_figs_12_13(load_path, save_path):
    # Figs. 12/13
    smart_model_folder = load_path.joinpath("well_generalizing_misleading_CAM")
    dumb_model_folder = load_path.joinpath("poorly_generalizing_good_CAM")
    cam_ind = 26
    model_paths = {
        "smart": smart_model_folder,
        "dumb": dumb_model_folder,
    }

    # Model choice
    # Setting the disorders and layers to plot
    disorders_to_plot = ["0", "0.15", "0.5", "1.0", "2.0"]
    layers_to_plot = [
        "clear",
        "conv2d1",
        # 'conv2d2',
        "cnn_seq_1",
        # 'conv2d3',
        "cnn_seq_2",
        "avg_pool",
    ]
    layer_new_names = [
        "Raw data",
        "Conv1",
        "Conv2",
        "Conv3",
        "GAP",
    ]
    umap_data = {}
    umap_viz = {}
    for model_type, model_path in model_paths.items():
        umap_dict = {}
        umap_viz[model_type] = {}
        for W in disorders_to_plot:
            umap_dict[W] = {}
            umap_viz[model_type][W] = {}
            for lay_num, chosen_layer in enumerate(layers_to_plot):
                df_umap = pd.read_csv(
                    model_path.joinpath(f"UMAP/{chosen_layer}_UMAP.csv")
                )
                df_filtered = df_umap[df_umap["class"].str.endswith(f"W={W}")].copy()
                df_filtered.loc[:, "class"] = df_filtered["class"].str.replace(
                    f"_W={W}", "", regex=True
                )
                df_filtered.loc[:, "pred"] = df_filtered["pred"].str.replace(
                    f"_W={W}", "", regex=True
                )
                umap_dict[W][chosen_layer] = {
                    "df": df_filtered.copy(),
                    "name": layer_new_names[lay_num],
                }
                umap_viz[model_type][W][chosen_layer] = {
                    "df": "pandas.DataFrame",
                    "name": layer_new_names[lay_num],
                }
        umap_data[model_type] = umap_dict

    clean_system_data = {}
    for model_type, model_path in model_paths.items():
        clean_system_data[model_type] = {}
        for chosen_layer in layers_to_plot:
            df = umap_data[model_type]["0"][chosen_layer]["df"]
            clean_system_data[model_type][chosen_layer] = df
            # Delete the dictionary entry for disorder 0
            del umap_data[model_type]["0"][chosen_layer]
            del umap_viz[model_type]["0"][chosen_layer]
        del umap_viz[model_type]["0"]
        del umap_data[model_type]["0"]

    # Plotting
    # Toggle use of latex serif font
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fontsize = 24
    ticksize = 24
    ticks_num = 5
    lw = 2.5
    cmap = plt.cm.get_cmap("tab20c")
    # subfig_labels = [
    #     "(a)",
    #     "(b)",
    # ]
    subfig_labels = [
        "(a) well-generalizing model",
        "(b) badly-generalizing model",
    ]
    data_type = "UMAP"
    no_ticks = False
    do_insets = True

    fig1 = plt.figure(layout="constrained", figsize=(20, 20))
    fig2 = plt.figure(layout="constrained", figsize=(20, 20))
    subfigs = [fig1, fig2]

    colors_shift = 0
    mult = 4
    pl = 3
    s_s = 10

    color_mis = 4

    for num_model, (model_type, model_data) in enumerate(umap_data.items()):
        subfig = subfigs[num_model]
        axs = subfig.subplots(len(layers_to_plot), len(disorders_to_plot) - 1)
        for num_disorder, (disorder, disorder_data) in enumerate(model_data.items()):
            for num_layer, (chosen_layer, layer_data) in enumerate(
                disorder_data.items()
            ):
                layer_df = layer_data["df"]
                new_layer_name = layer_data["name"]

                ax = axs[num_layer, num_disorder]

                clean_data = clean_system_data[model_type][chosen_layer]
                levels_p_cl, categories_p_cl = pd.factorize(clean_data["pred"])
                categories_p_cl = categories_p_cl.tolist()
                categories_p_cl = [c.lower() + " w/o disorder" for c in categories_p_cl]
                levels_p_cl *= mult
                levels_p_cl += pl + colors_shift * mult
                colors_p_cl = [cmap(i) for i in levels_p_cl]
                handles_p_cl = [
                    mpl.patches.Patch(color=cmap(i), label=c)
                    for i, c in zip(np.unique(levels_p_cl), categories_p_cl)
                ]

                levels_tr_cl, categories_tr_cl = pd.factorize(clean_data["class"])
                levels_tr_cl *= mult
                levels_tr_cl += pl + colors_shift * mult
                colors_tr_cl = [cmap(i) for i in levels_tr_cl]

                levels_p, categories_p = pd.factorize(layer_df["pred"])
                levels_tr, categories_tr = pd.factorize(layer_df["class"])

                misclassified = np.abs(levels_tr - levels_p)
                categories_mis = np.array(["Misclassified"])
                colors_mis = np.array([cmap(color_mis) for i in misclassified])
                handles_mis = [
                    mpl.patches.Patch(color=cmap(color_mis), label="Misclassified")
                ]

                categories_p = categories_p.tolist()
                categories_p = [c.lower() + " w/ disorder" for c in categories_p]
                levels_p *= mult
                levels_p += colors_shift * mult
                colors_p = np.array([cmap(i) for i in levels_p])
                handles_p = [
                    mpl.patches.Patch(color=cmap(i), label=c)
                    for i, c in zip(np.unique(levels_p), categories_p)
                ]

                levels_tr *= mult
                levels_tr += colors_shift * mult
                colors_tr = np.array([cmap(i) for i in levels_tr])

                categories_tr = categories_tr.tolist()
                categories_tr = [c.lower() + " w/ disorder" for c in categories_tr]
                handles_tr = [
                    mpl.patches.Patch(color=cmap(i), label=c)
                    for i, c in zip(np.unique(levels_tr), categories_tr)
                ]

                ax.scatter(
                    clean_data["x"],
                    clean_data["y"],
                    c=colors_p_cl,
                    cmap=cmap,
                    s=s_s,
                    marker="o",
                    alpha=0.5,
                    zorder=1,
                )
                ax.scatter(
                    layer_df["x"],
                    layer_df["y"],
                    c=colors_p,
                    cmap=cmap,
                    s=s_s,
                    marker="o",
                    alpha=0.5,
                    zorder=2,
                )

                if not do_insets:
                    ax.scatter(
                        layer_df["x"].to_numpy()[misclassified.astype(bool)],
                        layer_df["y"].to_numpy()[misclassified.astype(bool)],
                        c=colors_mis[misclassified.astype(bool)],
                        cmap=cmap,
                        s=s_s,
                        marker="o",
                        alpha=0.5,
                        zorder=3,
                    )

                ax.set_xlabel(f"{data_type}1", size=ticksize)

                # Fist subfigure (two columns)
                """
                Numeration:
                num_model - Type of model (dumb/smart) - subfigure (two columns) - 0/1
                num_disorder - Disorder strength (0.15, 3.0) - subplot column - 0/1
                num_layer - Layer number - subplot row
                """
                if num_model == 0:
                    # Subfigure 1
                    if num_layer == 0:
                        # Row 1

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=5)

                        # Inset axes for the Labels
                        if do_insets:
                            axin = ax.inset_axes([0.45, 0.35, 0.5, 0.5])
                            axin.set_ylim(1, 17)
                            axin.set_xlim(-7, 15)

                        # Row-specific settings
                        ax.set_ylim(1, 17)
                        ax.set_xlim(-7, 15)

                        ax.set_xlabel("")

                    elif num_layer == 1:
                        # Row 2

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=5)

                        # Inset axes for the Labels
                        if do_insets:
                            # axin = ax.inset_axes([0.15, 0.15, 0.5, 0.5])
                            axin = ax.inset_axes([0.5, 0.35, 0.45, 0.45])
                            # axin.set_ylim(-13, 10)
                            # axin.set_xlim(8, 15)

                        # Row-specific settings
                        # ax.set_ylim(-13, 10)
                        # ax.set_xlim(8, 12)

                        ax.set_xlabel("")

                    elif num_layer == 2:
                        # Row 3

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=10)

                        # Inset axes for the Labels
                        if do_insets:
                            # axin = ax.inset_axes([0.08, 0.46, 0.45, 0.42])
                            axin = ax.inset_axes([0.08, 0.48, 0.45, 0.4])
                            # axin.set_ylim(-18, 32)
                            # axin.set_xlim(-7.5, 17.5)

                        # Row-specific settings
                        # ax.set_ylim(-18, 26)
                        # ax.set_xlim(-7.5, 17.5)

                        ax.set_xlabel("")

                    elif num_layer == 3:
                        # Row 4

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=5)

                        # Inset axes for the Labels
                        if do_insets:
                            axin = ax.inset_axes([0.15, 0.15, 0.45, 0.45])
                            # axin.set_ylim(-18, 32)
                            # axin.set_xlim(-7.5, 17.5)

                        # Row-specific settings
                        # ax.set_ylim(-18, 26)
                        # ax.set_xlim(-7.5, 17.5)

                        ax.set_xlabel("")

                    elif num_layer == 4:
                        # Row 5

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=-10)

                        # Inset axes for the Labels
                        if do_insets:
                            axin = ax.inset_axes([0.24, 0.12, 0.45, 0.45])
                            axin.set_ylim(-11, 22)
                            # axin.set_xlim(-7.5, 17.5)

                        # Row-specific settings
                        # ax.set_ylim(-18, 26)
                        # ax.set_xlim(-7.5, 17.5)

                elif num_model == 1:
                    # Subfigure 2
                    if num_layer == 0:
                        # Row 1

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=5)

                        # Inset axes for the Labels
                        if do_insets:
                            # axin = ax.inset_axes([0.45, 0.35, 0.5, 0.5])
                            axin = ax.inset_axes([0.46, 0.1, 0.45, 0.5])
                            # ax.yaxis.set_major_locator(plt.FixedLocator([0, 2.5, 5, 7.5, 10]))

                        # Row-specific settings
                        # ax.set_ylim(-0.1, 11)
                        # ax.set_xlim(-5, 20)
                        # ax.yaxis.set_major_locator(plt.FixedLocator([0, 2.5, 5, 7.5, 10]))
                        # ax.xaxis.set_major_locator(plt.FixedLocator([-5, 0, 5, 10, 15, 20]))

                        ax.set_xlabel("")

                    elif num_layer == 1:
                        # Row 2

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=10)

                        # Inset axes for the Labels
                        if do_insets:
                            # axin = ax.inset_axes([0.56, 0.44, 0.42, 0.42])
                            # axin = ax.inset_axes([0.15, 0.15, 0.42, 0.42])
                            axin = ax.inset_axes([0.5, 0.35, 0.45, 0.45])
                            # axin.set_ylim(-5, 14)

                        # Row-specific settings
                        # ax.set_ylim(-5, 12)

                        ax.set_xlabel("")

                    elif num_layer == 2:
                        # Row 3

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=5)

                        # Inset axes for the Labels
                        if do_insets:
                            # axin = ax.inset_axes([0.28, 0.12, 0.5, 0.5])
                            axin = ax.inset_axes([0.12, 0.12, 0.5, 0.5])
                            # axin.set_ylim(-15, 32)

                        # Row-specific settings
                        # ax.set_ylim(-11, 25)

                        ax.set_xlabel("")

                    elif num_layer == 3:
                        # Row 4

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=5)

                        # Inset axes for the Labels
                        if do_insets:
                            # axin = ax.inset_axes([0.35, 0.375, 0.5, 0.5])
                            axin = ax.inset_axes([0.5, 0.43, 0.45, 0.45])
                            # axin.set_ylim(-15, 32)

                        # Row-specific settings
                        # ax.set_ylim(-11, 25)

                        ax.set_xlabel("")

                    elif num_layer == 4:
                        # Row 5
                        ax.set_ylim(-20, 28.5)

                        ax.set_ylabel(f"{data_type}2", size=ticksize, labelpad=-10)

                        # Inset axes for the Labels
                        if do_insets:
                            # ax.set_ylim(-25, 23)
                            ax.set_ylim(-15, 28.5)
                            ax.set_xlim(-8, 17.5)

                            # axin = ax.inset_axes([0.325, 0.09, 0.4, 0.4])
                            axin = ax.inset_axes([0.47, 0.32, 0.35, 0.35])
                            # axin = ax.inset_axes([0.33, 0.08, 0.365, 0.3875])
                            # axin.set_ylim(-15, 32)
                            axin.set_ylim(-19, 28.5)

                        if num_disorder in [1, 2]:
                            axin.text(
                                0.45,
                                1,
                                "Labels",
                                size=1 * fontsize,
                                ha="center",
                                va="bottom",
                                transform=axin.transAxes,
                            )

                        # Row-specific settings
                        # ax.set_ylim(-11, 25)

                if num_disorder in [1, 2, 3]:
                    # Remove the yaxis label and ticks
                    ax.set_ylabel("")
                    ax.set_yticklabels([])

                if do_insets:
                    axin.scatter(
                        clean_data["x"],
                        clean_data["y"],
                        c=colors_tr_cl,
                        cmap=cmap,
                        s=s_s,
                        marker=".",
                        alpha=1,
                    )
                    axin.scatter(
                        layer_df["x"],
                        layer_df["y"],
                        c=colors_tr,
                        cmap=cmap,
                        s=s_s,
                        marker=".",
                        alpha=1,
                    )
                    axin.tick_params(axis="both", which="both", direction="in")

                ax.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    labelsize=ticksize,
                    size=0.3 * ticksize,
                )
                axin.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    labelsize=0.95 * ticksize,
                    size=0.3 * ticksize,
                )

                ax.text(
                    0.5,
                    1.02,
                    "Predictions",
                    size=fontsize,
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                )

                if not (num_model == 1 and num_layer == 4 and num_disorder in [1, 2]):
                    axin.text(
                        0.5,
                        1.02,
                        "Labels",
                        size=fontsize,
                        ha="center",
                        va="bottom",
                        transform=axin.transAxes,
                    )

                # This is to make space for the centered titles
                ax.set_title(".", color="white", size=1.5 * fontsize)
                vpos_d = {0: 0.96, 1: 0.770, 2: 0.58, 3: 0.39, 4: 0.1975}
                hpos = 0.52

                # if num_disorder == 1:
                subfig.text(
                    hpos,
                    vpos_d[num_layer],
                    f"{new_layer_name}",
                    size=1.2 * fontsize,
                    ha="center",
                    va="center",
                    transform=subfig.transFigure,
                )
                if num_layer == 0:
                    # ax.set_title(f"W={disorder}", size=fontsize)
                    if num_disorder < len(disorders_to_plot) - 2:
                        ax.text(
                            0.5,
                            1.35,
                            r"$W \,/\,\textit{{{}}} = {}$".format("w", disorder),
                            size=1.4 * fontsize,
                            ha="center",
                            va="top",
                            transform=ax.transAxes,
                        )
                    else:
                        ax.text(
                            0.5,
                            1.35,
                            r"$W \,/\,\textit{{{}}} \geq {}$".format("w", disorder),
                            size=1.4 * fontsize,
                            ha="center",
                            va="top",
                            transform=ax.transAxes,
                        )

        # subfig.text(0.265 if num_model == 0 else 0.7825, 1.05, subfig_labels[num_model], size=1.2*fontsize, ha="center", va='center', transform=subfig.transFigure)

        if no_ticks:
            for ax in axs.flatten():
                #     ax.label_outer()
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    for subfig in subfigs:
        if do_insets:
            subfig.legend(
                handles=handles_p_cl + handles_tr,
                labels=categories_p_cl + categories_tr,
                loc="center",
                ncol=4,
                fontsize=fontsize,
                bbox_to_anchor=(0.5, 1.03),
            )
        else:
            subfig.legend(
                handles=handles_p_cl + handles_p + handles_mis,
                labels=categories_p_cl + categories_p + ["Misclassified"],
                loc="center",
                ncol=5,
                fontsize=fontsize,
                bbox_to_anchor=(0.5, 1.25),
            )

    for f_num, fig in enumerate(subfigs):
        fig.savefig(
            save_path.joinpath(f"Fig_{'12' if f_num == 0 else '13'}.pdf"),
            bbox_inches="tight",
        )
        save_path.joinpath("PNG").mkdir(exist_ok=True, parents=True)
        fig.savefig(
            save_path.joinpath("PNG").joinpath(
                f"Fig_{'12' if f_num == 0 else '13'}.png"
            ),
            bbox_inches="tight",
        )
