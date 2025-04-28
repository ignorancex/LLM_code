import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
import sympy as sp
import torch
import torch.nn.functional as F
from IPython.display import Image, display
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import integrate
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import classification_report, confusion_matrix
from termcolor import cprint

from src.thermometer_encoding import Thermometer


def test_model(
    model,
    testDataLoader,
    class_names,
    device,
    verbose=True,
    therm=False,
    therm_levels=100,
):
    # we can now evaluate the network on the test set
    if verbose:
        cprint("[INFO]", "magenta", end=" ")
        print("Evaluating network...")
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        # loop over the test set
        for (x, y) in testDataLoader:
            y = F.one_hot(y, num_classes=len(class_names))
            y = y.float()

            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
    # generate a classification report
    if verbose:
        print(
            classification_report(
                testDataLoader.dataset.Y.cpu().numpy(),
                np.array(preds),
                target_names=class_names,
            )
        )
    return classification_report(
        testDataLoader.dataset.Y.cpu().numpy(),
        np.array(preds),
        target_names=class_names,
        output_dict=True,
    )


def get_preds_from_model(
    model,
    testDataLoader,
    class_names,
    device,
    therm=False,
    therm_levels=100,
):
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        # loop over the test set
        for (x, y) in testDataLoader:
            # These three lines are essential to apply for our data!!!!!!
            y = F.one_hot(y, num_classes=len(class_names))
            y = y.float()

            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).detach().cpu().numpy())

    return np.array(preds)


def prepare_dfs(hparam_sets, real_no, n_slices, main_root):
    for params in hparam_sets:
        batch_size = params["batch_size"]
        arch = params["arch"]
        if 'perturb' in params.keys():
            perturb = params["perturb"]
            coral = None
        elif 'coral' in params.keys():
            perturb = None
            coral = params['coral']
        else:
            perturb = None
            coral = None
        lr = params["lr"]

        if arch != "CNN":
            load_root = main_root.joinpath(f"bs={batch_size}_arch={arch}_lr={lr:.0e}")
        elif perturb is not None:
            load_root = main_root.joinpath(f"bs={batch_size}_perturb={perturb}_lr={lr:.0e}")
        else:
            load_root = main_root.joinpath(f"bs={batch_size}_coral={coral}_lr={lr:.0e}")

        all_stats_csv_path = load_root.joinpath(
            f"all_stats_{real_no}_real_{n_slices}_slices.csv"
        )

        df = pd.read_csv(all_stats_csv_path)
        col_dict = dict(
            zip(
                df.columns,
                [
                    "Model name",
                    "Folder",
                    "Topo_Precision",
                    "Topo_Recall",
                    "Topo_F1",
                    "Trivial_Precision",
                    "Trivial_Recall",
                    "Trivial_F1",
                    "Macro avg Precision",
                    "Macro Avg Recall",
                    "Macro Avg F1",
                    "Accuracy",
                    "SSIM",
                    "RMSE",
                    "LPIPS",
                    "ToDrop",
                ],
            )
        )
        df.rename(columns=col_dict, inplace=True)
        if perturb is not None:
            df.drop(columns=["ToDrop"], inplace=True)

        # From each model in the load_root folder load the history file and read the training accuracy and validation accuracy, then add this to df created from csv file.
        trn_acc_tab = []
        val_acc_tab = []
        model_nos_tab = np.arange(100)
        model_nos_tab = sorted(model_nos_tab.astype(str))
        for i in model_nos_tab:
            p = load_root.joinpath(f"final_run_{i}").joinpath("trained_model_history.pickle")
            if p.exists():
                H = pickle.load(open(p, "rb"))
                trn_acc = H["train_acc"][-1]
                val_acc = H["val_acc"][-1]
                trn_acc_tab.append(trn_acc)
                val_acc_tab.append(val_acc)
            else:
                trn_acc_tab.append(np.nan)
                val_acc_tab.append(np.nan)
                
        df["Train Accuracy"] = trn_acc_tab
        df["Validation Accuracy"] = val_acc_tab


        p_to_test = pathlib.Path(
            df["Model name"].iloc[0].replace("\\", "/")
        ).parent.joinpath("generalization_stats.csv")
        df_test = pd.read_csv(p_to_test)
        df_mean = df_test.groupby("W").mean()
        df_mean = df_mean.loc[df_mean.index == 0.0]
        for i in range(1, df.shape[0]):
            p_to_test = pathlib.Path(
                df["Model name"].iloc[i].replace("\\", "/")
            ).parent.joinpath("generalization_stats.csv")
            df_test = pd.read_csv(p_to_test)
            df_mean = df_mean.append(
                df_test.groupby("W")
                .mean()
                .loc[df_test.groupby("W").mean().index == 0.0]
            )
        df_mean.reset_index(inplace=True)
        df_mean.drop(
            columns=[
                "W",
                "Real",
                "Topo_Precision",
                "Topo_F1",
                "Macro avg Precision",
                "Trivial_Precision",
                "Trivial_Recall",
                "Trivial_F1",
                "Macro Avg Recall",
            ],
            inplace=True,
        )
        df_mean.rename(
            columns={
                "Topo_Recall": "Clean Topo Recall",
                "Macro Avg F1": "Clean Avg F1",
                "Accuracy": "Clean Accuracy",
            },
            inplace=True,
        )
        df = pd.concat([df, df_mean], axis=1)
        del df_mean, df_test
        params["df"] = df
    return hparam_sets


def disp_general_around_thresh(
    hparam_sets,
    above,
    head_number,
    column_name="LPIPS",
    loc_in_head=0,
    thresh_proc=0.5,
    comp_axis_1="RMSE",
    comp_axis_2="Macro avg Precision",
    real_slice_no="10_100",
    sig=1.5,
    shown_stat="generalization",
    cam_ind=0,
):
    fontsize = 25
    ticksize = 18
    real_no, n_slices = real_slice_no.split("_")
    n_slices = int(n_slices)
    real_no = int(real_no)
    w_values = np.linspace(0.00, 5.00, n_slices, endpoint=True)
    w_labels = np.linspace(0.00, 5.00, 11, endpoint=True)
    v_ticks_num = 11
    v_tab = np.linspace(0, 2, v_ticks_num, endpoint=True)

    batch_size = hparam_sets["batch_size"]
    if 'perturb' in hparam_sets.keys():
        perturb = hparam_sets["perturb"]
        coral = None
    elif 'coral' in hparam_sets.keys():
        perturb = None
        coral = hparam_sets['coral']
    else:
        perturb = None
        coral = None
    lr = hparam_sets["lr"]
    df = hparam_sets["df"]

    thresh_max = df[column_name].max()
    thresh_min = df[column_name].min()
    thresh = thresh_min + thresh_proc * (thresh_max - thresh_min)

    df = df.copy()
    df_pl2 = df.copy()
    if above:
        df = df.loc[df[column_name] > thresh]
        ascending = True
    else:
        df = df.loc[df[column_name] < thresh]
        ascending = False
    df_sorted = df.sort_values(by=column_name, ascending=ascending).head(head_number)

    folder = df_sorted["Folder"].iloc[loc_in_head]
    model_name = df_sorted["Model name"].iloc[loc_in_head].replace("\\", "/")

    fig = plt.figure(figsize=(20, 10))
    plt.style.use("ggplot")

    if shown_stat == "generalization":
        axs = [fig.add_subplot(121, frame_on=False), fig.add_subplot(122)]
        ax = axs[0]
        ax_met = axs[1]
    elif shown_stat == "history":
        axs = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
        ax_loss = axs[0]
        ax_acc = axs[1]
        ax_met = axs[2]
    elif shown_stat in ["UMAP", "PCA", "CAMs"]:
        axs = [fig.add_subplot(111)]
        ax_met = axs[0]
    else:
        raise ValueError("Unknown shown_stat")

    # Add a second plots consisting of three subplots. Each representing the histogram of three metrics: RMSE, SSIM, LPIPS, and showing where in the distribution is the displayed model
    ax_met.set_xlabel(comp_axis_1, size=fontsize, labelpad=30)
    ax_met.set_ylabel(comp_axis_2, size=fontsize, labelpad=30)
    ax_met.xaxis.set_label_position("top")
    ax_met.yaxis.set_label_position("right")
    ax_met.xaxis.tick_top()
    ax_met.yaxis.tick_right()
    ax_met.xaxis.set_tick_params(labelsize=fontsize)
    ax_met.yaxis.set_tick_params(labelsize=fontsize)
    if df_pl2[comp_axis_1].min() > 0.9 and "Clean" in comp_axis_1:
        ax_met.set_xlim([0.90, 1])
    else:
        ax_met.set_xlim([0, 1])
    if df_pl2[comp_axis_2].min() > 0.9 and "Clean" in comp_axis_2:
        ax_met.set_ylim([0.90, 1])
    else:
        ax_met.set_ylim([0, 1])
    ax_met.plot(
        df_pl2[comp_axis_1],
        df_pl2[comp_axis_2],
        lw=0,
        marker="o",
        markersize=10,
        color="black",
        alpha=0.25,
    )
    ax_met.plot(
        df_sorted[comp_axis_1].iloc[loc_in_head],
        df_sorted[comp_axis_2].iloc[loc_in_head],
        lw=0,
        marker="o",
        markersize=20,
        color="red",
        alpha=0.75,
    )
    ax_met.axvline(x=df_sorted[comp_axis_1].iloc[loc_in_head], ls="dotted", color="red")
    ax_met.axhline(y=df_sorted[comp_axis_2].iloc[loc_in_head], ls="dotted", color="red")
    if comp_axis_1 == column_name:
        ax_met.axvline(x=thresh, ls="dotted", color="green")
    elif comp_axis_2 == column_name:
        ax_met.axhline(y=thresh, ls="dotted", color="green")

    if shown_stat == "generalization":
        im = pickle.load(
            open(
                f"./{pathlib.Path(model_name).parent}/generalization_data_raw.pkl", "rb"
            )
        )
        image = gaussian_filter(im[:, ::-1].T, sigma=sig)
        hmap = sns.heatmap(
            image, ax=ax, cmap="coolwarm", rasterized=True, vmin=0, vmax=1
        )
        ax.set_title(
            f"Model name : {model_name}\n\nFolder : {folder}\n{column_name} : {df_sorted[column_name].iloc[loc_in_head]}\nThreshold : {thresh:.3f}",
            fontsize=20,
        )
        plt.tick_params(size=fontsize)
        ymin, ymax = ax.get_ylim()
        ytick_pos = np.linspace(ymin, ymax, v_ticks_num)
        xmin, xmax = ax.get_xlim()
        xtick_pos = np.linspace(xmin, xmax, len(w_labels))
        # ax.set_title("Predictions", size=fontsize)
        ax.set_xticks(xtick_pos, size=fontsize)
        ax.set_yticks(ytick_pos, size=fontsize)
        ax.set_yticklabels([f"{t:.1f}" for t in v_tab], size=ticksize)
        ax.set_xticklabels([f"{t:.1f}" for t in w_labels], size=ticksize, rotation=45)
        ax.set_xlabel("W", size=fontsize)
        ax.set_ylabel("v", size=fontsize)

        fig.tight_layout()
        plt.show()
    elif shown_stat == "history":
        H = pickle.load(
            open(
                f"./{pathlib.Path(model_name).parent}/trained_model_history.pickle",
                "rb",
            )
        )
        ax_loss.plot(H["train_loss"], label="train_loss")
        ax_loss.plot(H["val_loss"], label="val_loss")

        ax_acc.plot(H["train_acc"], label="train_acc")
        ax_acc.plot(H["val_acc"], label="val_acc")

        ax_loss.set_xlabel("Epoch #", size=fontsize)
        ax_loss.set_ylabel("Loss/Accuracy", size=fontsize)
        ax_acc.set_xlabel("Epoch #", size=fontsize)
        ax_acc.set_ylabel("Loss/Accuracy", size=fontsize)

        ax_loss.set_title(f"Training Loss", size=fontsize)
        ax_acc.set_title(f"Training Accuracy", size=fontsize)

        ax_loss.legend(loc="upper left", fontsize=fontsize)
        ax_acc.legend(loc="lower left", fontsize=fontsize)

        ax_acc.xaxis.set_tick_params(labelsize=ticksize)
        ax_acc.yaxis.set_tick_params(labelsize=ticksize)
        ax_loss.xaxis.set_tick_params(labelsize=ticksize)
        ax_loss.yaxis.set_tick_params(labelsize=ticksize)

        fig.tight_layout()
        plt.show()
    else:
        fig.tight_layout()
        plt.show()
        if shown_stat == "CAMs":
            # Find in model_name.parent a file with 'cam_ind' ind in name
            parent_dir = pathlib.Path(model_name).parent.joinpath("CAM")
            cam_files = list(parent_dir.glob(f"ind={cam_ind}_*.png"))
            if len(cam_files) == 0:
                display("File is not generated yet")
            else:
                p_file = cam_files[0]
                display(Image(p_file))
        else:
            if shown_stat == "UMAP":
                p_file = pathlib.Path(f"./{pathlib.Path(model_name).parent}/UMAP.png")
            elif shown_stat == "PCA":
                p_file = pathlib.Path(f"./{pathlib.Path(model_name).parent}/PCA.png")

            if p_file.exists():
                display(Image(p_file))
            else:
                display("File is not generated yet")


def recreate_fig_6(save_path, data_path):

    data = pd.read_pickle(data_path.joinpath("targets_25_reals.pkl"))

    v, w, k = sp.symbols("v w k", real=True)

    # v = 1.5

    # w = 1.0

    h = sp.Matrix([[v, w], [w * sp.exp(sp.I * k), v]])

    dkh = sp.simplify(sp.Derivative(h, k).doit())

    hinv = sp.Inverse(h).doit()

    h1_fin = hinv @ dkh

    expr1 = sp.Trace(h1_fin).doit() / (2 * sp.pi * sp.I)

    expr2 = sp.simplify(sp.Derivative(sp.log(v + w * sp.exp(sp.I * k)), k).doit()) / (
        2 * sp.pi * sp.I
    )

    # expr1.subs([(v, 0), (w, 1)])

    int1 = sp.integrate(expr1, (k, -sp.pi, sp.pi - np.finfo(float).eps))

    int2 = sp.integrate(expr2, (k, -sp.pi, sp.pi - np.finfo(float).eps))

    print(int1)
    print(int2)

    #%%

    ww = 1

    vv_tab = np.linspace(0, 2, 200)

    wnum1_tab = []

    wnum2_tab = []

    for vv in vv_tab:
        # vv= 0.999
        integrand1 = sp.lambdify([k, v, w], expr1.doit(), "scipy")

        integrand2 = sp.lambdify([k, v, w], expr2)

        r1, r1err = integrate.quad(
            integrand1, -np.pi, np.pi - np.finfo(float).eps, args=(vv, ww)
        )

        r2, r2err = integrate.quad(
            integrand2, -np.pi, np.pi - np.finfo(float).eps, args=(vv, ww)
        )

        wnum1_tab.append(r1)
        wnum2_tab.append(r2)

        # print(f"{r1:.2f} , {r2:.2f}")

    ss = 100
    fs = 25
    tl = 20
    ts = 8

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    # fig.suptitle("Numerical integration", fontsize=fs)
    plt.tight_layout()
    ax.scatter(
        vv_tab,
        wnum1_tab,
        s=ss,
        lw=0,
        marker=7,
        alpha=0.75,
        color="C0",
        label=r"$W \,/\,\textit{w} = 0$",
    )
    # ax.scatter(vv_tab, wnum1_tab, s=ss, lw=0, marker=6, alpha=0.75, color="C0", label=r"$\frac{1}{2 \pi i} \int_{-\pi}^{\pi} Tr\left[h^{-1} \partial_k h\right]$ dk")
    # ax.scatter(vv_tab, wnum2_tab, s=ss, lw=0, marker=7, alpha=0.75, color="C3", label=r"$\frac{1}{2 \pi i} \sum_{j=1}^{\mathcal{D}/2} \int_{-\pi}^{\pi}  \partial_{k} \log{h_j}$ dk")
    # ax.scatter(vv_tab, data[50, :], s=ss, lw=0, marker=6, alpha=0.75, color="C1", label=r"$W = 0.5$")
    ax.scatter(
        vv_tab,
        data[100, :],
        s=ss,
        lw=0,
        marker=6,
        alpha=0.75,
        color="C1",
        label=r"$W \,/\,\textit{w}= 1$",
    )
    ax.scatter(
        vv_tab,
        data[200, :],
        s=ss,
        lw=0,
        marker=5,
        alpha=0.75,
        color="C2",
        label=r"$W \,/\,\textit{w}= 2$",
    )
    ax.scatter(
        vv_tab,
        data[300, :],
        s=ss,
        lw=0,
        marker=4,
        alpha=0.75,
        color="C3",
        label=r"$W \,/\,\textit{w}= 3$",
    )
    ax.set_ylabel(r"winding number $\mathcal{\vartheta}$", fontsize=fs)
    ax.set_xlabel(r"$\textit{v}\,/\,\textit{w}$", fontsize=fs)
    ax.set_yticks([0, 1])
    ax.set_xlim(-0.02, 2.02)
    ax.set_xticks(np.linspace(0, 2, 11))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.axvline(
        x=1.0,
        ls="dotted",
        label="Infinite system \nphase transition point",
        color="black",
    )
    at = AnchoredText(r"$w=1$", prop=dict(size=20), frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    ax.tick_params(axis="both", which="major", direction="in", labelsize=tl, size=ts)
    ax.tick_params(axis="both", which="minor", direction="in", size=0.5 * ts)
    ax.legend(loc="lower left", fontsize=tl)

    fig.savefig(save_path.joinpath("Fig_6.pdf"), bbox_inches="tight")
    save_path.joinpath("PNG").mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path.joinpath("PNG").joinpath("Fig_6.png"), bbox_inches="tight")


def real_vs_therm_comparison(therm, data_sample, levels=15, eigenstate_no=25, fs=15):
    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    ax = axs[0]
    ax.stem(
        data_sample.squeeze().numpy()[:, eigenstate_no],
        linefmt="C5:",
        markerfmt="C3.",
        basefmt="C2--",
    )

    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.set_ylim(0, 1.1)

    ax.set_ylabel("Amplitude [arb.u.]", size=fs)

    ax.set_title("Real valued wavefunction", size=fs)

    ax = axs[1]

    therm = ax.imshow(
        therm.squeeze().numpy()[:, eigenstate_no][:, ::-1].T,
        cmap="cividis",
        aspect="auto",
    )
    ax.set_yticks(np.arange(0, levels, 1))
    ax.set_yticklabels(np.arange(0, levels, 1)[::-1])

    ax.set_xticks(np.arange(-0.5, 50, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, levels, 1), minor=True)

    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_ylabel("Thermometer encoding channel $c$", size=fs)

    axins = inset_axes(
        ax,
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(therm, cax=axins, orientation="vertical")
    cbar.set_label("Channel population", rotation=270, labelpad=30, size=fs)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1], size=fs)

    ax.set_title("Thermometer encoded wavefunction", size=fs)

    for ax in axs:
        ax.set_xticks([0, 9, 19, 29, 39, 49], size=fs)
        ax.set_xticklabels([0, 10, 20, 30, 40, 50], size=fs, rotation=45, ha="right")
        ax.set_xlabel("Physical site index $j$", size=fs)

    plt.show()


def plot_thermometer_demonstration(
    ds, img_ind=20, levels=15, chosen_level=3, lin_fig_size=15, fs=20
):
    # Preprocess the data
    test_loader = ds.get_test_loader()

    data_sample = ds.test_data[20]

    data_sample = torch.from_numpy(data_sample).float().reshape([1, 1, 50, 50])

    therm = Thermometer(data_sample, levels)

    therm = therm.reshape([50, 50, levels])

    therm = therm.numpy()

    # Plot the data
    XX, YY = np.meshgrid(np.linspace(0, 50, 50), np.linspace(0, 50, 50))
    # Plot this as a bar plot, and add coloring according to the height of the bar
    colors = plt.cm.coolwarm(data_sample.flatten() / float(data_sample.max()))

    fig = plt.figure(figsize=(lin_fig_size, lin_fig_size))
    col = "y"
    alp = 0.5

    rect = [0.15, 0.25, 0.2, 0.2]

    axs = [
        fig.add_subplot(211, projection="3d"),
        fig.add_subplot(212, projection="3d"),
        fig.add_axes(rect, anchor="NW"),
    ]

    ax = axs[0]

    colors = plt.cm.coolwarm(data_sample.flatten() / float(data_sample.max()))
    ax.bar3d(
        XX.ravel(),
        YY.ravel(),
        np.zeros(XX.shape).ravel(),
        1,
        1,
        data_sample.ravel(),
        color=colors,
        alpha=0.5,
        cmap="viridis",
    )
    ax.set_zticks(
        np.linspace(data_sample.min(), data_sample.max(), levels).round(2),
        size=0.5 * fs,
    )
    ax.set_zlabel("Probability amplitude", fontsize=fs, labelpad=fs)
    ax.set_title("a)", fontsize=fs)

    ax = axs[1]

    for i in range(levels):
        ax.bar3d(
            XX.ravel(),
            YY.ravel(),
            np.zeros(XX.shape).ravel(),
            1,
            1,
            therm[:, :, : i + 1].sum(axis=2).ravel(),
            color=colors,
            alpha=0.5,
            cmap="viridis",
        )
    ax.set_zticks(np.linspace(0, levels, levels + 1).astype(int), size=0.5 * fs)
    ax.set_zlabel("Thermometer encoding\nchannel", fontsize=fs, labelpad=fs)
    ax.set_title("b)", fontsize=fs)

    xs, zs = np.meshgrid(
        np.linspace(0, 50, 50), np.linspace(chosen_level, chosen_level + 1, 50)
    )
    ys = xs * 0
    ax.plot_surface(xs, ys, zs, alpha=alp, color=col)
    ax.plot_surface(xs, ys + 50, zs, alpha=alp, color=col)

    ys, zs = np.meshgrid(
        np.linspace(0, 50, 50), np.linspace(chosen_level, chosen_level + 1, 50)
    )
    xs = ys * 0
    ax.plot_surface(xs, ys, zs, alpha=alp, color=col)
    ax.plot_surface(xs + 50, ys, zs, alpha=alp, color=col)

    ax = axs[2]

    ax.set_title(
        "c)\nView of thermometer\nencoded channel"
        + r"$c = {}$".format(chosen_level + 1),
        fontsize=0.8 * fs,
        pad=fs,
    )
    ax.imshow(therm[:, :, chosen_level], cmap="binary", vmin=0, vmax=1)
    # Make the axes yellow to match the plane
    for sp in ax.spines.values():
        sp.set_color(col)
        sp.set_linewidth(2)

    ax = axs[1]

    for num, ax in enumerate(axs):
        if num < 2:
            ax.view_init(25, -60)
        ax.tick_params(axis="both", which="major", labelsize=0.6 * fs, direction="in")
        ax.set_xticks([0, 9, 19, 29, 39, 49], size=fs)
        ax.set_xticklabels([0, 10, 20, 30, 40, 50], size=fs, rotation=45, ha="right")

        ax.set_yticks([0, 9, 19, 29, 39, 49][::-1], size=fs)
        ax.set_yticklabels(
            [0, 10, 20, 30, 40, 50], size=fs, rotation=0, ha="right", va="center"
        )

        ax.set_xlabel("Eigenstate index $i$", size=fs, labelpad=0.9 * fs)
        ax.set_ylabel("Physical site index $j$", size=fs, labelpad=0.1 * fs)

    ax = axs[2]

    ax.set_xticks([0, 9, 19, 29, 39, 49], size=fs)
    ax.set_xticklabels([0, 10, 20, 30, 40, 50], size=fs, rotation=45, ha="right")

    ax.set_yticks([0, 9, 19, 29, 39, 49][::-1], size=fs)
    ax.set_yticklabels(
        [0, 10, 20, 30, 40, 50], size=fs, rotation=0, ha="right", va="center"
    )

    ax.set_xlabel("Eigenstate index $i$", size=fs, labelpad=0.5 * fs)
    ax.set_ylabel("Physical site index $j$", size=fs, labelpad=0.5 * fs)
