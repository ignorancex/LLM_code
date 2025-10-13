import json
import os
import sys

import numpy as np  # type: ignore
import torch  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore

from distance import get_indices  # type: ignore
from path import GNN_DIR, RESULT_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore
from utils import calc_rmse_wo_outliers, load_dataset  # type: ignore


def weight_l1_plot(
    dataset_name: str, model: str, metric: str, pooling: str, norm: str, kernel: str
) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    # weight distribution
    path = os.path.join(
        RESULT_DIR,
        dataset_name,
        model,
        f"l=3_p={pooling}_d=64_s=0",
        metric,
        f"d=4_{norm}_l1=0.0",
    )
    data = load_dataset(dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, 4, False, False)
    tree.load_parameter(os.path.join(path, "model_final.pt"))
    axes.hist(tree.weight.detach().numpy(), bins=100)
    axes.set_yscale("log")
    axes.set_title("Weight distribution", fontsize=25)
    axes.set_xlabel("Weight value", fontsize=20)
    axes.set_ylabel("Frequency", fontsize=20)
    axes.set_xticklabels(axes.get_xticklabels(), fontsize=15)
    axes.set_yticklabels(axes.get_yticklabels(), fontsize=15)
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "../paper",
            f"weight_{dataset_name}_{model}_{metric}_{pooling}_{norm}.pdf",
        ),
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close()
    # weight and rmse
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    lambdas = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    non_zeros = np.zeros((len(lambdas), 5))
    rmses = np.zeros((len(lambdas), 5))
    rmses_struc = np.zeros(5)
    dis_mx_struc = torch.load(
        os.path.join(RESULT_DIR, dataset_name, f"{kernel}_d=4.pt")
    )
    for i, seed in enumerate(range(5)):
        dis_mx_mpnn = torch.load(
            os.path.join(
                GNN_DIR,
                dataset_name,
                model,
                f"l=3_p={pooling}_d=64_s={seed}",
                f"dist_{metric}_last.pt",
            )
        )
        indices = get_indices(data)
        for j, lamb in enumerate(lambdas):
            path = os.path.join(
                RESULT_DIR,
                dataset_name,
                model,
                f"l=3_p={pooling}_d=64_s={seed}",
                metric,
                f"d=4_{norm}_l1={lamb}",
            )
            tree = WeisfeilerLemanLabelingTree(data, 4, False, False)
            tree.load_parameter(os.path.join(path, "model_final.pt"))
            non_zeros[j, i] = torch.sum(tree.weight > 0).item() / len(tree.weight)
            dis_mx = torch.load(
                os.path.join(
                    path,
                    "WILT.pt",
                )
            )
            rmses[j, i] = calc_rmse_wo_outliers(
                dis_mx.flatten().numpy(), dis_mx_mpnn.flatten()[indices].numpy()
            )[3]
        rmses_struc[i] = calc_rmse_wo_outliers(
            dis_mx_struc.flatten().numpy(), dis_mx_mpnn.flatten()[indices].numpy()
        )[3]

    axes.errorbar(
        x=[0, 1, 2, 3, 4, 5],
        y=np.mean(rmses, axis=1),
        yerr=np.std(rmses, axis=1),
        marker="o",
        label="RMSE of WILT",
        color="blue",
    )
    axes.axhline(
        np.mean(rmses_struc),
        color="blue",
        linestyle="--",
        label="RMSE of WLOA" if kernel == "WLOA" else "RMSE of WWL",
    )
    axes.axhspan(
        np.mean(rmses_struc) - np.std(rmses_struc),
        np.mean(rmses_struc) + np.std(rmses_struc),
        color="blue",
        alpha=0.2,
    )
    axes1 = axes.twinx()
    axes1.errorbar(
        x=[0, 1, 2, 3, 4, 5],
        y=np.mean(non_zeros, axis=1),
        yerr=np.std(non_zeros, axis=1),
        marker="x",
        label="Ratio of non-zero weights",
        color="red",
    )
    h1, l1 = axes.get_legend_handles_labels()
    h2, l2 = axes1.get_legend_handles_labels()
    axes.legend(h1 + h2, l1 + l2, loc="center", bbox_to_anchor=(0.35, 0.3))
    axes.set_title(r"Results with different $\lambda$", fontsize=25)
    axes.set_xlabel("L1 coefficient", fontsize=20)
    axes.set_ylabel("RMSE", fontsize=20)
    axes.set_xticks([0, 1, 2, 3, 4, 5])
    axes.set_xticklabels(lambdas, fontsize=15)
    axes.set_yticklabels(axes.get_yticklabels(), fontsize=15)
    axes1.set_ylabel("Ratio of non-zero weights", fontsize=20)
    print(np.mean(rmses, axis=1))
    print(np.mean(non_zeros, axis=1))
    # save
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "../paper",
            f"l1_{dataset_name}_{model}_{metric}_{pooling}_{norm}.pdf",
        ),
        bbox_inches="tight",
        pad_inches=0.06,
    )


def rmse_wwl_wloa_wilt(model: str):
    for dataset_name in ["Mutagenicity", "ENZYMES", "Lipo", "IMDB-BINARY", "COLLAB"]:
        data = load_dataset(dataset_name)
        indices = get_indices(data)
        for pooling in ["mean", "sum"]:
            rmses = {
                "WWL": [],
                "WLOA": [],
                ("WILT", "size"): [],
                ("WILT", "dummy"): [],
            }
            for seed in range(5):
                dis_mx_mpnn = torch.load(
                    os.path.join(
                        GNN_DIR,
                        dataset_name,
                        model,
                        f"l=3_p={pooling}_d=64_s={seed}",
                        f"dist_l2_last.pt",
                    )
                )
                for metric in [
                    "WWL",
                    "WLOA",
                    ("WILT", "size"),
                    ("WILT", "dummy"),
                ]:
                    if type(metric) == str:
                        dis_mx = torch.load(
                            os.path.join(
                                RESULT_DIR,
                                dataset_name,
                                f"{metric}_d=4.pt",
                            )
                        )
                    else:
                        dis_mx = torch.load(
                            os.path.join(
                                RESULT_DIR,
                                dataset_name,
                                model,
                                f"l=3_p={pooling}_d=64_s={seed}",
                                # metric[0],
                                "l2",
                                f"d=4_{metric[1]}_l1=0.0",
                                f"WILT.pt",
                            )
                        )
                    rmses[metric].append(
                        calc_rmse_wo_outliers(
                            dis_mx.flatten().numpy(),
                            dis_mx_mpnn.flatten()[indices].numpy(),
                        )[3]
                    )
            for metric in [
                "WWL",
                "WLOA",
                ("WILT", "size"),
                ("WILT", "dummy"),
            ]:
                print(
                    dataset_name,
                    pooling,
                    metric,
                    np.mean(np.array(rmses[metric])) * 100,
                    np.std(np.array(rmses[metric])) * 100,
                )


def weights():
    fig, axes = plt.subplots(4, 5, figsize=(25, 20), sharey=True)
    for i, dataset_name in enumerate(
        ["Mutagenicity", "ENZYMES", "Lipo", "IMDB-BINARY", "COLLAB"]
    ):
        data = load_dataset(dataset_name)
        for j, model in enumerate(["gcn", "gin"]):
            for k, pooling in enumerate(["sum", "mean"]):
                norm = "dummy" if pooling == "sum" else "size"
                path = os.path.join(
                    RESULT_DIR,
                    dataset_name,
                    model,
                    f"l=3_p={pooling}_d=64_s=0",
                    "l2",
                    f"d=4_{norm}_l1=0.0",
                )
                tree = WeisfeilerLemanLabelingTree(data, 4, False, False)
                tree.load_parameter(os.path.join(path, "model_final.pt"))
                axes[j * 2 + k, i].hist(tree.weight.detach().numpy(), bins=100)
                axes[j * 2 + k, i].set_yscale("log")
                if j == 0 and k == 0:
                    axes[0, i].set_title(
                        dataset_name if dataset_name != "Lipo" else "Lipophilicity",
                        fontsize=25,
                    )
                if i == 0:
                    axes[j * 2 + k, 0].set_ylabel(
                        f"{model.upper()}/{pooling}", fontsize=20
                    )
                    axes[j * 2 + k, i].set_yticklabels(
                        axes[j * 2 + k, i].get_yticklabels(), fontsize=15
                    )
                axes[j * 2 + k, i].xaxis.set_major_locator(
                    ticker.MaxNLocator(integer=True)
                )
                axes[j * 2 + k, i].set_xticklabels(
                    axes[j * 2 + k, i].get_xticklabels(), fontsize=15
                )
    fig.supxlabel("Weight value", fontsize=40)
    fig.supylabel("Frequency", fontsize=40)
    fig.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "../paper", "weights.pdf"),
        bbox_inches="tight",
        pad_inches=0.06,
    )


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.subplot.bottom"] = 0.15
    weight_l1_plot("Mutagenicity", "gcn", "l2", "sum", "dummy", "WLOA")
    rmse_wwl_wloa_wilt("gcn")
    weights()
