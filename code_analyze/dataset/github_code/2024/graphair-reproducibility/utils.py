import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from models.fairgraph.utils.utils import (
    scipysp_to_pytorchsp,
    spearman_correlation,
    weighted_homophily,
)

def parse_arguments():
    # standard arguments
    parser = argparse.ArgumentParser(description="Grid Search Script")
    parser.add_argument(
        "--epochs", nargs="+", type=int, default=[500], help="Number of epochs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pokec_z",
        choices=["pokec_z", "pokec_n", "nba", "citeseer", "cora", "pubmed"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model_lr", nargs="+", type=float, default=[1e-4], help="model learning rate"
    )
    parser.add_argument(
        "--use_best_params",
        action="store_true",
        default=False,
        help="Ignore input args, use best hparams",
    )

    # model exclusive arguments
    subparsers = parser.add_subparsers(dest="model")

    # graphair
    graphair_parser = subparsers.add_parser("graphair")
    lp_graphair_parser = subparsers.add_parser("lpgraphair")
    for gparser in [graphair_parser, lp_graphair_parser]:
        gparser.add_argument(
            "--alpha", nargs="+", type=float, default=[1.0], help="List of alpha values"
        )
        gparser.add_argument(
            "--beta", nargs="+", type=float, default=[1.0], help="List of alpha values"
        )
        gparser.add_argument(
            "--gamma", nargs="+", type=float, default=[1.0], help="List of gamma values"
        )
        gparser.add_argument(
            "--lam", nargs="+", type=float, default=[1.0], help="List of lambda values"
        )
        gparser.add_argument(
            "--classifier_lr",
            nargs="+",
            type=float,
            default=[1e-3],
            help="classifier learning rate",
        )
        gparser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
        gparser.add_argument(
            "--classifier_hidden",
            nargs="+",
            type=int,
            default=[128],
            help="Classifier hidden size",
        )
        gparser.add_argument(
            "--model_hidden",
            nargs="+",
            type=int,
            default=[64],
            help="Model hidden size",
        )
        gparser.add_argument(
            "--test_epochs", type=int, default=500, help="Number of test epochs"
        )
        gparser.add_argument(
            "--disable_ep",
            nargs='+',  
            type=lambda x: x.lower() in ('true', '1', 't', 'y', 'yes'), 
            default=[False],
            help="Specify whether to disable the ep component in graphair. Accepts multiple boolean values.",
        )
        gparser.add_argument(
            "--disable_fm",
            nargs='+',  
            type=lambda x: x.lower() in ('true', '1', 't', 'y', 'yes'), 
            default=[False],
            help="Specify whether to disable the fm component in graphair. Accepts multiple boolean values.",
        )
        gparser.add_argument(
            "--homophily",
            action="store_true",
            default=False,
            help="Specify wether claim 3 (plotting homophiy and spearman) should be enabled or not.",
        )

    # fairdrop
    fairdrop_parser = subparsers.add_parser("fairdrop")
    fairdrop_parser.add_argument(
        "--hidden_gcn", nargs="+", type=int, default=[64], help="Hidden size for GCN"
    )
    fairdrop_parser.add_argument(
        "--hidden_gat", nargs="+", type=int, default=[64], help="Hidden size for GCN"
    )
    fairdrop_parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=[1],
        help="Number of heads, used for GAT",
    )
    fairdrop_parser.add_argument(
        "--model_type", nargs="+", type=str, default=['GCN'], help="Model type for Fairdrop, either GCN or GAT"
    )
    fairdrop_parser.add_argument(
        "--delta", nargs="+", type=float, default=[0.16], help="Delta parameter for FairDrop"
    )

    # fairadj
    fairadj_parser = subparsers.add_parser("fairadj")
    fairadj_parser.add_argument(
        "--hidden1",
        nargs="+",
        type=int,
        default=[32],
        help="Hidden size for layer 1 of GCNModelVAE",
    )
    fairadj_parser.add_argument(
        "--hidden2",
        nargs="+",
        type=int,
        default=[16],
        help="Hidden size for layer 2 of GCNModelVAE",
    )
    fairadj_parser.add_argument(
        "--dropout",
        nargs="+",
        type=float,
        default=[0.0],
        help="Dropout rate, calculated as 1 - keep_prob",
    )
    fairadj_parser.add_argument(
        "--outer_epochs", nargs="+", type=int, default=[4], help="Number of epochs to train."
    )
    fairadj_parser.add_argument(
        "--eta", nargs="+", type=float, default=[0.2], help="Learning rate for adjacency matrix."
    )
    fairadj_parser.add_argument("--T1", nargs="+", type=int, default=[50])
    fairadj_parser.add_argument("--T2", nargs="+", type=int, default=[20])

    args = parser.parse_args()
    
    # set default parser to Graphair
    if args.model is None:
        args = parser.parse_args(['graphair'])
        
    return args


def set_seed(seed=20):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_best_params(dataset):
    with open("best_hparams.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    if dataset in data:
        return data[dataset]
    else:
        return None


def plot_homophily(dataset, result):
    plot_params = {
        "nba": {"yrange": [0.325, 0.525], "smooth": [13, 13]},
        "pokec_n": {"yrange": [0.1, 0.8], "smooth": [15, 37]},
        "pokec_z": {"yrange": [0.1, 0.8], "smooth": [20, 35]},
    }

    plot_params = plot_params[dataset.name.lower()]

    adj = scipysp_to_pytorchsp(dataset.adj).to_dense()
    homophily_values_fair = result["homophily"]
    homophily_values_original = weighted_homophily(adj, dataset.sens)

    sns.kdeplot(
        homophily_values_fair,
        bw_adjust=plot_params["smooth"][0],
        label="Fair view",
        color="orange",
    )
    sns.kdeplot(
        homophily_values_original,
        bw_adjust=plot_params["smooth"][1],
        label="Original",
        color="blue",
    )
    plt.xlim(0, 1)
    plt.ylim(plot_params["yrange"][0], plot_params["yrange"][1])
    plt.axvline(x=np.mean(homophily_values_fair), color="orange", linestyle="--")
    plt.axvline(x=np.mean(homophily_values_original), color="blue", linestyle="--")
    plt.text(
        np.mean(homophily_values_fair),
        plt.ylim()[0],
        f"{np.mean(homophily_values_fair):.2f}",
        color="orange",
        ha="left",
        va="bottom",
    )
    plt.text(
        np.mean(homophily_values_original),
        plt.ylim()[0],
        f"{np.mean(homophily_values_original):.2f}",
        color="blue",
        ha="left",
        va="bottom",
    )
    plt.title(f"{dataset.name.upper()}")
    plt.xlabel("Homophily Value", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.legend()
    plt.savefig(f"plots/homophily/homophily_{dataset.name}.png")
    plt.clf()


def plot_spearman(dataset, result):
    spearman_correlations_fair = result["spearman"][0]
    spearman_correlations_original = spearman_correlation(
        dataset.features, dataset.sens
    )

    indexed_list = list(enumerate(spearman_correlations_original))
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    top_10_indices = [index for index, _ in sorted_list[:10]]
    spearman_correlations_fair = [spearman_correlations_fair[i] for i in top_10_indices]
    spearman_correlations_original = [
        spearman_correlations_original[i] for i in top_10_indices
    ]

    indices = range(10)
    plt.bar(indices, spearman_correlations_original, width=0.4, label="Original")
    plt.bar(
        [i + 0.4 for i in indices],
        spearman_correlations_fair,
        width=0.4,
        label="Fair view",
    )
    plt.xlabel("Feature index", fontsize=15)
    plt.ylabel("Spearman correlation", fontsize=15)
    plt.title(dataset.name.upper())
    plt.legend()
    plt.xticks([i + 0.2 for i in indices], indices)
    plt.savefig(f"plots/spearman/spearman_{dataset.name}.png")
    plt.clf()
