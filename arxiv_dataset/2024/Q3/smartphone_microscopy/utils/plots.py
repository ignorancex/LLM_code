"""
This script is used to plot the precision-recall curve for a given model and sample type.

Usage:
    python utils/plot_pr_curve.py \
        --base_dir /path/to/base/dir \
        --model_type model_type \
        --sample_type sample_type \
        --fold fold \
        --save True \
        --result_json /path/to/result.json \
        --plot_type pr_curve

Arguments:
    base_dir: path to the base directory where the outputs are stored
    model_type: the model type for which the pr curve is to be plotted
    sample_type: the sample type for which the pr curve is to be plotted
    fold: the fold for which the pr curve is to be plotted
    save: whether to save the plot or not
    result_json: path to the result json file
    plot_type: the type of plot to be plotted. Pr curve or iou conf heatmap
"""

import os.path
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from .metrics import calculate_precision_recall_f1


def plot_pr_curve(args: Namespace):
    base_dir = args.base_dir
    model_type = args.model_type
    sample_type = args.sample_type
    fold = args.fold

    gt_annotation_file = os.path.join(
        base_dir,
        f"cysts_dataset_all/{sample_type}/fold_{fold}/{sample_type}_coco_annos_val.json",
    )

    pred_annotation_file = os.path.join(
        base_dir, f"outputs/{sample_type}/{model_type}/fold_{fold}/results.bbox.json"
    )

    if not os.path.exists(pred_annotation_file):
        print(f"Predictions for {model_type} not found")
        return

    confidence_thresholds = np.arange(0.7, 0.0, -0.05)
    precisions = {
        0: np.zeros(len(confidence_thresholds)),
        1: np.zeros(len(confidence_thresholds)),
    }

    recalls = {
        0: np.zeros(len(confidence_thresholds)),
        1: np.zeros(len(confidence_thresholds)),
    }

    for i, conf_threshold in enumerate(confidence_thresholds):
        metrics_df = calculate_precision_recall_f1(
            pred_annotation_file, gt_annotation_file, conf_threshold
        )
        precisions[0][i] = metrics_df[metrics_df["category"] == "crypto"][0][
            "precision"
        ]
        precisions[1][i] = metrics_df[metrics_df["category"] == "giardia"][0][
            "precision"
        ]

        recalls[0][i] = metrics_df[metrics_df["category"] == "crypto"][0]["recall"]
        recalls[1][i] = metrics_df[metrics_df["category"] == "giardia"][0]["recall"]

    crypto_metrics_df = pd.DataFrame(
        {
            "precision": precisions[0],
            "recall": recalls[0],
        }
    )
    giardia_metrics_df = pd.DataFrame(
        {
            "precision": precisions[1],
            "recall": recalls[1],
        }
    )
    # plot giardia and crypto pr curves for the given sample type
    sns.lineplot(
        data=crypto_metrics_df, x="recall", y="precision", label="Cryptosporidium"
    )
    sns.lineplot(data=giardia_metrics_df, x="recall", y="precision", label="Giardia")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.title(
        f"Precision-Recall Curve for {sample_type.replace('_', ' ')} using {model_type.replace('_', ' ').capitalize()}"
    )
    if args.save:
        save_path = os.path.join(
            base_dir,
            f"outputs/{sample_type}/{model_type}/pr_curve.png",
        )
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved pr curve at {save_path}")


def plot_iou_conf_heatmaps(args: Namespace):
    base_dir, model_type, sample_type, metrics_df = metrics_at_multiple_iou_conf(args)
    # A subplot grid with 2 rows, 2 columns, first row for gcrypto, second for giardia, each column for precision and recall
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Precision-Recall Heatmap for {sample_type.replace('_', ' ')} using {model_type.replace('_', ' ').capitalize()}"
    )

    for i in range(2):
        for j in range(3):
            if i == 0:
                category = "Crypto"
            else:
                category = "Giardia"
            if j == 0:
                metric = "precision"
            elif j == 1:
                metric = "recall"
            else:
                metric = "f1_score"

            ax = sns.heatmap(
                metrics_df[metrics_df["category"] == category].pivot(
                    "conf_threshold", "iou_threshold", metric
                ),
                fmt=".2f",
                ax=axs[i, j],
            )
            ax.invert_yaxis()
            axs[i, j].set_title(f"{category.capitalize()} {metric.capitalize()}")
            axs[i, j].set_xlabel("IoU Threshold")
            axs[i, j].set_ylabel("Confidence Threshold")

    if args.save:
        save_path = os.path.join(
            base_dir,
            f"outputs/{sample_type}/{model_type}/pr_heatmap.png",
        )
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved pr heatmap at {save_path}")

def metrics_at_multiple_iou_conf(args):
    base_dir = args.base_dir
    model_type = args.model_type
    sample_type = args.sample_type
    fold = args.fold
    results_json = args.result_json

    gt_annotation_file = os.path.join(
        base_dir,
        f"cysts_dataset_all/{sample_type}/fold_{fold}/{sample_type}_coco_annos_val.json",
    )

    pred_annotation_file = results_json if results_json else os.path.join(
        base_dir, f"outputs/{sample_type}/{model_type}/fold_{fold}/results.bbox.json"
    )

    if not os.path.exists(pred_annotation_file):
        print(f"Predictions for {model_type} not found")
        return

    confidence_thresholds = np.arange(0.1, 1.0, 0.1)
    iou_thresholds = np.arange(0.1, 1.0, 0.1)

    metrics_df = pd.DataFrame(
        columns=["conf_threshold", "iou_threshold", "precision", "recall", "f1_score"]
    )

    for conf_threshold in confidence_thresholds:
        for iou_threshold in iou_thresholds:
            print(
                f"Calculating metrics for conf_threshold: {conf_threshold} and iou_threshold: {iou_threshold}"
            )
            metrics = calculate_precision_recall_f1(
                pred_annotation_file,
                gt_annotation_file,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            metrics["conf_threshold"] = conf_threshold.round(2)
            metrics["iou_threshold"] = iou_threshold.round(2)

            metrics_df = pd.concat([metrics_df, metrics], axis=0)
            print("Done")

    metrics_df.reset_index(inplace=True)
    print(metrics_df)
    return base_dir,model_type,sample_type,metrics_df


def plot_pr_scatter(args):
    base_dir, model_type, sample_type, metrics_df = metrics_at_multiple_iou_conf(args)
    # 1x2 subplot with scatter plot for gcrypto and giardia
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"Precision-Recall Scatter Plot for {sample_type.replace('_', ' ')} using {model_type.replace('_', ' ').capitalize()}"
    )

    for i in range(2):
        if i == 0:
            category = "Crypto"
        else:
            category = "Giardia"

        ax = sns.scatterplot(
            data=metrics_df[metrics_df["category"] == category],
            x="precision",
            y="recall",
            hue="iou_threshold",
            palette="tab10",
            style="conf_threshold",
            ax=axs[i],
        )
        axs[i].set_title(f"{category.capitalize()}")
        axs[i].set_xlabel("Precision")
        axs[i].set_ylabel("Recall")
        axs[i].set_xticks(np.arange(0, 1.1, 0.1))
        axs[i].set_yticks(np.arange(0, 1.1, 0.1))
        if i == 0:
            axs[i].get_legend().remove()

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    if args.save:
        save_path = os.path.join(
            base_dir,
            f"outputs/{sample_type}/{model_type}/pr_scatter_{sample_type}_{model_type}.png",
        )
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved pr scatter plot at {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/Enterprise/safal/AI_assisted_microscopy_system/",
    )
    parser.add_argument("--model_type", type=str, default="retinanet")
    parser.add_argument("--sample_type", type=str, default="brightfield_sample")
    parser.add_argument(
        "--fold", type=int, default=1, help="Fold to use for evaluation"
    )
    parser.add_argument("--result_json", type=str, default="")
    parser.add_argument(
        "--plot_type", type=str, default="heatmap", help="heatmap or pr_curve or pr_scatter"
    )
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()
    if args.plot_type == "heatmap":
        plot_iou_conf_heatmaps(args)
    elif args.plot_type == "pr_scatter":
        plot_pr_scatter(args)
    else:
        plot_pr_curve(args)
