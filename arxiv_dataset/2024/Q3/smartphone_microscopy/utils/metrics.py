import json
import os.path
from os import PathLike

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou
from argparse import ArgumentParser, Namespace


def calculate_precision_recall_f1(
    pred_annotations_file: PathLike,
    gt_annotations_file: PathLike,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
):
    """
    This function calculates the precision, recall and f1 score for each class
    in the given ground truth and predicted annotations.

    Arguments:
        pred_annotations_file: path to the predicted annotations file
        gt_annotations_file: path to the ground truth annotations file
        conf_threshold: confidence threshold to filter out predictions

    Returns:
        metrics_df: dataframe containing the precision, recall and f1 score for each class

    """
    # Precision x Recall is obtained individually by each class
    # Loop through each class and calculate the precision and recall

    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    with open(gt_annotations_file) as f:
        gt_annotations = json.load(f)

    with open(pred_annotations_file) as f:
        pred_annotations = json.load(f)

    if len(pred_annotations) == 0:
        print("No predictions found")
        return

    # change gt_annos id value to image names
    for i in range(len(gt_annotations["images"])):
        gt_annotations["images"][i]["image_id"] = (
            gt_annotations["images"][i]["file_name"].rsplit("/")[-1].split(".")[0]
        )

    gt_annotations_df = pd.DataFrame(gt_annotations["annotations"])
    pred_annotations_df = pd.DataFrame(pred_annotations)

    # change bbox width and height to x2, y2
    pred_annotations_df["bbox"] = pred_annotations_df["bbox"].apply(
        lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]]
    )
    gt_annotations_df["bbox"] = gt_annotations_df["bbox"].apply(
        lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]]
    )

    images_df = pd.DataFrame(gt_annotations["images"])

    # replace image_id of gt_annotations_df with image_id of images_df
    gt_annotations_df["image_id"] = gt_annotations_df["image_id"].apply(
        lambda x: images_df[images_df["id"] == x]["image_id"].values[0]
    )

    # replace image_id of pred_annotations_df with image_id of images_df
    # comment this line for yolov8
    # pred_annotations_df["image_id"] = pred_annotations_df["image_id"].apply(
    #     lambda x: images_df[images_df["id"] == x]["image_id"].values[0]
    # )

    categories = sorted(gt_annotations_df.category_id.unique())

    # dataframe to store the precision, recall and f1 score for each class
    metrics_df = pd.DataFrame(
        columns=["category", "precision", "recall", "f1_score", "TP", "FP"]
    )

    for category in categories:
        # get the ground truth annotations for the current class
        gt_annotations_df_class = gt_annotations_df[
            gt_annotations_df.category_id == category
        ]
        # get the predicted annotations for the current class
        pred_annotations_df_class = pred_annotations_df[
            pred_annotations_df.category_id == category
        ]

        # sort the predicted annotations by score
        pred_annotations_df_class = pred_annotations_df_class.sort_values(
            by="score", ascending=False
        )

        # filter predictions with score > conf_threshold
        pred_annotations_df_class = pred_annotations_df_class[
            pred_annotations_df_class.score > conf_threshold
        ]

        true_positives_class = 0
        false_positives_class = 0

        # get image ids for the current class from both ground truth and predicted annotations
        image_ids = pred_annotations_df_class["image_id"].unique()

        for image in image_ids:
            # get the ground truth annotations for the current image
            gt_annotations_df_image = gt_annotations_df_class[
                gt_annotations_df_class.image_id == image
            ]
            # get the predicted annotations for the current image
            pred_annotations_df_image = pred_annotations_df_class[
                pred_annotations_df_class.image_id == image
            ]

            # get the ground truth bounding boxes
            gt_bboxes = torch.tensor(gt_annotations_df_image.bbox.to_list())

            gt_matched = np.zeros(len(gt_bboxes))

            # get the predicted bounding boxes
            pred_bboxes = torch.tensor(pred_annotations_df_image.bbox.to_list())

            if len(gt_bboxes) == 0:
                false_positives_class += len(pred_bboxes)
                continue

            # get the intersection over union for each predicted bounding box
            ious = box_iou(pred_bboxes, gt_bboxes)

            # get the maximum iou for each ground truth bounding box
            max_ious, max_idxs = torch.max(ious, dim=1)

            for iou_pred, gt_idx in zip(max_ious, max_idxs):
                if (iou_pred > iou_threshold).item() and (gt_matched[gt_idx] == 0):
                    true_positives_class += 1
                    gt_matched[gt_idx] = 1
                else:
                    false_positives_class += 1

        # calculate the precision and recall
        eps = torch.finfo(torch.float32).eps
        precision = true_positives_class / max(
            true_positives_class + false_positives_class, eps
        )
        recall = true_positives_class / max(gt_annotations_df_class.shape[0], eps)

        category_name = gt_annotations["categories"][category]["name"]
        f1_score = 2 * (precision * recall) / max(precision + recall, eps)

        category_metrics_df = pd.DataFrame(
            {
                "category": category_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "TP": true_positives_class,
                "FP": false_positives_class,
            },
            index=[0],
        )

        # concatenate the metrics for the current class to the metrics dataframe
        metrics_df = pd.concat([metrics_df, category_metrics_df], axis=0)

    return metrics_df


def calculate_5_fold_precision_recall_f1(
    sample_type,
    base_dir,
    model_name,
    conf_threshold=0.5,
    iou_threshold=0.5,
    save_metrics=False,
    mode="val",
):
    """
    Calculates the precision, recall and f1 score for the 5 folds of the dataset

    Args:
        sample_type (str): The type of sample to calculate the metrics for
        base_dir (str): The base directory of the project
        model_name (str): The name of the model to calculate the metrics for
        conf_threshold (float): The confidence threshold to use for calculating the metrics
        save_metrics (bool): Whether to save the metrics to a csv file

    Returns:
        metrics_df (pd.DataFrame): The metrics for the 5 folds
    """
    metrics_df = None

    for fold in range(1, 6):
        print(f"Calculating metrics for fold {fold}")
        # get the ground truth and predicted annotations for the current fold
        if mode == "val":
            gt_annotation_file = os.path.join(
                base_dir,
                f"cysts_dataset_all/{sample_type}/fold_{fold}/{sample_type}_coco_annos_val.json",
            )

            pred_annotation_file = os.path.join(
                base_dir,
                f"outputs/{sample_type}/{model_name}/fold_{fold}/results.bbox.json",
            )
        else:
            gt_annotation_file = os.path.join(
                base_dir,
                f"cysts_dataset_all/{sample_type}_test/{sample_type}_test_coco_annos.json",
            )

            pred_annotation_file = os.path.join(
                base_dir,
                f"outputs/{sample_type}/{model_name}/fold_{fold}/results_test.bbox.json",
            )
        if not os.path.exists(pred_annotation_file):
            print(f"Skipping fold {fold} as no predictions were made")
            continue

        # calculate the precision, recall and f1 score for the current fold
        fold_metrics_df = calculate_precision_recall_f1(
            pred_annotation_file,
            gt_annotation_file,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        # concatenate the metrics for the current fold to the metrics dataframe
        metrics_df = (
            fold_metrics_df
            if fold == 1
            else pd.concat([metrics_df, fold_metrics_df], axis=0)
        )

    metrics_df = metrics_df.groupby("category").agg(
        {
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1_score": ["mean", "std"],
        }
    )
    metrics_df.columns = ["_".join(x) for x in metrics_df.columns]
    metrics_df = metrics_df.reset_index()

    if save_metrics:
        if mode == "val":
            save_path = os.path.join(
                base_dir, f"outputs/{sample_type}/{model_name}/metrics_pr.csv"
            )
        else:
            save_path = os.path.join(
                base_dir, f"outputs/{sample_type}/{model_name}/metrics_pr_test.csv"
            )
        metrics_df.to_csv(
            save_path,
            index=False,
            float_format=r"%.3f",
        )
        print(
            f"Saved metrics to {save_path}, iou_threshold={iou_threshold}, conf_threshold={conf_threshold}"
        )
    return metrics_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_type", type=str, default="brightfield_reference")
    parser.add_argument("--model_name", type=str, default="faster_rcnn")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--save_metrics", type=bool, default=False)
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/Enterprise/safal/AI_assisted_microscopy_system",
    )
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--fold", type=int, default=1)

    args = parser.parse_args()

    if args.fold == 1:
        metrics = calculate_precision_recall_f1(
            pred_annotations_file=os.path.join(
                args.base_dir,
                f"outputs/{args.sample_type}/{args.model_name}/results.bbox.json",
            ),
            gt_annotations_file=os.path.join(
                args.base_dir,
                f"cysts_dataset_all/{args.sample_type}/{args.sample_type}_coco_annos.json",
            ),
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
        )
        if args.save_metrics:
            if args.mode == "val":
                save_path = os.path.join(
                    args.base_dir,
                    f"outputs/{args.sample_type}/{args.model_name}/metrics_pr.csv",
                )
            else:
                save_path = os.path.join(
                    args.base_dir,
                    f"outputs/{args.sample_type}/{args.model_name}/metrics_pr_test.csv",
                )
            metrics.to_csv(
                save_path,
                index=False,
                float_format=r"%.3f",
            )
            print(
                f"Saved metrics to {save_path}, iou_threshold={args.iou_threshold}, conf_threshold={args.conf_threshold}"
            )
    else:
        metrics = calculate_5_fold_precision_recall_f1(
            base_dir=args.base_dir,
            sample_type=args.sample_type,
            model_name=args.model_name,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            save_metrics=args.save_metrics,
            mode=args.mode,
        )
    print(metrics)
