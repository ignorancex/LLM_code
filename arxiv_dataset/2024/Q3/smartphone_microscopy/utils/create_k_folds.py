"""
This script is used to create k-folds from coco annotations.
The folds are saved in different json files in the specified folder.
"""
import argparse
import json
import os.path
import pandas as pd

from sklearn.model_selection import KFold


# Path: utils/create_k_folds.py
def create_k_folds(
    src_annotations_path: str,
    folds: int = 5,
    seed: int = 42,
    output_dir: str = None,
):
    """
    This function is used to create k-folds from coco annotations.
    The folds are saved in different json files in the specified folder.

    Arguments:
        src_annotations_path {str} -- Path to the coco annotations file
        folds {int} -- Number of folds
        seed {int} -- Seed for the random state
        output_dir {str} -- Path to the output directory
    """
    with open(src_annotations_path, "r") as f:
        coco = json.load(f)
    images = coco["images"]
    info = coco["info"]
    categories = coco["categories"]
    annotations = coco["annotations"]

    # create a dataframe from the annotations
    annotations_df = pd.DataFrame(annotations)
    # create a new column with the image id
    annotations_df["image_id"] = annotations_df["image_id"]
    # create a new column with the category id
    annotations_df["category_id"] = annotations_df["category_id"]

    # create images dataframe
    images_df = pd.DataFrame(images)
    # create a new column with the image id
    images_df["id"] = images_df["id"]

    print("Total number of images: ", len(images_df))
    # kfold
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (train_index, val_index) in enumerate(kf.split(images_df)):
        print(
            f"Fold {fold}, number of images: Train: {len(train_index)}, Val: {len(val_index)}"
        )
        train_images_df = images_df.iloc[train_index]
        val_images_df = images_df.iloc[val_index]

        # filter the annotations dataframe for test and val images
        train_annotations_df = annotations_df[
            annotations_df["image_id"].isin(train_images_df["id"])
        ]
        val_annotations_df = annotations_df[
            annotations_df["image_id"].isin(val_images_df["id"])
        ]

        # create train and val annotations
        train_annotations = {
            "info": info,
            "categories": categories,
            "images": train_images_df.to_dict("records"),
            "annotations": train_annotations_df.to_dict("records"),
        }
        val_annotations = {
            "info": info,
            "categories": categories,
            "images": val_images_df.to_dict("records"),
            "annotations": val_annotations_df.to_dict("records"),
        }

        # Save the annotations in a JSON file
        if output_dir is None:
            output_dir = os.path.dirname(src_annotations_path)
        train_annotations_path = os.path.join(
            output_dir,
            f"fold_{fold + 1}",
            os.path.basename(src_annotations_path).replace(".json", "_train.json"),
        )
        val_annotations_path = os.path.join(
            output_dir,
            f"fold_{fold + 1}",
            os.path.basename(src_annotations_path).replace(".json", "_val.json"),
        )
        os.makedirs(os.path.dirname(train_annotations_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_annotations_path), exist_ok=True)
        with open(train_annotations_path, "w") as f:
            json.dump(train_annotations, f)
        with open(val_annotations_path, "w") as f:
            json.dump(val_annotations, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the train and validation files from coco annotations"
    )
    parser.add_argument(
        "--src_annotations_path",
        type=str,
        help="Path to the coco annotations file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random state",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds",
    )
    args = parser.parse_args()
    create_k_folds(
        args.src_annotations_path,
        args.folds,
        args.seed,
        args.output_dir,
    )
