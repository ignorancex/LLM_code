from os import PathLike
import pandas as pd
import os.path
import json
from argparse import ArgumentParser

BASE_DIR = "/mnt/Enterprise/safal/AI_assisted_microscopy_system"

annotation_file = (
    "/mnt/Enterprise/safal/AI_assisted_microscopy_system/"
    "cysts_dataset_all/smartphone_sample_test/"
    "smartphone_sample_test_coco_annos.json"
)


model_name = "faster_rcnn"

prediction_file = os.path.join(
    BASE_DIR,
    "outputs/smartphone_sample",
    model_name,
    "fold_1",
    "results_test.bbox.json",
)


def prediction_count(
    prediction_file: PathLike,
    annotation_file: PathLike,
    model_name: str,
    conf_threshold: float = 0.5,
):
    with open(annotation_file) as f:
        test_annos_json = json.load(f)

    # map image id to image name
    image_id_to_name = {
        image["id"]: image["file_name"].split("/")[-1].split(".")[0]
        for image in test_annos_json["images"]
    }

    prediction_df = pd.read_json(prediction_file)

    # filter out predictions with confidence less than threshold
    prediction_df = prediction_df[prediction_df["score"] > conf_threshold]

    if model_name == "yolov8":
        prediction_df["image_name"] = prediction_df["image_id"]
    else:
        prediction_df["image_name"] = prediction_df["image_id"].map(image_id_to_name)

    prediction_df = (
        prediction_df.groupby(["image_name", "category_id"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for _, image_name in image_id_to_name.items():
        if image_name not in prediction_df["image_name"].values:
            prediction_df = pd.concat(
                [
                    prediction_df,
                    pd.DataFrame(
                        {
                            "image_name": [image_name],
                            0: [0],
                            1: [0],
                        }
                    ),
                ]
            )

    prediction_df.sort_values(by="image_name", inplace=True)

    return prediction_df


def prediction_count_five_fold(
    base_dir: PathLike,
    model_name: str,
    annotation_file: PathLike,
    conf_threshold: float = 0.5,
):
    prediction_df_five_fold = None
    for fold in range(1, 6):
        prediction_file = os.path.join(
            base_dir,
            "outputs/smartphone_sample",
            model_name,
            f"fold_{fold}",
            "results_test.bbox.json",
        )

        prediction_df = prediction_count(
            prediction_file=prediction_file,
            annotation_file=annotation_file,
            model_name=model_name,
            conf_threshold=conf_threshold,
        )

        prediction_df = prediction_df.reset_index(drop=True)

        print(prediction_df.head())
        if fold == 1:
            prediction_df_five_fold = prediction_df
        else:
            # add the predictions of categories 0 and 1
            prediction_df_five_fold[0] = prediction_df_five_fold[0] + prediction_df[0]

            prediction_df_five_fold[1] = prediction_df_five_fold[1] + prediction_df[1]

    # average the predictions
    prediction_df_five_fold[0] = prediction_df_five_fold[0] / 5
    prediction_df_five_fold[1] = prediction_df_five_fold[1] / 5
    prediction_df_five_fold = prediction_df_five_fold.rename(
        columns={0: "Crypto", 1: "Giardia"}
    )
    return prediction_df_five_fold


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/Enterprise/safal/AI_assisted_microscopy_system",
    )
    parser.add_argument("--model_name", type=str, default="faster_rcnn")
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="/mnt/Enterprise/safal/AI_assisted_microscopy_system/"
        "cysts_dataset_all/smartphone_sample_test/"
        "smartphone_sample_test_coco_annos.json",
    )
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/Enterprise/safal/AI_assisted_microscopy_system/outputs",
    )
    args = parser.parse_args()

    prediction_df_five_fold = prediction_count_five_fold(
        base_dir=args.base_dir,
        model_name=args.model_name,
        annotation_file=args.annotation_file,
        conf_threshold=args.conf_threshold,
    )

    save_path = os.path.join(args.output_dir, f"{args.model_name}_prediction_count.csv")
    prediction_df_five_fold.to_csv(
        save_path,
        index=False,
    )

    print(f"Saved prediction count to {save_path}")
