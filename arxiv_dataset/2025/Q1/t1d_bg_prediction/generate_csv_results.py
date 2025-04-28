import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

HYPO = 70.0
HYPER = 180.0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    type=str,
    default="outputs",
    help="Path to save model and results",
)

if __name__ == "__main__":
    args = parser.parse_args()

    file_prefixes = [file for file in os.listdir(args.output_path) if "output" in file]

    cumulative_results = []
    condition_results = []

    for file in os.listdir(args.output_path):
        if "output" not in file:
            continue

        test_set = pd.read_csv(f"{args.output_path}/{file}")
        test_set["bgClass"] = test_set["target"].apply(
            lambda x: "Hypo" if x < HYPO else ("Hyper" if x > HYPER else "Normal")
        )

        maes, mapes, rmses = [], [], []
        for subject in test_set["subject_id"].unique():
            x = test_set[test_set["subject_id"] == subject]
            maes.append(mean_absolute_error(x["target"], x["y_pred"]))
            mapes.append(mean_absolute_percentage_error(x["target"], x["y_pred"]) * 100)
            rmses.append(root_mean_squared_error(x["target"], x["y_pred"]))

        cumulative_results.append(
            [
                file[:-11].upper(),
                f"{np.mean(maes):.2f}({np.std(maes):.2f})",
                f"{np.mean(mapes):.2f}({np.std(mapes):.2f})",
                f"{np.mean(rmses):.2f}({np.std(rmses):.2f})",
            ]
        )

        for condition in ["Normal", "Hyper", "Hypo"]:
            dummy = test_set[test_set["bgClass"] == condition]
            maes, mapes, rmses = [], [], []
            for subject in test_set["subject_id"].unique():
                x = dummy[dummy["subject_id"] == subject]
                if x.empty:
                    continue
                maes.append(mean_absolute_error(x["target"], x["y_pred"]))
                mapes.append(
                    mean_absolute_percentage_error(x["target"], x["y_pred"]) * 100
                )
                rmses.append(root_mean_squared_error(x["target"], x["y_pred"]))

            condition_results.append(
                [
                    file[:-11].upper(),
                    condition,
                    f"{np.mean(maes):.2f}({np.std(maes):.2f})",
                    f"{np.mean(mapes):.2f}({np.std(mapes):.2f})",
                    f"{np.mean(rmses):.2f}({np.std(rmses):.2f})",
                ]
            )

    cumulative_df = pd.DataFrame(
        cumulative_results, columns=["Model", "MAE", "MAPE", "RMSE"]
    )
    cumulative_df.to_csv("cumulative_results.csv", index=False)

    condition_df = pd.DataFrame(
        condition_results, columns=["Model", "Condition", "MAE", "MAPE", "RMSE"]
    )
    condition_pivot = condition_df.pivot(
        index="Model", columns="Condition", values=["MAE", "MAPE", "RMSE"]
    )
    condition_pivot.columns = [
        f"{metric} ({condition})" for metric, condition in condition_pivot.columns
    ]
    condition_pivot.reset_index(inplace=True)
    condition_pivot.to_csv("condition_results.csv", index=False)
