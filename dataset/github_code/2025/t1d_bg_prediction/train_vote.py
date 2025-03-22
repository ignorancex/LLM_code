import argparse
import pandas as pd
from utils_data import *


class CFG:
    subjects = [29, 30, 31, 36, 37, 38, 39, 42, 45, 46, 47, 49, 50, 51, 52, 53, 54]
    train_split = [30, 31, 36, 38, 39, 46, 47, 49, 50, 53, 54]
    test_split = [29, 37, 42, 45, 51, 52]
    horizons = list(range(0, -60, -5))
    output_csv_header = ["date", "subject_id", "bgClass", "target", "y_pred"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path", type=str, default="outputs", help="Path to csv outputs"
)
parser.add_argument(
    "--model_names", nargs="+", required=True, help="List of models for voting"
)

if __name__ == "__main__":
    args = parser.parse_args()

    exp_name = "vote"
    df = pd.DataFrame()
    for model_name in args.model_names:
        file = pd.read_csv(f"{args.output_path}/{model_name}_output.csv")
        file = file.rename(columns={"y_pred": model_name})
        df = file if df.empty else pd.concat([df, file[model_name]], axis=1)
        exp_name = exp_name + "_" + model_name.split("_")[0]
    df["y_pred"] = df[args.model_names].mean(axis=1)
    df = df[CFG.output_csv_header]

    print_results(df)

    df.to_csv(f"{args.output_path}/{exp_name}_output.csv", index=False)
