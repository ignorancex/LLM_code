import pickle
import argparse
from sklearn.linear_model import LinearRegression
from utils_data import *


class CFG:
    l_bound = 20.0
    u_bound = 420.0
    subjects = [29, 30, 31, 36, 37, 38, 39, 42, 45, 46, 47, 49, 50, 51, 52, 53, 54]
    train_split = [30, 31, 36, 38, 39, 46, 47, 49, 50, 53, 54]
    test_split = [29, 37, 42, 45, 51, 52]
    horizons = list(range(0, -60, -5))
    output_csv_header = ["date", "subject_id", "bgClass", "target", "y_pred"]


parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="outputs")
parser.add_argument("--stack_features_path", type=str, default="stack_features")
parser.add_argument(
    "--model_names", nargs="+", required=True, help="List of models for voting"
)
parser.add_argument("--seed", type=int, default=42)


if __name__ == "__main__":
    args = parser.parse_args()

    exp_name = "stack"
    train_set = pd.DataFrame()
    for model_name in args.model_names:
        sub_df = pd.read_csv(f"{args.stack_features_path}/{model_name}_output.csv")
        sub_df = sub_df.rename(columns={"y_pred": f"{model_name}"})
        if train_set.empty:
            train_set = sub_df
        else:
            train_set[f"{model_name}"] = sub_df[f"{model_name}"]
        exp_name = exp_name + "_" + model_name.split("_")[0]
    train_set = scale_data(train_set, args.model_names + ["target"])

    # Train model
    model = LinearRegression()
    model.fit(train_set[args.model_names], train_set["target"])
    with open(f"{args.output_path}/{exp_name}.pickle", "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n")

    # Evaluate
    test_set = pd.DataFrame()
    for model_name in args.model_names:
        sub_df = pd.read_csv(f"{args.output_path}/{model_name}_output.csv")
        sub_df = sub_df.rename(columns={"y_pred": f"{model_name}"})
        if test_set.empty:
            test_set = sub_df
        else:
            test_set[f"{model_name}"] = sub_df[f"{model_name}"]

    test_set = scale_data(test_set, args.model_names + ["target"])
    test_set["y_pred"] = model.predict(test_set[args.model_names])
    test_set = rescale_data(test_set, ["target", "y_pred"])
    test_set = test_set[CFG.output_csv_header]

    print_results(test_set)

    test_set.to_csv(f"{args.output_path}/{exp_name}_output.csv", index=False)
