import os
import pickle
import argparse
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
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
parser.add_argument("--data_path", default="DiaTrend", type=str)
parser.add_argument("--output_path", type=str, default="outputs")
parser.add_argument("--stack_features_path", type=str, default="stack_features")
parser.add_argument(
    "--exp_name",
    type=str,
    choices=["svr", "rf", "lgb"],
    help="Model name. Choose from [svr, rf, lgb]",
    required=True,
)
parser.add_argument("--seed", type=int, default=42)


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if not os.path.exists(args.stack_features_path):
        os.mkdir(args.stack_features_path)

    train_set, test_set, X_cols, y_cols = get_data(
        args.data_path,
        subjects=CFG.subjects,
        horizons=CFG.horizons,
        train_split=CFG.train_split,
        test_split=CFG.test_split,
        single_output=True,
    )

    # Train model
    if args.exp_name == "svr":
        model = SVR()
    elif args.exp_name == "rf":
        model = RandomForestRegressor(random_state=args.seed)
    elif args.exp_name == "lgb":
        model = lgb.LGBMRegressor(random_state=args.seed)

    model.fit(train_set[X_cols], train_set[y_cols[-1]])
    with open(f"{args.output_path}/{args.exp_name}.pickle", "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n")

    # Evaluate
    test_set["y_pred"] = model.predict(test_set[X_cols])
    test_set = test_set.rename(columns={y_cols[-1]: "target"})
    test_set = rescale_data(test_set, ["target", "y_pred"])
    test_set = test_set[CFG.output_csv_header]

    print_results(test_set)

    test_set.to_csv(f"{args.output_path}/{args.exp_name}_output.csv", index=False)

    # Generate outputs for stacking purposes
    train_set["y_pred"] = model.predict(train_set[X_cols])
    train_set = train_set.rename(columns={y_cols[-1]: "target"})
    train_set = rescale_data(train_set, ["target", "y_pred"])
    train_set = train_set[CFG.output_csv_header]

    train_set.to_csv(
        f"{args.stack_features_path}/{args.exp_name}_output.csv", index=False
    )
