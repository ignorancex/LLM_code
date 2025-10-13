from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve
import pandas as pd
import os
from sklearn.model_selection import KFold
import numpy as np
import random
from datasets import get_dataset_registry
from fair_icl_methods import (
    correlation_remover,
    uncertainty_based_sample_selection,
    group_balanced_sampling,
    class_group_balanced_sampling,
)

from metrics import fair_metrics_map
from models import get_model
from args import add_input_args
import torch


# seed
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def fit(base_model, data, data2, args):
    clf = get_model(base_model)
    kf = KFold(n_splits=5, shuffle=True)
    to_ret = {"acc": [], "f1": [], "auc": [], "size_of_training_omitted": []}
    for f in fair_metrics_map.keys():
        to_ret[f] = []

    for i, (train_index, test_index) in enumerate(kf.split(data[0])):
        print(f"---------------------------->>> Running fold = {i+1}/5")
        # Extract training data for current fold
        X_train, y_train, s_train = (
            data[0][train_index],
            data[1][train_index],
            data[2][train_index],
        )
        # Extract test data for current fold
        X_test, y_test, s_test = (
            data[0][test_index],
            data[1][test_index],
            data[2][test_index],
        )

        train_idx = np.random.choice(len(X_train), size=len(X_train), replace=False)

        if args.selection_method.endswith("certain"):
            print("=============== COMPUTING UNCERTAINTY SETS ==================== ")
            base_classifier = get_model(args.base_model_uncertainty)
            # Get demonstration based on Uncertainty of sensitive attribute.
            _, train_idx = uncertainty_based_sample_selection(
                base_classifier,
                X_train,
                (data2[0], data2[2]),
                cp_alpha=args.cp_alpha,
            )
        elif args.selection_method == "class_group_balanced":
            print(
                f"=============== SAMPLING CLASS BALANCED SET OF SIZE {args.train_size}  ==================== "
            )
            train_idx = class_group_balanced_sampling(
                X_train, y_train, s_train, train_size=args.train_size
            )
        elif args.selection_method == "group_balanced":
            print(
                f"=============== SAMPLING GROUP BALANCED SET OF SIZE {args.train_size}  ==================== "
            )

            train_idx = group_balanced_sampling(X_train, s_train)
        elif args.selection_method == "correlation_remover":
            print("=============== APPLYING CORRELATION REMOVER ==================== ")
            if args.cr_sanitization_strategy == 2:
                # We apply correlation remover on the training set only
                X_train, cr = correlation_remover(X_train, s_train, alpha=args.cp_alpha)
                # X_augmented = np.hstack((np.expand_dims(s_test, axis=1), X_test))
                # X_test = cr.transform(X_augmented)
            elif args.cr_sanitization_strategy == 1:
                # Default: we apply correlation remover on the testing and the testing sets
                X_train, cr = correlation_remover(X_train, s_train, alpha=args.cp_alpha)
                # X_test, _ = correlation_remover(X_test, s_test)
                X_augmented = np.hstack((np.expand_dims(s_test, axis=1), X_test))
                X_test = cr.transform(X_augmented)
            else:
                raise Exception("Correlation remover sanitization strategy not found")
        elif args.selection_method == "vanilla":
            train_idx = np.random.choice(train_idx, size=len(train_idx), replace=False)
        else:
            raise Exception("ICL method not found")

        to_ret["size_of_training_omitted"].append(
            ((len(X_train) - len(train_idx)) / len(X_train)) * 100
        )
        print(f"Size of training omitted: {to_ret['size_of_training_omitted'][-1]}%")

        if args.train_size > 0 and args.train_size < len(train_idx):
            # subsample the demonstration (training) set size
            print(
                f"=============== SAMPLING THE TRAINING SET OF SIZE {args.train_size} ==================== "
            )
            train_idx = np.random.choice(train_idx, size=args.train_size, replace=False)

        if base_model == "tabpfn" and len(train_idx) >= 10000:
            # sub-sample the maximum of 10000 samples that TabPFN can handle
            print(
                "=============== SUBSAMPLING TO 10000 SAMPLES FOR TabPFN==================== "
            )
            train_idx = np.random.choice(train_idx, size=10000, replace=False)

        X_train = X_train[train_idx, :]
        y_train = y_train[train_idx]

        print("================== FITTING THE MODEL =================")
        print("Lenght Train ======>", X_train.shape)
        clf.fit(X_train, y_train)

        print("================== PREDICTING ON TEST SET =================")
        y_pred = clf.predict(X_test)

        print(
            "================== COMPUTING METRICS: ACC, DP, EODS, EOP ================="
        )

        for fairness in fair_metrics_map.keys():
            unfairness_measure = fair_metrics_map[fairness]

            unfair = unfairness_measure(
                y_true=y_test, y_pred=y_pred, sensitive_features=s_test
            )
            to_ret[fairness].append(unfair)

        to_ret["acc"].append(accuracy_score(y_true=y_test, y_pred=y_pred))
        to_ret["f1"].append(f1_score(y_true=y_test, y_pred=y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        to_ret["auc"].append(auc(fpr, tpr))

        # to_ret.update({"acc": [acc]})
    return to_ret


def main(args):
    exp_folder_name = "analysis/results"
    if args.exp_name:
        exp_folder_name = f"analysis/{args.exp_name}/"

    out_path = "{}/{}/t_size_{}/{}".format(
        exp_folder_name, args.dataset, args.train_size, args.base_model
    )

    if args.selection_method == "uncertain":
        out_path += "/{}_{}/epsilon_{}".format(
            args.selection_method,
            args.base_model_uncertainty,
            args.cp_alpha,
        )
    elif args.selection_method == "correlation_remover":
        out_path += f"/{args.selection_method}_{args.cr_sanitization_strategy}/alpha_{args.cp_alpha}"
    else:
        out_path += f"/{args.selection_method}"

    datasets = get_dataset_registry()
    train_data, data2 = datasets[args.dataset]()

    out_file = "{}/seed_{}.csv".format(out_path, args.seed)
    res = fit(args.base_model, train_data, data2, args)
    print(res)
    df = pd.DataFrame(res)
    os.makedirs(out_path, exist_ok=True)
    df.to_csv(out_file, encoding="utf-8", index=False)
    print("================== COMPLETED =================")
    print(out_file)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run fairness experiments with different models and selection methods"
    )
    parser = add_input_args(parser)

    arg = parser.parse_args()

    seed_everything(arg.seed)

    main(arg)
