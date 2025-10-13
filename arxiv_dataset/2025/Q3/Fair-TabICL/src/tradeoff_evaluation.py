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
)

from metrics import fair_metrics_map
from models import get_model
import torch
from sklearn.model_selection import train_test_split
from args import add_input_args

import mpi4py.rc
from mpi4py import MPI


# seed
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def split(container, count):
    return [container[_i::count] for _i in range(count)]


def process_result(epsilons, results):
    results_dict = {"dp": [], "eop": [], "eodds": [], "acc": [], "f1": []}

    for res in results:
        for metric in results_dict.keys():
            results_dict[metric] = results_dict[metric] + res[metric]

    results_dict["epsilon"] = epsilons
    return results_dict


def fit(base_model, train_data, test_data, data2, epsilon, args):
    clf = get_model(base_model)
    kf = KFold(n_splits=5, shuffle=True)
    to_ret = {"acc": [], "f1": [], "auc": []}
    for f in fair_metrics_map.keys():
        to_ret[f] = []
    X_train, y_train, s_train = train_data
    X_test, y_test, s_test = test_data
    train_idx = np.random.choice(len(X_train), size=len(X_train), replace=False)

    if args.selection_method.endswith("certain"):
        print("=============== COMPUTING UNCERTAINTY SETS ==================== ")
        base_classifier = get_model(args.base_model_uncertainty)
        # Get demonstration based on Uncertainty of sensitive attribute.
        _, train_idx = uncertainty_based_sample_selection(
            base_classifier,
            X_train,
            (data2[0], data2[2]),
            cp_alpha=epsilon,
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
        elif args.cr_sanitization_strategy == 3:
            # We apply correlation remover on the testing set only
            X_test, _ = correlation_remover(X_test, s_test, alpha=args.cp_alpha)
        else:
            # Default: we apply correlation remover on the testing and the testing sets
            X_train, cr = correlation_remover(X_train, s_train, alpha=args.cp_alpha)
            # X_test, _ = correlation_remover(X_test, s_test)
            X_augmented = np.hstack((np.expand_dims(s_test, axis=1), X_test))
            X_test = cr.transform(X_augmented)
    elif args.selection_method == "vanilla":
        train_idx = np.random.choice(train_idx, size=len(train_idx), replace=False)
    else:
        raise Exception("ICL method not found")

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

    print("================== COMPUTING METRICS: ACC, DP, EODS, EOP =================")

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


def main(args, epsilons):
    mpi4py.rc.threads = False

    selection_method = args.selection_method

    if args.selection_method.endswith("certain"):
        selection_method += f"_{args.base_model_uncertainty}"
    elif args.selection_method == "correlation_remover":
        selection_method += f"_{args.cr_sanitization_strategy}"
    
    out_path = "analysis/tradeoff/{}/t_size_{}/{}/{}".format(
        args.dataset, args.train_size, args.base_model, selection_method
    )
    datasets = get_dataset_registry()
    data1, data2 = datasets[args.dataset]()

    X_train, y_train, s_train = data1

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_train, y_train, s_train, test_size=0.2
    )

    train_data = X_train, y_train, s_train

    test_data = X_test, y_test, s_test

    print("Lenght Test ======>", X_test.shape)
    print("Lenght Train ======>", X_train.shape)

    COMM = MPI.COMM_WORLD

    print("COMM_SIZE = ", COMM.size)

    if COMM.rank == 0:
        jobs = split(epsilons, COMM.size)
    else:
        jobs = None

    jobs = COMM.scatter(jobs, root=0)
    results = []

    for epsilon in jobs:
        print("---------------------------->>> epsilon = {}".format(epsilon))

        res = fit(args.base_model, train_data, test_data, data2, epsilon, args)
        if res is not None:
            results.append(res)
    # Gather results on rank 0.
    results = MPI.COMM_WORLD.gather(results, root=0)
    if COMM.rank == 0:
        results = [_i for temp in results for _i in temp]
        processed = process_result(epsilons, results)

        out_file = "{}/{}_model_{}.csv".format(out_path, args.base_model, args.seed)
        df = pd.DataFrame(processed)
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(out_file, encoding="utf-8", index=False)
        print(out_file)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run fairness experiments with different models and selection methods"
    )
    parser = add_input_args(parser)

    arg = parser.parse_args()

    # epsilons
    """ epsilon_range = np.arange(0.701, 0.991, 0.004)
    base = np.arange(0.00, 0.7, 0.05)  # [0.01, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
    epsilon_range = list(base) + list(epsilon_range) + [0.9999]
    epsilons = [round(x, 3) for x in epsilon_range]  """

    epsilon_range = np.arange(0.701, 0.991, 0.004)
    base = [0.0, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
    epsilon_range = base + list(epsilon_range) + [0.9999]
    epsilons = [round(x, 3) for x in epsilon_range]  # 300 values 

    if arg.selection_method == "uncertain":
        epsilons = 1 - np.array(epsilons)
        epsilons = np.array([round(x, 3) for x in epsilons])
        epsilons = epsilons[epsilons > 0]
        epsilons = epsilons[epsilons < 1]
    if arg.selection_method == "vanilla" or arg.selection_method == "group_balanced":
        epsilons = [1]
    seed_everything(arg.seed)

    main(arg, epsilons)
