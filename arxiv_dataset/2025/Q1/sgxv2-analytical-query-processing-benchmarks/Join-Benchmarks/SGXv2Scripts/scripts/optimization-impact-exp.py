#!/usr/bin/python3
import sys
from itertools import chain, combinations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.commons import powerset
from helpers.runner import compile_and_run

phase_order = ["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2", "join_total", "build", "probe"]


def plot_throughput(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput")]
    data = data[(data["threads"] == 16)]
    data["optimizations"] = data["lock"] + "_" + data["unroll"] + '_' + data["sort"]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    data = data.drop(["threads", "lock", "unroll", "sort", "size_r", "size_s"], axis=1)

    sns.catplot(data, y="value", hue="mode", x="alg", col="optimizations", row="size", kind="bar",
                sharey="row")
    plot_filename = f"../img/{experiment_name}-throughput.png"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)


def plot_phases(filename_detail, experiment_name, alg, threads):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == threads)]
    data = data[(data["alg"] == alg)]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)
    data["optimizations"] = data["lock"] + "_" + data["unroll"] + '_' + data["sort"]

    data = data.drop(columns=["threads", "alg", "lock", "unroll", "sort", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", col="optimizations", row="size", kind="bar",
                sharey=False, order=phase_order)

    plot_filename = f"../img/{experiment_name}-{alg}-{threads}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def plot_improvements_alg(filename_detail: str, experiment_name: str, alg: str):
    data = pd.read_csv(filename_detail, header=0)
    # data = data[(data["threads"] == threads)]
    data = data[(data["alg"] == alg)]
    # data = data[(data["mode"] == "sgx")]
    data = data[(data["measurement"] == "throughput")]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)
    data["optimizations"] = data["lock"] + "_" + data["unroll"]
    if alg == "RSM":
        data["optimizations"] = data["optimizations"] + "_" + data["sort"]

    # data = data.drop(columns=["mode", "alg", "lock", "unroll", "sort", "size_r", "size_s"])

    sns.catplot(data, y="value", hue="mode", x="optimizations", col="size", row="threads", kind="bar",
                sharey="row")

    plot_filename = f"../img/{experiment_name}-{alg}-improvements-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def main():
    experiment_name = "optimizations-impact"

    if len(sys.argv) < 2:
        print("Please specify run/plot/both")
        exit(-1)

    execution_command = sys.argv[1]

    if len(sys.argv) < 3 and execution_command != "plot":
        print(f"Experiment name defaults to {experiment_name}. Are you sure?")
        user_input = input("y/N: ")
        if user_input.lower().strip() != "y":
            exit()
    elif execution_command != "plot":
        experiment_name = sys.argv[2]

    filename_detail = f"../data/{experiment_name}.csv"

    config = dict()
    config["experiment"] = True
    config["threads"] = [1, 16]
    config["algorithms"] = ["RHO", "RHT", "RSM"]
    config["reps"] = 5
    config["modes"] = ["native", "sgx"]
    config["flags"] = powerset(["SPIN_LOCK", "UNROLL", "SIMD_SORT"])

    MB = 2 ** 20
    TUPLE_SIZE = 8
    TUPLE_PER_MB = MB/TUPLE_SIZE

    config["sizes"] = [(10 * TUPLE_PER_MB, 40 * TUPLE_PER_MB), (100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        compile_and_run(config, filename_detail)
    if execution_command in ["plot", "both"]:
        # plot_throughput(filename_detail, experiment_name)
        #plot_phases(filename_detail, experiment_name, "RHO", 1)
        # plot_phases(filename_detail, experiment_name, "RHO", 16)
        plot_improvements_alg(filename_detail, experiment_name, "RHO")
        plot_improvements_alg(filename_detail, experiment_name, "RHT")
        plot_improvements_alg(filename_detail, experiment_name, "RSM")


if __name__ == '__main__':
    main()
