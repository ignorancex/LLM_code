#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.runner import compile_and_run

phase_order = ["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2", "join_total", "build", "probe"]


def plot_throughput(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data_throughput = data[(data["measurement"] == "throughput")]

    data_throughput = data_throughput.drop(["lock"], axis=1)

    data_throughput["size"] = data_throughput["size_r"].astype(str) + '_' + data_throughput["size_s"].astype(str)

    sns.catplot(data_throughput, y="value", hue="mode", x="alg", col="size", row="threads", kind="bar",
                sharey="row")
    plot_filename = f"../img/{experiment_name}-throughput.png"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)


def plot_phases(filename_detail, experiment_name, threads):
    data = pd.read_csv(filename_detail, header=0)
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)
    data = data[data["threads"] == threads]

    data = data.drop(columns=["lock", "threads", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", row="size", col="alg", kind="bar",
                sharey=False, order=phase_order)

    plot_filename = f"../img/{experiment_name}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def plot_phases_one_algorithm(filename_detail, experiment_name, alg):
    data = pd.read_csv(filename_detail, header=0)
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)
    data = data[data["alg"] == alg]
    data = data.drop(columns=["lock", "alg", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", row="size", col="threads", kind="bar",
                sharey=False, order=phase_order)

    plot_filename = f"../img/{experiment_name}-{alg}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def main():
    experiment_name = "compare-sgxv1"

    if len(sys.argv) < 2:
        print("Please specify run/plot/both")
        exit(-1)

    execution_command = sys.argv[1]

    if len(sys.argv) < 3:
        print(f"Experiment name defaults to {experiment_name}. Are you sure?")
        user_input = input("y/N: ")
        if user_input.lower().strip() != "y":
            exit()
    else:
        experiment_name = sys.argv[2]

    filename_detail = f"../data/{experiment_name}.csv"

    config = dict()
    config["experiment"] = True
    config["threads"] = [16]
    # config["algorithms"] = ["RHO", "RHT", "RSM", "CHT", "PHT", "MWAY", "INL"]
    config["algorithms"] = ["CHT", "PHT"]
    config["reps"] = 3
    config["modes"] = ["native", "sgx"]

    MB = 2 ** 20
    TUPLE_SIZE = 8
    TUPLE_PER_MB = MB/TUPLE_SIZE

    config["sizes"] = [(1 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (2 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (4 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (8 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (16 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (32 * TUPLE_PER_MB, 400 * TUPLE_PER_MB),
                       (64 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        compile_and_run(config, filename_detail)
    if execution_command in ["plot", "both"]:
        plot_throughput(filename_detail, experiment_name)
        # plot_phases(filename_detail, experiment_name, 4)
        for algorithm in config["algorithms"]:
            plot_phases_one_algorithm(filename_detail, experiment_name, algorithm)


if __name__ == '__main__':
    main()
