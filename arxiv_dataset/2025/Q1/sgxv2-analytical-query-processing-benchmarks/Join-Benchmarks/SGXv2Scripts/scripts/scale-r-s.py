#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.runner import compile_and_run

phase_order = ["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2", "join_total", "build", "probe"]


def plot_throughput(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data_throughput = data[(data["measurement"] == "throughput") & (data["alg"] == "RHO")]

    data_throughput = data_throughput.drop(["lock", "alg"], axis=1)

    sns.catplot(data_throughput, y="value", hue="size_r", x="size_s", row="threads", col="mode", kind="bar",
                sharey="row")
    plot_filename = f"../img/{experiment_name}-throughput.png"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)


def plot_phases(algorithms: list[str], filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    for alg in algorithms:
        alg_data = data[data["alg"] == alg]

        alg_data = alg_data.drop(columns=["lock", "alg", "size_r", "size_s"])

        sns.catplot(alg_data, y="value", x="measurement", hue="mode", row="size", col="threads", kind="bar",
                    sharey=False, order=phase_order)

        plot_filename = f"../img/{experiment_name}-{alg}.pdf"
        plt.savefig(plot_filename, transparent=False, dpi=150)


def main():
    experiment_name = "scale-r-s"

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

    filename_avg = f"../data/{experiment_name}-avg.csv"
    filename_detail = f"../data/{experiment_name}.csv"

    config = dict()
    config["experiment"] = True
    config["threads"] = [1, 16]
    config["algorithms"] = ["RHO"]
    config["reps"] = 1
    config["modes"] = ["native", "sgx"]

    # subtract 3 from the exponent to get data sizes instead of tuple counts
    r_sizes = [2 ** (p-3) for p in [10, 11, 12, 13, 14, 15, 20, 25]]
    s_sizes = [2 ** (p-3) for p in [15, 20, 25, 30]]

    config["sizes"] = [(r, s) for s in s_sizes for r in r_sizes if r <= s]

    if execution_command in ["run", "both"]:
        compile_and_run(config, filename_avg, filename_detail, flags=["SPIN_LOCK"])
    if execution_command in ["plot", "both"]:
        plot_throughput(filename_detail, experiment_name)
        plot_phases(config["algorithms"], filename_detail, experiment_name)


if __name__ == '__main__':
    main()
