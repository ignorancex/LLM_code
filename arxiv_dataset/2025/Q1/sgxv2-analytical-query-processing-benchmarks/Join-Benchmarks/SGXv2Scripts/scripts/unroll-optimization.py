#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

from helpers.runner import compile_and_run

phase_order = ["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2", "join_total", "build",
               "probe"]
matplotlib.use('qtagg')
MB = 2 ** 20
TUPLE_SIZE = 8
TUPLE_PER_MB = MB/TUPLE_SIZE

def plot_throughput(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput")]
    data = data[(data["threads"] == 1)]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    # data = data.drop(["threads", "lock", "unroll", "sort", "size_r", "size_s"], axis=1)

    sns.catplot(data, y="value", hue="mode", x="unroll", row="size", kind="bar",
                sharey="row")
    plot_filename = f"../img/{experiment_name}-throughput.png"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)


def plot_phases(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == 1)]
    data = data[(data["alg"] == "RHO")]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    # data = data.drop(columns=["threads", "alg", "lock", "unroll", "sort", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", col="unroll", row="size", kind="bar",
                sharey=False, order=phase_order)

    plot_filename = f"../img/{experiment_name}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def paper_figure_unroll_optimization(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["measurement"] == "throughput")]
    data = data[(data["threads"] == 1)]
    data = data[(data["alg"] == "RHO")]
    data = data[data["size_r"] == (100 * TUPLE_PER_MB)]

    data["Setting"] = data["mode"]
    data["Setting"].replace({"sgx": "SGX Data in Enclave", "native": "Plain CPU"}, inplace=True)
    data["Unrolling"] = data["unroll"].replace({"scalar": "Scalar", "unrolled": "Unrolled"})
    data["Throughput in $10^6$ rows/s"] = data["value"]

    c_palette_1 = ["#D56257", "#8e8e8e", "#29292b", "#176582"]
    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette(sns.color_palette(c_palette_1))

    sns.barplot(data, x="Unrolling", y="Throughput in $10^6$ rows/s", hue="Setting")

    plt.ylim(bottom=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"../img/paper-figure-unroll.pdf", dpi=300)
    plt.close()


def main():
    experiment_name = "unroll-impact"

    if len(sys.argv) < 2:
        print("Please specify run/plot/both")
        exit(-1)

    execution_command = sys.argv[1]

    if len(sys.argv) < 3 and execution_command != "plot":
        print(f"Experiment name defaults to {experiment_name}. Are you sure?")
        user_input = input("y/N: ")
        if user_input.lower().strip() != "y":
            exit()
    elif len(sys.argv) == 3:
        experiment_name = sys.argv[2]

    filename_detail = f"../data/{experiment_name}.csv"

    config = dict()
    config["experiment"] = True
    config["threads"] = [1]
    config["algorithms"] = ["RHO"]
    config["reps"] = 3
    config["modes"] = ["native", "sgx"]
    config["flags"] = [["SPIN_LOCK"], ["SPIN_LOCK", "UNROLL"]]

    config["sizes"] = [(10 * TUPLE_PER_MB, 40 * TUPLE_PER_MB), (100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        compile_and_run(config, filename_detail)
    if execution_command in ["plot", "both"]:
        #plot_throughput(filename_detail, experiment_name)
        #plot_phases(filename_detail, experiment_name)
        paper_figure_unroll_optimization(filename_detail, experiment_name)


if __name__ == '__main__':
    main()
