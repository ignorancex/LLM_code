#!/usr/bin/python3
import sys
from itertools import chain, combinations

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.runner import compile_and_run

phase_order = ["total", "build", "probe"]
MB = 2 ** 20
TUPLE_SIZE = 8
TUPLE_PER_MB = MB/TUPLE_SIZE


def plot_throughput(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput")]
    data = data[(data["threads"] == 1)]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    data = data.drop(["threads", "lock", "unroll", "sort", "size_r", "size_s"], axis=1)

    sns.catplot(data, y="value", hue="mode", x="part", row="size", kind="bar",
                sharey="row")
    plot_filename = f"../img/{experiment_name}-throughput.png"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


def plot_phases(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == 1)]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    sns.catplot(data, y="value", x="measurement", hue="mode", row="size", col="alg", kind="bar", sharey="row",
                order=phase_order)

    native = data.loc[data["mode"] == "native", "value"].reset_index(drop=True)
    sgx = data.loc[data["mode"] == "sgx", "value"].reset_index(drop=True)

    relative = sgx / native
    relative = pd.concat([data.loc[data["mode"] == "native", ["alg", "size_r", "measurement"]], relative], axis=1)

    print(relative.to_string())

    plot_filename = f"../img/{experiment_name}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)
    plt.close()


def plot_phases_unroll(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == 1)]
    data = data[data["alg"] == "NPO_no"]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    sns.catplot(data, y="value", x="measurement", hue="mode", row="size", col="unroll", kind="bar", sharey="row",
                order=phase_order)

    native = data.loc[data["mode"] == "native", "value"].reset_index(drop=True)
    sgx = data.loc[data["mode"] == "sgx", "value"].reset_index(drop=True)

    relative = sgx / native
    relative = pd.concat([data.loc[data["mode"] == "native", ["alg", "unroll", "size_r", "measurement"]], relative],
                         axis=1)

    print(relative.to_string())

    plot_filename = f"../img/{experiment_name}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)
    plt.close()


def plot_unroll_speedup(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data.loc[(data["threads"] == 1) & (data["alg"] == "NPO_no") & (data["measurement"] != "throughput"), :]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    data = data.drop(["size_r", "size_s", "alg", "threads", "lock", "sort", "part"], axis=1)

    scalar = data.loc[data["unroll"] == "scalar", :]
    unrolled = data.loc[data["unroll"] == "unrolled", "value"]
    unrolled.name = "unrolled"
    unrolled.reset_index(drop=True, inplace=True)

    combined = scalar.reset_index(drop=True)
    combined["unrolled"] = unrolled

    combined["improvement"] = 1 - combined["unrolled"] / combined["value"]

    sns.catplot(combined, y="improvement", x="measurement", hue="mode", row="size", kind="bar", sharey="row",
                order=phase_order)

    print(combined.to_string())

    plot_filename = f"../img/{experiment_name}-improvement.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)
    plt.close()


def main():
    experiment_name = "no-part-investigation"

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
    config["algorithms"] = ["NPO_no"]
    config["reps"] = 3
    config["modes"] = ["native", "sgx"]
    config["flags"] = [[], ["UNROLL"]]

    config["sizes"] = [(1 * TUPLE_PER_MB, 1000 * TUPLE_PER_MB),
                       (50 * TUPLE_PER_MB, 1000 * TUPLE_PER_MB),
                       (100 * TUPLE_PER_MB, 1000 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        compile_and_run(config, filename_detail)
    if execution_command in ["plot", "both"]:
        pass
        # plot_throughput(filename_detail, experiment_name)
        plot_phases_unroll(filename_detail, experiment_name)
        # paper_figure_random_access_join_absolut(filename_detail, experiment_name)
        # paper_figure_random_access_join_relative(filename_detail, experiment_name)


if __name__ == '__main__':
    main()
