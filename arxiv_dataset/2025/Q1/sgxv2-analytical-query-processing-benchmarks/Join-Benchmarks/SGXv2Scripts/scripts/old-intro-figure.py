#!/usr/bin/python3
import sys
from itertools import chain, combinations

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.runner import compile_and_run


MB = 2 ** 20
TUPLE_SIZE = 8
TUPLE_PER_MB = MB/TUPLE_SIZE


def plot_phases(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == 1)]
    data = data[(data["alg"] == "RHO")]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    data = data.drop(columns=["threads", "alg", "lock", "unroll", "sort", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", col="part", row="size", kind="bar",
                sharey="row", order=["total", "partition", "partition_1", "partition_r", "partition_s",
                                     "partition_2", "join_total", "build", "probe"])

    plot_filename = f"../img/{experiment_name}-phases.png"
    # plt.savefig(plot_filename, transparent=False, dpi=150)
    plt.show()
    plt.close()


def paper_figure_intro():
    sgxv1 = pd.read_csv("../data/intro-sgxv1.csv", header=0)
    sgxv2 = pd.read_csv("../data/intro-sgxv2.csv", header=0)
    sgxv1["version"] = 1
    sgxv2["version"] = 2

    data = pd.concat([sgxv1, sgxv2])
    data = data[(data["threads"] == 8)]
    data = data[(data["alg"] == "RHO")]
    data = data[data["size_r"] == (10 * TUPLE_PER_MB)]
    data = data[data["measurement"] == "throughput"]

    data["Setting"] = data["mode"].replace({"sgx": "SGX Data in Enclave", "native": "Plain CPU"})

    data["SGX Version & Optimization"] = data["version"].astype(str) + data["unroll"]
    data["SGX Version & Optimization"].replace({"1scalar": "SGXv1",
                                                "1unrolled": "SGXv1 Optimized",
                                                "2scalar": "SGXv2",
                                                "2unrolled": "SGXv2 Optimized"}, inplace=True)
    data["Throughput in $10^6$ rows/s"] = data["value"]

    c_palette_1 = ["#D56257", "#8e8e8e", "#29292b", "#176582"]
    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette(sns.color_palette(c_palette_1))

    sns.barplot(data, y="Throughput in $10^6$ rows/s", x="SGX Version & Optimization", hue="Setting",
                order=["SGXv1", "SGXv2", "SGXv2 Optimized"])
    # plt.xticks(rotation=90)
    plt.ylim(bottom=0, top=1000)
    plt.xlabel("")
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-intro.pdf"
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    plt.close()


def paper_figure_intro_2():
    sgxv1 = pd.read_csv("../data/intro-sgxv1-10.csv", header=0)
    sgxv2 = pd.read_csv("../data/intro-sgxv2-10.csv", header=0)
    sgxv1["version"] = 1
    sgxv2["version"] = 2

    data = pd.concat([sgxv1, sgxv2])
    # data = sgxv2
    data = data[(data["threads"] == 8)]
    data = data[(data["alg"] == "RHO")]
    data = data[data["size_r"] == (100 * TUPLE_PER_MB)]
    data = data[data["measurement"] == "throughput"]

    data["Setting"] = data["mode"].replace({"sgx": "SGX Data in Enclave", "native": "Plain CPU"})

    data["SGX Version & Optimization"] = data["version"].astype(str) + data["unroll"]
    data["SGX Version & Optimization"].replace({"1scalar": "SGXv1",
                                                "1unrolled": "SGXv1 Optimized",
                                                "2scalar": "SGXv2",
                                                "2unrolled": "SGXv2 Optimized"}, inplace=True)

    data["Setting Version"] = data["version"].astype(str) + data["unroll"] + data["mode"]
    data["Setting Version"].replace({
        "2unrollednative": "Plain CPU\nXeon Gold 6326",
        "1scalarsgx": "SGXv1\nXeon E-2288G",
        "2scalarsgx": "SGXv2 without\nOptimizations",
        "2unrolledsgx": "SGXv2 with\nour Optimizations"
    }, inplace=True)

    data["Throughput in $10^6$ rows/s"] = data["value"]

    c_palette_1 = ["#29292b", "#176582", "#8e8e8e", "#D56257"]
    sns.set_style("ticks")
    sns.set_context("notebook")
    # sns.set_palette("deep")

    fig = plt.figure(figsize=(6, 3))
    p = sns.color_palette("deep")
    plot = sns.barplot(data,
                       y="Setting Version",
                       x="Throughput in $10^6$ rows/s",
                       order=["Plain CPU\nXeon Gold 6326", "SGXv1\nXeon E-2288G", "SGXv2 without\nOptimizations",
                       "SGXv2 with\nour Optimizations"],
                       errorbar="sd",
                       palette=[p[0], p[3], p[4], p[2]])

    # for bar in plot.containers:
    #    for child in bar:
    #        child.set_zorder(0)
    for line in plot.get_lines():
        line.set_zorder(1)

    averages = data.groupby("Setting Version")["Throughput in $10^6$ rows/s"].mean()

    sgxv1_percent = averages['SGXv1\nXeon E-2288G'] / averages["Plain CPU\nXeon Gold 6326"] * 100
    sgxv2_wo_percent = averages['SGXv2 without\nOptimizations'] / averages["Plain CPU\nXeon Gold 6326"] * 100
    sgxv2_w_percent = averages['SGXv2 with\nour Optimizations'] / averages["Plain CPU\nXeon Gold 6326"] * 100

    plot.bar_label(plot.containers[0],
                   labels=["100%", f"{sgxv1_percent:.0f}%", f"{sgxv2_wo_percent:.0f}%", f"{sgxv2_w_percent:.0f}%"],
                   padding=3)

    plt.xlim(left=0, right=800)
    plt.ylabel("")
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-intro.pdf"
    plt.savefig(plot_filename, dpi=300)
    # plt.show()
    plt.close()


def main():
    experiment_name = "intro"

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
    config["threads"] = [8]
    config["algorithms"] = ["RHO"]
    config["reps"] = 10
    config["modes"] = ["native", "sgx"]
    config["flags"] = [["NOAVX", "FORCE_2_PHASES", "CONSTANT_RADIX_BITS"],
                       ["SPIN_LOCK", "UNROLL", "NOAVX", "FORCE_2_PHASES"]]

    config["sizes"] = [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        compile_and_run(config, filename_detail)
    if execution_command in ["plot", "both"]:
        pass
        # plot_phases(filename_detail, experiment_name)
        paper_figure_intro_2()


if __name__ == '__main__':
    main()
