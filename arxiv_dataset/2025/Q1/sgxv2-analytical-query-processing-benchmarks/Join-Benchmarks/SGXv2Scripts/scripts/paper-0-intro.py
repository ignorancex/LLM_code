#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
from seaborn import axes_style

from helpers.commons import TUPLE_PER_MB, delete_all_configurations
from helpers.runner import compile_and_run_simple_flags, ExperimentConfig


def paper_intro_plot(filename_detail):
    joins = ["Join optimized\nfor SGXv1", "State-of-the-art\nradix join", "+Our\noptimization",
             "+Our optimization\nwithout SGX"]

    data = pd.read_csv(filename_detail, header=0)
    data = data[data["measurement"] == "throughput"]
    data["Join Throughput in\n$10^6$ rows/s"] = data["value"]

    data["Join"] = (data["mode"] + data["alg"] + data["flags"].str.contains("UNROLL").astype(str)).replace({
        "nativeCRKJFalse": "",
        "nativeCRKJTrue": "",
        "sgxCRKJFalse": joins[0],
        "sgxCRKJTrue": joins[0],
        "nativeRHOFalse": "",
        "nativeRHOTrue": joins[3],
        "sgxRHOFalse": joins[1],
        "sgxRHOTrue": joins[2],
    })

    plt.xticks(rotation=90)

    f = plt.figure(figsize=(6, 2.5))
    p = sns.barplot(data,  y="Join Throughput in\n$10^6$ rows/s", x="Join", hue="Join", order=joins, hue_order=joins)
    p.set(xlabel=None)
    p.set_ylim(bottom=0, top=1900)
    means = data.groupby("Join")["Join Throughput in\n$10^6$ rows/s"].mean()

    f.axes[0].yaxis.set_label_coords(-0.12, 0.43)
    f.axes[0].axvline(2.5, linestyle="--", color="black", linewidth=1)

    for container in p.containers:
        p.bar_label(container, fmt="%.1f")

    hatches = ['', '//', '\\\\', '--']

    for hatch, bar in zip(hatches, p.patches):
        bar.set_hatch(hatch)

    plt.xticks(ticks=[0, 1, 2, 3], labels=["Join designed\nfor SGXv1", "Radix Join designed\nfor Plain CPU",
                                           "+New\nOptimization", "Radix Join\n+New Optimization"], fontsize=9)

    # p.bar_label(p.containers[0]) #, labels=[f"{means[joins[i]]:.0f}" for i in range(len(joins))], padding=3)

    plt.grid(axis="y")
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-intro-new.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()


def paper_intro_plot_mitigation(filename_detail):
    joins = ["SGXv1-optimized\nJoin in SGXv2", "Radix Join\nin SGXv2", "SGXv2-optimized\nRadix Join",
             "Radix Join\noutside Enclave", "Radix Join\nSSB mitigation"]

    data = pd.read_csv(filename_detail, header=0)
    data = data[data["measurement"] == "throughput"]
    data["Throughput in\n$10^6$ rows/s"] = data["value"]

    data["Join"] = (data["mode"] + data["alg"] + data["flags"].str.contains("UNROLL").astype(str) + data[
        "mitigation"].astype(str)).replace({
        "sgxCRKJFalseFalse": joins[0],
        "sgxCRKJTrueFalse": joins[0],
        "nativeRHOTrueFalse": joins[3],
        "sgxRHOFalseFalse": joins[1],
        "sgxRHOTrueFalse": joins[2],
        "nativeRHOTrueTrue": joins[4],
    })

    plt.xticks(rotation=90)

    plt.figure(figsize=(6, 2.5))
    p = sns.barplot(data, y="Throughput in\n$10^6$ rows/s", x="Join", hue="Join", order=joins[:-1], hue_order=joins[:-1])
    p.set(xlabel=None)
    p.set_ylim(bottom=0, top=1600)
    means = data.groupby("Join")["Throughput in\n$10^6$ rows/s"].mean()

    for container in p.containers:
        p.bar_label(container, fmt="%.1f")

    # p.bar_label(p.containers[0]) #, labels=[f"{means[joins[i]]:.0f}" for i in range(len(joins))], padding=3)

    plt.grid(axis="y")
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-intro-mitigation.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()


def main():
    experiment_name = "intro"

    if len(sys.argv) < 2:
        print("Please specify run/plot/both")
        exit(-1)

    execution_command = sys.argv[1]

    if len(sys.argv) < 3 and execution_command == "run":
        print(f"Experiment name defaults to {experiment_name}. Are you sure?")
        user_input = input("y/N: ")
        if user_input.lower().strip() != "y":
            exit()
    elif len(sys.argv) == 3:
        experiment_name = sys.argv[2]

    filename_detail = f"../data/{experiment_name}.csv"

    config = ExperimentConfig(
        ["native", "sgx"],
        [["UNROLL", "FORCE_2_PHASES"], ["FORCE_2_PHASES"]],
        [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)],
        ["RHO", "CRKJ"],
        [16],
        [False],
        10,
        mitigation=[False, True]
    )

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run_simple_flags(config, filename_detail)
    if execution_command in ["plot", "both"]:
        sns.set_style("ticks")
        sns.set_context("notebook")
        sns.set_palette("deep")

        so.Plot.config.theme.update(axes_style("ticks"))
        paper_intro_plot(filename_detail)


if __name__ == '__main__':
    main()
