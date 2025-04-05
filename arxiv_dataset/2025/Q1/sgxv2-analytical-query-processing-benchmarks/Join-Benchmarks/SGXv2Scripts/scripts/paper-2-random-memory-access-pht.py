#!/usr/bin/python3
import sys
from itertools import chain, combinations

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.commons import delete_all_configurations
from helpers.runner import compile_and_run
phase_order = ["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2", "join_total", "build",
               "probe"]
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
    data = data[(data["alg"] == "PHT")]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    data = data.drop(columns=["threads", "alg", "lock", "unroll", "sort", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", col="part", row="size", kind="bar",
                sharey="row", order=phase_order)

    plot_filename = f"../img/{experiment_name}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)
    plt.close()


def paper_figure_random_access_join_absolut(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput")]
    data = data[(data["threads"] == 1)]
    data = data[data["part"] == "max"]
    data = data[data["size_r"] == (100 * TUPLE_PER_MB)]

    data["Setting"] = data["mode"]
    data["Setting"].replace({"sgx": "SGX DiE", "native": "Plain CPU"}, inplace=True)
    data["Throughput in $10^6$ rows/s"] = data["value"]

    c_palette_1 = ["#D56257", "#8e8e8e", "#29292b", "#176582"]
    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette(sns.color_palette(c_palette_1))

    sns.barplot(data, y="Throughput in $10^6$ rows/s", x="Setting")

    plt.ylim(bottom=0)
    plt.tight_layout()
    # plt.show()
    plot_filename = f"../img/paper-figure-random-memory-access-overhead-absolute.pdf"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


def paper_figure_random_access_join_relative(filename_detail):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput") & (data["alg"] == "NPO_st") & (data["threads"] == 1)]

    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})

    sgx_join = data[data["Setting"] == "SGX DiE"]
    plain_join = data[data["Setting"] == "Plain CPU"]

    sgx_join.set_index("size_r", inplace=True)
    plain_join.set_index("size_r", inplace=True)

    relative = sgx_join["value"] / plain_join["value"]

    relative_df = relative.reset_index()
    relative_df["Relative Throughput"] = relative_df["value"]
    relative_df["Hash Size"] = relative_df["size_r"].replace({1 * TUPLE_PER_MB: "1 MB",
                                                              50 * TUPLE_PER_MB: "50 MB",
                                                              100 * TUPLE_PER_MB: "100 MB"})

    plt.figure(figsize=(6, 3.25))
    sns.barplot(relative_df, y="Hash Size", x="Relative Throughput",
                color=sns.color_palette("deep")[2], errorbar="sd")

    print(relative_df.groupby("Hash Size").mean().to_string())

    plt.xlim(left=0)
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-random-memory-access-overhead-PHT.pdf"
    plt.savefig(plot_filename, dpi=300)
    # plt.show()
    plt.close()


def paper_figure_phases(filename_detail):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == 1)
                & (data["alg"] == "NPO_st")
                & (data["size_r"] == 100 * TUPLE_PER_MB)
                & (data["measurement"]).isin(["build", "probe"])]

    data["Runtime in s"] = data["value"] / 2900 / 10 ** 6
    data["Phase"] = data["measurement"].replace({"build": "Build", "probe": "Probe"})
    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})

    sns.barplot(data, y="Runtime in s", x="Phase", hue="Setting", order=["Build", "Probe"])

    plt.ylim(bottom=0)
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-NPO-phases.pdf"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_NPO_combined(filename_detail):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == 1) & (data["alg"] == "NPO_st")]
    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})

    data_slowdown = data[(data["measurement"] == "throughput")].reset_index().copy()

    sgx_join = data_slowdown[data_slowdown["Setting"] == "SGX DiE"]
    plain_join = data_slowdown[data_slowdown["Setting"] == "Plain CPU"]

    sgx_join.set_index("size_r", inplace=True)
    plain_join.set_index("size_r", inplace=True)

    relative = sgx_join["value"] / plain_join["value"]

    relative_df = relative.reset_index()
    relative_df["Relative Throughput"] = relative_df["value"]
    relative_df["Hash Size"] = relative_df["size_r"].replace({1 * TUPLE_PER_MB: "1 MB",
                                                              50 * TUPLE_PER_MB: "50 MB",
                                                              100 * TUPLE_PER_MB: "100 MB"})

    data_phases = data[data["measurement"].isin(["build", "probe"]) & (data["size_r"] == 100 * TUPLE_PER_MB)].copy()

    data_phases["Runtime in s"] = data_phases["value"] / 2900 / 10 ** 6
    data_phases["Phase"] = data_phases["measurement"].replace({"build": "Build", "probe": "Probe"})

    data_cache_misses = data[data["measurement"].isin(["build_miss", "probe_miss"])]
    mean_cache_misses = data_cache_misses.groupby(["size_r", "measurement", "Setting"])["value"].mean()

    print(mean_cache_misses.to_string())

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex="none", sharey="none", figsize=(6, 2.5), squeeze=True)

    sns.barplot(relative_df, x="Hash Size", y="Relative Throughput", color=sns.color_palette("deep")[2],
                errorbar="sd", ax=ax1)
    sns.barplot(data_phases, y="Runtime in s", x="Phase", hue="Setting", order=["Build", "Probe"],
                palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[2]], ax=ax2)
    sns.move_legend(
        ax2, "lower center",
        bbox_to_anchor=(-0.2, 1),
        frameon=False,
        title=None,
        ncols=2
    )

    for ax in [ax1, ax2]:
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
    ax1.set_ylim(top=1)

    hatches = ['', '\\\\']

    for bar in ax1.patches:
        bar.set_hatch(hatches[1])

    for i, bar in enumerate(ax2.patches):
        if i // 2 == 0:
            bar.set_hatch(hatches[0])
        elif i // 2 == 1:
            bar.set_hatch(hatches[1])

    for hatch, handle in zip(hatches, ax2.get_legend().legend_handles):
        handle.set_hatch(hatch)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    plot_filename = f"../img/paper-figure-NPO-combined.pdf"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def main():
    experiment_name = "random-access-pht"

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
    config["algorithms"] = ["NPO_st"]
    config["reps"] = 10
    config["modes"] = ["native", "sgx"]
    config["flags"] = [[]]

    config["sizes"] = [(1 * TUPLE_PER_MB, 1000 * TUPLE_PER_MB),
                       (50 * TUPLE_PER_MB, 1000 * TUPLE_PER_MB),
                       (100 * TUPLE_PER_MB, 1000 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run(config, filename_detail)
    if execution_command in ["plot", "both"]:
        sns.set_style("ticks")
        sns.set_context("notebook")
        sns.set_palette("deep")

        #paper_figure_random_access_join_relative(filename_detail)
        #paper_figure_phases(filename_detail)
        paper_figure_NPO_combined(filename_detail)


if __name__ == '__main__':
    main()
