#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.commons import delete_all_configurations
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
    plt.close()


def plot_phases(filename_detail, experiment_name, threads):
    data = pd.read_csv(filename_detail, header=0)
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)
    data = data[data["threads"] == threads]

    data = data.drop(columns=["lock", "threads", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", row="size", col="alg", kind="bar",
                sharey=False, order=phase_order)

    plot_filename = f"../img/{experiment_name}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)
    plt.close()


def plot_phases_one_algorithm(filename_detail, experiment_name, alg):
    data = pd.read_csv(filename_detail, header=0)
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)
    data = data[data["alg"] == alg]
    data = data.drop(columns=["lock", "alg", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", row="size", col="threads", kind="bar",
                sharey=False, order=phase_order)

    plot_filename = f"../img/{experiment_name}-{alg}-phases.png"
    plt.savefig(plot_filename, transparent=False, dpi=150)
    plt.close()


def paper_plot_join_overview(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput") & (data["alg"].isin(["PHT", "RHO", "MWAY", "INL", "CrkJoin"]))]

    data["Throughput in\n$10^6$ rows/s"] = data["value"]
    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})
    data["Join Algorithm"] = data["alg"]

    # c_palette_1 = ["#176582", "#D56257"]
    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette([sns.color_palette("deep")[0], sns.color_palette("deep")[2]])

    f = plt.figure(figsize=(6, 2.5))
    plot = sns.barplot(data, y="Throughput in\n$10^6$ rows/s", hue="Setting", x="Join Algorithm",
                       order=["PHT", "RHO", "MWAY", "INL", "CrkJoin"], errorbar="sd")
    sns.move_legend(plot, "lower center", frameon=False, bbox_to_anchor=(0.5, 0.95), title=None, ncols=2)

    hatches = ['', '\\\\']

    for i, bar in enumerate(plot.patches):
        if i // 5 == 0:
            bar.set_hatch(hatches[0])
        elif i // 5 == 1:
            bar.set_hatch(hatches[1])

    for hatch, handle in zip(hatches, f.axes[0].get_legend().legend_handles):
        handle.set_hatch(hatch)

    plt.grid(axis="y")
    plt.ylim(bottom=0)
    plt.tight_layout()

    grouped = data.groupby(["mode", "alg"])["value"].mean()
    relative = grouped["sgx"] / grouped["native"]

    print(relative.to_string())

    plot_filename = f"../img/paper-figure-join-overview.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()


def paper_plot_join_overview_two_size(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput") & (data["alg"].isin(["PHT", "RHO", "MWAY", "INL", "CrkJoin"]))]

    data["Throughput in\n$10^6$ rows/s"] = data["value"]
    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})
    data["Join Algorithm"] = data["alg"]
    data["Size"] = data["size_r"].astype(int).replace({13107200: "100/400", 1310720: "10/4000"})

    # c_palette_1 = ["#176582", "#D56257"]
    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette([sns.color_palette("deep")[0], sns.color_palette("deep")[2]])

    plot = sns.catplot(data, y="Throughput in\n$10^6$ rows/s", hue="Setting", x="Join Algorithm", col="Size",
                       order=["PHT", "RHO", "MWAY", "INL", "CrkJoin"], errorbar="sd", kind="bar", height=2.8, aspect=1)
    sns.move_legend(plot, "upper right", frameon=True, bbox_to_anchor=(0.57, 0.85))

    axes = plot.axes.flatten()
    for ax in axes:
        ax.grid(axis="y")
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    grouped = data.groupby(["mode", "alg", "Size"])["value"].mean()
    relative = grouped["sgx"] / grouped["native"]

    print(relative.to_string())

    plot_filename = f"../img/paper-figure-join-overview-two-size.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()


def main():
    experiment_name = "join-overview"

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
    config["algorithms"] = ["PHT", "RHO", "MWAY", "INL", "CrkJoin"]
    config["reps"] = 10
    config["modes"] = ["native", "sgx"]
    config["flags"] = [["FORCE_2_PHASES"]]

    MB = 2 ** 20
    TUPLE_SIZE = 8
    TUPLE_PER_MB = MB/TUPLE_SIZE

    config["sizes"] = [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run(config, filename_detail)
    if execution_command in ["plot", "both"]:
        #plot_throughput(filename_detail, experiment_name)
        paper_plot_join_overview(filename_detail, experiment_name)
        # plot_phases(filename_detail, experiment_name, 4)
        #for algorithm in config["algorithms"]:
            #plot_phases_one_algorithm(filename_detail, experiment_name, algorithm)


if __name__ == '__main__':
    main()
