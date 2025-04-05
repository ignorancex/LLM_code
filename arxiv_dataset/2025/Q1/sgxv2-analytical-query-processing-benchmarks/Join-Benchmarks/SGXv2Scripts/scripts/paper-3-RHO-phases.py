#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.commons import TUPLE_PER_MB, delete_all_configurations
from helpers.runner import ExperimentConfig, compile_and_run_simple_flags


def plot_phases(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == 1)]
    data = data[(data["alg"] == "RHO")]
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    data = data.drop(columns=["threads", "alg", "lock", "unroll", "sort", "size_r", "size_s"])

    sns.catplot(data, y="value", x="measurement", hue="mode", col="part", row="size", kind="bar",
                sharey="row", order=["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2",
                                     "join_total", "build", "probe"])

    plot_filename = f"../img/{experiment_name}-phases.png"
    # plt.savefig(plot_filename, transparent=False, dpi=150)
    plt.show()
    plt.close()


def paper_figure_RHO_phases(filename_detail):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == 1)]
    data = data[(data["alg"] == "RHO")]
    data = data[data["size_r"] == (100 * TUPLE_PER_MB)]
    data = data[data["measurement"].isin(["partition_r", "partition_s", "build", "probe"])]
    data = data[data["unroll"] == "scalar"]

    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})
    data["Phase"] = data["measurement"].replace({
        "partition_r": "Histogram",
        "partition_s": "Partition Copy",
        "build": "Build Hash Table",
        "probe": "Probe + Count"
    })
    phase_order = ["Histogram", "Partition Copy", "Build Hash Table", "Probe + Count"]
    data["Time in ms"] = data["value"] / (2.9 * 10 ** 6)

    c_palette_1 = ["#D56257", "#8e8e8e", "#29292b", "#176582"]
    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette(sns.color_palette(c_palette_1))

    plt.figure(figsize=(6, 3.25))
    sns.barplot(data, y="Time in ms", x="Phase", hue="Setting", order=phase_order)
    # plt.xticks(rotation=90)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-RHO-phases.pdf"
    # plt.show()
    plt.savefig(plot_filename, dpi=300)


def paper_figure_RHO_phases_both(filename_detail, threads):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == threads)]
    data = data[(data["alg"] == "RHO")]
    # data = data[data["size_r"] == (100 * TUPLE_PER_MB)]
    data = data[data["measurement"].isin(["partition_r", "partition_s", "partition_2_c", "partition_2_h", "build",
                                          "probe"])]

    # data = data[data["unroll"] == "scalar"]

    PHASES = ["Hist. 1", "Copy 1", "Hist. 2", "Copy 2", "Build", "Probe"]

    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})
    data["Phase"] = data["measurement"].replace({
        "partition_r": PHASES[0],
        "partition_s": PHASES[1],
        "partition_2_h": PHASES[2],
        "partition_2_c": PHASES[3],
        "build": PHASES[4],
        "probe": PHASES[5],
    })

    data["Time in ms"] = data["value"] / (2.9 * 10 ** 6)
    data["Unrolling"] = data["flags"].str.contains("UNROLL").replace({False: "No", True: "Yes"})

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette([sns.color_palette("deep")[0], sns.color_palette("deep")[2]])

    plot = sns.catplot(data, y="Time in ms", x="Phase", hue="Setting", row="Unrolling", order=PHASES,
                       row_order=["No", "Yes"], kind="bar", height=1.6, aspect=2.6, sharey=True, errorbar="sd")

    axes = plot.axes.flatten()
    axes[0].set_title("Not Optimized", fontdict={"fontsize": 13})
    axes[1].set_title("Optimized with Unrolling + Reordering", fontdict={"fontsize": 13})

    plt.ylim(bottom=0)
    if threads == 1:
        plt.yticks([100, 200, 300, 400])
    elif threads == 16:
        plt.yticks([5, 10, 15, 20, 25])
    plt.xlabel("")
    sns.move_legend(plot, "upper right", frameon=True, bbox_to_anchor=(1, 0.92))
    for ax in axes:
        ax.grid(axis="y")
        #for tick in ax.get_xticklabels():
        #    tick.set_rotation(-30)
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-RHO-phases-both-{threads}.pdf"
    # plt.show()
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)


def paper_figure_RHO_phases_mitigation(filename_detail, threads):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == threads)]
    data = data[(data["alg"] == "RHO")]
    # data = data[data["size_r"] == (100 * TUPLE_PER_MB)]
    data = data[data["measurement"].isin(["partition_r", "partition_s", "partition_2_c", "partition_2_h", "build",
                                          "probe"])]

    # data = data[data["unroll"] == "scalar"]

    PHASES = ["Hist. 1", "Copy 1", "Hist. 2", "Copy 2", "Build", "Probe"]

    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})
    data["Phase"] = data["measurement"].replace({
        "partition_r": PHASES[0],
        "partition_s": PHASES[1],
        "partition_2_h": PHASES[2],
        "partition_2_c": PHASES[3],
        "build": PHASES[4],
        "probe": PHASES[5],
    })

    data["Time in ms"] = data["value"] / (2.9 * 10 ** 6)
    data["Unrolling"] = data["flags"].str.contains("UNROLL")

    data["M/U"] = data["mitigation"].astype(int) + data["Unrolling"].astype(int) * 2

    # data = data[data["M/U"].isin([0, 1, 3])]

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette([sns.color_palette("deep")[0], sns.color_palette("deep")[2]])

    plot = sns.catplot(data, y="Time in ms", x="Phase", hue="Setting", row="M/U", order=PHASES,
                       row_order=[0, 1, 2, 3], kind="bar", height=1.6, aspect=2.6, sharey=True, errorbar="sd")

    axes = plot.axes.flatten()
    axes[0].set_title("SSB Mitigation Off, Not Optimized", fontdict={"fontsize": 13})
    axes[1].set_title("SSB Mitigation On, Not Optimized", fontdict={"fontsize": 13})
    axes[2].set_title("SSB Mitigation Off, Optimized", fontdict={"fontsize": 13})
    axes[3].set_title("SSB Mitigation On, Optimized", fontdict={"fontsize": 13})

    plt.ylim(bottom=0)
    if threads == 1:
        plt.yticks([100, 200, 300, 400])
    elif threads == 16:
        plt.yticks([5, 10, 15, 20, 25])
    plt.xlabel("")
    sns.move_legend(plot, "upper right", frameon=True, bbox_to_anchor=(1, 0.92))
    for ax in axes:
        ax.grid(axis="y")
        #for tick in ax.get_xticklabels():
        #    tick.set_rotation(-30)
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-RHO-phases-mitigation.pdf"
    # plt.show()
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)


def paper_figure_RHO_phases_mitigation_color(filename_detail, threads):
    data = pd.read_csv(filename_detail, header=0)

    data["Time in ms"] = data["value"] / (2.9 * 10 ** 6)
    data["Unrolling"] = data["flags"].str.contains("UNROLL")

    SETTINGS = ["Plain CPU", "SGX DiE", "Plain CPU M"]
    PHASES = ["Hist. 1", "Copy 1", "Hist. 2", "Copy 2", "Build", "Probe"]

    data["Setting"] = (data["mode"] + data["mitigation"].astype(str)).replace({
        "sgxFalse": SETTINGS[1],
        "sgxTrue": SETTINGS[1],
        "nativeFalse": SETTINGS[0],
        "nativeTrue": SETTINGS[2]
    })
    data["Phase"] = data["measurement"].replace({
        "partition_r": PHASES[0],
        "partition_s": PHASES[1],
        "partition_2_h": PHASES[2],
        "partition_2_c": PHASES[3],
        "build": PHASES[4],
        "probe": PHASES[5],
    })

    data = data[(data["threads"] == threads)]
    data = data[(data["alg"] == "RHO")]
    totals = data[data["measurement"] == "total"]
    data = data[data["measurement"].isin(["partition_r", "partition_s", "partition_2_c", "partition_2_h", "build",
                                          "probe"])]

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette([sns.color_palette("deep")[0], sns.color_palette("deep")[2], sns.color_palette("deep")[3]])

    plot = sns.catplot(data, y="Time in ms", x="Phase", hue="Setting", row="Unrolling", order=PHASES,
                       row_order=[False, True], kind="bar", height=1.3, aspect=3.2, sharey=True, errorbar="sd",
                       hue_order=SETTINGS)

    axes = plot.axes.flatten()
    axes[0].set_title("Not Optimized", fontdict={"fontsize": 13})
    axes[1].set_title("Optimized with Unroll + Reorder", fontdict={"fontsize": 13})
    axes[0].set_ylabel("")
    axes[1].yaxis.set_label_coords(-0.11,1.25)

    hatches = ['', '\\\\', '--']

    for ax in axes:
        for i, bar in enumerate(ax.patches):
            if i // 6 == 1:
                bar.set_hatch(hatches[1])
            elif i // 6 == 2:
                bar.set_hatch(hatches[2])
        ax.grid(axis="y")
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    plt.ylim(bottom=0)
    if threads == 1:
        plt.yticks([100, 200, 300, 400])
    elif threads == 16:
        plt.yticks([5, 10, 15, 20, 25])
    plt.xlabel("")
    sns.move_legend(plot, "lower center", frameon=False, bbox_to_anchor=(0.5, 0.9), ncols=3, title=None)

    for hatch, handle in zip(hatches, plot._legend.legend_handles):
        handle.set_hatch(hatch)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plot_filename = f"../img/paper-figure-RHO-phases-mitigation-color.pdf"
    # plt.show()
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)

    means = totals.groupby(["Unrolling", "Setting"])["Time in ms"].mean()
    print(means.to_string())

    print(f"in_sgx_improvement = {1 - means[True]['SGX DiE'] / means[False]['SGX DiE']}")
    print(f"relative_throughput_before = {means[False]['Plain CPU'] / means[False]['SGX DiE']}")
    print(f"relative_throughput_after = {means[True]['Plain CPU'] / means[True]['SGX DiE']}")
    print(f"relative_throughput_after_mitigation = {means[True]['Plain CPU M'] / means[True]['SGX DiE']}")


def paper_figure_RHO_phases_only_default(filename_detail, threads):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == threads)]
    data = data[(data["alg"] == "RHO")]
    # data = data[data["size_r"] == (100 * TUPLE_PER_MB)]
    data = data[data["measurement"].isin(["partition_r", "partition_s", "partition_2_c", "partition_2_h", "build",
                                          "probe"])]
    data = data[~data["flags"].str.contains("UNROLL")]

    # data = data[data["unroll"] == "scalar"]

    PHASES = ["Hist. 1", "Copy 1", "Hist. 2", "Copy 2", "Build", "Probe"]

    data["Setting"] = data["mode"].replace({"sgx": "SGX DiE", "native": "Plain CPU"})
    data["Phase"] = data["measurement"].replace({
        "partition_r": PHASES[0],
        "partition_s": PHASES[1],
        "partition_2_h": PHASES[2],
        "partition_2_c": PHASES[3],
        "build": PHASES[4],
        "probe": PHASES[5],
    })

    data["Time in ms"] = data["value"] / (2.9 * 10 ** 6)

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette([sns.color_palette("deep")[0], sns.color_palette("deep")[2]])

    plt.figure(figsize=(6, 3.5))
    plot = sns.barplot(data, y="Time in ms", x="Phase", hue="Setting", order=PHASES, errorbar="sd")

    plt.ylim(bottom=0)
    if threads == 1:
        plt.yticks([100, 200, 300, 400])
    elif threads == 16:
        plt.yticks([5, 10, 15, 20, 25])
    plt.xlabel("Radix Join Phase")
    sns.move_legend(plot, "upper right", frameon=True, bbox_to_anchor=(1, 0.92))
    plot.grid(axis="y")
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-RHO-phases-only-default.png"
    # plt.show()
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)


def main():
    experiment_name = "RHO-phases"

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

    config = ExperimentConfig(
        ["native", "sgx"],
        [["FORCE_2_PHASES"], ["FORCE_2_PHASES", "UNROLL"]],
        [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)],
        ["RHO"],
        [1],
        [False],
        3,
        mitigation=[False, True]
    )

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run_simple_flags(config, filename_detail)
    if execution_command in ["plot", "both"]:
        # plot_phases(filename_detail, experiment_name)
        # paper_figure_RHO_phases(filename_detail)
        paper_figure_RHO_phases_mitigation_color(filename_detail, 1)


if __name__ == '__main__':
    main()
