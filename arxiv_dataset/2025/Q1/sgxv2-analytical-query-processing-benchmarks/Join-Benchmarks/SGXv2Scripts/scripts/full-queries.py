#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
from seaborn import axes_style

from helpers.commons import delete_all_configurations
from helpers.tpch_runner import compile_and_run_simple_flags, ExperimentConfig, TPCH_PHASE_ORDER


def plot_stacked(filename_detail, experiment_name, sf):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] != "throughput")]
    #data = data.rename_axis("index1").reset_index()

    data = data.pivot_table(index=["query", "alg", "sf"], columns="measurement", values="value")
    data["rest"] = data["total"] - data["selection"] - data["join"]
    data = data.drop(["total"], axis=1)
    data = data.melt(value_vars=['join', 'selection', 'rest'], value_name="Query Runtime in ms", ignore_index=False).reset_index()
    data["Query"] = "Q" + data["query"].apply(str)
    data["Query Runtime in ms"] /= 1000
    data["Join Algorithm"] = data["alg"]
    data["Phase"] = data["measurement"].str.capitalize()
    data["Scale Factor"] = data["sf"]

    p = so.Plot(data, x="Query", y="Query Runtime in ms", color="Join Algorithm", alpha="Phase")
    p = p.add(so.Bar(), so.Agg(), so.Dodge(by=["color"]), so.Stack())
    p = p.facet(col="Scale Factor")
    p = p.scale(color="deep")
    p = p.label(
        x="TPC-H Query",
        alpha=str.capitalize,
        title="Scale Factor {}".format
    )
    p = p.share(y=False)
    #p = p.scale(stack=so.Nominal(order=["join", "selection", "rest"]))

    plot_filename = f"../img/{experiment_name}-SF{sf}-stacked.png"
    p.save(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)
    #plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)
    #plt.show()
    #plt.close()


def plot_total(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["measurement"] == "total")]
    data["Query"] = "Q" + data["query"].apply(str)

    sns.catplot(data, y="value", x="Query", hue="mode", col="scale_factor", row="threads", kind="bar", sharey="row")

    plot_filename = f"../img/{experiment_name}-total.pdf"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def plot_phases(filename_detail, experiment_name, threads):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == threads) & (data["measurement"] != "throughput")]
    data["Query"] = "Q" + data["query"].apply(str)

    sns.catplot(data, y="value", x="measurement", hue="mode", col="Query", row="scale_factor", kind="bar", sharey="row",
                order=TPCH_PHASE_ORDER)

    plot_filename = f"../img/{experiment_name}-phases-{threads}.pdf"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def plot_unroll_effect(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["measurement"] == "throughput")
                & (data["flags"].isin(["SPIN_LOCK CHUNKED_TABLE", "SPIN_LOCK UNROLL CHUNKED_TABLE"]))
                & (data["threads"] == 1)]
    data["unrolled"] = data["flags"].str.contains("UNROLL")

    print(data.to_string())

    sns.catplot(data, y="value", x="mode", hue="unrolled", row="scale_factor", kind="bar", sharey="row")
    plt.tight_layout()
    plt.show()


def main():
    experiment_name = "full-queries"

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
        ["tpch"],
        [["SPIN_LOCK", "CHUNKED_TABLE", "SIMD", "UNROLL", "FORCE_2_PHASES"]],
        [3],
        [10],
        ["RHO"],
        [16],
        10
    )

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    so.Plot.config.theme.update(axes_style("ticks"))

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run_simple_flags(config, filename_detail)
    if execution_command in ["plot", "both"]:
        plot_total(filename_detail, experiment_name)
        plot_phases(filename_detail, experiment_name, 1)
        plot_phases(filename_detail, experiment_name, 16)
        #plot_stacked(filename_detail, experiment_name, 1)
        #plot_stacked(filename_detail, experiment_name, 10)


if __name__ == '__main__':
    main()
