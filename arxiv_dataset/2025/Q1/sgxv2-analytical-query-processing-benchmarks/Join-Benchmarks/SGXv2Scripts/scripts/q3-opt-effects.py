#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
from seaborn import axes_style

from helpers.commons import delete_all_configurations, powerset
from helpers.tpch_runner import compile_and_run_simple_flags, ExperimentConfig, TPCH_PHASE_ORDER


def plot(filename_detail, experiment_name, threads):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == threads)
                & (data["flags"].isin(["SPIN_LOCK CHUNKED_TABLE", "SPIN_LOCK UNROLL CHUNKED_TABLE"]))]

    sns.catplot(data, y="value", x="measurement", hue="mode", col="flags", row="scale_factor", kind="bar", sharey="row",
                order=TPCH_PHASE_ORDER)

    plot_filename = f"../img/{experiment_name}-{threads}.pdf"
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
    experiment_name = "q3-opt-effects"

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
        ["tpch-native", "tpch"],
        powerset(["SPIN_LOCK", "UNROLL", "CHUNKED_TABLE"]),
        [3],
        [1, 10],
        ["RHO"],
        [1, 16],
        3
    )

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run_simple_flags(config, filename_detail)
    if execution_command in ["plot", "both"]:
        sns.set_style("ticks")
        sns.set_context("notebook")
        sns.set_palette("deep")

        so.Plot.config.theme.update(axes_style("ticks"))
        plot(filename_detail, experiment_name, 1)
        plot(filename_detail, experiment_name, 16)
        plot_unroll_effect(filename_detail, experiment_name)


if __name__ == '__main__':
    main()
