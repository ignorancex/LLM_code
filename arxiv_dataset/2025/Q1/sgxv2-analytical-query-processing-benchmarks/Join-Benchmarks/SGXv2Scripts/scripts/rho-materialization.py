#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
from seaborn import axes_style

from helpers.commons import TUPLE_PER_MB, powerset, delete_all_configurations
from helpers.runner import compile_and_run_simple_flags, ExperimentConfig

phase_order = ["total", "partition", "partition_1", "partition_2", "join_total", "build", "probe"]


def plot(filename_detail, experiment_name, threads, size_mb):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["threads"] == threads) & (data["size_r"] == size_mb * TUPLE_PER_MB)]
    data["flags"] = data["flags"].fillna("NONE")

    sns.catplot(data, y="value", x="measurement", hue="mode", row="flags", col="materialize", kind="bar", sharey="row",
                order=phase_order)

    plot_filename = f"../img/{experiment_name}-{size_mb}-{threads}.pdf"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def main():
    experiment_name = "rho-materialization"

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
        powerset(["UNROLL", "SPIN_LOCK", "CHUNKED_TABLE"]),
        [(10 * TUPLE_PER_MB, 40 * TUPLE_PER_MB), (100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)],
        ["RHO"],
        [1, 16],
        [False, True],
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
        for size in [10, 100]:
            for threads in config.threads:
                plot(filename_detail, experiment_name, threads, size)


if __name__ == '__main__':
    main()
