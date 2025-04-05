#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
from seaborn import axes_style

from helpers.commons import TUPLE_PER_MB, delete_all_configurations
from helpers.runner import compile_and_run_simple_flags, ExperimentConfig


def paper_plot_throughput(filename_detail):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput")]

    settings = ["Plain CPU\nLock-Free", "Plain CPU\nMutex", "SGX\nLock-Free", "SGX\nMutex"]

    data["Throughput in\n$10^6$ rows/s"] = data["value"]
    data["Setting"] = (data["mode"] + data["flags"].str.contains("MUTEX_QUEUE").astype(str)).replace({
        "nativeFalse": settings[0],
        "nativeTrue": settings[1],
        "sgxFalse": settings[2],
        "sgxTrue": settings[3],
    })

    color_1 = sns.color_palette("deep")[0]
    color_2 = sns.color_palette("deep")[2]

    f = plt.figure(figsize=(6, 2.5))
    sns.barplot(data, y="Throughput in\n$10^6$ rows/s", x="Setting", hue="Setting", order=settings,
                palette=[color_1, color_2] * 2, errorbar="sd")

    plt.xlabel(None)

    for i, bar in enumerate(f.axes[0].patches):
        if i % 2 == 1:
            bar.set_hatch('\\\\')

    plt.ylim(bottom=0)
    plt.grid(axis="y")
    plt.tight_layout()

    plt.tight_layout()
    plot_filename = f"../img/paper-figure-mutex-contention.pdf"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()


def main():
    experiment_name = "mutex-contention"

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
        [["UNROLL", "CONSTANT_RADIX_BITS"], ["UNROLL", "CONSTANT_RADIX_BITS", "MUTEX_QUEUE"]],
        [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)],
        ["RHO"],
        [16],
        [False],
        10
    )

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run_simple_flags(config, filename_detail)
    if execution_command in ["plot", "both"]:
        sns.set_style("ticks")
        sns.set_context("notebook")
        sns.set_palette("deep")

        so.Plot.config.theme.update(axes_style("ticks"))
        paper_plot_throughput(filename_detail)


if __name__ == '__main__':
    main()
