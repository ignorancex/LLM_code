#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.commons import delete_all_configurations
from helpers.runner import compile_and_run

phase_order = ["total", "partition", "partition_1", "partition_r", "partition_s", "partition_2", "join_total", "build", "probe"]


def paper_plot_radix_improvements(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["measurement"] == "throughput")]

    data["Throughput in $10^6$ rows/s"] = data["value"]
    data["Setting"] = data["mode"].replace({"sgx": "SGX Data in Enclave", "native": "Plain CPU"})
    data["Algorithm"] = data["alg"]

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette([sns.color_palette("deep")[0], sns.color_palette("deep")[2]])

    plt.figure(figsize=(6, 3.25))
    sns.barplot(data, y="Throughput in $10^6$ rows/s", hue="Setting", x="Algorithm", errorbar="sd")

    plt.ylim(bottom=0)
    plt.grid(axis="y")
    plt.tight_layout()

    plot_filename = f"../img/paper-radix-improvements.pdf"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1,  dpi=300)
    # plt.show()


def main():
    experiment_name = "radix-improvements"

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
    config["algorithms"] = ["RHO", "RHT", "RSM"]
    config["reps"] = 10
    config["modes"] = ["native", "sgx"]
    config["flags"] = [["UNROLL", "FORCE_2_PHASES"]]

    MB = 2 ** 20
    TUPLE_SIZE = 8
    TUPLE_PER_MB = MB/TUPLE_SIZE

    config["sizes"] = [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)]

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run(config, filename_detail)
    if execution_command in ["plot", "both"]:
        paper_plot_radix_improvements(filename_detail, experiment_name)


if __name__ == '__main__':
    main()
