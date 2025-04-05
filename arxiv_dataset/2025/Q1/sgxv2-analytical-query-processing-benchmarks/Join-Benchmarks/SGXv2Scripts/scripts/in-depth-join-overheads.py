#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.runner import compile_and_run

phase_order = ["total", "partition", "partition_1", "partition_2", "join_total", "build", "probe"]


def plot(algorithms: list[str], filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data["size"] = (data['size_r'] / 2 ** 17).astype(str) + "_" + (data['size_s'] / 2 ** 17).astype(str)

    for alg in algorithms:
        alg_data = data[data["alg"] == alg]

        alg_data = alg_data.drop(columns=["lock", "alg", "size_r", "size_s"])

        sns.catplot(alg_data, y="value", x="measurement", hue="mode", row="size", col="threads", kind="bar",
                    sharey=False, order=phase_order)

        plot_filename = f"../img/{experiment_name}-{alg}.png"
        plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)


def main():
    experiment_name = "in-depth-join-overheads"

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

    filename_avg = f"../data/{experiment_name}-avg.csv"
    filename_detail = f"../data/{experiment_name}.csv"

    config = dict()
    config["experiment"] = True
    config["threads"] = [1, 16]
    config["algorithms"] = ["PHT", "CHT", "RHO", "RSM", "RHT"]
    config["reps"] = 1
    config["modes"] = ["native", "sgx"]
    config["sizes"] = [#(2 ** 10, 2 ** 10),
                       #(2 ** 10, 2 ** 20),
                       (2 ** 20, 2 ** 20),
                       (10 * 2 ** 17, 40 * 2 ** 17),  # 10 and 40 MiB of 8 Byte tuples
                       #(100 * 2 ** 17, 400 * 2 ** 17),  # 200 and 400 MiB of 8 Byte tuples
                       #(2 ** 26, 2 ** 28), # 512 MiB and 2 GiB of 8 Byte tuples
                       ]

    if execution_command in ["run", "both"]:
        compile_and_run(config, filename_avg, filename_detail)
    if execution_command in ["plot", "both"]:
        plot(config["algorithms"], filename_detail, experiment_name)


if __name__ == '__main__':
    main()
