#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
from seaborn import axes_style

from helpers.commons import TUPLE_PER_MB, delete_all_configurations
from helpers.runner import compile_and_run_simple_flags, ExperimentConfig


def plot(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[data["measurement"] == "throughput"]
    data["Enclave Node"] = data["init_core"].replace({16: 1})
    data["Number of Join Threads"] = data["threads"]
    data["Throughput in 10^6 rows/s"] = data["value"]

    sns.barplot(data, y="Throughput in 10^6 rows/s", x="Number of Join Threads", hue="Enclave Node", palette="deep")

    plot_filename = f"../img/{experiment_name}.png"
    plt.savefig(plot_filename, transparent=False, dpi=300)


def plot_phases(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["measurement"] != "throughput")]
    data["Setup"] = data["mode"] + data["init_core"].replace({16: 1}).astype(str) + data["threads"].astype(str)
    data["Setup"] = data["Setup"].replace({
        "native016": "Plain CPU Local",
        "native116": "Plain CPU Fully Remote",
        "native032": "Plain CPU Half Local",
        "native132": "Plain CPU Half Local (r)",
        "sgx016": "SGX Local",
        "sgx116": "SGX Fully Remote",
        "sgx032": "SGX Half Local",
        "sgx132": "SGX Half Local (r)"
    })

    sns.barplot(data, y="value", x="measurement", hue="Setup",
                order=["total", "partition_r", "partition_s", "partition_2_h", "partition_2_c", "build", "probe"])

    plt.tight_layout()
    plot_filename = f"../img/{experiment_name}-phases.png"
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=150)
    # plt.show()
    plt.close()


def carsten_plot(filename_detail):
    data = pd.read_csv(filename_detail, header=0)

    # NAMES = ["SGX Join\nSingle Node", "SGX Join\nFully Remote", "SGX Join\nHalf Local", "Native Join\nNUMA Local"]
    NAMES = ["Plain CPU Join\nSingle Socket", "SGX Join\nSingle Socket", "Plain CPU Join\nDual Socket",
             "SGX Join\nDual Socket"]

    data = data[(data["measurement"] == "throughput") & (data["size_s"] == 52428800.0)]
    data["Setup"] = data["mode"] + data["init_core"].replace({16: 1}).astype(str) + data["threads"].astype(str)
    data["Setup"] = data["Setup"].replace({
        "native016": NAMES[0],
        "native116": "Plain CPU Fully Remote",
        "native032": "Plain CPU Half Local",
        "native132": "Plain CPU Half Local (r)",
        "sgx016": NAMES[1],
        "sgx116": "",
        "sgx032": NAMES[3],
        "sgx132": "SGX Half Local (r)"
    })

    hypothetical_native = data[data["Setup"] == NAMES[0]].copy()
    hypothetical_native["value"] *= 2
    hypothetical_native["Setup"] = NAMES[2]

    data = pd.concat([data, hypothetical_native]).reset_index(drop=True)
    data = data[data["Setup"].isin(NAMES)]
    data["Throughput in\n$10^6$ rows/s"] = data["value"]

    print(data[["Setup", "value"]].to_string())

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[2]] * 2
    f = plt.figure(figsize=(6, 2.5))
    sns.barplot(data, x="Setup", hue="Setup", palette=colors, y="Throughput in\n$10^6$ rows/s", order=NAMES,
                hue_order=NAMES)

    f.axes[0].axhline(data.loc[data["Setup"] == NAMES[1], "value"].mean(), linestyle="--", color="grey", linewidth=1)
    f.axes[0].axvline(1.5, linestyle="--", color="black", linewidth=1)

    hatches = ['', '\\\\']

    for i, bar in enumerate(f.axes[0].patches):
        if i % 2 == 0:
            bar.set_hatch(hatches[0])
        elif i % 2 == 1:
            bar.set_hatch(hatches[1])

    plt.ylim(0, 3500)
    plt.xticks(ticks=[0, 1, 2, 3], labels=["Plain CPU", "SGX DiE"] * 2)

    plt.xlabel(None)
    plt.grid(axis="y")
    plt.tight_layout()
    plot_filename = f"../img/paper-figure-joins-numa.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)


def paper_plot(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    NAMES = ["Local\n16 Threads", "Fully Remote\n16 Threads", "Half Local\n32 Threads", "NUMA Local\n32 Threads"]

    data = data[(data["measurement"] == "throughput")]
    data["Setup"] = data["init_core"].replace({16: 1}).astype(str) + data["threads"].astype(str)
    data["Setup"] = data["Setup"].replace({
        "016": NAMES[0],
        "116": NAMES[1],
        "032": NAMES[2],
        "132": NAMES[2],
    })

    hypothetical_local = data[data["Setup"] == NAMES[0]].copy()
    hypothetical_local["value"] *= 2
    hypothetical_local["Setup"] = NAMES[3]

    data = pd.concat([data, hypothetical_local]).reset_index(drop=True)
    data["Throughput in\n$10^6$ rows/s"] = data["value"]

    print(data[["Setup", "mode", "value"]].to_string())

    plt.figure(figsize=(6, 3.25))
    sns.barplot(data, x="Setup", y="Throughput in\n$10^6$ rows/s", hue="mode",
                order=NAMES)
    plt.xlabel(None)
    plt.tight_layout()
    plot_filename = f"../img/{experiment_name}-throughput.png"
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    #plt.show()


def main():
    experiment_name = "rho-numa"

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
        [["UNROLL", "FORCE_2_PHASES"]],
        [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)],
        ["RHO"],
        [16, 32],
        [False],
        10,
        [0, 16]
    )

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run_simple_flags(config, filename_detail)
    if execution_command in ["plot", "both"]:
        sns.set_style("ticks")
        sns.set_context("notebook")
        sns.set_palette("deep")

        so.Plot.config.theme.update(axes_style("ticks"))
        carsten_plot(filename_detail)


if __name__ == '__main__':
    main()
