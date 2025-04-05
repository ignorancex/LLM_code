#!/usr/bin/python3
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from helpers.commons import delete_all_configurations, TUPLE_PER_MB
from helpers.runner import compile_and_run_simple_flags, ExperimentConfig


sizes = [(1, 400),
         (100, 400),
         (4000, 4000)]
sizes = [(r * TUPLE_PER_MB, s * TUPLE_PER_MB) for r, s in sizes]
size_strings = ["1 MB/400 MB", "100 MB/400 MB", "4000 MB/4000 MB"]
size_dict = dict(zip(sizes, size_strings))


def paper_plot_scaling(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    pht_times = data[(data["measurement"].isin(["build", "probe"])) & (data["alg"] == "PHT")]

    data = data[(data["measurement"] == "throughput")]

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    deep = sns.color_palette("deep")

    settings = ["Plain CPU O", "Plain CPU O+M", "SGX DiE O"]
    palette = [deep[4], deep[5], deep[7]]
    hatches = ['||', '++', 'oo']
    skewness = ["No Skew", "Mild Skew", "Heavy Skew"]

    data["Throughput in\n$10^6$ rows/s"] = data["value"]
    data["Setting"] = (data["mode"] + data["mitigation"].astype(str)).replace({
        "nativeFalse": settings[0],
        "nativeTrue": settings[1],
        "sgxFalse": settings[2],
        "sgxTrue": settings[2],
    })
    data["Algorithm"] = data["alg"]
    data["Skewness"] = data["skew"].replace({-1.0: skewness[0], 0.5: skewness[1], 1.5: skewness[2]})

    #test = data[(data["Skewness"] == skewness[0]) & (data["Algorithm"] == "RHO")]
    #print(test.to_string())

    f = sns.catplot(data, y="Throughput in\n$10^6$ rows/s",
                hue="Setting", hue_order=settings, palette=palette,
                x="Algorithm",
                col="Skewness", col_order=skewness,
                errorbar="sd", kind="bar",
                sharey=True,
                height=2, aspect=0.7, zorder=2)
    f.set_titles(col_template="{col_name}")

    axes = f.axes.flatten()

    for ax_index, ax in enumerate(axes):
        for patch_index, bar in enumerate(ax.patches):
            if patch_index // 2 == 0:
                bar.set_hatch(hatches[0])
            elif patch_index // 2 == 1:
                bar.set_hatch(hatches[1])
            elif patch_index // 2 == 2:
                bar.set_hatch(hatches[2])

        # ax.grid(axis="y", linewidth=0.5)
        ax.xaxis.label.set_visible(False)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    for hatch, handle in zip(hatches, f._legend.legend_handles):
        handle.set_hatch(hatch)

    sns.move_legend(
        f, "lower center",
        bbox_to_anchor=(.5, 0.9), ncol=3, title=None, frameon=False
    )

    # axes[0].yaxis.set_label_coords(-0.5, 0.8)
    axes[1].yaxis.label.set_visible(False)
    axes[2].yaxis.label.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    plot_filename = f"../img/paper-{experiment_name}.pdf"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)

    pht_means = pht_times.groupby(["mitigation", "skew", "measurement", "mode"])["value"].mean()
    print(pht_means.to_string())

    # plt.show()


def main():
    experiment_name = "skew"

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
        [["FORCE_2_PHASES", "UNROLL"]],
        [(100 * TUPLE_PER_MB, 400 * TUPLE_PER_MB)],
        ["RHO", "PHT"],
        [16],
        [False],
        10,
        mitigation=[False, True],
        skew=[-1.0, 0.5, 1.5]  # Negative skew = no skew
    )

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run_simple_flags(config, filename_detail)
    if execution_command in ["plot", "both"]:
        paper_plot_scaling(filename_detail, experiment_name)


if __name__ == '__main__':
    main()
