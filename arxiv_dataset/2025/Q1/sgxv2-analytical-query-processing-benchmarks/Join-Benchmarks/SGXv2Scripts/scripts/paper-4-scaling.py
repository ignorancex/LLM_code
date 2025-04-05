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

    data = data[(data["measurement"] == "throughput")]

    settings = ["Plain CPU", "Plain CPU O", "Plain CPU O+M", "SGX DiE", "SGX DiE O"]

    data["Throughput in\n$10^6$ rows/s"] = data["value"]
    data["unroll"] = data["flags"].str.contains("UNROLL")
    data["Setting"] = (data["mode"] + data["unroll"].astype(str) + data["mitigation"].astype(str)).replace({
        "nativeFalseFalse": settings[0],
        "nativeTrueFalse": settings[1],
        "nativeTrueTrue": settings[2],
        "sgxFalseFalse": settings[3],
        "sgxTrueFalse": settings[4],
    })
    data["Algorithm"] = data["alg"]

    data["Size"] = data.loc[:, ["size_r", "size_s"]].astype(int).apply(lambda x: size_dict.get((x["size_r"], x["size_s"]), ""), axis=1)

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    deep = sns.color_palette("deep")
    palette = [deep[0], deep[4], deep[5], deep[2], deep[7]]
    hatches = ['', '||', '++', '\\\\', 'oo']

    f = sns.catplot(data, y="Throughput in\n$10^6$ rows/s",
                hue="Setting", hue_order=settings, palette=palette,
                x="Algorithm",
                col="Size", col_order=size_strings,
                errorbar="sd", kind="bar",
                sharey=False,
                height=2, aspect=0.7, zorder=2)
    f.set_titles(col_template="{col_name}")

    axes = f.axes.flatten()

    for ax_index, ax in enumerate(axes):
        for patch_index, bar in enumerate(ax.patches):
            if patch_index // 2 == 1:
                bar.set_hatch(hatches[1])
            elif patch_index // 2 == 2:
                bar.set_hatch(hatches[2])
            elif patch_index // 2 == 3:
                bar.set_hatch(hatches[3])
            elif patch_index // 2 == 4:
                bar.set_hatch(hatches[4])

        # ax.grid(axis="y", linewidth=0.5)
        upper_bound = 4000/(2**ax_index)
        ax.set_ylim(bottom=0, top=upper_bound)
        ax.set_yticks([0, upper_bound//4, upper_bound//2, upper_bound//4*3, upper_bound])
        ax.xaxis.label.set_visible(False)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        # Add horizontal lines
        best_algo = "PHT" if ax_index == 0 else "RHO"
        ax_algo_filter = (data["Size"] == size_dict[sizes[ax_index]]) & (data["Algorithm"] == best_algo)

        baseline_setting = settings[1] if ax_index <= 1 else settings[2]
        baseline_color = palette[1] if ax_index <= 1 else palette[2]
        baseline_value = data.loc[ax_algo_filter & (data["Setting"] == baseline_setting), "value"].mean()
        ax.axhline(baseline_value, linestyle="--", color=baseline_color, linewidth=1, zorder=1)

        ours = data.loc[ax_algo_filter & (data["Setting"] == settings[4]), "value"].mean()
        ours_color = palette[4]
        ax.axhline(ours, linestyle="--", color=ours_color, linewidth=1, zorder=1)

    for hatch, handle in zip(hatches, f._legend.legend_handles):
        handle.set_hatch(hatch)

    sns.move_legend(
        f, "lower center",
        bbox_to_anchor=(.55, 0.9), ncol=5, title=None, frameon=False, fontsize=8, columnspacing=0.8
    )

    # axes[0].yaxis.set_label_coords(-0.5, 0.8)
    axes[1].yaxis.label.set_visible(False)
    axes[2].yaxis.label.set_visible(False)

    means = (data
        .loc[data["Size"].isin(size_strings)]
        .groupby(["mode", "unroll", "mitigation", "Algorithm", "Size"])["Throughput in\n$10^6$ rows/s"]
        .mean())

    fastest = means.groupby(level=4).idxmax()

    sgx_normal = means["sgx", False, False]
    sgx_opt = means["sgx", True, False]
    native_normal = means["native", False, False]
    native_opt = means["native", True, False]
    native_opt_mit = means["native", True, True]

    unroll_improvement = (sgx_opt / sgx_normal) - 1
    relative_perf_normal = sgx_normal / native_normal
    relative_perf_opt = sgx_opt / native_opt
    relative_perf_mit = sgx_opt / native_opt_mit

    print("Absolute Throughput")
    print(means.to_string())
    print("Fastest Setting")
    print(fastest.to_string())
    print("Improvement due to unrolling")
    print(unroll_improvement.to_string())
    print("Relative performance before optimization")
    print(relative_perf_normal.to_string())
    print("Relative performance after optimization")
    print(relative_perf_opt.to_string())
    print("Performance relative to native with mitigation")
    print(relative_perf_mit.to_string())
    print("Relative performance PHT/RHO:")
    print((sgx_opt["PHT"] / sgx_opt["RHO"]).to_string())

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.3)

    plot_filename = f"../img/paper-{experiment_name}.pdf"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()


def build_slowdown(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    data = data[(data["alg"] == "PHT") & (data["measurement"] == "probe")]

    settings = ["Plain CPU", "Plain CPU O", "Plain CPU O+M", "SGX DiE", "SGX DiE O"]

    data["unroll"] = data["flags"].str.contains("UNROLL")
    data["Setting"] = (data["mode"] + data["unroll"].astype(str) + data["mitigation"].astype(str)).replace({
        "nativeFalseFalse": settings[0],
        "nativeTrueFalse": settings[1],
        "nativeTrueTrue": settings[2],
        "sgxFalseFalse": settings[3],
        "sgxTrueFalse": settings[4],
    })

    data = data[data["Setting"].isin(settings)]

    data["Runtime ms"] = data["value"] / 2900 / 1000

    data["Size"] = data.loc[:, ["size_r", "size_s"]].astype(int).apply(lambda x: size_dict.get((x["size_r"], x["size_s"]), ""), axis=1)

    means = data.groupby(["Size", "Setting"])["Runtime ms"].mean()

    print(means.to_string())

def main():
    experiment_name = "scaling-perf"

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
        [["FORCE_2_PHASES"], ["FORCE_2_PHASES", "UNROLL"]],
        sizes,
        ["RHO", "PHT"],
        [16],
        [False],
        10,
        mitigation=[False, True]
    )

    if execution_command in ["run", "both"]:
        delete_all_configurations()
        compile_and_run_simple_flags(config, filename_detail)
    if execution_command in ["plot", "both"]:
        paper_plot_scaling(filename_detail, experiment_name)
        build_slowdown(filename_detail, experiment_name)


if __name__ == '__main__':
    main()
