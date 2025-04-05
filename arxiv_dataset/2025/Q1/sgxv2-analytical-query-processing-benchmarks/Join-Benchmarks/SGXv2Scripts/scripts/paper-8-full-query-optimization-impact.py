#!/usr/bin/python3
import inspect
import itertools
import sys

import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
import matplotlib.legend as legend
from seaborn import axes_style

from helpers.commons import delete_all_configurations
from helpers.tpch_runner import compile_and_run_simple_flags, ExperimentConfig


def plot_total(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)
    data = data[(data["measurement"] == "total")]
    data["Query"] = "Q" + data["query"].apply(str)
    data["Unrolled"] = data["flags"].str.contains("UNROLL")

    sns.catplot(data, y="value", x="Query", hue="mode", col="Unrolled", row="threads", kind="bar", sharey="row")

    plot_filename = f"../img/{experiment_name}-total.pdf"
    plt.savefig(plot_filename, transparent=False, dpi=150)


def move_legend_fig_to_ax(fig, ax, loc, bbox_to_anchor=None, **kwargs):
    if fig.legends:
        old_legend = fig.legends[-1]
    else:
        raise ValueError("Figure has no legend attached.")

    old_boxes = old_legend.get_children()[0].get_children()

    legend_kws = inspect.signature(legend.Legend).parameters
    props = {
        k: v for k, v in old_legend.properties().items() if k in legend_kws
    }

    props.pop("bbox_to_anchor")
    title = props.pop("title")
    if "title" in kwargs:
        title.set_text(kwargs.pop("title"))
    title_kwargs = {k: v for k, v in kwargs.items() if k.startswith("title_")}
    for key, val in title_kwargs.items():
        title.set(**{key[6:]: val})
        kwargs.pop(key)
    kwargs.setdefault("frameon", old_legend.legendPatch.get_visible())

    # Remove the old legend and create the new one
    props.update(kwargs)
    fig.legends = []
    new_legend = ax.legend(
        [], [], loc=loc, bbox_to_anchor=bbox_to_anchor, **props
    )
    new_legend.get_children()[0].get_children().extend(old_boxes)


def paper_plot_stacked(filename_detail, experiment_name):
    data = pd.read_csv(filename_detail, header=0)

    # phases_original = ["join1", "join2", "join3", "selection1", "selection2", "selection3", "copy"]
    # phases_plot = ["Join 1", "Join 2", "Join 3", "Selection 1", "Selection 2", "Selection 3", "Rest"]

    phases_original = ["join", "selection"]
    phases_plot = ["Join", "Selection"]

    setting_order = {"Plain CPU": 0, "Plain CPU O": 100, "SGX DiE": 200,
                     "SGX DiE O": 300}
    phase_order = {"Join": 0, "Selection": 10, "Rest": 20}
    query_order = {"Q3": 0, "Q10": 1, "Q12": 2, "Q19": 3}

    data = data[(data["measurement"].isin(phases_original))]
    # data = data.rename_axis("index1").reset_index()

    data["Query"] = "Q" + data["query"].apply(str)
    data["Runtime in ms"] = data["value"] / 1000
    data["Phase"] = data["measurement"].replace({orig: new for orig, new in zip(phases_original, phases_plot)})
    data["Unrolled"] = data["flags"].str.contains("UNROLL")
    data["Scale Factor"] = data["scale_factor"]
    data["Setting"] = (data["mode"] + data["Unrolled"].astype(str)).replace({
        "tpch-nativeFalse": "Plain CPU",
        "tpch-nativeTrue": "Plain CPU O",
        "tpchFalse": "SGX DiE",
        "tpchTrue": "SGX DiE O"
    })

    means = data.groupby(["mode", "Unrolled", "Scale Factor", "Query", "Phase"])["Runtime in ms"].mean()

    default_sgx = means["tpch", False]
    default_sgx_totals = default_sgx.groupby(level=[0, 1]).sum()
    optimized_sgx = means["tpch", True]
    optimized_sgx_totals = optimized_sgx.groupby(level=[0, 1]).sum()
    plain = means["tpch-native", False]
    plain_totals = plain.groupby(level=[0, 1]).sum()
    plain_opt = means["tpch-native", True]
    plain_opt_totals = plain_opt.groupby(level=[0, 1]).sum()

    sgx_improvement = 1 - (optimized_sgx_totals / default_sgx_totals)
    sgx_overhead = default_sgx_totals / plain_totals
    sgx_optimized_overhead = optimized_sgx_totals / plain_opt_totals

    print("Average SGX Improvement:", sgx_improvement.mean())
    print(sgx_improvement.to_string())
    print("Average SGX Overhead:", sgx_overhead.mean())
    print(sgx_overhead.to_string())
    print("Average SGX Optimized Overhead:", sgx_optimized_overhead.mean())
    print(sgx_optimized_overhead.to_string())
    print("Average overhead reduction:", (sgx_overhead - sgx_optimized_overhead).mean())
    print((sgx_overhead - sgx_optimized_overhead).to_string())

    # data = data[data["Setting"] != "tpch-nativeTrue"]
    data["sort_column"] = data["Phase"].map(phase_order) + data["Query"].map(query_order) + data["Setting"].map(
        setting_order)

    data = data.sort_values(by="sort_column")

    deep = sns.color_palette("deep")
    palette = [deep[0], deep[4], deep[2], deep[7]]
    hatches = ['', '||', '\\\\', 'oo']

    f, [ax0, ax1] = plt.subplots(ncols=2, figsize=(6, 3.5), squeeze=True)
    (so.Plot(data[data["Scale Factor"] == 10], x="Query", y="Runtime in ms", color="Setting", alpha="Phase")
     .add(so.Bar(), so.Agg(), so.Dodge(by=["color"]), so.Stack())
     .scale(
        color=palette,
        alpha=so.Nominal(order=phases_plot))
     .label(x="TPC-H Query")
     .layout(engine='tight')
     .share(y=True)
     .on(ax0)
     .plot())

    (so.Plot(data[data["Scale Factor"] == 100], x="Query", y="Runtime in ms", color="Setting", alpha="Phase")
     .add(so.Bar(), so.Agg(), so.Dodge(by=["color"]), so.Stack())
     .scale(
        color=palette,
        alpha=so.Nominal(order=phases_plot))
     .label(x="TPC-H Query")
     .layout(engine='tight')
     .share(y=True)
     .on(ax1)
     .plot())

    for l in f.legends:
        l.set_visible(False)

    l1 = plt.legend(ax0.patches[::2], setting_order.keys(), loc="lower center", bbox_to_anchor=(-0.25, 1.17), frameon=False, ncols=4, columnspacing=0.8)

    for hatch, color, handle in zip(hatches, palette, l1.legend_handles):
        handle.set_color(color)
        handle.set_linewidth(0)
        handle.set_edgecolor("white")
        handle.set_hatch(hatch)

    l2 = ax0.legend(ax0.patches[:2], ["Join", "Selection"], loc="lower center", bbox_to_anchor=(1.1, 1.07), frameon=False, ncols=2, columnspacing=0.8)

    l2.legend_handles[0].set_linewidth(1)
    l2.legend_handles[1].set_linewidth(1.7)

    ax0.grid(axis="y")
    ax1.grid(axis="y")

    ax0.set_title("Scale Factor 10")
    ax1.set_title("Scale Factor 100")

    ax0.set_ylim(bottom=0, top=140)
    ax1.set_ylim(bottom=0, top=1400)

    for ax in [ax0, ax1]:
        for i, (bar1, bar2) in enumerate(itertools.batched(ax.patches, 2)):
            bar1.set_hatch(hatches[i % 4])
            bar1.set_linewidth(0)
            bar1.set_edgecolor("white")
            bar2.set_hatch(hatches[i % 4])

    ax1.set_ylabel("")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.27)
    plot_filename = f"../img/paper-figure-full-query-optimization-10-100.pdf"
    plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)

    # plt.savefig(plot_filename, transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=150)
    # plt.show()
    # plt.close()


def main():
    experiment_name = "full-query-optimization-impact"

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
        ["tpch-native", "tpch"],
        [["FORCE_2_PHASES", "CHUNKED_TABLE", "SIMD"],
         ["FORCE_2_PHASES", "CHUNKED_TABLE", "SIMD", "UNROLL"]],
        [3, 10, 12, 19],
        [10, 100],
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
        # plot_total(filename_detail, experiment_name)
        paper_plot_stacked(filename_detail, experiment_name)


if __name__ == '__main__':
    main()
