import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from seaborn import color_palette


def plot_both(file, measurement):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)

    data["setting"] = "sgx: " + data["sgx"] + " preload: " + data["preload"]

    plot = sns.relplot(data, x="num_bins", y=measurement, hue="setting",
                       markers=True, col="workload", kind="line", facet_kws={'sharey': True, 'sharex': True})

    plt.xscale('log')
    plt.xlim([1, 2 ** 20])
    plt.ylim(bottom=0)
    plot.axes[0, 0].set_xticks([2 ** x for x in range(0, 20)],
                               [str(2 ** x) + unit for unit in ["", "ki"] for x in range(0, 10)],
                               rotation=90)
    plot.axes[0, 1].set_xticks([2 ** x for x in range(0, 20)],
                               [str(2 ** x) + unit for unit in ["", "ki"] for x in range(0, 10)],
                               rotation=90)
    # ax2 = plot.axes[0, 0].twinx()
    # ax3 = plot.axes[0, 1].twinx()
    # sns.lineplot(data[data["workload"] == "hist"], x="num_keys", y="L1D-misses", hue="sgx", markers=True,
    #             palette=color_palette()[2:], ax=ax2)
    # sns.lineplot(data[data["workload"] == "copy"], x="num_keys", y="L1D-misses", hue="sgx", markers=True,
    #             palette=color_palette()[2:], ax=ax3)
    plt.savefig(f"img/{file}-{measurement}.png", dpi=150)
    plt.close()


def filter_workload(df: DataFrame, workload: str) -> DataFrame:
    return df[df["workload"] == workload]


def unroll_comparison():
    not_unrolled = pd.read_csv(f"data/results.csv", header=0, skipinitialspace=True)
    not_unrolled = filter_workload(not_unrolled, "copy")
    unrolled_4 = pd.read_csv(f"data/unroll-copy-4.csv", header=0, skipinitialspace=True)
    unrolled_4 = filter_workload(unrolled_4, "copy")
    unrolled_8 = pd.read_csv(f"data/unroll-copy-8.csv", header=0, skipinitialspace=True)
    unrolled_8 = filter_workload(unrolled_8, "copy")

    not_unrolled["unroll"] = 0
    unrolled_4["unroll"] = 4
    unrolled_8["unroll"] = 8

    data = pd.concat([not_unrolled, unrolled_4, unrolled_8])

    plot = sns.relplot(data, x="num_keys", y="timeMicroSec", hue="unroll", markers=True, col="sgx", kind="line",
                       facet_kws={'sharey': True, 'sharex': True}, palette=color_palette())

    plt.xscale('log')
    plt.xlim([1, 2 ** 20])
    plt.ylim(bottom=0)
    plot.axes[0, 0].set_xticks([2 ** x for x in range(0, 20)],
                               [str(2 ** x) + unit for unit in ["", "ki"] for x in range(0, 10)],
                               rotation=90)
    plot.axes[0, 1].set_xticks([2 ** x for x in range(0, 20)],
                               [str(2 ** x) + unit for unit in ["", "ki"] for x in range(0, 10)],
                               rotation=90)

    plot.tight_layout()

    plt.savefig(f"img/copy-unroll-comp.png", dpi=150)


def paper_figure_histogram(file):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)
    data["Setting"] = data["sgx"] + data["preload"]
    data["Setting"].replace({2: "SGX Data in Enclave", 1: "SGX Data outside Enclave", 0: "Plain CPU"},
                            inplace=True)
    data = data[(data["workload"] == "hist")
                & (data["mode"] == "loop")
                & data["unroll_factor"].isin([1, 9])
                & ~data["ssb_mitigation"]]

    data["Optimization"] = data["unroll_factor"].replace({
        1: "No Optimization",
        9: "Unrolling and Reordering"
    })
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Histogram Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("notebook")

    plt.figure(figsize=(6, 4))
    plot = sns.lineplot(data, x="Number of Histogram Bins", y="Runtime in ms", hue="Setting", style="Optimization",
                        hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"],
                        markers=True, errorbar="sd", palette=sns.color_palette("deep")[:3], legend="auto")

    upper_bound = 14
    # plt.legend()

    plot.axvline(x=2 ** 12 * 3, ymin=0, ymax=1, linestyle="--", color="grey")
    plt.text(2 ** 12 * 3, 1, "L1", rotation=90, ha='right', va='bottom', color="grey")

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0, top=215)
    plt.grid(axis="y")

    plot.set_xticks([2 ** x for x in range(0, upper_bound + 1)],
                    [str(2 ** x) + unit for unit in ["", "ki"] for x in range(0, 10)][:upper_bound + 1],
                    rotation=90)

    sns.move_legend(plot, "center right", bbox_to_anchor=(1, 0.65))

    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig(f"img/paper-figure-histogram.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_histogram_facet(file):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)
    data["Setting"] = data["sgx"].astype(int) * 2 + data["ssb_mitigation"]
    data["Setting"].replace({3: "SGX DiE", 2: "SGX DiE", 1: "Plain CPU M", 0: "Plain CPU"},
                            inplace=True)
    data = data[(data["workload"] == "hist")
                & (data["mode"] == "loop")
                & data["unroll_factor"].isin([1, 9])
                & ~(data["sgx"].astype(bool) & ~data["preload"].astype(bool))]

    data["Optimization"] = data["unroll_factor"].replace({
        1: "No Optimization",
        9: "Unroll + Reorder"
    })
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Histogram Bins"] = data["num_bins"]

    means = data[(data["Optimization"] == "Unroll + Reorder") & (data["Setting"].isin(["Plain CPU", "SGX DiE"]))].groupby(["Setting", "num_bins"])["Runtime in ms"].mean()

    native = means["Plain CPU"]
    sgx = means["SGX DiE"]

    relative = sgx / native

    print(relative.to_string())

    sns.set_style("ticks")
    sns.set_context("notebook")

    palette = [sns.color_palette("deep")[0], sns.color_palette("deep")[3], sns.color_palette("deep")[2]]

    order = ["Plain CPU", "Plain CPU M", "SGX DiE"]
    plot = sns.relplot(data, x="Number of Histogram Bins", y="Runtime in ms", hue="Setting", style="Setting",
                       col="Optimization", markers=True, hue_order=order, style_order=order, errorbar="sd",
                       palette=palette, kind="line", height=1.8, aspect=1.3)

    upper_bound = 14
    # plt.legend()

    for ax in plot.axes.flat:
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlim(left=1, right=2 ** upper_bound)
        ax.set_ylim(bottom=0, top=215)
        ax.set_xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
                      [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4])
        ax.set_yticks([0, 100,  200])
        ax.grid(axis="y")
        ax.axvline(x=2 ** 12 * 3, ymin=0, ymax=1, linestyle="--", color="grey")
        ax.text(2 ** 12 * 3, 1, "L1", rotation=90, ha='right', va='bottom', color="grey")
        ax.set_frame_on(True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    sns.move_legend(plot, "lower center", bbox_to_anchor=(0.5, 0.9), frameon=False, title=None, ncols=3, borderpad=0)

    plot.set_titles(col_template="{col_name}")

    plt.tight_layout()
    plt.savefig(f"img/paper-figure-histogram-facet-mit.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_partitioning(file):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)
    data["Setting"] = data["sgx"] + data["preload"]
    data["Setting"].replace({"yesyes": "SGX Data in Enclave", "yesno": "SGX Data outside Enclave", "nono": "Plain CPU"},
                            inplace=True)
    data["Unrolled"] = data["unroll"].replace({
        "no": 1,
        "yes": 4
    })
    data = data[data["workload"] == "copy"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Partitions"] = data["num_bins"]

    plot = sns.lineplot(data, x="Number of Partitions", y="Runtime in ms", hue="Setting", style="Unrolled",
                        hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"],
                        markers=True)

    upper_bound = 12

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0)

    plot.set_xticks([2 ** x for x in range(0, upper_bound + 1)],
                    [str(2 ** x) + unit for unit in ["", "ki"] for x in range(0, 10)][:upper_bound + 1],
                    rotation=90)

    # sns.move_legend(plot, "center right", bbox_to_anchor=(1, 0.65))

    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig(f"img/paper-figure-partition-copy-dis.png", dpi=300)
    plt.close()


def paper_figure_unrolling_factors():
    data = pd.read_csv(f"data/paper-histogram-data.csv", header=0, skipinitialspace=True)
    data["Setting"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                               2: "SGX Data in Enclave"})
    data = data[(data["mode"] == "loop")
                & (data["sgx"] + data["preload"]).isin([0, 2])
                & data["unroll_factor"].isin([1, 2, 4, 8, 9, 10, 16])
                & (data["num_bins"] == 32)
                & ~data["ssb_mitigation"]]
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000

    sns.set_style("ticks")
    sns.set_context("notebook")

    plt.figure(figsize=(6, 2.5))
    plot = sns.barplot(data, x="Unroll Factor", y="Runtime in ms", errorbar="sd", hue="Setting",
                       palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[2]],
                       legend="auto")

    # plt.legend()

    # sns.move_legend(plot, "center left", bbox_to_anchor=(1, 0.5))

    plt.grid(axis="y")
    plt.ylim(bottom=0, top=215)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"img/paper-figure-unrolling-improvement.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_avx_unrolling_factors():
    data = pd.read_csv(f"data/paper-histogram-data.csv", header=0, skipinitialspace=True)
    data["Setting"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                               2: "SGX Data in Enclave"})

    data = data[(data["mode"] == "simd_loop_unrolled")
                & (data["sgx"] + data["preload"]).isin([0, 2])
                & data["unroll_factor"].isin([1, 2, 3, 4, 5, 6])
                & (data["num_bins"] == 32)
                & ~data["ssb_mitigation"]]
    data["Unroll Factor"] = data["unroll_factor"] * 8
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000

    sns.set_style("ticks")
    sns.set_context("notebook")

    plt.figure(figsize=(6, 2.5))
    plot = sns.barplot(data, x="Unroll Factor", y="Runtime in ms", errorbar="sd", hue="Setting",
                       palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[2]],
                       legend="auto")

    # plt.legend()

    # sns.move_legend(plot, "center left", bbox_to_anchor=(1, 0.5))

    plt.grid(axis="y")
    plt.ylim(bottom=0, top=215)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"img/paper-figure-unrolling-avx-improvement.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_unrolling_factors_facet():
    data = pd.read_csv(f"data/paper-histogram-data.csv", header=0, skipinitialspace=True)
    data["Setting"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                               2: "SGX DiE"})

    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000

    data_scalar = data[(data["mode"] == "loop")
                       & (data["sgx"] + data["preload"]).isin([0, 2])
                       & data["unroll_factor"].isin([1, 2, 4, 8, 9, 10, 16])
                       & (data["num_bins"] == 32)
                       & ~data["ssb_mitigation"]]
    data_scalar["Unroll Factor"] = data_scalar["unroll_factor"]

    data_avx = data[(data["mode"] == "simd_loop_unrolled")
                    & (data["sgx"] + data["preload"]).isin([0, 2])
                    & data["unroll_factor"].isin([1, 2, 3, 4, 5, 6])
                    & (data["num_bins"] == 32)
                    & ~data["ssb_mitigation"]]
    data_avx["Unroll Factor"] = data_avx["unroll_factor"] * 8

    means = data_avx.groupby(["Setting", "Unroll Factor"])["Runtime in ms"].mean()

    sgx_overhead = means["SGX DiE"] / means["Plain CPU"] - 1
    print(sgx_overhead.to_string())

    sns.set_style("ticks")
    sns.set_context("notebook")

    f, axes = plt.subplots(ncols=2, figsize=(6, 2.25), tight_layout=True, width_ratios=(7, 6))

    sns.barplot(data_scalar, x="Unroll Factor", y="Runtime in ms", errorbar="sd", hue="Setting",
                palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[2]],
                legend="auto", ax=axes[0])

    sns.barplot(data_avx, x="Unroll Factor", y="Runtime in ms", errorbar="sd", hue="Setting",
                palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[2]],
                legend=False, ax=axes[1])

    sns.move_legend(axes[0], "lower center", bbox_to_anchor=(0.85, 1.2), frameon=False, title=None, ncols=3, borderpad=0)

    hatches = ['', '\\\\']

    num_patches = [7, 6]

    for ax, np in zip(axes, num_patches):
        ax.grid(axis="y")
        ax.set_ylim(bottom=0, top=215)

        for i, bar in enumerate(ax.patches):
            if i // np == 1:
                bar.set_hatch(hatches[1])

    for hatch, handle in zip(hatches, axes[0].get_legend().legend_handles):
        handle.set_hatch(hatch)

    axes[0].set_title("Scalar")
    axes[0].set_yticks([0, 100, 200])
    axes[1].set_title("AVX")
    axes[1].set_ylabel("")
    axes[1].set_yticks([0, 100, 200], ["", "", ""])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(f"img/paper-figure-unrolling-improvement-combined.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_simplest():
    data = pd.read_csv(f"data/simple-test-05-09.csv", header=0, skipinitialspace=True)
    data = data[(data["mode"] == "loop_const") & ((data["unroll_factor"] == 1) | (data["unroll_factor"] == 12))]
    data["SGX"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                           2: "SGX Data in Enclave"})
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("notebook")

    plt.figure(figsize=(6, 4))
    plot = sns.lineplot(data, x="Number of Bins", y="Runtime in ms", hue="SGX", style="Unroll Factor",
                        hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"], markers=True,
                        errorbar="sd", palette=sns.color_palette("deep")[:3], legend="auto")

    sns.move_legend(plot, "center left", bbox_to_anchor=(0, 0.55))

    upper_bound = 14

    plot.axvline(x=2 ** 12 * 3, ymin=0, ymax=1, linestyle="--", color="grey")
    plt.text(2 ** 12 * 3, 1, "L1", rotation=90, ha='right', va='bottom', color="grey")

    plt.grid(axis="y")
    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0)

    plt.xticks([2 ** x for x in range(0, upper_bound + 1)],
               [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1],
               rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    # plt.show()
    plt.savefig(f"img/paper-figure-simplified-algorithm.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_simplest_facet(file):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)
    data["Setting"] = data["sgx"] + data["preload"]
    data["Setting"].replace({2: "SGX DiE", 1: "SGX DoE", 0: "Plain CPU"},
                            inplace=True)
    data = data[(data["workload"] == "simple")
                & (data["mode"] == "loop_const")
                & data["unroll_factor"].isin([1, 9])
                & ~data["ssb_mitigation"]]

    data["Optimization"] = data["unroll_factor"].replace({
        1: "No Optimization",
        9: "Unroll + Reorder"
    })
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Write Positions"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("notebook")

    plot = sns.relplot(data, x="Number of Write Positions", y="Runtime in ms", hue="Setting", col="Optimization",
                       hue_order=["Plain CPU", "SGX DoE", "SGX DiE"],
                       markers=True, errorbar="sd", palette=sns.color_palette("deep")[:3], kind="line",
                       height=4, aspect=0.6)

    upper_bound = 14
    # plt.legend()

    for ax in plot.axes.flat:
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlim(left=1, right=2 ** upper_bound)
        ax.set_ylim(bottom=0, top=215)
        ax.set_xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
                      [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4])
        ax.grid(axis="y")
        ax.axvline(x=2 ** 12 * 3, ymin=0, ymax=1, linestyle="--", color="grey")
        ax.text(2 ** 12 * 3, 1, "L1", rotation=90, ha='right', va='bottom', color="grey")

    sns.move_legend(plot, "center right", bbox_to_anchor=(0.95, 0.75), frameon=True)

    plot.set_titles(col_template="{col_name}")

    plt.tight_layout()
    plt.savefig(f"img/paper-figure-simplified-algorithm-facet.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_mitigation_comparison(file):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)
    data = data[(data["workload"] == "hist")
                & (data["mode"] == "loop")
                & (data["unroll_factor"] == 1)
                & (data["sgx"] == data["preload"])]

    data["Setting"] = data["sgx"] + data["preload"] + data["ssb_mitigation"]
    data = data[data["Setting"].isin([0, 1, 2])]
    data["Setting"].replace({2: "SGX Data in Enclave", 1: "Plain CPU mitigation", 0: "Plain CPU No Mitigation"},
                            inplace=True)

    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Histogram Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("notebook")

    plt.figure(figsize=(6, 3.25))
    plot = sns.lineplot(data, x="Number of Histogram Bins", y="Runtime in ms", hue="Setting", style="Setting",
                        hue_order=["Plain CPU No Mitigation", "SGX Data in Enclave", "Plain CPU mitigation"],
                        style_order=["Plain CPU No Mitigation", "SGX Data in Enclave", "Plain CPU mitigation"],
                        markers=True, errorbar="sd",
                        palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[2],
                                 sns.color_palette("deep")[3]], legend="auto")

    upper_bound = 14
    # plt.legend()

    plot.axvline(x=2 ** 12 * 3, ymin=0, ymax=1, linestyle="--", color="grey")
    plt.text(2 ** 12 * 3, 1, "L1", rotation=90, ha='right', va='bottom', color="grey")

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0, top=215)
    plt.grid(axis="y")

    plot.set_xticks([2 ** x for x in range(0, upper_bound + 1)],
                    [str(2 ** x) + unit for unit in ["", "ki"] for x in range(0, 10)][:upper_bound + 1],
                    rotation=90)

    sns.move_legend(plot, "center right", bbox_to_anchor=(1, 0.65))

    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig(f"img/paper-mitigation-comparison.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def figure_histogram_scalar(file):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)
    data["Setting"] = data["sgx"] + data["preload"]
    data["Setting"].replace({2: "SGX Data in Enclave", 1: "SGX Data outside Enclave", 0: "Plain CPU"},
                            inplace=True)
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Output Data Size"] = data["num_bins"] * 4

    sns.set_style("ticks")
    sns.set_context("notebook")

    plt.figure(figsize=(6, 4))
    plot = sns.lineplot(data, x="Output Data Size", y="Runtime in ms", hue="Setting", style="Setting",
                        hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"],
                        markers=True, errorbar="sd", palette=sns.color_palette("deep"), legend="auto")

    upper_bound = 22
    lower_bound = 2
    # plt.legend()

    plt.xscale('log')
    plt.xlim([2 ** lower_bound, 2 ** upper_bound])
    plt.ylim(bottom=0)

    plot.set_xticks([2 ** x for x in range(lower_bound, upper_bound + 1)],
                    [str(2 ** x) + unit for unit in ["B", "kiB", "MiB"] for x in range(0, 10)][
                    lower_bound:upper_bound + 1],
                    rotation=90)

    # sns.move_legend(plot, "center right", bbox_to_anchor=(1, 0.65))

    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig(f"img/histogram-scalar.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def detailed_comparison():
    data = pd.read_csv(f"data/pointer.csv", header=0, skipinitialspace=True)
    data["SGX"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                           2: "SGX Data in Enclave"})
    data["Mode"] = data["mode"].replace(to_replace="simd_loop_unrolled", value="simd_unrolled")
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("paper")

    # plt.figure(figsize=(6, 3.75))
    plot = sns.relplot(data, x="Number of Bins", y="Runtime in ms", hue="SGX", col="Unroll Factor", row="Mode",
                       hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"],
                       errorbar="sd", palette=sns.color_palette("deep"), legend="auto", kind="line")

    upper_bound = 25
    # plt.legend()

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0)

    for ax in plot.axes.flat:
        ax.set_xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
                      [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4],
                      rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    plt.show()
    # plt.savefig(f"img/paper-figure-histogram.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def detailed_comparison_one_var():
    data = pd.read_csv(f"data/opt-dis-hist.csv", header=0, skipinitialspace=True)
    data = data[data["mode"] == "loop"]
    data["SGX"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                           2: "SGX Data in Enclave"})
    data["Mode"] = data["mode"].replace(to_replace="simd_loop_unrolled", value="simd_unrolled")
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("notebook")

    plot = sns.relplot(data, x="Number of Bins", y="Runtime in ms", hue="SGX", col="Unroll Factor", col_wrap=4,
                       hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"],
                       errorbar="sd", palette=sns.color_palette("deep"), legend="auto", kind="line")

    sns.move_legend(plot, "lower right", bbox_to_anchor=(1, 0))

    upper_bound = 25
    # plt.legend()

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0)

    for ax in plot.axes.flat:
        ax.set_xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
                      [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4],
                      rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    plt.show()
    # plt.savefig(f"img/paper-figure-histogram.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def detailed_comparison_one_var_old():
    data = pd.read_csv(f"data/simd-256.csv", header=0, skipinitialspace=True)
    data = data[data["unroll_factor"].isin([1, 4, 5, 6])]
    data["SGX"] = data["sgx"].replace({0: "Plain CPU", 1: "SGX Data in Enclave"})
    data["Mode"] = data["mode"].replace(to_replace="simd_loop_unrolled", value="simd_unrolled")
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("paper")

    plot = sns.relplot(data, x="Number of Bins", y="Runtime in ms", hue="SGX", col="Unroll Factor",
                       hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"],
                       errorbar="sd", palette=sns.color_palette("deep"), legend="auto", kind="line")

    upper_bound = 20
    # plt.legend()

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0)

    for ax in plot.axes.flat:
        ax.set_xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
                      [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4],
                      rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    plt.show()
    # plt.savefig(f"img/paper-figure-histogram.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def sgx_comparison_one_var():
    data = pd.read_csv(f"data/histogram-unrolled-1-16.csv", header=0, skipinitialspace=True)
    data = data[(data["mode"] == "unrolled") & (data["sgx"] == 1) & data["unroll_factor"].isin([1, 2, 4, 8, 9, 10, 16])]
    data["Mode"] = data["mode"].replace(to_replace="simd_loop_unrolled", value="simd_unrolled")
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("talk")

    plt.figure(figsize=(6, 3.25))
    plot = sns.lineplot(data, x="Number of Bins", y="Runtime in ms", hue="Unroll Factor", style="Unroll Factor",
                        errorbar="sd", palette=sns.color_palette("deep"), markers=True, legend="auto")

    upper_bound = 20
    # plt.legend()

    sns.move_legend(plot, "center left", bbox_to_anchor=(1, 0.5))

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0, top=500)

    plt.xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
               [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4],
               rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    # plt.show()
    plt.savefig(f"img/paper-figure-unrolling-improvement.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def pragma_comp():
    data = pd.read_csv(f"data/pragma-comparison.csv", header=0, skipinitialspace=True)
    data["SGX"] = data["sgx"].replace({0: "Plain CPU", 1: "SGX Data in Enclave"})
    data["Unrolling"] = data["unroll_factor"].replace({1: "Scalar", 8: "Pragma"})
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("talk")

    plot = sns.relplot(data, x="Number of Bins", y="Runtime in ms", hue="SGX", col="Unrolling",
                       hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"],
                       errorbar="sd", palette=sns.color_palette("deep"), kind="line",
                       facet_kws=dict(legend_out=True))

    sns.move_legend(plot, "center left", bbox_to_anchor=(1, 0.5))

    upper_bound = 20
    # plt.legend()

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0)

    for ax in plot.axes.flat:
        ax.set_xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
                      [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4],
                      rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    plt.show()
    # plt.savefig(f"img/paper-figure-histogram.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def simple_unroll_factor():
    data = pd.read_csv(f"data/cache-user.csv", header=0, skipinitialspace=True)
    data = data[(data["mode"] == "loop") & (data["unroll_factor"] == 11)]
    data["SGX"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                           2: "SGX Data in Enclave"})
    data["Mode"] = data["mode"].replace(to_replace="simd_loop_unrolled", value="simd_unrolled")
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("talk")

    plot = sns.lineplot(data, x="Number of Bins", y="Runtime in ms", hue="SGX",
                        hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"],
                        errorbar="sd", palette=sns.color_palette("deep"), legend="auto")

    # sns.move_legend(plot, "lower right", bbox_to_anchor=(1, 0))

    upper_bound = 12
    # plt.legend()

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0, top=0.25)

    plt.xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
               [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4],
               rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    plt.show()
    # plt.savefig(f"img/paper-figure-histogram.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def histogram_scalar_opt_comparison():
    data = pd.read_csv(f"data/opt-dis-hist.csv", header=0, skipinitialspace=True)
    data = data[(data["mode"] == "loop") & ((data["unroll_factor"] == 1) | (data["unroll_factor"] == 9))]
    data["SGX"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                           2: "SGX Data in Enclave"})
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("notebook")

    plot = sns.lineplot(data, x="Number of Bins", y="Runtime in ms", hue="SGX", style="Unroll Factor",
                        hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"], markers=True,
                        errorbar="sd", palette=sns.color_palette("deep"), legend="auto")

    sns.move_legend(plot, "center", bbox_to_anchor=(0.5, 0.55))

    upper_bound = 20
    # plt.legend()

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0)

    plt.xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
               [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4],
               rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    # plt.show()
    plt.savefig(f"img/opt-hist-dis.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def pointer_scalar():
    data = pd.read_csv(f"data/pointer-comparison.csv", header=0, skipinitialspace=True)
    data = data[(data["mode"] == "ptr_loop") & ((data["unroll_factor"] == 1) | (data["unroll_factor"] == 9))]
    data["SGX"] = (data["sgx"] + data["preload"]).replace({0: "Plain CPU", 1: "SGX Data outside Enclave",
                                                           2: "SGX Data in Enclave"})
    data["Unroll Factor"] = data["unroll_factor"]
    data["Runtime in ms"] = data["cpuCycles"] / 2900 / 1000
    data["Number of Bins"] = data["num_bins"]

    sns.set_style("ticks")
    sns.set_context("notebook")

    plot = sns.lineplot(data, x="Number of Bins", y="Runtime in ms", hue="SGX", style="Unroll Factor",
                        hue_order=["Plain CPU", "SGX Data outside Enclave", "SGX Data in Enclave"], markers=True,
                        errorbar="sd", palette=sns.color_palette("deep"), legend="auto")

    # sns.move_legend(plot, "lower right", bbox_to_anchor=(1, 0))

    upper_bound = 25
    # plt.legend()

    plt.xscale('log')
    plt.xlim([1, 2 ** upper_bound])
    plt.ylim(bottom=0)

    plt.xticks([2 ** x for x in range(0, upper_bound + 1, 4)],
               [str(2 ** x) + unit for unit in ["", "ki", "Mi"] for x in range(0, 10)][:upper_bound + 1:4],
               rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    # plt.show()
    plt.savefig(f"img/pointer-comparison.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def main():
    # unroll_comparison()

    # sns.set_palette(sns.color_palette(c_palette_1))

    # for file in ["radix-bits"]:
    #     for measurement in ["timeMicroSec", "instructions", "L1D-misses", "L1I-misses", "LLC-misses", "branch-misses", "dTLB", "iTLB"]:
    #         plot_both(file, measurement)
    # paper_figure_histogram("histogram")
    # detailed_comparison_one_var()
    # figure_histogram_scalar("histogram-user-check")
    # simple_scalar()
    # pointer_scalar()
    # histogram_scalar_opt_comparison()
    # sgx_comparison_one_var()
    # paper_figure_partitioning("part-dis-opt")
    paper_figure_histogram_facet("paper-histogram-data")
    # paper_figure_unrolling_factors()
    # paper_figure_avx_unrolling_factors()
    paper_figure_unrolling_factors_facet()
    # paper_figure_simplest()
    # paper_figure_simplest_facet("simple-test-05-09")
    # paper_figure_mitigation_comparison("paper-histogram-data")


if __name__ == '__main__':
    main()
