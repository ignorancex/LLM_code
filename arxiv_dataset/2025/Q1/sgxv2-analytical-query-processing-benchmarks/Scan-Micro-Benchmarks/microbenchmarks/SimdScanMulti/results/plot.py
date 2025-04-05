import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns

matplotlib.use('qtagg')
CYCLES_PER_SECOND = 2.9 * 10 ** 9
GiB = 2 ** 30
GB = 10 ** 9
DATA_SIZE_UNIT = GiB


SETTINGS = ["Plain CPU", "SGX DoE", "SGX DiE"]


def plot_single_thread_lines(df: DataFrame, threads, mode):
    data = df[df["numThreads"] == threads]

    time = data["cpuCycles"] / data["reruns"] / CYCLES_PER_SECOND
    throughput = data["entries"] * data["numRuns"] / time / DATA_SIZE_UNIT

    unsafe: Series = throughput.loc[(df["writeMode"] == mode) &
                                    (df["enclaveMode"] == "native")]
    user: Series = throughput.loc[(df["writeMode"] == mode) &
                                  (df["enclaveMode"] == "enclave") &
                                  (df["dataLoading"] == "noPreload")]
    preload: Series = throughput.loc[(df["writeMode"] == mode) &
                                     (df["enclaveMode"] == "enclave") &
                                     (df["dataLoading"] == "preload")]

    unsafe.reset_index(drop=True, inplace=True)
    user.reset_index(drop=True, inplace=True)
    preload.reset_index(drop=True, inplace=True)

    plot_df = pd.concat([unsafe, user, preload], axis=1)
    plot_df.columns = ["unsafe", "user", "preload"]

    plot_df.plot()

    plt.ylim(bottom=0)
    plt.xticks(range(21), ([str(2 ** x) + unit for unit in ["KB", "MB", "GB"] for x in range(0, 10)])[:21], rotation=90)
    plt.grid()
    plt.show()


def paper_plot_cache_scan():
    data = pd.read_csv("data/single-thread.csv", header=0, skipinitialspace=True)
    data = data[data["numThreads"] == 1]
    data = data[data["writeMode"] == "bitvector"]
    data = data[data["entries"] >= 2 ** 12]
    data = data[data["selectivity"] == 0.01]


    # drop_index = data[(data["numRuns"] == 100) & (data["datasizeKiB"] <= 1048576)].index
    # data.drop(drop_index, inplace=True)


    test_1000 = data[(data["numRuns"] == 1000) & (data["datasizeKiB"].isin([262144, 524288, 1048576]))]
    test_100 = data[(data["numRuns"] == 100) & (data["datasizeKiB"].isin([262144, 524288, 1048576]))]

    test_1000_means = test_1000.groupby(["enclaveMode", "dataLoading", "datasizeKiB"])[["timeMicroSec", "cpuCycles"]].mean()
    test_100_means = test_100.groupby(["enclaveMode", "dataLoading", "datasizeKiB"])[["timeMicroSec", "cpuCycles"]].mean()

    relative = test_100_means * 10 / test_1000_means

    print(test_1000_means.to_string())
    print(test_100_means.to_string())
    print(relative.to_string())

    data["time"] = data["cpuCycles"] / data["reruns"] / CYCLES_PER_SECOND
    data["Scan Throughput\nin GiB/s"] = data["entries"] * data["numRuns"] / data["time"] / DATA_SIZE_UNIT
    data["Setting"] = data["enclaveMode"] + data["dataLoading"]
    data["Setting"].replace({"enclavepreload": SETTINGS[2],
                             "enclavenoPreload": SETTINGS[1],
                             "nativenoPreload": SETTINGS[0]},
                            inplace=True)

    data["Data Size"] = data["entries"]

    means = data.groupby(["Setting", "Data Size"])["Scan Throughput\nin GiB/s"].mean()

    native = means[SETTINGS[0]]
    out_enclave = means[SETTINGS[1]]
    in_enclave = means[SETTINGS[2]]

    out_enclave_relative = (out_enclave / native).rename("Relative").reset_index()
    in_enclave_relative = (in_enclave / native).rename("Relative").reset_index()

    out_enclave_relative["Setting"] = SETTINGS[1]
    in_enclave_relative["Setting"] = SETTINGS[2]

    relative = pd.concat([out_enclave_relative, in_enclave_relative])

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    fig, (ax1, ax2) = plt.subplots(nrows=2, squeeze=True, figsize=(6, 3.25), sharex="col", height_ratios=(3, 6))

    ax1 = sns.lineplot(relative, x="Data Size", y="Relative", hue="Setting", style="Setting",
                       hue_order=[SETTINGS[1], SETTINGS[2]],
                       style_order=[SETTINGS[1], SETTINGS[2]], markers=True,
                       ax=ax1, legend=False, errorbar="sd", palette=sns.color_palette("deep")[1:])
    ax1.lines[0].set_linestyle("--")
    ax1.lines[0].set_marker("X")
    ax1.lines[1].set_linestyle(":")
    ax1.lines[1].set_marker("s")
    ax1.lines[1].set_markersize(4.2)

    ax1.set_yticks([1, 1.1, 1.2, 1.3])

    ax2 = sns.lineplot(data, x="Data Size", y="Scan Throughput\nin GiB/s", hue="Setting", style="Setting",
                       hue_order=SETTINGS,
                       style_order=SETTINGS, markers=True,
                       errorbar="sd", ax=ax2)

    ax2.set_yticks([0, 25, 50, 75])

    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xlim([2 ** 12, 2 ** 33])
        ax.minorticks_off()
        ax.grid(visible=True, axis="y")

        ax.axvline(x=2 ** 14 * 3, linestyle="--", color="grey", label="L1 Cache Size")
        ax.axvline(x=2 ** 18 * 5, linestyle="--", color="grey", label="L2 Cache Size")
        ax.axvline(x=2 ** 23 * 3, linestyle="--", color="grey", label="L3 Cache Size")

    ax1.yaxis.set_label_coords(-0.09, 0.5)
    ax2.yaxis.set_label_coords(-0.07, 0.5)

    ax1.set_ylim(bottom=0.95)
    ax2.text(2 ** 14 * 3, 3, "L1", rotation=90, ha='right', color='grey')
    ax2.text(2 ** 18 * 5, 3, "L2", rotation=90, ha='right', color='grey')
    ax2.text(2 ** 23 * 3, 3, "L3", rotation=90, ha='right', color='grey')

    ax2.set_ylim(bottom=0)
    ax2.set_xticks([2 ** x for x in [12, 15, 20, 25, 30, 34]],
                   ["4KB", "32KB", "1MB", "32MB", "1GB", "16GB"])

    ax1.legend(ax2.lines, SETTINGS, loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.87), title=None, ncols=3)
    ax2.legend_.set_visible(False)

    plt.tight_layout()

    # plt.show()
    plt.savefig(f"img/paper-figure-in-cache-scan-new.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_plot_cache_scan_16():
    data = pd.read_csv("data/single-thread-16bit.csv", header=0, skipinitialspace=True)
    data = data[data["numThreads"] == 1]
    data = data[data["writeMode"] == "bitvector"]
    data = data[data["entries"] >= 2 ** 12]
    data = data[data["selectivity"] == 0.01]


    # drop_index = data[(data["numRuns"] == 100) & (data["datasizeKiB"] <= 1048576)].index
    # data.drop(drop_index, inplace=True)


    test_1000 = data[(data["numRuns"] == 1000) & (data["datasizeKiB"].isin([262144, 524288, 1048576]))]
    test_100 = data[(data["numRuns"] == 100) & (data["datasizeKiB"].isin([262144, 524288, 1048576]))]

    test_1000_means = test_1000.groupby(["enclaveMode", "dataLoading", "datasizeKiB"])[["timeMicroSec", "cpuCycles"]].mean()
    test_100_means = test_100.groupby(["enclaveMode", "dataLoading", "datasizeKiB"])[["timeMicroSec", "cpuCycles"]].mean()

    relative = test_100_means * 10 / test_1000_means

    print(test_1000_means.to_string())
    print(test_100_means.to_string())
    print(relative.to_string())

    data["time"] = data["cpuCycles"] / data["reruns"] / CYCLES_PER_SECOND
    data["Scan Throughput\nin GiB/s"] = data["entries"] * data["numRuns"] / data["time"] / DATA_SIZE_UNIT
    data["Setting"] = data["enclaveMode"] + data["dataLoading"]
    data["Setting"].replace({"enclavepreload": SETTINGS[2],
                             "enclavenoPreload": SETTINGS[1],
                             "nativenoPreload": SETTINGS[0]},
                            inplace=True)

    data["Data Size"] = data["entries"]

    means = data.groupby(["Setting", "Data Size"])["Scan Throughput\nin GiB/s"].mean()

    native = means[SETTINGS[0]]
    out_enclave = means[SETTINGS[1]]
    in_enclave = means[SETTINGS[2]]

    out_enclave_relative = (out_enclave / native).rename("Relative").reset_index()
    in_enclave_relative = (in_enclave / native).rename("Relative").reset_index()

    out_enclave_relative["Setting"] = SETTINGS[1]
    in_enclave_relative["Setting"] = SETTINGS[2]

    relative = pd.concat([out_enclave_relative, in_enclave_relative])

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    fig, (ax1, ax2) = plt.subplots(nrows=2, squeeze=True, figsize=(6, 3.25), sharex="col", height_ratios=(3, 6))

    ax1 = sns.lineplot(relative, x="Data Size", y="Relative", hue="Setting", style="Setting",
                       hue_order=[SETTINGS[1], SETTINGS[2]],
                       style_order=[SETTINGS[1], SETTINGS[2]], markers=True,
                       ax=ax1, legend=False, errorbar="sd", palette=sns.color_palette("deep")[1:])
    ax1.lines[0].set_linestyle("--")
    ax1.lines[0].set_marker("X")
    ax1.lines[1].set_linestyle(":")
    ax1.lines[1].set_marker("s")
    ax1.lines[1].set_markersize(4.2)

    ax1.set_yticks([1, 1.1, 1.2, 1.3])

    ax2 = sns.lineplot(data, x="Data Size", y="Scan Throughput\nin GiB/s", hue="Setting", style="Setting",
                       hue_order=SETTINGS,
                       style_order=SETTINGS, markers=True,
                       errorbar="sd", ax=ax2)

    ax2.set_yticks([0, 25, 50, 75])

    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xlim([2 ** 12, 2 ** 33])
        ax.minorticks_off()
        ax.grid(visible=True, axis="y")

        ax.axvline(x=2 ** 14 * 3, linestyle="--", color="grey", label="L1 Cache Size")
        ax.axvline(x=2 ** 18 * 5, linestyle="--", color="grey", label="L2 Cache Size")
        ax.axvline(x=2 ** 23 * 3, linestyle="--", color="grey", label="L3 Cache Size")

    ax1.yaxis.set_label_coords(-0.09, 0.5)
    ax2.yaxis.set_label_coords(-0.07, 0.5)

    ax1.set_ylim(bottom=0.95)
    ax2.text(2 ** 14 * 3, 3, "L1", rotation=90, ha='right', color='grey')
    ax2.text(2 ** 18 * 5, 3, "L2", rotation=90, ha='right', color='grey')
    ax2.text(2 ** 23 * 3, 3, "L3", rotation=90, ha='right', color='grey')

    ax2.set_ylim(bottom=0)
    ax2.set_xticks([2 ** x for x in [12, 15, 20, 25, 30, 34]],
                   ["4KB", "32KB", "1MB", "32MB", "1GB", "16GB"])

    ax1.legend(ax2.lines, SETTINGS, loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.87), title=None, ncols=3)
    ax2.legend_.set_visible(False)

    plt.tight_layout()

    # plt.show()
    plt.savefig(f"img/paper-figure-in-cache-scan-16bit.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def compare_8_16():
    data_8 = pd.read_csv("data/single-thread.csv", header=0, skipinitialspace=True)
    data_8 = data_8[data_8["numThreads"] == 1]
    data_8 = data_8[data_8["writeMode"] == "bitvector"]
    data_8 = data_8[data_8["entries"] >= 2 ** 12]
    data_8 = data_8[data_8["selectivity"] == 0.01]

    drop_index = data_8[(data_8["numRuns"] == 100) & (data_8["datasizeKiB"] <= 1048576)].index
    data_8.drop(drop_index, inplace=True)

    data_16 = pd.read_csv("data/single-thread-16bit.csv", header=0, skipinitialspace=True)
    data_16 = data_16[data_16["numThreads"] == 1]
    data_16 = data_16[data_16["writeMode"] == "bitvector"]
    data_16 = data_16[data_16["entries"] >= 2 ** 12]
    data_16 = data_16[data_16["selectivity"] == 0.01]

    drop_index = data_16[(data_16["numRuns"] == 100) & (data_16["datasizeKiB"] <= 1048576)].index
    data_16.drop(drop_index, inplace=True)

    data_8_mean = data_8.groupby(["enclaveMode", "dataLoading", "entries"])[["timeMicroSec", "cpuCycles"]].mean()
    data_16_mean = data_16.groupby(["enclaveMode", "dataLoading", "entries"])[["timeMicroSec", "cpuCycles"]].mean()

    relative = data_16_mean / data_8_mean

    print(data_8_mean.to_string())
    print(data_16_mean.to_string())
    print(relative.to_string())

    relative.reset_index(inplace=True)

    relative["Setting"] = relative["enclaveMode"] + relative["dataLoading"]
    relative["Setting"].replace({"enclavepreload": SETTINGS[2],
                             "enclavenoPreload": SETTINGS[1],
                             "nativenoPreload": SETTINGS[0]}, inplace=True)

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    plt.figure(figsize=(6, 2.5))
    ax2 = sns.lineplot(relative, y="cpuCycles", x="entries", hue="Setting", style="Setting", hue_order=SETTINGS,
                       style_order=SETTINGS, markers=True)

    ax2.set_xscale('log')
    ax2.set_xlim([2 ** 12, 2 ** 34])
    ax2.minorticks_off()
    ax2.grid(visible=True, axis="y")

    ax2.set_ylim(bottom=0.85, top=1.15)
    ax2.set_yticks([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])
    ax2.set_xticks([2 ** x for x in [12, 15, 20, 25, 30, 34]],
                   ["4KB", "32KB", "1MB", "32MB", "1GB", "16GB"])

    ax2.set_ylabel("16 bit Relative Runtime\n$<1\\rightarrow$16 bit faster")
    ax2.set_xlabel("Data Size")

    sns.move_legend(ax2, "lower center", frameon=False, bbox_to_anchor=(0.5, 0.95), title=None, ncols=3)

    plt.tight_layout()

    # plt.show()
    plt.savefig(f"img/paper-figure-16bit-relative.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

    #plt.show()


def plot_no_cache_scan():
    data = pd.read_csv("data/out-of-cache.csv", header=0, skipinitialspace=True)
    data = data[data["numThreads"] == 1]
    data = data[data["writeMode"] == "bitvector"]
    data["time"] = data["cpuCycles"] / data["reruns"] / CYCLES_PER_SECOND
    data["Scan Throughput\nin GiB/s"] = data["entries"] * data["numRuns"] / data["time"] / DATA_SIZE_UNIT
    data["Setting"] = data["enclaveMode"] + data["dataLoading"]
    data["Setting"].replace({"enclavepreload": SETTINGS[2],
                             "enclavenoPreload": SETTINGS[1],
                             "nativenoPreload": SETTINGS[0]},
                            inplace=True)

    data["Data Size"] = data["entries"]

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    plt.figure(figsize=(6, 3.25))
    plot = sns.lineplot(data, x="Data Size", y="Scan Throughput\nin GiB/s", hue="Setting", style="Setting",
                        hue_order=[SETTINGS[2], SETTINGS[1], SETTINGS[0]],
                        style_order=[SETTINGS[2], SETTINGS[1], SETTINGS[0]], markers=True)

    plt.xscale('log')
    plt.xlim([2 ** 10, 2 ** 30])
    plt.ylim(bottom=0)

    plot.set_xticks([2 ** x for x in range(10, 31)],
                    ([str(2 ** x) + unit for unit in ["KB", "MB", "GB"] for x in range(0, 10)])[:21],
                    rotation=90)

    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig(f"img/paper-figure-out-of-cache-scan.pdf", dpi=300)
    plt.close()


def paper_figure_multithreading():
    data = pd.read_csv("data/scale-up.csv", header=0, skipinitialspace=True)
    data = data[data["writeMode"] == "bitvector"]
    data["time"] = data["cpuCycles"] / data["reruns"] / CYCLES_PER_SECOND
    data["Scan Throughput\nin GiB/s"] = data["entries"] * data["numRuns"] / data["time"] / DATA_SIZE_UNIT
    data["Setting"] = data["enclaveMode"] + data["dataLoading"]
    data["Setting"].replace({"enclavepreload": SETTINGS[2],
                             "enclavenoPreload": SETTINGS[1],
                             "nativenoPreload": SETTINGS[0]},
                            inplace=True)
    data["Data Size"] = data["entries"]
    data["Threads"] = data["numThreads"]

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    f = plt.figure(figsize=(6, 2.5))
    plt.grid(axis='y')

    plot = sns.barplot(data, y="Scan Throughput\nin GiB/s", x="Threads", hue="Setting",
                       hue_order=SETTINGS, errorbar="sd", ax=f.axes[0])

    sns.move_legend(plot, "lower center", frameon=False, bbox_to_anchor=(0.5, 0.95), title=None, ncols=3)

    hatches = ['', '//', '\\\\']

    for i, bar in enumerate(plot.patches):
        if i // 5 == 1:
            bar.set_hatch(hatches[1])
        elif i // 5 == 2:
            bar.set_hatch(hatches[2])

    for hatch, handle in zip(hatches, f.axes[0].get_legend().legend_handles):
        handle.set_hatch(hatch)

    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"img/paper-figure-scan-multithreading.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def paper_figure_write_rate():
    data = pd.read_csv("data/write-rate.csv", header=0, skipinitialspace=True)
    data = data[data["writeMode"] == "noindex"]
    data["time"] = data["cpuCycles"] / data["reruns"] / CYCLES_PER_SECOND
    data["Scan Throughput\nin GiB/s"] = data["entries"] * data["numRuns"] / data["time"] / DATA_SIZE_UNIT
    data["Setting"] = data["enclaveMode"] + data["dataLoading"]
    data["Setting"].replace({"enclavepreload": SETTINGS[2],
                             "enclavenoPreload": SETTINGS[1],
                             "nativenoPreload": SETTINGS[0]},
                            inplace=True)
    data["Write Rate"] = data["selectivity"] * 8

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    f = plt.figure(figsize=(6, 2.5))
    plt.grid(axis='y')

    plot = sns.barplot(data, y="Scan Throughput\nin GiB/s", x="Write Rate", hue="Setting",
                       hue_order=SETTINGS,
                       err_kws={'linewidth': 2}, errorbar="sd", ax=f.axes[0])

    sns.move_legend(plot, "lower center", frameon=False, bbox_to_anchor=(0.5, 0.95), title=None, ncols=3)

    hatches = ['', '//', '\\\\']

    for i, bar in enumerate(plot.patches):
        if i // 11 == 1:
            bar.set_hatch(hatches[1])
        elif i // 11 == 2:
            bar.set_hatch(hatches[2])

    for hatch, handle in zip(hatches, f.axes[0].get_legend().legend_handles):
        handle.set_hatch(hatch)

    plt.ylim(bottom=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"img/paper-figure-scan-write-rate.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def numa_absolute_values_2():
    data = pd.read_csv("data/cross-numa.csv", header=0, skipinitialspace=True)

    data["time"] = data["cpuCycles"] / data["reruns"] / CYCLES_PER_SECOND
    data["Scan Throughput\nin GiB/s"] = data["entries"] * data["numRuns"] / data["time"] / DATA_SIZE_UNIT
    data["Setting"] = data["enclaveMode"] + data["dataLoading"] + data["numa"]
    data["Setting"].replace({"enclavepreloadyes": "SGX cross-NUMA",
                             "nativenoPreloadyes": "P. CPU cross-NUMA",
                             "nativenoPreloadno": "P. CPU local-NUMA",},
                            inplace=True)

    data = data[data["Setting"].isin(["SGX cross-NUMA", "P. CPU cross-NUMA", "P. CPU local-NUMA"])]
    data["Threads"] = data["numThreads"]

    c_palette_1 = ["#D56257", "#8e8e8e", "#29292b", "#176582"]

    sns.set_style("ticks")
    sns.set_context("notebook")

    f = plt.figure(figsize=(6, 2.5))
    plt.grid(axis='y')

    plot = sns.barplot(data, y="Scan Throughput\nin GiB/s", x="Threads", hue="Setting",
                       hue_order=["P. CPU local-NUMA", "P. CPU cross-NUMA", "SGX cross-NUMA"],
                       errorbar="sd", palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[3],
                                               sns.color_palette("deep")[5]], ax=f.axes[0])

    sns.move_legend(plot, "lower center", frameon=False, bbox_to_anchor=(0.5, 0.95), title=None, ncols=3, fontsize=9, columnspacing=0.8)

    hatches = ['', '--', 'oo']

    for i, bar in enumerate(plot.patches):
        if i // 5 == 1:
            bar.set_hatch(hatches[1])
        elif i // 5 == 2:
            bar.set_hatch(hatches[2])

    for hatch, handle in zip(hatches, f.axes[0].get_legend().legend_handles):
        handle.set_hatch(hatch)

    plt.ylim(bottom=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"img/paper-figure-numa-absolute.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def plot_threads(df: DataFrame, mode):
    data = df[df["mode"] == mode]

    time = data["total_cycles"] / data["reruns"] / CYCLES_PER_SECOND
    throughput = data["entries"] * data["runs_eq"] / time / DATA_SIZE_UNIT

    for thread in [1, 2, 4, 8, 16]:
        thread_series = throughput.loc[df["threads"] == thread]
        thread_series.reset_index(drop=True, inplace=True)
        plt.plot(thread_series, label=str(thread))

    plt.legend()
    plt.ylim(bottom=0)
    plt.xticks(range(21), ["4KB", "8KB", "16KB", "32KB", "64KB", "128KB", "256KB", "512KB",
                           "1MB", "2MB", "4MB", "8MB", "16MB", "32MB", "64MB", "128MB", "256MB", "512MB",
                           "1GB", "2GB", "4GB"], rotation=90)
    plt.grid()
    plt.show()


def main():
    data = pd.read_csv("data/dict-pattern-data.csv", header=0, skipinitialspace=True)

    plot_single_thread_lines(data.copy(), 1, "dict")
    # plot_threads(data, "bitvector_unsafe")


if __name__ == '__main__':
    # main()
    paper_plot_cache_scan()  # figure 13
    paper_figure_multithreading()  # figure 14
    paper_figure_write_rate()  # figure 15
    numa_absolute_values_2()  # figure 16
    paper_plot_cache_scan_16()  # figure 19 (revision letter)
    compare_8_16()  # figure 20 (revision letter)
