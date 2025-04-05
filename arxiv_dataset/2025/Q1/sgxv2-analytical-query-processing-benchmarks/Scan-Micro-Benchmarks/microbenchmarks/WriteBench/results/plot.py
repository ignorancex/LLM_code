import math

import matplotlib.axes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SETTINGS = ["Plain CPU", "SGX Data Outside Enclave", "SGX Data in Enclave"]
SETTINGS_DICT = {
    "yesyes": SETTINGS[2],
    "yesno": SETTINGS[1],
    "nono": SETTINGS[0]
}
UNITS = [str(2 ** x) + unit for unit in ["B", "KiB", "MiB", "GiB"] for x in range(0, 10)]
EXPONENTS = [2 ** x for x in range(40)]
L1 = 2 ** 14 * 3
L2 = 2 ** 18 * 5
L3 = 2 ** 23 * 3


def plot_measurements(file):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)

    data["Array Size"] = data["totalDataSize"]
    data["Setting"] = (data["sgx"] + data["preload"]).replace(SETTINGS_DICT)

    f, axes = plt.subplots(7, 1, sharex="col", squeeze=True, figsize=(8, 15))
    axes: list[matplotlib.axes.Axes]
    for i, measurement in enumerate(["timeMicroSec", "L1D-misses", "L1I-misses", "LLC-misses", "branch-misses", "dTLB",
                                     "iTLB"]):
        sns.lineplot(data, x="Array Size", y=measurement, hue="Setting", style="Setting", hue_order=SETTINGS,
                     style_order=SETTINGS, markers=True, ax=axes[i])
        axes[i].grid(visible=True, axis="x")
        axes[i].set_ylim(bottom=0)

    plt.xlim([2 ** 3, 2 ** 35])
    plt.xscale('log')
    plt.xticks(EXPONENTS[3:36], UNITS[3:36], rotation=90)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"img/figure-random-write-perf-counters.pdf", dpi=300)
    plt.close()


def write_overhead(file="write_unrolled_big"):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)

    data["Runtime in s"] = data["timeMicroSec"] / 1000 / 1000
    data["Array Size"] = data["totalDataSize"]
    native = data.loc[data["sgx"] == "no", :]
    sgx = data.loc[(data["sgx"] == "yes") & (data["preload"] == "yes"), :]

    native.reset_index(inplace=True, drop=True)
    sgx.reset_index(inplace=True, drop=True)

    sgx["Rel. Throughput"] = native["Runtime in s"] / sgx["Runtime in s"]

    plt.figure(figsize=(6, 3.25))
    ax = sns.lineplot(sgx, x="Array Size", y="Rel. Throughput", markers=True)
    ax.axvline(x=L1, ymin=0, ymax=1, linestyle="--", color="grey")
    ax.axvline(x=L2, ymin=0, ymax=1, linestyle="--", color="grey")
    ax.axvline(x=L3, ymin=0, ymax=1, linestyle="--", color="grey")
    plt.text(L1, 0.010, "L1", rotation=90, ha='right', va='bottom', color="grey")
    plt.text(L2, 0.010, "L2", rotation=90, ha='right', va='bottom', color="grey")
    plt.text(L3, 0.010, "L3", rotation=90, ha='right', va='bottom', color="grey")

    plt.xscale('log')
    plt.xlim([8, 2 ** 35])
    plt.xticks(EXPONENTS[3:36], UNITS[3:36], rotation=90)
    plt.ylim(bottom=0)
    plt.minorticks_off()
    plt.grid(visible=True, axis="y")

    plt.tight_layout()
    plt.savefig(f"img/figure-random-write-overhead.pdf", dpi=300)


def write_comparison(file="write_unrolled_big"):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)

    data["Runtime in s"] = data["timeMicroSec"] / 1000 / 1000
    data["Array Size"] = data["totalDataSize"]
    data["Setting"] = (data["sgx"] + data["preload"]).replace(SETTINGS_DICT)

    plt.figure(figsize=(6, 3.25))
    ax = sns.lineplot(data, x="Array Size", y="Runtime in s", hue="Setting", style="Setting", markers=True,
                      hue_order=SETTINGS, style_order=SETTINGS)
    ax.lines[1].set_linestyle(":")
    ax.axvline(x=L1, ymin=0, ymax=1, linestyle="--", color="grey")
    ax.axvline(x=L2, ymin=0, ymax=1, linestyle="--", color="grey")
    ax.axvline(x=L3, ymin=0, ymax=1, linestyle="--", color="grey")
    plt.text(L1, 0.010, "L1", rotation=90, ha='right', va='bottom', color="grey")
    plt.text(L2, 0.010, "L2", rotation=90, ha='right', va='bottom', color="grey")
    plt.text(L3, 0.010, "L3", rotation=90, ha='right', va='bottom', color="grey")

    plt.xscale('log')
    plt.xlim([8, 2 ** 35])
    plt.xticks(EXPONENTS[3:36], UNITS[3:36], rotation=90)
    plt.ylim(bottom=0)
    plt.minorticks_off()

    plt.tight_layout()
    plt.savefig(f"img/figure-random-write-comparison.pdf", dpi=300)


CYCLES_PER_SECOND = 2.9 * 10 ** 9
GiB = 2 ** 30
GB = 10 ** 9
DATA_SIZE_UNIT = GiB


def paper_figure_random_write(file="write_area_increase"):
    X_AXIS_STRING = "Array Size"
    Y_AXIS_STRING = "Write Latency in ns"
    Z_AXIS_REL_STRING = "Rel. Latency"
    HUE_STRING = "Setting"

    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)
    data["time"] = data["cpuCycles"] / CYCLES_PER_SECOND
    data[Y_AXIS_STRING] = data["time"] / data["writeOps"] * 10 ** 9
    data[HUE_STRING] = data["sgx"] + data["preload"]
    data[HUE_STRING].replace(SETTINGS_DICT, inplace=True)

    data[X_AXIS_STRING] = data["totalDataSize"]

    data = data.set_index(X_AXIS_STRING, drop=False)

    native = data.loc[data[HUE_STRING] == SETTINGS[0], Y_AXIS_STRING]
    out_enclave = data.loc[data[HUE_STRING] == SETTINGS[1], Y_AXIS_STRING]
    in_enclave = data.loc[data[HUE_STRING] == SETTINGS[2], Y_AXIS_STRING]

    out_enclave_relative = (out_enclave / native).rename(Z_AXIS_REL_STRING).reset_index()
    in_enclave_relative = (in_enclave / native).rename(Z_AXIS_REL_STRING).reset_index()

    out_enclave_relative[HUE_STRING] = SETTINGS[1]
    in_enclave_relative[HUE_STRING] = SETTINGS[2]

    relative = pd.concat([out_enclave_relative, in_enclave_relative])

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    fig, (ax1, ax2) = plt.subplots(nrows=2, squeeze=True, figsize=(6, 4), sharex="col", height_ratios=(2, 3))

    ax1 = sns.lineplot(relative, x=X_AXIS_STRING, y=Z_AXIS_REL_STRING, hue=HUE_STRING, style=HUE_STRING,
                       hue_order=SETTINGS[1:], style_order=SETTINGS[1:], markers=True, ax=ax1, legend=False,
                       errorbar="sd", palette=sns.color_palette("deep")[1:3])
    ax1.lines[0].set_linestyle("--")
    ax1.lines[0].set_marker("X")
    ax1.lines[1].set_linestyle(":")
    ax1.lines[1].set_marker("s")
    ax1.lines[1].set_markersize(4.5)
    ax1.set_yticks([1, 1.5, 2, 2.5])
    ax2 = sns.lineplot(data, x=X_AXIS_STRING, y=Y_AXIS_STRING, hue=HUE_STRING, style=HUE_STRING, hue_order=SETTINGS,
                       style_order=SETTINGS, markers=True, errorbar="sd", ax=ax2)

    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xlim([2 ** 3, 2 ** 35])
        ax.minorticks_off()

        ax.axvline(x=L1, linestyle="--", color="grey", label="L1 Cache Size")
        ax.axvline(x=L2, linestyle="--", color="grey", label="L2 Cache Size")
        ax.axvline(x=L3, linestyle="--", color="grey", label="L3 Cache Size")

        ax.yaxis.set_label_coords(-0.09, 0.5)
        ax.grid(visible=True, axis="y")

    ax1.set_ylim(bottom=0.95)
    ax1.text(L1, 2.5, "L1", rotation=90, ha='right', color='grey')
    ax1.text(L2, 2.5, "L2", rotation=90, ha='right', color='grey')
    ax1.text(L3, 2.5, "L3", rotation=90, ha='right', color='grey')
    ax2.text(L1, 10, "L1", rotation=90, ha='right', color='grey')
    ax2.text(L2, 10, "L2", rotation=90, ha='right', color='grey')
    ax2.text(L3, 10, "L3", rotation=90, ha='right', color='grey')

    ax2.set_ylim(bottom=0)
    ax2.set_xticks(EXPONENTS[3:36], UNITS[3:36], rotation=90)

    plt.tight_layout()

    # plt.show()
    plt.savefig(f"img/paper-figure-random-write.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def write_size_comparison(file="write_size_large"):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)

    data["Runtime in s"] = data["timeMicroSec"] / 1000 / 1000
    data["totalWritten"] = data["writeOps"] * data["writeSize"] / 2 ** 30
    data["Throughput in GB/s"] = data["totalWritten"] / data["Runtime in s"]
    data["Write Size"] = data["writeSize"]
    data["Setting"] = (data["sgx"] + data["preload"]).replace(SETTINGS_DICT)

    plt.figure(figsize=(6, 3.25))
    ax = sns.lineplot(data, x="Write Size", y="Throughput in GB/s", hue="Setting", style="Setting", markers=True,
                      hue_order=SETTINGS, style_order=SETTINGS)

    plt.xscale('log')
    size_min = data["Write Size"].min()
    size_max = data["Write Size"].max()
    size_min_log = int(math.log2(size_min))
    size_max_log = int(math.log2(size_max))
    plt.xlim([size_min, size_max])
    plt.xticks([2 ** x for x in range(size_min_log, size_max_log + 1)],
               (UNITS[
                size_min_log:size_max_log + 1]),
               rotation=90)
    plt.ylim(bottom=0)
    plt.minorticks_off()

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"img/figure-write-size-comparison.pdf", dpi=300)


def write_size_overhead(file="write_size_combined"):
    data = pd.read_csv(f"data/{file}.csv", header=0, skipinitialspace=True)

    data["Runtime in s"] = data["timeMicroSec"] / 1000 / 1000
    data["totalWritten"] = data["writeOps"] * data["writeSize"] / 2 ** 30
    data["Throughput in GB/s"] = data["totalWritten"] / data["Runtime in s"]
    data["Write Size"] = data["writeSize"]
    native = data.loc[data["sgx"] == "no", :]
    sgx = data.loc[(data["sgx"] == "yes") & (data["preload"] == "yes"), :]

    native.reset_index(inplace=True, drop=True)
    sgx.reset_index(inplace=True, drop=True)

    sgx["Rel. Throughput"] = sgx["Throughput in GB/s"] / native["Throughput in GB/s"]

    plt.figure(figsize=(6, 5))
    ax = sns.lineplot(sgx, x="Write Size", y="Rel. Throughput", markers=True)

    plt.xscale('log')
    size_min = data["Write Size"].min()
    size_max = data["Write Size"].max()
    size_min_log = int(math.log2(size_min))
    size_max_log = int(math.log2(size_max))
    plt.xlim([size_min, size_max])
    plt.xticks([2 ** x for x in range(size_min_log, size_max_log + 1)],
               (UNITS[
                size_min_log:size_max_log + 1]),
               rotation=90)
    plt.yticks([x / 10 for x in range(0, 12)])
    plt.ylim(bottom=0, top=1)
    plt.minorticks_off()
    plt.grid(visible=True, axis="y")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"img/figure-write-size-overhead.pdf", dpi=300)


def paper_figure_increment():
    data = pd.read_csv("data/increment.csv", header=0, skipinitialspace=True)

    sns.set_style("ticks")
    sns.set_context("notebook")

    data["Runtime in s"] = data["timeMicroSec"] / 1000 / 1000
    data["Array Size"] = data["totalDataSize"]
    data["Setting"] = data["sgx"].replace({
        "yes": "SGX Data in Enclave",
        "no": "Plain CPU"
    })

    data = data[data["totalDataSize"] <= 2 ** 21]

    plt.figure(figsize=(6, 3.25))
    ax = sns.lineplot(data, x="Array Size", y="Runtime in s", hue="Setting", style="Setting", markers=True,
                      hue_order=["Plain CPU", "SGX Data in Enclave"], style_order=["Plain CPU", "SGX Data in Enclave"],
                      palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[2]])
    ax.lines[1].set_linestyle(":")
    ax.axvline(x=L1, ymin=0, ymax=1, linestyle="--", color="grey")
    #ax.axvline(x=L2, ymin=0, ymax=1, linestyle="--", color="grey")
    plt.text(L1, 0.010, "L1", rotation=90, ha='right', va='bottom', color="grey")
    #plt.text(L2, 0.010, "L2", rotation=90, ha='right', va='bottom', color="grey")

    plt.xscale('log')
    # plt.yscale('log')
    plt.grid(axis="y")
    plt.xlim([8, 2 ** 16])
    plt.xticks([2 ** x for x in range(3, 17)],
               UNITS[3:17],
               rotation=90)
    plt.ylim(bottom=0)
    plt.minorticks_off()

    plt.tight_layout()
    plt.savefig(f"img/paper-figure-increment.pdf", dpi=300)
    plt.close()


def overview_figure():
    data = pd.concat([pd.read_csv("data/random-write.csv", header=0, skipinitialspace=True),
                      # pd.read_csv("data/random-increment.csv", header=0, skipinitialspace=True),
                      # pd.read_csv("data/non-temp-write.csv", header=0, skipinitialspace=True),
                      pd.read_csv("data/random_write_unrolled.csv", header=0, skipinitialspace=True)])

    ax = sns.lineplot(data, x="totalDataSize", y="timeMicroSec", hue="sgx", style="workload", markers=True)
    ax.axvline(x=L1, ymin=0, ymax=1, linestyle="--", color="k", label="L1 Cache Size")
    ax.axvline(x=L2, ymin=0, ymax=1, linestyle="--", color="r", label="L2 Cache Size")
    ax.axvline(x=L3, ymin=0, ymax=1, linestyle="--", color="b", label="L3 Cache Size")
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlim([1, 2 ** 31])
    plt.xticks([2 ** x for x in range(0, 31)],
               UNITS[:31],
               rotation=90)
    plt.legend()
    plt.show()

    # data.set_index("totalDataSize", inplace=True)
    # data.groupby("sgx")["timeMicroSec"].plot(kind="line", logx=True, legend=True)
    # plt.show()


def paper_figure_random_access_mean(write_file, read_file):
    access_types = ["Random Read", "Random Write"]

    # prepare random write data
    df_write = pd.read_csv(f"data/{write_file}.csv", header=0, skipinitialspace=True)
    df_write = df_write[~((df_write["sgx"] == "yes") & (df_write["preload"] == "no"))]
    df_write["Runtime in s"] = df_write["timeMicroSec"] / 1000 / 1000
    df_write["Data Size"] = df_write["totalDataSize"]

    mean_runtime = df_write.groupby(["sgx", "Data Size"])["Runtime in s"].mean()

    df_write = mean_runtime.reset_index()

    write_native = df_write.loc[df_write["sgx"] == "no", :]
    write_sgx = df_write.loc[df_write["sgx"] == "yes", :]

    write_native.reset_index(inplace=True, drop=True)
    write_sgx.reset_index(inplace=True, drop=True)

    write_sgx["Rel. Performance"] = write_native["Runtime in s"] / write_sgx["Runtime in s"]
    write_sgx["Access Type"] = access_types[1]

    # prepare random independent read data
    df_read = pd.read_csv(f"data/{read_file}.csv", header=0, skipinitialspace=True)
    df_read = df_read[~((df_read["sgx"] == "yes") & (df_read["preload"] == "no"))]
    df_read["Runtime in s"] = df_read["timeMicroSec"] / 1000 / 1000
    df_read["Data Size"] = df_read["totalDataSize"]

    mean_runtime = df_read.groupby(["sgx", "Data Size"])["Runtime in s"].mean()
    df_read = mean_runtime.reset_index()

    read_native = df_read.loc[df_read["sgx"] == "no", :]
    read_sgx = df_read.loc[df_read["sgx"] == "yes", :]

    read_native.reset_index(inplace=True, drop=True)
    read_sgx.reset_index(inplace=True, drop=True)

    read_sgx["Rel. Performance"] = read_native["Runtime in s"] / read_sgx["Runtime in s"]
    read_sgx["Access Type"] = access_types[0]

    combined_df = pd.concat([write_sgx[["Access Type", "Data Size", "Rel. Performance"]],
                             read_sgx[["Access Type", "Data Size", "Rel. Performance"]]])

    sns.set_style("ticks")
    sns.set_context("notebook")

    plt.figure(figsize=(6, 2))
    plot = sns.lineplot(combined_df, x='Data Size', y='Rel. Performance', hue='Access Type', style='Access Type',
                        hue_order=access_types, style_order=access_types, errorbar="sd", markers=True,
                        palette=sns.color_palette("deep"))

    plt.xscale('log')
    plt.xlim([2 ** 10, 2 ** 35])
    plt.ylim(bottom=0)

    plot.set_xticks([2 ** x for x in range(10, 36, 5)],
                    ([str(2 ** x) + unit for unit in ["KB", "MB", "GB"] for x in range(0, 10)])[:26:5])

    plot.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25])

    plt.axvline(x=2 ** 14 * 3, linestyle="--", color="grey")
    plt.axvline(x=2 ** 18 * 5, linestyle="--", color="grey")
    plt.axvline(x=2 ** 23 * 3, linestyle="--", color="grey")
    plt.text(2 ** 14 * 3, 0.02, "L1", rotation=90, ha='right', va='bottom', color="grey")
    plt.text(2 ** 18 * 5, 0.02, "L2", rotation=90, ha='right', va='bottom', color="grey")
    plt.text(2 ** 23 * 3 + 2 ** 20, 0.02, "L3", rotation=90, ha='left', va='bottom', color="grey")
    sns.move_legend(plot, "lower center", frameon=False, bbox_to_anchor=(0.5, 0.95), title=None, ncols=2)
    #plot.yaxis.set_label_coords(-0.1, 0.4)
    plot.axhline(1, linestyle="-", color="grey", alpha=0.5, linewidth=1)

    plt.grid(axis="y")
    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig('img/paper-figure-random.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    percentages = combined_df.loc[combined_df["Data Size"] >= 2 ** 33, :]
    mean_percentages = percentages.groupby(["Access Type", "Data Size"]).agg(["mean", "median", "min", "max"])
    print(mean_percentages.to_string())


def main():
    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette("deep")

    paper_figure_random_write()
    paper_figure_increment()
    paper_figure_random_access_mean("write_area_increase", "read")


if __name__ == '__main__':
    main()
