import marimo

__generated_with = "0.7.8"
app = marimo.App(width="medium")


@app.cell
def __(benchmark_files, mo):
    mo.md(f"# Benchmark Analysis {benchmark_files}")


@app.cell
def __(mo, npad_bench_data):
    npad_plotting_data = npad_bench_data[
        (npad_bench_data["sparsity"] == "sparse")
        & (npad_bench_data["params.ntrunc"] == 3)
    ]
    mo.ui.dataframe(npad_plotting_data)
    return (npad_plotting_data,)


@app.cell
def __():
    import polars as pl

    return (pl,)


@app.cell
def __():
    return


@app.cell
def __():
    import plotly.express as px

    return (px,)


@app.cell
def __(npad_plotting_data, pl):
    npad_speedup = pl.DataFrame(npad_plotting_data).select(
        [
            pl.col("params.chain_size").cast(pl.Int64).alias("Size"),
            pl.col("stats.mean").max().cast(pl.Float64).alias("Speedup"),
        ]
    )
    npad_speedup
    return (npad_speedup,)


@app.cell
def __(npad_speedup, px):
    px.bar(
        npad_speedup,
        x="Size",
        y="Speedup",
    )


@app.cell
def __(magnus_bench_data, npad_plotting_data, plt, sns):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    npad_bmark_times = sns.barplot(
        npad_plotting_data,
        x="ndim",
        y="stats.mean",
        hue="res",
        hue_order=["qutip", "cpu", "gpu"],
        ax=ax[0],
    )
    ax[0].set(
        yscale="log",
        title="NPAD Benchmarks",
        # xscale="log",
    )
    magnus_bmark_times = sns.barplot(
        magnus_bench_data,
        x="ndim",
        y="stats.mean",
        hue="res",
        ax=ax[1],
    )
    ax[1].set(
        yscale="log",
        ylim=(1e-3, 1e2),
        title="Mangus Benchmarks",
        # xscale="log",
    )
    fig
    return ax, fig, magnus_bmark_times, npad_bmark_times


app._unparsable_cell(
    r"""
    import polars as 
    """,
    name="__",
)


@app.cell
def __(ax, magnus_bench_data, sns):
    sns.barplot(
        magnus_bench_data,
        x="ndim",
        y="stats.mean",
        hue="res",
        ax=ax[1],
    )


@app.cell
def __(benchmark_files, json, mo, pd):
    # Extract relevant statistics
    bench_data = pd.concat(
        [
            pd.json_normalize(
                json.loads(benchmark_json.contents.decode("utf-8"))["benchmarks"]
            )
            for benchmark_json in benchmark_files.value
        ]
    )
    # tag the various benchmarks
    # First device (cpu/gpu)
    bench_data.loc[bench_data["name"].str.find("cpu") != -1, "res"] = "cpu"
    bench_data.loc[bench_data["name"].str.find("gpu") != -1, "res"] = "gpu"
    # Then method (qutip/npad/magnus)
    bench_data.loc[bench_data["name"].str.find("npad") != -1, "benchmark"] = "npad"
    bench_data.loc[bench_data["name"].str.find("qutip") != -1, "benchmark"] = "qutip"
    bench_data.loc[bench_data["name"].str.find("qutip") != -1, "res"] = "qutip"
    bench_data.loc[bench_data["name"].str.find("magnus") != -1, "benchmark"] = "magnus"
    # Sparsity
    bench_data.loc[bench_data["name"].str.find("sparse") != -1, "sparsity"] = "sparse"
    bench_data.loc[bench_data["name"].str.find("dense") != -1, "sparsity"] = "dense"

    mo.ui.dataframe(bench_data)
    return (bench_data,)


@app.cell
def __(bench_data, mo, pd):
    sparse_npad_bench_data = bench_data[
        (bench_data["benchmark"] == "npad") & (bench_data["sparsity"] == "sparse")
    ]
    dense_npad_bench_data = bench_data[
        (bench_data["benchmark"] == "npad") & (bench_data["sparsity"] == "dense")
    ]
    qutip_bench_data = bench_data[bench_data["res"] == "qutip"]
    npad_bench_data = pd.concat(
        [
            sparse_npad_bench_data,
            dense_npad_bench_data,
            qutip_bench_data,
        ]
    )[
        [
            "res",
            "sparsity",
            "benchmark",
            "params.chain_size",
            "params.ntrunc",
            "stats.min",
            "stats.max",
            "stats.mean",
            "stats.stddev",
            "stats.median",
        ]
    ]
    npad_bench_data["ndim"] = (
        npad_bench_data["params.ntrunc"] ** npad_bench_data["params.chain_size"]
    )  # .astype(int)
    mo.ui.dataframe(npad_bench_data)
    return (
        dense_npad_bench_data,
        npad_bench_data,
        qutip_bench_data,
        sparse_npad_bench_data,
    )


@app.cell
def __(bench_data, mo):
    magnus_bench_data = (bench_data[bench_data["benchmark"] == "magnus"])[
        [
            "res",
            "params.chain_size",
            "params.num_intervals",
            "stats.min",
            "stats.max",
            "stats.mean",
            "stats.stddev",
            "stats.median",
        ]
    ]
    magnus_bench_data["ndim"] = (
        2 ** magnus_bench_data["params.chain_size"]
    )  # .astype(int)
    mo.ui.dataframe(magnus_bench_data)
    return (magnus_bench_data,)


@app.cell
def __():
    import json

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    mpl.rc("font", size=20)

    benchmark_files = mo.ui.file(
        kind="button",
        label="Benchmark files:",
        filetypes=[".json"],
        multiple=True,
    )
    return benchmark_files, json, mo, mpl, pd, plt, sns


if __name__ == "__main__":
    app.run()
