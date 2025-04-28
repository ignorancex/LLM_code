import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    from qcheff.jaynes_cummings_hubbard.models import (
        JCHModel,
        jch_scqubits_hilbertspace,
    )
    from qcheff.jaynes_cummings_hubbard.utils import JCHAnalysis

    return JCHAnalysis, JCHModel, jch_scqubits_hilbertspace


@app.cell
def __(mo):
    jch_param_form = mo.ui.batch(
        mo.md(
            r"""
            # Jaynes-Cummings-Hubbard Model Parameters

            Chain size $N$: {n}

            Qubit frequency range $\omega_q$ : {wq_range} GHz

            Resonator detuning $\Delta = \omega_r - \omega_q$ : {delta} GHz, {nr} levels, {ndelta} values.

            Interaction strength $g$ : {g} GHz

            Cavity Tunneling Rate $\kappa$ : {kappa} GHz
            """
        ),
        {
            "wq_range": mo.ui.range_slider(
                start=5,
                stop=10,
                step=0.1,
                value=(5, 7),
                show_value=True,
            ),
            "delta": mo.ui.range_slider(
                start=1.9,
                stop=2.1,
                step=0.01,
                value=[1.98, 2.01],
                show_value=True,
            ),
            "ndelta": mo.ui.number(
                start=2,
                stop=1000,
                step=1,
                value=50,
            ),
            "g": mo.ui.number(
                start=0.0,
                stop=10.0,
                step=0.001,
                value=0.105,
            ),
            "kappa": mo.ui.number(
                start=0.0,
                stop=10.0,
                step=0.01,
                value=0.01,
            ),
            "nr": mo.ui.number(
                start=2,
                stop=20,
                step=1,
                value=5,
            ),
            "n": mo.ui.number(
                start=1,
                stop=20,
                step=1,
                value=2,
            ),
        },
    ).form()
    jch_param_form
    return (jch_param_form,)


@app.cell
def __(JCHModel, copy, np):
    def models_gen(form_data: dict):
        model_params_in_form = ["wq_range", "g", "kappa", "nr", "n"]
        delta_range = np.linspace(*form_data["delta"], form_data["ndelta"])
        model_template = copy(form_data)
        for delta in delta_range.flat:
            model_param_dict = {
                key: value
                for key, value in form_data.items()
                if key in model_params_in_form
            }
            yield JCHModel(**(model_param_dict | {"delta": float(delta)}))

    return (models_gen,)


@app.cell
def __(itertools, repeat, test_model):
    test_levels = list(
        map(
            tuple,
            map(
                itertools.chain,
                itertools.product([0, 1], repeat=test_model.n),
                repeat(tuple([0] * test_model.n)),
            ),
        )
    )
    "\n".join(map(lambda x: " ,".join(map(str, x)), test_levels))
    # print(test_levels)
    return (test_levels,)


@app.cell
def __(jch_param_form, models_gen):
    test_model = next(models_gen(jch_param_form.value))
    return (test_model,)


@app.cell
def __(jch_scqubits_hilbertspace, test_model):
    test_hs = jch_scqubits_hilbertspace(test_model)
    return (test_hs,)


@app.cell
def __():
    import itertools

    return (itertools,)


@app.cell
def __():
    import scqubits as scq

    return (scq,)


@app.cell
def __(test_hs):
    test_ham = test_hs.hamiltonian()[:]
    return (test_ham,)


@app.cell
def __():
    import more_itertools

    return (more_itertools,)


@app.cell
def __(mo):
    mo.md("""# Testing accuracy""")


@app.cell
def __(
    JCHAnalysis,
    itertools,
    jch_param_form,
    mo,
    models_gen,
    pl,
    test_levels,
):
    data_df = pl.concat(
        mo.status.progress_bar(
            itertools.chain(
                *(
                    JCHAnalysis(model=model, level_labels=test_levels).analyse(
                        batch_size=5,
                        # methods=["scqubits", "npad_cpu"],
                        max_iter=200,
                    )
                    for model in list(models_gen(jch_param_form.value))
                )
            ),
            total=jch_param_form.value["ndelta"] * 3,
            title="Analyzing JCH System",
            subtitle="Method: ",
        )
    )
    data_df
    return (data_df,)


@app.cell
def __(data_df, pl):
    zz_df = (
        data_df.pivot(
            on="state",
            values="energy",
        )
        .with_columns(
            (pl.col("1100") + pl.col("0000") - pl.col("1000") - pl.col("0100")).alias(
                "ZZ"
            )
        )
        .pivot(
            on="Method",
            values="ZZ",
            index=["delta", "g", "kappa"],
        )
    ).rename({"npad_cpu": "NPAD CPU", "npad_gpu": "NPAD GPU", "scqubits": "scQubits"})
    zz_error_df = zz_df.with_columns(
        ((pl.col("NPAD CPU") - pl.col("scQubits")) / pl.col("scQubits"))
        .abs()
        .fill_nan(0)  # in case of divide-by-zero
        .alias("NPAD CPU Error"),
        ((pl.col("NPAD GPU") - pl.col("scQubits")) / pl.col("scQubits"))
        .abs()
        .fill_nan(0)  # in case of divide-by-zero
        .alias("NPAD GPU Error"),
    )

    zz_df
    return zz_df, zz_error_df


@app.cell
def __(pl, plt, sns, zz_df, zz_error_df):
    with sns.plotting_context("paper"):
        zz_fig, zz_ax = plt.subplots(2, 1, sharex=True, figsize=(4, 7))
        zz_plot = sns.lineplot(
            data=zz_df.select(pl.all().exclude(["g", "kappa"]))
            .unpivot(
                index="delta",
                variable_name="Method",
                value_name="zz",
            )
            .select(pl.col("delta"), pl.col("Method"), pl.col("zz") * 1e6),
            x="delta",
            y="zz",
            hue="Method",
            markers=True,
            palette={
                "scQubits": "slategray",
                "NPAD CPU": (0 / 255, 104 / 255, 181 / 255),
                "NPAD GPU": (118 / 255, 185 / 255, 0 / 255),
            },
            style="Method",
            ax=zz_ax[0],
        )
        zz_plot.set(xlim=(1.97, 2.01), ylabel="ZZ (kHz)")
        zz_err_plot = sns.lineplot(
            data=zz_error_df.select(
                [
                    "delta",
                    "NPAD CPU Error",
                    "NPAD GPU Error",
                ]
            ).unpivot(
                index="delta",
                variable_name="Method",
                value_name="error",
            ),
            x="delta",
            y="error",
            hue="Method",
            style="Method",
            palette={
                "scQubits": "slategray",
                "NPAD CPU Error": (0 / 255, 104 / 255, 181 / 255),
                "NPAD GPU Error": (118 / 255, 185 / 255, 0 / 255),
            },
            markers=True,
            alpha=0.6,
            ax=zz_ax[1],
        )
        zz_err_plot.set(
            ylim=(1e-10, 1),
            yscale="log",
            xlabel=r"$\delta$ (GHz)",
            ylabel="Relative ZZ error",
        )
        sns.despine(fig=zz_fig)
    zz_fig
    return zz_ax, zz_err_plot, zz_fig, zz_plot


@app.cell
def __():
    from copy import copy
    from itertools import product
    from time import sleep

    import cupy as cp
    import cupyx
    import cupyx.scipy.sparse as cpsparse
    import cupyx.scipy.sparse.linalg as cpla
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px
    import polars as pl
    import qutip as qt
    import scipy.sparse as spsparse
    import seaborn as sns
    from tqdm.notebook import tqdm

    from qcheff.iswt.iswt import NPAD
    from qcheff.operators.operator_base import qcheffOperator

    return (
        NPAD,
        copy,
        cp,
        cpla,
        cpsparse,
        cupyx,
        mo,
        mpl,
        np,
        pl,
        plt,
        product,
        px,
        qcheffOperator,
        qt,
        sleep,
        sns,
        spsparse,
        tqdm,
    )


if __name__ == "__main__":
    app.run()
