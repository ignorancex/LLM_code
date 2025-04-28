import marimo

__generated_with = "0.7.0"
app = marimo.App(width="medium", app_title="Transmon Chain ZZ")


@app.cell
def __(system_params_form):
    system_params_form


@app.cell
def __(mo, num_resonators, product):
    zz_level_labels = [
        tuple(l + ([0] * num_resonators) + r) for l, r in product([[0], [1]], repeat=2)
    ]
    mo.md(f" ZZ levels: {zz_level_labels}")
    return (zz_level_labels,)


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(
    d1,
    d2,
    mo,
    npad_auto_find_evals,
    npad_diag_find_evals,
    num_resonators,
    qutip_overlap_find_evals,
    scq_auto_find_evals,
    scq_overlap_find_evals,
    wr,
    zz_level_labels,
):
    zz_args = {
        "level_labels": zz_level_labels,
        "del1": d1,
        "del2": d2,
        "wr": wr,
        "nr": num_resonators,
        "debug": True,
    }
    mo.md(
        f"""
        # Comparing different ZZ calculation methods
        |Method | ZZ interaction ( GHz)| 
        | ------|----------------|
        |SCQubits-auto| {scq_auto_find_evals(**zz_args)}|
        |SCQubits-Hamiltonian-overlap| {scq_overlap_find_evals(**zz_args)} |
        |QuTiP-Hamiltonian-Overlap| {qutip_overlap_find_evals(**zz_args)} |
        |NPAD-auto| {npad_auto_find_evals(**zz_args)} |
        |NPAD-Diag| {npad_diag_find_evals(**zz_args)} | 
        """
    )
    return (zz_args,)


@app.cell
def __(zz_sweep_form):
    zz_sweep_form


@app.cell
def __(np, zz_sweep_form):
    d1_arr = np.linspace(
        *zz_sweep_form.value["d1_range"],
        zz_sweep_form.value["d1_pts"],
    )
    return (d1_arr,)


@app.cell
def __(
    d2,
    mo,
    npad_auto_find_evals,
    npad_diag_find_evals,
    num_resonators,
    pl,
    qutip_overlap_find_evals,
    scq_auto_find_evals,
    scq_overlap_find_evals,
    wr,
    zz_level_labels,
):
    def do_evals_sweep(d1_arr):
        sweep_arglist = [
            {
                "level_labels": zz_level_labels,
                "del1": d1_temp,
                "del2": d2,
                "wr": wr,
                "nr": num_resonators,
                "debug": False,
            }
            for d1_temp in d1_arr
        ]
        method_funcs = [
            scq_auto_find_evals,
            scq_overlap_find_evals,
            qutip_overlap_find_evals,
            npad_auto_find_evals,
            npad_diag_find_evals,
        ]
        method_names = [
            method.__name__.removesuffix("_find_evals") for method in method_funcs
        ]

        evals_sweeps = [
            {"del1": args["del1"], "method": method_name}
            | dict(
                zip(
                    ["E_" + "".join(map(str, labels)) for labels in zz_level_labels],
                    method(**args),
                    strict=False,
                )
            )
            for args in mo.status.progress_bar(sweep_arglist, title="Configurations: ")
            for method, method_name in zip(method_funcs, method_names, strict=False)
        ]

        return pl.json_normalize(evals_sweeps)

    return (do_evals_sweep,)


@app.cell
def __(evals_df, mo, px):
    mo.ui.plotly(
        px.line(
            data_frame=evals_df.unpivot(
                index=["del1", "method"],
                value_name="energy",
                variable_name="level",
            ),
            x="del1",
            y="energy",
            color="method",
            template="simple_white",
            symbol="level",
            # opacity=0.8,
        )
    )


@app.cell
def __(mo, px, zz_df):
    mo.ui.plotly(
        px.line(
            zz_df,
            x="del1",
            y="zz",
            template="simple_white",
            color="method",
            labels={"zz": "ZZ (GHz)"},
        )
    )


@app.cell
def __(evals_df, pl):
    zz_df = evals_df.select(
        pl.col("del1"),
        pl.col("method"),
        (pl.col("E_101") + pl.col("E_000") - pl.col("E_001") - pl.col("E_100")).alias(
            "zz"
        ),
    )
    zz_df
    return (zz_df,)


@app.cell
def __(mo, px, zz_err_df):
    mo.ui.plotly(
        px.scatter(
            data_frame=zz_err_df,
            x="del1",
            y="error",
            color="method",
            template="simple_white",
            log_y=True,
            trendline="rolling",
            labels={"error": "Relative error"},
            trendline_options=dict(window=5),
        )
    )


@app.cell
def __(pl, zz_df):
    zz_err_df = (
        zz_df.pivot(index="del1", on="method")
        .select(
            pl.col("del1"),
            ((pl.col("scq_overlap") - pl.col("scq_auto")) / pl.col("scq_auto"))
            .abs()
            .alias("scq_overlap"),
            ((pl.col("qutip_overlap") - pl.col("scq_auto")) / pl.col("scq_auto"))
            .abs()
            .alias("qutip_overlap"),
            ((pl.col("npad_auto") - pl.col("scq_auto")) / pl.col("scq_auto"))
            .abs()
            .alias("npad_auto"),
            ((pl.col("npad_diag") - pl.col("scq_auto")) / pl.col("scq_auto"))
            .abs()
            .alias("npad_diag"),
        )
        .unpivot(index="del1", variable_name="method", value_name="error")
    ).filter(pl.col("error").is_not_nan())
    zz_err_df
    return (zz_err_df,)


@app.cell
def __(d1_arr, do_evals_sweep):
    evals_df = do_evals_sweep(d1_arr)
    evals_df
    return (evals_df,)


@app.cell
def __(system_params_form):
    d1 = system_params_form.value["d1_num"]
    d2 = system_params_form.value["d2_num"]
    a1 = system_params_form.value["a1_num"]
    a2 = system_params_form.value["a2_num"]
    wr = system_params_form.value["wr_num"]
    dwr = system_params_form.value["dwr_num"]
    num_resonators = int(system_params_form.value["nr_num"])
    global_ntrunc = int(system_params_form.value["ntrunc_num"])
    system_dims = [global_ntrunc] * (num_resonators + 2)
    return (
        a1,
        a2,
        d1,
        d2,
        dwr,
        global_ntrunc,
        num_resonators,
        system_dims,
        wr,
    )


@app.cell
def __():
    import numpy as np

    return (np,)


@app.cell
def __():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def __(mpl):
    mpl.rcParams.update({"font.size": 22})
    mpl.rcParams["lines.linewidth"] = 2


@app.cell
def __():
    import qutip as qt

    return (qt,)


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import matplotlib as mpl

    return (mpl,)


@app.cell
def __():
    from itertools import product

    return (product,)


@app.cell
def __():
    import scqubits as scq

    return (scq,)


@app.cell
def __():
    from qcheff.duffing.duffing_chain_utils import (
        create_linear_spectrum_zz_chain,
        duffing_chain_qutip_ham,
        duffing_chain_scq_hilbertspace,
    )

    return (
        create_linear_spectrum_zz_chain,
        duffing_chain_qutip_ham,
        duffing_chain_scq_hilbertspace,
    )


@app.cell
def __():
    from methods import (
        npad_auto_find_evals,
        npad_diag_find_evals,
        qutip_overlap_find_evals,
        scq_auto_find_evals,
        scq_overlap_find_evals,
    )

    return (
        npad_auto_find_evals,
        npad_diag_find_evals,
        qutip_overlap_find_evals,
        scq_auto_find_evals,
        scq_overlap_find_evals,
    )


@app.cell
def __():
    from itertools import pairwise

    return (pairwise,)


@app.cell
def __(mo):
    zz_sweep_form = mo.ui.batch(
        mo.md(
            r""" ## ZZ Parameter Sweep

            Range of \(\Delta_1\) {d1_range} GHz, {d1_pts} points 

            """
        ),
        {
            "d1_range": mo.ui.range_slider(
                start=-1.0,
                stop=5.0,
                step=0.01,
                value=[-0.5, 0.5],
                show_value=True,
                # full_width=True,
            ),
            "d1_pts": mo.ui.number(
                start=1,
                stop=100,
                step=1,
                value=10,
            ),
        },
    ).form()
    return (zz_sweep_form,)


@app.cell
def __(mo):
    system_params_form = mo.ui.batch(
        mo.md(
            r"""
        # System parameters

        ## Qubit parameters

        \(\Delta_1=\) {d1_num} GHz             \(\alpha_1=\) {a1_num} GHz

        \(\Delta_2=\) {d2_num} GHz             \(\alpha_2=\) {a2_num} GHz

        ## Resonator parameters

        \(\omega_r=\) {wr_num} GHz 

        \(\delta\omega_r=\) {dwr_num} GHz  

        \(n_r=\) {nr_num}

        ## Other

        Global truncation level: {ntrunc_num}
        """
        ),
        {
            "d1_num": mo.ui.slider(
                start=-4.0,
                stop=12.0,
                step=0.01,
                value=1.42,
                show_value=True,
            ),
            "a1_num": mo.ui.slider(
                start=-0.5,
                stop=0.5,
                step=0.01,
                value=-0.3,
                show_value=True,
            ),
            "d2_num": mo.ui.slider(
                start=-4.0,
                stop=12.0,
                step=0.01,
                value=0.38,
                show_value=True,
            ),
            "a2_num": mo.ui.slider(
                start=-0.5,
                stop=0.5,
                step=0.01,
                value=-0.3,
                show_value=True,
            ),
            "wr_num": mo.ui.slider(
                start=2,
                stop=20.0,
                step=0.01,
                value=5,
                show_value=True,
            ),
            "dwr_num": mo.ui.slider(
                start=-10,
                stop=10.0,
                step=0.01,
                value=0.6,
                show_value=True,
            ),
            "nr_num": mo.ui.number(
                start=1,
                stop=5,
                step=2,
                value=1,
            ),
            "ntrunc_num": mo.ui.number(
                start=3,
                stop=10,
                step=1,
                value=3,
            ),
        },
    ).form(
        bordered=False,
    )
    return (system_params_form,)


@app.cell
def __(
    create_linear_spectrum_zz_chain,
    d1,
    d2,
    duffing_chain_scq_hilbertspace,
    num_resonators,
    wr,
):
    test_system = create_linear_spectrum_zz_chain(
        delta1=d1,
        delta2=d2,
        omega_res=wr,
        num_resonators=num_resonators,
    )
    test_hs = duffing_chain_scq_hilbertspace(test_system)

    # test_hs
    return test_hs, test_system


@app.cell
def __():
    import polars as pl

    return (pl,)


@app.cell
def __():
    import plotly.express as px

    return (px,)


if __name__ == "__main__":
    app.run()
