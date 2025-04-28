import marimo

__generated_with = "0.7.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    from qcheff.jaynes_cummings import (
        jc_energies,
        jc_num_ham,
        jc_qutip_ham,
        jc_scq_paramsweep,
        npad_rabi_splitting,
        qt_rabi_splitting,
        qutip_overlap_find_evals,
        scq_rabi_splitting,
        theory_rabi_splitting,
    )

    return (
        jc_energies,
        jc_num_ham,
        jc_qutip_ham,
        jc_scq_paramsweep,
        npad_rabi_splitting,
        qt_rabi_splitting,
        qutip_overlap_find_evals,
        scq_rabi_splitting,
        theory_rabi_splitting,
    )


@app.cell
def __():
    import qutip as qt

    return (qt,)


@app.cell
def __(
    delta_arr,
    jc_num_ham,
    jc_qutip_ham,
    models_gen,
    np,
    npad_diagonalize,
):
    test_model = next(models_gen(delta_arr))
    (
        (lambda x: x - x.min())(jc_qutip_ham(test_model).eigenenergies()),
        np.sort(
            np.real(
                (lambda x: x - x.min())(
                    npad_diagonalize(jc_num_ham(test_model), max_rots=4).diagonal()
                )
            )
        ),
    )
    return (test_model,)


@app.cell
def __():
    from qcheff.npad.sparse.npad_cpu import npad_diagonalize

    return (npad_diagonalize,)


@app.cell
def __(delta_arr, jc_num_ham, models_gen, np, npad_diagonalize, plt, qt):
    plt.plot(
        delta_arr,
        np.sort(
            [
                np.real(npad_diagonalize(jc_num_ham(model)).diagonal()) / model.g
                for model in models_gen(delta_arr)
            ]
        ),
        label=[str(lab) for lab in qt.state_number_enumerate([2, 5])],
    )
    plt.legend()


@app.cell
def __():
    import scqubits as scq

    return (scq,)


@app.cell
def __(jc_param_form):
    jc_param_form


@app.cell
def __():
    from qcheff.jaynes_cummings import models_gen

    return (models_gen,)


@app.cell
def __(fig, mo):
    mo.mpl.interactive(fig)


@app.cell
def __(
    jc_param_form,
    models_gen,
    np,
    npad_rabi_splitting,
    qt_rabi_splitting,
    theory_rabi_splitting,
):
    delta_min, delta_max = jc_param_form.value["delta"]
    num_delta = jc_param_form.value["num_delta"]
    delta_arr = np.linspace(delta_min, delta_max, num_delta)

    nlist = range(1, 5)

    th_rabi_amps = np.array(
        [
            [theory_rabi_splitting(model, n=n) for model in models_gen(delta_arr)]
            for n in nlist
        ]
    ).T
    # scq_rabi_amps = np.array(
    #     [
    #         [scq_rabi_splitting(model, n=n) for model in models_gen(delta_arr)]
    #         for n in nlist
    #     ]
    # ).T
    qt_rabi_amps = np.array(
        [
            [qt_rabi_splitting(model, n=n) for model in models_gen(delta_arr)]
            for n in nlist
        ]
    ).T

    npad_rabi_amps = np.array(
        [
            [npad_rabi_splitting(model, n=n) for model in models_gen(delta_arr)]
            for n in nlist
        ]
    ).T
    return (
        delta_arr,
        delta_max,
        delta_min,
        nlist,
        npad_rabi_amps,
        num_delta,
        qt_rabi_amps,
        th_rabi_amps,
    )


@app.cell
def __(delta_arr, nlist, npad_rabi_amps, plt, qt_rabi_amps, th_rabi_amps):
    (
        fig,
        (
            th_ax,
            # scq_ax,
            qt_ax,
            # npad_ax,
        ),
    ) = plt.subplots(
        1,
        2,
        layout="constrained",
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )

    axs = (
        th_ax,
        # scq_ax,
        qt_ax,
        # npad_ax,
    )
    rabi_amps = (
        th_rabi_amps,
        # scq_rabi_amps,
        qt_rabi_amps,
        # npad_rabi_amps,
    )

    # for ax, rabi_amp in zip(axs, rabi_amps):
    #     ax.plot(
    #         delta_arr,
    #         rabi_amp,
    #         ".-",
    #         label=list(map(lambda x: f"n={x}", range(1, 4))),
    #     )
    #     ax.set(
    #         xlabel=r"$\Delta/g$",
    #         ylabel=r"Rabi Splitting $\Omega_n/g$",
    #         title="Theory",
    #         xlim=(delta_arr.min(), delta_arr.max()),
    #     )
    #     ax.legend()
    plot_labels = list(map(lambda x: f"n={x}", nlist))
    th_ax.plot(
        delta_arr,
        qt_rabi_amps[:, [3, 1]] - (qt_rabi_amps)[:, 1][:, None],
        ".-",
        # label=plot_labels,
    )
    th_ax.set(
        xlabel=r"$\Delta/g$",
        ylabel=r"Rabi Splitting $\Omega_n/g$",
        title="Theory",
        xlim=(delta_arr.min(), delta_arr.max()),
    )
    th_ax.legend()
    # scq_ax.plot(
    #     delta_arr,
    #     th_rabi_amps,
    #     ".-",
    #     label=plot_labels,
    # )
    # scq_ax.set(
    #     xlabel=r"$\Delta/g$",
    #     title="SCQubits",
    #     xlim=(delta_arr.min(), delta_arr.max()),
    # )
    # scq_ax.legend()
    qt_ax.plot(
        delta_arr,
        npad_rabi_amps[:, [1, 3]] - npad_rabi_amps[:, 3][:, None],
        ".-",
        # label=plot_labels,
    )
    qt_ax.set(
        xlabel=r"$\Delta/g$",
        title="NPAD",
        xlim=(delta_arr.min(), delta_arr.max()),
    )
    qt_ax.legend()
    # npad_ax.plot(
    #     delta_arr,
    #     npad_rabi_amps,
    #     ".-",
    #     label=plot_labels,
    # )
    # npad_ax.set(
    #     xlabel=r"$\Delta/g$",
    #     title="NPAD",
    #     xlim=(delta_arr.min(), delta_arr.max()),
    # )
    # npad_ax.legend()
    None
    return axs, fig, plot_labels, qt_ax, rabi_amps, th_ax


@app.cell
def __():
    import numpy as np

    return (np,)


@app.cell
def __():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def __(mo):
    jc_param_form = mo.ui.batch(
        mo.md(
            r"""
            # Jaynes-Cummings Model Parameters

            Qubit frequency $\omega_q$ : {wq} GHz

            Resonator detuning $\Delta = \omega_r - \omega_q$ : {delta} GHz, {num_delta} values.

            Interaction strength $g$ : {g} GHz
            """
        ),
        {
            "wq": mo.ui.number(
                start=5,
                stop=10,
                step=0.1,
                value=5,
            ),
            "delta": mo.ui.range_slider(
                start=-5,
                stop=5,
                step=0.1,
                value=[-1, 1],
                show_value=True,
            ),
            "num_delta": mo.ui.number(
                start=10,
                stop=1000,
                step=1,
                value=51,
            ),
            "g": mo.ui.number(
                start=0.0,
                stop=10.0,
                step=0.01,
                value=0.1,
            ),
        },
    ).form()
    return (jc_param_form,)


@app.cell
def __():
    from itertools import product

    return (product,)


if __name__ == "__main__":
    app.run()
