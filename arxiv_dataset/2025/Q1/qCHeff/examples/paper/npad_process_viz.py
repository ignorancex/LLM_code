import marimo

__generated_with = "0.7.5"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import numpy as np

    return (np,)


@app.cell
def __():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    return mpl, plt


@app.cell
def __():
    from qcheff.duffing.duffing_chain_utils import (
        create_linear_spectrum_zz_chain,
        duffing_chain_num_ham,
    )

    return create_linear_spectrum_zz_chain, duffing_chain_num_ham


@app.cell
def __(mpl, np, plt):
    def plot_matelm_mag(H: np.ndarray, scale="log", plot_ax=None):
        if plot_ax == None:
            plot_fig, plot_ax = plt.subplots(1, 1, layout="constrained")
        else:
            plot_fig = plot_ax.get_figure()
        absH = np.abs(H)
        P1 = plot_ax.matshow(
            absH,
            cmap="binary",
            norm=(mpl.colors.LogNorm(vmin=1e-10, vmax=1) if scale == "log" else None),
        )
        plot_ax.set(xticks=[], yticks=[])
        plt.colorbar(P1, ax=plot_ax)
        return plot_fig, plot_ax

    def plot_offdiag_mag(H: np.ndarray, scale="log", plot_ax=None):
        return plot_matelm_mag(
            H - np.diag(np.diagonal(H)), scale=scale, plot_ax=plot_ax
        )

    return plot_matelm_mag, plot_offdiag_mag


@app.cell
def __(create_linear_spectrum_zz_chain, duffing_chain_num_ham):
    test_ham = duffing_chain_num_ham(
        create_linear_spectrum_zz_chain(
            delta1=1.2,
            delta2=1.7,
            omega_res=5,
            num_resonators=3,
        )
    )
    return (test_ham,)


@app.cell
def __():
    from scipy import sparse

    return (sparse,)


@app.cell
def __():
    import scipy

    return (scipy,)


@app.cell
def __(
    find_largest_coupling,
    mo,
    np,
    npad_eliminate_coupling,
    sparse,
    test_form,
    test_ham,
):
    H_list = [test_ham]
    coupling_idx_list = []
    off_diag_norm_list = []
    diff_H_list = []
    sorted_eigenvalue_l2_norm_list = []
    true_spectrum = np.linalg.eigvalsh(test_ham.toarray())
    for i in mo.status.progress_bar(range(test_form.value["test_rot_number"])):
        current_ham = H_list[-1]
        coupling_idx = find_largest_coupling(current_ham)
        new_ham = npad_eliminate_coupling(current_ham, *coupling_idx)
        H_list.append(new_ham)
        change_ham = np.abs(new_ham) - np.abs(current_ham)
        off_diag_norm_list.append(
            sparse.linalg.norm(new_ham - sparse.diags(new_ham.diagonal()))
        )
        sorted_eigenvalue_l2_norm_list.append(
            np.linalg.norm(np.real(np.sort(new_ham.diagonal())) - true_spectrum)
        )
        diff_H_list.append(change_ham)
        coupling_idx_list.append(coupling_idx)
    return (
        H_list,
        change_ham,
        coupling_idx,
        coupling_idx_list,
        current_ham,
        diff_H_list,
        i,
        new_ham,
        off_diag_norm_list,
        sorted_eigenvalue_l2_norm_list,
        true_spectrum,
    )


@app.cell
def __(mo):
    test_form = mo.ui.batch(
        mo.md(
            """
            How many Givens rotations? {test_rot_number}. 

            Plot every {plot_every}
            """
        ),
        {
            "test_rot_number": mo.ui.number(start=2, stop=100000, step=1, value=200),
            "plot_every": mo.ui.number(start=1, stop=100000, step=1, value=20),
        },
    ).form()
    test_form
    return (test_form,)


@app.cell
def __(
    H_list,
    coupling_idx_list,
    diff_H_list,
    mo,
    mpl,
    np,
    off_diag_norm_list,
    plot_matelm_mag,
    plt,
    sorted_eigenvalue_l2_norm_list,
    test_form,
    test_ham,
):
    num_plot_cols = test_form.value["test_rot_number"] // test_form.value["plot_every"]

    fig, (top_ax, bottom_ax) = plt.subplots(
        2,
        num_plot_cols + 1,
        layout="constrained",
        figsize=(3 * num_plot_cols + 1, 4),
    )
    plot_matelm_mag(test_ham.toarray(), plot_ax=top_ax[0])
    top_ax[0].set(xticks=[], yticks=[], xlabel="Starting Matrix")
    bottom_ax[0].plot(off_diag_norm_list, label="Offdiag norm")
    bottom_ax[0].plot(sorted_eigenvalue_l2_norm_list, label="Spectrum Error")
    bottom_ax[0].set(xlabel="# Rotations", yscale="log")
    bottom_ax[0].legend()
    for index, (top_axis, bottom_axis, ham, coupling, change) in enumerate(
        zip(
            top_ax[1:],
            bottom_ax[1:],
            H_list[1 :: test_form.value["plot_every"]],
            coupling_idx_list[:: test_form.value["plot_every"]],
            diff_H_list[:: test_form.value["plot_every"]],
            strict=False,
        )
    ):
        counter = index * test_form.value["plot_every"]
        ham_plot = plot_matelm_mag(ham.toarray(), plot_ax=top_axis)
        bottom_axis.scatter(*coupling, marker="*", color="tab:red", s=10)
        top_axis.set(xticks=[], yticks=[], xlabel=f"Step {counter+1}")
        change_plot_lim = np.max(np.abs(change))
        change_plot = bottom_axis.matshow(
            change.toarray(),
            cmap="RdBu_r",
            norm=mpl.colors.SymLogNorm(
                linthresh=0.1,
                vmin=-change_plot_lim,
                vmax=change_plot_lim,
                base=10,
            ),
        )
        bottom_axis.set(
            xticks=[], yticks=[], xlabel=f"Rotation {counter+1}: {coupling}"
        )
        plt.colorbar(change_plot)

    mo.mpl.interactive(fig)
    return (
        bottom_ax,
        bottom_axis,
        change,
        change_plot,
        change_plot_lim,
        counter,
        coupling,
        fig,
        ham,
        ham_plot,
        index,
        num_plot_cols,
        top_ax,
        top_axis,
    )


@app.cell
def __():
    from qcheff.npad.sparse.npad_cpu import npad_eliminate_coupling
    from qcheff.npad.sparse.utils_cpu import find_largest_coupling

    return find_largest_coupling, npad_eliminate_coupling


if __name__ == "__main__":
    app.run()
