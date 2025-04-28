from dataclasses import asdict

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.utils import polyval_rows
from src.demo.paper.simulated.simulated import compose_dir, load_variable_reflectivity
from src.outputs.visualization import RcParamsOptions, SubplotsOptions, plot_custom, savefig_dir_list


def visualize_reflectivity(
        options: list[np.ndarray],
        experiment_name: str,
):
    rc_params = RcParamsOptions(fontsize=17)
    subplots_options = SubplotsOptions(figsize=(9, 4.4))
    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig_tr, axs_tr = plt.subplots(**asdict(subplots_options))
    wavenumbers = np.linspace(0.7, 2.9, int(1e4))
    for i_refl, reflectivity_coeffs in enumerate(options):
        reflectivity = polyval_rows(coefficients=reflectivity_coeffs, interval=wavenumbers)
        mathcal_r = r"$\mathcal{R}$"
        if reflectivity_coeffs.size == 1:
            label = f"{mathcal_r}={reflectivity_coeffs[0, 0]:.2f}"
        else:
            label = f"Variable {mathcal_r}"
        plot_custom(
            axs=axs_tr[0, 0],
            x_array=wavenumbers,
            array=reflectivity[0],
            label=label,
            color=f"C{i_refl}",
            linewidth=2,
            ylabel="Intensity",
            xlabel=r"Wavenumbers $\sigma$ [um$^{-1}$]",
            ylim=[-0.1, 0.9],
        )
        axs_tr[0, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')

    plt.show()

    savefig_dir_list(
        fig=fig_tr,
        filename="reflectivity.pdf",
        directories_list=[
            compose_dir(report_type="figures", experiment_name=experiment_name, save_dir_init="reports"),
            compose_dir(report_type="paper_figures", experiment_name=experiment_name, save_dir_init="paper"),
        ],
        subdirectory="",
        fmt="pdf",
        bbox_inches="tight",
    )


def main():
    experiment_name = "simulated/reflectivity_levels"
    options = [
        np.array([[0.2]]),
        np.array([[0.4]]),
        np.array([[0.7]]),
        load_variable_reflectivity()[2],
    ]

    visualize_reflectivity(
        options=options,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    main()
