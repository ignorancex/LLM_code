from dataclasses import asdict, replace

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.utils import convert_zero_to_infty_latex
from src.demo.paper.snr.utils import experiment_dir_convention, experiment_subdir_convention
from src.demo.paper.snr.visualize import experiment_figures_subdir_convention
from src.interface.configuration import load_config
from src.outputs.visualization import RcParamsOptions, SubplotsOptions, savefig_dir_list, plot_custom


def visualize_spectrum_compare(
        experiment_id: int,
        dataset_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
        acquisition_index: int,
        is_plot_show: bool = False,
):
    config = load_config()
    db = config.database()

    experiment_config = db.experiments[experiment_id]

    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"
    reconstruction_dir = experiment_dir_convention(dir_type="reconstruction", experiment_id=experiment_id)
    figures_dir_list = [
        experiment_dir_convention(dir_type="figures", experiment_id=experiment_id),
        experiment_dir_convention(dir_type="paper_figures", experiment_id=experiment_id, custom_dir=paper_dir),
    ]
    save_subdir = experiment_figures_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=-1,
        noise_level_index=noise_level_index,
        folder_name="spectrum_comparison",
    )

    spectra_ref_scaled = db.dataset_spectrum(ds_id=dataset_id).rescale(new_max=1, axis=-2)

    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig, axes = plt.subplots(**asdict(subplots_options))

    spectra_ref_scaled.visualize(
        axs=axes[0, 0],
        acq_ind=acquisition_index,
        label="Reference",
        color="C0",
        linewidth=3,
        title="",
        **plot_options,
    )

    for i_ifm, interferometer_id in enumerate(experiment_config.interferometer_ids):
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")

        load_subdir = experiment_subdir_convention(
            dataset_id=dataset_id,
            interferometer_id=interferometer_id,
            noise_level_index=noise_level_index,
            inversion_protocol_id=inversion_protocol_id,
        )

        spectra_rec = replace(
            spectra_ref_scaled,
            data=np.load(file=reconstruction_dir / load_subdir / "spectra_rec.npy"),
        )
        spectra_rec_matched, _ = spectra_rec.match_stats(reference=spectra_ref_scaled, axis=-2)

        # TODO: Visualization options either in database or in schema or in
        if experiment_id == 3:
            label = r"$\delta_{min}$=" + f"{db.interferometers[interferometer_id].opds.start:.2f}"
            title = (f"{db.interferometers[interferometer_id].type.value} "
                     f"model")
        elif experiment_id == 4 or experiment_id == 5:
            reflectance_coeffs = np.array(db.interferometers[interferometer_id].reflectance_coefficients)
            if reflectance_coeffs.size == 1:
                label = r"$\mathcal{R}$=" + f"{reflectance_coeffs[0, 0]:.2f}"
            else:
                label = r"Variant $\mathcal{R}$"
            title = (
                f"{db.interferometers[interferometer_id].type.value} "
                f"{convert_zero_to_infty_latex(db.interferometers[interferometer_id].order)}-wave "
                f"model"
            )
        else:
            label = f"Interferometer {interferometer_id}"
            title = f"model"

        spectra_rec_matched.visualize(
            axs=axes[0, 0],
            acq_ind=acquisition_index,
            label=label,
            color=f"C{i_ifm + 1}",
            linewidth=1.3,
            title=title,
            **plot_options,
        )

    filename = f"acquisition_{acquisition_index:03}.pdf"
    if is_plot_show:
        plt.show()
    savefig_dir_list(
        fig=fig,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )


def visualize_reflectance(
        experiment_id: int,
        dataset_id: int,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
        is_plot_show: bool = False,
):
    config = load_config()
    db = config.database()

    experiment_config = db.experiments[experiment_id]

    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"
    figures_dir_list = [
        experiment_dir_convention(dir_type="figures", experiment_id=experiment_id),
        experiment_dir_convention(dir_type="paper_figures", experiment_id=experiment_id, custom_dir=paper_dir),
    ]
    save_subdir = experiment_figures_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=-1,
        noise_level_index=-1,
        folder_name="reflectance_comparison",
    )

    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
    wavenumbers = spectra_ref.wavenumbers

    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig, axes = plt.subplots(**asdict(subplots_options))

    for i_ifm, interferometer_id in enumerate(experiment_config.interferometer_ids):
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")

        array = db.interferometer(interferometer_id=interferometer_id).reflectance(wavenumbers=wavenumbers)[0]

        # TODO: Visualization options either in database or in schema or in
        if experiment_id == 3:
            label = r"$\delta_{min}$=" + f"{db.interferometers[interferometer_id].opds.start:.2f}"
            title = (f"{db.interferometers[interferometer_id].type.value} "
                     f"model")
        elif experiment_id == 4 or experiment_id == 5:
            reflectance_coeffs = np.array(db.interferometers[interferometer_id].reflectance_coefficients)
            if reflectance_coeffs.size == 1:
                label = r"$\mathcal{R}$=" + f"{reflectance_coeffs[0, 0]:.2f}"
            else:
                label = r"Variant $\mathcal{R}$"
            title = (
                f"{db.interferometers[interferometer_id].type.value} "
                f"{convert_zero_to_infty_latex(db.interferometers[interferometer_id].order)}-wave "
                f"model"
            )
        else:
            label = f"Interferometer {interferometer_id}"
            title = f"model"

        plot_custom(
            axs=axes[0, 0],
            x_array=wavenumbers,
            array=array,
            label=label,
            color=f"C{i_ifm + 1}",
            linewidth=2,
            title=title,
            ylabel="Intensity",
            xlabel=rf"Wavenumbers $\sigma$ [{spectra_ref.wavenumbers_unit}]",
            **plot_options,
        )

    filename = f"reflectance.pdf"
    if is_plot_show:
        plt.show()
    savefig_dir_list(
        fig=fig,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )


def visualize_one_experiment(
        experiment_id: int,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
        acquisition_indices: list[int],
        is_plot_show: bool = False,
):
    db = load_config().database()

    experiment_config = db.experiments[experiment_id]
    for i_ds, dataset_id in enumerate(experiment_config.dataset_ids):
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")

        for noise_level_index in experiment_config.noise_level_indices:
            print(f"\t\tNoise Level: {db.noise_levels[noise_level_index]:.0f}")

            inversion_protocol_id = experiment_config.inversion_protocol_ids[0]

            visualize_spectrum_compare(
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                noise_level_index=noise_level_index,
                inversion_protocol_id=inversion_protocol_id,
                rc_params=rc_params,
                subplots_options=subplots_options,
                plot_options=plot_options,
                acquisition_index=acquisition_indices[i_ds],
                is_plot_show=is_plot_show,
            )

            visualize_reflectance(
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                rc_params=rc_params,
                subplots_options=subplots_options,
                plot_options=plot_options,
                is_plot_show=is_plot_show,
            )


def visualize_list_experiments(
        experiment_ids: list,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
        acquisition_indices: list[int],
        is_plot_show: bool = False,
):
    for experiment_id in experiment_ids:
        visualize_one_experiment(
            experiment_id=experiment_id,
            rc_params=rc_params,
            subplots_options=subplots_options,
            plot_options=plot_options,
            acquisition_indices=acquisition_indices,
            is_plot_show=is_plot_show,
        )


def main():
    experiment_ids = [3, 4, 5, 6]
    rc_params = RcParamsOptions(fontsize=17)
    subplots_options = SubplotsOptions()
    plot_options = {"ylim": [-0.2, 1.4]}
    acquisition_indices = [0, 13, 13]
    is_plot_show = True
    visualize_list_experiments(
        experiment_ids=experiment_ids,
        rc_params=rc_params,
        subplots_options=subplots_options,
        plot_options=plot_options,
        acquisition_indices=acquisition_indices,
        is_plot_show=is_plot_show,
    )


if __name__ == "__main__":
    main()
