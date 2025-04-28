from dataclasses import asdict, replace
from enum import Enum
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import match_stats
from src.demo.paper.monochromatic.utils import visualize_matching_central_wavenumbers
from src.direct_model.interferometer import simulate_interferogram
from src.interface.configuration import load_config
from src.outputs.visualization import RcParamsOptions, SubplotsOptions, savefig_dir_list


def visualize_datasets(
        experiment_id: int,
        subplots_options: SubplotsOptions,
        rc_params: RcParamsOptions,
        imshow_options: dict,
):
    config = load_config()
    reports_dir = config.directory_paths.reports
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    figures_dir_list = [
        reports_dir / f"experiment_{experiment_id}" / "figures",
        paper_dir / "figures" / f"{experiment_config.type}",
    ]

    for ds_id in experiment_config.dataset_ids:
        dataset = db.dataset_interferogram(ds_id=ds_id)
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"
        for char_id in experiment_config.interferometer_ids:
            characterization_subdir = f"{dataset_subdir}/{db.characterizations[char_id].title}"
            save_subdir = f"{characterization_subdir}/dataset"

            plt.rcParams['font.size'] = str(rc_params.fontsize)

            fig, axes = plt.subplots(**asdict(subplots_options))
            dataset.visualize_matrix(fig=fig, axs=axes[0, 0], vmax=0.4*np.max(dataset.data), **imshow_options)
            filename = "dataset.pdf"
            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=save_subdir,
            )


def visualize_transfer_matrices(
        experiment_id: int,
        subplots_options: SubplotsOptions,
        rc_params: RcParamsOptions,
        transmat_imshow_options: dict,
        singvals_plot_options: dict,
        opd_response_plot_options: dict,
        dct_opd_plot_options: dict,
        opd_idx: int = 0,
):
    config = load_config()
    reports_dir = config.directory_paths.reports
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    figures_dir_list = [
        reports_dir / f"experiment_{experiment_id}" / "figures",
        paper_dir / "figures" / f"{experiment_config.type}",
    ]

    for ds_id in experiment_config.dataset_ids:
        dataset_wavenumbers = db.dataset_central_wavenumbers(dataset_id=ds_id)
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"
        for char_id in experiment_config.interferometer_ids:
            characterization = db.characterization(characterization_id=char_id)
            characterization_subdir = f"{dataset_subdir}/{db.characterizations[char_id].title}"
            transfer_matrix = characterization.transmittance_response(wavenumbers=dataset_wavenumbers)
            transfer_matrix_subdir = f"{characterization_subdir}/transfer_matrix"

            plt.rcParams['font.size'] = str(rc_params.fontsize)

            fig, axes = plt.subplots(**asdict(subplots_options))
            transfer_matrix.visualize(fig=fig, axs=axes[0, 0], **transmat_imshow_options)
            filename = "transfer_matrix.pdf"
            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=transfer_matrix_subdir,
            )

            fig, axes = plt.subplots(**asdict(subplots_options))
            transfer_matrix.visualize_singular_values(axs=axes[0, 0], **singvals_plot_options)
            filename = "singular_values.pdf"
            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=transfer_matrix_subdir,
            )

            fig, axes = plt.subplots(**asdict(subplots_options))
            transfer_matrix.visualize_opd_response(axs=axes[0, 0], opd_idx=opd_idx, **opd_response_plot_options)
            filename = f"opd_response_idx_{opd_idx}.pdf"
            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=transfer_matrix_subdir,
            )

            fig, axes = plt.subplots(**asdict(subplots_options))
            transfer_matrix.visualize_dct(axs=axes[0, 0], opd_idx=opd_idx, **dct_opd_plot_options)
            filename = f"dct_opd_idx_{opd_idx}.pdf"
            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=transfer_matrix_subdir,
            )


def visualize_interferogram_matrices(
        experiment_id: int,
        subplots_options: SubplotsOptions,
        rc_params: RcParamsOptions,
        imshow_options: dict,
):
    config = load_config()
    reports_dir = config.directory_paths.reports
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    figures_dir_list = [
        reports_dir / f"experiment_{experiment_id}" / "figures",
        paper_dir / "figures" / f"{experiment_config.type}",
    ]
    reconstruction_dir = reports_dir / f"experiment_{experiment_id}" / "reconstruction"
    metrics_dir = reports_dir / f"experiment_{experiment_id}" / "metrics"

    for ds_id in experiment_config.dataset_ids:
        interferograms_ref = db.dataset_interferogram(ds_id=ds_id).rescale()
        central_wavenumbers = db.dataset_central_wavenumbers(dataset_id=ds_id)
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"
        for char_id in experiment_config.interferometer_ids:
            characterization = db.characterization(characterization_id=char_id)
            characterization_subdir = f"{dataset_subdir}/{db.characterizations[char_id].title}"
            save_subdir = f"{characterization_subdir}/interferogram_matrices"
            transfer_matrix = characterization.transmittance_response(wavenumbers=central_wavenumbers)

            # Save the reference
            plt.rcParams['font.size'] = str(rc_params.fontsize)

            fig, axes = plt.subplots(**asdict(subplots_options))
            interferograms_ref.visualize_matrix(fig=fig, axs=axes[0, 0], **imshow_options)
            filename = f"reference.pdf"
            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=save_subdir,
            )

            for ip_id in experiment_config.inversion_protocol_ids:
                inverter_subdir = f"{characterization_subdir}/{db.inversion_protocols[ip_id].title}"
                spectra_rec_all = np.load(file=reconstruction_dir / inverter_subdir / "spectra_rec_all.npy")
                rmse_diagonal = np.load(file=metrics_dir / inverter_subdir / "rmse_diagonal.npy")
                best_idx = np.argmin(rmse_diagonal)
                spectra_rec = Spectrum(data=spectra_rec_all[best_idx], wavenumbers=central_wavenumbers)
                interferograms_rec = simulate_interferogram(
                    transmittance_response=transfer_matrix,
                    spectrum=spectra_rec,
                )
                interferograms_rec_matched, _ = interferograms_rec.match_stats(
                    reference=interferograms_ref,
                    axis=-2,
                )

                # Save the reconstruction
                plt.rcParams['font.size'] = str(rc_params.fontsize)

                fig, axes = plt.subplots(**asdict(subplots_options))
                interferograms_rec_matched.visualize_matrix(fig=fig, axs=axes[0, 0], **imshow_options)
                filename = f"{db.inversion_protocols[ip_id].title}.pdf"
                savefig_dir_list(
                    fig=fig,
                    filename=filename,
                    directories_list=figures_dir_list,
                    subdirectory=save_subdir,
                )


def visualize_spectrum_matrices(
        experiment_id: int,
        subplots_options: SubplotsOptions,
        rc_params: RcParamsOptions,
        imshow_options: dict,
):
    config = load_config()
    reports_dir = config.directory_paths.reports
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    figures_dir_list = [
        reports_dir / f"experiment_{experiment_id}" / "figures",
        paper_dir / "figures" / f"{experiment_config.type}",
    ]
    reconstruction_dir = reports_dir / f"experiment_{experiment_id}" / "reconstruction"
    metrics_dir = reports_dir / f"experiment_{experiment_id}" / "metrics"

    for ds_id in experiment_config.dataset_ids:
        dataset_wavenumbers = db.dataset_central_wavenumbers(dataset_id=ds_id)
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"
        spectra_ref = Spectrum(data=np.eye(dataset_wavenumbers.size), wavenumbers=dataset_wavenumbers)
        for char_id in experiment_config.interferometer_ids:
            characterization_subdir = f"{dataset_subdir}/{db.characterizations[char_id].title}"
            save_subdir = f"{characterization_subdir}/spectrum_matrices"

            # Save the reference
            plt.rcParams['font.size'] = str(rc_params.fontsize)

            fig, axes = plt.subplots(**asdict(subplots_options))
            spectra_ref.visualize_matrix(fig=fig, axs=axes[0, 0], **imshow_options)
            filename = f"reference.pdf"
            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=save_subdir,
            )

            for ip_id in experiment_config.inversion_protocol_ids:
                inverter_subdir = f"{characterization_subdir}/{db.inversion_protocols[ip_id].title}"
                spectra_rec_all = np.load(file=reconstruction_dir / inverter_subdir / "spectra_rec_all.npy")
                rmse_diagonal = np.load(file=metrics_dir / inverter_subdir / "rmse_diagonal.npy")
                best_idx = np.argmin(rmse_diagonal)
                spectra_rec_matched, _ = match_stats(
                    array=spectra_rec_all[best_idx],
                    reference=spectra_ref.data,
                    axis=-2
                )
                spectra_rec = replace(spectra_ref, data=spectra_rec_matched)

                # Save the reconstruction
                plt.rcParams['font.size'] = str(rc_params.fontsize)

                fig, axes = plt.subplots(**asdict(subplots_options))
                spectra_rec.visualize_matrix(fig=fig, axs=axes[0, 0], **imshow_options)
                filename = f"{db.inversion_protocols[ip_id].title}.pdf"
                savefig_dir_list(
                    fig=fig,
                    filename=filename,
                    directories_list=figures_dir_list,
                    subdirectory=save_subdir,
                )


def visualize_interferogram_comparison(
        experiment_id: int,
        acquisition_index_ratio: float,
        ip_indices: list,
        subplots_options: SubplotsOptions,
        rc_params: RcParamsOptions,
        plot_options: dict,
):
    config = load_config()
    reports_dir = config.directory_paths.reports
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    figures_dir_list = [
        reports_dir / f"experiment_{experiment_id}" / "figures",
        paper_dir / "figures" / f"{experiment_config.type}",
    ]
    reconstruction_dir = reports_dir / f"experiment_{experiment_id}" / "reconstruction"
    metrics_dir = reports_dir / f"experiment_{experiment_id}" / "metrics"

    for ds_id in experiment_config.dataset_ids:
        interferograms_ref = db.dataset_interferogram(ds_id=ds_id).rescale()
        dataset_wavenumbers = db.dataset_central_wavenumbers(dataset_id=ds_id)
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"

        for char_id in experiment_config.interferometer_ids:
            characterization = db.characterization(characterization_id=char_id)
            transfer_matrix = characterization.transmittance_response(wavenumbers=dataset_wavenumbers)
            characterization_subdir = f"{dataset_subdir}/{db.characterizations[char_id].title}"
            save_subdir = f"{characterization_subdir}/interferogram_comparison"
            acquisition_index = int(acquisition_index_ratio * dataset_wavenumbers.size)
            filename = f"acquisition_{acquisition_index:3}.pdf"
            central_wavenumber = dataset_wavenumbers[acquisition_index]
            fig_title = rf"Central wavenumber $\sigma$={central_wavenumber:.2f}"

            plt.rcParams['font.size'] = str(rc_params.fontsize)
            fig, axes = plt.subplots(**asdict(subplots_options))

            # Plot the reference
            interferograms_ref.visualize(
                axs=axes[0, 0],
                acq_ind=acquisition_index,
                label="Reference",
                color="C0",
                linewidth=3,
                title=fig_title,
                **plot_options,
            )

            for i_ip, ip_id in enumerate([experiment_config.inversion_protocol_ids[ip_index] for ip_index in ip_indices]):
                inverter_subdir = f"{characterization_subdir}/{db.inversion_protocols[ip_id].title}"
                spectra_rec_all = np.load(file=reconstruction_dir / inverter_subdir / "spectra_rec_all.npy")
                rmse_diagonal = np.load(file=metrics_dir / inverter_subdir / "rmse_diagonal.npy")
                best_idx = np.argmin(rmse_diagonal)
                spectra_rec = Spectrum(data=spectra_rec_all[best_idx], wavenumbers=dataset_wavenumbers)
                interferograms_rec = simulate_interferogram(
                    transmittance_response=transfer_matrix,
                    spectrum=spectra_rec
                )
                interferograms_rec_matched, _ = match_stats(
                    array=interferograms_rec.data,
                    reference=interferograms_ref.data,
                    axis=-2,
                )
                interferograms_rec = replace(interferograms_ref, data=interferograms_rec_matched)

                # Plot the reconstruction
                interferograms_rec.visualize(
                    axs=axes[0, 0],
                    acq_ind=acquisition_index,
                    label=db.inversion_protocols[ip_id].title.upper(),
                    color=f"C{i_ip + 1}",
                    linewidth=1.3,
                    title=fig_title,
                    **plot_options
                )

            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=save_subdir,
            )


def visualize_spectrum_comparison(
        experiment_id: int,
        acquisition_index_ratio: float,
        ip_indices: list,
        subplots_options: SubplotsOptions,
        rc_params: RcParamsOptions,
        plot_options: dict,
):
    config = load_config()
    reports_dir = config.directory_paths.reports
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    figures_dir_list = [
        reports_dir / f"experiment_{experiment_id}" / "figures",
        paper_dir / "figures" / f"{experiment_config.type}",
    ]
    reconstruction_dir = reports_dir / f"experiment_{experiment_id}" / "reconstruction"
    metrics_dir = reports_dir / f"experiment_{experiment_id}" / "metrics"

    for ds_id in experiment_config.dataset_ids:
        dataset_wavenumbers = db.dataset_central_wavenumbers(dataset_id=ds_id)
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"
        spectra_ref = Spectrum(data=np.eye(dataset_wavenumbers.size), wavenumbers=dataset_wavenumbers)
        for char_id in experiment_config.interferometer_ids:
            characterization_subdir = f"{dataset_subdir}/{db.characterizations[char_id].title}"
            save_subdir = f"{characterization_subdir}/spectrum_comparison"
            acquisition_index = int(acquisition_index_ratio * dataset_wavenumbers.size)
            filename = f"acquisition_{acquisition_index:3}.pdf"
            central_wavenumber = dataset_wavenumbers[acquisition_index]
            fig_title = rf"Central wavenumber $\sigma$={central_wavenumber:.2f}"

            plt.rcParams['font.size'] = str(rc_params.fontsize)
            fig, axes = plt.subplots(**asdict(subplots_options))

            # Plot the reference
            spectra_ref.visualize(
                axs=axes[0, 0],
                acq_ind=acquisition_index,
                label="Reference",
                color="C0",
                linewidth=3,
                title=fig_title,
                **plot_options,
            )

            for i_ip, ip_id in enumerate(
                    [experiment_config.inversion_protocol_ids[ip_index] for ip_index in ip_indices]):
                inverter_subdir = f"{characterization_subdir}/{db.inversion_protocols[ip_id].title}"
                spectra_rec_all = np.load(file=reconstruction_dir / inverter_subdir / "spectra_rec_all.npy")
                rmse_diagonal = np.load(file=metrics_dir / inverter_subdir / "rmse_diagonal.npy")
                best_idx = np.argmin(rmse_diagonal)
                spectra_rec_matched, _ = match_stats(
                    array=spectra_rec_all[best_idx],
                    reference=spectra_ref.data,
                    axis=-2
                )
                spectra_rec = replace(spectra_ref, data=spectra_rec_matched)

                # Plot the reconstruction
                spectra_rec.visualize(
                    axs=axes[0, 0],
                    acq_ind=acquisition_index,
                    label=db.inversion_protocols[ip_id].title.upper(),
                    color=f"C{i_ip + 1}",
                    linewidth=1.3,
                    title=fig_title,
                    **plot_options
                )

            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=save_subdir,
            )


def visualize_matching_intensity(
        experiment_id: int,
        target_type: str,  # "diagonal" or "maxima"
        subplots_options: SubplotsOptions,
        rc_params: RcParamsOptions,
        plot_options: dict,
):
    config = load_config()
    reports_dir = config.directory_paths.reports
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    figures_dir_list = [
        reports_dir / f"experiment_{experiment_id}" / "figures",
        paper_dir / "figures" / f"{experiment_config.type}",
    ]
    reconstruction_dir = reports_dir / f"experiment_{experiment_id}" / "reconstruction"
    metrics_dir = reports_dir / f"experiment_{experiment_id}" / "metrics"

    for ds_id in experiment_config.dataset_ids:
        dataset_wavenumbers = db.dataset_central_wavenumbers(dataset_id=ds_id)
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"
        spectra_ref = Spectrum(data=np.eye(dataset_wavenumbers.size), wavenumbers=dataset_wavenumbers)
        for char_id in experiment_config.interferometer_ids:
            characterization_subdir = f"{dataset_subdir}/{db.characterizations[char_id].title}"
            save_subdir = f"{characterization_subdir}/matching_{target_type}_intensity"

            for ip_id in experiment_config.inversion_protocol_ids:
                inverter_subdir = f"{characterization_subdir}/{db.inversion_protocols[ip_id].title}"
                spectra_rec_all = np.load(file=reconstruction_dir / inverter_subdir / "spectra_rec_all.npy")
                rmse_diagonal = np.load(file=metrics_dir / inverter_subdir / "rmse_diagonal.npy")
                rmcw = np.load(file=metrics_dir / inverter_subdir / "rmcw.npy")
                best_idx = np.argmin(rmse_diagonal)
                spectra_rec = replace(spectra_ref, data=spectra_rec_all[best_idx])
                spectra_rec, _ = spectra_rec.match_stats(reference=spectra_ref, axis=-2)

                plt.rcParams['font.size'] = str(rc_params.fontsize)

                plot_options["title"] = f"No. of MCW = {rmcw[best_idx]}"
                fig, axes = plt.subplots(**asdict(subplots_options))
                visualize_matching_central_wavenumbers(
                    spectra=spectra_rec,
                    axs=axes[0, 0],
                    target_type=target_type,
                    **plot_options,
                )
                filename = f"{db.inversion_protocols[ip_id].title}.pdf"
                savefig_dir_list(
                    fig=fig,
                    filename=filename,
                    directories_list=figures_dir_list,
                    subdirectory=save_subdir,
                )


class VisualizationOptions(str, Enum):
    DATASETS = "datasets"
    TRANSFER_MATRICES = "transfer_matrices"
    INTERFEROGRAM_MATRICES = "interferogram_matrices"
    SPECTRUM_MATRICES = "spectrum_matrices"
    INTERFEROGRAM_COMPARISON = "interferogram_comparison"
    SPECTRUM_COMPARISON = "spectrum_comparison"
    MATCHING_CENTRAL_WAVENUMBERS_INTENSITY = "matching_central_wavenumbers_intensity"


def visualization_function_factory(option: VisualizationOptions) -> Callable:
    if option == VisualizationOptions.DATASETS:
        return visualize_datasets

    if option == VisualizationOptions.TRANSFER_MATRICES:
        return visualize_transfer_matrices

    if option == VisualizationOptions.INTERFEROGRAM_MATRICES:
        return visualize_interferogram_matrices

    if option == VisualizationOptions.SPECTRUM_MATRICES:
        return visualize_spectrum_matrices

    if option == VisualizationOptions.INTERFEROGRAM_COMPARISON:
        return visualize_interferogram_comparison

    if option == VisualizationOptions.SPECTRUM_COMPARISON:
        return visualize_spectrum_comparison

    if option == VisualizationOptions.MATCHING_CENTRAL_WAVENUMBERS_INTENSITY:
        return visualize_matching_intensity


def visualization_inputs_factory(experiment_id: int, option: VisualizationOptions) -> dict:
    if option == VisualizationOptions.DATASETS:
        inputs_dict = {
            "experiment_id": experiment_id,
            "subplots_options": SubplotsOptions(),
            "rc_params": RcParamsOptions(fontsize=17),
            "imshow_options": {
                "title": "",
            },
        }
    elif option == VisualizationOptions.TRANSFER_MATRICES:
        inputs_dict = {
            "experiment_id": experiment_id,
            "subplots_options": SubplotsOptions(),
            "rc_params": RcParamsOptions(fontsize=17),
            "transmat_imshow_options": {
                "title": "",
            },
            "singvals_plot_options": {
                "title": f" ",
            },
            "opd_response_plot_options": {
                "title": None,
                "show_full_title": False,
            },
            "dct_opd_plot_options": {
                "title": None,
                "show_full_title": False,
            },
            "opd_idx": 20,
        }
    elif option == VisualizationOptions.INTERFEROGRAM_MATRICES:
        inputs_dict = {
            "experiment_id": experiment_id,
            "subplots_options": SubplotsOptions(),
            "rc_params": RcParamsOptions(fontsize=17),
            "imshow_options": {
                "title": "",
                "vmin": 0.,
                "vmax": 0.7,
            },
        }
    elif option == VisualizationOptions.SPECTRUM_MATRICES:
        inputs_dict = {
            "experiment_id": experiment_id,
            "subplots_options": SubplotsOptions(),
            "rc_params": RcParamsOptions(fontsize=17),
            "imshow_options": {
                "title": "",
                "vmin": -0.2,
                "vmax": 1,
            },
        }
    elif option == VisualizationOptions.INTERFEROGRAM_COMPARISON:
        inputs_dict = {
            "experiment_id": experiment_id,
            "acquisition_index_ratio": 0.4,
            "ip_indices": [2, 3, 9],
            "subplots_options": SubplotsOptions(),
            "rc_params": RcParamsOptions(fontsize=17),
            "plot_options": {
                "linestyle": "-",
                "ylabel": "Normalized Intensity",
                "ylim": [-0.1, 1.1],
            },
        }
    elif option == VisualizationOptions.SPECTRUM_COMPARISON:
        inputs_dict = {
            "experiment_id": experiment_id,
            "acquisition_index_ratio": 0.4,
            "ip_indices": [2, 3, 9],
            "subplots_options": SubplotsOptions(),
            "rc_params": RcParamsOptions(fontsize=17),
            "plot_options": {
                "linestyle": "-",
                "ylabel": "Normalized Intensity",
                "ylim": [-0.1, 1.1],
            },
        }
    elif option == VisualizationOptions.MATCHING_CENTRAL_WAVENUMBERS_INTENSITY:
        inputs_dict = {
            "experiment_id": experiment_id,
            "target_type": "maxima",
            "subplots_options": SubplotsOptions(),
            "rc_params": RcParamsOptions(fontsize=17),
            "plot_options": {
                "linestyle": "-",
                "xlim": [0.85, 3.],
                "ylabel": "Normalized Intensity",
                "ylim": [-0.1, 1.1],
            },
        }
    else:
        raise ValueError(f"Option '{option}' is not supported.")
    return inputs_dict


def visualize_options(experiment_id: int, options: list[VisualizationOptions]):
    for option in options:
        function = visualization_function_factory(option=option)
        inputs = visualization_inputs_factory(experiment_id=experiment_id, option=option)
        function(**inputs)


def main():
    visualization_options = [
        # VisualizationOptions.DATASETS,
        # VisualizationOptions.TRANSFER_MATRICES,
        # VisualizationOptions.INTERFEROGRAM_MATRICES,
        VisualizationOptions.SPECTRUM_MATRICES,
        # VisualizationOptions.INTERFEROGRAM_COMPARISON,
        # VisualizationOptions.SPECTRUM_COMPARISON,
        VisualizationOptions.MATCHING_CENTRAL_WAVENUMBERS_INTENSITY,
    ]

    experiment_id_options = [1, 2]
    for experiment_id in experiment_id_options:
        visualize_options(options=visualization_options, experiment_id=experiment_id)


if __name__ == "__main__":
    main()
