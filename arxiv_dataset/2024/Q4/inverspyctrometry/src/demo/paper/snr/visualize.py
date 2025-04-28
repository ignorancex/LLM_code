from dataclasses import asdict, replace

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.interferogram import Interferogram
from src.demo.paper.snr.utils import experiment_subdir_convention, experiment_dir_convention
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_load_list
from src.outputs.visualization import savefig_dir_list, RcParamsOptions, SubplotsOptions


def experiment_figures_subdir_convention(
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        folder_name: str,
) -> str:
    figures_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
    )
    figures_subdir = f"{figures_subdir}/{folder_name}"
    return figures_subdir


def visualize_transfer_matrix(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
):
    config = load_config()
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    figures_dir_list = [
        experiment_dir_convention(dir_type="figures", experiment_id=experiment_id),
        experiment_dir_convention(dir_type="paper_figures", experiment_id=experiment_id, custom_dir=paper_dir),
    ]
    save_subdir = experiment_figures_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=-1,
        folder_name="transfer_matrix",
    )

    # Load
    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
    interferometer = db.interferometer(interferometer_id=interferometer_id)
    transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)

    # Visualize
    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig, axes = plt.subplots(**asdict(subplots_options))
    transfer_matrix.visualize(fig=fig, axs=axes[0, 0], **plot_options)

    # Save
    filename = "transfer_matrix.pdf"
    savefig_dir_list(
        fig=fig,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )


def visualize_spectrum_comparison(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        acquisition_index: int,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
        inversion_protocol_indices: list,
        inverspectrometer: str = "haar",
):
    config = load_config()
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    reconstruction_dir = experiment_dir_convention(dir_type="reconstruction", experiment_id=experiment_id)
    figures_dir_list = [
        experiment_dir_convention(dir_type="figures", experiment_id=experiment_id),
        experiment_dir_convention(dir_type="paper_figures", experiment_id=experiment_id, custom_dir=paper_dir),
    ]
    save_subdir = experiment_figures_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
        folder_name="spectrum_comparison",
    )
    inversion_protocol_list = [
        db.experiments[experiment_id].inversion_protocol_ids[ip_id]
        for ip_id in inversion_protocol_indices
    ]

    # Load
    spectra_ref_scaled = db.dataset_spectrum(ds_id=dataset_id).rescale(new_max=1, axis=-2)

    # Visualize
    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig, axes = plt.subplots(**asdict(subplots_options))
    spectra_ref_scaled.visualize(
        axs=axes[0, 0],
        acq_ind=acquisition_index,
        label="Reference",
        color="C0",
        linestyle="-",
        linewidth=3,
        title="",
        **plot_options,
    )

    load_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
    ) + "/haar"

    # Load
    spectra_rec_best = replace(
        spectra_ref_scaled,
        data=np.load(file=reconstruction_dir / load_subdir / "spectra_rec_best.npy"),
    )
    spectra_rec_matched, _ = spectra_rec_best.match_stats(reference=spectra_ref_scaled, axis=-2)

    # Visualize
    spectra_rec_matched.visualize(
        axs=axes[0, 0],
        acq_ind=acquisition_index,
        label="HAAR",
        color=f"C1",
        linestyle="--",
        linewidth=1.3,
        title="",
        **plot_options
    )

    for i_ip, ip_id in enumerate(inversion_protocol_list):
        load_subdir = experiment_subdir_convention(
            dataset_id=dataset_id,
            interferometer_id=interferometer_id,
            noise_level_index=noise_level_index,
            inversion_protocol_id=ip_id,
        )

        # Load
        spectra_rec_best = replace(
            spectra_ref_scaled,
            data=np.load(file=reconstruction_dir / load_subdir / "spectra_rec_best.npy"),
        )
        spectra_rec_matched, _ = spectra_rec_best.match_stats(reference=spectra_ref_scaled, axis=-2)

        # Visualize
        spectra_rec_matched.visualize(
            axs=axes[0, 0],
            acq_ind=acquisition_index,
            label=db.inversion_protocols[ip_id].title.upper(),
            color=f"C{i_ip + 2}",
            linestyle="--",
            linewidth=1.3,
            title="",
            **plot_options
        )

    # Save
    filename = f"acquisition_{acquisition_index:03}.pdf"
    savefig_dir_list(
        fig=fig,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )


def visualize_interferograms_noisy(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        acquisition_index: int,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
):
    config = load_config()
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    simulation_dir = experiment_dir_convention(dir_type="simulation", experiment_id=experiment_id)
    interferometer_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
    )
    figures_dir_list = [
        experiment_dir_convention(dir_type="figures", experiment_id=experiment_id),
        experiment_dir_convention(dir_type="paper_figures", experiment_id=experiment_id, custom_dir=paper_dir),
    ]
    save_subdir = experiment_figures_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=-1,
        folder_name="interferograms_noisy",
    )

    # Load
    data, opds = numpy_load_list(
        filenames=["data.npy", "opds.npy"],
        directory=simulation_dir,
        subdirectory=interferometer_subdir,
    )
    interferograms_rescaled = Interferogram(data=data, opds=opds).rescale(new_max=1, axis=-2)

    # Visualize
    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig, axes = plt.subplots(**asdict(subplots_options))
    interferograms_rescaled.visualize(
        axs=axes[0, 0],
        acq_ind=acquisition_index,
        label="Reference",
        color="C0",
        linewidth=3,
        title="",
        **plot_options,
    )

    for i_nl, noise_level_index in enumerate(db.experiments[experiment_id].noise_level_indices):
        # Load
        np.random.seed(0)
        interferograms_noisy = interferograms_rescaled.add_noise(snr_db=db.noise_levels[noise_level_index])

        # Visualize
        interferograms_noisy.visualize(
            axs=axes[0, 0],
            acq_ind=acquisition_index,
            label=f"{int(db.noise_levels[noise_level_index])} dB",
            color=f"C{i_nl + 1}",
            linewidth=1.3,
            title="",
            **plot_options
        )

    # Save
    filename = f"acquisition_{acquisition_index:03}.pdf"
    savefig_dir_list(
        fig=fig,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )


def main():
    experiment_id = 0

    config = load_config()
    db = config.database()

    experiment_config = db.experiments[experiment_id]

    for ds_id in experiment_config.dataset_ids:
        for ifm_id in experiment_config.interferometer_ids:
            visualize_transfer_matrix(
                experiment_id=experiment_id,
                dataset_id=ds_id,
                interferometer_id=ifm_id,
                rc_params=RcParamsOptions(fontsize=17),
                subplots_options=SubplotsOptions(),
                plot_options={
                    "title": "",
                },
            )
            for nl_idx in experiment_config.noise_level_indices:
                visualize_interferograms_noisy(
                    experiment_id=experiment_id,
                    dataset_id=ds_id,
                    interferometer_id=ifm_id,
                    acquisition_index=0,
                    rc_params=RcParamsOptions(fontsize=17),
                    subplots_options=SubplotsOptions(),
                    plot_options={},
                )
                visualize_interferograms_noisy(
                    experiment_id=experiment_id,
                    dataset_id=ds_id,
                    interferometer_id=ifm_id,
                    acquisition_index=13,
                    rc_params=RcParamsOptions(fontsize=17),
                    subplots_options=SubplotsOptions(),
                    plot_options={},
                )
                visualize_spectrum_comparison(
                    experiment_id=experiment_id,
                    dataset_id=ds_id,
                    interferometer_id=ifm_id,
                    noise_level_index=nl_idx,
                    acquisition_index=0,
                    rc_params=RcParamsOptions(fontsize=17),
                    subplots_options=SubplotsOptions(),
                    plot_options={
                        "ylabel": "Normalized Intensity",
                        "ylim": [-0.1, 1.1],
                    },
                    inversion_protocol_indices=[2, 3, 4],
                    inverspectrometer="haar",
                )
                visualize_spectrum_comparison(
                    experiment_id=experiment_id,
                    dataset_id=ds_id,
                    interferometer_id=ifm_id,
                    noise_level_index=nl_idx,
                    acquisition_index=13,
                    rc_params=RcParamsOptions(fontsize=17),
                    subplots_options=SubplotsOptions(),
                    plot_options={
                        "ylabel": "Normalized Intensity",
                        "ylim": [-0.1, 1.1],
                    },
                    inversion_protocol_indices=[2, 3, 4],
                    inverspectrometer="haar",
                )


if __name__ == "__main__":
    main()
