from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.common_utils.custom_vars import EnumInvalidOptionError
from src.common_utils.interferogram import Interferogram
from src.common_utils.utils import calculate_rmse
from src.demo.paper.snr.utils import experiment_dir_convention, experiment_subdir_convention
from src.interface.configuration import load_config
from src.inverse_model.analytical_inverter import HaarInverter
from src.outputs.serialize import numpy_save_list, numpy_load_list


# TODO:
#  To be automatized:
#   Directory convention
#  Separate DB functionalities from processing: Load the fb components from a higher level and process them internally
#  Separate saving and loading?


@dataclass
class ExtrapolateOptions:
    kind: str = "cubic"
    fill_value: str | float | tuple = 0.


def simulate(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        wn_factor: float,
        is_verbose: bool = False,
):
    """
    Simulates interferograms from continuous (oversampled) spectra
    """
    config = load_config()
    db = config.database()

    if is_verbose:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")

    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
    opds = load_config().database().interferometers[interferometer_id].opds
    wn_bandwidth = 1 / (2 * opds.step)
    wn_step_2_wave = 1 / (2 * opds.max)
    wn_step = wn_step_2_wave / wn_factor  # wn_factor should have a minimum of N from the "N-wave" approximation
    wavenumbers_cont = np.arange(start=wn_step / 2, stop=wn_bandwidth, step=wn_step)
    valid_indices = np.logical_and(
        spectra_ref.wavenumbers[0] <= wavenumbers_cont,
        wavenumbers_cont <= spectra_ref.wavenumbers[-1]
    )
    wavenumbers_cont = wavenumbers_cont[valid_indices]
    spectra_cont = spectra_ref.interpolate(wavenumbers=wavenumbers_cont)

    interferometer = db.interferometer(interferometer_id=interferometer_id)
    interferograms = interferometer.acquire_interferogram(spectrum=spectra_cont)

    simulation_dir = experiment_dir_convention(
        dir_type="simulation",
        experiment_id=experiment_id,
    )
    interferometer_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
    )
    # TODO: interferograms.serialize()/.dump()
    numpy_save_list(
        filenames=["data.npy", "opds.npy"],
        arrays=[interferograms.data, interferograms.opds],
        directories=[simulation_dir],
        subdirectory=interferometer_subdir,
    )


def invert_haar(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        extrap_opts: ExtrapolateOptions,
        haar_order: int,
        noise_seed: int = None,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()

    if is_verbose:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")
        print(f"\t\tSNR: {db.noise_levels[noise_level_index]} dB")
        print(f"\t\t\tInversion Protocol: HAAR")

    # Load the simulated interferograms
    simulation_dir = experiment_dir_convention(
        dir_type="simulation",
        experiment_id=experiment_id,
    )
    interferometer_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
    )
    interferograms_ref = Interferogram.load_from_dir(directory=simulation_dir / interferometer_subdir)

    # Extrapolate the missing OPDs
    if extrap_opts.kind != "none":
        interferograms_ref = interferograms_ref.extrapolate(
            kind=extrap_opts.kind,
            fill_value=extrap_opts.fill_value,
        )

    # Rescale (and center?) the interferograms
    interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
    # interferograms_ref = interferograms_ref.center(new_mean=0)

    # Add noise to the interferograms
    snr_db = db.noise_levels[noise_level_index]
    np.random.seed(seed=noise_seed)
    interferograms_noisy = interferograms_ref.add_noise(snr_db=snr_db)

    # Load the transfer matrix / Knowledge of the acquisition (direct) model
    interferometer = db.interferometer(interferometer_id=interferometer_id)
    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)

    haar_inv = HaarInverter(
        transmittance=interferometer.transmittance_coefficients[0],
        wavenumbers=spectra_ref.wavenumbers,
        reflectance=interferometer.reflectance_coefficients[0],
        order=haar_order,
        is_mean_center=True,
    )
    spectrum_haar = haar_inv.reconstruct_spectrum(interferogram=interferograms_noisy)

    spectra_rec_all = spectrum_haar.data[None, :]
    rmse_lambdaas = calculate_rmse(
        array=spectra_rec_all,
        reference=spectra_ref.data,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    argmin_rmse = np.argmin(rmse_lambdaas)
    spectra_rec_best = spectra_rec_all[argmin_rmse]

    # Save reconstruction
    reconstruction_dir = experiment_dir_convention(
        dir_type="reconstruction",
        experiment_id=experiment_id,
    )
    inverter_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
    ) + "/haar"
    numpy_save_list(
        filenames=["argmin_rmse.npy", "spectra_rec_best.npy"],
        arrays=[argmin_rmse, spectra_rec_best],
        directories=[reconstruction_dir],
        subdirectory=inverter_subdir,
    )


def invert(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        extrap_opts: ExtrapolateOptions,
        noise_seed: int = None,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()

    if is_verbose:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")
        print(f"\t\tSNR: {db.noise_levels[noise_level_index]} dB")
        print(f"\t\t\tInversion Protocol: {db.inversion_protocols[inversion_protocol_id].title.upper()}")

    # Load the simulated interferograms
    simulation_dir = experiment_dir_convention(
        dir_type="simulation",
        experiment_id=experiment_id,
    )
    interferometer_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
    )
    interferograms_ref = Interferogram.load_from_dir(directory=simulation_dir / interferometer_subdir)

    # Extrapolate the missing OPDs
    if extrap_opts.kind != "none":
        interferograms_ref = interferograms_ref.extrapolate(
            kind=extrap_opts.kind,
            fill_value=extrap_opts.fill_value,
        )

    # Rescale (and center?) the interferograms
    interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
    # interferograms_ref = interferograms_ref.center(new_mean=0)

    # Add noise to the interferograms
    snr_db = db.noise_levels[noise_level_index]
    np.random.seed(seed=noise_seed)
    interferograms_noisy = interferograms_ref.add_noise(snr_db=snr_db)

    # Load the transfer matrix / Knowledge of the acquisition (direct) model
    interferometer = db.interferometer(interferometer_id=interferometer_id)
    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
    transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
    transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

    # Reconstruct the spectrum / Invert the interferograms
    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=inversion_protocol_id)
    spectra_rec_all = np.zeros(shape=(lambdaas.size, *spectra_ref.data.shape))
    for i_lmd, lambdaa in enumerate(lambdaas):
        inverter = db.inversion_protocol(inv_protocol_id=inversion_protocol_id, lambdaa=lambdaa)
        spectra_rec = inverter.reconstruct_spectrum(
            interferogram=interferograms_noisy, transmittance_response=transfer_matrix
        )
        spectra_rec_all[i_lmd] = spectra_rec.data
    rmse_lambdaas = calculate_rmse(
        array=spectra_rec_all,
        reference=spectra_ref.data,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    argmin_rmse = np.argmin(rmse_lambdaas)
    spectra_rec_best = spectra_rec_all[argmin_rmse]

    # Save reconstruction
    reconstruction_dir = experiment_dir_convention(
        dir_type="reconstruction",
        experiment_id=experiment_id,
    )
    inverter_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
        inversion_protocol_id=inversion_protocol_id,
    )
    numpy_save_list(
        filenames=["argmin_rmse.npy", "spectra_rec_best.npy"],
        arrays=[argmin_rmse, spectra_rec_best],
        directories=[reconstruction_dir],
        subdirectory=inverter_subdir,
    )


def metrics_haar(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()

    if is_verbose:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")
        print(f"\t\tSNR: {db.noise_levels[noise_level_index]} dB")
        print(f"\t\t\tInversion Protocol: HAAR")

    # Load reconstruction
    reconstruction_dir = experiment_dir_convention(
        dir_type="reconstruction",
        experiment_id=experiment_id,
    )
    inverter_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
    ) + "/haar"
    argmin, spectra_rec_best = numpy_load_list(
        filenames=["argmin_rmse.npy", "spectra_rec_best.npy"],
        directory=reconstruction_dir,
        subdirectory=inverter_subdir,
    )

    # Compute metrics
    lambdaas = np.array([0.])
    lambdaa_min = lambdaas[argmin]
    spectra_ref = db.dataset_spectrum(ds_id=dataset_id).data
    rmse_min = calculate_rmse(
        array=spectra_rec_best,
        reference=spectra_ref,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    if is_verbose:
        print(f"\t\t\t\tLambda: {lambdaa_min:.3f}")
        print(f"\t\t\t\tRMSE:   {rmse_min:.3f}")

    # Save metrics
    metrics_dir = experiment_dir_convention(
        dir_type="metrics",
        experiment_id=experiment_id,
    )
    numpy_save_list(
        filenames=["lambdaa_min", "rmse_min"],
        arrays=[lambdaa_min, rmse_min],
        directories=[metrics_dir],
        subdirectory=inverter_subdir,
    )


def metrics(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()

    if is_verbose:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")
        print(f"\t\tSNR: {db.noise_levels[noise_level_index]} dB")
        print(f"\t\t\tInversion Protocol: {db.inversion_protocols[inversion_protocol_id].title.upper()}")

    reconstruction_dir = experiment_dir_convention(
        dir_type="reconstruction",
        experiment_id=experiment_id,
    )
    metrics_dir = experiment_dir_convention(
        dir_type="metrics",
        experiment_id=experiment_id,
    )
    inverter_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
        inversion_protocol_id=inversion_protocol_id,
    )

    # Load reconstruction
    argmin, spectra_rec_best = numpy_load_list(
        filenames=["argmin_rmse.npy", "spectra_rec_best.npy"],
        directory=reconstruction_dir,
        subdirectory=inverter_subdir,
    )

    # Compute metrics
    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=inversion_protocol_id)
    lambdaa_min = lambdaas[argmin]
    spectra_ref = db.dataset_spectrum(ds_id=dataset_id).data
    rmse_min = calculate_rmse(
        array=spectra_rec_best,
        reference=spectra_ref,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    if is_verbose:
        print(f"\t\t\t\tLambda: {lambdaa_min:.3f}")
        print(f"\t\t\t\tRMSE:   {rmse_min:.3f}")

    # Save metrics
    numpy_save_list(
        filenames=["lambdaa_min", "rmse_min"],
        arrays=[lambdaa_min, rmse_min],
        directories=[metrics_dir],
        subdirectory=inverter_subdir,
    )


def visualize(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()
    print(db)


@dataclass(frozen=True)
class SubExperiment:
    """The flow of a single experiment taking into account having only one set of interferograms [Opds x Acqs]"""
    experiment_id: int
    dataset_id: int
    interferometer_id: int
    noise_level_index: int
    inversion_protocol_id: int


def main():
    exp_id = 0

    config = load_config()
    db = config.database()

    exp_config = db.experiments[exp_id]

    wn_factor = 10.
    extra_opts = ExtrapolateOptions(kind="none", fill_value=(0., 0.))
    noise_seed = 0
    haar_order = 10

    is_simulate = False
    if is_simulate:
        for ds_id in exp_config.dataset_ids:
            for ifm_id in exp_config.interferometer_ids:
                simulate(
                    experiment_id=exp_id,
                    dataset_id=ds_id,
                    wn_factor=wn_factor,
                    interferometer_id=ifm_id,
                    is_verbose=False,
                )

    is_invert = False
    if is_invert:
        for ds_id in exp_config.dataset_ids:
            for ifm_id in exp_config.interferometer_ids:
                for nl_idx in exp_config.noise_level_indices:
                    invert_haar(
                        experiment_id=exp_id,
                        dataset_id=ds_id,
                        interferometer_id=ifm_id,
                        noise_level_index=nl_idx,
                        extrap_opts=extra_opts,
                        haar_order=haar_order,
                        noise_seed=noise_seed,
                        is_verbose=True,
                    )
                    for ip_id in exp_config.inversion_protocol_ids:
                        invert(
                            experiment_id=exp_id,
                            dataset_id=ds_id,
                            interferometer_id=ifm_id,
                            noise_level_index=nl_idx,
                            inversion_protocol_id=ip_id,
                            extrap_opts=extra_opts,
                            noise_seed=noise_seed,
                            is_verbose=True,
                        )

    is_metrics = True
    if is_metrics:
        for ds_id in exp_config.dataset_ids:
            for ifm_id in exp_config.interferometer_ids:
                for nl_idx in exp_config.noise_level_indices:
                    metrics_haar(
                        experiment_id=exp_id,
                        dataset_id=ds_id,
                        interferometer_id=ifm_id,
                        noise_level_index=nl_idx,
                        is_verbose=True,
                    )
                    for ip_id in exp_config.inversion_protocol_ids:
                        metrics(
                            experiment_id=exp_id,
                            dataset_id=ds_id,
                            interferometer_id=ifm_id,
                            noise_level_index=nl_idx,
                            inversion_protocol_id=ip_id,
                            is_verbose=True,
                        )


if __name__ == "__main__":
    main()
