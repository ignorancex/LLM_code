from dataclasses import replace

import numpy as np

from src.demo.paper.snr.utils import experiment_dir_convention, experiment_subdir_convention
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_save_list


def invert_core(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        is_compensate: bool,
):
    db = load_config().database()

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

    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
    interferometer = db.interferometer(interferometer_id=interferometer_id)

    interferograms_ref = interferometer.acquire_interferogram(spectrum=spectra_ref)
    interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
    interferograms_ref = interferograms_ref.center(new_mean=0, axis=-2)

    transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
    transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

    snr_db = db.noise_levels[noise_level_index]
    np.random.seed(0)
    interferograms_noisy = interferograms_ref.add_noise(snr_db=snr_db)

    lambdaa = db.inversion_protocol_lambdaas(inv_protocol_id=inversion_protocol_id)[0]
    inverter = db.inversion_protocol(inv_protocol_id=inversion_protocol_id, lambdaa=lambdaa)
    spectra_rec = inverter.reconstruct_spectrum(
        interferogram=interferograms_noisy, transmittance_response=transfer_matrix
    )

    if is_compensate:
        reflectance = interferometer.reflectance(wavenumbers=spectra_ref.wavenumbers)[0]
        transmittance = interferometer.transmittance(wavenumbers=spectra_ref.wavenumbers)[0]
        quotient = 1 / (1 - reflectance ** 2)
        compensation = transmittance ** 2 * quotient * reflectance
        spectra_rec = replace(
            spectra_rec,
            data=spectra_rec.data / compensation[:, None],
            wavenumbers=spectra_rec.wavenumbers,
        )

    numpy_save_list(
        filenames=["spectra_rec.npy"],
        arrays=[spectra_rec.data],
        directories=[reconstruction_dir],
        subdirectory=inverter_subdir,
    )


def invert_one_experiment(
        experiment_id,
        is_compensate: bool,
):
    db = load_config().database()

    experiment_config = db.experiments[experiment_id]
    for dataset_id in experiment_config.dataset_ids:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")

        for interferometer_id in experiment_config.interferometer_ids:
            print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")

            for noise_level_index in experiment_config.noise_level_indices:
                print(f"\t\tNoise Level: {db.noise_levels[noise_level_index]:.0f}")

                inversion_protocol_id = experiment_config.inversion_protocol_ids[0]

                invert_core(
                    experiment_id=experiment_id,
                    dataset_id=dataset_id,
                    interferometer_id=interferometer_id,
                    noise_level_index=noise_level_index,
                    inversion_protocol_id=inversion_protocol_id,
                    is_compensate=is_compensate,
                )


def invert_list_experiments(
        experiment_ids: list,
        is_compensate: bool = True,
):
    for experiment_id in experiment_ids:
        invert_one_experiment(
            experiment_id=experiment_id,
            is_compensate=is_compensate,
        )


def main():
    experiment_ids = [3, 4, 5, 6]
    is_compensate = True
    invert_list_experiments(experiment_ids=experiment_ids, is_compensate=is_compensate)


if __name__ == "__main__":
    main()
