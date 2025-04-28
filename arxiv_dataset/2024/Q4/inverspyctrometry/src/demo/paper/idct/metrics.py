from src.common_utils.utils import calculate_rmse
from src.demo.paper.snr.utils import experiment_dir_convention, experiment_subdir_convention
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_save_list, numpy_load_list


def metrics_core(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        is_verbose: bool,
):
    db = load_config().database()

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

    spectra_rec, = numpy_load_list(
        filenames=["spectra_rec.npy"],
        directory=reconstruction_dir,
        subdirectory=inverter_subdir,
    )

    spectra_ref = db.dataset_spectrum(ds_id=dataset_id).data
    lambdaa_min = db.inversion_protocol_lambdaas(inv_protocol_id=inversion_protocol_id)[0]
    rmse_min = calculate_rmse(
        array=spectra_rec,
        reference=spectra_ref,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    if is_verbose:
        print(f"\t\t\t\tRMSE:   {rmse_min:.3f}")

    numpy_save_list(
        filenames=["lambdaa_min", "rmse_min"],
        arrays=[lambdaa_min, rmse_min],
        directories=[metrics_dir],
        subdirectory=inverter_subdir,
    )


def metrics_one_experiment(
        experiment_id: int,
        is_verbose: bool,
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

                metrics_core(
                    experiment_id=experiment_id,
                    dataset_id=dataset_id,
                    interferometer_id=interferometer_id,
                    noise_level_index=noise_level_index,
                    inversion_protocol_id=inversion_protocol_id,
                    is_verbose=is_verbose,
                )


def metrics_list_experiments(
        experiment_ids: list,
        is_verbose: bool = True,
):
    for experiment_id in experiment_ids:
        metrics_one_experiment(
            experiment_id=experiment_id,
            is_verbose=is_verbose,
        )


def main():
    experiment_ids = [3, 4, 5]
    is_verbose = True
    metrics_list_experiments(experiment_ids=experiment_ids, is_verbose=is_verbose)


if __name__ == "__main__":
    main()
