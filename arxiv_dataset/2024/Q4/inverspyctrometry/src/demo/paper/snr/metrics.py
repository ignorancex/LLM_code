from pprint import pprint

from src.common_utils.utils import calculate_rmse
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_load_list, numpy_save_list


def main():
    experiment_id = 0

    config = load_config()
    reports_folder = config.directory_paths.reports

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    pprint(dict(experiment_config))

    reconstruction_dir = reports_folder / f"experiment_{experiment_id}" / "reconstruction"
    metrics_dir = reports_folder / f"experiment_{experiment_id}" / "metrics"

    for ds_id in experiment_config.dataset_ids:
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"

        spectra_ref = db.dataset_spectrum(ds_id=ds_id)

        for ifm_id in experiment_config.interferometer_ids:
            print(f"\tInterferometer: {db.interferometers[ifm_id].title.upper()}")
            interferometer_subdir = f"{dataset_subdir}/{db.interferometers[ifm_id].title}"

            for nl_idx in experiment_config.noise_level_indices:
                print(f"\t\tSNR: {db.noise_levels[nl_idx]} dB")
                noise_level_subdir = f"{interferometer_subdir}/{int(db.noise_levels[nl_idx])}_db"

                for ip_id in experiment_config.inversion_protocol_ids:
                    print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")
                    inverter_subdir = f"{noise_level_subdir}/{db.inversion_protocols[ip_id].title}"

                    # Load reconstruction
                    argmin, spectra_rec_best = numpy_load_list(
                        filenames=["argmin_rmse.npy", "spectra_rec_best.npy"],
                        directory=reconstruction_dir,
                        subdirectory=inverter_subdir,
                    )

                    # Compute metrics
                    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                    lambdaa_min = lambdaas[argmin]
                    rmse_min = calculate_rmse(
                        array=spectra_rec_best,
                        reference=spectra_ref.data,
                        is_match_axis=-2,
                        is_match_stats=True,
                        is_rescale_reference=True,
                    )
                    print(f"\t\t\t\tLambda: {lambdaa_min:.3f}")
                    print(f"\t\t\t\tRMSE:   {rmse_min:.3f}")

                    # Save metrics
                    numpy_save_list(
                        filenames=["lambdaa_min", "rmse_min"],
                        arrays=[lambdaa_min, rmse_min],
                        directories=[metrics_dir],
                        subdirectory=inverter_subdir,
                    )


if __name__ == "__main__":
    main()
