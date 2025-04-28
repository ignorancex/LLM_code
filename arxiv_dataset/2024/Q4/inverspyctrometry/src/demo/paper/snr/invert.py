from pprint import pprint

import numpy as np

from src.common_utils.utils import calculate_rmse
from src.direct_model.interferometer import simulate_interferogram
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_save_list


def main():
    experiment_id = 0

    config = load_config()
    reports_folder = config.directory_paths.reports

    db = config.database()
    experiment_config = db.experiments[experiment_id]
    pprint(dict(experiment_config))

    reconstruction_dir = reports_folder / f"experiment_{experiment_id}" / "reconstruction"

    for ds_id in experiment_config.dataset_ids:
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"

        spectra_ref = db.dataset_spectrum(ds_id=ds_id)

        for ifm_id in experiment_config.interferometer_ids:
            print(f"\tInterferometer: {db.interferometers[ifm_id].title.upper()}")
            interferometer_subdir = f"{dataset_subdir}/{db.interferometers[ifm_id].title}"

            interferometer = db.interferometer(interferometer_id=ifm_id)

            transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
            interferograms_ref = simulate_interferogram(transmittance_response=transfer_matrix, spectrum=spectra_ref)

            interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
            # interferograms_ref = interferograms_ref.center(new_mean=0)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

            for nl_idx in experiment_config.noise_level_indices:
                print(f"\t\tSNR: {db.noise_levels[nl_idx]} dB")
                noise_level_subdir = f"{interferometer_subdir}/{int(db.noise_levels[nl_idx])}_db"

                snr_db = db.noise_levels[nl_idx]

                np.random.seed(0)
                interferograms_noisy = interferograms_ref.add_noise(snr_db=snr_db)

                for ip_id in experiment_config.inversion_protocol_ids:
                    print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")
                    inverter_subdir = f"{noise_level_subdir}/{db.inversion_protocols[ip_id].title}"

                    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                    spectra_rec_all = np.zeros(shape=(lambdaas.size, *spectra_ref.data.shape))
                    for i_lmd, lambdaa in enumerate(lambdaas):
                        inverter = db.inversion_protocol(inv_protocol_id=ip_id, lambdaa=lambdaa)
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
                    numpy_save_list(
                        filenames=["argmin_rmse.npy", "spectra_rec_best.npy"],
                        arrays=[argmin_rmse, spectra_rec_best],
                        directories=[reconstruction_dir],
                        subdirectory=inverter_subdir,
                    )

                # Save noisy interferograms
                numpy_save_list(
                    filenames=["interferograms_noisy"],
                    arrays=[interferograms_noisy.data],
                    directories=[reconstruction_dir],
                    subdirectory=noise_level_subdir,
                )


if __name__ == "__main__":
    main()
