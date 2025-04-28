from pprint import pprint

import numpy as np

from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import calculate_rmse, match_stats
from src.demo.paper.monochromatic.utils import print_metrics, calculate_rmcw
from src.interface.configuration import load_config


def main():
    config = load_config()
    db = config.database()

    experiment_id = 1
    experiment_params = db.experiments[experiment_id]
    pprint(dict(experiment_params))
    print()

    for ds_id in experiment_params.dataset_ids:
        interferograms_ref = db.dataset_interferogram(ds_id=ds_id)
        wavenumbers_ifgm = db.dataset_central_wavenumbers(dataset_id=ds_id)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")

        interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)

        spectra_ref = Spectrum(data=np.eye(wavenumbers_ifgm.size), wavenumbers=wavenumbers_ifgm)

        for char_id in experiment_params.interferometer_ids:
            characterization = db.characterization(characterization_id=char_id)
            print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")

            transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers_ifgm)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

            for ip_id in experiment_params.inversion_protocol_ids:
                lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")

                rmse = np.zeros_like(a=lambdaas)
                rmse_diagonal = np.zeros_like(a=lambdaas)
                rmcw = np.zeros_like(a=lambdaas, dtype=int)

                for il, lambdaa in enumerate(lambdaas):
                    inverter = db.inversion_protocol(inv_protocol_id=ip_id, lambdaa=lambdaa)
                    spectra_rec = inverter.reconstruct_spectrum(
                        interferogram=interferograms_ref, transmittance_response=transfer_matrix
                    )

                    # EXTRA: Metrics
                    reconstruction, reference = match_stats(
                        array=spectra_rec.data,
                        reference=spectra_ref.data,
                        axis=-2,
                    )
                    rmse[il] = calculate_rmse(
                        array=reconstruction,
                        reference=reference,
                    )
                    rmse_diagonal[il] = calculate_rmse(
                        array=np.diag(reconstruction),
                        reference=np.diag(reference),
                    )
                    rmcw[il] = calculate_rmcw(monochromatic_array=reconstruction)

                    nb_tabs = int(3)
                    print_metrics(
                        nb_tabs,
                        "i_lambda",
                        il,
                        lambdaas[il],
                        rmse[il],
                        rmse_diagonal[il],
                        rmcw[il],
                        wavenumbers_ifgm.size
                    )

                print(f"\t\t\t----")
                nb_tabs = int(2)
                best_idx = np.argmin(rmse_diagonal)
                print_metrics(
                    nb_tabs,
                    "RMSE_MIN",
                    best_idx,
                    lambdaas[best_idx],
                    rmse[best_idx],
                    rmse_diagonal[best_idx],
                    rmcw[best_idx],
                    wavenumbers_ifgm.size
                )
                best_idx = np.argmin(-rmcw)
                print_metrics(
                    nb_tabs,
                    "RMCW_MAX",
                    best_idx,
                    lambdaas[best_idx],
                    rmse[best_idx],
                    rmse_diagonal[best_idx],
                    rmcw[best_idx],
                    wavenumbers_ifgm.size
                )


if __name__ == "__main__":
    main()
