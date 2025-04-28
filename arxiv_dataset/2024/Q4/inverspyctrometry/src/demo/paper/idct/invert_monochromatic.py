from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import calculate_rmse, match_stats
from src.demo.paper.monochromatic.utils import calculate_rmcw
from src.interface.configuration import load_config
from src.inverse_model.protocols import inversion_protocol_factory


def main():
    config = load_config()
    db = config.database()
    plot_options = {
        "figsize": (8, 6),
        "fontsize": 25,
    }

    experiment_id = 6
    experiment_params = db.experiments[experiment_id]
    pprint(dict(experiment_params))

    for ds_id in experiment_params.dataset_ids:
        interferograms_ref = db.dataset_interferogram(ds_id=ds_id)
        wavenumbers_ifgm = np.load(db.datasets[ds_id].wavenumbers_path)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")

        interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
        interferograms_ref = interferograms_ref.center(new_mean=0, axis=-2)
        spectra_ref = Spectrum(data=np.eye(wavenumbers_ifgm.size), wavenumbers=wavenumbers_ifgm)

        fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=plot_options["figsize"], tight_layout=True)
        spectra_ref.visualize_matrix(fig=fig, axs=axs[0, 0])

        for char_id in experiment_params.interferometer_ids:
            characterization = db.characterization(characterization_id=char_id)
            char_ds_id = db.characterizations[char_id].source_dataset_id
            wavenumbers_char = np.load(db.datasets[char_ds_id].wavenumbers_path)
            print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")

            reflectance = characterization.reflectance(wavenumbers=wavenumbers_ifgm)
            transmittance = characterization.transmittance(wavenumbers=wavenumbers_ifgm)
            transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers_ifgm)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

            for ip_id in experiment_params.inversion_protocol_ids:
                ip_schema = db.inversion_protocols[ip_id]
                lambdaas = ip_schema.lambdaas_schema.as_array()
                print(f"\t\t\tInvProtocol: {ip_schema.title.upper()}")

                for lambdaa in lambdaas:
                    ip_kwargs = ip_schema.parameters(lambdaa=lambdaa)
                    inverter = inversion_protocol_factory(option=ip_schema.type, parameters=ip_kwargs)
                    spectra_rec = inverter.reconstruct_spectrum(
                        interferogram=interferograms_ref, transmittance_response=transfer_matrix
                    )

                    # EXTRA: Calculate some statistics
                    reconstruction, reference = match_stats(
                        array=spectra_rec.data,
                        reference=spectra_ref.data,
                        axis=-2,
                    )
                    rmse = calculate_rmse(
                        array=reconstruction,
                        reference=reference,
                    )
                    rmse_diagonal = calculate_rmse(
                        array=np.diag(reconstruction),
                        reference=np.diag(reference),
                    )
                    rmse_max = calculate_rmse(
                        array=np.max(reconstruction, axis=-2),
                        reference=np.max(reference, axis=-2),
                    )
                    rmcw = calculate_rmcw(monochromatic_array=reconstruction)

                    print(
                        f"\t\t\t\t"
                        f"Lambda: {lambdaa:.4f},\t"
                        f"RMSE: {rmse:.4f},\t"
                        f"RMSE_DIAG: {rmse_diagonal:.4f},\t"
                        f"RMSE_MAX: {rmse_max:.4f},\t"
                        f"RMCW: {rmcw:.4f}"
                    )

                    spectra_rec = Spectrum(data=reconstruction, wavenumbers=spectra_rec.wavenumbers)
                    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=plot_options["figsize"], tight_layout=True)
                    spectra_rec.visualize_matrix(fig=fig, axs=axs[0, 0])
                plt.show()


if __name__ == "__main__":
    main()
