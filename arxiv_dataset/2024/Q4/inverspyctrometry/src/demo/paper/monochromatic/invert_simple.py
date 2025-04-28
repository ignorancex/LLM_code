from dataclasses import replace
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import match_stats
from src.interface.configuration import load_config


# TODO: Crop the acquisitions and wavenumbers within an interval of interest


def main():
    config = load_config()
    db = config.database()

    experiment_id = 1
    experiment_params = db.experiments[experiment_id]

    for ds_id in experiment_params.dataset_ids:
        interferograms_ref = db.dataset_interferogram(ds_id=ds_id)
        interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
        wavenumbers_ref = db.dataset_central_wavenumbers(dataset_id=ds_id)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")

        for char_id in experiment_params.interferometer_ids:
            characterization = db.characterization(characterization_id=char_id)
            print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")

            transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers_ref)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

            for ip_id in experiment_params.inversion_protocol_ids:
                lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")

                for lambdaa in lambdaas:
                    inverter = db.inversion_protocol(inv_protocol_id=ip_id, lambdaa=lambdaa)
                    spectra_rec = inverter.reconstruct_spectrum(
                        interferogram=interferograms_ref, transmittance_response=transfer_matrix
                    )

                    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
                    spectra_rec.visualize(axs=axs[0, 0], acq_ind=0)
                    plt.show()


if __name__ == "__main__":
    main()
