"""
Minimal working template for simulated inversion, without comments and extra functionality (saving arrays, etc.)
"""

import numpy as np
from matplotlib import pyplot as plt

from src.direct_model.interferometer import simulate_interferogram
from src.interface.configuration import load_config
from src.inverse_model.protocols import inversion_protocol_factory


def main():
    config = load_config()
    db = config.database()

    experiment_id = 0
    experiment_params = db.experiments[experiment_id]

    for ds_id in experiment_params.dataset_ids:
        spectra_ref = db.dataset_spectrum(ds_id=ds_id)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")

        for ifm_id in experiment_params.interferometer_ids:
            interferometer = db.interferometer(interferometer_id=ifm_id)
            print(f"\tInterferometer: {db.interferometers[ifm_id].title.upper()}")

            transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
            interferograms = simulate_interferogram(transmittance_response=transfer_matrix, spectrum=spectra_ref)

            fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
            interferograms.visualize(axs=axs[0, 0], acq_ind=0)
            plt.show()


if __name__ == "__main__":
    main()
