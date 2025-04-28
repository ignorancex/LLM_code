from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from src.interface.configuration import load_config


def main():
    config = load_config()
    db = config.database()

    characterization_id = 0
    characterization_schema = db.characterizations[characterization_id]
    pprint(dict(characterization_schema))

    characterization = db.characterization(characterization_id=characterization_id)

    ds_id = db.characterizations[characterization_id].source_dataset_id
    interferograms = db.dataset_interferogram(ds_id=ds_id)

    central_wavenumbers = np.load(db.datasets[ds_id].wavenumbers_path)
    transmittance_response = characterization.transmittance_response(wavenumbers=central_wavenumbers)

    plot_options = {
        "figsize": (8, 6),
        "fontsize": 25,
    }
    plt.rcParams['font.size'] = str(plot_options["fontsize"])

    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=plot_options["figsize"], tight_layout=True)
    interferograms.visualize_matrix(fig=fig, axs=axs[0, 0])

    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=plot_options["figsize"], tight_layout=True)
    transmittance_response.visualize(fig=fig, axs=axs[0, 0])

    plt.show()


if __name__ == "__main__":
    main()
