from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from src.interface.configuration import load_config


def main():
    db = load_config().database()

    for interferometer_id in range(len(db.interferometers)):
        interferometer_schema = db.interferometers[interferometer_id]
        pprint(interferometer_schema)

        interferometer = db.interferometer(interferometer_id=interferometer_id)
        pprint(interferometer.transmittance_coefficients)
        pprint(interferometer.opds)

        transmat = interferometer.transmittance_response(wavenumbers=np.linspace(start=0, stop=2.85, num=276))
        plt.imshow(transmat.data)
        plt.show()

        print()


if __name__ == "__main__":
    main()
