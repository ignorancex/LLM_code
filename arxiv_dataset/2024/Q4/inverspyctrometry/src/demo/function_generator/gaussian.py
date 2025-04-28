from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.function_generator import GaussianGenerator


def main():
    options = {
        "single_component": {
            "coefficients": np.array([[1.], [0.5]]).T,
            "means": np.array([[1.5], [1.]]).T,
            "stds": np.array([[0.08], [0.15]]).T,
        },
        "multiple_components": {
            "coefficients": np.array([[1., 0.9, 0.75], [1.1, 0.6, 0.5]]).T,
            "means": np.array([[1.5, 2.1, 2.7], [3.2, 2.1, 1.]]).T,
            "stds": np.array([[0.08, 0.28, 0.1], [0.2, 0.4, 0.15]]).T,
        },
    }

    wavenumbers = np.linspace(start=0, stop=5, num=501, endpoint=True)
    gauss_gen = GaussianGenerator(**options["multiple_components"])
    gauss_funcs = gauss_gen.generate_data(variable=wavenumbers)

    fig, axs = plt.subplots(1, 1, squeeze=False)
    axs[0, 0].plot(wavenumbers, gauss_funcs[:, 0], color="C0")
    axs[0, 0].plot(wavenumbers, gauss_funcs[:, 1], color="C1")
    plt.show()


if __name__ == "__main__":
    main()
