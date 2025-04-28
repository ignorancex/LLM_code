from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.function_generator import VoigtGenerator


def main():
    options = {
        "single_component": {
            "coefficients": np.array([[1.], [0.5]]).T,
            "centers": np.array([[2.5], [3.75]]).T,
            "gauss_stds": np.array([[0.25], [0.125]]).T,
            "lorentz_scales": np.array([[0.25], [0.125]]).T,
        },
        "multiple_components": {
            "coefficients": np.array([[1., 0.9, 0.75], [1.1, 0.6, 0.5]]).T,
            "centers": np.array([[1.5, 2.1, 2.7], [3.2, 2.1, 1.]]).T,
            "gauss_stds": np.array([[0.08, 0.28, 0.1], [0.2, 0.4, 0.15]]).T,
            "lorentz_scales": np.array([[0.08, 0.28, 0.1], [0.2, 0.4, 0.15]]).T,
        },
    }

    wavenumbers = np.arange(0, 5, 0.01)
    voigt_gen = VoigtGenerator(**options["multiple_components"])
    voigt_funcs = voigt_gen.generate_data(variable=wavenumbers)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(10, 6))
    axs[0, 0].plot(wavenumbers, voigt_funcs[:, 0], color="C0", label='Voigt 1', linestyle='dashed')
    axs[0, 0].plot(wavenumbers, voigt_funcs[:, 1], color="C1", label='Voigt 2', linestyle='dashed')
    plt.legend()
    plt.title('Voigt Profile')
    plt.xlabel('variable')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
