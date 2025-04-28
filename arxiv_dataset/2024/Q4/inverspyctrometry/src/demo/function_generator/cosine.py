from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.function_generator import CosineGenerator


def main():
    wavenumbers = np.linspace(start=0, stop=5, num=501, endpoint=True)
    cos_gen = CosineGenerator(
        coefficients=np.array([[1., 0.3], [1., 0.3]]).T,
        frequencies=np.array([[1., 2.], [1., 3.1]]).T,  # Hz
    )
    cos_funcs = cos_gen.generate_data(variable=wavenumbers)

    fig, axs = plt.subplots(1, 1, squeeze=False)
    axs[0, 0].plot(wavenumbers, cos_funcs[:, 0], color="C0")
    axs[0, 0].plot(wavenumbers, cos_funcs[:, 1], color="C1")
    plt.show()


if __name__ == "__main__":
    main()
