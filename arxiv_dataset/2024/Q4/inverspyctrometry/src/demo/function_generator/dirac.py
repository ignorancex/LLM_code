from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.function_generator import DiracGenerator


def main():
    wavenumbers = np.linspace(start=0, stop=5, num=21, endpoint=True)
    dirac_gen = DiracGenerator(
        coefficients=np.array([[0.5, 0.75, 1.], [1.1, 0.85, 0.6]]).T,
        shifts=np.array([[1.5, 2.7, 4.5,], [1., 2., 4.,]]).T,
    )
    dirac_funcs = dirac_gen.generate_data(variable=wavenumbers)

    fig, axs = plt.subplots(1, 1, squeeze=False)
    dirac_funcs = np.where(dirac_funcs != 0., dirac_funcs, np.nan)
    axs[0, 0].stem(wavenumbers, dirac_funcs[:, 0], linefmt="C0", markerfmt="^")
    axs[0, 0].stem(wavenumbers, dirac_funcs[:, 1], linefmt="C1", markerfmt="^")
    plt.show()


if __name__ == "__main__":
    main()
