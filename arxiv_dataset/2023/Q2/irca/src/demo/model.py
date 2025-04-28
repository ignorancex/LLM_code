from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.characterization.preprocessing import (
    Characterization,
    Parameters,
    CharacterizationPixel,
)


def interferometer_example() -> None:
    opd = 10
    reflectivity = 0.13
    gain = 1
    phase_shift = 0
    model = 0
    wavenumbers = np.linspace(1, 2, 10000, endpoint=False)

    interferometer = Characterization(
        Parameters(
            opd=opd,
            reflectivity=reflectivity,
            gain=gain,
            phase_shift=phase_shift,
        ),
        model=model,
    )
    interferometer.visualize(wavenumbers=wavenumbers)
    # plt.show()


def comparison_example() -> None:
    interferometer = 40
    project_folder = Path(__file__).resolve().parents[2]
    reference_folder = project_folder / "data/acquisitions/imspoc_uv_2"
    acquisition = CharacterizationPixel.load(folder=reference_folder)
    transfer = acquisition.data[interferometer, :][np.newaxis, :]
    wavenumbers = acquisition.central_wavenumbers

    opd = 8.58
    reflectivity = 0.29
    gain = 4.73e+9
    phase_shift = 3.083
    model = 0

    interferometer = Characterization(
        Parameters(
            opd=opd,
            reflectivity=reflectivity,
            gain=gain,
            phase_shift=phase_shift,
        ),
        model=model,
    )
    _, ax = interferometer.visualize_compare(
        transfer=transfer, wavenumbers=wavenumbers
    )
    ax.set_ylim((0, 3))
    plt.show()


def main() -> None:
    interferometer_example()
    comparison_example()


if __name__ == "__main__":
    main()
