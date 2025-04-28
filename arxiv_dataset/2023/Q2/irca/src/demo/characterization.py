from pathlib import Path

import matplotlib.pyplot as plt

from src.interface.characterization import CharacterizationInterface


def main() -> None:

    acquisition_subfolder = "acquisitions/imspoc_uv_2"
    options_file = "characterization/imspoc_uv_2_characterization_options.json"
    device_file = "device/imspoc_uv_2.json"
    interferometer = 40

    data_folder = Path(__file__).resolve().parents[2] / "data"
    characterization = CharacterizationInterface.from_file(
        acquisition_folder=data_folder / acquisition_subfolder,
        method_json=data_folder / options_file,
        device_json=data_folder / device_file,
    )

    characterization.acquisition.visualize(interferometer=interferometer)
    parametrization = characterization.characterization()
    parametrization.visualize_opd()
    parametrization.visualize_reflectivity(
        wavenumber=characterization.acquisition.central_wavenumbers
    )
    characterization.visualize(interferometer=interferometer)

    plt.show()


if __name__ == "__main__":
    main()
