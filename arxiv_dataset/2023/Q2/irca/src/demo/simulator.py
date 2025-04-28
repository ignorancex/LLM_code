from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

from src.interface.input import ImspocOptions, Imspoc
from src.characterization.preprocessing import Characterization
from src.interface.imspoc_simulator import HyperspectralImage


def imspoc_visualization(
    hyperspectral: HyperspectralImage,
    imspoc: Imspoc,
    rgb: tuple[int, int, int],
    point_x: int = 30,
    point_y: int = 40,
):
    point = (point_y, point_x)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(17, 10))
    hyperspectral.visualize(rgb=rgb, point=point, figure=(fig, ax[0, 0]))
    hyperspectral.visualize_spectrum(point=point, figure=(fig, ax[0, 1]))
    imspoc.visualize(points=[point], figure=(fig, ax[1, 0]))
    imspoc.visualize_interferogram(point=point, figure=(fig, ax[1, 1]))
    ax[0, 0].set_title("Hyperspectral image", fontsize=20)
    ax[1, 0].set_title("Multi-aperture FP spectrometer image", fontsize=20)
    ax[1, 0].set_xlim((1000, 1400))
    ax[1, 0].set_ylim((320, 700))
    ax[0, 1].set_ylim([0, 1])
    ax[1, 1].set_ylim([0.8, 1.3])
    fig.tight_layout()


def main():
    characterization_subfolder = "characterization/imspoc_uv_2"
    device_subfolder = "device/imspoc_uv_2.json"
    hyperspectral_subfolder = "hyperspectral/color_checker"
    rgb = (70, 53, 19)
    x_limits = (1000, 1400)
    y_limits = (320, 700)

    data_folder = Path(__file__).resolve().parents[2] / "data"

    characterization_folder = data_folder / characterization_subfolder
    device_folder = data_folder / device_subfolder
    hyperspectral_folder = data_folder / hyperspectral_subfolder

    device = ImspocOptions.parse_file(device_folder)
    hyperspectral = HyperspectralImage.load(hyperspectral_folder)
    characterization = Characterization.load(characterization_folder)
    imspoc = hyperspectral.to_imspoc(
        device=device,
        characterization=characterization,
    )

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

    hyperspectral.visualize(rgb=rgb, figure=(fig, ax[0]))
    imspoc.visualize(figure=(fig, ax[1]))
    # cube = imspoc.cube()  # Returns the subimages stacked as a datacube
    ax[0].set_title("Hyperspectral image")
    ax[1].set_title("Multi-aperture Fabry-Perot spectrometer image (zoom)")
    ax[1].set_xlim(x_limits)
    ax[1].set_ylim(y_limits)
    fig.tight_layout()

    # Fixed visualizetion
    imspoc_visualization(hyperspectral=hyperspectral, imspoc=imspoc, rgb=rgb)

    plt.show()


if __name__ == "__main__":
    main()
