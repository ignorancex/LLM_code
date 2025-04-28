from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.characterization.preprocessing import Characterization
from src.lib.custom_vars import Row, Col, Chn, Ifm, OriginEnum, Fig, Axs
from src.interface.input import Imspoc, ImspocOptions
from src.lib.serialize import NumpySerializer
from src.lib.visualize import plot_1d


@dataclass(frozen=True)
class SimulatedImspoc:
    image: np.ndarray[tuple[Row, Col, Chn], np.float_]
    transfer: np.ndarray[tuple[Row, Col, Ifm, Chn], np.float_]
    centers: np.ndarray[tuple[Ifm, Literal[2]], np.float_]
    image_size: Optional[tuple[int, int]] = None
    modulation: Optional[tuple[Row, Col, Ifm], np.float_] = None
    origin: OriginEnum = OriginEnum.LOWER

    def __post_init__(self):
        if self.transfer.ndim != 4 or self.image.ndim != 3:
            raise ValueError("Dimensions not consistent!")
        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError("Centers dimensions are not consistent")
        if self.centers.shape[0] != self.transfer.shape[2]:
            raise ValueError("Interferometer dimensions are not consistent")
        if not self._validator_size(self.image, self.transfer[:, :, 0, :]):
            raise ValueError("Image dimensions must be consistent")
        if self.modulation is None:
            object.__setattr__(self, "modulation", np.ones((1, 1, 1)))
        if self.modulation.ndim != 3:
            raise ValueError("Modulation dimensions must be consistent")
        if not self._validator_size(self.image, self.modulation):
            raise ValueError("Modulation dimensions not consistent")
        if self.image_size is None:
            shape = np.ndarray(self.image.shape)[:2][np.newaxis, :]
            image_size = np.max(self._corners + shape, axis=0)
            object.__setattr__(self, "image_size", image_size)

    @staticmethod
    def _validator_size(a: np.ndarray, b: np.ndarray) -> bool:
        for shape_1, shape_2 in zip(a.shape, b.shape):
            if not (shape_1 == shape_2 or shape_1 == 1 or shape_2 == 1):
                return False
        return True

    @classmethod
    def from_device(
        cls,
        image: np.ndarray[tuple[Row, Col, Chn], np.float_],
        transfer: np.ndarray[tuple[Row, Col, Ifm, Chn], np.float_],
        device: ImspocOptions,
        fov: float,
    ) -> "SimulatedImspoc":
        modulation = cls._angle_modulation(
            fov=fov, subimage_size=device.subimage_size
        )
        return cls(
            image=image,
            transfer=transfer,
            centers=device.centers,
            image_size=device.image_size,
            modulation=modulation[:, :, np.newaxis],
        )

    @staticmethod
    def _angle_modulation(
        fov: float = 0, subimage_size: tuple[int, int] = (1, 1)
    ) -> np.ndarray[tuple[Row, Col], np.float_]:
        half_size = np.array(subimage_size) / 2
        row = np.arange(subimage_size[0]) - half_size[0]
        col = np.arange(subimage_size[1]) - half_size[1]
        coord = np.sqrt(row[:, np.newaxis] ** 2 + col[np.newaxis, :] ** 2)
        theta_max = fov / 180 * np.pi / coord[0, 0]
        return np.cos(theta_max * coord)

    @property
    def _corners(self) -> np.ndarray[tuple[Ifm, Literal[2]], np.int_]:
        half_size = np.array(self.image.shape[:2])[np.newaxis, :] // 2
        return np.array(self.centers - half_size, np.int_)

    def datacube(self) -> np.ndarray[tuple[Row, Col, Ifm], np.float_]:
        image = self.image[::-1, :, :] if self.origin == OriginEnum.LOWER else self.image
        out = np.sum(image[:, :, np.newaxis, :] * self.transfer, axis=3)
        out = out * self.modulation
        return out

    def _unstacked_image(self) -> np.ndarray[tuple[int, int], np.float_]:
        datacube = self.datacube()
        corner = self._corners
        shape = datacube.shape[:2]
        image = np.zeros(self.image_size)
        max_pad = np.array(datacube.shape[:2]) - np.array(self.image_size)
        for ii in np.arange(corner.shape[0]):
            pad_min = np.maximum(- corner[ii, :], 0)
            pad_max = np.maximum(corner[ii, :] + max_pad, 0)
            sli_i = [slice(pad_min[i], shape[i] - pad_max[i]) for i in [0, 1]]
            sli_o = [slice(corner[ii, i] + pad_min[i], corner[ii, i] + shape[i] - pad_max[i]) for i in [0, 1]]
            image[sli_o[0], sli_o[1]] = datacube[sli_i[0], sli_i[1], ii]
        return image

    def imspoc(self) -> Imspoc:
        image = self._unstacked_image()[:, :, np.newaxis]
        return Imspoc(
            centers=self.centers,
            size=self.image.shape[:2],
            image=image,
            image_size=image.shape[:2],
        )


@dataclass(frozen=True)
class HyperspectralImage:
    image: np.ndarray[tuple[Row, Col, Chn], np.float_]
    wavenumbers: np.ndarray[tuple[Chn], np.float_]

    def __post_init__(self) -> None:
        image = self.image / np.max(self.image)
        object.__setattr__(self, "image", image)

    def _get_transfer_matrix(
        self,
        characterization: Characterization,
    ) -> np.ndarray[tuple[Row, Col, Ifm, Chn], np.float_]:
        transfer = Characterization(
            parameters=characterization.parameters,
            model=characterization.model,
            polynomial=characterization.polynomial,
        ).transfer_function(wavenumber=self.wavenumbers)
        return transfer[np.newaxis, np.newaxis, :, :]

    def to_imspoc(
        self,
        device: ImspocOptions,
        characterization: Characterization,
        fov: float = 0,
    ) -> Imspoc:
        transfer = self._get_transfer_matrix(
            characterization=characterization
        )
        sim = SimulatedImspoc.from_device(
            image=self.image,
            device=device,
            transfer=transfer / np.mean(transfer, axis=3, keepdims=True),
            fov=fov,
        )
        return sim.imspoc()

    def visualize(
        self,
        rgb: tuple[int, int, int],
        point: tuple[int, int] = None,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        fig, ax = plt.subplots() if figure is None else figure
        image_rgb = self.image[:, :, rgb]
        ax.imshow(image_rgb / np.max(image_rgb))
        if point is not None:
            ax.scatter(x=point[1], y=point[0], marker="+", facecolor="r")
        return fig, ax

    def visualize_spectrum(
        self,
        point: tuple[int, int] = None,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        if point is None:
            point = self.image.shape[0] // 2, self.image.shape[1] // 2
        wavenumbers = self.wavenumbers * 10000
        fig, ax = plot_1d(
            x=wavenumbers,
            y=self.image[point[0], point[1], :],
            x_label=r"Wavenumbers [$\mathrm{c m^{-1}}$]",
            y_label="Normalized intensity",
            figure=figure,
            x_limit=(wavenumbers.min(), wavenumbers.max()),
            color="C4",
            scientific=True,
            reciprocal_axis=True,
            reciprocal_label=r"Wavelengths [$\mathrm{nm}$]",
        )
        ax.set_title("Spectrum", fontsize=20)
        return fig, ax

    def dump(self, save_folder: Path | str) -> None:
        NumpySerializer.dump(instance=self, folder_path=save_folder)

    @classmethod
    def load(cls, load_folder: Path | str) -> "HyperspectralImage":
        return NumpySerializer.load(class_obj=cls, folder_path=load_folder)
