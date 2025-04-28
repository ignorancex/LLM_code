from __future__ import annotations
from typing import Literal, Tuple, Optional
from pathlib import Path
import json
from dataclasses import dataclass, field
from pydantic import BaseModel, conint, confloat

import numpy as np
import matplotlib.pyplot as plt

from src.lib.custom_vars import Ifm, Acq, Row, Col, Fig, Axs, OriginEnum
from src.lib.visualize import imshow_enhanced, plot_1d


class ImspocOptions(BaseModel):
    image_size: Tuple[conint(ge=0), conint(ge=0)]
    subimage_size: Tuple[conint(ge=0), conint(ge=0)]
    subimage_amount: Tuple[conint(ge=0), conint(ge=0)]
    subimage_center: Tuple[confloat(ge=0), confloat(ge=0)]
    subimage_arrangement: str
    thickness_minimum: confloat(ge=0)
    thickness_step: confloat(ge=0)
    fov: Optional[confloat(ge=0)] = 0

    @property
    def centers(self) -> np.ndarray[tuple[Ifm, Literal[2]], np.float_]:
        amount = self.subimage_amount
        start = self.subimage_center
        size = self.subimage_size
        if self.subimage_arrangement == "vertical":
            idx = np.indices(amount)
            idx[1::2, :, :] = idx[1::2, :, ::-1]
            idx[0, :, :] = start[0] + size[0] * idx[0, :, :]
            idx[1, :, :] = start[1] + size[1] * idx[1, :, :]
            idx = np.reshape(idx, newshape=(2, -1), order="f").T
        else:
            idx = np.indices(amount)
            idx[:, 1::2, :] = idx[:, 1::2, ::-1]
            idx[0, :, :] = start[0] + size[0] * idx[0, :, :]
            idx[1, :, :] = start[1] + size[1] * idx[1, :, :]
            idx = idx[:, ::-1, ::-1]
            idx = np.reshape(idx, newshape=(2, -1), order="a").T
        return idx

    @property
    def opd(self) -> np.ndarray[tuple[Ifm], np.float_]:
        amount = np.prod(self.subimage_amount)
        thickness_vector = np.arange(amount) * self.thickness_step
        return (self.thickness_minimum + thickness_vector) / 500


@dataclass(frozen=True)
class Imspoc:
    image: np.ndarray[tuple[Row, Col, Acq], np.float_]
    centers: np.ndarray[tuple[Ifm, Literal[2]], np.float_]
    image_size: tuple[int, int]
    size: tuple[int, int]
    _corners: np.ndarray[tuple[Ifm, Literal[2]], np.float_] = field(init=False, default=None)
    _cube: np.ndarray[tuple[Row, Col, Ifm, Acq], np.float_] = field(init=False, default=None)

    @classmethod
    def from_file(
        cls, options: str | Path, image: str | Path = None,
    ) -> "Imspoc":
        with open(options) as json_file:
            json_dict = json.load(json_file)
            imspoc_options = ImspocOptions(**json_dict)
        if image is None:
            image = np.zeros((1, 1, 1))
        else:
            image = np.load(image)
        return cls(
            image=image,
            centers=imspoc_options.centers,
            size=imspoc_options.subimage_size,
            image_size=imspoc_options.image_size,
        )

    @classmethod
    def from_cube(
        cls,
        cube: np.ndarray[tuple[Row, Col, Ifm, Acq], np.float_],
        options: str | Path,
    ) -> "Imspoc":
        with open(options) as json_file:
            json_dict = json.load(json_file)
            imspoc_options = ImspocOptions(**json_dict)
        centers = imspoc_options.centers
        size = imspoc_options.subimage_size
        image = cube_to_image(
            cube=cube,
            image_size=imspoc_options.image_size,
            corners=evaluate_corners(size=size, centers=centers)
        )
        return cls(
            image=image,
            centers=centers,
            size=size,
            image_size=imspoc_options.image_size,
        )

    def grid(self) -> np.ndarray[tuple[Literal[2], Row, Col], np.float_]:
        return np.indices((self.size[0], self.size[1])) + 0.5

    @property
    def corners(self) -> np.ndarray[tuple[Ifm, Literal[2]], np.int32]:
        if self._corners is not None:
            return self._corners
        corners = evaluate_corners(size=self.size, centers=self.centers)
        object.__setattr__(self, "_corners", corners)
        return corners

    def cube(self) -> np.ndarray[tuple[Row, Col, Ifm, Acq], np.float_]:
        if self._cube is not None:
            return self._cube
        size = (self.size[0], self.size[1], self.centers.shape[0])
        shape = (self.image.shape[0], self.image.shape[1])
        out = np.zeros((size[0], size[1], size[2], self.image.shape[2]))
        corners = self.corners
        for ii in np.arange(corners.shape[0]):
            sx1 = np.maximum(-corners[ii, 0], 0)
            sx2 = np.maximum(corners[ii, 0] + size[0] - shape[0], 0)
            sy1 = np.maximum(-corners[ii, 1], 0)
            sy2 = np.maximum(corners[ii, 1] + size[1] - shape[1], 0)
            rng_ix = slice(sx1, size[0] - sx2)
            rng_iy = slice(sy1, size[1] - sy2)
            rng_ox = slice(corners[ii, 0]+size[0]-sx2-1, corners[ii, 0]+sx1-1, -1)
            rng_oy = slice(corners[ii, 1]+sy1, corners[ii, 1]+size[1]-sy2)
            out[rng_ix, rng_iy, ii, :] = self.image[rng_ox, rng_oy, :]
        object.__setattr__(self, "_cube", out)
        return out

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Returns the shape of the cube"""
        size = (self.size[0], self.size[1])
        return size[0], size[1], self.centers.shape[0], self.image.shape[2]

    def visualize(
        self,
        acquisition: int = 0,
        points: list[tuple[float, float]] = None,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        """
        Visualizes a single acquisition as cropped subimages.
        The method also allows displaying extra points in each subimage.
        """
        corners = self.corners
        if points is not None:
            points = np.array(points)[np.newaxis, :, :]
            points[:, :, 0] = self.shape[0] - points[:, :, 0]
            corners_ext = corners[:, np.newaxis, :]
            points = (points + corners_ext).reshape((-1, 2))
        size_broadcast = np.broadcast_to([self.size], corners.shape)
        rectangle = np.hstack((corners, size_broadcast))
        return imshow_enhanced(
            image=self.image[:, :, acquisition],
            rectangle=rectangle,
            origin=OriginEnum.LOWER,
            points=points,
            figure=figure,
            center_color=None,
            point_style={"marker": "+", "c": "r"},
        )

    def visualize_datacube(
        self,
        interferometer: int = 0,
        acquisition: int = 0,
        point: tuple[float, float] = None,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        """
        Visualizes a single acquisition as cropped subimages.
        The method also allows displaying extra points in each subimage.
        """
        fig, ax = plt.subplots() if figure is None else figure
        cube = self.cube()[:, :, interferometer, acquisition]
        ax.imshow(cube)
        if point is not None:
            ax.scatter(x=point[1], y=point[0], marker="+", facecolor="r")
        return fig, ax

    def visualize_interferogram(
        self,
        acquisition: int = 0,
        point: tuple[float, float] = None,
        figure: tuple[Fig, Axs] = None,
        normalize: bool = True,
    ) -> tuple[Fig, Axs]:
        """
        Visualizes a single acquisition as cropped subimages.
        The method also allows displaying extra points in each subimage.
        """
        if point is None:
            point = self.shape[0] // 2, self.shape[1] // 2
        interferogram = self.cube()[point[0], point[1], :, acquisition]
        y_label = "Intensity"
        if normalize:
            interferogram = interferogram / np.mean(interferogram)
            y_label = "Mean scaled Intensity"
        interferogram_shape = self.shape[2]
        fig, ax = plot_1d(
            x=np.arange(interferogram_shape),
            y=interferogram,
            x_label="Interferometers' index",
            y_label=y_label,
            figure=figure,
            x_limit=(0, interferogram_shape-1),
            color="y",
        )
        ax.set_title("Interferogram", fontsize=20)
        ax.set_facecolor("k")
        ax.grid(color="gray")
        return fig, ax


def evaluate_corners(
    size: tuple[int, int],
    centers: np.ndarray[tuple[Ifm, Literal[2]], np.float_]
) -> np.ndarray[tuple[Ifm, Literal[2]], np.float_]:
    size = np.array(size)[np.newaxis, :]
    return np.around(np.array(centers) - size / 2).astype(np.int32)


def cube_to_image(
    cube: np.ndarray[tuple[Row, Col, Ifm, Acq], np.float_],
    image_size: tuple[int, int],
    corners: np.ndarray[tuple[Ifm, Literal[2]]],
) -> np.ndarray[tuple[Row, Col, Acq], np.float_]:
    shape = image_size
    size = cube.shape[:2]
    image = np.zeros((shape[0], shape[1], cube.shape[3]))
    for ii in np.arange(corners.shape[0]):
        sx1 = np.maximum(-corners[ii, 0], 0)
        sx2 = np.maximum(corners[ii, 0] + size[0] - shape[0], 0)
        sy1 = np.maximum(-corners[ii, 1], 0)
        sy2 = np.maximum(corners[ii, 1] + size[1] - shape[1], 0)
        rng_ix = slice(sx1, size[0] - sx2)
        rng_iy = slice(sy1, size[1] - sy2)
        rng_ox = slice(corners[ii, 0] + size[0] - sx2 - 1,
                       corners[ii, 0] + sx1 - 1, -1)
        rng_oy = slice(corners[ii, 1] + sy1, corners[ii, 1] + size[1] - sy2)
        image[rng_ox, rng_oy, :] = cube[rng_ix, rng_iy, ii, :]
    return image
