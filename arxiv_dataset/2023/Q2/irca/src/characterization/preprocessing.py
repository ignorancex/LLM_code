from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field, asdict, replace, fields
from pathlib import Path
import functools
import json

import numpy as np
import numpy.polynomial as npp

from src.lib.custom_vars import (
    Ifm,
    Acq,
    Chn,
    SerializerEnum,
    NumpyPolynomial,
    Fig,
    Axs, Row, Col,
)
from src.characterization.interface import ParameterSetSchema, \
    CallablePolynomial
from src.lib.visualize import plot_1d, plot_shaded
from src.lib.serialize import serializer_factory
from src.characterization.interferometry import (
    Interferometer,
    poly_eval,
    poly_eval_2d,
)


@dataclass(frozen=True)
class CharacterizationPixel:
    data: np.ndarray[tuple[Ifm, Chn], np.float64]
    central_wavenumbers: np.ndarray[tuple[Chn], np.float64] = field(
        metadata={"unit": "1 / micrometer"}
    )
    norm: np.ndarray[tuple[Ifm, Chn], np.float64] = None
    mean: np.ndarray[tuple[Ifm, Chn], np.float64] = None

    def __post_init__(self) -> None:
        mean = self.mean if self.mean is not None else self.data
        object.__setattr__(self, "mean", mean)
        if self.norm is None:
            norm = np.mean(self.data, axis=1, keepdims=True)
            norm = np.broadcast_to(array=norm, shape=self.data.shape)
        else:
            norm = self.norm
        object.__setattr__(self, "norm", norm)
        assert np.all(
            self.central_wavenumbers[:-1] <= self.central_wavenumbers[1:]
        )
        assert self.data.ndim == 2
        assert self.central_wavenumbers.ndim == 1
        assert self.data.shape == self.norm.shape
        assert self.data.shape == self.mean.shape
        assert self.data.shape[1] == self.central_wavenumbers.shape[0]

    def normalize(self) -> "CharacterizationPixel":
        """Normalizes the acquisition by the norm (which is set to 1)."""
        return replace(
            self,
            data=self.data/self.norm,
            mean=self.mean/self.norm,
            norm=np.ones_like(self.norm),
        )

    def select_data(
            self, flag_mean: bool
    ) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
        """Selects the mean data if the flag is true"""
        return self.mean if flag_mean else self.data

    def __getitem__(self, item: tuple[slice, slice]) -> "CharacterizationPixel":
        item = (item, slice(None)) if isinstance(item, slice) else item
        return replace(
            self,
            data=self.data[item],
            norm=self.norm[item],
            mean=self.mean[item],
            central_wavenumbers=self.central_wavenumbers[item[1]]
        )

    @classmethod
    def load(cls, folder: Path | str) -> "CharacterizationPixel":
        """Loads data from a serialized file"""
        serializer = serializer_factory(fmt=SerializerEnum.NUMPY)
        return serializer.load(class_obj=cls, folder_path=folder)

    def dump(self, folder: Path | str) -> None:
        serializer = serializer_factory(fmt=SerializerEnum.NUMPY)
        serializer.dump(instance=self, folder_path=folder)

    def visualize(
        self,
        interferometer: int = 0,
        font_size: int = 16,
        flag_mean: bool = False,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        """Plots the characterization acquisition, together with its norm"""
        data = self.mean if flag_mean else self.data
        data_mean = np.mean(data[interferometer, :])
        data_mean_one = data[interferometer, :] / data_mean
        norm_mean = np.mean(self.norm[interferometer, :])
        norm_mean_one = self.norm[interferometer, :] / norm_mean
        normalized_data = data_mean_one / norm_mean_one

        title = f"Acquisition (Interferometer index: {interferometer})"
        legend = ["Raw acquisition", "Flat field", "Normalized acquisition"]
        wavenumbers = self.central_wavenumbers * 10000
        fig, ax = plot_1d(
            x=wavenumbers,
            y=np.vstack((data_mean_one, norm_mean_one, normalized_data)),
            x_label=r"Central wavenumbers [$\mathrm{cm^{-1}}$]",
            y_label="Mean scaled intensity",
            legend=legend,
            font_size=font_size,
            figure=figure,
            x_limit=(wavenumbers.min(), wavenumbers.max()),
            scientific=True,
            reciprocal_axis=True,
            reciprocal_label=r"Central wavelengths [$\mathrm{nm}$]",
        )
        ax.set_title(title, fontsize=font_size+4)
        return fig, ax

    def visualize_compare(
        self,
        estimation: np.ndarray[tuple[Ifm, Acq], np.float64],
        interferometer: int = 0,
        difference: bool = False,
        font_size: int = 16,
        flag_mean: bool = False,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        """Compares the transfer function with an estimation"""
        data = self.mean if flag_mean else self.data
        scale_factor = np.mean(data[interferometer, :])
        data_scaled = data[interferometer, :] / scale_factor
        estimation_scaled = estimation[interferometer, :] / scale_factor
        if difference:
            y = data_scaled - estimation_scaled
            title = f"Fitting mismatch"
            legend = None
        else:
            y = np.vstack((data_scaled, estimation_scaled))
            title = f"Characterization"
            legend = ["Raw acquisition", "Fit transfer function"]
        wavenumbers = self.central_wavenumbers * 10000
        fig, ax = plot_1d(
            x=wavenumbers,
            y=y,
            x_label=r"Central wavenumbers [$\mathrm{c m^{-1}}$]",
            y_label="Mean scaled intensity",
            legend=legend,
            font_size=font_size,
            figure=figure,
            x_limit=(wavenumbers.min(), wavenumbers.max()),
            scientific=True,
            reciprocal_axis=True,
            reciprocal_label=r"Central wavelengths [$\mathrm{n m}$]",
        )
        title = title + f" (Interferometer index: {interferometer + 1})"
        ax.set_title(title, fontsize=font_size+4)
        return fig, ax


@dataclass(frozen=True)
class Parameters:
    opd: np.ndarray[tuple[Ifm, Any], np.float64] | float = field(
        metadata={"unit": "micrometer"}
    )
    gain: np.ndarray[tuple[Ifm, Any], np.float64] | float
    reflectivity: np.ndarray[tuple[Ifm, Any], np.float64] | float
    phase_shift: np.ndarray[tuple[Ifm, Any], np.float64] | float

    def __post_init__(self):
        """Validation checks"""
        for attrib in fields(self):
            attrib_array = np.array(getattr(self, attrib.name))
            if attrib_array.ndim == 0:
                attrib_array = attrib_array[np.newaxis, np.newaxis]
            if attrib_array.ndim == 1:
                attrib_array = attrib_array[np.newaxis, :]
            object.__setattr__(self, attrib.name, attrib_array)
            if attrib_array.ndim != 2:
                raise ValueError(f"{attrib.name} is not a 2-dimensional array")
            if getattr(self, attrib.name).shape[0] != self.opd.shape[0]:
                raise ValueError(f"{attrib.name} rows are not consistent")

    def __getitem__(self, item: slice | list[int]) -> "Parameters":
        """Selects a subset of interferometers"""
        dictionary = {}
        for key, value in asdict(self).items():
            dictionary.update({key: value[item, :]})
        return replace(self, **dictionary)

    @classmethod
    def init_zeros(cls, interferometers: int = 1) -> "Parameters":
        """Initializes the parameters with zeros, given the amount of rows"""
        out_dict = {}
        for attrib in fields(cls):
            name = attrib.name
            out_dict.update({name: np.zeros((interferometers, 1))})
        return cls(**out_dict)

    @classmethod
    def from_csv(cls, load_folder: Path) -> "Parameters":
        """Loads data from a serialized csv file"""
        serializer = serializer_factory(fmt=SerializerEnum.CSV)
        return serializer.load(class_obj=cls, folder_path=load_folder)

    def to_csv(self, save_folder: Path) -> None:
        """Dumps data to a serialized csv"""
        serializer = serializer_factory(fmt=SerializerEnum.CSV)
        serializer.dump(instance=self, folder_path=save_folder)

    def dump(self, save_folder: Path | str) -> None:
        serializer = serializer_factory(fmt=SerializerEnum.NUMPY)
        serializer.dump(instance=self, folder_path=save_folder)

    @classmethod
    def load(cls, load_folder: Path | str) -> "Parameters":
        """Loads data from a serialized numpy file"""
        serializer = serializer_factory(fmt=SerializerEnum.NUMPY)
        return serializer.load(class_obj=cls, folder_path=load_folder)

    def standardize(self, options: ParameterSetSchema) -> "Parameters":
        """
        Changes the numbers of columns of the attributes to match the
        degree specified by the options
        """
        replace_dict = {}
        for key, val in asdict(self).items():
            dimensions = (val.shape[0], getattr(options, key).degree + 1)
            val_new = np.zeros(dimensions)
            max_dimension = (np.minimum(val.shape[1], dimensions[1]))
            val_new[:, :max_dimension] = val[:, :max_dimension]
            replace_dict.update({key: val_new})
        return replace(self, **replace_dict)

    def to_parameters_matrix(
            self, fov: float = 0, subimage_size: tuple[int, int] = (1, 1),
    ) -> ParametersMatrix:
        cos_theta = angle_modulation(fov=fov, subimage_size=subimage_size)
        cos_theta = cos_theta.flatten()
        final_sizes = subimage_size + (self.opd.shape[0], -1)
        opd = np.kron(cos_theta[:, np.newaxis], self.opd).reshape(final_sizes)
        param_dict = {}
        for param in ["gain", "reflectivity", "phase_shift"]:
            ele = np.repeat(getattr(self, param), cos_theta.shape[0], axis=0)
            param_dict.update({param: ele.reshape(final_sizes)})
        return ParametersMatrix(opd=opd, **param_dict)


def angle_modulation(
    fov: float = 0, subimage_size: tuple[int, int] = (1, 1)
) -> np.ndarray[tuple[Row, Col], np.float_]:
    half_size = np.array(subimage_size) / 2
    row = np.arange(subimage_size[0]) - half_size[0]
    col = np.arange(subimage_size[1]) - half_size[1]
    coord = np.sqrt(row[:, np.newaxis] ** 2 + col[np.newaxis, :] ** 2)
    theta_max = fov / 180 * np.pi / coord[0, 0]
    return np.cos(theta_max * coord)


@dataclass(frozen=True)
class ParametersMatrix:
    opd: np.ndarray[tuple[Row, Col, Ifm, Any], np.float64] = field(
        metadata={"unit": "micrometer"}
    )
    gain: np.ndarray[tuple[Row, Col, Ifm, Any], np.float64]
    reflectivity: np.ndarray[tuple[Row, Col, Ifm, Any], np.float64]
    phase_shift: np.ndarray[tuple[Row, Col, Ifm, Any], np.float64]

    def to_parameters(self) -> Parameters:
        out_sizes = (np.prod(self.opd.shape[:3]), -1)
        params = ["opd", "gain", "reflectivity", "phase_shift"]
        out_dict = {x: getattr(self, x).reshape(out_sizes) for x in params}
        return Parameters(**out_dict)


@dataclass(frozen=True)
class Characterization:
    parameters: Parameters
    model: int = 0
    polynomial: NumpyPolynomial = npp.Polynomial

    @classmethod
    def load(cls, load_folder: Path | str) -> "Characterization":
        parameters = Parameters.load(load_folder=load_folder)
        load_json = Path(f"{load_folder}") / "options.json"
        with open(load_json) as json_file:
            json_dict = json.load(fp=json_file)
        model = int(json_dict["model"])
        polynomial = CallablePolynomial.direct(string=json_dict["polynomial"])
        return cls(parameters=parameters, model=model, polynomial=polynomial)

    def dump(self, save_folder: Path | str) -> None:
        self.parameters.dump(save_folder=save_folder)
        polynomial = CallablePolynomial.inverse(function=self.polynomial)
        dict_save = {"model": self.model, "polynomial": f"{polynomial}"}
        save_json = save_folder / "options.json"
        with open(save_json, "w", encoding="utf-8") as json_file:
            json.dump(dict_save, json_file, ensure_ascii=False, indent=4)

    def to_matrix(
        self,
        wavenumber: np.ndarray[tuple[Chn], np.float64],
    ) -> Interferometer:
        """
        Expresses the parameters in matrix form (as function of wavenumber
        samples in um^{-1})
        """
        out_dict = {"wavenumber": wavenumber}
        func = functools.partial(poly_eval, func=self.polynomial, x=wavenumber)
        for key, val in asdict(self.parameters).items():
            update = np.apply_along_axis(func1d=func, axis=1, arr=val)
            out_dict.update({key: update})
        return Interferometer(**out_dict)

    def transfer_function(
        self,
        wavenumber: np.ndarray[tuple[Chn], np.float64],
    ) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
        interferometer = self.to_matrix(wavenumber=wavenumber)
        return interferometer.transfer(model=self.model, is_scaled=True)

    def evaluate_parameter(
        self,
        wavenumber: np.ndarray[tuple[Chn], np.float64],
        parameter_name: str,
    ) -> np.ndarray[tuple[Ifm, Chn], np.float_]:
        return poly_eval_2d(
            coef=getattr(self.parameters, parameter_name),
            func=self.polynomial,
            x=wavenumber
        )

    def visualize_reflectivity(
        self,
        wavenumber: np.ndarray[tuple[Any], np.float_],
        font_size: float = 16,
        save_filepath: Path | str = None,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        reflectivity_show = self.evaluate_parameter(wavenumber, "reflectivity")
        wavenumber = wavenumber * 10000
        fig, ax = plot_shaded(
            y=reflectivity_show.T,
            x=wavenumber,
            save_filepath=save_filepath,
            x_label=r"Wavenumbers [$\mathrm{c m^{-1}}$]",
            y_label="Estimated reflectivity",
            font_size=font_size,
            figure=figure,
            x_limit=(np.min(wavenumber), np.max(wavenumber)),
            scientific=True,
            reciprocal_axis=True,
            reciprocal_label=r"Wavenumbers [$\mathrm{nm}$]",
        )
        return fig, ax

    def visualize_gain(
        self,
        wavenumber: np.ndarray[tuple[Any], np.float_],
        font_size: float = 16,
        save_filepath: Path | str = None,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        gain_show = poly_eval_2d(
            coef=self.parameters.gain,
            func=self.polynomial,
            x=wavenumber
        )
        fig, ax = plot_shaded(
            y=gain_show.T,
            x=wavenumber * 10000,
            save_filepath=save_filepath,
            x_label=r"Wavenumbers [$\mathrm{c m^{-1}}$]",
            y_label="Estimated gain",
            font_size=font_size,
            figure=figure,
            scientific=True,
            reciprocal_axis=True,
            reciprocal_label=r"Wavenumbers [$\mathrm{nm}$]",
        )
        return fig, ax

    def visualize_opd(
        self,
        color: str = None,
        wavenumber: np.ndarray[tuple[Chn], np.float64] = None,
        font_size: float = 16,
        save_filepath: Path | str = None,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        """Plots the characterization acquisition, together with its norm"""
        if wavenumber is None:
            opd_show = self.parameters.opd[:, 0]
        else:
            opd_show = poly_eval_2d(
                coef=self.parameters.opd, func=self.polynomial, x=wavenumber
            )
            opd_show = np.mean(opd_show, axis=1)

        return plot_1d(
            x=np.arange(len(opd_show)) + 1,
            y=opd_show,
            x_label=f"Interferometers' indices",
            y_label=r"Estimated OPD [$\mathrm{\mu m}$]",
            line_style="",
            marker="*",
            font_size=font_size,
            save_filepath=save_filepath,
            figure=figure,
            color=color,
        )

    def visualize(
        self,
        wavenumbers: np.ndarray[tuple[Chn], np.float64],
        interferometer: int = 0,
        font_size: float = 16,
        save_filepath: Path | str = None,
        figure: tuple[Fig, Axs] = None,
        normalize: bool = False,
        color: str = None,
    ) -> tuple[Fig, Axs]:
        """Plots the characterization acquisition, together with its norm"""
        transfer = self.transfer_function(wavenumber=wavenumbers)
        transfer_choice = transfer[interferometer, :]
        y_label = "Intensity"
        if normalize:
            transfer_choice = transfer_choice / np.mean(transfer_choice)
            y_label = "Mean scaled intensity"
        fig, ax = plot_1d(
            x=wavenumbers * 10000,  # to show in cm^{-1}
            y=transfer_choice,
            x_label=r"Wavenumbers [$\mathrm{c m^{-1}}$]",
            y_label=y_label,
            font_size=font_size,
            save_filepath=save_filepath,
            figure=figure,
            color=color,
            scientific=True,
            reciprocal_axis=True,
            reciprocal_label=r"Wavelengths [$\mathrm{nm}$]",
        )
        ax.set_title("Transfer function", fontsize=font_size+4)
        return fig, ax

    def visualize_compare(
        self,
        wavenumbers: np.ndarray[tuple[Chn], np.float64],
        transfer: np.ndarray[tuple[Ifm, Chn], np.float64],
        interferometer: int = 0,
        font_size: float = 16,
        save_filepath: Path | str = None,
        figure: tuple[Fig, Axs] = None,
    ) -> tuple[Fig, Axs]:
        """Compares the characterization with a given transfer function"""
        transfer_pick = transfer[interferometer, :]
        transfer_mean = np.mean(transfer_pick)
        transfer_pick = transfer_pick / transfer_mean
        transfer_local = self.transfer_function(wavenumber=wavenumbers)
        transfer_local = transfer_local[interferometer, :] / transfer_mean
        transfer_total = np.vstack((transfer_pick, transfer_local))
        wavenumbers = wavenumbers * 10000
        fig, ax = plot_1d(
            x=wavenumbers,  # to show in cm^{-1}
            y=transfer_total,
            x_label=r"Wavenumbers [$\mathrm{cm^{-1}}$]",
            y_label="Mean scaled intensity",
            font_size=font_size,
            save_filepath=save_filepath,
            figure=figure,
            legend=["Reference", "Estimated"],
            x_limit=(wavenumbers.min(), wavenumbers.max()),
            scientific=True,
            reciprocal_axis=True,
            reciprocal_label=r"Wavelengths [$\mathrm{nm}$]",
        )
        ax.set_title("Characterization", fontsize=font_size+4)
        return fig, ax
