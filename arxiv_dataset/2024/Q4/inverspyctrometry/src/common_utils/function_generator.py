from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import convolve

from src.common_utils.custom_vars import Acq, Wvn


@dataclass
class FunctionGenerator(ABC):
    coefficients: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    @abstractmethod
    def generate_data(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        pass


@dataclass
class DiracGenerator(FunctionGenerator):
    shifts: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def generate_data(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        distances_from_shifts = np.abs(self.shifts[:, :, None] - variable[None, None, :])
        shifts_idxs = np.argmin(distances_from_shifts, axis=-1)
        acq_idxs = np.arange(self.shifts.shape[-1])
        data = np.zeros(shape=(variable.size, self.shifts.shape[-1]))
        data[shifts_idxs[:, acq_idxs], acq_idxs] = self.coefficients[:, acq_idxs]
        return data


@dataclass
class CosineGenerator(FunctionGenerator):
    frequencies: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def generate_data(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        variable_oscillated = 2 * np.pi * self.frequencies[:, :, None] * variable[None, None, :]
        cosine_funcs = np.cos(variable_oscillated)
        data = np.sum(self.coefficients[:, :, None] * cosine_funcs, axis=0).T
        return data


@dataclass
class GaussianGenerator(FunctionGenerator):
    means: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]
    stds: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    @property
    def fwhm(self):
        """Full width at half maximum"""
        return 2 * np.sqrt(2 * np.log(2)) * self.stds

    def rescale_parameters(
            self,
            ref_min,
            ref_max,
            new_min,
            new_max,
    ):
        means = new_min + (self.means - ref_min) / ref_max * (new_max - new_min)
        stds = self.stds / ref_max * (new_max - new_min)
        return replace(self, means=means, stds=stds)

    def generate_funcs(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ):
        variable_centered = (variable[None, None, :] - self.means[:, :, None]) / self.stds[:, :, None]
        gaussian_funcs = np.exp(- variable_centered ** 2)
        return gaussian_funcs

    def generate_data(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        gaussian_funcs = self.generate_funcs(variable=variable)
        data = np.sum(self.coefficients[:, :, None] * gaussian_funcs, axis=0).T
        return data


@dataclass
class LorentzianGenerator(FunctionGenerator):
    locations: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]  # location of the peak of the distribution
    scales: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]  # half-width at half-maximum (HWHM)

    @property
    def fwhm(self):
        """Full width at half maximum"""
        return 2 * self.scales

    def generate_funcs(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ):
        variable_centered = (variable[None, None, :] - self.locations[:, :, None]) / self.scales[:, :, None]
        lorentzian_funcs = 1 / (1 + variable_centered ** 2)
        return lorentzian_funcs

    def generate_data(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        lorentzian_funcs = self.generate_funcs(variable=variable)
        data = np.sum(self.coefficients[:, :, None] * lorentzian_funcs, axis=0).T
        return data


@dataclass
class VoigtGenerator(FunctionGenerator):
    centers: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]
    gauss_stds: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]
    lorentz_scales: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def generate_data(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        gauss_gen = GaussianGenerator(
            coefficients=np.array([[0.4]]),
            means=self.centers,
            stds=self.gauss_stds
        )
        gauss_funcs = gauss_gen.generate_funcs(variable=variable)

        lorentz_gen = LorentzianGenerator(
            coefficients=np.array([[0.31]]),
            locations=self.centers,
            scales=self.lorentz_scales
        )
        lorentz_funcs = lorentz_gen.generate_funcs(variable=variable)

        voigt_funcs = convolve(in1=gauss_funcs, in2=lorentz_funcs, mode="same") * np.mean(np.diff(variable))

        data = np.sum(self.coefficients[:, :, None] * voigt_funcs, axis=0).T
        return data
