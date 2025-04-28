from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from scipy import interpolate

from src.common_utils.custom_vars import Wvn, Acq
from src.common_utils.utils import convert_meter_units, standardize, min_max_normalize, rescale, add_noise, match_stats, \
    calculate_rmse, center
from src.outputs.serialize import numpy_save_list, numpy_load_list
from src.outputs.visualization import plot_custom


@dataclass(frozen=True)
class Spectrum:
    data: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]
    wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    wavenumbers_unit: str = r"um$^{-1}$"

    def visualize(
            self,
            axs,
            acq_ind: int,
            linestyle: str = "-",
            label: str = None,
            color: str = "C0",
            linewidth: float = 1.5,
            marker: str = "",
            markevery: int = 1,
            title: str = None,
            ylabel: str = None,
            ylim: list = None,
            x_ticklabel_format: bool = None,
    ):
        if title is None:
            title = "Spectral Radiance"
        if ylabel is None:
            ylabel = "Intensity"
        xlabel = rf"Wavenumbers $\sigma$ [{self.wavenumbers_unit}]"
        plot_custom(
            axs=axs,
            x_array=self.wavenumbers,
            array=self.data[:, acq_ind],
            linestyle=linestyle,
            label=label,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markevery=markevery,
            title=title,
            xlabel=xlabel,
            xlim=None,
            ylabel=ylabel,
            ylim=ylim,
            x_ticklabel_format=x_ticklabel_format,
        )

    def visualize_matrix(
            self,
            fig: Figure,
            axs,
            title: str = None,
            vmin: float = None,
            vmax: float = None
    ):
        pos = axs.imshow(
            self.data,
            # aspect='auto',
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(pos, ax=axs)

        wavenumber_ticks = np.linspace(start=0, stop=self.wavenumbers.size-1, num=6, dtype=int)
        wavenumber_labels = np.around(a=self.wavenumbers[wavenumber_ticks], decimals=2)
        axs.set_yticks(ticks=wavenumber_ticks, labels=wavenumber_labels)

        if title is None:
            title = "Spectrum Acquisitions"
        axs.set_title(title)
        axs.set_ylabel(rf"Wavenumbers $\sigma$ [{self.wavenumbers_unit}]")
        axs.set_xlabel(r"Acquisitions index $n \in \{1, \dots, N\}$")

    @classmethod
    def from_wavelength(
            cls,
            data: np.ndarray[tuple[int, Acq], np.dtype[np.float_]],
            wavelengths: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
            wavelengths_unit: str,
            is_interpolate: bool = False,
            interpolate_options: Optional[dict] = None,
            new_wavelength_unit: str = "cm",
    ) -> "Spectrum":
        unit_conversion_coefficient = convert_meter_units(values=1, from_=wavelengths_unit, to_=new_wavelength_unit)
        wavenumbers = 1 / wavelengths[::-1] / unit_conversion_coefficient
        spectrum = cls(
            data=data[::-1, :],
            wavenumbers=wavenumbers,
            wavenumbers_unit=f"1/{new_wavelength_unit}",
        )
        if is_interpolate:
            if interpolate_options is None:
                wavenumbers = np.linspace(start=wavenumbers[0], stop=wavenumbers[-1], num=wavenumbers.size)
            elif interpolate_options is not None and "wavenumbers" in interpolate_options:
                wavenumbers = interpolate_options["wavenumbers"] / unit_conversion_coefficient
            else:
                raise ValueError(f"The dictionary interpolate_options must either be {None} or contain the field 'wavenumbers'.")
            spectrum = spectrum.interpolate(wavenumbers=wavenumbers)
        return spectrum

    def interpolate(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
            kind: str = "cubic",
            fill_value: float | tuple | str = (0., 0.),
    ) -> "Spectrum":
        spectrum_out = interpolate.interp1d(
            x=self.wavenumbers,
            y=self.data,
            axis=-2,
            kind=kind,
            fill_value=fill_value,
            bounds_error=False,
        )(wavenumbers)
        return Spectrum(
            data=spectrum_out,
            wavenumbers=wavenumbers,
            wavenumbers_unit=self.wavenumbers_unit,
        )

    def add_noise(self, snr_db) -> Spectrum:
        noisy_data = add_noise(array=self.data, snr_db=snr_db)
        return replace(self, data=noisy_data)

    def center(self, new_mean: float = 0., axis: int = -2) -> Spectrum:
        centered_data = center(array=self.data, new_mean=new_mean, axis=axis)
        return replace(self, data=centered_data)

    def rescale(self, new_max: float = 1., axis: int = -2) -> Spectrum:
        rescaled_data = rescale(array=self.data, new_max=new_max, axis=axis)
        return replace(self, data=rescaled_data)

    def min_max_normalize(
            self,
            new_min: float = 0.,
            new_max: float = 1.,
            axis: int = -2
    ) -> Spectrum:
        normalized_data = min_max_normalize(array=self.data, new_min=new_min, new_max=new_max, axis=axis)
        return replace(self, data=normalized_data)

    def standardize(
            self,
            new_mean: float = 0.,
            new_std: float = 1.,
            axis: int = -2
    ) -> Spectrum:
        standardized_data = standardize(array=self.data, new_mean=new_mean, new_std=new_std, axis=axis)
        return replace(self, data=standardized_data)

    def match_stats(
            self,
            reference: Spectrum,
            axis: int = -2,
            is_rescale_reference: bool = False,
    ) -> tuple[Spectrum, Spectrum]:
        matched_data, scaled_reference = match_stats(
            array=self.data,
            reference=reference.data,
            axis=axis,
            is_rescale_reference=is_rescale_reference,
        )
        return replace(self, data=matched_data), replace(reference, data=scaled_reference)

    def calculate_rmse(
            self,
            reference: Spectrum,
            is_match_stats: bool = False,
            is_rescale_reference: bool = False,
            is_match_axis: int = -2,
    ) -> np.ndarray[..., np.dtype[np.float_]]:
        rmse = calculate_rmse(
            array=self.data,
            reference=reference.data,
            is_match_stats=is_match_stats,
            is_rescale_reference=is_rescale_reference,
            is_match_axis=is_match_axis,
        )
        return rmse

    def crop_wavenumbers(
            self,
            new_range: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> Spectrum:
        mask = (new_range[0] <= self.wavenumbers) & (self.wavenumbers <= new_range[-1])
        new_wavenumbers = self.wavenumbers[mask]
        new_data = self.data[mask]
        return replace(self, data=new_data, wavenumbers=new_wavenumbers)

    @classmethod
    def from_oscillations(
            cls,
            amplitudes: np.ndarray,
            opds: np.ndarray,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> Spectrum:
        radiance = generate_radiance_from_oscillations(
            amplitudes=amplitudes,
            opds=opds,
            wavenumbers=wavenumbers,
        ).reshape((-1, 1))
        return Spectrum(data=radiance, wavenumbers=wavenumbers)

    def save_numpy(
            self,
            directories: list[Path],
            subdirectory: str = "",
    ):
        numpy_save_list(
            filenames=["data.npy", "wavenumbers.npy"],
            arrays=[self.data, self.wavenumbers],
            directories=directories,
            subdirectory=subdirectory,
        )

    @classmethod
    def load_numpy(
            cls,
            directory: Path,
            subdirectory: str = "",
    ) -> Spectrum:
        data, wavenumbers = numpy_load_list(
            filenames=["data.npy", "wavenumbers.npy"],
            directory=directory,
            subdirectory=subdirectory,
        )
        return Spectrum(data=data, wavenumbers=wavenumbers)


def generate_radiance_from_oscillations(
        amplitudes: np.ndarray,
        opds: np.ndarray,
        wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> np.ndarray[tuple[Wvn], np.dtype[np.float_]]:
    if amplitudes.size != opds.size:
        raise ValueError("Amplitudes and OPDs must have the same size.")
    list_of_cosines = amplitudes[:, None] * np.cos(2 * np.pi * opds[:, None] * wavenumbers[None, :])
    radiance = np.sum(list_of_cosines, axis=0)
    return radiance


def convert_to_wavenumbers(
    spectrum_in_wavelengths: np.ndarray[tuple[int, Acq], np.dtype[np.float_]],
    wavelengths: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    wavelengths_unit: str = "nm",
    wavenumbers_limits: tuple[float, float] = None,
    wavenumbers_samples: int = None,
) -> Spectrum:
    spectrum_in = Spectrum.from_wavelength(
        data=spectrum_in_wavelengths,
        wavelengths=wavelengths,
        wavelengths_unit=wavelengths_unit,
    )

    if wavenumbers_limits is None:
        wavenumbers_limits = (1/wavelengths[-1], 1/wavelengths[0])
    if wavenumbers_samples is None:
        wavenumbers_samples = wavelengths.size
    wavenumbers = np.linspace(
        start=wavenumbers_limits[0],
        stop=wavenumbers_limits[1],
        num=wavenumbers_samples,
        endpoint=False,
    )
    spectrum_out = spectrum_in.interpolate(wavenumbers=wavenumbers)

    return spectrum_out
