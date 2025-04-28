from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from src.common_utils.custom_vars import Wvn
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import generate_wavenumbers_from_opds, polyval_rows
from src.outputs.visualization import imshow_custom


# TODO: Supports only constant transmittance and reflectance
#  (otherwise, we would need information on the coefficients and the wavenumbers)


@dataclass(frozen=True)
class AnalyticalInverter(ABC):
    transmittance: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]]

    @abstractmethod
    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
    ) -> Spectrum:
        pass


class MichelsonInverter(AnalyticalInverter):

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
    ) -> Spectrum:
        interferogram_compensated = interferogram.data - 1 / 2 * interferogram.data[0, :]
        # This is equivalent to: x = 2 * scipy.fft.idct(y - 1/2 * y[0], type=2. norm=None) / (2*T)
        spectrum = fft.idct(interferogram_compensated, axis=-2) / self.transmittance[:, None]
        wavenumbers = generate_wavenumbers_from_opds(
            wavenumbers_num=interferogram.opds.size,
            del_opd=np.mean(np.diff(interferogram.opds)),
        )
        return Spectrum(data=spectrum, wavenumbers=wavenumbers)


@dataclass(frozen=True)
class HaarInverter(AnalyticalInverter):
    """
    This method is based on expanding the DFT of the interferogram and the spectrum by a Haar or box function.
    """
    reflectance: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    order: int
    is_mean_center: bool = True

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
    ) -> Spectrum:
        if self.is_mean_center:
            interferogram = interferogram.center(new_mean=0., axis=-2)

        interferogram_dft = compute_interferogram_dft(interferogram, norm="ortho")

        fp_obj = SimpleNamespace(
            transmittance_coeffs=self.transmittance,
            reflectance_coeffs=self.reflectance,
            order=0,
        )
        b_matrix = assert_haar_check(fp_obj, interferogram_dft, self.wavenumbers, self.order)
        spectrum_rec = replace(interferogram_dft, data=np.real(np.linalg.pinv(b_matrix) @ interferogram_dft.data))

        spectrum_rec = spectrum_rec.interpolate(
            wavenumbers=self.wavenumbers,
            kind="linear",
            fill_value="extrapolate",
        )

        return spectrum_rec

    def plot_transfer_matrix(
            self,
            transfer_matrix,
            wavenumbers_dct,
    ):
        fig, axs = plt.subplots(1, 1, squeeze=False)
        imshow_custom(
            fig=fig,
            axs=axs[0, 0],
            image=transfer_matrix,
            title=f"Harmonics Order M = {self.order}",
            x_variable=wavenumbers_dct,
            y_variable=wavenumbers_dct,
            x_label="Wavenumbers [cm^-1]",
            y_label="Wavenumbers [cm^-1]",
            vmin=0,
            vmax=0.01,
            interpolation="nearest",
        )


def compute_interferogram_dft(interferogram, norm):
    dft_data = fft.fft(interferogram.data, axis=-2, norm=norm)
    dft_data = dft_data[:interferogram.opds.size // 2]
    dft_support = fft.fftfreq(n=interferogram.opds.size, d=np.mean(np.diff(interferogram.opds)))
    dft_support = dft_support[:interferogram.opds.size // 2]
    dft = Spectrum(data=dft_data, wavenumbers=dft_support, wavenumbers_unit=f"1/{interferogram.opds_unit}")
    return dft


def calculate_airy_fourier_coeffs(fp, haar_order):
    variable = np.linspace(start=0, stop=1, num=int(1e4), endpoint=False)

    numerator = fp.transmittance_coeffs ** 2
    phase_difference = 2 * np.pi * variable
    denominator = 1 + fp.reflectance_coeffs ** 2 - 2 * fp.reflectance_coeffs * np.cos(phase_difference)
    kernel = numerator / denominator

    n_values = np.arange(haar_order + 1)
    cosines = np.cos(2 * np.pi * n_values[:, None] * variable[None, :])
    integrands = kernel[None, :] * cosines
    variable_differential = np.mean(np.diff(variable))
    coefficients = 2 * np.sum(integrands * variable_differential, axis=-1)
    coefficients[0] /= 2
    # coefficients = np.sum(integrands * variable_differential, axis=-1)

    # print(coefficients)
    return coefficients


def assert_haar_check(fp, interferogram_dft, wavenumbers, haar_order):
    k_vals = np.where(
        np.logical_and(
            wavenumbers[0] <= interferogram_dft.wavenumbers,
            interferogram_dft.wavenumbers <= wavenumbers[-1],
        )
    )

    a_cap = calculate_airy_fourier_coeffs(fp, haar_order)

    k_cap = interferogram_dft.wavenumbers.size
    b_matrix = np.zeros(shape=(k_cap, k_cap))
    for mu in k_vals[0]:
        for n in range(1, haar_order + 1):
            for k in k_vals[0]:
                for i in range(n):
                    if n * k + i == mu:
                        b_matrix[mu, k] += a_cap[n] / n

    return b_matrix
