from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.custom_vars import Wvn, Opd, InterferometerType, Deg
from src.common_utils.utils import polyval_rows


def interferometer_factory(
        option: InterferometerType,
        transmittance_coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]],
        opds: np.ndarray[tuple[Opd], np.dtype[np.float_]],
        phase_shift: np.ndarray[tuple[Opd], np.dtype[np.float_]],
        reflectance_coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]] = None,
        order: int = 0,
) -> Interferometer:
    if option == InterferometerType.MICHELSON:
        interferometer = MichelsonInterferometer(
            transmittance_coefficients=transmittance_coefficients,
            opds=opds,
            phase_shift=phase_shift,
        )
    elif option == InterferometerType.FABRY_PEROT:
        interferometer = FabryPerotInterferometer(
            transmittance_coefficients=transmittance_coefficients,
            opds=opds,
            phase_shift=phase_shift,
            reflectance_coefficients=reflectance_coefficients,
            order=order,
        )
    else:
        raise ValueError(f"Option '{option.value}' is not supported")
    return interferometer


@dataclass(frozen=True)
class Interferometer(ABC):
    transmittance_coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    phase_shift: np.ndarray[tuple[Opd], np.dtype[np.float_]]

    @abstractmethod
    def transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
            is_correct_transmittance: bool = False,
    ) -> TransmittanceResponse:
        pass

    def acquire_interferogram(
            self,
            spectrum: Spectrum,
    ) -> Interferogram:
        transmittance_response = self.transmittance_response(wavenumbers=spectrum.wavenumbers)
        return simulate_interferogram(transmittance_response=transmittance_response, spectrum=spectrum)

    def phase_difference(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        return calculate_phase_difference(
            opds=self.opds,
            wavenumbers=wavenumbers,
            phase_shift=self.phase_shift,
        )

    def coeffs_to_polynomials(
            self,
            coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]],
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        if coefficients.shape[0] == 1:
            coefficients = np.tile(coefficients.reshape(1, -1), (self.opds.size, 1))
            assert coefficients.ndim == 2
        return polyval_rows(coefficients=coefficients, interval=wavenumbers)

    def transmittance(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        return self.coeffs_to_polynomials(coefficients=self.transmittance_coefficients, wavenumbers=wavenumbers)

    @abstractmethod
    def reflectance(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        pass

    @abstractmethod
    def harmonic_order(self, threshold: float = 0.001) -> int:
        pass

    @property
    def average_opd_step(self) -> float:
        return np.mean(np.diff(self.opds))


@dataclass(frozen=True)
class MichelsonInterferometer(Interferometer):

    def transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
            is_correct_transmittance: bool = False,
    ) -> TransmittanceResponse:
        transmittance = self.transmittance(wavenumbers=wavenumbers)
        phase_difference = self.phase_difference(wavenumbers=wavenumbers)
        # This is equivalent to: y = (1/2 * y[0]) + 1/2 * scipy.fft.dct(2*T*x, type=2, norm=None)
        transmittance_response = 2 * transmittance * (1 + np.cos(phase_difference))
        return TransmittanceResponse(
            data=transmittance_response,
            wavenumbers=wavenumbers,
            opds=self.opds,
        )

    def reflectance(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        return 1 - self.transmittance(wavenumbers=wavenumbers)

    def harmonic_order(self, threshold: float = 0.001) -> int:
        return 2


@dataclass(frozen=True)
class FabryPerotInterferometer(Interferometer):
    reflectance_coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]]
    order: int = 0

    def transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
            is_correct_transmittance: bool = False,
    ) -> TransmittanceResponse:
        reflectance = self.reflectance(wavenumbers=wavenumbers)
        transmittance = self.transmittance(wavenumbers=wavenumbers)
        if is_correct_transmittance:
            transmittance = transmittance * (1 - reflectance) * (1 + reflectance)
        else:
            transmittance = transmittance ** 2
        phase_difference = self.phase_difference(wavenumbers=wavenumbers)

        if self.order == 0:
            # Using the infinity-wave model and geometric series formula
            denominator = 1 + reflectance ** 2 - 2 * reflectance * np.cos(phase_difference)
            transmittance_response = transmittance / denominator

        else:
            # Using the N-wave model approximation and the Poisson kernel formula
            quotient = 1 / (1 - reflectance ** 2)
            n_values = np.arange(1, self.order)
            reflectance_factors = reflectance[None, :] ** n_values[:, None, None]
            cosine_factors = np.cos(n_values[:, None, None] * phase_difference[None, :])
            series_sum = 1 + 2 * np.sum(reflectance_factors * cosine_factors, axis=0)
            transmittance_response = transmittance * quotient * series_sum

        return TransmittanceResponse(
            data=transmittance_response,
            wavenumbers=wavenumbers,
            opds=self.opds,
        )

    def reflectance(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        return self.coeffs_to_polynomials(coefficients=self.reflectance_coefficients, wavenumbers=wavenumbers)

    def harmonic_order(self, threshold: float = 0.001) -> int:
        """
        Number of coefficients until we have C_{N} / C_1 < 0.001
          (We choose 0.001 because it looks like the closest threshold to 2-wave)
          In other words, where:
            R^{N-1} < 0.001
            (N-1) * log(R) < log(0.001)
            N < log(0.001) / log(R) + 1
            N = ceil(N)
        """
        reflectance = self.reflectance(np.array([1])).mean()
        order_float = np.log(threshold) / np.log(reflectance) + 1
        order = np.ceil(order_float)
        return order


def simulate_interferogram(
        transmittance_response: TransmittanceResponse,
        spectrum: Spectrum,
) -> Interferogram:
    interferogram_data = transmittance_response.data @ spectrum.data
    return Interferogram(data=interferogram_data, opds=transmittance_response.opds)


def calculate_phase_difference(
        opds: np.ndarray[tuple[Opd], np.dtype[np.float_]],
        wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
        phase_shift: np.ndarray[tuple[Opd], np.dtype[np.float_]],
) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
    return 2 * np.pi * opds[:, None] * wavenumbers[None, :] - phase_shift[:, None]
