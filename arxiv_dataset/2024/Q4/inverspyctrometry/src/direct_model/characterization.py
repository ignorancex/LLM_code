from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import interpolate

from src.common_utils.custom_vars import Opd, Deg, InterferometerType, Wvn
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.utils import polyval_rows
from src.direct_model.interferometer import Interferometer, interferometer_factory


@dataclass(frozen=True)
class Characterization:
    interferometer_type: InterferometerType
    transmittance_coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    phase_shift: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    reflectance_coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]]
    order: int

    def sort_opds(self):
        new_indices = np.argsort(self.opds)
        char_sorted = replace(
            self,
            transmittance_coefficients=self.transmittance_coefficients[new_indices],
            opds=self.opds[new_indices],
            phase_shift=self.phase_shift[new_indices],
            reflectance_coefficients=self.reflectance_coefficients[new_indices],
        )
        return char_sorted

    def interferometer(self) -> Interferometer:
        interferometer = interferometer_factory(
            option=self.interferometer_type,
            transmittance_coefficients=self.transmittance_coefficients,
            opds=self.opds,
            phase_shift=self.phase_shift,
            reflectance_coefficients=self.reflectance_coefficients,
            order=self.order,
        )
        return interferometer

    def transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> TransmittanceResponse:
        interferometer = self.interferometer()
        return interferometer.transmittance_response(wavenumbers=wavenumbers, is_correct_transmittance=True)

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

    def reflectance(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        return self.coeffs_to_polynomials(coefficients=self.reflectance_coefficients, wavenumbers=wavenumbers)

    def extrapolate_opds(
            self,
            support_resampler: str,
    ) -> Characterization:
        opd_mean_step = np.mean(np.diff(self.opds))
        lowest_missing_opds = np.arange(start=0., stop=self.opds.min(), step=opd_mean_step)
        
        if support_resampler == "resample_all":
            # opds = np.arange(start=0., stop=self.opds.max() + opd_mean_step, step=opd_mean_step)
            raise ValueError(f"Support resampling option '{support_resampler}' is not yet supported.")
        elif support_resampler == "concatenate_missing":
            opds = np.concatenate((lowest_missing_opds, self.opds))
        else:
            raise ValueError(f"Support resampling option '{support_resampler}' is not supported.")
        
        def extrapolate_coeffs(coefficients: np.ndarray) -> np.ndarray:
            """
            I'm taking the mean here because theoretically interferometers sharing the similar materials should have
              the same the transmittance and reflectance (independent of the OPD as a physical parameter), so then
              the "missing interferometers" will have that.
            """
            coeffs_mean = np.mean(coefficients, axis=-2, keepdims=True)
            coeffs_missing = np.tile(coeffs_mean, reps=(lowest_missing_opds.size, 1))
            concatenated_coeffs = np.concatenate((coeffs_missing, coefficients), axis=-2)
            return concatenated_coeffs
        
        transmittance_coeffs = extrapolate_coeffs(self.transmittance_coefficients)
        reflectance_coeffs = extrapolate_coeffs(self.reflectance_coefficients)
        phase_shift = interpolate.interp1d(
            x=self.opds,
            y=self.phase_shift,
            kind="zero",
            fill_value=0.,
            bounds_error=False,
        )(opds)

        return replace(
            self,
            transmittance_coefficients=transmittance_coeffs,
            opds=opds,
            phase_shift=phase_shift,
            reflectance_coefficients=reflectance_coeffs,
        )
