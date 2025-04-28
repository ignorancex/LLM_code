from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from abc import ABC, abstractmethod
from typing import Callable, Literal
import functools

import numpy as np
from scipy.optimize import least_squares

from src.characterization.interface import (
    CharacterizeOptionsSchema,
    CharacterizeOptionsListSchema,
)
from src.characterization.interferometry import (
    nyquist_cutoff_opd,
    interferometry_transfer,
    interferometry_transfer_scaled,
    poly_eval,
    poly_eval_2d,
)
from src.lib.custom_vars import (
    Ifm,
    Chn,
    Smp,
    Deg,
    DgO,
    Fig,
    Axs,
    CharacterizeMethodEnum,
    NumpyPolynomial,
)
from src.characterization.preprocessing import (
    Parameters,
    CharacterizationPixel,
    Characterization,
)
from src.lib.visualize import plot_1d


@dataclass(frozen=True)
class CharacterizationProtocol:
    calibration: CharacterizationPixel
    options: CharacterizeOptionsListSchema | CharacterizeOptionsSchema
    init: Parameters = None
    _estimation: Parameters = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.options, CharacterizeOptionsListSchema):
            object.__setattr__(self, "options", [self.options])

    def estimate(self) -> Parameters:
        """Estimates the stack of characterization protocols"""
        if self._estimation is None:
            init = self.init
            for options in self.options:
                protocol = CharacterizationFactory(
                    calibration=self.calibration,
                    init=init,
                    options=options,
                )
                init = protocol.estimate()
            object.__setattr__(self, "_estimation", init)
        return self._estimation

    def characterize(self, model: int = None) -> Characterization:
        """Returns the characterization of the device"""
        model = self.options[-1].model if model is None else model
        return Characterization(
            parameters=self.estimate(),
            model=model,
            polynomial=self.options[-1].callable,
        )

    def transfer_estimated(self) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
        wavenumber = self.calibration.central_wavenumbers
        return self.characterize().transfer_function(wavenumber=wavenumber)

    def visualize(self, interferometer: int) -> tuple[Fig, Axs]:
        """
        Plots a comparison between the device calibration and the estimated
        characterization
        """
        transfer_estimated = self.transfer_estimated()[interferometer, :]
        transfer_calibration = self.calibration.data[interferometer, :]
        scale_factor = np.mean(transfer_calibration)
        calibration_scaled = transfer_calibration / scale_factor
        estimation_scaled = transfer_estimated / scale_factor
        return plot_1d(
            x=self.calibration.central_wavenumbers*10000,
            y=np.vstack(tup=(calibration_scaled, estimation_scaled)),
            x_label=r"Central wavenumbers [$\r{mathrm{cm}^{-1}}$]",
            y_label="Mean scaled intensity",
            title="Transfer function curve fitting",
            legend=["Raw acquisition", "Fitted transfer function"],
            scientific=True,
            reciprocal_axis=True,
            reciprocal_label=r"Central wavelengths [$\r{mathrm{nm}$]",
        )


@dataclass(frozen=True)
class CharacterizationFactory:
    calibration: CharacterizationPixel
    options: CharacterizeOptionsSchema = CharacterizeOptionsSchema()
    init: Parameters = None
    _estimation: Parameters = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        ifm = self.calibration.data.shape[0]
        init = Parameters.init_zeros(ifm) if self.init is None else self.init
        init = init.standardize(options=self.options.parameters)
        object.__setattr__(self, "init", init)

    def update(
        self, options: CharacterizeOptionsSchema
    ) -> "CharacterizationFactory":
        protocol = replace(self, options=options)
        return replace(protocol, init=protocol.estimate())

    def estimate(self) -> Parameters:
        if self._estimation is None:
            object.__setattr__(self, "_estimation", self.create().estimate())
        return self._estimation

    def characterize(self) -> Characterization:
        return Characterization(
            parameters=self.estimate(),
            model=self.options.model,
            polynomial=self.options.polynomial,
        )

    def _load_data(self) -> np.ndarray[tuple[Ifm, Chn], np.float64]:
        if self.options.average:
            return self.calibration.mean
        return self.calibration.data

    def create(self) -> CharacterizationMethod:
        enum = CharacterizeMethodEnum
        dictionary = {
            (enum.GAUSS_NEWTON, GaussNewtonCharacterization),
            (enum.MAXIMUM_LIKELIHOOD, MaximumLikelihoodCharacterization),
            (enum.EXHAUSTIVE_SEARCH, ExhaustiveSearchCharacterization),
        }
        mth = next(dic[1] for dic in dictionary if dic[0] == self.options.id)
        return mth.create(
            calibration=self.calibration,
            options=self.options,
            init=self.init,
        )


class CharacterizationMethod(ABC):
    """Interface for the characterization protocols"""

    @classmethod
    @abstractmethod
    def create(
        cls,
        calibration: CharacterizationPixel,
        options: CharacterizeOptionsSchema,
        init: Parameters,
    ) -> "CharacterizationMethod":
        """Constructor for the characterization method."""

    @abstractmethod
    def estimate(self) -> Parameters:
        """
        Estimates the parameters of a transfer function with the maximum
        likelihood method acquisition is a 2D numpy array, where each column
        defines the acquisition for a given central wavenumber samples defines
        the amount of samples explored in the OPD range
        """


@dataclass(frozen=True)
class MaximumLikelihoodCharacterization(CharacterizationMethod):
    acquisition: np.ndarray[tuple[Ifm, Chn], np.float64]
    wavenumber: np.ndarray[tuple[Chn], np.float64]  # in 1 / micrometer
    gain_calibration: np.ndarray[tuple[Ifm, Chn], np.float64]

    opd_samples: np.ndarray[tuple[Smp], np.float64]  # in micrometer
    opd_mask: np.ndarray[tuple[Ifm, Smp], np.float64]
    gain_degree: int  # polynomial degree for the gain
    gain_fit: Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]]

    @classmethod
    def create(
        cls,
        calibration: CharacterizationPixel,
        options: CharacterizeOptionsSchema,
        init: Parameters,
    ) -> "MaximumLikelihoodCharacterization":
        wavenumbers = calibration.central_wavenumbers
        init_opd = poly_eval_2d(
            coef=init.opd, func=options.callable, x=wavenumbers
        )
        init_opd = np.mean(init_opd, axis=1)
        opd_max = nyquist_cutoff_opd(wavenumber=wavenumbers)
        mask = options.parameters.opd.mask(init=init_opd / opd_max)
        opd_samples = np.linspace(
            0, 1, num=options.parameters.opd.samples, endpoint=False
        ) * opd_max
        return MaximumLikelihoodCharacterization(
            acquisition=calibration.select_data(flag_mean=options.average),
            wavenumber=wavenumbers,
            opd_samples=opd_samples,
            gain_degree=options.parameters.gain.degree,
            gain_calibration=calibration.norm,
            gain_fit=options.callable,
            opd_mask=mask,
        )

    def estimate(self) -> Parameters:
        gain = estimate_gain(
            acquisition=self.acquisition,
            calibration=self.gain_calibration,
            wavenumber=self.wavenumber,
            fit_function=self.gain_fit,
            degree=self.gain_degree,
        )
        gain_full = poly_eval_2d(
            coef=gain, func=self.gain_fit, x=self.wavenumber
        )
        acq = self.acquisition / gain_full - 1
        opd = self.opd(acquisition=acq)
        phase_diff = 2 * np.pi * opd * self.wavenumber[np.newaxis, :]
        shift = self.phase_shift(acquisition=acq, phase_diff=phase_diff)
        reflect = self.reflectivity(acquisition=acq, phase_diff=phase_diff)
        return Parameters(
            opd=opd, reflectivity=reflect, phase_shift=shift, gain=gain
        )

    def opd(
        self, acquisition: np.ndarray[tuple[Ifm, Chn], np.float64]
    ) -> np.ndarray[tuple[Ifm, Literal[1]], np.float64]:
        """Optical path difference estimation by periodogram maximization."""
        phase_term = np.outer(-2j * np.pi * self.wavenumber, self.opd_samples)
        acq_fft = np.abs(acquisition @ np.exp(phase_term))
        np.place(arr=acq_fft, mask=np.logical_not(self.opd_mask), vals=0)
        index_max = np.argmax(acq_fft, axis=1)
        return np.take(self.opd_samples, index_max)[:, np.newaxis]

    @staticmethod
    def phase_shift(
        acquisition: np.ndarray[tuple[Ifm, Chn], np.float64],
        phase_diff: np.ndarray[tuple[Ifm, Chn], np.float64],
    ) -> np.ndarray[tuple[Ifm, Literal[1]], np.float64]:
        """Phase shift estimator"""
        numerator = np.sum(acquisition * np.sin(phase_diff), axis=1)
        denominator = np.sum(acquisition * np.cos(phase_diff), axis=1)
        phase_shift = np.arctan2(numerator, denominator) % (2 * np.pi)
        return phase_shift[:, np.newaxis]

    @staticmethod
    def reflectivity(
        acquisition: np.ndarray[tuple[Ifm, Chn], np.float64],
        phase_diff: np.ndarray[tuple[Ifm, Chn], np.float64],
    ) -> np.ndarray[tuple[Ifm, Literal[1]], np.float64]:
        """Reflectivity estimation from the amplitude of a sinusoid"""
        statistic = np.sum(acquisition * np.exp(-1j * phase_diff), axis=1)
        amplitude = 2 / acquisition.shape[1] * np.abs(statistic)
        reflectivity = (1 - np.sqrt(1 - amplitude**2)) / amplitude
        return reflectivity[:, np.newaxis]


@dataclass(frozen=True)
class GaussNewtonCharacterization(CharacterizationMethod):
    acquisition: np.ndarray[tuple[Ifm, Chn], np.float64]
    wavenumber: np.ndarray[tuple[Chn], np.float64]
    init: Parameters
    choice: list[str]
    fit_func: NumpyPolynomial
    model: int = 0

    _len: list[int] = field(init=False, repr=False)
    _slice: list[slice] = field(init=False, repr=False)
    _excluded: list[str] = field(init=False, repr=False)
    _len_exc: list[int] = field(init=False, repr=False)
    _slice_exc: list[slice] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_len", self._length_slice(self.choice))
        object.__setattr__(self, "_slice", self._get_slice(self._len))
        exclude = [key for key in asdict(self.init) if key not in self.choice]
        object.__setattr__(self, "_excluded", exclude)
        object.__setattr__(self, "_len_exc", self._length_slice(exclude))
        object.__setattr__(self, "_slice_exc", self._get_slice(self._len_exc))

    @classmethod
    def create(
        cls,
        calibration: CharacterizationPixel,
        options: CharacterizeOptionsSchema,
        init: Parameters,
    ) -> "GaussNewtonCharacterization":
        return cls(
            acquisition=calibration.select_data(flag_mean=options.average),
            wavenumber=calibration.central_wavenumbers,
            init=init,
            choice=options.choice,
            model=options.model,
            fit_func=options.callable,
        )

    def estimate(self) -> Parameters:
        acq_mean = np.mean(self.acquisition, axis=1, keepdims=True)
        acq_no_mean = self.acquisition / acq_mean
        init = replace(self.init, gain=self.init.gain / acq_mean)
        new = replace(self, init=init, acquisition=acq_no_mean)
        out = np.empty((new.acquisition.shape[0], new._slice[-1].stop))
        x0 = np.empty((new.acquisition.shape[0], new._len[-1]))
        x1 = np.empty((new.acquisition.shape[0], new._len_exc[-1]))
        for key, sli in zip(new.choice, new._slice):
            x0[:, sli] = getattr(new.init, key)
        for key, sli in zip(new._excluded, new._slice_exc):
            x1[:, sli] = getattr(new.init, key)
        for ii in np.arange(new.acquisition.shape[0]):
            partial = functools.partial(
                new.cost_function,
                init=x1[ii, :],
                reference=new.acquisition[ii, :],
            )
            out[ii, :] = least_squares(fun=partial, x0=x0[ii, :]).x
        out_dict = {}
        for key, sli in zip(new.choice, new._slice):
            out_dict.update({key: out[:, sli]})
        out = replace(init, **out_dict)
        out = replace(out, gain=out.gain * acq_mean)
        return out

    def cost_function(
        self,
        x: np.ndarray[tuple[Deg], np.float64],
        init: np.ndarray[tuple[DgO], np.float64],
        reference: np.ndarray[tuple[Chn], np.float64],
    ) -> np.ndarray:
        dict_eval = {}
        for key, sli in zip(self.choice, self._slice):
            upd = poly_eval(coef=x[sli], func=self.fit_func, x=self.wavenumber)
            dict_eval.update({key: upd})
        estimation = self._interferometer_partial(x=init)(**dict_eval)[0, :]
        return estimation - reference

    def _length_slice(self, choice: list[str]) -> list[int]:
        length = [0] + [getattr(self.init, key).shape[1] for key in choice]
        return np.cumsum(length).tolist()

    @staticmethod
    def _get_slice(lst: list[int]) -> list[slice]:
        return [slice(lst[ii], lst[ii + 1]) for ii in range(len(lst) - 1)]

    def _interferometer_partial(
        self, x: np.ndarray[tuple[DgO], np.float64]
    ) -> Callable:
        exclude = {}
        for key, sli in zip(self._excluded, self._slice_exc):
            upd = poly_eval(coef=x[sli], func=self.fit_func, x=self.wavenumber)
            exclude.update({key: upd})
        return functools.partial(
            interferometry_transfer_scaled,
            model=self.model,
            wavenumber=self.wavenumber,
            **exclude,
        )


@dataclass(frozen=True)
class ExhaustiveSearchCharacterization(CharacterizationMethod):
    acquisition: np.ndarray[tuple[Ifm, Chn], np.float64]
    wavenumber: np.ndarray[tuple[Chn], np.float64]  # in 1 / micrometer
    gain_calibration: np.ndarray[tuple[Ifm, Chn], np.float64]
    opd_samples: np.ndarray[tuple[int], np.float_]
    reflectivity_samples: np.ndarray[tuple[int], np.float_]
    phase_shift_samples: np.ndarray[tuple[int], np.float_]
    gain_degree: int  # polynomial degree for the gain
    gain_fit: Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]]
    opd_mask: np.ndarray[tuple[Ifm, int], np.float_]
    reflectivity_mask: np.ndarray[tuple[Ifm, int], np.float_]
    model: int = 0

    @classmethod
    def create(
        cls,
        calibration: CharacterizationPixel,
        options: CharacterizeOptionsSchema,
        init: Parameters,
    ) -> "ExhaustiveSearchCharacterization":
        wavenumbers = calibration.central_wavenumbers
        parameters = ["opd", "reflectivity", "phase_shift"]
        maxima = (nyquist_cutoff_opd(wavenumber=wavenumbers), 1, 2 * np.pi)
        out = {}
        for param, maximum in zip(parameters, maxima):
            num = getattr(options.parameters, param).samples
            out.update(
                {param: np.linspace(0, 1, num=num, endpoint=False) * maximum}
            )
        acquisition = calibration.select_data(flag_mean=options.average)

        parameters_mask = ["opd", "reflectivity"]
        maximum_mask = [maxima[0], maxima[1]]
        out_mask = {}
        for param, maximum in zip(parameters_mask, maximum_mask):
            init_matrix = poly_eval_2d(
                coef=getattr(init, param), func=options.callable, x=wavenumbers
            )
            init_v = np.mean(init_matrix, axis=1)
            cur = getattr(options.parameters, param).mask(init=init_v/maximum)
            index_ones = ~np.all(~cur, axis=0)
            cur = cur[:, index_ones]
            out[param] = out[param][index_ones]
            out_mask.update({param: cur})

        return ExhaustiveSearchCharacterization(
            acquisition=acquisition,
            wavenumber=wavenumbers,
            opd_samples=out["opd"],
            reflectivity_samples=out["reflectivity"],
            phase_shift_samples=out["phase_shift"],
            gain_degree=options.parameters.gain.degree,
            gain_calibration=calibration.norm,
            gain_fit=options.callable,
            opd_mask=out_mask["opd"],
            reflectivity_mask=out_mask["reflectivity"],
        )

    def estimate(self) -> Parameters:
        gain = estimate_gain(
            acquisition=self.acquisition,
            calibration=self.gain_calibration,
            wavenumber=self.wavenumber,
            fit_function=self.gain_fit,
            degree=self.gain_degree,
        )
        gain_full = poly_eval_2d(
            coef=gain, func=self.gain_fit, x=self.wavenumber
        )
        acq = self.acquisition / gain_full
        acq_fft = np.abs(np.fft.fft(acq, axis=1))

        ifm = self.acquisition.shape[0]
        opd_o, refl_o = (np.zeros((ifm,)), np.zeros((ifm,)))
        shape_fft = (len(self.opd_samples), len(self.reflectivity_samples))
        fft_function = np.empty(shape_fft + self.wavenumber.shape)
        for ii, opd in enumerate(self.opd_samples):
            for jj, refl in enumerate(self.reflectivity_samples):
                current = interferometry_transfer(
                    opd=np.full(shape=(1, 1), fill_value=opd),
                    reflectivity=np.full(shape=(1, 1), fill_value=refl),
                    gain=np.ones((1, 1)),
                    phase_shift=np.zeros((1, 1)),
                    model=self.model,
                    wavenumber=self.wavenumber,
                )[0, :]
                fft_function[ii, jj, :] = np.abs(np.fft.fft(current))
        for kk in np.arange(ifm):
            opd_index = self.opd_mask[kk, :]
            refl_index = self.reflectivity_mask[kk, :]
            fft_subset = fft_function[opd_index, :, :][:, refl_index, :]
            acq_fft_select = acq_fft[kk, :][np.newaxis, np.newaxis, :]
            cost = np.sum((fft_subset - acq_fft_select) ** 2, axis=2)
            min_idx = np.unravel_index(np.argmin(cost), cost.shape)
            opd_o[kk] = self.opd_samples[opd_index][min_idx[0]]
            refl_o[kk] = self.reflectivity_samples[refl_index][min_idx[1]]

        shape = (len(self.phase_shift_samples), self.wavenumber.shape[0])
        interferogram = np.empty(shape)
        phase_shift_o = np.empty((ifm,))
        for kk in np.arange(ifm):
            for ii, phase_shift in enumerate(self.phase_shift_samples):
                interferogram[ii, :] = interferometry_transfer_scaled(
                    opd=opd_o[kk, np.newaxis],
                    reflectivity=refl_o[kk, np.newaxis],
                    gain=np.ones((1, 1)),
                    phase_shift=np.full(shape=(1, 1), fill_value=phase_shift),
                    model=self.model,
                    wavenumber=self.wavenumber,
                )
            cst = np.sum((interferogram - acq[np.newaxis, kk, :]) ** 2, axis=1)
            idx_min = np.argmin(cst, axis=0)
            phase_shift_o[kk] = self.phase_shift_samples[idx_min]

        return Parameters(
            opd=opd_o[:, np.newaxis],
            gain=gain,
            reflectivity=refl_o[:, np.newaxis],
            phase_shift=phase_shift_o[:, np.newaxis],
        )


def estimate_gain(
    acquisition: np.ndarray[tuple[Ifm, Chn], np.float64],
    calibration: np.ndarray[tuple[Ifm, Chn], np.float64],
    wavenumber: np.ndarray[tuple[Chn], np.float64],
    fit_function: Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]],
    degree: int = 0,
) -> np.ndarray[tuple[Ifm, Chn], np.float64]:

    norm_mean = np.mean(calibration, axis=1, keepdims=True)
    norm = calibration / norm_mean
    acquisition_mean = np.mean(acquisition, axis=1, keepdims=True)

    gain = np.empty((acquisition.shape[0], degree+1))
    cond = np.sum((calibration - calibration[0, :][np.newaxis, :]) ** 2) < 1e-9
    max_loop = [0] if cond else np.arange(acquisition.shape[0])
    for ii in max_loop:
        gain[ii, :] = least_squares(
            fun=cost_function,
            x0=np.zeros((degree + 1,)),
            kwargs={
                "wavenumber": wavenumber,
                "reference": norm[ii, :],
                "func": fit_function,
            },
        ).x
    if cond:  # Avoids calculation for all interferometers if all are equal
        gain = np.broadcast_to(gain[0, :][np.newaxis, :], gain.shape)
    return gain * acquisition_mean


def cost_function(
    estimation: np.ndarray[tuple[Deg], np.float_],
    func: Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]],
    wavenumber: np.ndarray[tuple[Chn], np.float_],
    reference: np.ndarray[tuple[Chn], np.float_],
) -> np.float_:
    return poly_eval(coef=estimation, func=func, x=wavenumber) - reference
