from __future__ import annotations
from typing import Tuple, List, Optional, Union
from collections.abc import Sequence

from pydantic import BaseModel, conint, confloat, validator
import numpy.polynomial as npp
import numpy as np

from src.lib.custom_vars import (
    Smp,
    Ifm,
    NumpyPolynomialEnum,
    CharacterizeMethodEnum,
    NumpyPolynomial,
)


class ParameterSchema(BaseModel):
    """
    Schema for the options of the parameters model
    The minimum and maximum are relative to the top estimable value for a
    particular parameter
    (eg: 2*pi for the phase shift, the Shannon theorem limit for the opd)
    """

    samples: conint(ge=0) = 1000
    degree: conint(ge=0) = 0
    choice: bool = True
    relative_minimum: confloat(ge=0) = 0
    relative_maximum: confloat(ge=0) = 1
    relative_span: confloat(ge=0) = 1

    def mask(
        self, init: np.ndarray[tuple[Ifm], np.float64]
    ) -> np.ndarray[tuple[Ifm, Smp], np.float64]:
        """
        Returns a mask from the given parameters.
        init must be an array of values between 0 and 1
        """
        mask = np.zeros(shape=(init.shape[0], self.samples), dtype=bool)
        mask_min = self.relative_minimum * self.samples
        mask_max = self.relative_maximum * self.samples
        mask[:, int(mask_min): int(mask_max)] = True
        range_samples = np.arange(stop=self.samples)[np.newaxis, :]
        range_samples = range_samples / self.samples
        mask_low = range_samples > init[:, np.newaxis] - self.relative_span
        mask_high = range_samples < init[:, np.newaxis] + self.relative_span
        mask = np.logical_and(np.logical_and(mask, mask_low), mask_high)
        return mask


class ParameterSetSchema(BaseModel):
    opd: ParameterSchema = ParameterSchema()
    reflectivity: ParameterSchema = ParameterSchema()
    phase_shift: ParameterSchema = ParameterSchema()
    gain: ParameterSchema = ParameterSchema()

    def choice(self) -> list[str]:
        return [key for key in self.__dict__ if self.__dict__[key].choice]


class CharacterizeOptionsSchema(BaseModel):
    id: CharacterizeMethodEnum = CharacterizeMethodEnum.MAXIMUM_LIKELIHOOD
    average: bool = False
    model: int = 0
    polynomial: NumpyPolynomialEnum = NumpyPolynomialEnum.POLYNOMIAL
    parameters: ParameterSetSchema = ParameterSetSchema()

    @property
    def callable(self) -> NumpyPolynomial:
        return CallablePolynomial.direct(string=self.polynomial)

    @property
    def choice(self) -> list[str]:
        return self.parameters.choice()


class CallablePolynomial:
    associations = {
        (NumpyPolynomialEnum.POLYNOMIAL, npp.Polynomial),
        (NumpyPolynomialEnum.CHEBYSHEV, npp.Chebyshev),
        (NumpyPolynomialEnum.LEGENDRE, npp.Legendre),
        (NumpyPolynomialEnum.LAGUERRE, npp.Laguerre),
        (NumpyPolynomialEnum.HERMITE, npp.Hermite),
        (NumpyPolynomialEnum.HERMITE_E, npp.HermiteE),
    }

    @staticmethod
    def direct(string: NumpyPolynomialEnum | str) -> NumpyPolynomial:
        associations = CallablePolynomial.associations
        label = f"{string}"
        return next(dic[1] for dic in associations if label == f"{dic[0]}")

    @staticmethod
    def inverse(function: NumpyPolynomial) -> NumpyPolynomialEnum:
        associations = CallablePolynomial.associations
        return next(dic[0] for dic in associations if function == dic[1])


class CharacterizeOptionsListSchema(Sequence, BaseModel):
    __root__: List[CharacterizeOptionsSchema] = [CharacterizeOptionsSchema()]

    def __getitem__(self, item: int) -> CharacterizeOptionsSchema:
        return self.__root__[item]

    def __len__(self) -> int:
        return len(self.__root__)

    def __next__(self) -> CharacterizeOptionsSchema:
        return next(self)

    def label(self) -> str:
        """Shorthand representation of the method"""
        choice = 1 if self.__root__[-1].parameters.gain.choice else 0
        return f"{self.__root__[-1].id} m{self.__root__[-1].model} g{choice}"


class CharacterizationPreprocessingSchema(BaseModel):
    window: Union[conint(ge=1), Tuple[conint(ge=1), conint(ge=1)]] = (1, 1)
    percentile: confloat(ge=0, le=100) = 98
    average: bool = False
    local: bool = False
    crop: Optional[
        Tuple[
            Tuple[conint(ge=0), conint(ge=0)],
            Tuple[conint(ge=0), conint(ge=0)],
        ]
    ] = None
    flat: bool = True

    @validator("window")
    def window_tuple(cls, val) -> tuple[int, int]:
        if not isinstance(val, tuple):
            return val, val


class CharacterizationSchema(BaseModel):
    acquisition_id: int
    preprocessing: CharacterizationPreprocessingSchema
    methods: CharacterizeOptionsListSchema


class OptionsList(BaseModel):
    options_list: List[CharacterizeOptionsListSchema]
