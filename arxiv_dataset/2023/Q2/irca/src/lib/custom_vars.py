"""This module defines the custom types and enums used within the project"""

from __future__ import annotations
from typing import TypeVar, NewType, List, Callable, Any, Union
from enum import Enum

import numpy as np


Row = NewType("Row", np.int32)  # Rows
Col = NewType("Col", np.int32)  # Columns
Ifm = NewType("Ifm", np.int32)  # Interferometers
Acq = NewType("Acq", np.int32)  # Acquisitions
Chn = NewType("Chn", np.int32)  # Channels
Smp = NewType("Smp", np.int32)  # Samples of the discrete space
Deg = NewType("Deg", np.int32)  # Degree of chosen parameter(s)
DgO = NewType("DgO", np.int32)  # Degree of other parameters
Mtd = NewType("Mtd", np.int32)  # Reconstruction/inversion methods
Qix = NewType("Qix", np.int32)  # Quality indices
XAx = TypeVar("XAx", bound=np.int32)   # X-axis length
Idx = NewType("Idx", Union[int, slice, List[int], np.ndarray, None])
Fig = NewType("Fig", Any)  # matplotlib.figure.Figure
Axs = NewType("Axs", Any)  # matplotlib.axes._subplots.AxesSubplot


UnitLike = TypeVar("UnitLike", int, float, List[int], List[float], np.ndarray)
DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)
NumpyPolynomial = NewType(
    "NumpyPolynomial",
    Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]],
)


class NumpyPolynomialEnum(str, Enum):
    POLYNOMIAL = "Polynomial"
    CHEBYSHEV = "Chebyshev"
    LEGENDRE = "Legendre"
    LAGUERRE = "Laguerre"
    HERMITE = "Hermite"
    HERMITE_E = "HermiteE"


class OriginEnum(str, Enum):
    """origin parameter of matplotlib imshow"""
    UPPER = "upper"
    LOWER = "lower"


class AcquisitionCategoryEnum(str, Enum):
    """Available categories of imspoc acquisitions"""
    CHARACTERIZATION = "characterization"
    COREGISTRATION = "coregistration"
    INSITU = "insitu"
    FLAT = "flat"
    CENTER = "center_spot"
    HYPERSPECTRAL = "hyperspectral"


class CharacterizeMethodEnum(str, Enum):
    """Available list of implemented characterization methods."""
    MAXIMUM_LIKELIHOOD = "ml"
    GAUSS_NEWTON = "gn"
    EXHAUSTIVE_SEARCH = "es"


class InversionMethodEnum(str, Enum):
    """Available list of implemented inversion methods."""
    RIDGE_REGRESSION = "rr"
    TSVD = "tsvd"
    LORIS_VERHOEVEN = "lv"  # "loris"


class FileFormatEnum(str, Enum):
    """Available list of formats for dictionary generation files"""
    JSON = "json"


class SerializerEnum(str, Enum):
    """Available list of dumping formats"""
    CSV = "csv"
    NUMPY = "numpy"
    PICKLE = "pickle"
