from enum import Enum
from typing import NewType, Type

Wvn = NewType(name='Wvn', tp=int)
Opd = NewType(name='Opd', tp=int)
Acq = NewType(name='Acq', tp=int)
Deg = NewType(name='PolyCoef', tp=int)


class DatasetTitle(str, Enum):
    SOLAR = "solar"
    SHINE = "shine"
    SPECIM = "specim"
    MC_451 = "mc-451"
    MC_651 = "mc-651"


class DatasetCategory(str, Enum):
    SPECTRUM = "spectrum"
    INTERFEROGRAM = "interferogram"


class DeviceType(str, Enum):
    SOLAR = "N/A"
    SHINE = "shine"
    SPECIM = "specim"
    IMSPOC_UV_2 = "imspoc_uv_2"


class RegularizationParameterKey(str, Enum):
    NOT_APPLICABLE = "N/A"
    TSVD = "penalization_ratio"
    RIDGE_REGRESSION = "penalization"
    LORIS_VERHOEVEN = "regularization_parameter"


class RegularizationParameterListSpace(str, Enum):
    NOT_APPLICABLE = "N/A"
    LINSPACE = "linspace"
    LOGSPACE = "logspace"


class InterferometerType(str, Enum):
    MICHELSON = "Michelson"
    FABRY_PEROT = "Fabry-Perot"


class InversionProtocolType(str, Enum):
    HAAR = "haar"  # Haar method inversion
    IDCT = "idct"  # Inverse Discrete Cosine Transform
    PSEUDO_INVERSE = "pseudo_inverse"
    TSVD = "truncated_svd"  # Truncated Singular Value Decomposition
    RIDGE_REGRESSION = "ridge_regression"
    LORIS_VERHOEVEN = "loris_verhoeven"
    ADMM = "admm"  # Alternating Direction Method of Multipliers


class LinearOperatorMethod(str, Enum):
    NOT_APPLICABLE = "N/A"
    IDENTITY = "identity"
    DCT = "dct"  # Discrete Cosine Transform
    TV = "tv"  # Total Variation


class NormOperatorType(list, Enum):
    NOT_APPLICABLE = ["N/A", []]
    L1O = ["l1o", []]
    L2O = ["l2o", []]
    L112 = ["l112", [2, 0, 1]]
    L121 = ["l121", [1, 0, 2]]
    L211 = ["l211", [0, 1, 2]]


class EnumInvalidOptionError(Exception):
    """
    Enhanced ValueError for entering an invalid option among an Enum's options,
        that takes the Enum and the option as inputs,
        and displays the available options of the Enum.
    """
    def __init__(self, option: str, enum_class: Type[Enum]):
        self.option = option
        self.enum_class = enum_class
        message = self.compose_error_message()
        super().__init__(message)

    def compose_error_message(self):
        invalid_option_str = self.invalid_option_message()
        enum_options_str = self.enum_options_message()
        message = invalid_option_str + enum_options_str
        return message

    def invalid_option_message(self):
        message = f"\nOption '{self.option}' is not supported for {self.enum_class.__name__}.\n"
        return message

    def enum_options_message(self):
        message = f"The supported options for {self.enum_class.__name__} are:\n"
        for member in self.enum_class:
            message += f"- {member.value}\n"
        return message
