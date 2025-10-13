import attr
from typing import Tuple, Optional, Union
import numpy as np
from py_factor_graph.utils.matrix_utils import (
    get_quat_from_rotation_matrix,
    _check_transformation_matrix,
    get_rotation_matrix_from_transformation_matrix,
    get_translation_from_transformation_matrix,
    get_theta_from_transformation_matrix,
)
from py_factor_graph.utils.attrib_utils import (
    optional_float_validator,
    make_rot_matrix_validator,
    make_variable_name_validator,
)
from typing import Callable, List
from py_factor_graph.variables import PoseVariable2D
from attrs import define, field
from py_factor_graph.measurements import PoseToLandmarkMeasurement2D


@attr.s(frozen=True)
class BooleanVariable:
    name: str = attr.ib()
    true_value: bool = attr.ib(validator=attr.validators.instance_of(bool))
    timestamp: Optional[float] = attr.ib(default=None)


@define(frozen=True)
class ToyExamplePrior:
    name: str = attr.ib()
    center: np.ndarray = attr.ib()
    offset: float = attr.ib()
    Q: np.ndarray = attr.ib()

    @property
    def dimension(self) -> int:
        return self.true_value.shape[0]


@attr.s(frozen=True)
class VectorVariable:
    name: str = attr.ib()
    true_value: np.ndarray = attr.ib()
    timestamp: Optional[float] = attr.ib(default=None)

    @property
    def dimension(self) -> int:
        return self.true_value.shape[0]


@attr.s(frozen=False)
class UnknownDataAssociationMeasurement:
    name: str = attr.ib()
    measurement_list: List[Union[ToyExamplePrior, PoseToLandmarkMeasurement2D]] = (
        attr.ib()
    )
    boolean_variables: List[BooleanVariable] = attr.ib()
    timestamp: Optional[float] = attr.ib(default=None)
