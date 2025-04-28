from typing import Optional, Sequence, List

import numpy as np
from pydantic import BaseModel, RootModel

from src.common_utils.custom_vars import InversionProtocolType, NormOperatorType, LinearOperatorMethod, \
    RegularizationParameterKey, RegularizationParameterListSpace
from src.inverse_model.operators import CTVOperator, NormOperator, LinearOperator
from src.inverse_model.protocols import InversionProtocol, inversion_protocol_factory


class LambdaasSchema(BaseModel):
    key: RegularizationParameterKey
    start: float
    stop: float
    num: int
    space: RegularizationParameterListSpace

    def as_array(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self.space == RegularizationParameterListSpace.LINSPACE:
            lambdaas = np.linspace(start=self.start, stop=self.stop, num=self.num)
        elif self.space == RegularizationParameterListSpace.LOGSPACE:
            lambdaas = np.logspace(start=self.start, stop=self.stop, num=self.num)
        elif self.space == RegularizationParameterListSpace.NOT_APPLICABLE:
            lambdaas = np.zeros(shape=self.num)
        else:
            raise ValueError(f"Option {self.space} is not supported. Only 'linspace' and 'logspace' are supported.")
        return lambdaas


class InversionProtocolSchema(BaseModel):
    """
    Schema for inversion protocol inputs used to launch experiments, not to directly initialize an Inversion Protocol
    """
    id: int
    title: str
    experiment_id: int
    experiment_description: str
    type: InversionProtocolType
    lambdaas_schema: LambdaasSchema
    norm_operator: NormOperatorType
    linear_operator: LinearOperatorMethod
    nb_iters: int

    @property
    def prox_functional(self) -> CTVOperator:
        return CTVOperator(norm=NormOperator.from_norm(norm=self.norm_operator))

    @property
    def domain_transform(self) -> LinearOperator:
        return LinearOperator.from_method(method=self.linear_operator)

    def parameters(
            self,
            lambdaa: float,
            is_compute_and_save_cost: bool = False,
            experiment_id: int = -1,
    ) -> dict:
        parameters = {
            self.lambdaas_schema.key.value: lambdaa,
            "prox_functional": self.prox_functional,
            "domain_transform": self.domain_transform,
            "nb_iters": self.nb_iters,
            "is_compute_and_save_cost": is_compute_and_save_cost,
            "experiment_id": experiment_id,
        }
        return parameters

    def inversion_protocol(
            self,
            lambdaa: float,
            is_compute_and_save_cost: bool = False,
            experiment_id: int = -1,
    ) -> InversionProtocol:
        inversion_protocol = inversion_protocol_factory(
            option=self.type,
            parameters=self.parameters(
                lambdaa=lambdaa,
                is_compute_and_save_cost=is_compute_and_save_cost,
                experiment_id=experiment_id,
            )
        )
        return inversion_protocol

    def inversion_protocol_list(self) -> list[InversionProtocol]:
        lambdaas = self.lambdaas_schema.as_array()
        inversion_protocol_list = [self.inversion_protocol(lambdaa=lambdaa) for lambdaa in lambdaas]
        return inversion_protocol_list


class InversionProtocolListSchema(Sequence, RootModel):
    root: List[InversionProtocolSchema]

    def __getitem__(self, item) -> InversionProtocolSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
