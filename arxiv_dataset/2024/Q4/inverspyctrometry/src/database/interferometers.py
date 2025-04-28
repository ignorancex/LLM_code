from typing import Optional, Sequence, List

import numpy as np
from pydantic import BaseModel, RootModel

from src.common_utils.custom_vars import InterferometerType, Opd
from src.common_utils.utils import generate_sampled_opds
from src.direct_model.interferometer import Interferometer, interferometer_factory


class OPDSchema(BaseModel):
    num: int
    step: float
    start: float

    def as_array(self) -> np.ndarray[tuple[Opd], np.dtype[np.float64]]:
        return generate_sampled_opds(nb_opd=self.num, opd_step=self.step, opd_min=self.start)

    @property
    def max(self) -> float:
        return self.start + self.step * (self.num - 1)


class InterferometerSchema(BaseModel):
    id: int
    title: str
    category: str
    type: InterferometerType
    opds: OPDSchema
    transmittance_coefficients: list[list[float]]
    reflectance_coefficients: list[list[float]]
    phase_shift: list[float]
    order: int

    def interferometer(self) -> Interferometer:
        transmittance_coefficients = np.array(self.transmittance_coefficients)
        reflectance_coefficients = np.array(self.reflectance_coefficients)
        opds = self.opds.as_array()
        phase_shift = np.array(self.phase_shift)

        if phase_shift.size == 1:
            phase_shift = np.tile(phase_shift, self.opds.num)

        interferometer = interferometer_factory(
            option=self.type,
            transmittance_coefficients=transmittance_coefficients,
            opds=opds,
            reflectance_coefficients=reflectance_coefficients,
            order=self.order,
            phase_shift=phase_shift,
        )
        return interferometer


class InterferometerListSchema(Sequence, RootModel):
    root: List[InterferometerSchema]

    def __getitem__(self, item: int) -> InterferometerSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
