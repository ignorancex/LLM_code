from typing import Sequence

from pydantic import BaseModel, RootModel


class ExperimentSchema(BaseModel):
    id: int
    title: str
    type: str
    dataset_ids: list[int]
    interferometer_ids: list[int]
    noise_level_indices: list[int]
    inversion_protocol_ids: list[int]
    description: str


class ExperimentListSchema(Sequence, RootModel):
    root: list[ExperimentSchema]

    def __getitem__(self, item: int):
        return self.root[item]

    def __len__(self):
        return len(self.root)
