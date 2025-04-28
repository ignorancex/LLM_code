from pathlib import Path
from typing import Sequence, List, Optional

import numpy as np
from pydantic import BaseModel, RootModel, FilePath, field_validator

from src.common_utils.custom_vars import DatasetCategory
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.outputs.resolver import resolve_path


# TODO: Replace the type hints with Enums
# TODO: Switch Path to FilePath


class DatasetSchema(BaseModel):
    id: int
    title: str
    category: str
    device: str
    source: str
    path: FilePath
    wavenumbers_path: Optional[FilePath] = None
    wavenumbers_unit: Optional[str] = None
    opds_path: Optional[FilePath] = None
    opds_unit: Optional[str] = None

    @field_validator("*", mode="before")
    def resolve_filepaths(cls, filepath: FilePath) -> FilePath:
        """Takes a field and completes / resolves it based on what was registered in the OmegaConf resolver before"""
        filepath_resolved = resolve_path(path=filepath)
        return filepath_resolved

    def load(self) -> Spectrum | Interferogram:
        if self.category == DatasetCategory.SPECTRUM:
            return self.spectrum()
        elif self.category == DatasetCategory.INTERFEROGRAM:
            return self.interferogram()
        else:
            raise ValueError(f"Requested dataset (id={self.id}) category is not supported.")

    def spectrum(self) -> Spectrum:
        if self.category != DatasetCategory.SPECTRUM:
            raise ValueError(f"Requested dataset id={self.id} does not refer to a Spectrum dataset.")
        data = np.load(file=self.path)
        wavenumbers = np.load(file=self.wavenumbers_path)
        wavenumbers_unit = rf"{self.wavenumbers_unit}"
        return Spectrum(
            data=data,
            wavenumbers=wavenumbers,
            wavenumbers_unit=wavenumbers_unit,
        )

    def interferogram(self) -> Interferogram:
        if self.category != DatasetCategory.INTERFEROGRAM:
            raise ValueError(f"Requested dataset id={self.id} does not refer to an Interferogram dataset.")
        data = np.load(file=self.path)
        opds = np.load(file=self.opds_path)
        opds_unit = self.opds_unit
        return Interferogram(
            data=data,
            opds=opds,
            opds_unit=opds_unit,
        )


class DatasetListSchema(Sequence, RootModel):
    root: List[DatasetSchema]

    def __getitem__(self, item) -> DatasetSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
