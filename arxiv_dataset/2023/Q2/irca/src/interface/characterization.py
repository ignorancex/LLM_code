from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, replace

import numpy as np

from src.characterization.protocols import CharacterizationProtocol
from src.characterization.interface import CharacterizeOptionsListSchema
from src.characterization.preprocessing import (
    CharacterizationPixel,
    Parameters,
    Characterization,
)
from src.interface.input import ImspocOptions
from src.lib.custom_vars import Fig, Axs


@dataclass(frozen=True)
class CharacterizationInterface:
    acquisition: CharacterizationPixel
    method: CharacterizeOptionsListSchema
    device: ImspocOptions
    _characterization: Characterization | None = None

    @classmethod
    def from_file(
        cls,
        acquisition_folder: str | Path,
        method_json: str | Path,
        device_json: str | Path,
    ) -> "CharacterizationInterface":
        acquisition = CharacterizationPixel.load(folder=acquisition_folder)
        method = CharacterizeOptionsListSchema.parse_file(f"{method_json}")
        device = ImspocOptions.parse_file(f"{device_json}")
        return cls(
            acquisition=acquisition,
            method=method,
            device=device,
        )

    def characterization(self) -> Characterization:
        if self._characterization is not None:
            return self._characterization
        interferometers = self.acquisition.data.shape[0]
        init = Parameters.init_zeros(interferometers=interferometers)
        init = replace(init, opd=self.device.opd[:, np.newaxis])
        characterize_protocol = CharacterizationProtocol(
            calibration=self.acquisition,
            options=self.method,
            init=init,
        )
        characterization = characterize_protocol.characterize(model=0)
        object.__setattr__(self, "_characterization", characterization)
        return characterization

    def visualize(self, interferometer: int = 0) -> tuple[Fig, Axs]:
        characterization_pixel = self.acquisition
        characterization = self.characterization()
        wavenumbers = self.acquisition.central_wavenumbers
        transfer = characterization.transfer_function(wavenumber=wavenumbers)
        return characterization_pixel.visualize_compare(
            estimation=transfer,
            interferometer=interferometer,
        )
