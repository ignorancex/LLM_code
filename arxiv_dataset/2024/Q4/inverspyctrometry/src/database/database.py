import numpy as np
from pydantic import BaseModel

from src.common_utils.custom_vars import Wvn, DatasetCategory
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.database.characterizations import CharacterizationSchema
from src.database.datasets import DatasetSchema
from src.database.experiments import ExperimentSchema
from src.database.interferometers import InterferometerSchema
from src.database.inversion_protocols import InversionProtocolSchema
from src.direct_model.characterization import Characterization
from src.direct_model.interferometer import Interferometer
from src.inverse_model.protocols import InversionProtocol


class DatabaseSchema(BaseModel):
    characterizations: list[CharacterizationSchema]
    datasets: list[DatasetSchema]
    experiments: list[ExperimentSchema]
    interferometers: list[InterferometerSchema]
    inversion_protocols: list[InversionProtocolSchema]
    noise_levels: list[float]

    def characterization(self, characterization_id: int) -> Characterization:
        return self.characterizations[characterization_id].characterization()

    def characterization_wavenumbers(self, char_id: int) -> np.ndarray[tuple[Wvn], np.dtype[np.float64]]:
        char_ds_id = self.characterizations[char_id].source_dataset_id
        wavenumbers = np.load(self.datasets[char_ds_id].wavenumbers_path)
        return wavenumbers

    def characterization_dataset(self, char_id: int) -> Interferogram:
        dataset_id = self.characterizations[char_id].source_dataset_id
        interferograms = self.dataset(dataset_id=dataset_id)
        return interferograms

    def dataset(self, dataset_id: int) -> Spectrum | Interferogram:
        dataset = self.datasets[dataset_id].load()
        return dataset

    def dataset_spectrum(self, ds_id: int) -> Spectrum:
        spectrum = self.datasets[ds_id].spectrum()
        return spectrum

    def dataset_interferogram(self, ds_id: int) -> Interferogram:
        interferogram = self.datasets[ds_id].interferogram()
        return interferogram

    def dataset_central_wavenumbers(self, dataset_id: int) -> np.ndarray[tuple[Wvn], np.dtype[np.float64]]:
        central_wavenumbers = np.load(self.datasets[dataset_id].wavenumbers_path)
        return central_wavenumbers

    def interferometer(self, interferometer_id: int) -> Interferometer:
        interferometer = self.interferometers[interferometer_id].interferometer()
        return interferometer

    def inversion_protocol_list(self, inv_protocol_id: int) -> list[InversionProtocol]:
        inversion_protocol_list = self.inversion_protocols[inv_protocol_id].inversion_protocol_list()
        return inversion_protocol_list

    def inversion_protocol(
            self,
            inv_protocol_id: int,
            lambdaa: float,
            is_compute_and_save_cost: bool = False,
            experiment_id: int = -1,
    ) -> InversionProtocol:
        inversion_protocol = self.inversion_protocols[inv_protocol_id].inversion_protocol(
            lambdaa=lambdaa,
            is_compute_and_save_cost=is_compute_and_save_cost,
            experiment_id=experiment_id,
        )
        return inversion_protocol

    def inversion_protocol_lambdaas(self, inv_protocol_id: int) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        lambdaas = self.inversion_protocols[inv_protocol_id].lambdaas_schema.as_array()
        return lambdaas
