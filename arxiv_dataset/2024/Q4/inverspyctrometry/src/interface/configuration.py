import functools
import json
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, DirectoryPath, FilePath, model_validator, field_validator, Field

from src.database.database import DatabaseSchema
from src.outputs.resolver import resolve_path


class DirectoryPathSchema(BaseModel):
    data: DirectoryPath = Field(title="DATA_DIR")
    datasets: DirectoryPath = Field(title="DATASETS_DIR")
    notebooks: DirectoryPath = Field(title="NOTEBOOKS_DIR")
    reports: DirectoryPath = Field(title="REPORTS_DIR")
    src: DirectoryPath = Field(title="SRC_DIR")

    @model_validator(mode="before")
    def register_project_dir(cls, values):
        """Registers the PROJECT_DIR resolver using OmegaConf"""
        resolver_func = functools.partial(lambda x: x, x=cls.__project_dir())
        OmegaConf.register_new_resolver(name="PROJECT_DIR", resolver=resolver_func, replace=True)
        return values

    @field_validator("*", mode="before")
    def resolve_dir(cls, directory: DirectoryPath) -> DirectoryPath:
        """Takes a field and completes / resolves it based on what was registered in the OmegaConf resolver before"""
        directory_resolved = resolve_path(path=directory)
        return directory_resolved

    @model_validator(mode="after")
    def register_attributes_dirs(cls, values):
        for field, value in cls.model_fields.items():
            directory = getattr(values, field).as_posix()
            resolver_func = functools.partial(lambda x: x, x=directory)
            OmegaConf.register_new_resolver(name=value.title, resolver=resolver_func, replace=True)
        return values

    @property
    def project(self):
        return self.__project_dir()

    @staticmethod
    def __project_dir() -> DirectoryPath:
        return Path(__file__).resolve().parents[2]


class DatabasePathSchema(BaseModel):
    characterizations: FilePath
    datasets: FilePath
    interferometers: FilePath
    inversion_protocols: FilePath
    noise_levels: FilePath
    experiments: FilePath

    @field_validator("*", mode="before")
    def resolve_filepaths(cls, filepath: FilePath) -> FilePath:
        """Takes a field and completes / resolves it based on what was registered in the OmegaConf resolver before"""
        filepath_resolved = resolve_path(path=filepath)
        return filepath_resolved

    def open(self) -> DatabaseSchema:
        characterizations_dict = self._load_dict_from_json(path=self.characterizations)
        datasets_dict = self._load_dict_from_json(path=self.datasets)
        experiments = self._load_dict_from_json(path=self.experiments)
        interferometers_dict = self._load_dict_from_json(path=self.interferometers)
        inversion_protocols_dict = self._load_dict_from_json(path=self.inversion_protocols)
        noise_levels_dict = self._load_dict_from_json(path=self.noise_levels)
        return DatabaseSchema(
            **characterizations_dict,
            **datasets_dict,
            **experiments,
            **interferometers_dict,
            **inversion_protocols_dict,
            **noise_levels_dict,
        )

    @staticmethod
    def _load_dict_from_json(path: Path) -> dict:
        with open(path) as json_opened:
            database_dict = json.load(json_opened)
        # database_obj = OmegaConf.create(database_dict)
        # database_dict = OmegaConf.to_container(database_obj, resolve=True)
        return database_dict


class ConfigSchema(BaseModel):
    directory_paths: DirectoryPathSchema
    database_paths: DatabasePathSchema

    def database(self) -> DatabaseSchema:
        return self.database_paths.open()


def load_config(config_sub_path: Path | str = "data/database/config.json") -> ConfigSchema:
    project_path = Path(__file__).resolve().parents[2]
    config_path = project_path / config_sub_path
    with open(config_path) as config_file:
        config_dict = json.load(fp=config_file)
    return ConfigSchema(**config_dict)
