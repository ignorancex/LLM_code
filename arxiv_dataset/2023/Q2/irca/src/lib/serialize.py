from __future__ import annotations
from typing import Type, Callable, Any
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from dataclasses import dataclass, fields

import numpy as np

from src.lib.custom_vars import DataClass, DataClassType, SerializerEnum


class Serializer(ABC):
    @staticmethod
    @abstractmethod
    def dump(instance: DataClass, folder_path: Path | str) -> None:
        """Dumps a dataclass whose field are made up only of numpy arrays"""
        ...

    @staticmethod
    @abstractmethod
    def load(class_obj: DataClassType, folder_path: Path | str) -> DataClass:
        """Loads a class from dumped files"""
        ...

    @classmethod
    def load_or_none(
            cls, class_obj: DataClassType, folder_path: Path | str
    ) -> DataClass | None:
        """Loads a class from dumped files"""
        try:
            return cls.load(class_obj=class_obj, folder_path=folder_path)
        except (OSError, FileNotFoundError, TypeError):
            return None


class NumpySerializer(Serializer):
    @staticmethod
    def dump(instance: DataClass, folder_path: Path | str) -> None:
        for attrib in fields(instance):
            name = attrib.name
            save_file = Path(folder_path).joinpath(f"{name}.npy")
            np.save(file=f"{save_file}", arr=getattr(instance, name))

    @staticmethod
    def load(class_obj: DataClassType, folder_path: Path | str) -> DataClass:
        out_dict = {}
        for attrib in fields(class_obj):
            name = attrib.name
            array = NumpyArraySaver(name=name, folder_path=folder_path).load()
            out_dict.update({name: array})
        return class_obj(**out_dict)


class PickleSerializer(Serializer):
    @staticmethod
    def dump(instance: DataClass, folder_path: Path | str) -> None:
        filename = f"{instance.__class__.__name__}.pickle"
        save_path = Path(folder_path).joinpath(filename)
        with open(save_path, "wb") as file:
            pickle.dump(obj=instance, file=file)

    @staticmethod
    def load(class_obj: DataClassType, folder_path: Path | str) -> DataClass:
        filename = f"{class_obj.__name__}.pickle"
        save_path = Path(folder_path).joinpath(filename)
        with open(save_path, "rb") as file:
            out = pickle.load(file=file)
        return out


def serializer_factory(fmt: SerializerEnum) -> Type[Serializer]:
    """
    Factory function to generate a class in charge to serialize data of a
    dataclass uniquely composed of numpy arrays with the same height.
    """
    dictionary = {
        (SerializerEnum.PICKLE, PickleSerializer),
        (SerializerEnum.NUMPY, NumpySerializer),
    }
    return next(dic[1] for dic in dictionary if fmt == dic[0])


class ArraySaver(ABC):
    name: str
    folder_path: Path | str

    @abstractmethod
    def file_path(self) -> Path:
        """Defines the save file path"""

    @abstractmethod
    def dump(self, array: np.ndarray) -> None:
        """Dumps a 1d or 2d numpy array to a save file"""

    @abstractmethod
    def load(self) -> np.ndarray:
        """Loads a 1d or 2d numpy array from a save file"""

    def exists(self) -> bool:
        """Checks if save file can load correctly"""
        return self.file_path().exists()

    def dynamic_load(
        self, function: Callable[[Any], np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Loads the array from the savefile if it exists, else computes the
        function and saves it to the target position
        """
        if self.exists():
            out = self.load()
        else:
            out = function(**kwargs)
            self.dump(array=out)
        return out


@dataclass(frozen=True)
class NumpyArraySaver(ArraySaver):
    name: str
    folder_path: Path | str

    def file_path(self) -> Path:
        filename = f"{self.name}.npy"
        return Path(self.folder_path).joinpath(filename)

    def dump(self, array: np.ndarray) -> None:
        save_file = self.file_path()
        np.save(file=f"{save_file}", arr=array)

    def load(self) -> np.ndarray:
        save_file = self.file_path()
        return np.load(file=f"{save_file}")


@dataclass(frozen=True)
class PickleArraySaver(ArraySaver):
    name: str
    folder_path: Path | str

    def file_path(self) -> Path:
        filename = f"{self.name}.pickle"
        return Path(self.folder_path).joinpath(filename)

    def dump(self, array: np.ndarray) -> None:
        save_path = self.file_path()
        with open(save_path, "wb") as file:
            pickle.dump(obj=array, file=file)

    def load(self) -> np.ndarray:
        save_path = self.file_path()
        with open(save_path, "rb") as file:
            out = pickle.load(file=file)
        return out


def array_saver_factory(
    kind: SerializerEnum,
    name: str,
    folder_path: Path | str,
    **kwargs,
) -> ArraySaver:
    """
    Factory function to generate a class in charge to serialize data of a
    dataclass uniquely composed of numpy arrays with the same height.
    """
    dictionary = {
        (SerializerEnum.PICKLE, PickleArraySaver),
        (SerializerEnum.NUMPY, NumpyArraySaver),
    }
    array_saver = next(dic[1] for dic in dictionary if kind == dic[0])
    return array_saver(name=name, folder_path=folder_path, **kwargs)
