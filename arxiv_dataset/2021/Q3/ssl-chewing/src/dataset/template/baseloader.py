from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, NoReturn

import numpy as np


class BaseLoader(ABC):
    """
    Abstract class for a loader: an object that loads individual samples from a dataset.

    Child classes should implement the following methods: ``load``.
    """

    @abstractmethod
    def load(self, i: int) -> np.ndarray:
        """
        Load the i-th sample from a collection (can be an entire dataset, file, etc)

        :param i:
        :return:
        """
        raise NotImplementedError("Do not call load from BaseLoader")


class BaseFileLoader(BaseLoader, ABC):
    """
    Abstract class for a file-based loader.
    """

    def __init__(self, file: Path):
        assert isinstance(file, Path)

        self._file: Path = file
        self._io: IO or None = None

    def open(self) -> NoReturn:
        if self._io is not None:
            raise ValueError("IO is already opened")
        self._io: IO = open(self._file, 'rb')

    def close(self) -> NoReturn:
        if self._io is None:
            raise ValueError("IO is not opened")
        self._io.close()
        self._io = None
