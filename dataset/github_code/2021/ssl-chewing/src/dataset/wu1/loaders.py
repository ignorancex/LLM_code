from collections.abc import Iterable
from enum import Enum
from typing import List

import numpy as np

from dataset.augmentors import BaseAugmentor
from dataset.normalizers import BaseNormalizer
from dataset.reshapers import BaseReshaper, DefaultReshaper
from dataset.template.baseloader import BaseLoader
from dataset.wu1.utils import load_x_file
from dataset.wu1.wu1metadata import WU1Metadata, get_x_file_path
from utilities.numpyutils import is_numpy_2d_vector


class LoaderPadding(Enum):
    NONE = 0
    BEGINNING = 1
    MIDDLE = 2
    END = 3
    RANDOM = 4


class LoaderTrimming(Enum):
    NONE = 0
    BEGINNING = 1
    MIDDLE = 2
    END = 3
    RANDOM = 4


class Loader(BaseLoader):
    WSIZE_ACTUAL: float = -1.0

    def __init__(self,
                 wu1metadata: WU1Metadata,
                 wsize: int = WSIZE_ACTUAL,
                 padding: LoaderPadding = LoaderPadding.NONE,
                 trimming: LoaderTrimming = LoaderTrimming.NONE,
                 normalizer: BaseNormalizer = None,
                 augmentors: List[BaseAugmentor] = None,
                 reshaper: BaseReshaper = DefaultReshaper(),
                 preload: bool = False):
        """
        Class that loads the actual audio data of the dataset.

        :param wu1metadata: The metadata of the WU1 dataset.
        :param wsize: The length (in samples) of the audio segments to be loaded. If set to WSIZE_ACTUAL, audio segments
                      are returned in their original size, however, this will cause problems with model training and is
                      included only for debugging purposes.
        :param padding: Type of padding to apply to the audio segment if its length is less than wsize.
        :param trimming: Type of trimming to apply to the audio segment if its length is more than wsize.
        :param normalizer: An object with a normalize function that normalizes a training sample.
        :param augmentors: An object with an augment function that augments a training sample. You can pass multiple
                           augmentors as a list.
        :param reshaper: An object with a reshape function that reshapes a training sample.
        :param preload: If set to True, all audio segments are preloaded into memory to increase performance.
        """
        assert isinstance(wu1metadata, WU1Metadata)
        assert isinstance(wsize, int)
        assert isinstance(padding, LoaderPadding)
        assert isinstance(trimming, LoaderTrimming)
        assert isinstance(normalizer, BaseNormalizer) or normalizer is None
        if augmentors is not None:
            assert isinstance(augmentors, list)
            for augmentor in augmentors:
                assert isinstance(augmentor, BaseAugmentor)
        assert isinstance(reshaper, BaseReshaper) or reshaper is None
        assert isinstance(preload, bool)

        self._md: WU1Metadata = wu1metadata
        self._wsize: int = wsize
        self._padding: LoaderPadding = padding
        self._trimming: LoaderTrimming = trimming
        self._normalizer: BaseNormalizer = normalizer
        self._augmentors: List[BaseAugmentor]
        if augmentors is None:
            self._augmentors = list()
        elif isinstance(augmentors, BaseAugmentor):
            self._augmentors = (augmentors,)
        elif isinstance(augmentors, Iterable):
            self._augmentors = augmentors
        else:
            raise ValueError("Unsupported type for augmentors")

        self.reshaper: BaseReshaper = reshaper
        self._preload: bool = preload

        # Check valid input
        if self._wsize == Loader.WSIZE_ACTUAL and (
                self._padding is not LoaderPadding.NONE or self._trimming is not LoaderTrimming.NONE):
            raise ValueError('Invalid input combination: if wsize is equal to LoaderConfiguration.WSIZE_ACTUAL, ' +
                             'both padding and trimming should be NONE')

        self._loader: BaseLoader
        if self._preload:
            self._loader = _PreLoader(wu1metadata)
        else:
            self._loader = _NaiveLoader(wu1metadata)

    def load(self, idx: int, augment: bool = False) -> np.ndarray:
        # assert isinstance(idx, int)

        # Load/get audio
        x = self._loader.load(idx)
        # Note - x is a numpy ndarray with shape=(n, 1)

        # Augment
        if augment and self._augmentors is not None:
            for augmentor in self._augmentors:
                x = augmentor.augment_single(x)

        # Trim/pad
        if self._wsize == Loader.WSIZE_ACTUAL:
            pass
        elif len(x) == self._wsize:
            pass
        elif len(x) < self._wsize:
            x = _pad_x(x, self._wsize, self._padding)
        elif len(x) > self._wsize:
            x = _trim_x(x, self._wsize, self._trimming)

        # Normalize
        if self._normalizer is not None:
            x = self._normalizer.normalize(x)

        # Reshape
        x = self.reshaper.reshape(x)

        return x

    def get_shape(self):
        if not isinstance(self.reshaper, DefaultReshaper):
            return self.reshaper.get_output_shape()
        elif self._wsize != self.WSIZE_ACTUAL:
            return self._wsize, 1
        else:
            return None, 1


class _PreLoader(BaseLoader):
    """
    Loader for chews/bouts that pre-loads all chews/bouts in memory.
    """

    def __init__(self, wu1metadata: WU1Metadata):
        assert isinstance(wu1metadata, WU1Metadata)

        self._x = list()
        for i in range(wu1metadata.length):
            file_name = get_x_file_path(wu1metadata, i)
            self._x.append(load_x_file(file_name))

    def load(self, idx: int) -> np.ndarray:
        return self._x[idx]


class _NaiveLoader(BaseLoader):
    """
    Loader for chews/bouts that loads each chew/bout from mat-files upon request.
    """

    def __init__(self, wu1metadata: WU1Metadata):
        assert isinstance(wu1metadata, WU1Metadata)

        self._md = wu1metadata

    def load(self, idx: int) -> np.ndarray:
        file_name = get_x_file_path(self._md, idx)

        return load_x_file(file_name)


def _pad_x(x: np.ndarray, wsize: int, padding: LoaderPadding) -> np.ndarray:
    assert is_numpy_2d_vector(x)
    assert isinstance(wsize, int)
    assert isinstance(padding, LoaderPadding)
    assert len(x) < wsize

    n = wsize - len(x)  # number of items (samples) to add
    if padding is LoaderPadding.NONE:
        return x  # no padding requested
    elif padding is LoaderPadding.BEGINNING:
        return np.pad(x, ((n, 0), (0, 0)), 'constant')
    elif padding is LoaderPadding.MIDDLE:
        n1 = int(n / 2)
        n2 = n - n1
        return np.pad(x, ((n1, n2), (0, 0)), 'constant')
    elif padding is LoaderPadding.END:
        return np.pad(x, ((0, n), (0, 0)), 'constant')
    elif padding is LoaderPadding.RANDOM:
        n1 = np.random.randint(0, n)
        n2 = n - 1
        return np.pad(x, ((n1, n2), (0, 0)), 'constant')
    else:
        raise ValueError('Unsupported padding: ' + str(padding))


def _trim_x(x: np.ndarray, wsize: int, trimming: LoaderTrimming) -> np.ndarray:
    assert is_numpy_2d_vector(x)
    assert isinstance(wsize, int)
    assert isinstance(trimming, LoaderTrimming)
    assert 0 < wsize < len(x)

    n = len(x) - wsize  # number of items (samples) to be trimmed
    if trimming is LoaderTrimming.NONE:
        return x
    elif trimming is LoaderTrimming.BEGINNING:
        return x[n:]
    elif trimming is LoaderTrimming.MIDDLE:
        n1 = int(n / 2)
        return x[n1:n1 + wsize]
    elif trimming is LoaderTrimming.END:
        return x[:wsize]
    elif trimming is LoaderTrimming.RANDOM:
        n1 = np.random.randint(0, n)
        return x[n1:n1 + wsize]
    else:
        raise ValueError('Unsupported trimming: ' + str(trimming))


def load_durations(wu1metadata: WU1Metadata, loader: Loader = None, in_sec: bool = False) -> np.ndarray:
    """
    Get a list of durations (in samples or seconds) for all chews or bouts.

    NOTE - this will load all chews/bouts from storage in order to measure durations.

    :param wu1metadata: The dataset metadata.
    :param loader: If provided, this loader is used to obtain durations, otherwise, a new loader is instantiated for a
                   one-time use.
    :param in_sec: If true, durations are returned in seconds, otherwise in number of samples.
    :return: The list of durations.
    """
    assert isinstance(wu1metadata, WU1Metadata)
    assert isinstance(loader, Loader) or loader is None
    assert isinstance(in_sec, bool)

    if loader is None:
        loader = Loader(wu1metadata)

    if in_sec:
        dtype = float
    else:
        dtype = int

    durations = np.zeros(wu1metadata.length, dtype=dtype)
    for i in range(wu1metadata.length):
        x = loader.load(i, False)
        durations[i] = len(x)

    if in_sec:
        durations /= wu1metadata.fs_hz

    return durations
