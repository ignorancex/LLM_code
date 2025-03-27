"""
Module that provides various audio augmentors.

Each augmentor class should provide a function called 'augment' with the following signature:
    augment(x)
where x is a numpy ndarray (len(x.shape) should be equal to 1), and should return an ndarray
with the same shape that contains the augmented version of x.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Tuple, List

import numpy as np
from scipy.signal import buttord, butter, lfilter, sosfilt

from dataset.template.commons import PureAbstractError
from utilities.numpyutils import is_numpy_vector
from utilities.typingutils import is_typed_list, is_typed_tuple


class BaseAugmentor(ABC):
    """
    Base class for augmentors.
    """

    @abstractmethod
    def augment_single(self, x: np.ndarray) -> np.ndarray:
        """Augments a single sample."""
        raise PureAbstractError()

    def augment_batch(self, batch: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
        """
        Augments an entire batch by calling ``augment`` on each sample.

        :param batch: The batch to augment. It can either be a list (where each item is a sample) or a numpy array
                      (where the first dimension, axis=0, is the batch-index dimension).
        :return: The augmented batch (type is the same as the input)
        """
        assert isinstance(batch, List) or isinstance(batch, np.ndarray)

        augmented_batch: List = [self.augment_single(v) for v in batch]

        if isinstance(batch, List):
            return augmented_batch
        if isinstance(batch, np.ndarray):
            return np.stack(augmented_batch)
        raise ValueError("This should never be reached")


class RandomAugmentor(BaseAugmentor):
    def __init__(self, augmentors: List[BaseAugmentor], p: List[float] = None):
        """
        Applies a random augmentation.

        Each input is augment by randomly selecting an augmentor from a pool of augmentors.

        :param augmentors: A list of augmentors to randomly select from
        :param p: The probability distribution by which an augmentor is selected each time (None for uniform distribution).
        """
        assert is_typed_list(augmentors, BaseAugmentor) and len(augmentors) > 1
        assert is_typed_list(p, float) or p is None

        if p is None:
            p = [1 / len(augmentors) for augmentor in augmentors]

        assert len(augmentors) == len(p)

        self._augmentors: List[BaseAugmentor] = augmentors
        self._p: np.ndarray = np.array(p)
        self._cp: np.ndarray = np.cumsum(p)

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        idx: int = np.sum(np.random.random_sample() > self._cp)

        return self._augmentors[idx].augment_single(x)


class MultiAugmentor(BaseAugmentor):
    """
    A simple wrapper that is used to apply multiple augmentors, one after the other
    """

    def __init__(self, augmentors: List[BaseAugmentor] = None):
        """
        Create an augmentor that applies a list of augmentors. The list can be defined by the constructor, or can be
        modified at any time by accessing the ``augmentors: List[BaseAugmentor]`` attribute.

        :param augmentors: The list of augmentors to apply
        """
        assert is_typed_list(augmentors, BaseAugmentor) or augmentors is None

        self.augmentors: List[BaseAugmentor]
        if augmentors is None:
            self.augmentors = []
        else:
            self.augmentors = augmentors

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        for augmentor in self.augmentors:
            x = augmentor.augment_single(x)

        return x


class OffsetAugmentor(BaseAugmentor):
    def __init__(self, max_offset: int, min_offset: int = 0, step: int = 1):
        """
        Create a simple augmentor that offsets the beginning of the window.

        Each time the augment function is called, a random integer offset is chosen in [min_offset, max_offset], that
        defines how much the window is off-set from the original beginning.

        Example: if min_offset=0 and max_offset=10, a random integer offset can be 6, which means the first 6 samples of
        x will be discarded.

        :param max_offset: Maximum offset.
        :param min_offset: Minimum offset.
        :param step: The number of samples to advance from the beginning.
        """
        assert isinstance(max_offset, int)
        assert isinstance(min_offset, int)
        assert isinstance(step, int)
        assert 0 <= min_offset <= max_offset
        assert 1 <= step

        self._max_offset: int = max_offset
        self._min_offset: int = min_offset

        self._idxs: np.ndarray = np.arange(min_offset, max_offset, step)

    def augment_single(self, x: np.ndarray, return_offset: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(x, np.ndarray)
        assert isinstance(return_offset, bool)

        offset: np.ndarray = np.random.choice(self._idxs)

        y = x[offset:]
        if return_offset:
            return y, offset
        else:
            return y


class TrimmerAugmentor(BaseAugmentor):
    """An augmentor that trims the input vector if its length exceeds a fixed threshold."""

    def __init__(self, max_size: int):
        assert isinstance(max_size, int)

        self._max_size: int = max_size

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        if x.size > self._max_size:
            x = x[:self._max_size]

        return x


class NoiseAugmentor(BaseAugmentor):
    class NoiseType(Enum):
        NORMAL = 0
        UNIFORM = 1

    def __init__(self, noise_type: NoiseType = NoiseType.NORMAL, p1: float = 0.0, p2: float = 1.0):
        """
        Create a simple augmentor that adds noise to the window.

        The following types of noise are supported:

        - NORMAL: normal distribution random samples. In this case, p1 is the distribution mean, and p2 is the
                  distribution standard deviation.
        - UNIFORM: uniform distribution random samples. In this case, [p1, p2) is the half-open interval of the uniform
                   distribution.

        :param noise_type: Type of noise to add
        :param p1: first distribution parameter
        :param p2: second distribution parameter
        """
        assert isinstance(noise_type, NoiseAugmentor.NoiseType)
        assert isinstance(p1, float)
        assert isinstance(p2, float)

        self._noise_type: NoiseAugmentor.NoiseType = noise_type
        self._p1: float = p1
        self._p2: float = p2

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        if self._noise_type is NoiseAugmentor.NoiseType.NORMAL:
            v = np.random.randn(*x.shape)
            v = self._p2 * (v + self._p1)
        elif self._noise_type is NoiseAugmentor.NoiseType.UNIFORM:
            v = self._p1 + (self._p2 - self._p1) * np.random.random_sample(x.shape)
        else:
            raise ValueError('Unsupported noise_type: ' + str(self._noise_type))

        return x + v


class IIRFilterAugmentorOrig(BaseAugmentor):
    def __init__(self, b: np.ndarray, a: np.ndarray):
        assert is_numpy_vector(b)
        assert is_numpy_vector(a)

        self._b: np.ndarray = b
        self._a: np.ndarray = a

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        return lfilter(self._b, self._a, x)


class IIRFilterAugmentor(BaseAugmentor):
    def __init__(self, sos):
        self._sos = sos

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        return sosfilt(self._sos, x)


def easy_butterworth_augmentor(fs_Hz: float, wp: Union[float, Tuple[float, float]],
                               ws: Union[float, Tuple[float, float]], gpass_dB: float, gstop_dB: float, btype: str) \
        -> IIRFilterAugmentor:
    """
    Create an ``IIRFilterAugmentor`` that uses a Butterworth filter.

    :param fs_Hz: Sampling frequency (in Hz)
    :param wp: Pass frequency (digital, in [0, 1]). It can be a list of 2 frequencies for band-pass filters
    :param ws: Stop frequency (digital, in [0, 1]). It can be a list of 2 frequencies for band-pass filters
    :param gpass_dB: Maximum ripple in pass-band (in dB)
    :param gstop_dB: Minimum attenuation in stop-band (in dB)
    :param btype: Type of filter: 'lowpass', 'highpass', 'bandpass'
    :return: The IIRFilterAugmentor object
    """

    def check_w(w):
        return isinstance(w, float) or (is_typed_tuple(w, float) and len(w) == 2)

    assert isinstance(fs_Hz, float)
    assert check_w(wp)
    assert check_w(ws)
    assert isinstance(gpass_dB, float)
    assert isinstance(gstop_dB, float)
    assert isinstance(btype, str)

    hfs: float = fs_Hz / 2

    if isinstance(wp, float):
        wp = wp / hfs
        ws = ws / hfs
    else:
        wp = (wp[0] / hfs, wp[1] / hfs)
        ws = (ws[0] / hfs, ws[1] / hfs)

    n, wn = buttord(wp, ws, gpass_dB, gstop_dB)

    # ba = butter(n, wn, btype)
    sos = butter(n, wn, btype, output='sos')

    # return IIRFilterAugmentorOrig(ba[0], ba[1])
    return IIRFilterAugmentor(sos)


class RandomAmplifierAugmentor(BaseAugmentor):
    def __init__(self, min_g: float, max_g: float):
        assert isinstance(min_g, float)
        assert isinstance(max_g, float)

        self._offset: float = min_g
        self._range: float = max_g - min_g

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        g: float = self._range * np.random.random_sample() + self._offset

        return g * x
