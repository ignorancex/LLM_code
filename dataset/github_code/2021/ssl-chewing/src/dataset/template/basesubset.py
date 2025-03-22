from abc import ABC, abstractmethod
from typing import Union, List, Tuple, NoReturn, Optional
from warnings import warn

import numpy as np
from tensorflow.keras.utils import Sequence

from dataset.commons import SubsetType
from dataset.template.commons import PureAbstractError
from utilities.matlabutils import strcmp, find_unique
from utilities.typingutils import is_typed_list


class BaseSubset(Sequence, ABC):
    """
    Base class for a subset. Subsets inherit ``Sequence`` and so can be used directly as generators in Keras.

    Child classes should overwrite the following methods: ``shuffle``.
    Child classes can overwrite the following methods: ``__len__``, ``on_epoch_end``.
    """

    def __init__(self, subset_type: SubsetType, length: int, batch_size: int, shuffle_on_epoch_end: bool = True):
        assert isinstance(subset_type, SubsetType)
        assert isinstance(length, int)
        assert isinstance(batch_size, int)
        assert isinstance(shuffle_on_epoch_end, bool)

        self.subset_type: SubsetType = subset_type
        self.length: int
        self.batch_size: int = batch_size
        self._shuffle: bool = shuffle_on_epoch_end

    def __len__(self) -> int:
        """
        Returns the number of batches of the subset. The last batch may contain less than ``batch_size`` samples.
        """

        return np.ceil(self.length / self.batch_size).astype(np.int)

    def shuffle(self) -> NoReturn:
        """
        Dummy implementation that does nothing.
        """
        warn("This method has not been implemented. No shuffling will be done.")

    def on_epoch_end(self) -> NoReturn:
        """
        Shuffles the subset if ``shuffle_on_epoch_end`` was passed as ``True`` in the constructor.

        """
        if self._shuffle:
            self.shuffle()


class BaseSubsetBuilder(ABC):
    """
    A base class for creating train, test, and optionally validation subsets for experiments for datasets that are
    organized in "partitions" (i.e. subjects). Each dataset should implement one such class.

    Child classes should implement the following methods: ``get_partitions``, ``_get_subsets``.
    """

    @abstractmethod
    def get_partitions(self) -> List[str]:
        """
        Return a list of labels corresponding to the partitions. For example, if the partitions are subjects (as with
        typical LOSO configurations), this method should return a list of names or identifiers.
        """
        raise PureAbstractError()

    @abstractmethod
    def _get_subsets(self, train_idxs: List[int], test_idxs: List[int], validation_idxs: List[int] = None) \
            -> Tuple[BaseSubset, BaseSubset, Optional[BaseSubset]]:
        """
        Main method that returns the three subsets based on the partition indices. The following checks are performed
        before this method is called:

        - type checking on the three input arguments
        - sanity check: all three intersections are empty, and the union of all indices is equal to ``get_partitions()``

        :param train_idxs: The partition indices for the train subset
        :param test_idxs: The partition indices for the test subset
        :param validation_idxs: The partition indices for the validation subset (or None)
        :return: A tuple of the three subsets, in order (train, test, validation). Validation subset can be ``None``
        """
        raise PureAbstractError()

    def _get_train_subset_idxs(self, tst_idxs: List[int], validation_idxs: Optional[List[int]] = None) -> List[int]:
        """
        Convenience method for computing the train indices based on the test and validation indices.
        """
        assert is_typed_list(tst_idxs, int)
        assert is_typed_list(validation_idxs, int) or validation_idxs is None

        if validation_idxs is None:
            validation_idxs = []

        n: int = len(self.get_partitions())
        tst_len: int = len(tst_idxs)
        val_len: int = len(validation_idxs)

        assert 0 < tst_len
        assert 0 <= val_len
        assert tst_len + val_len < n

        all_idxs: np.ndarray = np.arange(n)
        tst_idxs: np.ndarray = np.array(tst_idxs)
        val_idxs: np.ndarray = np.array(validation_idxs)

        assert np.unique(tst_idxs).size == tst_len
        assert np.unique(val_idxs).size == val_len
        assert np.all(np.logical_and(0 <= tst_idxs, tst_idxs < n))
        assert np.all(np.logical_and(0 <= val_idxs, val_idxs < n))
        assert np.intersect1d(tst_idxs, val_idxs).size == 0  # TODO -

        # TODO - something about the type warning
        return np.setdiff1d(all_idxs, np.union1d(tst_idxs, val_idxs)).tolist()

    def split(self, test_partitions: List[str], validation_partitions: List[str] = None) \
            -> Tuple[BaseSubset, BaseSubset, Optional[BaseSubset]]:
        """
        Splits the dataset in train, test, and optionally validation subsets. If ``validation_partitions is None`` then
        the third item of the returned tuple is ``None``.

        :param test_partitions: The partition IDs to be included in the test subset
        :param validation_partitions: The partition IDs to be included in the validation subset
        :return: Train subset, test subset, [validation subset]
        """
        assert is_typed_list(test_partitions, str)
        assert is_typed_list(validation_partitions, str) or validation_partitions is None

        partitions: List[str] = self.get_partitions()

        test_idxs: List[int] = [find_unique(strcmp(partitions, p)) for p in test_partitions]

        validation_idxs: Optional[List[int]]
        if validation_partitions is not None:
            validation_idxs = [find_unique(strcmp(partitions, p)) for p in validation_partitions]
        else:
            validation_idxs = None

        train_idxs: List[int] = self._get_train_subset_idxs(test_idxs, validation_idxs)

        return self._get_subsets(train_idxs, test_idxs, validation_idxs)

    def random_split(self, test_length: int, validation_length: int = 0) \
            -> Tuple[BaseSubset, BaseSubset, Union[BaseSubset, None]]:
        """
        Splits the dataset randomly in train, test, and optionally validation subsets. If ``validation_length == 0``
        then the third item of the returned tuple is ``None``.

        :param test_length: The number of partitions to be included in the test subset
        :param validation_length: The number of partitions to be included in the validation subset
        :return: Train subset, test subset, [validation subset]
        """
        # The two positivity checks are obsolete, and used only to avoid raising numpy exceptions
        assert isinstance(test_length, int) and 0 < test_length
        assert isinstance(validation_length, int) and 0 <= validation_length

        all_idxs: np.ndarray = np.arange(len(self.get_partitions()))
        test_idxs_np: np.ndarray = np.random.choice(all_idxs, test_length, False)
        rem_idxs: np.ndarray = np.setdiff1d(all_idxs, test_idxs_np)
        validation_idxs_np: np.ndarray = np.random.choice(rem_idxs, validation_length, False)

        # TODO - something about the type warnings
        test_idxs: List[int] = test_idxs_np.tolist()

        validation_idxs: Optional[List[int]]
        if validation_length > 0:
            validation_idxs = validation_idxs_np.tolist()
        else:
            validation_idxs = None

        train_idxs: List[int] = self._get_train_subset_idxs(test_idxs, validation_idxs)

        return self._get_subsets(train_idxs, test_idxs, validation_idxs)

    def lopo_split(self, left_out_index: int, validation_length: int = 0, random_validation: bool = False) \
            -> Tuple[BaseSubset, BaseSubset, Optional[BaseSubset]]:
        """
        Split the dataset for a leave-one-partition-out (LOPO) experiment. If ``validation_length == 0`` then
        the third item of the returned tuple is ``None``.

        :param left_out_index: The index of the partition that is used for testing
        :param validation_length: The number of partitions to be  included in the validation subset
        :param random_validation: If true, validation indices are chosen in random, otherwise the are the partitions
                                  exactly before the test partition
        :return: Train subset, test subset, [validation subset]
        """
        n: int = len(self.get_partitions())

        assert isinstance(left_out_index, int) and 0 <= left_out_index < n
        assert isinstance(validation_length, int) and 0 <= validation_length
        assert isinstance(random_validation, bool)

        test_idxs: List[int] = [left_out_index]

        validation_idxs: Optional[List[int]]
        if validation_length > 0:
            if random_validation:
                validation_idxs = np.random.choice(
                    np.setdiff1d(np.arange(n), left_out_index), validation_length, False).tolist()
            else:
                validation_idxs = [(left_out_index - 1 - i) % n for i in range(validation_length)]
        else:
            validation_idxs = None

        train_idxs: List[int] = self._get_train_subset_idxs(test_idxs, validation_idxs)

        return self._get_subsets(train_idxs, test_idxs, validation_idxs)
