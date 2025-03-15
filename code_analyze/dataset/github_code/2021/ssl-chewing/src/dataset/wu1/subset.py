from typing import List, Tuple, Union

import numpy as np

from dataset.commons import PartitionMode, SubsetType
from dataset.template.basesubset import BaseSubset
from dataset.wu1.commons import LabelMode
from dataset.wu1.wu1experiment import WU1Experiment
from utilities.numpyutils import is_numpy_1d_vector


def _split_lopo_simple(items: np.ndarray, ids: np.ndarray, test_idxs: np.ndarray, validation):
    assert is_numpy_1d_vector(items)
    assert is_numpy_1d_vector(ids)
    assert is_numpy_1d_vector(test_idxs)
    assert isinstance(validation, int) or isinstance(validation, np.ndarray)

    # Handle validation input argument
    if isinstance(validation, int):
        idxs = np.arange(0, len(ids))
        validation_idxs = np.random.choice(np.setdiff1d(idxs, test_idxs), validation, False)
        pass
    elif isinstance(validation, np.ndarray):
        validation_idxs = validation
    else:
        raise ValueError("Unsupported type for validation: " + str(type(validation)))

    # Split IDs
    test_ids = ids[test_idxs]
    validation_ids = ids[validation_idxs]
    train_ids = np.setdiff1d(ids, np.union1d(test_ids, validation_ids))

    # Create the 3 sets (find their items' indices)
    # np.where returns a tuple, we need only the contents of the 1st dimension, hence [0] at the end
    train_set_idxs = np.where(np.in1d(items, train_ids))[0]
    validation_set_idxs = np.where(np.in1d(items, validation_ids))[0]
    test_set_idxs = np.where(np.in1d(items, test_ids))[0]

    return train_set_idxs, validation_set_idxs, test_set_idxs


def create_lopo_experiment_subsets(wu1experiment: WU1Experiment, test_idx: int, as_dict: bool = False,
                                   dict_names: List[str] = None):
    assert isinstance(wu1experiment, WU1Experiment)
    assert isinstance(test_idx, int)
    assert isinstance(as_dict, bool)
    assert isinstance(dict_names, list) is dict_names is None

    test_idx = np.array((test_idx,))

    if wu1experiment.partition_mode is PartitionMode.LOSO_SIMPLE:
        items = wu1experiment.wu1md.items.user_id
    elif wu1experiment.partition_mode is PartitionMode.LOFTO_SIMPLE:
        items = wu1experiment.wu1md.items.food_type_id
    else:
        raise ValueError("Unsupported value for partition mode")

    train_idxs, validation_idxs, test_idxs = _split_lopo_simple(items, wu1experiment.ids, test_idx,
                                                                wu1experiment.validation_items)

    train_set = Subset(SubsetType.TRAIN, wu1experiment, train_idxs, as_dict, dict_names)
    validation_set = Subset(SubsetType.VALIDATION, wu1experiment, validation_idxs, as_dict, dict_names)
    test_set = Subset(SubsetType.TEST, wu1experiment, test_idxs, as_dict, dict_names)

    return train_set, validation_set, test_set


class Subset(BaseSubset):
    def __init__(self, subset_type: SubsetType, wu1experiment: WU1Experiment, idxs: np.ndarray, as_dict: bool = False,
                 dict_names: List[str] = None):
        super().__init__(subset_type, wu1experiment.batch_size, True)

        assert isinstance(subset_type, SubsetType)
        assert isinstance(wu1experiment, WU1Experiment)
        assert is_numpy_1d_vector(idxs)
        assert isinstance(as_dict, bool)
        assert isinstance(dict_names, list) or dict_names is None

        self.type = subset_type
        self._wu1experiment = wu1experiment
        self._idxs = idxs
        self._as_dict = as_dict
        self._dict_names = dict_names

        self._augment = subset_type is SubsetType.TRAIN
        self._shuffled_idxs = np.copy(idxs)

        self.on_epoch_end()

    def get_len(self):
        return len(self._idxs)

    def shuffle(self):
        np.random.shuffle(self._shuffled_idxs)

    def __len__(self) -> int:
        """
        Override from keras.utils.Sequence
        """
        return int(np.ceil(self.get_len() / self._wu1experiment.batch_size))

    def __getitem__(self, batch_idx: int, only_idxs: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Override from keras.utils.Sequence
        """
        assert isinstance(batch_idx, int)
        assert isinstance(only_idxs, bool)

        # Useful renames
        batch_size = self._wu1experiment.batch_size

        i1 = batch_idx * batch_size
        i2 = min((batch_idx + 1) * batch_size, self.get_len())
        batch_idxs = np.arange(i1, i2)

        if only_idxs:
            return batch_idxs

        x = list()
        for idx in batch_idxs:
            x.append(self.load_x(idx))
        x = np.array(x)

        if self._as_dict:
            n = x.shape[1]
            if self._dict_names is None:
                dict_names = ["input_" + str(i + 1) for i in range(n)]
            else:
                dict_names = self._dict_names
            x_dict = dict()
            for i in range(x.shape[1]):
                x_dict[dict_names[i]] = x[:, i, :].reshape((x.shape[0], *x.shape[2:]))
            x = x_dict

        y = self.get_y(batch_idxs)

        return x, y

    def get_global_idx(self, set_idx=None) -> np.ndarray:
        """
        Returns the global indices of the set samples.

        :param set_idx: The set-indices to return. If set to None, all of the set's indices are returned.
        :return: The global indices.
        """
        if set_idx is None:
            return self._idxs
        else:
            return self._idxs[set_idx]

    def load_x(self, idx: int, is_set_indexed: bool = True):
        # assert isinstance(idx, int)
        assert isinstance(is_set_indexed, bool)

        if is_set_indexed:
            global_idx = self.get_global_idx(idx)
        else:
            global_idx = idx

        x = self._wu1experiment.loader.load(global_idx, self._augment)
        if np.any(np.isnan(x)):
            raise ValueError("Sample " + str(idx) + " contains NaNs.")

        return x

    def get_y(self, idxs: np.ndarray = None, is_set_indexed: bool = True):
        assert isinstance(is_set_indexed, bool)

        # Useful renames
        exp_type = self._wu1experiment.label_mode
        lbl = self._wu1experiment.wu1md.items.labels

        if idxs is None:
            global_idxs = self.get_global_idx()
        else:
            if is_set_indexed:
                global_idxs = self.get_global_idx(idxs)
            else:
                global_idxs = idxs

        if exp_type is LabelMode.CRISP_VS_NONCRISP:
            cols = [0]
        elif exp_type is LabelMode.WET_VS_DRY:
            cols = [1]
        elif exp_type is LabelMode.CHEWY_VS_NON_CHEWY:
            cols = [2]
        elif exp_type is LabelMode.ALL_BINARY:
            cols = [0, 1, 2]
        else:
            raise ValueError('Unsupported experiment_type: ' + str(exp_type))
        cols = np.array(cols)

        return np.array(lbl[global_idxs, :][:, cols], int)
