"""
This is a private module that organizes various metadata of the dataset. It should not be accessed directly by any
module outside of the ones in the same package.
"""
import csv
import warnings
from types import SimpleNamespace

import numpy as np

import globalconfig as g_cfg
from dataset.commons import SampleType, PartitionMode
from dataset.wu1.utils import get_database_csv_filename, fs2str
from utilities.matlabutils import integer2path

_NOOF_LABELS: int = 4


class WU1Metadata:
    def __init__(self, sample_type: SampleType, fs_hz: int):
        assert isinstance(sample_type, SampleType)
        assert isinstance(fs_hz, int)

        self.data_type = sample_type
        self.fs_hz = fs_hz

        if sample_type is SampleType.CHEW:
            self.items = _load_metadata_chews(fs_hz)
        elif sample_type is SampleType.BOUT:
            self.items = _load_metadata_bouts(fs_hz)
        else:
            raise ValueError("Unsupported data_type: " + str(sample_type))

        self.length = len(self.items.user_id)
        self.user_ids = np.unique(self.items.user_id)
        self.food_type_ids = np.unique(self.items.food_type)

    def get_partition_ids(self, partition_mode: PartitionMode):
        assert isinstance(partition_mode, PartitionMode)

        if partition_mode is PartitionMode.LOSO_SIMPLE:
            return self.user_ids
        elif partition_mode is PartitionMode.LOFTO_SIMPLE:
            return self.food_type_ids
        else:
            raise ValueError('Unsupported partition_mode: ' + str(partition_mode))


def _load_metadata_bouts(fs_hz: int):
    file_name = get_database_csv_filename(SampleType.BOUT, fs_hz)

    x = list()
    with file_name.open() as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            x.append(row)

    n = len(x)
    b = np.zeros(n, bool)
    bouts = SimpleNamespace()
    bouts.user_id = np.zeros(n)
    bouts.event_id = np.zeros(n)
    bouts.food_type = list()
    bouts.labels = np.zeros((n, _NOOF_LABELS))
    bouts.t = np.zeros(n)
    bouts.x_path = list()

    for i in range(n):
        bouts.user_id[i] = x[i]['user_id']
        bouts.event_id[i] = x[i]['event_id']
        bouts.food_type.append(x[i]['food_type'])
        b[i], bouts.labels[i, :] = _get_food_type_attributes(bouts.food_type[i])
        bouts.t[i] = x[i]['t']
        bouts.x_path.append(x[i]['x_path'])

    bouts.user_id = bouts.user_id[b]
    bouts.event_id = bouts.event_id[b]
    bouts.food_type = np.array(bouts.food_type)
    bouts.food_type = bouts.food_type[b]
    bouts.labels = bouts.labels[b, :]
    bouts.t = bouts.t[b]
    bouts.x_path = np.array(bouts.x_path)
    bouts.x_path = bouts.x_path[b]

    return bouts


def _load_metadata_chews(fs_hz: int):
    file_name = get_database_csv_filename(SampleType.CHEW, fs_hz)

    x = list()
    with file_name.open() as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            x.append(row)

    n = len(x)
    b = np.zeros(n, bool)
    chews = SimpleNamespace()
    chews.user_id = np.zeros(n)
    chews.event_id = np.zeros(n)
    chews.bout_id = np.zeros(n)
    chews.food_type = list()
    chews.labels = np.zeros((n, _NOOF_LABELS))
    chews.t = np.zeros(n)
    chews.x_path = list()

    for i in range(n):
        chews.user_id[i] = x[i]['user_id']
        chews.event_id[i] = x[i]['event_id']
        chews.bout_id[i] = x[i]['bout_id']
        chews.food_type.append(x[i]['food_type'])
        b[i], chews.labels[i, :] = _get_food_type_attributes(chews.food_type[i])
        chews.t[i] = x[i]['t']
        chews.x_path.append(x[i]['x_path'])

    chews.user_id = chews.user_id[b]
    chews.event_id = chews.event_id[b]
    chews.bout_id = chews.bout_id[b]
    chews.food_type = np.array(chews.food_type)
    chews.food_type = chews.food_type[b]
    chews.labels = chews.labels[b, :]
    chews.t = chews.t[b]
    chews.x_path = np.array(chews.x_path)
    chews.x_path = chews.x_path[b]

    return chews


def _get_food_type_attributes(food_type: str):
    """
    Get texture attributes for a food type.

    The attributes that are supported currently are:
    0: crispy vs. non-crispy
    1: wet vs. dry
    2: chewy vs. non-chewy
    3: ??? vs. ???

    :param food_type: A string that contains the food type
    :return: A vector of the texture attributes
    """
    assert isinstance(food_type, str)

    if food_type in ('potato_chips', 'cookie'):
        return 1, np.array((1, 0, 0, 0))
    elif food_type in ('apple', 'lettuce'):
        return 1, np.array((1, 1, 0, 0))
    elif food_type in ('bread', 'cake'):
        return 1, np.array((0, 0, 0, 0))
    elif food_type in ('banana', 'strawberry'):
        return 1, np.array((0, 1, 0, 0))
    elif food_type in ('candy_bar', 'toffee', 'toffee_wrapper'):
        return 1, np.array((0, 0, 1, 0))
    elif food_type in ('pureed_apple', 'vanilla_custard', 'yoghurt'):
        return 1, np.array((0, 1, 0, 1))
    else:
        warnings.warn('Unknown food type: ' + food_type + ', skipping...')
        return 0, np.array((0, 0, 0, 0))


def _load_metadata_deprecated(sample_type: SampleType, fs_hz: int):
    """
    Loads the metadata for the dataset.

    :param sample_type: Used to select between chews and bouts.
    :param fs_hz: The sampling frequency of audio to use.
    :return: The dataset metadata.
    """
    assert isinstance(sample_type, SampleType)
    assert isinstance(fs_hz, int)

    metadata = SimpleNamespace()
    metadata.fs_Hz = fs_hz
    metadata.data_type = sample_type
    if sample_type == SampleType.CHEW:
        metadata.x = _load_metadata_chews(fs_hz)
    elif sample_type == SampleType.BOUT:
        metadata.x = _load_metadata_bouts(fs_hz)
    metadata.user_ids = np.array(np.unique(metadata.x.user_id))
    metadata.food_types = np.array(np.unique(metadata.x.food_type))
    metadata.len = len(metadata.x.x_path)

    return metadata


def get_x_file_path(wu1metadata: WU1Metadata, idx: int):
    """
    Gets the full path to the mat-file of a chew or bout.

    :param wu1metadata: Metadata of the WU1 dataset
    :param idx: The index of the item (chew/bout).
    :return: The item's (chew/bout) audio.
    """
    assert isinstance(wu1metadata, WU1Metadata)
    assert isinstance(idx, int)

    if wu1metadata.data_type is SampleType.CHEW:
        s = 'chews'
    elif wu1metadata.data_type is SampleType.BOUT:
        s = 'bouts'
    else:
        raise ValueError('Unsupported load_type: ' + str(wu1metadata.data_type))

    folder_name = s + '_exported_' + fs2str(wu1metadata.fs_hz)
    file_name = integer2path(idx, wu1metadata.length) + '.mat'

    return g_cfg.get_generated_path() / s / folder_name / 'x' / file_name
