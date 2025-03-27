"""
This is a private module that organizes various metadata of the dataset. It should not be accessed directly by any
module outside of the ones in the same package.

It requires a mat-file, wu2md.mat, to be present in the main resources of the project.
"""
from copy import deepcopy
from datetime import date
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
from scipy.io import loadmat

import globalconfig as g_conf
from dataset.commons import SampleType
from utilities.listutils import select_by_index
from utilities.matlabutils import find_unique, strcmp
from utilities.numpyutils import is_numpy_matrix, setdiff1d_listint
from utilities.typingutils import is_typed_list


class SessionMetadata:
    def __init__(self, user_name: str, user_id: int, datalogger_id: int, date_time: date, comment: str, ignore: bool):
        assert isinstance(user_name, str)
        assert isinstance(user_id, int)
        assert isinstance(datalogger_id, int)
        assert isinstance(date_time, date)
        assert isinstance(comment, str)
        assert isinstance(ignore, bool)

        self.user_name: str = user_name
        self.user_id: int = user_id
        self.datalogger_id: int = datalogger_id
        self.date_time: date = date_time
        self.comment: str = comment
        self.ignore: bool = ignore


class SensorMetadata:
    def __init__(self, channel_file_names: List[Path], fs_hz: float, duration_sec: float, offset_sec: float):
        assert is_typed_list(channel_file_names, Path)
        assert isinstance(fs_hz, float)
        assert isinstance(duration_sec, float)
        assert isinstance(offset_sec, float)

        self.file_name: List[Path] = channel_file_names
        self.fs_hz: float = fs_hz
        self.duration_sec: float = duration_sec
        self.offset_sec: float = offset_sec


class GroundTruth:
    def __init__(self, chews: np.ndarray = None, bouts: np.ndarray = None, meals: np.ndarray = None):
        assert chews is None or is_numpy_matrix(chews, cols=2)
        assert bouts is None or is_numpy_matrix(bouts, cols=2)
        assert meals is None or is_numpy_matrix(meals, cols=2)

        self.chews: np.ndarray = chews
        self.bouts: np.ndarray = bouts
        self.meals: np.ndarray = meals


class WU2Metadata:
    def __init__(self, sample_type: SampleType, dataset_path: Path = None):
        assert isinstance(sample_type, SampleType)
        assert isinstance(dataset_path, Path) or dataset_path is None

        if sample_type is SampleType.WINDOW:
            self.data_type: SampleType = sample_type
        elif sample_type is SampleType.CHEW:
            self.data_type: SampleType = sample_type
        elif sample_type is SampleType.MEAL:
            self.data_type: SampleType = sample_type
        else:
            raise ValueError("Unsupported data_type: " + str(sample_type))

        if dataset_path is None:
            dataset_path: Path = g_conf.get_wu2_path()

        mat_file: Path = g_conf.get_res_main() / 'wu2' / 'wu2.mat'
        md: np.ndarray = loadmat(str(mat_file))["wu2"][0]

        self.length: int = len(md)

        self.session_md: List[SessionMetadata] = []
        self.audio_md: List[SensorMetadata] = []
        self.ppg_md: List[SensorMetadata] = []
        self.accelerometer_md: List[SensorMetadata] = []
        self.ground_truth: List[GroundTruth] = []

        for i in range(self.length):
            date_time = md['DateTime'][i][0]
            ses_md: SessionMetadata = SessionMetadata(
                str(md['Username'][i][0]),
                int(md['ParticipantID'][i][0, 0]),
                int(md['Datalogger'][i][0, 0]),
                date(date_time[0], date_time[1], date_time[2]),
                str(md['Comment'][i][0]),
                bool(md['ignore'][i][0, 0]))

            aud_md: SensorMetadata = SensorMetadata(
                [dataset_path / str(md['audfile'][i][0])[1:]],
                float(md['audfs'][i][0, 0]),
                float(md['auddur'][i][0, 0]),
                float(md['audoffset'][i][0, 0]))

            ppg_md: SensorMetadata = SensorMetadata(
                [dataset_path / str(md['ppgfile' + str(j + 1)][i][0])[1:] for j in range(3)],
                float(md['ppgfs'][i][0, 0]),
                float(md['ppgdur'][i][0, 0]),
                float(md['ppgoffset'][i][0, 0]))

            acc_md: SensorMetadata = SensorMetadata(
                [dataset_path / str(md['accfile' + j][i][0])[1:] for j in ['x', 'y', 'z']],
                float(md['accfs'][i][0, 0]),
                float(md['accdur'][i][0, 0]),
                float(md['accoffset'][i][0, 0]))

            gt: GroundTruth = GroundTruth(
                chews=np.array(md['groundtruth'][i][0, 0][0]),
                meals=np.array(md['groundtruth'][i][0, 0][1]))

            self.session_md.append(ses_md)
            self.audio_md.append(aud_md)
            self.ppg_md.append(ppg_md)
            self.accelerometer_md.append(acc_md)
            self.ground_truth.append(gt)

        self.partitions: List[str] = self._create_partitions()

        self.partition_idxs: List[int] = [
            find_unique(strcmp(self.partitions, smd.user_name)) for smd in self.session_md]

        self.session_idxs: List[int] = [i for i in range(self.length)]

    def _create_partitions(self) -> List[str]:
        p: List[str] = [ses_md.user_name for ses_md in self.session_md]
        p = list(set(p))
        p.sort()

        return p


def select_by_partition_index(md: WU2Metadata, partition_idxs: List[int]) -> WU2Metadata:
    assert isinstance(md, WU2Metadata)
    assert is_typed_list(partition_idxs, int)

    # Find the sessions that need to be copied
    session_idxs: List[int] = []
    for i, partition_idx in enumerate(md.partition_idxs):
        if partition_idx in partition_idxs:
            session_idxs.append(i)

    select = partial(select_by_index, idxs=session_idxs)

    # Copy metadata to a new object in order to perform the selection
    split_md: WU2Metadata = deepcopy(md)

    # Select
    split_md.partition_idxs = partition_idxs  # Partition indices are directly set to user's choice
    split_md.partitions = select_by_index(split_md.partitions, partition_idxs)

    split_md.session_idxs = select(split_md.session_idxs)
    split_md.session_md = select(split_md.session_md)
    split_md.length = len(split_md.session_md)  # Update length based on sessions' metadata
    split_md.audio_md = select(split_md.audio_md)
    split_md.ppg_md = select(split_md.ppg_md)
    split_md.accelerometer_md = select(split_md.accelerometer_md)
    split_md.ground_truth = select(split_md.ground_truth)

    return split_md


def split(md: WU2Metadata, partition_idxs: List[int]) -> (WU2Metadata, WU2Metadata):
    """
    Split metadata into 2 new objects. The first object contains metadata only the the partitions defined by partition
    indices ``partition_idxs``, and the second object contains metadata for the remaining partitions.
    """
    assert isinstance(md, WU2Metadata)
    assert is_typed_list(partition_idxs, int)

    other_partition_idxs: List[int] = setdiff1d_listint(md.partition_idxs, partition_idxs)
    md1: WU2Metadata = select_by_partition_index(md, partition_idxs)
    md2: WU2Metadata = select_by_partition_index(md, other_partition_idxs)

    return md1, md2


if __name__ == "__main__":
    md: WU2Metadata = WU2Metadata(SampleType.WINDOW, Path("/tmp/wu2"))
    partition_idxs: List[int] = [0, 1, 2, 5, 9, 10]
    md1, md2 = split(md, partition_idxs)
    print("done")
