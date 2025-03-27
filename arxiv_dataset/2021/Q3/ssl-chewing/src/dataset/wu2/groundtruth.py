from typing import List, Union

import numpy as np

from dataset.commons import GroundTruthType
from dataset.template.basegroundtruth import BaseGroundTruth
from dataset.wu2.wu2metadata import WU2Metadata, SensorMetadata
from utilities.matlabutils import find_unique
from utilities.typingutils import is_int


class AudioGroundTruth(BaseGroundTruth):
    def __init__(self, wu2md: WU2Metadata, gt_type: GroundTruthType):
        assert isinstance(wu2md, WU2Metadata)
        assert isinstance(gt_type, GroundTruthType)

        self._md: WU2Metadata = wu2md
        self._gt_type: GroundTruthType = gt_type

        if gt_type is GroundTruthType.CHEW:
            self._gt: List[np.ndarray] = [gt.chews for gt in wu2md.ground_truth]
        elif gt_type is GroundTruthType.BOUT:
            raise ValueError("Unsupported ground-truth type: " + str(gt_type))
        elif gt_type is GroundTruthType.MEAL:
            self._gt: List[np.ndarray] = [gt.meals for gt in wu2md.ground_truth]
        else:
            raise ValueError("Unsupported ground-truth type: " + str(gt_type))

    def get_label(self, session_idx: int, t1: float, t2: float) -> Union[bool, int, List[int], np.ndarray]:
        # Set is_inside as the default way to get ground-truth

        return self.is_inside(session_idx, t1, t2)

    def is_inside(self, session_idx: int, t1: float, t2: float) -> bool:
        assert is_int(session_idx)
        assert isinstance(t1, float)
        assert isinstance(t2, float)

        assert session_idx in self._md.session_idxs

        idx: int = find_unique((np.array(self._md.session_idxs) == session_idx).tolist())

        md: SensorMetadata = self._md.audio_md[idx]  # Ability to change sensor stream here
        assert md.offset_sec <= t1 <= md.offset_sec + md.duration_sec
        assert md.offset_sec <= t1 <= md.offset_sec + md.duration_sec
        t: float = (t1 + t2) / 2 + md.offset_sec

        gt: np.ndarray = self._gt[idx]
        b1: np.ndarray = gt[:, 0] <= t
        b2: np.ndarray = t <= gt[:, 1]
        test: int = sum(np.logical_and(b1, b2))

        if test == 0:
            return False
        elif test == 1:
            return True
        else:
            raise ValueError(
                "Bad ground truth. Moment t=" + str(t) + " is inside " + str(test) + " ground-truth intervals")
