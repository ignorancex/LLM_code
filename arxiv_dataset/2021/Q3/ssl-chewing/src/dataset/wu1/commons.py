from enum import Enum
from typing import List


class LabelMode(Enum):
    CRISP_VS_NONCRISP = 0  # '1=crisp, 0=non-crisp'
    WET_VS_DRY = 1  # '1=wet, 0=dry'
    CHEWY_VS_NON_CHEWY = 2  # '1=chewy, 0=non-chewy'
    ALL_BINARY = 3


def label_mode_length(label_mode: LabelMode) -> int:
    assert isinstance(label_mode, LabelMode)

    if label_mode is LabelMode.CRISP_VS_NONCRISP:
        return 1
    elif label_mode is LabelMode.WET_VS_DRY:
        return 1
    elif label_mode is LabelMode.CHEWY_VS_NON_CHEWY:
        return 1
    elif label_mode is LabelMode.ALL_BINARY:
        return 3
    else:
        raise ValueError('Unsupported label_mode: ' + str(label_mode))


def label_mode_str(label_mode: LabelMode) -> List[str]:
    assert isinstance(label_mode, LabelMode)

    if label_mode is LabelMode.CRISP_VS_NONCRISP:
        return ['crisp']
    elif label_mode is LabelMode.WET_VS_DRY:
        return ['wet']
    elif label_mode is LabelMode.CHEWY_VS_NON_CHEWY:
        return ['chewy']
    elif label_mode is LabelMode.ALL_BINARY:
        return ['crisp', 'wet', 'chewy']
    else:
        raise ValueError('Unsupported label_mode: ' + str(label_mode))
