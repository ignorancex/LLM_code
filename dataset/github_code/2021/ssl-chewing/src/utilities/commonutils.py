import numpy as np


def sec2samples(sec: float, fs_hz: float) -> int:
    return int(np.floor(sec * fs_hz))
