"""
Utilities related to the functionality of MATLAB's buffer function.
"""
from utilities.typingutils import is_int


def get_noof_windows(n: int, wsize: int, wstep: int) -> int:
    assert isinstance(n, int)
    assert isinstance(wsize, int)
    assert isinstance(wstep, int)

    if n < wsize:
        return 0

    return int((n - wsize) / wstep) + 1


def get_window_timestamp(wsize_sec: float, wstep_sec: float, i: int) -> [float, float]:
    assert isinstance(wsize_sec, float)
    assert isinstance(wstep_sec, float)
    assert is_int(i)

    t_start: float = wstep_sec * i
    t_stop: float = t_start + wsize_sec

    return t_start, t_stop
