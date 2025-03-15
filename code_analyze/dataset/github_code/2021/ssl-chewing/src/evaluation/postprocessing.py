import numpy as np

from utilities.numpyutils import ensure_numpy_1d_vector
from utilities.postprocessing import get_meals, get_bouts


def chewing2meals(chewing: np.ndarray, fs_hz: float):
    chewing = ensure_numpy_1d_vector(chewing)
    assert isinstance(fs_hz, float)

    meals_sec = get_meals(get_bouts(chewing, fs_hz))
    y_pred = np.zeros(chewing.shape)
    t_sec = np.arange(y_pred.size) / fs_hz
    for meal in meals_sec:
        b = np.logical_and(meal[0] <= t_sec, t_sec <= meal[1])
        y_pred[b] = 1

    return y_pred
