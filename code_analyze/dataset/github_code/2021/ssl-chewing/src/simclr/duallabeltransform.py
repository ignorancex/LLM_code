import numpy as np

from dataset.labeltransform import BaseLabelTransform
from utilities.numpyutils import is_numpy_2d_vector


class DualLabelTransform(BaseLabelTransform):
    def transform_batch(self, x: np.ndarray) -> np.ndarray:
        assert is_numpy_2d_vector(x)

        return np.vstack((x, x))
