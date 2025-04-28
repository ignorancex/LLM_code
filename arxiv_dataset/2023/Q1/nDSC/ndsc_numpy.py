import numpy as np


def check_inputs(y_pred: np.ndarray, y: np.ndarray) -> None:
    """
    Check that input arrays are numpy.ndarray-s, binary masks, and of the similar shape.
    :param y_pred: predicted binary segmentation mask
    :param y: ground truth segmentation mask
    """
    def check_binary_mask(mask):
        unique = np.unique(mask)
        if np.sum(np.isin(unique, test_elements=[0.0, 1.0], invert=True)) != 0.0:
            return False
        return True

    instance = bool(isinstance(y_pred, np.ndarray) * isinstance(y, np.ndarray))
    binary_mask = bool(check_binary_mask(y_pred) * check_binary_mask(y))
    dimensionality = y_pred.shape == y.shape

    if not instance * binary_mask * dimensionality:
        raise ValueError(f"Inconsistent input to metric function. Failed in instance: {instance},"
                         f"binary mask: {binary_mask}, dimensionality: {dimensionality}.")


def ndsc_metric(y_pred: np.ndarray, y: np.ndarray, effective_load: float = 0.001, check: bool = False) -> float:
    """
    Compute an nDSC for numpy arrays.
    :param y_pred: predicted binary segmentation mask for a single class, no batch or class dimensions.
    :param y: ground truth binary segmentation mask of the same shape as `y_pred`
    :param effective_load: effective load
    :param check: if True, `y` and `y_pred` inputs will be checked for type, shape, binarisation.
    :return: nDSC value
    """
    if check:
        check_inputs(y_pred, y)

    if np.sum(y_pred) + np.sum(y) > 0:
        scaling_factor = 1.0 if np.sum(y) == 0 else (1 - effective_load) * np.sum(y) / (effective_load * (len(y.flatten()) - np.sum(y)))
        tp = np.sum(y_pred[y == 1])
        fp = np.sum(y_pred[y == 0])
        fn = np.sum(y[y_pred == 0])
        fp_scaled = scaling_factor * fp
        return 2 * tp / (fp_scaled + 2 * tp + fn)
    return 1.0
