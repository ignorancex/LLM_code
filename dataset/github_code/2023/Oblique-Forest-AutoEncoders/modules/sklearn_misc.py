################################################################################################### 
#                               Load modules
###################################################################################################
import numbers
import numpy as np
from sklearn.utils.validation import check_array, check_non_negative





################################################################################################### 
#                               Helper functions
###################################################################################################
def _num_samples(x):
    message = "Expected sequence or array-like, got %s" % type(x)

    if hasattr(x, "fit") and callable(x.fit):
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )

        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error






def _check_sample_weight(sample_weight, X, dtype = None, copy = False, only_non_negative = False):
    n_samples = _num_samples(X)

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype = dtype)
    
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(n_samples, sample_weight, dtype = dtype)

    else:
        if dtype is None:
            dtype = [np.float64, np.float32]
        sample_weight = check_array(
            sample_weight,
            accept_sparse = False,
            ensure_2d = False,
            dtype = dtype,
            order = "C",
            copy = copy,
            input_name = "sample_weight",
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples, ):
            raise ValueError(
                "sample_weight.shape == {}, expected {}!".format(
                    sample_weight.shape, (n_samples,)
                )
            )

    if only_non_negative:
        check_non_negative(sample_weight, "`sample_weight`")

    return sample_weight





