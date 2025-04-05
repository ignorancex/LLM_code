"""General measure funcitons for ease of executing an arbitrary measure on data
that may hav have multiple values per sample, such as having an empirical
sampling of a conditional distribution in order to obtain a distribution of the
measure.

Notes
-----
All this assumes discrete distributions.
"""
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder

#from psych_metric.distrib.conditional.G

# TODO consider adding axes for: conditionals_axis=None, target_axis=0
def measure(measure_func, targets, preds):
    """Obtain measure per conditional sample

    Parameters
    ----------
    measure_func : callable
        Function that performs some measure on the target and pred expecting
        them in order of targets and preds (ie. measure_func(targets, preds).
    targets : np.ndarray
        targets is a numpy array with shape of [number of targets, classes],
        where classes is the number of classes or discrete dimensions of a
        sample.
    preds : np.ndarray
        preds is a numpy array with shape of [number of targets, classes],
        where classes is the number of classes or discrete dimensions of a
        sample, XOR preds has the shape of [number of targets, conditionals,
        classes] where conditionals is the number of multiple pred samples per
        target sample. This latter shape is
    """
    if len(preds.shape) == 1 or (len(preds.shape) == 2 and preds.shape[1] == 1):
        # Assumed structure is then shape of 2, and non-target axis is the
        # n-dim discrete sample
        return measure_func(targets, preds)

    if len(preds.shape) != 3 or len(targets.shape) > 2:
        raise ValueError(' '.join([
            'preds and targets are expected to have 3 and 2 dimensions',
            'respectively.',
            f'Given preds has shape of {preds.shape}.',
            f'Given targets has shape of {targets.shape}.',
        ]))

    # NOTE assumes preds of shape [targets, conditionals, classes]

    conditionals_measurements = []
    for c_idx in range(preds.shape[1]):
        conditionals_measurements.append(
            measure_func(targets, preds[:, c_idx, :]),
        )

    return np.array(conditionals_measurements)


def credible_interval(vector, sample_density, version='highest_density'):
    """Wraps the density based credible interval functions."""
    if version == 'highest_density':
        return highest_density_credible_interval(vector, sample_density)

    if version == 'left':
        return one_tailed_credible_interval(vector, sample_density, True)

    if version == 'right':
        return one_tailed_credible_interval(vector, sample_density, False)

    raise ValueError(' '.join([
        'Expected `version` to be one of "highest density", "left", or',
        f'"right"; not {version}',
    ]))


def highest_density_credible_interval(vector, sample_density):
    """Finds the highest density credible interval (aka highest posterior
    density interval) for a vector of values. This interval is found by sorting
    the vector vlaues, then iterating through the data checking for which
    interval of some % size (ie. 95% credible interval) has the smallest
    difference of high quantile - low quantile. Given a static interval size,
    the smallest distance in values of the given vector is the highest density
    credible interval of that size.

    The granularity used to check the intervals is always by 1 sample.

    Parameters
    ----------
    vector : np.ndarray
        The vector of values whose credible interval is to be calculated.
    sample_density : float
        Float from (0,1) that represents the amount of data points that the
        credible interval is to contain (ie. .95 results in a credible interval
        that contains 95% of the sorted data with the highest density).

    Returns
    -------
    (low_quantile, high_quantile) : tuple
        A tuple of the lower and upper quantiles that bound the highest density
        crediblity interval of the given size.
    """
    if sample_density <= 0 or sample_density >= 1:
        raise ValueError(
            'The Credible interval size must be within the range (0,1)',
        )

    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)

    if len(vector.shape) > 1:
        # If given multiple dim array, flatten into 1d
        vector = np.ravel(vector)

    # Sort vector
    sorted_vec = np.sort(vector)
    vec_size = len(sorted_vec)
    start_idx = int(np.ceil(vec_size * sample_density))

    # Get the paired low and high quantiles
    high_quantiles = sorted_vec[start_idx:]
    low_quantiles = sorted_vec[:vec_size - start_idx]

    # Iterate through the vector with static interval size, check differences
    # minimum distance is the highest density credible interval
    idx = np.argmin(high_quantiles - low_quantiles)

    return low_quantiles[idx], high_quantiles[idx]


def one_tailed_credible_interval(vector, sample_density, left_tail=True):
    """Finds the one-tailed credible interval (left or right) for a vector of
    values. This interval is found by sorting the vector vlaues, then using the
    interval size to index the data and obtain the one tailed credible interval
    of the desired % size (ie. 95% credible interval).

    The granularity used to check the intervals is always by 1 sample.

    Parameters
    ----------
    vector : np.ndarray
        The vector of values whose credible interval is to be calculated.
    sample_density : float
        Float from (0,1) that represents the amount of data points that the
        credible interval is to contain (ie. .95 results in a credible interval
        that contains 95% of the sorted data with the highest density).
    left : bool
        Defaults to performing a left tailed test. Set to False for a right
        tailed test.

    Returns
    -------
    int | float
        The location of the credible interval boundary.
    """
    if sample_density <= 0 or sample_density >= 1:
        raise ValueError(
            'The Credible interval size must be within the range (0,1)',
        )

    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)

    if len(vector.shape) > 1:
        # If given multiple dim array, flatten into 1d
        vector = np.ravel(vector)

    # Sort vector
    sorted_vec = np.sort(vector)
    vec_size = len(sorted_vec)
    start_idx = int(np.ceil(vec_size * sample_density))

    if left_tail:
        return sorted_vec[vec_size - start_idx]
    # Right tailed credible interval
    return sorted_vec[start_idx]


def kldiv_probs(p, q, axis=1):
    """Given two discrete probability vectors, calculate Kullback-Lebler
    Divergence.
    """
    # TODO add and subtract a very small value from the probs when zeros are
    # encountered.
    return (p * np.log2(p / q)).sum(axis=axis)


def entropy_probs(p, axis=1):
    """Given a discrete probability vector, calculate entropy."""
    return -(p * np.log2(p)).sum(axis=1)


def discretize_multidim_continuous(x, bins, copy=True):
    """Returns discretized multidimensional continuous random variable."""
    if copy:
        x = x.copy()

    # Bin each dimension (column) of the RV's by the dim's respective quantiles
    for i in range(x.shape[1]):
        x[:, i] = np.digitize(x[:, i], np.quantile(x[:, i], bins))

    # Treat each row as a symbol.
    le = LabelEncoder()
    return le.fit_transform([f'{row}' for row in x])


def binned_mutual_information(
    x,
    y,
    num_bins=10,
    cpus=1,
    simplex=True,
    normalized=False,
):
    """Normalized Mutual information for multi dimensional continuous random
    variables. Uses quantiles to bin to result in essentially uniform marginal
    distributions, and thus able to calculate Mutual Information as the Copula
    Entropy.

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    method : str
        If 'bin' then discretizes the
    bins : int
        number of bins to discretize each element by.
    quantile_binning : bool
        The binning is done based on the quantiles if True, otherwise binning
        is evenly spaced bins within the interval.
    """
    if x.shape != y.shape:
        raise ValueError(
            f'x and y must have the same shape: {x.shape} and {y.shape}'
        )

    # Create the 'bins', as in the quantiles to use to create the bins
    bins = np.linspace(0, 1, num_bins + 1)

    if cpus <= 1:
        x_discrete = discretize_multidim_continuous(x, bins)
        y_discrete = discretize_multidim_continuous(y, bins)
    else:
        with Pool(processes=2) as pool:
            x_discrete, y_discrete = pool.starmap(
                discretize_multidim_continuous,
                [(x, bins, False), (y, bins, False)],
            )

    if normalized:
        return normalized_mutual_info_score(y_discrete, x_discrete)
    return mutual_info_score(y_discrete, x_discrete)

    # TODO possibly compute yourself using below Copula entropy technique

    print('finished discretization of marginals')

    unique, counts = np.unique(
        [f'{x_discrete[i]} {y_discrete[i]}' for i in range(len(x_discrete))],
        return_counts=True,
    )

    print(unique)
    print(unique.shape)
    print(counts)
    print(counts.shape)


    # MI based on copula entropy
    probs = counts / len(x_discrete)
    print(probs)
    mutual_information_est = (probs * np.log2(probs)).sum()
    return mutual_information_est
    # how to normalize? need to estimate entropy as well?
