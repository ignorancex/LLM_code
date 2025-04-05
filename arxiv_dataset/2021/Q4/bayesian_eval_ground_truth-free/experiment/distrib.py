"""Helper functions for getting parameters for distributions."""
from numbers import Number

import numpy as np


def get_dirichlet_params(
    concentration=None,
    num_classes=None
):
    """Either packages the parameters into a dict or creates the parameters
    first from some specification on how to create them (ie. drawing from
    normal distribution).

    Parameters
    ----------
    concentration : float | list(float), optional
        Either a postive float as the initial value of the scale, or a dict
        containing the loc and standard deviation of a normal distribution
        which this loc is drawn from randomly. If
    num_classes : int
        The number of classes of the source Dirichlet-Multinomial. Only
        required when the given a single float for `concentration`.
        `concentration` is then turned into a list of length `num_classes`
        where ever element is the single float given by `concentration`.

    Returns
    -------
    dict
        Dictionary of the values of the parameters.
    """
    params = {}

    # concentration
    if isinstance(concentration, Number) and  isinstance(num_classes, Number):
        params['concentration'] = [concentration] * num_classes
    elif isinstance(concentration, list) or isinstance(concentration, np.ndarray):
        params['concentration'] = concentration
    elif isinstance(concentration, dict) and isinstance(num_classes, Number):
        # TODO Create concentration as a discrete, ordinal normal distribution?
        # or perhaps sample the concentrations from that normal... just as a
        # form of controled randomization.
        raise NotImplementedError
    elif isinstance(num_classes, Number):
        params['concentration'] = np.random.uniform(0, 100, num_classes)
    else:
        raise TypeError(
            'Wrong type for `concentration` and `num_classes` not given. '
            + f'Type recieved: {type(concentration)}'
        )

    return params


def get_dirichlet_multinomial_params(
    total_count=None,
    concentration=None,
    num_classes=None
):
    """Either packages the parameters into a dict or creates the parameters
    first from some specification on how to create them (ie. drawing from
    normal distribution).

    Parameters
    ----------
    total_count : float | dict, optional
        Either a float as the initial value of the loc, or a dict containing
        the loc and standard deviation of a normal distribution which this
        loc is drawn from randomly.
    concentration : float | list(float), optional
        Either a postive float as the initial value of the scale, or a dict
        containing the loc and standard deviation of a normal distribution
        which this loc is drawn from randomly. If
    num_classes : int
        The number of classes of the source Dirichlet-Multinomial. Only
        required when the given a single float for `concentration`.
        `concentration` is then turned into a list of length `num_classes`
        where ever element is the single float given by `concentration`.

    Returns
    -------
    dict
        Dictionary of the values of the parameters.
    """
    params = {}

    # total count
    if isinstance(total_count, dict):
        params['total_count'] = np.abs(np.random.normal(**total_count))
    elif isinstance(total_count, Number):
        params['total_count'] = total_count
    else:
        params['total_count'] = np.random.uniform(0, 100)

    # concentration
    if isinstance(concentration, Number) and  isinstance(num_classes, Number):
        params['concentration'] = [concentration] * num_classes
    elif isinstance(concentration, list) or isinstance(concentration, np.ndarray):
        params['concentration'] = concentration
    elif isinstance(concentration, dict) and isinstance(num_classes, Number):
        # TODO Create concentration as a discrete, ordinal normal distribution?
        # or perhaps sample the concentrations from that normal... just as a
        # form of controled randomization.
        raise NotImplementedError
    elif isinstance(num_classes, Number):
        params['concentration'] = np.random.uniform(0, 100, num_classes)
    else:
        raise TypeError(
            'Wrong type for `concentration` and `num_classes` not given. '
            + f'Type recieved: {type(concentration)}'
        )

    return params


def get_multivariate_normal_full_cov_params(
    loc=None,
    covariance_matrix=None,
    sample_dim=None,
):
    """Either packages the parameters into a dict or creates the parameters
    first from some specification on how to create them (ie. drawing from
    normal distribution).

    Parameters
    ----------
    loc : float | list(float) | dict, optional
    covariance_matrix : float | np.ndarray(float) | dict, optional
    sample_dim : int, optional
        Number of the dimensions of the individual sample to be drawn from this
        distribution (ie. same as number of discrete bins / classes). Necessary
        if `loc` or `covariance_matrix` are not dicts with `size` included as a
        key.

    Returns
    -------
    dict
        Dictionary of the values of the parameters.
    """
    params = {}

    # Check if sample_dim is of correct type or able to be specified implicitly.
    if not isinstance(sample_dim, Number):
        if isinstance(loc, np.ndarray):
            sample_dim = len(loc)
        elif isinstance(loc, dict) and 'size' in loc:
            sample_dim = loc['size']
        elif isinstance(covariance_matrix, np.ndarray):
            sample_dim = len(covariance_matrix)
        else:
            raise TypeError(
                '`sample_dim` must be given and of type `int` when neither '
                + '`loc` or `covariance` are of type `np.ndarray`. '
                + f'Type of `sample_dim` recieved: {type(sample_dim)}'
            )

    # loc
    if isinstance(loc, dict):
        # Draw param value from normal distribution
        if 'size' not in loc:
            params['loc'] = np.random.normal(size=sample_dim, **loc)
        else:
            params['loc'] = np.random.normal(**loc)
    elif isinstance(loc, Number):
        # Set param to a repeating list of size sample_dim with the same Number
        params['loc'] = [loc] * sample_dim
    else:
        # Defaults to zeros.
        params['loc'] = np.zeros(sample_dim)

    # covariance matrix
    if isinstance(covariance_matrix, np.ndarray):
        # Simply store the covariance matrix as is.
        params['covariance_matrix'] = covariance_matrix
    elif isinstance(covariance_matrix, Number):
        # Set the diagonal covariances to the same value, the rest zeros
        params['covariance_matrix'] = covariance_matrix * np.eye(sample_dim)
    elif isinstance(covariance_matrix, dict):
        # Create symmetric matrix from uniform distribution.
        params['covariance_matrix'] = np.random.uniform(
            size=[sample_dim, sample_dim],
            **covariance_matrix,
        )

        # Ensure symmetric about diagonal
        params['covariance_matrix'] = (params['covariance_matrix'] + params['covariance_matrix'].T) / 2

        # Ensure positive definite matrix
        params['covariance_matrix'] += sample_dim * np.eye(sample_dim)
    elif covariance_matrix is None:
        # Create symmetric matrix from hardcoded uniform distribution.
        params['covariance_matrix'] = np.random.uniform(
            -0.01,
            0.01,
            [sample_dim, sample_dim],
        )

        # Ensure a positive diagonal
        for i in range(sample_dim):
            params['covariance_matrix'][i, i] = np.absolute(
                params['covariance_matrix'][i, i],
            )

        # Ensure symmetric about diagonal
        params['covariance_matrix'] = (params['covariance_matrix'] + params['covariance_matrix'].T) / 2

        # Ensure positive definite matrix
        params['covariance_matrix'] += 0.01 * np.eye(sample_dim)
    else:
        raise TypeError(
            'Wrong type for `covariance_matrix`. '
            + f'Type recieved: {type(covariance_matrix)}'
        )

    return params


def get_normal_params(loc=None, scale=None):
    """Either packages the parameters into a dict or creates the parameters
    first from some specification on how to create them (ie. drawing from
    normal distribution).

    Parameters
    ----------
    loc : float | dict | None, optional
    scale : float | dict | None, optional

    Returns
    -------
    dict
        Dictionary of the values of the parameters.
    """
    params = {}

    if isinstance(loc, dict):
        params['loc'] = np.random.normal(**loc)
    elif isinstance(loc, Number):
        params['loc'] = loc
    else:
        params['loc'] = np.random.uniform(-100, 100)

    if isinstance(scale, dict):
        params['scale'] = np.abs(np.random.normal(**scale))
    elif isinstance(scale, Number):
        params['scale'] = scale
    else:
        params['scale'] = np.random.uniform(0, 5)

    return params
