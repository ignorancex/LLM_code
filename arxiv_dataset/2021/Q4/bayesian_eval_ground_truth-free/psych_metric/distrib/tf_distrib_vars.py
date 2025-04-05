"""Convenience functions for getting Tensorflow variables for distribution
parameters.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib import tfp_mvst


def get_distrib_param_vars(
    distrib_id,
    init_params,
    const_params=None,
    num_class=None,
    random_seed=None,
    alt_distrib=False
):
    """Creates tfp.distribution and tf.Variables for the distribution's
    parameters.

    Parameters
    ----------
    distrib_id : str
        Name of the distribution being created.
    init_params : dict
        Initial parameters of the distribution.
    const_params : list(str), optional
        The names of the parameters to be kept constant, all others are assumed
        to be variables.
    num_class : int, optional
        only used when discrete random variable and number of classes is known.
    random_seed : int
        The random seed to use for initializing the distribution.

    Returns
    -------
    (tfp.distribution.Distribution, dict('param': tf.Variables))
        the distribution and tf.Variables as its parameters.
    """
    distrib_id = distrib_id.lower()

    if (
        distrib_id == 'dirichletmultinomial'
        or distrib_id == 'dirichlet_multinomial'
    ):
        params = get_dirichlet_multinomial_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.DirichletMultinomial(**params),
            params,
        )
    elif distrib_id == 'dirichlet' and not alt_distrib:
        params = get_dirichlet_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.Dirichlet(**params),
            params,
        )
    elif distrib_id == 'dirichlet' and alt_distrib:
        if 'concentration' in init_params:
            precision = np.sum(init_params['concentration'])
            # mean needs to be the discrete probability distrib.
            if np.isclose(precision, 1):
                precision = len(init_params['concentration'])
                mean = init_params['concentration']
            elif precision > 1:
                # Make mean be the normalized mean so all elements within [0,1]
                mean = init_params['concentration'] / precision
            else:
                # If precision is less than 1, that's a problem
                # TODO normalize the mean when it sums to less than 1.
                raise ValueError(' '.join([
                    '`precision`, the sum of the values in `concentration`,',
                    'is less than 1. This means that `mean`\'s values will',
                    'need normalized to fit within the range [0,1] and the',
                    'precision will be set to the sum of the uniform vector',
                    'of ones, which is the number dimensions.',
                ]))

            params = get_dirichlet_alt_param_vars(
                random_seed=random_seed,
                const_params=const_params,
                mean=mean,
                precision=precision,
            )
        else:
            params = get_dirichlet_alt_param_vars(
                random_seed=random_seed,
                const_params=const_params,
                **init_params,
            )
        return (
            tfp.distributions.Dirichlet(params['mean'] * params['precision']),
            params,
        )
    elif distrib_id == 'normal':
        params = get_normal_param_vars(
            random_seed=random_seed,
            const_params=const_params,
            **init_params,
        )
        return (
            tfp.distributions.Normal(**params),
            params,
        )
    elif distrib_id == 'multivariatestudentt':
        params = get_mvst_param_vars(const_params=const_params, **init_params)
        return (
            tfp_mvst.MultivariateStudentT(**params),
            params,
        )
    elif distrib_id == 'multivariatecauchy':
        params = get_mvc_param_vars(const_params=const_params, **init_params)
        return (
            tfp_mvst.MultivariateCauchy(**params),
            params,
        )
    else:
        raise NotImplementedError(
            f'{distrib_id} is not a supported '
            + 'distribution for `get_param_vars()`.'
        )


def tf_var_const(value_id, value, const_params=None, dtype=tf.float32):
    """Creates either a constant or variable of the value with its name."""
    return tf.constant(
        value=value,
        dtype=dtype,
        name=value_id,
    ) if const_params and value_id in const_params else tf.Variable(
        initial_value=value,
        dtype=dtype,
        name=value_id,
    )


def get_dirichlet_param_vars(
    num_classes=None,
    max_concentration=None,
    concentration=None,
    const_params=None,
    random_seed=None,
    name='dirichlet_params',
    dtype=tf.float32,
):
    """Create tf.Variable parameters for the Dirichlet distribution."""
    with tf.name_scope(name):
        if num_classes and max_concentration:
            return {
                'concentration': tf.Variable(
                    initial_value=np.random.uniform(
                        0,
                        max_concentration,
                        num_classes,
                    ),
                    dtype=dtype,
                    name='concentration',
                ),
            }
        elif concentration is not None:
            return {
                'concentration': tf_var_const(
                    'concentration',
                    concentration,
                    const_params,
                    dtype,
                )
            }
        else:
            raise ValueError(' '.join([
                'Must pass either both and `concentration` xor pass',
                '`num_classes`, and `max_concentration`',
            ]))


def get_dirichlet_alt_param_vars(
    num_classes=None,
    mean=None,
    max_precision=None,
    precision=None,
    const_params=None,
    random_seed=None,
    name='dirichlet_mean_precision_params',
    dtype=tf.float32,
):
    """Create tf.Variable parameters for the Dirichlet distribution using mean
    and precision.
    """
    with tf.name_scope(name):
        if num_classes and max_precision:
            return {
                'mean': tf.Variable(
                    initial_value=np.random.uniform(0, 1, num_classes),
                    dtype=dtype,
                    name='mean',
                ),
                'precision': tf.Variable(
                    initial_value=np.random.uniform(
                        0,
                        max_precision,
                        num_classes,
                    ),
                    dtype=dtype,
                    name='precision',
                ),
            }
        elif mean is not None and precision is not None:
            return {
                'mean': tf_var_const('mean', mean, const_params, dtype),
                'precision': tf_var_const(
                    'precision',
                    precision,
                    const_params,
                    dtype,
                )
            }
        else:
            raise ValueError(' '.join([
                'Must pass either both `mean` and `precision` xor pass',
                '`num_classes`, and `max_precision`',
            ]))


def get_dirichlet_multinomial_param_vars(
    num_classes=None,
    max_concentration=None,
    max_total_count=None,
    total_count=None,
    concentration=None,
    const_params=None,
    random_seed=None,
    name='dirichlet_multinomial_params',
    dtype=tf.float32,
):
    """Create tf.Variable parameters for the Dirichlet distribution."""
    with tf.name_scope(name):
        if num_classes and max_concentration and max_total_count:
            return {
                'total_count': tf.Variable(
                    initial_value=np.random.uniform(
                        1,
                        max_total_count,
                        num_classes,
                    ),
                    dtype=dtype,
                    name='total_count',
                ),
                'concentration': tf.Variable(
                    initial_value=np.random.uniform(
                        1,
                        max_concentration,
                        num_classes,
                    ),
                    dtype=dtype,
                    name='concentration',
                ),
            }
        elif total_count is not None and concentration is not None:
            return {
                'total_count': tf_var_const(
                    'total_count',
                    total_count,
                    const_params,
                    dtype,
                ),
                'concentration': tf_var_const(
                    'concentration',
                    concentration,
                    const_params,
                    dtype,
                )
            }
        else:
            raise ValueError('Must pass either both `total_count` and '
                + '`concentration` xor pass `num_classes`, `max_total_count` '
                + 'and `max_concentration`'
            )


def get_normal_param_vars(
    loc,
    scale,
    random_seed=None,
    const_params=None,
    name='normal_params',
    dtype=tf.float32,
):
    """Create tf.Variable parameters for the normal distribution.

    Parameters
    ----------
    loc : float | dict
        either a float as the initial value of the loc, or a dict containing
        the loc and standard deviation of a normal distribution which this
        loc is drawn from randomly.
    scale : float | dict
        either a float as the initial value of the scale, or a dict
        containing the loc and standard deviation of a normal distribution
        which this loc is drawn from randomly.
    """
    with tf.name_scope(name):
        params = {}

        # Get loc
        if isinstance(loc, dict):
            params['loc'] = (tf.constant(
                value=np.random.normal(**loc),
                dtype=dtype,
                name='loc',
            ) if const_params and 'loc' in const_params else tf.Variable(
                initial_value=np.random.normal(**loc),
                dtype=dtype,
                name='loc',
            ))
        elif isinstance(loc, float):
            params['loc'] = tf_var_const('loc', loc, const_params, dtype)
        else:
            raise TypeError(' '.join([
                '`loc` must be either a float xor dict containing a loc and ',
                'scale for sampling from a normal distribution to select the ',
                f'initial values. But recieved type: {type(loc)}',
            ]))

        # Get scale
        if isinstance(scale, dict):
            params['scale'] = (tf.constant(
                value=np.random.normal(**scale),
                dtype=dtype,
                name='scale',
            ) if const_params and 'scale' in const_params else tf.Variable(
                initial_value=np.random.normal(**scale),
                dtype=dtype,
                name='scale',
            ))
        elif isinstance(scale, float):
            params['scale'] = tf_var_const('scale', scale, const_params, dtype)
        else:
            raise TypeError(' '.join([
                '`scale` must be either a float xor dict containing a scale',
                'and scale for sampling from a normal distribution to select',
                f'the initial values. But recieved type: {type(scale)}',
            ]))

        return params


def get_mvc_param_vars(
    loc,
    scale,
    const_params=None,
    name='multivariate_cauchy_params',
    dtype=tf.float32,
):
    """Create tf.Variable / constants for a Multivariate Cauchy"""
    params = {}

    with tf.name_scope(name):
        # Get loc
        if (
            isinstance(loc, float)
            or isinstance(loc, list)
            or isinstance(loc, np.ndarray)
        ):
            params['loc'] = tf_var_const('loc', loc, const_params, dtype)
        else:
            raise TypeError(
                '`loc` must be either a float xor vector of floats '
                + f'initial values. But recieved type: {type(loc)}'
            )

        # TODO Get scale matrix
        if (
            isinstance(scale, float)
            or isinstance(scale, list)
            or isinstance(scale, np.ndarray)
        ):
            params['scale'] = tf_var_const('scale', scale, const_params, dtype)
        else:
            raise TypeError(
                '`scale` must be either a float xor vector of floats '
                + f'initial values. But recieved type: {type(scale)}'
            )

        return params


def get_mvst_param_vars(
    df,
    loc,
    scale,
    const_params=None,
    name='multivariate_student_t_params',
    dtype=tf.float32,
):
    """Create tf.Variable parameters for the Multivariate Student distribution.

    Parameters
    ----------
    df : float
        Positive non-zero float for degrees of freedom.
    loc : np.ndarray(float)
        Vector of floats for the locations or means.
    scale : np.ndarray(floats), optional
        Symmetric positive definite matrix of floats for Sigma matrix.
        Expects Covariance Matrix that is held constant and is used along with
        df to calculate the scale.
    covariance_matrix : np.ndarray(floats), optional
        Symmetric positive definite matrix of floats for covariance matrix.
        Expects Covariance Matrix that is held constant and is used along with
        df to calculate the scale.
    """
    with tf.name_scope(name):
        params = {}

        # Get df
        if isinstance(df, float) and df > 2:
            params['df'] = tf_var_const('df', df, const_params, dtype)
        else:
            raise TypeError(
                '`df` must be either a positve float greater than 2.'
                + f'But recieved type: {type(df)}'
            )

        # TODO separate this function for df only var mvst, from general mvst
        params.update(get_mvc_param_vars(loc, scale, name=name))
    return params


def get_mvst_param_vars_df_only(
    df,
    loc,
    covariance_matrix,
    const_params=None,
    name='multivariate_student_t_params',
    dtype=tf.float32,
):
    raise NotImplementedError('This is for when df is the only param being optimized and scale or sigma matrix could be obtained from the covariance data of the matrix with san equation afterwards.')
    # TODO remove or clean up for use, but then loc and cov_matrix expected
    # to be constant.
    with tf.name_scope(name):
        params = {}

        # Get loc
        if (
            isinstance(loc, float)
            or isinstance(loc, list)
            or isinstance(loc, np.ndarray)
        ):
            params['loc'] = tf_var_const('loc', loc, const_params, dtype)
        else:
            raise TypeError(
                '`loc` must be either a float xor vector of floats '
                + f'initial values. But recieved type: {type(loc)}'
            )

        # TODO Get scale matrix

        # Get Sigma matrix
        # TODO make scale dependent on df and data covariance:
        # sigma = (df-2) / df * cov(data) when df > 2.
        cov = tf.constant(
            value=covariance_matrix,
            dtype=dtype,
            name='covariance_matrix',
        )

        #params['sigma'] = tf.linalg.LinearOperatorLowerTriangular(
        params['sigma'] = (df - 2) / df * cov
    return params
