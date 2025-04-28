import numpy as np


def mean_squared(infected, data_fit, params=None):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    return np.power(infected - data_fit, 2).mean()


def regular_nearest(infected, data_fit, c_vals, beta_val, W, alpha):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    normsq = np.power(beta_val*(W@c_vals - c_vals), 2)
    return np.power(infected - data_fit, 2).mean() + alpha*(normsq.sum())


def regular_l2(infected, data_fit, beta0, alpha=1e-02):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    norm1sq = np.power(beta0, 2).sum()
    return np.power(infected - data_fit, 2).mean() + alpha*(norm1sq)


def regular_l1(infected, data_fit, beta0, alpha=1e-02):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    norm1 = np.abs(beta0).sum()
    return np.power(infected - data_fit, 2).mean() + alpha*(norm1)
