import numpy as np


def expit(num_input):
    """
    Expit (a.k.a. logistic sigmoid).
    The expit function (sigmoid function)
    is defined as expit(x) = 1/(1+exp(-x)).
    It is the inverse of the logit function.
    Parameters
    nparray
        The nparray to apply expit to element-wise.
    Returns
        nparray
            An nparray of the same shape as xnum_input.
    """
    return 1/(1+np.exp(-num_input))


def finding_best(
    infected_array,
    data_to_fit,
    num_of_best=4,
    inc=10
):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    best = np.zeros(num_of_best, dtype=int)
    i = 0
    N = infected_array.shape[0]
    for k in range(0, inc, num_of_best-1):
        score_old = 1e+10
        for j in range(N):
            score = np.abs(infected_array[j] - data_to_fit[k]).mean()
            if score < score_old:
                best[i] = j
                score_old = score
        i += 1
    return best
