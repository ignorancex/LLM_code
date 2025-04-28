

def momentum(gradient, vt_val, mu, learning_rate):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    vt_val = mu * vt_val + (1 - mu) * gradient
    difference = learning_rate*vt_val
    return difference, vt_val
