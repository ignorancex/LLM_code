import torch
import numpy as np
from scipy.interpolate import CubicSpline

def log_predictor(a):
    x = []
    y = []
    if type(a) == torch.Tensor:
        a = a.cpu()
    n = len(a)
    for i in range(n):
        if a[i] > 0:
            x.append(i)
            y.append(-np.log(a[i]))
            
    x.append(x[0] + 512)
    y.append(y[0])
    func = CubicSpline(x,y, bc_type="periodic")
    desiredXs = np.arange(n)
    result = func(desiredXs)
    return result
