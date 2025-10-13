import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

def num2str_decimal(x):
    s = str(x)
    c = ''
    for i in range(len(s)):
        if s[i] == '0':
            c = c + 'z'
        elif s[i] == '.':
            c = c + 'p'
        elif s[i] == '-':
            c = c + 'n'
        else:
            c = c + s[i]

    return c

def phi(n, x):
    return np.sin(n * np.pi * x)

def dphi(n, x):
    return n * np.pi * np.cos(n * np.pi * x)

def coeff_2_vec(coeff, x):
    """
    Convert a list of coefficients to a vector evaluated at points x.
    """
    res=np.zeros_like(x)
    for i, c in enumerate(coeff):
        res += c * np.sin((i + 1) * np.pi * x)
    return res

def get_average_relative_error(pd, ref):
    N, d = pd.shape
    k = np.arange(1, d+1)

    l2 = np.sqrt(np.sum((pd - ref)**2, axis=1))
    l2_ref = np.sqrt(np.sum(ref**2, axis=1))

    avg_relative_l2 = np.mean(l2 / l2_ref)

    h1 = np.sqrt(np.sum((1+(k * np.pi) ** 2) * (pd - ref) ** 2, axis=1))
    h1_ref = np.sqrt(np.sum((1+(k * np.pi) ** 2)  * ref ** 2, axis=1))
    avg_relative_h1 = np.mean(h1/h1_ref)

    return avg_relative_l2, avg_relative_h1

def get_average_error(pd, ref):
    N, d = pd.shape
    k = np.arange(1, d+1)

    # l2 = np.mean(np.sqrt(np.sum((pd - ref)**2, axis=1)))
    # h1 = np.mean(np.sqrt(np.sum((1+(k * np.pi) ** 2) * (pd - ref) ** 2, axis=1)))

    l2e = np.mean((np.sum((pd - ref) ** 2, axis=1)))
    h1 = np.mean((np.sum((1 + (k * np.pi) ** 2) * (pd - ref) ** 2, axis=1)))

    return l2e, h1