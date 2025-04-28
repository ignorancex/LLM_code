import numpy as np
from math import comb, factorial

def all_multindices(n, d):
    if n == 1:
        yield (d,)
    else:
        for i in range(d+1):
            for rest in all_multindices(n-1, d-i):
                yield (i,) + rest

def quantum_feature_map_vector(x, d, a=1.0):
    norm_x = np.linalg.norm(x)
    if d % 2 == 0:
        c = -1.0 - a * norm_x
    else:
        c = 1.0 - a * norm_x
    n = len(x)
    dim = comb(n + d, d)
    psi = np.zeros(dim, dtype=complex)
    idx = 0
    for k in all_multindices(n+1, d):
        num = (a**0.5) * (factorial(d)**0.5)
        den = 1.0
        prod = 1.0
        for i in range(n):
            prod *= x[i]**k[i]
            den *= factorial(k[i])
        prod *= (c**0.5)**k[n]
        den *= factorial(k[n])
        amp = num / (den**0.5)
        psi[idx] = amp * prod
        idx += 1
    norm_psi = np.linalg.norm(psi)
    if norm_psi > 0:
        psi /= norm_psi
    return psi

def quantum_feature_map_image(image, d, a=1.0):
    vec = image.flatten()
    return quantum_feature_map_vector(vec, d, a)

def encode_data(data1, data2, d=2, a=1.0):
    if isinstance(data1, np.ndarray) and len(data1.shape) == 1:
        if len(data1) != len(data2):
            raise ValueError("Vectors do not have the same length.")
        amps1 = quantum_feature_map_vector(data1, d, a)
        amps2 = quantum_feature_map_vector(data2, d, a)
    elif isinstance(data1, np.ndarray) and len(data1.shape) == 2:
        if data1.shape != data2.shape:
            raise ValueError("Images do not have the same dimensions.")
        amps1 = quantum_feature_map_image(data1, d, a)
        amps2 = quantum_feature_map_image(data2, d, a)
    else:
        raise TypeError("Unsupported datatype.")
    return amps1, amps2
