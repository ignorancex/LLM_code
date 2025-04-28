import numpy as np
from math import ceil, log2

def generate_random_w(n, D, sigma=1.0):
    return np.random.normal(0, 1.0 / sigma, size=(D, n))

def rff_feature_map_vector(x, w):
    D = w.shape[0]
    amps = np.zeros(2*D, dtype=float)
    for j in range(D):
        dot_val = np.dot(w[j], x)
        amps[2*j] = np.cos(dot_val)
        amps[2*j + 1] = np.sin(dot_val)
    norm_factor = np.sqrt(D)
    amps /= norm_factor
    return amps

def rff_feature_map_image(image, w):
    vec = image.flatten()
    return rff_feature_map_vector(vec, w)

def encode_data(data1, data2, D, w=None, sigma=1.0):
    if w is None:
        if data1.ndim == 1:
            w = generate_random_w(data1.shape[0], D, sigma)
        else:
            flat_dim = data1.shape[0] * data1.shape[1]
            w = generate_random_w(flat_dim, D, sigma)
    if isinstance(data1, np.ndarray) and data1.ndim == 1:
        if len(data1) != len(data2):
            raise ValueError("Vectors do not have the same length.")
        amps1 = rff_feature_map_vector(data1, w)
        amps2 = rff_feature_map_vector(data2, w)
    elif isinstance(data1, np.ndarray) and data1.ndim == 2:
        if data1.shape != data2.shape:
            raise ValueError("Images do not have the same dimensions.")
        amps1 = rff_feature_map_image(data1, w)
        amps2 = rff_feature_map_image(data2, w)
    else:
        raise TypeError("Unsupported datatype.")
    num_qubits = ceil(log2(2*D))
    return amps1, amps2, num_qubits
