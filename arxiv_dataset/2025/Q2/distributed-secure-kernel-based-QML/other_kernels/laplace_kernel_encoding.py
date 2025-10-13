import numpy as np
from math import ceil, log2, pi

def generate_random_w_cauchy(n, D, alpha=1.0):
    w = np.random.standard_cauchy(size=(D, n)) / alpha
    return w

def generate_random_phases(D):
    return np.random.uniform(0, 2*pi, size=D)

def rff_feature_map_vector(x, w, phases):
    D = w.shape[0]
    amps = np.zeros(2*D, dtype=float)
    for j in range(D):
        dot_val = np.dot(w[j], x) + phases[j]
        amps[2*j] = np.cos(dot_val)
        amps[2*j + 1] = np.sin(dot_val)
    amps /= np.sqrt(D)
    return amps

def rff_feature_map_image(image, w, phases):
    vec = image.flatten()
    return rff_feature_map_vector(vec, w, phases)

def encode_data(data1, data2, D, w=None, phases=None, alpha=1.0):
    if w is None:
        if data1.ndim == 1:
            w = generate_random_w_cauchy(data1.shape[0], D, alpha)
        else:
            flat_dim = data1.shape[0] * data1.shape[1]
            w = generate_random_w_cauchy(flat_dim, D, alpha)
    if phases is None:
        phases = generate_random_phases(D)
    if isinstance(data1, np.ndarray) and data1.ndim == 1:
        if data1.shape != data2.shape:
            raise ValueError("Vectors do not have the same length.")
        amps1 = rff_feature_map_vector(data1, w, phases)
        amps2 = rff_feature_map_vector(data2, w, phases)
    elif isinstance(data1, np.ndarray) and data1.ndim == 2:
        if data1.shape != data2.shape:
            raise ValueError("Images do not have the same dimensions.")
        amps1 = rff_feature_map_image(data1, w, phases)
        amps2 = rff_feature_map_image(data2, w, phases)
    else:
        raise TypeError("Unsupported data type.")
    num_qubits = ceil(log2(2*D))
    return amps1, amps2, num_qubits
