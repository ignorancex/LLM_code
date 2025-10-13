import numpy as np

def l2_normalize(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, 2, axis=-1)
    norms = np.expand_dims(norms, -1)
    return vectors / norms


def radius(vectors: np.ndarray):
    centroid = l2_normalize(np.mean(vectors.astype(np.float32), axis=0))
    distances = np.arccos(np.clip(vectors.astype(np.float32) @ centroid, -1, 1))
    return np.max(distances)
