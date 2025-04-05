from typing import Tuple

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import mode


def remove_labels_legacy(x_train: np.array, y_train: np.array) -> Tuple[np.array, np.array]:
    # "Remove" labels
    lost_idx = np.random.choice(
        len(y_train), size=int(len(y_train) - np.sqrt(len(y_train))), replace=False)

    x_lost = x_train[lost_idx]
    x_rest = np.delete(x_train, lost_idx, axis=0)
    y_lost = y_train[lost_idx]
    y_rest = np.delete(y_train, lost_idx, axis=0)

    if len(x_lost.shape) == 1:
        x_lost = x_lost.reshape(1, -1)
    if len(x_rest.shape) == 1:
        x_rest = x_rest.reshape(1, -1)

    # Impute data
    tree = KDTree(x_rest)
    _, idx = tree.query(x_lost, k=int(np.sqrt(np.sqrt(len(x_rest)))), p=1)
    y_lost = mode(y_rest[idx], axis=1)[0]

    y_lost = y_lost.reshape(-1)

    print(y_lost.shape, y_rest.shape)

    assert len(x_lost) == len(y_lost)

    x_train = np.concatenate((x_lost, x_rest), axis=0)
    y_train = np.concatenate((y_lost, y_rest), axis=0)
    return x_train, y_train
