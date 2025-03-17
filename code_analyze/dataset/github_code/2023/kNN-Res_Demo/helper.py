import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def is_iter(e):
    try:
        _ = iter(e)
        return True
    except TypeError:
        return False


def subsample(X: np.ndarray,
              n: int = 200
              ) -> Tuple[np.ndarray, np.ndarray]:
    m = X.shape[0]
    if m != n:
        mask = np.random.randint(0, m, m if m < n else n)
        print(mask.min(), mask.max(), X.shape)
        return X[mask], mask
    mask = np.arange(0, m).astype(np.int)
    return X, mask


def ndmeshgrid(dim, step=1, mi=0, mx=1):
    mi = mi if is_iter(mi) else [mi] * dim
    mx = mx if is_iter(mx) else [mx] * dim
    x = [np.arange(mi[i], mx[i], (mx[i] - mi[i]) / step) for i in range(dim)]
    X = np.meshgrid(*x, indexing='ij')
    print(X[0].shape)
    X = np.concatenate([XX.flatten().reshape(-1, 1) for XX in X], axis=1)
    return X


def ndmeshfromdata(source_train, target_train=None, points=20):
    target_train = source_train if target_train is None else target_train
    grid_xy_min = torch.minimum(source_train.min(0)[0], source_train.min(0)[0]).detach().numpy()
    grid_xy_max = torch.maximum(target_train.max(0)[0], target_train.max(0)[0]).detach().numpy()
    mesh_test = ndmeshgrid(dim=source_train.shape[-1],
                           step=(grid_xy_max[0] - grid_xy_min[0]) / points,
                           mi=grid_xy_min[0],
                           mx=grid_xy_max[0],
                           )
    return torch.as_tensor(mesh_test, dtype=torch.float, )


def plot_grid_warp(input, output, s1=10, s2=20, linewidth=1,
                   c1='blue', c2='orange', label1='', label2='', alpha=1,  ax=None):
    dim = input.shape[1]
    if ax is None:
        ax = plt.subplot(projection='3d') if dim == 3 else plt.subplot()
    input_ax = [input[:, i] for i in range(dim)]
    output_ax = [output[:, i] for i in range(dim)]



    for i in range(output.shape[0]):
        pxx = [input[i, 0], output[i, 0]]
        pyy = [input[i, 1], output[i, 1]]
        if output.shape[1] == 3:
            pzz = [input[i, 2], output[i, 2]]
            ax.plot(pxx, pyy, pzz, color='k', linewidth=linewidth, alpha=alpha)
        else:
            ax.plot(pxx, pyy, color='k', linewidth=linewidth, alpha=alpha)

    ax.scatter(*input_ax, s=s1, c=c1, label=label1)
    ax.scatter(*output_ax, s=s2, c=c2, label=label2)


def plot_knn_graph(points, knn_indices, alpha=.8):
    plt.scatter(points[:, 0], points[:, 1])
    for i in range(knn_indices.shape[0]):
        idx = knn_indices[i]
        n_points = points[idx]
        input = points[i]
        for j in range(n_points.shape[0]):
            output = n_points[j]
            pxx = [input[0], output[0]]
            pyy = [input[1], output[1]]
            plt.plot(pxx, pyy, color='k', linewidth=.4, alpha=alpha)
