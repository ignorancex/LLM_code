from functools import partial

import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from ...utils import seed_random_state
from .nn_cutils import get_constraints, check_feasibility
from .solvers import solve_lp, solve_qp


CONSTRAINTTOL = 1e-6


def get_constraints_linf(X, tar_x):
    """
    trnX: 2 dim
    tar_x: 1 dim
    """
    d = X.shape[1]
    assert len(tar_x) == d

    G = np.vstack((np.eye(d, dtype=np.float), -np.eye(d, dtype=np.float)))
    h = np.concatenate((np.inf * np.ones(d, dtype=np.float), -np.inf * np.ones(d, dtype=np.float)))
    for x in X:
        for di, xi in enumerate((x - tar_x) / 2):
            if xi > 0:
                h[di] = xi + tar_x[di]
            else:
                h[d + di] = xi + tar_x[di]
    return G, h

def get_data_1nn_constraints(X, y, n_samples=-1, random_state=None):
    X = X.astype(np.float64)
    n_classes = len(np.unique(y))
    Gs, hs = [], []
    dif_label_set = [X[y!=c] for c in range(n_classes)]
    dif_y_set = [y[y!=c] for c in range(n_classes)]
    if n_samples == -1:
        for i, x in enumerate(X):
            G, h = get_constraints(dif_label_set[y[i]], x)
            Gs.append(G)
            hs.append(h)
    else:
        # randomly select opposite labeled examples
        random_state = seed_random_state(random_state)
        for i, x in enumerate(X):
            tempX, tempy = dif_label_set[y[i]], dif_y_set[y[i]]
            portion = min(n_samples, len(tempX)) / len(tempX)

            if portion != 1:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=portion, random_state=random_state)
                sss.get_n_splits(tempX, tempy)
                test_idx = [test_idx for _, test_idx in sss.split(tempX, tempy)][0]
                tempX = tempX[test_idx]

            G, h = get_constraints(tempX, x)
            Gs.append(G)
            hs.append(h)

    return Gs, hs

def check_sat_constraints(x, G, h) -> bool:
    """ Check if the constraint is satisfiable
    """
    return np.all(np.dot(G, x) <= h)

def get_region_proj_fn_list(X, y, norm, n_samples=-1, n_jobs=4, random_state=None):
    X = np.copy(X).reshape(len(X), -1)
    Gs, hs = get_data_1nn_constraints(X, y, n_samples=n_samples, random_state=random_state)
    ret = []
    for G, h in zip(Gs, hs):
        h = h.reshape(-1, 1)
        ret.append(partial(project_with_constraints, G=G, h=h, norm=norm, n_jobs=n_jobs))
    return ret

def project_with_constraints(tar_x, G, h, norm, n_jobs=4):
    """Takes numpy arrays
    """
    if norm == 2:
        proj_fn = project_with_constraints_l2
    elif norm == np.inf:
        proj_fn = project_with_constraints_linf

    if check_sat_constraints(tar_x, G, h):
        ret = tar_x
    else:
        ret =  proj_fn(tar_x, G, h, n_jobs=n_jobs)
    return ret

def project_with_constraints_linf(tar_x, G, h, n_jobs=4):
    fet_dim = tar_x.shape[0]

    c = np.concatenate((np.zeros(fet_dim), np.ones(1))).reshape((-1, 1))

    G2 = np.hstack((np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G3 = np.hstack((-np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G = np.hstack((G, np.zeros((G.shape[0], 1))))
    G = np.vstack((G, G2, G3))
    h = np.concatenate((h, tar_x, -tar_x)).reshape((-1, 1))

    temph = h - CONSTRAINTTOL

    status, sol = solve_lp(c=c, G=G, h=temph, n_jobs=n_jobs)
    if status == 'optimal':
        ret = np.array(sol).reshape(-1)
        return ret[:-1]
    else:
        return None


def project_with_constraints_l2(tar_x, G, h, n_jobs=4):
    n_fets = tar_x.shape[0]

    Q = 2 * np.eye(n_fets)
    q = -2 * tar_x

    temph = h - CONSTRAINTTOL # make sure all constraints are met

    status, sol = solve_qp(np.array(Q), np.array(q), np.array(G),
                           np.array(temph), n_fets, n_jobs=n_jobs)
    if status == 'optimal':
        ret = sol.reshape(-1)
        return ret
    else:
        return None
