import gc

import numpy as np
import torch
from tqdm import tqdm

from .region import check_sat_constraints
from .pca_ellipsoid import get_Is
from ...utils import seed_random_state
from .nn_cutils import get_constraints


def torch_batch_project_sub_voronoi(x, cents, Gs, hs, device, step_size=0.1):
    x, Gs, hs = x.to(device), Gs.to(device), hs.to(device)
    alpha = torch.arange(0, 1+step_size, step_size).to(device).view(-1, 1, 1)
    n_tries = alpha.shape[0]
    batch_size, fet_size = x.shape[0], x.shape[1]
    pert = (x - cents).repeat(n_tries, 1).view(n_tries, batch_size, -1)
    Gs = Gs.repeat(n_tries, 1, 1).view(n_tries*batch_size, Gs.shape[1], fet_size)
    hs = hs.repeat(n_tries, 1).view(n_tries*batch_size, hs.shape[1])

    cents = cents.repeat(n_tries, 1).view(n_tries, batch_size, -1)
    tx = (cents + alpha * pert).view(n_tries*batch_size, fet_size, 1)
    ret = torch.bmm(Gs, tx)[:, :, 0] # (n_tries*n, keep_d)
    ret = torch.all(torch.le(ret, hs), axis=1).view(n_tries, batch_size)
    ret = ret.float() + alpha[:, :, 0] * 1e-5
    ret = ret.argmax(dim=0)
    ret = torch.gather(tx.view(n_tries, batch_size, -1), dim=0, index=ret.view(1, -1, 1).repeat(1, 1, fet_size))[0]
    return ret

def get_subvoronoi_constraints(X, y, sample_type, n_samples=-1, random_state=None):
    random_state = seed_random_state(random_state)
    X = X.astype(np.float64)
    n_classes = len(np.unique(y))
    Gs, hs = [], []

    if n_samples == -1:
        raise NotImplementedError()

    if sample_type == "rand":
        #dif_label_set = [X[y!=c] for c in range(n_classes)]
        #dif_y_set = [y[y!=c] for c in range(n_classes)]
        for i, x in tqdm(enumerate(X), total=len(X), desc="[get_subvoronoi_constraints]"):
            idx = random_state.choice(np.where(y != y[i])[0], size=n_samples, replace=False)
            assert len(idx) == n_samples
            tempX = X[idx.astype(int)].reshape(n_samples, -1)
            #tempX, tempy = dif_label_set[y[i]], dif_y_set[y[i]]
            #idx = random_state.choice(np.arange(len(tempX)), size=n_samples, replace=False)
            #assert len(idx) == n_samples
            #tempX = tempX[idx.astype(int)].reshape(n_samples, -1)
            G, h = get_constraints(tempX, x)
            assert check_sat_constraints(x.reshape(-1), G, h)
            Gs.append(G)
            hs.append(h)
    elif sample_type == "closest":
        Is = get_Is(X.astype(np.float32), y, n_samples=n_samples)
        for i, x in enumerate(X):
            tempX = X[Is[i].astype(int)].reshape(n_samples, -1)
            G, h = get_constraints(tempX, x)
            assert check_sat_constraints(x.reshape(-1), G, h)
            Gs.append(G)
            hs.append(h)
    else:
        NotImplementedError()

    return Gs, hs

def get_dist_to_constraints(cents, Gs, hs, scale_h=1):

    ret = np.zeros((len(cents), len(Gs[0])))
    for i, (c, G, h) in enumerate(zip(cents, Gs, hs)):
        if scale_h != 1:
            temp_h = np.copy(h)
            temp_h[h > 0] *= scale_h
            temp_h[h < 0] /= scale_h
        else:
            temp_h = h
        ret[i] = np.abs(np.dot(G, c) - temp_h) / np.linalg.norm(G, ord=2, axis=1)
    return ret

def get_sub_voronoi_info_list(X, y, scale_h=1, sample_type="closest", n_samples=-1,
        est_stepsize=False, n_jobs=4, random_state=None):
    print(f"sample_type: {sample_type}, n_samples: {n_samples}, scale_h: {scale_h}, est_stepsize: {est_stepsize}")
    X = X.reshape(len(X), -1)

    Gs, hs = get_subvoronoi_constraints(X, y, sample_type, n_samples=n_samples)
    if est_stepsize:
        stepsize = get_dist_to_constraints(X.reshape(len(X), -1), Gs, hs, scale_h=scale_h).max(1)
    ret = []
    for i, (G, h) in tqdm(enumerate(zip(Gs, hs)), desc="[get_sub_voronoi_info_list]", total=len(Gs)):
        G, h = torch.from_numpy(G[None, :, :]).float(), torch.from_numpy(h.reshape(1, -1)).float()
        cent = torch.from_numpy(X[i].reshape(1, -1)).float()
        ret.append(lambda x: torch_batch_project_sub_voronoi)
        ret[-1].torch_batch = "sub_voronoi"
        ret[-1].G = G
        ret[-1].cent = cent

        if scale_h != 1:
            temp_h = torch.clone(h)
            temp_h[h > 0] *= scale_h
            temp_h[h < 0] /= scale_h
            ret[-1].h = temp_h
        else:
            ret[-1].h = h

        if est_stepsize:
            ret[-1].stepsize = stepsize[i]

    del Gs
    del hs
    gc.collect()

    return ret