import contextlib
import os
import gc
from functools import partial
import logging

import torch
import numba as nb
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import joblib
from joblib import Parallel, delayed
import cvxpy as cp

from .oppo_dist import get_faiss_index

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def get_indexes(trnX, trny, norm=2):
    indexes = []
    ori_idx = []
    for yi in tqdm(sorted(np.unique(trny)), desc="[nearest_oppo_dist]"):
        index = get_faiss_index(int(np.prod(trnX.shape[1:])), norm)
        ind = (trny!=yi)
        index.add(trnX[ind].reshape(ind.sum(), -1))
        ori_idx.append(np.where(ind)[0])
        indexes.append(index)
    return indexes, ori_idx

def get_Is_ori(X, y, n_samples):
    indexes, ori_idx = get_indexes(X, y)

    Is = np.zeros((len(X), n_samples), np.int64)
    for i, yi in tqdm(enumerate(sorted(np.unique(y))), total=len(np.unique(y)), desc="[get Is]"):
        ind = np.where(y==yi)[0]
        _, I = indexes[i].search(X[ind].reshape(-1, np.prod(X.shape[1:])), k=n_samples)
        ori_I_shape = I.shape
        Is[ind] = ori_idx[i][I.reshape(-1)].reshape(ori_I_shape)
    return Is

def get_Is(X, y, n_samples, norm=2):
    """Should use less memory"""
    #indexes, ori_idx = get_indexes(X, y)

    Is = np.zeros((len(X), n_samples), np.int64)
    for _, yi in tqdm(enumerate(sorted(np.unique(y))), total=len(np.unique(y)), desc="[get Is]"):
        ind = np.where(y==yi)[0]

        index = get_faiss_index(int(np.prod(X.shape[1:])), norm)
        ind2 = (y!=yi)
        index.add(X[ind2].reshape(ind2.sum(), -1))
        ori_idx = np.where(ind2)[0]

        _, I = index.search(X[ind].reshape(-1, np.prod(X.shape[1:])), k=n_samples)
        ori_I_shape = I.shape
        Is[ind] = ori_idx[I.reshape(-1)].reshape(ori_I_shape)

        del index
        gc.collect()
    return Is

def get_pcas(X, y, n_samples, n_jobs, random_state, cache_filename=None):
    if (cache_filename is not None) and os.path.exists(cache_filename % (n_samples, )):
        logging.info(f"[get_pcas] Using cache file: {cache_filename}")
        return joblib.load(cache_filename % (n_samples, ))
    else:
        Is = get_Is(X, y, n_samples)

        def learn_pca(tX):
            return PCA(random_state=random_state).fit(tX)

        with tqdm_joblib(tqdm(desc="learn_pca", total=len(X))) as _:
            pcas = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(learn_pca)(
                X[Is[i].astype(int)].reshape(n_samples, -1)) for i in range(len(X)))

        if cache_filename is not None:
            joblib.dump(pcas, cache_filename % (n_samples, ))
        return pcas

def get_pca_components(pcas, keep_d):
    n_features = pcas[0].components_.shape[1]
    comp_vars = np.zeros((len(pcas), keep_d, 1), dtype=np.float32, order="c")
    components = np.zeros((len(pcas), keep_d, n_features), dtype=np.float32, order="c")
    for i, pca in enumerate(pcas):
        comp_vars[i, :, 0] = pca.singular_values_[:keep_d]
        components[i] = pca.components_[:keep_d]
    return comp_vars, components

def estimate_stepsize(cent, tt, norm):
    """length of to the longest axis
    """
    alpha = (1 - np.dot(cent, tt.T)).reshape(-1) / (tt * tt).sum(1)
    return np.linalg.norm(alpha.reshape(-1, 1) * tt, ord=norm, axis=1).max()

def get_proj_fns(components, comp_vals, cent_x, norm, lamb, method,
                 est_stepsize=False, img_clip=True):

    #@nb.guvectorize(["void(float32[:], float32[:, :], float32[:], float32[:])"],
    #                 "(m),(m,m),(m)->(m)")
    @nb.njit(fastmath=True, parallel=False)
    def line_bin_search_fn(tar_x, tt, cent_x, result):
        tol = np.float32(1e-2)
        #tar_x = tar_x.reshape(-1)
        #cent_x = cent_x.reshape(-1)
        dif = (tar_x - cent_x).astype(np.float32)

        def in_ellipsoid(dif, tt):
            return (np.linalg.norm(np.dot(dif, tt.T), ord=2) <= 1)

        if in_ellipsoid(tar_x, tt):
            result = tar_x
        else:
            low, high = np.float32(0), np.float32(1)
            while low <= high-tol:
                mid = np.float32((low + high) / 2)

                temp_x = (cent_x + mid * dif).astype(np.float32)
                if in_ellipsoid(temp_x, tt):
                    low = mid
                else:
                    high = mid - tol

            result = (cent_x + low * dif).astype(np.float32)

    def torch_batch_project_ellipsoid_fn():
        pass

    def project_ellipsoid_fn(tar_x, tt, cent_x, norm=np.inf, n_jobs=4, solver=cp.GUROBI):
        tar_x = tar_x.reshape(1, -1, order="C").astype(np.float32)
        fet_dim = tar_x.shape[1]
        if solver == cp.GUROBI:
            options = {'verbose': 0, 'Threads': n_jobs, 'FeasibilityTol': 1e-2, 'OptimalityTol': 1e-2, 'IterationLimit': 20}
        elif solver == cp.CVXOPT:
            options = {'feastol': 1e-2, 'abstol': 1e-2, 'reltol': 1e-1, 'max_iters': 20}
        else:
            options = {}

        v = cp.Variable(shape=(1, fet_dim))
        obj = cp.Minimize(cp.norm(tar_x-v, p=norm))
        constraints = [cp.norm((v-cent_x) @ tt.T, p=2) <= 1]
        if img_clip is True:
            constraints += [v <= 1, v>=0]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=solver, **options)

        if prob.status == 'optimal':
            return np.array(v.value).reshape(-1)
        else:
            print(prob.status)
            import ipdb; ipdb.set_trace()
            return None

    cent_x = cent_x.astype(np.float32).reshape(cent_x.shape, order="C")
    tt = (components * comp_vals / lamb).astype(np.float32).reshape(components.shape, order="C")
    ret = []
    for i in tqdm(range(len(components)), total=len(components), desc="[get_proj_fns]"):
        if method == "opt":
            ret.append(partial(project_ellipsoid_fn,
                    tt=tt[i], cent_x=cent_x[i].reshape(1, -1), norm=norm))
        elif method == "batch":
            temp_tt = torch.from_numpy(tt[i].reshape(1, tt[i].shape[0], tt[i].shape[1])).float()
            temp_cent_x = torch.from_numpy(cent_x[i].reshape(1, -1)).float()
            ret.append(lambda x: torch_batch_project_ellipsoid_fn)
            #ret.append(partial(project_ellipsoid_fn,
            #        tt=tt[i], cent_x=cent_x[i].reshape(1, -1), norm=norm))
            ret[-1].torch_batch = "ellipsoid"
            ret[-1].tt = temp_tt
            ret[-1].cent = temp_cent_x
            if est_stepsize:
                ret[-1].stepsize = estimate_stepsize(cent_x[i].reshape(1, -1), tt[i], norm)
        elif method == "bin":
            if img_clip:
                def _fn(tar_x):
                    tar_x = np.clip(tar_x, 0, 1)
                    ret = np.zeros(np.prod(tar_x.shape), dtype=np.float32)
                    line_bin_search_fn(tar_x, tt[i], cent_x[i].reshape(-1), ret)
                    return ret
            else:
                def _fn(tar_x):
                    ret = np.zeros(np.prod(tar_x.shape), dtype=np.float32)
                    line_bin_search_fn(tar_x, tt[i], cent_x[i].reshape(-1), ret)
                    return ret
            ret.append(_fn)
                #ret.append(partial(line_bin_search_fn,
                #        tt=tt[i], cent_x=cent_x[i].reshape(1, -1)))
        else:
            raise ValueError(f"[get_proj_fns] method {method} not supported")
        ret[-1].scale_fn = lambda x: np.dot(np.dot(x, tt[i].T), tt[i])
    return ret

def check_in_ellipsoid(x, tt, c):
    t = (x - c)
    return (np.linalg.norm(np.dot(t, tt.T), axis=1) <= 1)

def binary_search(l, r, tt, pred_fn, threshold=0.05, tol=1.0):
    while l < (r-tol):
        mid = (l + r) / 2
        out = pred_fn(tt / mid).mean()
        if out <= threshold:
            l = mid + tol
        else:
            r = mid
    return l

def find_lamb(X, y, comp_vals, components, n_samples, threshold=0.05, tol=0.1, n_jobs=2,
            info_cache_filename=None):
    if (info_cache_filename is not None) and os.path.exists(info_cache_filename):
        logging.info(f"[find_lamb] Using cache file: {info_cache_filename}")
        return joblib.load(info_cache_filename)
    else:
        Is = get_Is(X, y, n_samples)

        def _helper(comp_val, comp, x, c):
            tt = (comp_val * comp).astype(np.float32)
            return binary_search(
                1.0, 500., tt, threshold=threshold, tol=tol,
                pred_fn=lambda xx: check_in_ellipsoid(tt=xx, x=x, c=c)
            )

        with tqdm_joblib(tqdm(desc="binary search", total=len(X))) as _:
            ret = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_helper)(
                comp_vals[i], components[i],
                x=X[Is[i].astype(int)].reshape(n_samples, -1), c=X[i].reshape(1, -1)) for i in range(len(X)))
        ret = np.array(ret, dtype=np.float32).reshape(-1, 1, 1)
        if info_cache_filename is not None:
            joblib.dump(ret, info_cache_filename)

    #ret = np.zeros((len(X), 1, 1), np.float32)
    #for i in tqdm(range(len(X)), total=len(X), desc="binary search"):
    #    tt = (comp_vals[i] * components[i]).astype(np.float32)
    #    ret[i, 0, 0] = binary_search(
    #        1.0, 500., tt, threshold=threshold, tol=tol,
    #        pred_fn=lambda xx: check_in_ellipsoid(tt=xx, x=X[Is[i].astype(int)].reshape(n_samples, -1), c=X[i].reshape(1, -1))
    #    )
    return ret


def get_pca_ellipsoid_proj_fn_list(X, y, norm, keep_d, n_samples=200, threshold=0.05,
        method="opt", est_stepsize=False, n_jobs=-1, random_state=0, cache_filename=None,
        info_cache_filename=None, is_img_data=True):
    pcas = get_pcas(X, y, n_samples, n_jobs, random_state=random_state, cache_filename=cache_filename)
    comp_vals, components = get_pca_components(pcas, keep_d)
    del pcas
    gc.collect()
    lamb = find_lamb(X, y, comp_vals, components, n_samples, threshold=threshold,
                    info_cache_filename=info_cache_filename)
    lamb = lamb / 2
    print(lamb.reshape(-1))
    return get_proj_fns(components, comp_vals=comp_vals, cent_x=X, norm=norm,
                        lamb=lamb, method=method, est_stepsize=est_stepsize, img_clip=is_img_data)
