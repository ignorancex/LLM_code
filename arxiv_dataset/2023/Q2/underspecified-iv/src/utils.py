
"""
utility functions
"""
import numpy as np
import logging
logging.basicConfig(level=20)
from jax import numpy as jnp
import jax
from jax import random
import sys
import os

from scipy.spatial.distance import cdist
sys.path.insert(0, "../src")


def projection(P1):
    """
    computation of projection matrix

    Parameters
    ----------
    P1: ndarray

    Returns
    -------
    ndarray
    """
    return P1 @ np.linalg.solve(P1.T @ P1, P1.T)


def cosine_distance(mat1, mat2):
    """
    computation of cosine distance

    Parameters
    ----------
    mat1: nd.array
    mat2: nd.array

    Returns
    -------
    float

    """

    mat_comp = np.vstack([mat1, mat2])

    cdist_mat = cdist(mat_comp, mat2, 'cosine')
    cdist_mat[cdist_mat > 1] = 2 - cdist_mat[cdist_mat > 1]

    return np.sum(cdist_mat)


def generate_sparse_vector(key, n, p, perc=0.8):
    """
    generate sparse vector with *perc* zero values and *(1-perc)* uniform distributed values on the interval of [-5.0;
    5.0]

    Parameters
    ----------
    key: jax key
    n: int
        number of samples
    p: int
        number pf treatment variables
    perc: float
        percentage of zero values

    Returns
    -------
    nd.array with perc of zero entries
    """

    p0 = int(jnp.floor(p * perc))
    p1 = p - p0

    key, key_sub = jax.random.split(key)
    vec_non_zero = jax.random.uniform(key_sub, (p1, n), minval=-5.0, maxval=5.0).reshape(p1, n)
    vec_zeros = jnp.zeros((p0, n))
    vec = jnp.concatenate([vec_non_zero, vec_zeros])
    key, key_sub = jax.random.split(key)
    vec = jax.random.permutation(key_sub, vec, independent=True)
    return vec.reshape(p, n)


def generate_scenario(key, d, p, d_identify=None, alpha_sparse=0.3, do_permute=False, do_cluster=True):
    """
    generate a scenario of $\alpha$, $\beta$ and mixing matrix $M$ and $e$ for estimation

    Parameters
    ----------
    key: jax key
    d: int
    p: int
    d_identify: int
    conf_zero_perc: float

    Returns
    -------
    alpha_star: nd.array
    beta_star: ndarray
    e_confounder: ndarray
    M: ndarray
        mixing matrix for confounder

    """
    l = p
    if d_identify is None:
        d_identify = int(np.ceil(d / 2))

    key, key_sub = jax.random.split(key)
    beta_prelim = generate_sparse_vector(key_sub, 1, d_identify, 0).squeeze()
    beta_prelim = np.hstack([beta_prelim, np.zeros(p - d_identify)])

    key, key_sub = jax.random.split(key)
    alpha_prelim = generate_sparse_vector(key_sub, d, d_identify, alpha_sparse).T
    alpha_prelim = np.hstack([alpha_prelim, np.zeros((d, p - d_identify))])

    if do_cluster:
        eig_vals = np.zeros((d_identify,))
        while (eig_vals).any() < 0.2:
            key, key_sub = jax.random.split(key)
            lam_pick = jax.random.choice(key_sub, np.arange(d), (d_identify,), replace=False)
            alpha_cap = alpha_prelim[lam_pick, :]
            eig_vals, _ = np.linalg.eig(projection(alpha_cap[:, :d_identify]))
            print(lam_pick)
            print(eig_vals)
        dim_c1 = np.int32((d - d_identify) / 2)
        dim_c2 = d - d_identify - dim_c1

        key, key_sub = jax.random.split(key)
        alpha_cluster1 = np.repeat(alpha_cap[d_identify-1, :], dim_c1).reshape((p, dim_c1)).T + (
            jax.random.normal(key_sub, (dim_c1, p))) / 100
        key, key_sub = jax.random.split(key)
        alpha_cluster2 = np.repeat(alpha_cap[d_identify-2, :], dim_c2).reshape((p, dim_c2)).T + (
            jax.random.normal(key_sub, (dim_c2, p))) / 100
        alpha_prelim = np.vstack([alpha_cap, alpha_cluster1, alpha_cluster2])


    if do_permute:
        key, key_sub = jax.random.split(key)
        vec_permute = jax.random.permutation(key_sub, np.vstack([beta_prelim.T, alpha_prelim]).T, independent=False).T

        alpha_star = vec_permute[1:, :]
        beta_star = vec_permute[0, :]
    else:
        alpha_star = alpha_prelim
        beta_star = beta_prelim

    key, key_sub = jax.random.split(key)
    M = jax.random.normal(key_sub, (l, p)).reshape(l, p)

    key, key_sub = jax.random.split(key)
    e_confounder = jax.random.uniform(key_sub, (l, )) / jax.random.uniform(key_sub, (l, )).sum()

    return alpha_star, beta_star, e_confounder, M


def generate_rv(key, d, p, n):
    """
    generate random variables for the experiment

    Parameters
    ----------
    key: jax key
    d: int
        number of instruments
    p: int
        number of treatment variables
    n: int
        number of samples

    Returns
    -------
    z: nd array
        instrument matrix with n x d
    u: nd.array
        confounder matrix with n x l resp. nxp

    Notes
    --------
    l as the dimension of the confounder -> to match Janzing's paper, the confounder has the same dimensionality as
    the treatment variable
    """

    l = p

    # instruments
    key, key_sub = jax.random.split(key)
    #z = random.bernoulli(key_sub, 0.5, (n, d)).astype(float)
    z = random.rademacher(key_sub, (n, d)).astype(float)

    # confounder
    key, key_sub = jax.random.split(key)
    u = random.normal(key_sub, (n, l))

    return z, u


def generate_model(alpha, beta, z, u, e, M, lam, error_model="simple"):
    """
    generate samples from the instrumental variable model specified with the parameters $\alpha, \beta, e,
    M$ and the list $\lambda$ that specifies which instruments are currently switched on

    Parameters
    ----------
    alpha: ndarray
    beta: ndarray
    z: ndarray
        instrument matrix
    u: ndarray
        confounder matrix
    e: ndarray
        confounder vector
    M: ndarray
        mixing matrix
    lam: list of bools
        indication which instruments are currently "switched on"
    error_model: string
        error model, currently only supports "simple"

    Returns
    -------
    x: ndarray
    y: ndarray
    z: ndarray

    """
    alpha = alpha[lam, :]
    z = z[:, lam]

    if error_model == "simple":
        # simple noise model
        c_x = u@M
        c_y = u@e

    x = z @ alpha + c_x
    y = x @ beta + c_y

    return x, y, z


def save_scenario(key, alpha, beta, e, m, save_path, name_id="scen"):
    """
    Save generated scenario for reproducibility

    Parameters
    ----------
    key
    alpha
    beta
    e
    save_path

    Returns
    -------
    dict: to be saved
    """

    d, p = alpha.shape
    save_dict = {
        "key": key,
        "alpha": alpha,
        "d": d,
        "p": p,
        "beta": beta,
        "conf_e": e,
        "m": m
    }

    name = str(name_id)+ ".npy"
    np.save(os.path.join(save_path, name), save_dict)
    return save_dict


