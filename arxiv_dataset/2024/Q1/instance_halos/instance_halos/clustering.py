import numpy as np
from scipy.spatial.ckdtree import cKDTree
import logging
import time

class Timer():
    def __init__(self, log=logging.info):
        self.t0 = time.time()
        self.log = log
    def logdt(self, message, reset=True):
        t1 = time.time()
        self.log("%s (%.2fs)" % (message, t1-self.t0))
        if reset:
            self.t0 = t1

def sph_density(x, k=20, kernel="tophat", nthreads=8, get_neighbors=False):
    dim = x.shape[-1]
    
    tree = cKDTree(x)
    
    r, inn = tree.query(x,  k=k, workers=nthreads)
    h = r[:,-1]
    
    # For these two modes, the kernel width is defined at the evaluation point
    if kernel == "tophat":
        rho = k/r[:,-1]**3
    elif kernel == "linear":
        h = r[:,-1]
        u = r / h[:, np.newaxis]
        rho = np.sum(1.-u, axis=-1) / h**dim
    elif kernel == "other_tophat":
        weights = np.ones_like(r) / h[:,np.newaxis]**dim
        rho = np.bincount(inn.flat, weights=weights.flat)
    elif kernel == "other_linear":
        u = r / h[:,np.newaxis]
        weights = (1. - u) / h[:,np.newaxis]**dim
        rho = np.bincount(inn.flat, weights=weights.flat)
    else:
        raise ValueError("Unknown kernel %s" % kernel)
    
    if get_neighbors:
        return rho, inn
    else:
        return rho
        

from .cython_funcs import group_particles_descending

def meshfree_descending_clustering(x, kdens=20, klink=15, min_pers_ratio=4.2, kernel="tophat", full_output=False, nthreads=8):
    """More or less implements the density clustering of subfind (arXiv:0012055),
    but assigns not only the levelset, but the whole descending manifold"""
    assert kdens >= klink
    
    timer = Timer(log = logging.debug)
    rho, inn = sph_density(x, k=kdens, kernel=kernel, nthreads=nthreads, get_neighbors=True)
    assert(inn >= 0, "Error inn < 0")
    inn = inn[:,:klink]
    timer.logdt("meshfree_descending_clustering: Estimated Density and Neighbors")

    group_id, group_ptr, group_rhomax, group_rhosad = group_particles_descending(rho, np.int32(inn), min_pers_ratio=min_pers_ratio)
    
    ngroups = len(group_ptr)
    # Iterate from densest to lowest density particle

    timer.logdt("meshfree_descending_clustering: Determined Topology")

    while np.any(group_ptr[group_ptr] != group_ptr):
        group_ptr = group_ptr[group_ptr]

    mask = group_ptr != np.arange(0, len(group_ptr))

    # Relabel to omit now empty groups
    relabel = np.cumsum(~mask) - 1*(~mask)
    map_groups = relabel[group_ptr]

    group_id = map_groups[group_id]
    
    timer.logdt("meshfree_descending_clustering: Relabled")
    
    if full_output:
        return group_id, pers
    else:
        return group_id