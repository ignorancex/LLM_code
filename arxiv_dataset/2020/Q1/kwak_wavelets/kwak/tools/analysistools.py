import numpy as np
from mpmath import mp

__all__ = ['_empty_like', '_zeros_like', 'FixedResGlobal']

def _empty_like(obj):
    empty = [np.empty_like(lev, dtype=object) for lev in obj]
    return empty

def _zeros_like(obj):
    zeros = [np.zeros_like(lev, dtype=float) for lev in obj]
    return zeros

def FixedResGlobal(nsigma):
    n_levels = len(nsigma)
    q_perlevel = np.empty(n_levels, dtype=object)
    for l, lev in enumerate(nsigma):
        q_total = 0
        nQs = len(lev)
        for j, sig in enumerate(lev):
            pval = mp.erfc(abs(sig)/mp.sqrt(2))
            q = -2*mp.log(pval)
            q_total += q
        q_perlevel[l] = [q_total, nQs]

    nsigma_fixedres = np.zeros(n_levels)
    for i, entry in enumerate(q_perlevel):
        qT = entry[0]
        nQ = entry[1]
        Dchi2 = mp.gammainc(nQ, 0, 0.5*qT)/mp.gamma(nQ)
        nsigma_ell = mp.sqrt(2)*mp.erfinv(Dchi2)
        nsigma_fixedres[i] = nsigma_ell
    return nsigma_fixedres
