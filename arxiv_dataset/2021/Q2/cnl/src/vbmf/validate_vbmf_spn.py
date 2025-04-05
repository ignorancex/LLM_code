
import scipy as sp
import numpy as np
from .vbmf import VBMF2
from matrix_util import *
from random_matrices import *

def validate_vbmf_spn(dim, p_dim, sigma, min_singular, zero_dim, COMPLEX=False):
    assert min_singular < 1
    assert(zero_dim <= dim )
    assert(p_dim >= dim)

    A = random_from_diag(p_dim, dim, zero_dim, min_singular, COMPLEX)
    sample = signal_plus_noise(A, sigma,COMPLEX=COMPLEX)
    ### Analytic solution
    obj = VBMF2(sample)
    r = obj.get_rank_analytically()

    return r
