# pylint: disable=R,C,E1101
import math
from functools import lru_cache
import torch
import torch.cuda


def so3_sum_n(x):
    '''
    :param x: [l * m * n, batch, feature_in, complex]
    :return:  [l * m, batch, feature_in, complex]
    '''
    from s2cnn.utils.complex import complex_sum

    assert x.size(3) == 2
    nbatch = x.size(1)
    nfeature_in = x.size(2)
    nspec = x.size(0)
    nl = math.ceil((3 / 4 * nspec) ** (1 / 3))  # this is the band limit
    # note: Sum[(2*l+1)^2,{l,0,L-1}] == 1/3 * L * (4*L^2-1) (in Mathematica notation)
    assert nspec == nl * (4 * nl ** 2 - 1) // 3

    # We don't have a cuda implementation yet...
    # if x.is_cuda:
    #    return _cuda_SO3_mm.apply(x, y)

    Fz_list = []
    begin = 0
    for l in range(nl):
        L = 2 * l + 1
        size = L**2

        Fx = x[begin:begin + size]  # [m * n,   batch,    feature_in,  complex]

        Fx = Fx.view(L, L, nbatch, nfeature_in, 2)  # [m, n, batch, feature_in, complex]
        Fx = Fx.contiguous()

        Fz = complex_sum(Fx, 1)  # [m, batch, feature_in, complex]

        Fz_list.append(Fz)

        begin += size

    z = torch.cat(Fz_list, 0)  # [l * m, batch, feature_in, complex]
    return z
