import scipy as sp
import numpy as np
from vbmf import VBMF, VBMF2

from argparse import ArgumentParser
import logging

def options(logger=None):
    desc   = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = ArgumentParser(description = desc)

    # options
    parser.add_argument('-m', '--max_iter',
                        type     = int,
                        dest     = 'max_iter',
                        required = False,
                        default  =  20,
                        help     = "max_iter")
    parser.add_argument('-t', '--test_interval',
                        type     = int,
                        dest     = 'test_interval',
                        required = False,
                        default  =  5,
                        help     = "test_interval")
    parser.add_argument('-th', '--threshold',
                        type     = float,
                        dest     = 'threshold',
                        required = False,
                        default  =  0.1,
                        help     = "default:0.1")

    parser.add_argument('-d', '--use_decreament',
                        type     = bool,
                        dest     = 'use_decreament',
                        required = False,
                        default  =  False,
                        help     = "Use decrement(default:False)")

    parser.add_argument('-r', '--random',
                        type     = bool,
                        dest     = 'random',
                        required = False,
                        default  =  False,
                        help     = "random(default:False)")


    return parser.parse_args()

opt =options()
D = 64
min_singular = 0.3
num_zero = int(9*D/10)

sigma = 0.1
diag_A  = np.random.uniform( min_singular, 1, D)
for i in range(num_zero):
    diag_A[i] = 0

A = np.diag(diag_A)
Z = np.random.randn( D,D)/np.sqrt(D)
V = A + sigma*Z
### Iterative methods
obj = VBMF(V)
obj.optimize(opt.threshold, opt.max_iter, opt.test_interval, RANDOM=opt.random,use_decreament=opt.use_decreament)
print( "source:rank(V)=", np.linalg.matrix_rank(V))
print( "source:rank(A)=", np.linalg.matrix_rank(A))
print( "result:rank(A)=", np.linalg.matrix_rank(obj.A))
#import pdb; pdb.set_trace()

### Analytic solution
obj = VBMF2(V)
r = obj.get_rank_analytically()
