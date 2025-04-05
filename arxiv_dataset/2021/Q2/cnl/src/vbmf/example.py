import scipy as sp
import numpy as np
from vbmf import VBMF

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
L = 512
M = 256*3*3
H = 128
A = np.zeros((M,H))
for i in range (H):
    A[i][i] += 1
A += 0.5*np.random.randn(M,H)
#A = np.dot(A, A.T)

B = np.zeros((L,H))
for i in range (H):
    B[i][i] += 1

B += 0.5*np.random.randn(L,H)
#B = np.dot(B, B.T)
sigma = 1
Z = np.random.randn(L,M)
Z = sigma*Z
V = np.dot(B, A.transpose()) + Z
obj = VBMF(V)
obj.optimize(opt.threshold, opt.max_iter, opt.test_interval, RANDOM=opt.random,use_decreament=opt.use_decreament)
print( "source:rank(V)=", np.linalg.matrix_rank(V))
print( "source:rank(A)=", np.linalg.matrix_rank(A))
print( "result:rank(BA^T)=", np.linalg.matrix_rank(np.dot(B,A.T)))
print( "result:rank(A)=", np.linalg.matrix_rank(obj.A))
#import pdb; pdb.set_trace()
