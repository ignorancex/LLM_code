import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.special import expit
from copy import deepcopy
from scipy.stats import bernoulli
import math
from itertools import permutations
from scipy.spatial.distance import cdist, euclidean


def logreg_loss(x, args):
    A = args[0]
    y = args[1]
    l2 = args[2]
    sparse = args[3]
    assert l2 >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    degree1 = np.zeros(A.shape[0])
    if sparse == True:
        degree2 = -(A * x) * y
        l = np.logaddexp(degree1, degree2)
    else:
        degree2 = -A.dot(x) * y
        l = np.logaddexp(degree1, degree2)
    m = y.shape[0]
    return np.sum(l) / m + l2/2 * norm(x) ** 2

def logreg_grad(x, args):
    A = args[0]
    y = args[1]
    mu = args[2]
    sparse = args[3]
    assert mu >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    if sparse == True:
        degree = -y * (A * x)
        sigmas = expit(degree)
        loss_grad = -A.transpose() * (y * sigmas) / A.shape[0]
    else:
        degree = -y * (A.dot(x))
        sigmas = expit(degree)
        loss_grad = -A.T.dot(y * sigmas) / A.shape[0]
    assert len(loss_grad) == len(x)
    return loss_grad + mu * x

def logreg_grad_plus_lasso(x, args):
    return logreg_grad(x, args) + args[4] * np.sign(x)

def r(x, l1):
    assert (l1 >= 0)
    return l1 * norm(x, ord = 1)

def F(x, args):
    return logreg_loss(x, args) + r(x, args[4])
    
def prox_R(x, l1):
    res = np.abs(x) - l1 * np.ones(len(x))
    return np.where(res > 0, res, 0) * np.sign(x)

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def CM(W, s , G):
    N = math.ceil(len(W)/s)
    z = np.zeros((N, len(G[0])))
    for i in range(N):
        for k in range(i*s, min(len(W), (i+1)*s)):
            z[i] += G[W[k]-1]
    return np.median(z/s, axis=0)

def GM(W, s, G):
    N = math.ceil(len(W)/s)
    z = np.zeros((N, len(G[0])))
    for i in range(N):
        for k in range(i*s, min(len(W), (i+1)*s)):
            z[i] += G[W[k]-1]
    return geometric_median(z/s)