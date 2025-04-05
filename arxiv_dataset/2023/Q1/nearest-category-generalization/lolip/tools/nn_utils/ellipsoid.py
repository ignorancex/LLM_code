"""
https://cvxopt.org/examples/book/ellipsoids.html
"""
import os
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm
from math import log, pi
import numpy as np
import cvxpy as cp
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

from cvxopt import blas, lapack, solvers, matrix, sqrt, mul, cos, sin
solvers.options['show_progress'] = False

from ...utils import seed_random_state
from .solvers import solve_lp, solve_qp
from .region import get_constraints, get_constraints_linf


CONSTRAINTTOL = 1e-6

def check_in_ellipsoid(x, invBB, c) -> bool:
    """ Check if the constraint is satisfiable
    """
    return np.all(np.dot(np.dot((x-c).T, invBB), x-c) <= 1)

def get_analytic_ellipsoid_proj_fn_list(X, y, norm, lamb=1., n_samples=-1, n_jobs=4, random_state=None, cache_filename=None):

    if cache_filename is not None and os.path.exists(cache_filename):
        print(f"[get_ellipsoid_proj_fn_list] Using cache: {cache_filename}")
        ret = joblib.load(cache_filename)
    else:
        print(f"[get_ellipsoid_proj_fn_list] cache {cache_filename} don't exist. Calculating...")

        X = np.copy(X).reshape(len(X), -1)
        invBBs, cs = analytic_ellipsoid_constraints(X, y, norm=norm, n_samples=n_samples, n_jobs=n_jobs, random_state=random_state)
        ret = []
        for invBB, c in zip(invBBs, cs):
            ret.append(partial(project_ellipsoid, invBB=invBB, c=c, norm=norm, n_jobs=n_jobs))
            ret[-1].invBB = invBB
            ret[-1].c = c
            ret[-1].norm = norm

        if cache_filename is not None:
            joblib.dump(ret, cache_filename)
    return ret

def get_ellipsoid_proj_fn_list(X, y, norm, lamb=1., n_samples=-1, n_jobs=4, random_state=None, cache_filename=None):

    if cache_filename is not None and os.path.exists(cache_filename):
        print(f"[get_ellipsoid_proj_fn_list] Using cache: {cache_filename}")
        ret = joblib.load(cache_filename)
    else:
        print(f"[get_ellipsoid_proj_fn_list] cache {cache_filename} don't exist. Calculating...")

        X = np.copy(X).reshape(len(X), -1)
        invBBs, cs = get_max_ellipsoid(X, y, norm=norm, n_samples=n_samples, n_jobs=n_jobs, random_state=random_state)
        ret = []
        for invBB, c in zip(invBBs, cs):
            ret.append(partial(project_ellipsoid, invBB=invBB, c=c, norm=norm, n_jobs=n_jobs))
            ret[-1].invBB = invBB
            ret[-1].c = c
            ret[-1].norm = norm

        if cache_filename is not None:
            joblib.dump(ret, cache_filename)
    return ret

def project_ellipsoid(tar_x, invBB, c, norm, n_jobs=4):
    """Takes numpy arrays
    """
    if norm in [1, 2, np.inf]:
        proj_fn = project_ellipsoid
    else:
        raise ValueError(f"project ellipsoid norm: {norm} not supported")

    #tar_x = tar_x.reshape(-1, 1)
    if check_in_ellipsoid(tar_x, invBB, c):
        ret = tar_x
    else:
        ret = proj_fn(tar_x, invBB, c, n_jobs=n_jobs)
    return ret

def project_ellipsoid_fn(tar_x, B, c, norm=np.inf, n_jobs=4, solver=cp.GUROBI):
    fet_dim = tar_x.shape[0]
    options = {'threads': n_jobs}

    v = cp.Variable(shape=(fet_dim))
    obj = cp.Minimize(cp.norm(tar_x-v, p=norm))
    constraints = [cp.quad_form(v - c, B) <= 1]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, **options)

    if prob.status == 'optimal':
        return np.array(v.value).reshape(-1)
    else:
        return None

def constraints_each_coordinates(X):
    n_fets = X.shape[1]
    G = np.vstack((np.eye(n_fets), -np.eye(n_fets)))
    h = np.concatenate((X.max(axis=0), -X.min(axis=0)))
    return G, h

def get_max_ellipsoid(X, y, norm=2, n_samples=-1, n_jobs=1, random_state=None):
    X = np.asarray(X, np.float)
    n_classes = len(np.unique(y))
    invBBs, cs = [], []
    dif_label_set = [X[y!=c] for c in range(n_classes)]
    dif_y_set = [y[y!=c] for c in range(n_classes)]
    random_state = seed_random_state(random_state)

    for i, x in tqdm(enumerate(X), desc="Calculating the max volume ellipsiod"):
        tempX, tempy = dif_label_set[y[i]], dif_y_set[y[i]]
        if n_samples != -1:
            portion = min(n_samples, len(tempX)) / len(tempX)

            if portion != 1:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=portion, random_state=random_state)
                sss.get_n_splits(tempX, tempy)
                test_idx = [test_idx for _, test_idx in sss.split(tempX, tempy)][0]
                tempX = tempX[test_idx]

        if norm == 2:
            G, h = get_constraints(tempX, x)
        elif norm == np.inf:
            G, h = get_constraints_linf(tempX, x)
        G2, h2 = constraints_each_coordinates(X)
        G, h = np.vstack((G, G2)), np.concatenate((h, h2))
        B, c = max_volume_ellipsoid_constraints(G, h, n_jobs=n_jobs)
        invBBs.append(np.linalg.pinv(np.dot(B, B)))
        cs.append(c)

    return invBBs, cs

def analytic_ellipsoid_constraints(tar_x, G, h, n_jobs, solver=cp.SCS):
    """
    minimize     log det A^-1
    subject to   xk'*A*xk - 2*xk'*b + b'*A^1*b <= 1,  k=1,...,m
    """
    options = {"solver": solver}
    n_fets = len(tar_x)
    B_var = cp.Variable(shape=(n_fets, n_fets))
    c_var = cp.Variable(shape=(n_fets))
    obj = cp.Maximize(cp.log_det(B_var))
    constraints = [cp.norm(G@B_var, p=2, axis=1) + G@c_var <= h]

    problem = cp.Problem(obj, constraints)
    problem.solve(**options)

    assert problem.status == "optimal", problem.status

    return B_var.value, c_var.value

def max_volume_ellipsoid_constraints(G, h, n_jobs, solver=cp.MOSEK):
    """
    maximize     log det B
    subject to   || B G_i || + G_i^T d <= h_i,  for i = 1,...,m
    """
    options = {"solver": solver, "verbose": False}
    n_fets = G.shape[1]
    B_var = cp.Variable(shape=(n_fets, n_fets))
    c_var = cp.Variable(shape=(n_fets))
    obj = cp.Maximize(cp.log_det(B_var))
    constraints = [cp.norm(G@B_var, p=2, axis=1) + G@c_var <= h]

    problem = cp.Problem(obj, constraints)
    problem.solve(**options)

    assert problem.status == "optimal", problem.status

    return B_var.value, c_var.value


#def max_volume_ellipsoid_constraints(tar_x, G, h, n_jobs, solver=cp.MOSEK):
#    # Maximum volume enclosed ellipsoid center
#    #
#    # minimize    -log det B
#    # subject to  ||B * gk||_2 + gk'*c <= hk,  k=1,...,m
#    #
#    # with variables  B and c.
#    #
#    # minimize    -log det L
#    # subject to  ||L' * gk||_2^2 / (hk - gk'*c) <= hk - gk'*c,  k=1,...,m
#    #
#    # L lower triangular with positive diagonal and B*B = L*L'.
#    #
#    # minimize    -log x[0] - log x[2]
#    # subject to   g( Dk*x + dk ) <= 0,  k=1,...,m
#    #
#    # g(u,t) = u'*u/t - t
#    # Dk = [ G[k,0]   G[k,1]  0       0        0
#    #        0        0       G[k,1]  0        0
#    #        0        0       0      -G[k,0]  -G[k,1] ]
#    # dk = [0; 0; h[k]]
#    #
#    # 5 variables x = (L[0,0], L[1,0], L[1,1], c[0], c[1])
#    m = np.shape(G)[0]
#    n_fets = np.shape(G)[1]
#    x_size = n_fets*(1+n_fets)//2 + n_fets
#
#    diagonal_mask = []
#    c = 0
#    for i in range(n_fets):
#        diagonal_mask.append(int(c+i))
#        c += i+1
#
#    D = np.zeros((m, n_fets+1, x_size), dtype=np.float64)
#    for k in range(m):
#        c = 0
#        for i in range(n_fets):
#            D[k][i][c:c+n_fets-i] = G[k][i:]
#            c += n_fets-i
#        D[k][-1][-n_fets:] = -G[k]
#    d = [([0.0] * n_fets + [float(hk)]) for hk in h]
#
#    G, h = matrix(G, tc='d'), matrix(h, tc='d')
#    D = [matrix(Dk, tc='d') for Dk in D]
#    d = [matrix(dk, tc='d') for dk in d]
#
#    #@profile
#    def F(x=None, z=None):
#        if x is None:
#            #return m, matrix([ 1.0, 0.0, 1.0, 0.0, 0.0 ])
#            ret = np.zeros((x_size, 1), dtype=np.float)
#            ret[diagonal_mask, 0] = 1.
#            ret[-n_fets:, 0] = tar_x
#            return m, matrix(ret)
#        if min(min(x[diagonal_mask, 0]), min(h-G*x[-n_fets:])) <= 0.0:
#            return None
#        #center = np.array(x[-n_fets:, 0])
#        #if np.linalg.norm(tar_x-center, ord=2) > 0.1:
#        #    return None
#
#        y = [Dk*x + dk for Dk, dk in zip(D, d)]
#
#        f = np.zeros((m+1, 1))
#        #f = matrix(0.0, (m+1,1))
#        #-log(x[0]) - log(x[2])
#        for i in diagonal_mask:
#            f[0][0] += -log(x[int(i)])
#        f = matrix(f, tc='d')
#
#        for k in range(m):
#            f[k+1] = y[k][:n_fets].T * y[k][:n_fets] / y[k][n_fets] - y[k][n_fets]
#
#        Df = matrix(0.0, (m+1, x_size))
#        for i in diagonal_mask:
#            Df[0, i] = -1.0/x[i]
#
#        # gradient of g is ( 2.0*(u/t);  -(u/t)'*(u/t) -1)
#        for k in range(m):
#            a = y[k][:n_fets] / y[k][n_fets]
#            gradg = matrix(0.0, (n_fets+1,1))
#            gradg[:n_fets], gradg[n_fets] = 2.0 * a, -a.T*a - 1
#            Df[k+1,:] =  gradg.T * D[k]
#        if z is None:
#            return f, Df
#
#        H = matrix(0.0, (x_size, x_size))
#        for i in diagonal_mask:
#            H[i,i] = z[0] / x[i]**2
#        #H[0,0] = z[0] / x[0]**2
#        #H[2,2] = z[0] / x[2]**2
#
#        # Hessian of g is (2.0/t) * [ I, -u/t;  -(u/t)',  (u/t)*(u/t)' ]
#        for k in range(m):
#            a = y[k][:n_fets] / y[k][n_fets]
#            #hessg = matrix(0.0, (n_fets+1, n_fets+1))
#            #for i in range(n_fets):
#            #    hessg[i, i] = 1.
#            hessg = matrix(np.eye(n_fets+1), (n_fets+1, n_fets+1))
#            hessg[:n_fets, n_fets], hessg[n_fets, :n_fets] = -a, -a.T
#            hessg[n_fets, n_fets] = a.T * a
#            H += (z[k] * 2.0 / y[k][n_fets]) *  D[k].T * hessg * D[k]
#
#        return f, Df, H
#
#
#    # Extreme points (with first one appended at the end)
#    #X = matrix([ 0.55,  0.25, -0.20, -0.25,  0.00,  0.40,  0.55,
#    #            0.00,  0.35,  0.20, -0.10, -0.30, -0.20,  0.00 ], (7,2))
#    #m = X.size[0] - 1
#
#    # Inequality description G*x <= h with h = 1
#    #G, h = matrix(0.0, (m,2)), matrix(0.0, (m,1))
#    #G = (X[:m,:] - X[1:,:]) * matrix([0., -1., 1., 0.], (2,2))
#    #h = (G * X.T)[::m+1]
#    #G = mul(h[:,[0,0]]**-1, G)
#    #h = matrix(1.0, (m,1))
#
#    #GG = matrix(np.vstack((np.eye(x_size), -np.eye(x_size))), tc='d')
#    #hh = np.ones(2*x_size)
#    #hh[:n_fets*(1+n_fets)//2] = (np.max(X) - np.min(X))
#    #hh[x_size:x_size+n_fets*(1+n_fets)//2] = (np.max(X) - np.min(X))
#    #hh[n_fets*(1+n_fets)//2:] = np.max(X)
#    #hh[x_size+n_fets*(1+n_fets)//2:] = -np.min(X)
#    #hh = matrix(hh, tc='d')
#
#    sol = solvers.cp(F)
#    L = np.array(sol['x'][:-n_fets])
#    B = np.zeros((n_fets, n_fets))
#    for i in range(n_fets):
#        if i == 0:
#            B[i][:i+1] = L[:i+1, 0]
#        else:
#            B[i][:i+1] = L[diagonal_mask[i-1]+1:(diagonal_mask[i]+1), 0]
#    B = B.dot(B.T).T
#    c = np.array(sol['x'][-n_fets:])
#    #if sol['status'] != 'optimal':
#    #    import ipdb; ipdb.set_trace()
#    assert sol['status'] == "optimal", sol['status']
#    return B, c
