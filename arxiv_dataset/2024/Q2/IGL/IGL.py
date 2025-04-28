import utils
import time
import numpy as np
from scipy import spatial
from pyunlocbox import functions, solvers


def IGL(X, W_physics, mask = 1, dist_type='sqeuclidean', alpha=1, beta=1, gamma = 1, step=0.1, w0=None, maxit=9000, rtol=1e-5, retall=False, verbosity='NONE'):

    # Parse X
    N = X.shape[0]
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse stepsize
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    K, Kt = utils.weight2degmap(N)
    norm_K = np.sqrt(2 * (N - 1))

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gammma: np.maximum(0, w - (2 * gammma * z))

    f2 = functions.func()
    f2._eval = lambda w: - alpha * np.sum(np.log(np.maximum(
        np.finfo(np.float64).eps, K(w))))
    f2._prox = lambda d, gammma: np.maximum(
        0, 0.5 * (d + np.sqrt(d**2 + (4 * alpha * gammma))))



    
    f3 = functions.func()
    f3._eval = lambda w: beta * np.sum(w**2) + gamma*(np.sum(mask*((w - W_physics))**2))
    f3._grad = lambda w: 2 * beta * w + 2 * gamma * (mask*(w - W_physics))
    lipg = 2 * (beta + gamma)

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # Solve problem
    solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize)
    problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
                            rtol=rtol, verbosity=verbosity)

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem['sol'])

    if retall:
        return W, problem
    else:
        return W
