import numpy as np
from math import sqrt


def inexact_fb(X0, f, g, gradf, proxg, L, m, sigma, zeta, xi,
               maxiter_tot=1000, maxiter_inner=1000, backtrack=1, verbose=1,
               freq=10, tol_backtrack=0):

    r"""Inexact accelerated Forward Backward method for solving

        minimize f(x)+ g(x)

        Parameters
        ----------
        X0 : ndarray
            Initial guess.
        f : function, ndarray -> float
            Smooth part of the objective function.
        g : function, ndarray -> float
            Nonsmooth part of the objective function.
        gradf : function, ndarray -> ndarray
            Gradient of the smooth part f.
        proxg : function,
                (ndarray, ndarray, float, integer, float, float, float, float )
                -> (ndarray, ndarray, integer, float)
            Function that compute the inexact proximal operator of g.
        L : float
            Initial guess on the smoothness constant of f.
            If backtrack = 1, it can be a "bad" guess.
        m : float
            Value of the strong convexity parameter of g used in the algorithm.
        sigma : float
            Parameter between 0 and 1 that controls inexactness of
            prox computations relatively to the distance between
            consecutive iterates.
        zeta : float
            Parameter between 0 and 1 that controls inexactness of
            prox computations relatively to the subgradient norm of
            current iterate.
        xi : function, integer -> float
             Controls inexactness of prox computations in absolute value.
        maxiter_tot : integer, optional (default=1000)
            Maximal number of iterations (inner + outter).
        maxiter_inner : integer, optional (default=1000)
            Maximal number of inner iteration when calling proxg.
        backtrack: bool or integer, optional (default=1)
            If True or 1, use a backtracking lineseach strategy
            for the smoothness constant L.
        verbose : bool or integer, optional
            Amount of verbosity. 0/False is silent.
        freq : integer, optional (default=10)
            If verbose is True or 1, print infos every freq outter iterations.
        tol_backtrack: float, optional (default=0)
            tolerance in the inequalities used in backtracking linesearch
        Returns
        -------
        X: ndarray
            Approximate solution of the minimization problem.
        infos : ndarray, shape (_, 3)
            At each outer iterations, stores function values,
            precision reached in proxg and number of inner iterations of
            proxg.
        """

    X = X0.copy()
    Z = X0.copy()
    fx = f(X)
    gx = g(X)
    infos = np.array([fx+gx, 0, 0])
    Ak = 0
    k = 1
    while True:
        X_old = X.copy()
        stepsize = (1-sigma**2)/L
        eta = (1-zeta**2)*stepsize
        ak = (eta + 2*Ak*m*eta + sqrt(4*eta*Ak*(1+m*eta)*(1+Ak*m)
                                      + eta**2))/2

        Y = X_old + ak*(1+Ak*m)/(ak + Ak + Ak*(2*ak+Ak)*m)*(Z-X_old)

        gradf_y = gradf(Y)

        (X, V, proxiter, proxgap) = proxg(Y, gradf_y,
                                          stepsize, maxiter_inner,
                                          sigma, zeta, xi(k), m)

        fx = f(X)
        gx = g(X)
        infos = np.vstack([infos, np.array([fx+gx, proxiter, proxgap])])
        if verbose and (k % freq == 0):
            print("k = %i, proxIters = %i, proxGap = %e, f+g = %e, L = %e" %
                  (k, proxiter, proxgap, fx+gx, L))

        if backtrack:
            fy = f(Y)
            gradf_x = gradf(X)
            while (tol_backtrack < -fy + fx
                   + gradf_x.flatten().dot((Y-X).flatten())
                   + 1/2/L*np.sum(((gradf_x-gradf_y).flatten())**2)):
                if verbose:
                    print("Lipschitz estimate too small, increasing...")
                L *= 2
                stepsize = (1-sigma**2)/L
                eta = (1-zeta**2)*stepsize
                ak = (eta + 2*Ak*m*eta + sqrt(4*eta*Ak*(1+m*eta)*(1+Ak*m)
                                              + eta**2))/2

                Y = X_old + ak*(1+Ak*m)/(ak + Ak + Ak*(2*ak+Ak)*m)*(Z-X_old)

                gradf_y = gradf(Y)

                (X, V, proxiter, proxgap) = proxg(Y, gradf_y,
                                                  stepsize, maxiter_inner,
                                                  sigma, zeta, xi(k), m)

                fx = f(X)
                gx = g(X)
                fy = f(Y)
                gradf_x = gradf(X)
                infos = np.vstack([infos,
                                   np.array([fx+gx, proxiter, proxgap])])
                if verbose and (k % freq == 0):
                    print("k = %i, proxIters = %i, "
                          "proxGap = %e, f+g = %e, L = %e"
                          % (k, proxiter, proxgap, fx+gx, L))
                if np.sum(infos[:, 1]) >= maxiter_tot:
                    return (X, infos)
        # infos = np.vstack([infos,
        #                   np.array([fx+gx, proxiter, proxgap])])
        Z[:] -= ak/(1+m*(ak+Ak))*(V+gradf_y-m*(X-Z))
        Ak = ak+Ak

        k += 1

        if np.sum(infos[:, 1]) >= maxiter_tot:
            return (X, infos)

        if backtrack:
            L /= 1.1
