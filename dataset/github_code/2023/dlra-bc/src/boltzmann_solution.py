#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytic solution of "trianlge_inflow" in tensor form.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class BoltzmannSolution:
    """Tensor approximation on regular grid of exact solution via method of lines"""
    def __init__(self, x0, sigmax, v0, sigmav, E, Lx, Nx, Lv, Nv, rmax, tau):
        """

        Parameters
        ----------
        x0      spatial center (numpy.array)
        sigmax  spatial width
        v0      velocity center (numpy.array)
        sigmav  velocity width
        E       electrical field (numpy.array)
        Lx      spatial domain [-Lx,Lx]^2
        Nx      spatial mesh parameter in each dimension
        Lv      velocity domain [-Lv,Lv]^2
        Nv      velocity mesh parameter in each dimension
        rmax    maximal rank
        tau     tolernace for SVD approximation
        """
        self.x0 = x0
        self.sigmax = sigmax
        self.v0 = v0
        self.sigmav = sigmav
        self.E = E
        self.Lx = Lx
        self.Nx = Nx
        self.Lv = Lv
        self.Nv = Nv

        self.x = np.linspace(-self.Lx, self.Lx, Nx+1)
        self.v = np.linspace(-self.Lv, self.Lv, Nv+1)

        self.X, self.V = np.meshgrid(self.x, self.v, indexing='ij')

        self.rmax = rmax
        self.tau = tau

        # storage for factors
        self.Qx = [None]*2
        self.Qv = [None]*2
        self.sigma = [None]*2

        self.t = 0.0
        
        self.debug = False

        return

    def psi(self, Z):
        Z = np.abs(Z)
        return (Z <= 1) * (Z**2 * (2*Z-3) + 1)

    def u(self, t, x, v):
        """Evaluate solution via method of lines"""
        z =   self.psi((x[0] - v[0] * t - 0.5 * self.E[0] * t**2 - self.x0[0])/self.sigmax) \
            * self.psi((x[1] - v[1] * t - 0.5 * self.E[1] * t**2 - self.x0[1])/self.sigmax) \
            * self.psi((v[0] + self.E[0] * t - self.v0[0])/self.sigmav) \
            * self.psi((v[1] + self.E[1] * t - self.v0[1])/self.sigmav)
        return z

    def set_time(self, t):
        """Compute solution at time t"""
        self.t = t

        # compute factors for xi,vi
        for i in range(2):
            # coordinates for method of characteristics for xi, vi
            Xmod = self.X - self.V * self.t - 0.5 * self.E[i] * self.t**2 \
                - self.x0[i]
            Vmod = self.V + self.E[i] * self.t - self.v0[i]

            # solution for xi,vi
            U = self.psi(Xmod/self.sigmax) * self.psi(Vmod/self.sigmav)

            # compute truncated svd for xi,vi
            Qx, sigma, QvT = np.linalg.svd(U)
            Qv = QvT.T
            if self.debug: print(sigma)

            # truncate
            currank = len(sigma)
            sigma_flipped = np.flip(sigma)
            mask = np.cumsum(sigma_flipped ** 2) < self.tau ** 2
            kinv = np.argmin(mask)
            if mask[kinv] == True:  # all values smaller than rauc_tau
                kinv = currank
            rank = max(1, min(currank - kinv, self.rmax))
            if self.debug: print(rank)

            self.sigma[i] = sigma[:rank]
            self.Qx[i] = Qx[:, :rank]
            self.Qv[i] = Qv[:, :rank]

            Utrunc = (self.Qx[i] * self.sigma[i]) @ self.Qv[i].T
            err = np.linalg.norm(Utrunc - U)
            if self.debug: print(f"error truncation = {err}")

            if self.debug:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(self.X, self.V, U, cmap=cm.coolwarm,
                                        linewidth=0, antialiased=False)
                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.show()

        return

    def get_rank(self):
        return len(self.sigma[0]) * len(self.sigma[1])

    def get_ranks(self):
        return len(self.sigma[0]), len(self.sigma[1])

    def eval_factorS(self, i):
        r0, r1 = self.get_ranks()
        i0 = i % r0
        i1 = i // r0

        z = self.sigma[0][i0] * self.sigma[1][i1] 
        return z

    def eval_factorX(self, i, x):
        r0, r1 = self.get_ranks()
        i0 = i % r0
        i1 = i // r0

        z =   np.interp(x[0], self.x, self.Qx[0][:, i0]) \
            * np.interp(x[1], self.x, self.Qx[1][:, i1])
        return z

    def eval_factorV(self, i, v):
        r0, r1 = self.get_ranks()
        i0 = i % r0
        i1 = i // r0

        z =   np.interp(v[0], self.v, self.Qv[0][:, i0]) \
            * np.interp(v[1], self.v, self.Qv[1][:, i1])
        return z


if __name__ == "__main__":
    x0 = [-0.5, 0.0]
    sigmax = 0.5
    v0 = [1.0, 0.0]
    sigmav = 0.5
    E = [0.0, 0.0]
    N = 1000
    Lx = 1.0
    Nx = N
    Lv = 4.0
    Nv = N

    bs = BoltzmannSolution(x0, sigmax, v0, sigmav, E,
                           Lx, Nx, Lv, Nv, 100, 1e-4)
    bs.debug = True

    t = 0.25
    bs.set_time(t)
    r = bs.get_rank()
    print(f"rank = {r}")

    # test at random points
    errmax = 0.0
    for i in range(100):
        x = np.random.uniform(-Lx, Lx, 2)
        v = np.random.uniform(-Lv, Lv, 2)
        z = 0.0
        for i in range(r):
            z += bs.eval_factorX(i, x) * bs.eval_factorS(i) \
                * bs.eval_factorV(i, v)

        zref = bs.u(t, x, v)
        err = abs(z-zref)
        errmax = max(err, errmax)
        # print(z, zref, err)

    print(f"max error = {errmax : 10.2e}")
