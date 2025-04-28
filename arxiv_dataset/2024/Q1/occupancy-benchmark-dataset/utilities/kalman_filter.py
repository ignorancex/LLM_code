import numpy as np
from scipy import linalg


class KalmanFilter:
    def __init__(self, dt: float, Rk, Qk, A, C, m0, P0, B=None, D=None):
        self.Ak = linalg.expm(A*dt)
        self.Bk = None if B is None else (self.Ak - np.eye(self.Ak.shape[0])) / np.array(A) @ np.array(B)
        self.Ck = np.array(C)
        self.Dk = None if D is None else np.array(D)
        if self.Ak.shape[0] == np.array(m0).shape[0]:
            self.m = np.array(m0)
        else:
            raise ValueError("m0 and A have incompatible dimensions")
        if self.Ak.shape == np.array(P0).shape:
            self.P = np.array(P0)
        else:
            raise ValueError("P0 and A have different shapes")
        self.Qk = Qk
        self.Rk = Rk

    def predict(self, u: np.ndarray = None):
        if u is None:
            self.m = self.Ak @ self.m
        else:
            self.m = self.Ak @ self.m + self.Bk @ u
        self.P = self.Ak @ self.P @ self.Ak.T + self.Qk

    def update(self, y: np.ndarray, u: np.ndarray = None):
        S = self.Ck @ self.P @ self.Ck.T + self.Rk
        K = self.P @ linalg.solve(S, self.Ck).T
        if u is None:
            self.m = self.m + K @ (y - self.Ck @ self.m)
        else:
            self.m = self.m + K @ (y - self.Ck @ self.m - self.Dk @ u)
        self.P = self.P - K @ S @ K.T

    def predict_loop(self, steps: int) -> np.ndarray:
        m = self.m
        y = np.zeros((steps, 1))
        y[0, :] = self.Ck @ m
        for i in range(1, steps):
            m = self.Ak @ m
            y[i, :] = self.Ck @ m

        return y.flatten()


class UnscentedKalmanFilter:
    def __init__(self, nx, ny, f, h, dt, m0, P0, Qk, Rk, alpha=1e-3, beta=2):
        self.fk = lambda xk, uk, p: xk + dt*np.array(f(xk, uk, p))
        self.hk = h

        self.nx = nx
        self.ny = ny
        self.m = np.array(m0.reshape((nx, 1)))
        self.P = P0
        self.Qk = Qk
        self.Rk = Rk
        self.Wm, self.Wc, self.lmd = self.calcWeights(nx, alpha, beta)

    @staticmethod
    def calcWeights(nx, alpha, beta):
        kappa = 3-nx
        lmd = alpha ** 2 * (nx + kappa) - nx
        Wm = np.zeros((2*nx+1, 1))
        Wc = np.zeros((2*nx+1, 1))
        Wm[0] = lmd / (nx + lmd)
        Wc[0] = lmd / ((nx + lmd) + (1 - alpha ** 2 + beta))

        for i in range(1, nx+1):
            Wm[i] = 1.0 / (2 * (nx + lmd))
            Wc[i] = 1.0 / (2 * (nx + lmd))
            Wm[i+nx] = 1.0 / (2 * (nx + lmd))
            Wc[i+nx] = 1.0 / (2 * (nx + lmd))

        return Wm, np.diag(Wc[:, 0]), lmd

    def calcSigmaPoints(self):
        R = linalg.cholesky(self.P, lower=True)
        return np.tile(self.m, (1, 2*self.nx+1)) + np.sqrt(self.nx + self.lmd) * np.hstack((np.zeros((self.nx, 1)), R, -R))

    def predict(self, u=None, p=None):
        s = self.calcSigmaPoints()
        X = np.zeros((self.nx, self.nx * 2 + 1))
        for i in range(0, 2*self.nx+1):
            X[:, i] = self.fk(s[:, i], u, p)

        self.m = X @ self.Wm
        self.P = (X - self.m) @ self.Wc @ np.transpose(X - self.m) + self.Qk

    def update(self, y):
        s = self.calcSigmaPoints()
        Y = np.zeros((self.ny, self.nx*2+1))
        for i in range(0, 2*self.nx+1):
            Y[:, i] = self.hk(s[:, i])

        muk = Y @ self.Wm
        Sk = (Y - muk) @ self.Wc @ np.transpose(Y - muk) + self.Rk
        Ck = (s - self.m) @ self.Wc @ np.transpose(Y - muk)

        Kk = linalg.solve(Sk, Ck.T).T
        self.m = self.m + Kk @ (np.reshape(y, (self.ny, 1)) - muk)
        self.P = self.P - Kk @ Sk @ Kk.T

