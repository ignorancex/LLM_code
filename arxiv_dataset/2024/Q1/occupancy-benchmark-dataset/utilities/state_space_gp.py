import numpy as np
import math
from scipy import special, linalg


def MaternKernel(nu, lengthScale, magnitude):
    D = nu + 0.5
    if nu < 0:
        raise ValueError("nu must be positive")
    if D != round(D):
        raise ValueError("nu must be multiples of n*1/2")
    D = int(D)

    lmd = math.sqrt(2 * nu) / lengthScale  # lambda
    a = [-special.binom(D, i) * lmd ** (D - i) for i in range(D)]
    F = np.block([[np.zeros([D-1, 1]), np.eye(D-1)], a])
    L = np.block([[np.zeros([D-1, 1])], [1]])
    if D == 1:
        H = np.array([[1]])
    else:
        H = np.array([[1], [0]*(D-1)])
    Qc = magnitude ** 2 * math.factorial(D-1) ** 2 / math.factorial(2*D-2) * (2 * lmd) ** (2 * D - 1)
    Pinf = linalg.solve_lyapunov(F, -Qc*L@L.T)
    return F, Pinf, L, H, Qc


def PeriodicKernel(J, lengthScale, period, magnitude):
    qj2 = [2 * magnitude**2 * special.ive(j, lengthScale**(-2)) for j in range(J+1)]
    qj2[0] = qj2[0]/2

    omega = 2*math.pi/period
    F = np.kron(np.diag(range(J+1)), np.array([[0, -1], [1, 0]])*omega)
    Pinf = np.kron(np.diag(qj2), np.eye(2))
    L = np.eye(2*J)
    H = np.kron(np.ones([1, J+1]), [1, 0])
    Qc = np.zeros(2*(J+1))
    return F, Pinf, L, H, Qc


def QuasiPeriodicKernel(J, nu, lengthScale_p, period, magnitude_p, lengthScale_q, magnitude_q):
    Fp, Pinfp, Lp, Hp, Qcp = PeriodicKernel(J, lengthScale_p, period, magnitude_p)
    Fq, Pinfq, Lq, Hq, Qcq = MaternKernel(nu, lengthScale_q, magnitude_q)

    F = np.kron(Fp, np.eye(Fq.shape[0])) + np.kron(np.eye(Fp.shape[0]), Fq)
    Qc = np.kron(Pinfp, Qcq)
    Pinf = np.kron(Pinfp, Pinfq)
    L = np.kron(Lp, Lq)
    H = np.kron(Hp, Hq)
    return F, Pinf, L, H, Qc
