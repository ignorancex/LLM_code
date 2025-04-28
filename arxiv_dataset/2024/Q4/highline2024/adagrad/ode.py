import matplotlib.pyplot as plt
import numpy as np
from mcmc import mcmc
from tqdm import tqdm
from with_numba import gauss_hermite


class AdaGradODE:
    def __init__(self, kind, G=1, b_0=1, dt=1/256, e_var=0, seed=None):
        self.G = G
        self.b_0 = b_0
        self.dt = dt
        self.e_var = e_var
        if (kind not in ["least squares", "logistic regression"]):
            raise ValueError(
                "Type must be either 'least squares' or 'logistic regression'")
        self.kind = "".join([i[0] for i in kind.split(" ")])

        if (seed is not None):
            np.random.seed(seed)

    def get_losses(self, eigs, X_, T=3):
        eigs = np.array(eigs)
        d = len(eigs)
        losses = []
        steps = []
        running_sum = 0

        S_11 = np.zeros(d)
        S_12 = np.zeros(d)
        S_22 = X_ * eigs * X_
        S = np.array(
            [[S_11, S_12],
             [S_12, S_22]]
        )
        n_steps = int(T / self.dt)
        B = np.einsum("ijk->ij", S)
        losses.append(self.R(B))
        running_sum += 1/d * np.sum(eigs) * self.I(B)
        for _ in tqdm(range(n_steps)):

            integral = self.dt * running_sum
            step = self.G / np.sqrt(self.b_0**2 + integral)
            steps.append(step)

            S_eigs = np.einsum("ijk,k->ijk", S, eigs)

            dS = -2 * step * (np.einsum('ij,jlk->ilk', self.H(B).T, S_eigs) +
                              np.einsum("ijk,jl->ilk", S_eigs, self.H(B)))

            dS[0, 0] += (step**2/d) * eigs**2 * self.I(B)

            S += dS * self.dt

            B = np.einsum("ijk->ij", S)
            loss = self.R(B) - self.R_entropy(B)
            running_sum += 1/d * np.sum(eigs) * self.I(B)
            losses.append(loss)

        return losses, steps

    def R(self, B):
        match self.kind:
            case "lr":
                return gauss_hermite(B, which="R_"+self.kind)
            case "ls":
                return 0.5 * (B[0, 0] + B[1, 1] - B[0, 1] - B[1, 0] + self.e_var)

    def R_entropy(self, B):
        match self.kind:
            case "lr":
                return gauss_hermite(B, which="R_lr_entropy")
            case "ls":
                return 0

    def I(self, B):
        match self.kind:
            case "lr":
                return gauss_hermite(B, which="I_"+self.kind)
            case "ls":
                return (B[0, 0] + B[1, 1] - B[0, 1] - B[1, 0] + self.e_var)

    def H(self, B, h=1e-7):
        m, n = B.shape
        gradient = np.zeros((m, n))

        E1 = np.zeros((m, n))
        E1[0, 0] = h

        R_minus = self.R(B)
        R_plus = self.R(B+E1)

        gradient[0, 0] = (R_plus - R_minus) / (h)

        E2 = np.zeros((m, n))
        E2[1, 0] = h
        E2[0, 1] = h

        R_minus = self.R(B)
        R_plus = self.R(B+E2)

        gradient[1, 0] = (R_plus - R_minus) / (2 * h)
        return gradient

    def plot(self, losses, label, scaling=None):
        x = np.linspace(0, int(len(losses) * self.dt), len(losses))
        if scaling is None:
            plt.plot(x, losses, label=label)
        elif scaling == "loglog":
            plt.loglog(x, losses, label=label)
        else:
            raise ValueError("Scaling must be either None or 'loglog'")


if __name__ == "__main__":

    # def dist(x): return x**(-0.99)

    G = 1
    T = 10**5
    b_0 = 2
    ode = AdaGradODE("least squares", G=G, b_0=b_0, dt=1e-2)

    plt.rcParams['agg.path.chunksize'] = 101

    d = 100
    e_var = 0
    X_ = np.random.normal(size=d)
    X_ = X_ / np.sqrt((X_ @ X_))
    eigs = np.ones(d)
    ode.e_var = e_var
    losses, steps = ode.get_losses(eigs, X_, T=T)

    ode.plot(losses, label="Volterra", scaling="loglog")

    plt.title("g'(t) vs Volterra")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/ode.png", dpi=1000)
    plt.show()
