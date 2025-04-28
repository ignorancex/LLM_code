import matplotlib.pyplot as plt
import numpy as np
from mcmc import mcmc
from tqdm import tqdm
from utils import rand
from with_numba import gauss_hermite


def R(x, x_):
    return -x * np.exp(x_) / (1 + np.exp(x_)) + np.log(np.exp(x) + 1)


class AdaGrad:
    def __init__(self, kind, G=1, b_0=0.5, e_var=0, seed=None):
        self.G = G
        self.b_0 = b_0
        self.e_var = e_var

        if (kind not in ["least squares", "logistic regression"]):
            raise ValueError(
                "Type must be either 'least squares' or 'logistic regression'")

        self.kind = "".join([i[0] for i in kind.split(" ")])
        if (seed is not None):
            np.random.seed(seed)

    def get_losses(self, eigs, x_=None, T=3):
        d = len(eigs)
        cov = np.array(eigs)

        def noise():
            return self.e_var * np.random.normal()

        if x_ is None:
            x_ = rand(d)
            x_ = x_ / np.sqrt((x_ @ x_))

        x = np.zeros(d)

        def grad_f_lr(x):
            a = rand(d, cov)
            ex = np.exp(a @ x)
            ex_ = np.exp(a @ x_ + noise())
            return a * (ex / (1 + ex) - ex_ / (1 + ex_))

        def step_lr(x, b_2):
            g = 1 / d * grad_f_lr(x)
            b_2 = b_2 + np.linalg.norm(g) ** 2
            x = x - self.G / (np.sqrt(b_2)) * g
            return x, b_2, self.G / (np.sqrt(b_2))

        def loss_lr(x):
            B = np.array([[(x * cov) @ x, (x * cov) @ x_],
                          [(x * cov) @ x_, (x_ * cov) @ x_]])
            return gauss_hermite(B, "R_lr") - gauss_hermite(B, "R_lr_entropy")

        def grad_f_ls(x):
            a = rand(d, cov)
            b = a @ x_ + noise()
            return a * (a @ x - b)

        b_2 = self.b_0**2

        def step_ls(x, b_2):
            g = 1/d * grad_f_ls(x)
            b_2 = b_2 + np.linalg.norm(g)**2
            x = x - self.G/(np.sqrt(b_2)) * g
            return x, b_2, self.G/(np.sqrt(b_2))

        def loss_ls(x):
            return 0.5 * (((x - x_) * cov) @ (x - x_) + self.e_var**2)

        losses = []
        stepsizes = []
        # interval = round(d/16)
        interval = 1

        loss = loss_lr if self.kind == "lr" else loss_ls
        step = step_lr if self.kind == "lr" else step_ls

        for i in tqdm(range(T * d)):
            x, b_2, stepsize = step(x, b_2)
            if i % interval == 0:
                losses.append(loss(x))
                stepsizes.append(stepsize)

        return losses, stepsizes


if __name__ == "__main__":
    e = 0
    G = 1
    b_0 = 0.5
    sgd = AdaGrad(G, b_0, e)

    T = 100
    d = 2000

    plt.figure(figsize=(12, 6))

    def dist(x): return x**(-0.25)
    eigs = mcmc(d, dist, 0, 1)

    losses, steps = sgd.get_losses(eigs, T=T)
    x = np.linspace(0, T, len(losses))

    plt.subplot(1, 2, 1)
    plt.loglog(
        x, losses, label=f"Loss, e={e}")
    plt.title("Losses")

    plt.subplot(1, 2, 2)
    plt.loglog(
        x, steps, label=f"Stepsize, e={e}")
    plt.title("Stepsizes")

    plt.suptitle(r"No noise, eigs ~ $x^{-1/4}$")
    plt.savefig("plots/sgd.png", dpi=1000)
    plt.show()
