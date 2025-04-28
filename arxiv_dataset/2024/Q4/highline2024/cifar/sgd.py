import pickle

import matplotlib.pyplot as plt
import numpy as np
from real_data import get_real_data
from tqdm import tqdm
from volterra import Volterra
import os


class SGD:
    def __init__(self, F, b, K):
        self.F = F
        self.b = b
        self.K = K
        self.n, self.d = F.shape

    def loss(self, w):
        return 1/(2 * self.n) * np.linalg.norm(self.F @ w - self.b)**2

    def grad(self, w):
        return 1/self.n * (self.F.T @ (self.F @ w - self.b))

    def stochastic_grad(self, w):
        i = np.random.randint(self.n)
        return self.F[i] * (self.F[i] @ w - self.b[i])

    def get_losses(self, w_0, n_iters, grad_func='stochastic', eta=0.1, b_0=1):
        losses = []
        w = w_0
        integral = 0
        dt = 1/self.d
        sample = 1000

        grad_function = self.stochastic_grad if grad_func == 'stochastic' else self.grad

        for t in tqdm(range(n_iters)):
            gamma_t = eta/np.sqrt(b_0**2 + integral)
            gamma_t /= self.d
            g = grad_function(w)
            w = w - gamma_t * g
            if t % sample == 0:
                l = self.loss(w)
                losses.append((t/self.d, l))
                integral += np.linalg.norm(g)**2 * dt**2 * sample
        return w, losses

    def save_results(self, results, filename):
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

def run(SAMPLE_SIZE):
    F, b, K = get_real_data(SAMPLE_SIZE)

    sgd = SGD(F, b, K)
    volterra = Volterra(F, b, K)

    w_0 = np.zeros(sgd.d)
    t_max = 10**3
    n_steps_volterra = 10**3
    eta = 5
    b_0 = 0.1

    w, losses = sgd.get_losses(w_0, t_max * sgd.d, eta=eta, b_0=b_0)
    losses = [(t, l) for t, l in losses if t >= 1]
    times, loss_values = zip(*losses)

    t, psi = volterra.solve_psi(t_max, n_steps_volterra, w_0, eta=eta, b_0=b_0)
    psi = psi[t >= 1]
    t = t[t >= 1]

    # Save results
    sgd.save_results((times, loss_values, t, psi), f'cache/sgd_{sgd.n}.pkl')

def plot_all():
    plt.rcParams['font.size'] = 18

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    
    for i, n in enumerate([2**11, 2**12, 2**13, 2**14]):
        with open(f'cache/sgd_{n}.pkl', 'rb') as f:
            times, losses, t, psi = pickle.load(f)
        
        plt.loglog(times, losses, color=colors[i], linestyle='--', alpha=0.5)
        plt.loglog(t, psi, color=colors[i], label=f'n = {n}')
    
    plt.legend()
    plt.xlim(left=1e0)
    plt.ylim(bottom=2e-2, top=1e-1)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.xlabel("SGD Iterations/d")
    plt.ylabel("Empirical Risk")
    plt.title("CIFAR AdaGrad-Norm Least Squares")

    plt.tight_layout()
    # make sure the directory exists
    os.makedirs("figures/cifar", exist_ok=True)
    plt.savefig("figures/cifar/cifar_adagrad_all_d.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    sizes = [2**11, 2**12, 2**13, 2**14]
    for n in sizes:
        if os.path.exists(f'cache/sgd_{n}.pkl'):
            print(f'Skipping {n} because it already exists')
            continue
        print(f'Running for {n}')
        run(n)
    plot_all()
