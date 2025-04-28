import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Volterra:
    def __init__(self, F, b, K):
        self.eigvals_global, self.eigvecs_global = np.linalg.eigh(K)

        n, d = F.shape
        self.d = d  # Store d as instance variable
        self.n = n
        self.F = F
        self.b = b

        if n >= d:
            self.beta_star = np.linalg.inv(F.T @ F) @ (F.T @ b)
        else:
            self.beta_star = b

    def loss(self, w):
        return 1/(2 * self.n) * np.linalg.norm(self.F @ w - self.b)**2

    def gradient_flow(self, w_0, t):
        diff = self.eigvecs_global.T @ (w_0 - self.beta_star)
        expm_tK = self.eigvecs_global @ np.diag(
            np.exp(-t * self.eigvals_global))
        return self.beta_star + expm_tK @ diff

    def ker(self, t, Gamma_t, gamma_t):
        exponentials = np.exp(-2 * self.eigvals_global * Gamma_t)
        terms = self.eigvals_global**2 * exponentials
        terms = self.eigvecs_global @ np.diag(terms) @ self.eigvecs_global.T
        result = gamma_t**2 * np.trace(terms) / self.d
        return result

    def ker2(self, gamma_s, delta):
        exponentials = np.exp(-2 * self.eigvals_global * delta)
        terms = self.eigvals_global**2 * exponentials
        result = gamma_s**2 * np.sum(terms) / self.d
        return result

    def solve_psi(self, t_max, n_steps, w_0, eta=0.1, b_0=1):
        # Create logarithmic and linear spacing
        t = np.linspace(0, t_max, n_steps)
        n_steps = len(t)

        # Initialize arrays
        psi = np.zeros(n_steps)
        psi[0] = self.loss(w_0)
        result = np.zeros(n_steps)

        Gamma_t = np.zeros(n_steps)
        gamma_t = np.zeros(n_steps)
        X_t = np.zeros((n_steps, self.d))
        L_X_t = np.zeros(n_steps)
        L_X_t[0] = psi[0]
        kernel_values = np.zeros(n_steps)
        integral = 0
        integral_factor = 2 * np.sum(self.eigvals_global)/self.d
        gamma_t[0] = eta/b_0

        # Solve the Volterra equation using convolution
        for i in tqdm(range(1, n_steps)):
            dt = t[i] - t[i-1]
            gamma_t[i] = eta/np.sqrt(b_0**2 + integral_factor * integral)
            Gamma_t[i] = Gamma_t[i-1] + gamma_t[i] * dt
            X_t[i] = self.gradient_flow(w_0, Gamma_t[i])
            L_X_t[i] = self.loss(X_t[i])
            kernel_values = np.zeros(i+1)

            for j in range(i, -1, -1):
                delta = Gamma_t[i] - Gamma_t[j]
                kernel_values[j] = self.ker2(gamma_t[j], delta)
                if kernel_values[j] < kernel_values[i]/100:
                    break

            dt_vec = t[1:i+1] - t[:i]
            convolution = np.sum(kernel_values[:i] * psi[:i] * dt_vec)
            psi[i] = L_X_t[i] + convolution
            integral += psi[i] * dt
            result[i] = psi[i]

        return t, result
