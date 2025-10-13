import torch
import numpy as np
from numpy.polynomial.legendre import leggauss
from utils import *
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")


def phi(n, x):
    return np.sin(n * np.pi * x)

def dphi(n, x):
    return n * np.pi * np.cos(n * np.pi * x)

def coeff_2_vec(coeff, x):
    """
    Convert a list of coefficients to a vector evaluated at points x.
    """
    res = np.zeros_like(x)
    for i, c in enumerate(coeff):
        res += c * phi(i+1, x)
    return res
### compute matrix A for -Delta u + V u #########
def get_L(d, type='lognormal', alp1=1, beta1=1, num_quad=64):
    L = np.zeros((d, d))
    if type == 'lognormal':
        coeff_b = np.array([((n * np.pi) ** 2 + alp1) ** (-beta1 / 2) * np.random.randn()
                            for n in range(1, d + 1)])

        nodes, weights = leggauss(num_quad)
        # Transform nodes to [0, 1]: x = 0.5*(node + 1) and weights scale by 0.5.
        x_quad = 0.5 * (nodes + 1)
        w_quad = 0.5 * weights

        a_func = np.exp(coeff_2_vec(coeff_b, x_quad))

        for j in range(d):
            for k in range(d):
                # First term: I1 = ∫_0^1 a(x)* dphi(k,x)*dphi(j,x) dx
                integrand1 = a_func * dphi(k+1, x_quad) * dphi(j+1, x_quad)
                I1 = np.sum(w_quad * integrand1)
                # Second term: I2 = ∫_0^1 V(x)* sin(jπx)*sin(kπx) dx
                L[j, k] = I1

    elif type == 'const':
        for i in range(d):
            L[i, i] = ((i + 1) ** 2 * np.pi ** 2)

    elif type == 'const2':
        for i in range(d):
            L[i, i] = ((i + 1) ** 2 * np.pi ** 2 * 1e-3)
    elif type == 'const3':
        for i in range(d):
            L[i, i] = ((i + 1) ** 2 * np.pi ** 2 * 1e-1)
    return L

def random_rotation_mat(d):
    A = np.random.randn(d, d)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def generate_random(d, a=1, b=2):
    rd_vec = np.random.uniform(a, b, size=(d,))
    D = np.diag(rd_vec)
    Q = random_rotation_mat(d)
    Q_T = Q.T
    A = Q@D@Q_T
    return A


### V ~ piecewise constant U[1,2]
def compute_integral(i, j, x_start, x_end):
    """
    Computes int_{x_start}^{x_end} sin(iπx) sin(jπx) dx analytically.
    """
    if i == j:
        term1 = (x_end - x_start) / 2
        term2 = (np.sin(2 * i * np.pi * x_end) - np.sin(2 * i * np.pi * x_start)) / (4 * i * np.pi)
        return term1 - term2
    else:
        term1 = (np.sin((i - j) * np.pi * x_end) - np.sin((i - j) * np.pi * x_start)) / (2 * (i - j) * np.pi)
        term2 = (np.sin((i + j) * np.pi * x_end) - np.sin((i + j) * np.pi * x_start)) / (2 * (i + j) * np.pi)
        return term1 - term2

def generate_V_const(dv, a=1, b=2):
    N_coarse = dv
    V1 = np.random.uniform(a, b, size=(N_coarse,))
    return V1

def get_V_coeff(d, type='lognormal', num_quad=64, alpha=1, beta=1, dv=2, a=1, b=2):
    if type == 'lognormal':
        nodes, weights = leggauss(num_quad)
        # Transform nodes to [0, 1]: x = 0.5*(node + 1) and weights scale by 0.5.
        x_quad = 0.5 * (nodes + 1)
        w_quad = 0.5 * weights

        coeff_h = np.array([((n * np.pi) ** 2 + alpha) ** (-beta / 2) * np.random.randn()
                            for n in range(1, d + 1)])

        V_func = np.exp(coeff_2_vec(coeff_h, x_quad))


        V = np.zeros((d, d))

        for j in range(d):
            for k in range(d):
                # First term: I1 = ∫_0^1 a(x)* dphi(k,x)*dphi(j,x) dx
                integrand1 = V_func * phi(k+1, x_quad) * phi(j+1, x_quad)
                I1 = np.sum(w_quad * integrand1)
                # Second term: I2 = ∫_0^1 V(x)* sin(jπx)*sin(kπx) dx
                V[j, k] = I1

    elif type == 'lognormal2':
        nodes, weights = leggauss(num_quad)
        # Transform nodes to [0, 1]: x = 0.5*(node + 1) and weights scale by 0.5.
        x_quad = 0.5 * (nodes + 1)
        w_quad = 0.5 * weights

        coeff_h = np.array([((n * np.pi) ** 2 + alpha) ** (-beta / 2) * np.random.randn()
                            for n in range(1, d + 1)])

        V_func = np.exp(7*coeff_2_vec(coeff_h, x_quad))


        V = np.zeros((d, d))

        for j in range(d):
            for k in range(d):
                # First term: I1 = ∫_0^1 a(x)* dphi(k,x)*dphi(j,x) dx
                integrand1 = V_func * phi(k+1, x_quad) * phi(j+1, x_quad)
                I1 = np.sum(w_quad * integrand1)
                # Second term: I2 = ∫_0^1 V(x)* sin(jπx)*sin(kπx) dx
                V[j, k] = I1

    elif type == 'lognormal3':
        nodes, weights = leggauss(num_quad)
        # Transform nodes to [0, 1]: x = 0.5*(node + 1) and weights scale by 0.5.
        x_quad = 0.5 * (nodes + 1)
        w_quad = 0.5 * weights

        coeff_h = np.array([((n * np.pi) ** 2 + alpha) ** (-beta / 2) * np.random.randn()
                            for n in range(1, d + 1)])

        V_func = np.exp(2*coeff_2_vec(coeff_h, x_quad))


        V = np.zeros((d, d))

        for j in range(d):
            for k in range(d):
                # First term: I1 = ∫_0^1 a(x)* dphi(k,x)*dphi(j,x) dx
                integrand1 = V_func * phi(k+1, x_quad) * phi(j+1, x_quad)
                I1 = np.sum(w_quad * integrand1)
                # Second term: I2 = ∫_0^1 V(x)* sin(jπx)*sin(kπx) dx
                V[j, k] = I1

    elif type == 'piecewise':
        V_vec = generate_V_const(dv, a, b)
        # print(V_vec.shape)
        # zxc
        n_intervals = dv
        interval_length = 1.0 / n_intervals
        intervals = [(i * interval_length, (i + 1) * interval_length) for i in range(n_intervals)]

        V = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                integral_total = 0.0
                for k in range(n_intervals):
                    x_start, x_end = intervals[k]
                    V_k = V_vec[k]
                    integral = compute_integral(i + 1, j + 1, x_start, x_end)
                    integral_total += V_k * integral
                V[i, j] = integral_total

    elif type == 'random':
        V = generate_random(d, a, b)
    elif type == 'r_diagonal':
        V = np.random.uniform(1, 2, size=(1,)) * np.eye(d)
    return V


def generate_A_one(d, l1=1, l2=1, Ltype='lognormal', Vtype='lognormal', num_quad=64, alpha1=1, beta1=1, alpha2=1, beta2=1, dv=2, a=1, b=2):
    V = get_V_coeff(d, Vtype, num_quad, alpha2, beta2, dv, a, b)
    L = get_L(d, Ltype, alpha1, beta1)
    return l1*L+V*l2


def generate_A_inv(N, d, l1=1, l2=1, Ltype='lognormal', Vtype='lognormal', num_quad=64, alpha1=1, beta1=0.6, alpha2=1, beta2=1, dv=2, a=1, b=2):
    A = torch.zeros((N, d, d), device=device)
    for i in tqdm(range(N)):
        tmp_A =generate_A_one(d, l1, l2, Ltype, Vtype, num_quad, alpha1, beta1, alpha2, beta2, dv, a, b)
        tmp_A = np.linalg.inv(tmp_A)
        A[i, :, :] = torch.tensor(tmp_A, dtype=torch.float32, device=device)

    return A

def get_f(N, d, alp3=2, beta3=2, rho=None):
    if rho is None:
        l = torch.arange(1, d + 1)  # vector: [1, 2, ..., d]
        factors = ((l * torch.pi) ** 2 + alp3) ** (-beta3 / 2)
        random_samples = torch.randn(N, d)  # shape (N, d)
        res = (factors * random_samples).to(device)
    else:
        Sigma = equicorrelated_covariance(d, rho)
        mean = np.zeros(d)
        random_samples = np.random.multivariate_normal(mean, Sigma, size=N)
        res = torch.tensor(random_samples, dtype=torch.float32, device=device)
        print(rho)
    return res


def get_f_mat_eqv(N, n, d, rho=0):
    Sigma = equicorrelated_covariance(d, rho)
    mean = np.zeros(d)
    random_samples = np.random.multivariate_normal(mean, Sigma, size=N*n)
    return random_samples.reshape(N, n, d)

def generate_Yn(N, n, d, type='white_noise', alpha3=1, beta3=1, rho=0):
    if type == 'white_noise':
        x = torch.randn(N, n, d, device=device)  # Shape: (N, n, d)
        Yn = torch.bmm(x.transpose(1, 2), x) / n  # Shape: (N, d, d)
    elif type == 'lognormal':
        l = np.arange(1, d + 1)  # vector: [1, 2, ..., d]
        factors = ((l * np.pi) ** 2 + alpha3) ** (-beta3 / 2)
        random_samples = np.random.randn(N, n, d)  # shape (N, n, d)
        f = factors * random_samples
        Yn = np.matmul(f.transpose(0, 2, 1), f) / n
        Yn_tt = Yn.copy()
        n = np.arange(1, d + 1)
        theo_cov_diag = ((n * np.pi) ** 2 + alpha3) ** (-beta3)
        theo_cov = np.diag(theo_cov_diag)
        for i in tqdm(range(N)):
            Yn[i, ...] = theo_cov
        Yn = torch.tensor(Yn, dtype=torch.float32, device=device)
    elif type == 'equicorrelated':
        x = get_f_mat_eqv(N, n, d, rho)
        Yn = np.matmul(x.transpose(0, 2, 1), x) / n
        Yn = torch.tensor(Yn, dtype=torch.float32, device=device)

    return Yn

def generate_Yn_small(N, n, d, batch=100):
    num_batches = N // batch
    Yn = torch.zeros(N, d, d, device=device)
    for i in range(num_batches):
        Yn[i*batch:(i+1)*batch, ...] = generate_Yn(batch, n, d)
    return Yn

def generate_Ym(N, n, d, alp3, beta3):
    l = np.arange(1, d + 1)  # vector: [1, 2, ..., d]
    factors = ((l * np.pi) ** 2 + alp3) ** (-beta3 / 2)
    random_samples = np.random.randn(N, n, d)  # shape (N, n, d)
    f = factors * random_samples
    #Yn = np.matmul(f.transpose(0, 2, 1), f) / n  # Shape: (N, d, d)
    Yn = np.matmul(f.transpose(0, 2, 1), f) / n
    Yn = torch.tensor(Yn, dtype=torch.float32, device=device)
    return Yn

def generate_Ym_small(N, n, d, alp3, beta3, batch=100):
    num_batches = N // batch
    Yn = torch.zeros(N, d, d, device=device)
    for i in range(num_batches):
        Yn[i * batch:(i + 1) * batch, ...] = generate_Ym(batch, n, d, alp3, beta3)
    return Yn


def equicorrelated_covariance(d, rho):
    I = np.eye(d)
    J = np.ones((d, d))
    Sigma = (1 - rho) * I + rho * J
    return Sigma