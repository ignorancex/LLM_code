import torch
import numpy as np
import scipy


def dict_to_cpu(sample):
    for key, value in sample.items():
        sample[key] = value.cpu()
    return sample


def dict_to_cuda(sample):
    for key, value in sample.items():
        sample[key] = value.cuda()
    return sample


def dict_to_double(sample):
    for key, value in sample.items():
        sample[key] = value.double()
    return sample


def dict_to_float(sample):
    for key, value in sample.items():
        sample[key] = value.float()
    return sample


### FOR MLE

def logL(z, data, simulator, ):
    model = simulator.get_mu(torch.from_numpy(z).unsqueeze(0)).numpy().astype(np.float64)[0]
    logL = - (len(data)/2 * np.log(2 * np.pi) + len(data)/2 * np.log(simulator.sigma ** 2) + 1 /(2 * simulator.sigma ** 2) * sum((data - model) ** 2))
    return logL

def mle(data, simulator):
    res = scipy.optimize.minimize(
        fun = lambda z: - logL(z, data, simulator),
        x0 = np.array([0, 0, 0, 0]),
    )
    return res.x

def best_fit(data, simulator):
    best_z = mle(data, simulator)
    best_fit = simulator.get_mu(torch.from_numpy(best_z).unsqueeze(0))
    return best_fit


# FOR SNR

def get_sigma_epsilon_inv2(n, sigma=1):
    sigma2_inv = 1.0 / (sigma ** 2)
    return ((n * n).sum(dim=-1) * sigma2_inv)  # Shape: [N_is]

def get_epsilon(delta_x, n, sigma=1):
    sigma2_inv = 1.0 / (sigma ** 2)
    numerator = torch.matmul(delta_x * sigma2_inv, n.T) # Shape: [N_x, N_is]
    denominator = ((n * n).sum(dim=-1) * sigma2_inv).unsqueeze(0)  # Shape: [1, N_is]
    epsilon_x = numerator / denominator  # Shape: [N_x, N_is]
    return epsilon_x

def get_snr(delta_x, n, sigma=1):
    sigma2_inv = 1.0 / (sigma ** 2)
    numerator = torch.matmul(delta_x * sigma2_inv, n.T) # Shape: [N_x, N_is]
    denominator = torch.sqrt((n * n).sum(dim=-1) * sigma2_inv).unsqueeze(0) # Shape: [1, N_is]
    SNR = numerator / denominator # Shape: [N_x, N_is]
    return SNR
    
def get_b(max_snr, n, sigma=1):
    sigma2_inv = 1.0 / (sigma ** 2)
    return max_snr / torch.sqrt((n * n).sum(dim=-1) * sigma2_inv)