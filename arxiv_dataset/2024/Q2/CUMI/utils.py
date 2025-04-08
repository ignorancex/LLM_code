import torch
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Union
from torch.utils.data import Dataset
import scipy.io as io
import os


def pairwise_distances(x):
    bn = x.shape[0]
    x = x.view(bn, -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01#1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    return Ixy

def joint_entropy_tc(variables,sigmas):
    alpha = 1.01
    k = 1.0
    for variable, sigma in zip(variables, sigmas):
        variable = calculate_gram_mat(variable, sigma)
        k *= variable
    k /= torch.trace(k)
    eigv = torch.linalg.eigvalsh(k)
    eigv = torch.abs(eigv)
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy
    # k = 1.0
    # for (variable,sigma) in zip(variables,sigmas):
    #     variable = calculate_gram_mat(variable,sigma)
    #     k = torch.mul(k,variable)
    # k = k/torch.trace(k)
    # eigv,_ = torch.linalg.eigh(k)
    # eigv = torch.abs(eigv)
    # eig_pow =  eigv**alpha
    # entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    # return entropy

def calculate_TC(variables,sigmas,normlize = True):
    H_MTE = torch.tensor([reyi_entropy(variable, sigma=sigma) for variable, sigma in zip(variables, sigmas)])
    H = torch.sum(H_MTE)
    Hxy = joint_entropy_tc(variables, sigmas)
    TCxy = H - Hxy
    if normlize:
        return TCxy / torch.max(H_MTE)
    else:
        return TCxy
    # H_MTE = []
    # H = 0.0
    # for (variable,sigma) in zip(variables,sigmas):
    #     H_MTE.append( reyi_entropy(variable,sigma=sigma))
    # H_max = max(H_MTE)
    # H = np.array(H_MTE).sum()
    # H = torch.tensor(H)
    # Hxy = joint_entropy_tc(variables,sigmas)
    # TCxy = H-Hxy
    # if normlize:
    #     return TCxy/H_max
    # else:
    #     return TCxy

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)  # Python built-in random number generator
    np.random.seed(seed)  # Numpy's random number generator
    torch.manual_seed(seed)  # PyTorch's random number generator
    torch.cuda.manual_seed(seed)  # Random number generator for GPU when used
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility, but may decrease speed
    torch.backends.cudnn.benchmark = False  # Turn off automatic algorithm search for stability in experiments



def get_kernelsize(features: torch.Tensor, selected_param: Union[int, float]=0.15, select_type: str='meadian'):
    ### estimating kernelsize with data with the rule-of-thumb
    features = torch.flatten(features, 1).cpu().detach().numpy()
    k_features = squareform(pdist(features))
    if select_type=='min':
        kernelsize = np.sort(k_features, 1)[:, :int(selected_param)].mean()
    elif select_type=='max':
        kernelsize = np.sort(k_features, 1)[:, int(selected_param):].mean()
    elif select_type=='meadian':
        triu_indices = np.triu_indices(k_features.shape[0], 1)
        kernelsize = selected_param*np.median(k_features[triu_indices])
    else:
        kernelsize = 1.0
    # if kernelsize<EPSILON:
    #     kernelsize = torch.tensor(EPSILON, device=features.device)
    return kernelsize


