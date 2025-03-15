# this script takes the shortest distance matrix and returns the negative probability matrix
# different choices of transformations include linear, min-linear, max-linear, and alpha-linear transformation
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import DataLoader

def calculate_probability_linear(shrtst_dist):
    probability_norm = np.zeros_like(shrtst_dist)
    for row_indx in range(probability_norm.shape[0]):
        sum_row_dist = np.sum(shrtst_dist[row_indx])
        probability_norm[row_indx] = shrtst_dist[row_indx] / sum_row_dist
    return probability_norm

def min_unilinear_transform(shrtst_dist, device):
    probability_norm = calculate_probability_linear(shrtst_dist)
    all_one = np.ones_like(probability_norm)
    all_one_prob = all_one/all_one.sum(axis=1)
    res_probability = np.minimum(all_one_prob, probability_norm)
    np.fill_diagonal(res_probability, 0)
    probability_norm_tensor = torch.tensor(res_probability).float().to(device)
    return probability_norm_tensor

def max_unilinear_transform(shrtst_dist, device):
    probability_norm = calculate_probability_linear(shrtst_dist)
    all_one = np.ones_like(probability_norm)
    all_one_prob = all_one/all_one.sum(axis=1)
    res_probability = np.maximum(all_one_prob, probability_norm)
    np.fill_diagonal(res_probability, 0)
    probability_norm_tensor = torch.tensor(res_probability).float().to(device)
    return probability_norm_tensor

def linear_transform(shrtst_dist, device):
    probability_norm = calculate_probability_linear(shrtst_dist)
    probability_norm_tensor = torch.tensor(probability_norm).float().to(device)
    return probability_norm_tensor

def alpha_linear_transform(shrtst_dist, alpha, device):
    pow_shrtst_dist = np.power(shrtst_dist, alpha)
    probability_norm = calculate_probability_linear(pow_shrtst_dist)
    probability_norm_tensor = torch.tensor(probability_norm).float().to(device)
    return probability_norm_tensor