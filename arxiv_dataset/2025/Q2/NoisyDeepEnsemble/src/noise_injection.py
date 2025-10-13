import torch
import random
import numpy as np
import copy

def make_uni_noise(weight, scale, ratio):
    w = weight.cpu().detach().numpy().copy()
    size = w.size
    o_shape = w.shape
    w = w.reshape(-1)
    p = np.random.choice(size, int(size*ratio), replace=False)
    
    for i in range(p.size):
        noise = random.uniform(-scale, scale)
        tmp = w[p[i]]
        w[p[i]] += noise
        if(tmp * w[p[i]] < 0):
            w[p[i]] *= -1

    w = w.reshape(o_shape)
    new_weight = torch.from_numpy(w).clone()
    return new_weight

def make_norm_noise(weight, scale, mean, ratio):
    w = weight.cpu().detach().numpy().copy()
    size = w.size
    o_shape = w.shape
    w = w.reshape(-1)
    p = np.random.choice(size, int(size*ratio), replace=False)
    
    for i in range(p.size):
        noise = random.normalvariate(mean, scale)
        tmp = w[p[i]]
        w[p[i]] += noise
        if(tmp * w[p[i]] < 0):
            w[p[i]] *= -1
    w = w.reshape(o_shape)
    new_weight = torch.from_numpy(w).clone()
    return new_weight

def uni_noise(model, scale, ratio):
    params = [(n, p) for n, p in model.named_parameters() if all(pattern not in n for pattern in ['bias', 'bn', 'downsample'])]
    for name, param in params:
        model.state_dict()[name] = make_uni_noise(param, scale, ratio)
    return model

def norm_noise(model, scale, mean, ratio):
    params = [(n, p) for n, p in model.named_parameters() if all(pattern not in n for pattern in ['bias', 'bn', 'downsample'])]
    for name, param in params:
        model.state_dict()[name] = make_norm_noise(param, scale, mean, ratio)
    return model

def inject_noise(model, perturbation, perturbation_strength, perturbation_ratio, perturbation_mean):
    if perturbation == "uniform":
        return uni_noise(model, perturbation_strength, perturbation_ratio)
    elif perturbation == "normal":
        return norm_noise(model, perturbation_strength, perturbation_mean, perturbation_ratio)
    else:
        raise ValueError("Invalid perturbation type")