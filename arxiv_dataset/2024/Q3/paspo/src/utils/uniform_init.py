from action_distributions.helper_distributions import SafeTanhTransform
from utils.lp_sub_proc import LPSubProc
import torch
from scipy.stats import beta, truncnorm, norm
import numpy as np
import polytope as pc
from scipy.optimize import fmin_slsqp, minimize, curve_fit

import torch

def sample_tanh_transformed_normal(mu, sigma, min_max, size=1):
    normal_samples = np.random.normal(mu, sigma, size)
    samples =  np.tanh(normal_samples)
    samples = (samples * ((min_max[:, 1] - min_max[:, 0])/2.0)) + ((min_max[:, 0] + min_max[:, 1])/2.0)
    
    return samples


    


def fit_tanh_squashed_gaussian(data, min_max):
    mu = data.mean()
    sigma = data.std()
    
    scale = (min_max[:, 1] - min_max[:, 0]) /2.0
    loc = (min_max[:, 1] + min_max[:, 0]) / 2.0

    x = (data - loc) / scale
    
    x = np.arctanh(x)
    
    #filter out nan and inf values
    x = x[np.isfinite(x)]
    
    mu_fitted, sigma_fitted = norm.fit(x)
    
    return mu_fitted, sigma_fitted
    



#TODO this does not work for tuncnorm currently
def uniform_params(A, b, n_dim, dist="beta", n_points=10000):
    if dist == "beta":
        dist = beta
    elif dist == "truncnorm":
        dist = truncnorm
    elif dist == "squashed_gaussian":
        pass
    else:
        raise ValueError("dist must be beta or truncnorm")
    
    
    seed = 1 #TODO do not hardcode the seed here for uniform_params init
    rng = np.random.default_rng(seed)

    uniform_points = []
    points_already_sampled = 0
    while points_already_sampled < n_points: 
        n_points_raw = 1000000 #1000000
        points = rng.dirichlet([1.0]*n_dim, size=n_points_raw)
        polytope = pc.Polytope(A, b, normalize=False)
        indices = polytope.contains(points.T)
        uniform_points_batch = points[indices]
        
        uniform_points.append(uniform_points_batch)
        actual_batch_size = len(uniform_points_batch)
        points_already_sampled += actual_batch_size
    
    uniform_points = np.concatenate(uniform_points, axis=0)
    uniform_points = uniform_points[:n_points]

    params = []

    #sample_0 = beta.rvs(a1, b1, loc1, scale1, size=1000)

    # sample_0.mean()
    # sample_0.std()

    # uniform_points[:, 0].mean()
    # uniform_points[:, 0].std()

    n_points = len(uniform_points)

    A = A[None, ...].repeat(n_points, axis=0)
    b = b[None, ...].repeat(n_points, axis=0)

    unallocated = np.ones((n_points,1), dtype=np.float32)

    lp_sub_proc = LPSubProc(8) #TODO do not hardcode the number of processes here


    for dim in range(n_dim-1):

        n = A.shape[-1]

        lp_sub_proc.send([(A[i], b[i], 0, n, unall.item(), 1e-8, 1e-8, True) for i, unall in enumerate(unallocated)])


        min_max = lp_sub_proc.recv(unallocated.shape[0])

        min_max = np.array(min_max)
        
        #find entry in the transformed data that are not between the minimum and maximum (due to numerical imprecisions)
        cond = (uniform_points[:, dim] < min_max[:, 0]) | (uniform_points[:, dim] > min_max[:, 1]) | (min_max[:, 0] < 0) | (min_max[:, 1] < 0)
        indices = np.where(cond)
        
        #remove the entries that are not between the minimum and maximum
        uniform_points = np.delete(uniform_points, indices, axis=0)
        
        #remove the corresponding entries in the unallocated array
        unallocated = np.delete(unallocated, indices, axis=0)
        
        #remove the corresponding entries in the A and b arrays
        A = np.delete(A, indices, axis=0)
        b = np.delete(b, indices, axis=0)
        
        #remove the corresponding entries in the min_max array
        min_max = np.delete(min_max, indices, axis=0)
        floc=min_max[:, 0]
        fscale=min_max[:, 1]-min_max[:, 0]
        
        data = (np.ravel(uniform_points[:, dim]) - floc) / fscale
        if np.any(data <= 0) or np.any(data >= 1):
            #delete the entries that are not between 0 and 1 due to numerical imprecisions
            indices = np.where((data <= 0) | (data >= 1))
            uniform_points = np.delete(uniform_points, indices, axis=0)
            unallocated = np.delete(unallocated, indices, axis=0)
            A = np.delete(A, indices, axis=0)
            b = np.delete(b, indices, axis=0)
            min_max = np.delete(min_max, indices, axis=0)

        if dist == beta:
            a2, b2, loc2, scale2 = dist.fit(uniform_points[:, dim], floc=min_max[:, 0], fscale=min_max[:, 1]-min_max[:, 0])
        elif dist == "squashed_gaussian":
            a2, b2 = fit_tanh_squashed_gaussian(uniform_points[:, dim], min_max)
        else:
            floc = min_max[:, 0]
            fscale = min_max[:, 1]-min_max[:, 0]
            
            #fit_truncnorm_params(uniform_points[:, dim], min_max[:, 0], min_max[:, 1])
            #optimize_truncnorm_params(uniform_points[:, dim], min_max[:, 0], min_max[:, 1])
            
            data = (np.ravel(uniform_points[:, dim]) - floc) / fscale
            
            
            a2, b2, loc2, scale2 = dist.fit(data, fa=0, fb=1)
                        

        params.append((a2, b2))

        #sample_1 = beta.rvs(a2, b2, loc2[:1000], scale2[:1000], size=1000)

        #print(sample_1.mean())
        #print(sample_1.std())

        #print(uniform_points[:, dim].mean())
        #print(uniform_points[:, dim].std())
        
        unallocated -= uniform_points[:, dim][:, None]


        b = b - (uniform_points[:, dim][:, None] * A[...,0])
        A = A[...,1:]


    #print(params)
    
    lp_sub_proc.killall()
    
    return params



def sample(params, A, b, n_dim, n_samples, dist="beta"):
    if dist == "beta":
        dist = beta
    elif dist == "truncnorm":
        dist = truncnorm
    elif dist == "squashed_gaussian":
        pass
    else:
        raise ValueError("dist must be beta or truncnorm")

    lp_sub_proc = LPSubProc(32)

    A = A[None, ...].repeat(n_samples, axis=0)
    b = b[None, ...].repeat(n_samples, axis=0)

    unallocated = np.ones((n_samples,1), dtype=np.float32)

    all_samples = []
    for dim in range(n_dim -1):
        
        n = A.shape[-1]

        lp_sub_proc.send([(A[i], b[i], 0, n, unall.item(), 1e-8, 1e-8) for i, unall in enumerate(unallocated)])


        min_max = lp_sub_proc.recv(unallocated.shape[0])

        min_max = np.array(min_max)

        if dist == "squashed_gaussian":
            samples = sample_tanh_transformed_normal(*params[dim], min_max, size=n_samples)
        else:
            samples = dist.rvs(*params[dim], min_max[:, 0], min_max[:, 1]-min_max[:, 0], size=n_samples)
        
        
        all_samples.append(samples)
        
        unallocated -= samples[:, None]


        b = b - (samples[:, None] * A[...,0])
        A = A[...,1:]


    samples = np.stack([*all_samples, unallocated[:, 0]], axis=-1)
    return samples


if __name__ == "__main__":
    n_dim = 3
    #from envs.synth_env import SynthEnv

    from utils.polytope_loader import load_polytope
    
    
    A, b = load_polytope(n_dim=n_dim, storage_method="seed", generation_method="points", polytope_generation_data={"n_points": 100, "seed": 1})
    
    #env = SynthEnv(n_dim, 2, A=A, b=b)
    
    dist = "beta"
    dist = "truncnorm"

    params = uniform_params(A, b, n_dim, dist, n_points=100)
    samples = sample(params, A, b, n_dim, 1000, dist)
    print(samples.mean(axis=0))
    print(samples.std(axis=0))
