import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import config_gk
import config_ABC
import functions as f
import sys

def gk_barnes(param, s_sim, s_obs, delta, p, seed):
    np.random.seed(seed)
    nStats = s_obs.shape[1]

    gamma_vector = np.zeros(nStats,).astype('bool')
    queried_idx = []  # Initializing an empty list that will contain the indices of queried statistics

    k = 0
    d = delta + 1
    samples_current = param
    d_current = gaussian_kde(samples_current.T)
    resamples_current = d_current.resample(size = nRS).T

    while (k < nStats and d > delta):
        #print('Iteration no.' + str(k + 1))
        kld = np.zeros(nStats,)
        for j in range(nStats):
            if j not in queried_idx:  # Skipping the utility computation if the jth statistic has already been queried
                gamma_vector[j] = 1
                samples_future = f.regression_ABC(s_obs, param, s_sim, p, gamma_vector)
                d_future = gaussian_kde(samples_future.T)
                resamples_future = d_future.resample(size = nRS).T
                gamma_vector[j] = 0
                kld[j] = f.KLdivergence(resamples_future, resamples_current)
        
        d = kld.max()
        if d > delta: # Stopping criterion
            queried_idx.append(np.argmax(kld))
            gamma_vector[np.argmax(kld)] = 1
            samples_current = f.regression_ABC(s_obs, param, s_sim, p, gamma_vector)
            d_current = gaussian_kde(samples_current.T)
            resamples_current = d_current.resample(size = nRS).T
        #else:
            #print('We can stop \n')
        k = k + 1

    gamma_hat = gamma_vector.astype('int')
    print('Generating final ABC results...')
    samples_final = f.regression_ABC(s_obs, param, s_sim, p, gamma_hat.astype('bool'))

    return k-1, samples_final


if __name__ == "__main__":
    nR = config_gk.nR
    nRS = config_ABC.nGSamples
    delta = config_ABC.delta_barnes
    p = config_ABC.p
    
    nSim = int(sys.argv[1])
    
    print('Running ' + str(nR) + ' replications of the g-and-k experiment with Barnes with ' + str(nSim) + ' simulations.')
    print('----------')
    
    param = np.load('gk_param.npy') # Parameters
    s_sim = np.load('gk_ssim.npy')  # Simulated sum. stats.
    s_obs = np.load('gk_sobs.npy')  # Observed sum. stats.
    nStats = s_obs.shape[1]
    nParam = param.shape[2]
    
    k_tab = np.zeros(nR,)
    samples_tab = np.zeros((nR, int(nSim*p), nParam))
    
    for i in range(0,nR):
        print('Replication n.' + str(i+1))
        k_tab[i], samples_tab[i,:,:] = gk_barnes(param[i, :nSim, :], s_sim[i, :nSim, :], s_obs[[i], :], delta, p, i+1)
        print('----------')

    np.save('k_Barnes' + str(nSim), k_tab)
    np.save('samples_Barnes' + str(nSim), samples_tab)