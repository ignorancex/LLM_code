import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import config_gk
import config_ABC
import functions as f
import sys
import time

def gk_hitl(param, s_sim, s_obs, rho, pi, delta, nGSamples, p, gamma_true, criterion, seed):
    np.random.seed(seed)
    
    nStats = s_obs.shape[1]
    nParam = param.shape[1]

    # Computing prior predictive probability of f
    predProb_f1 = rho * pi + (1 - rho) * (1 - pi)
    predProb_f0 = 1 - predProb_f1

    queried_idx = []  # Initializing an empty list that will contain the indices of queried statistics
    feedbacks = []  # Initializing an empy list that will contain the feedbacks

    k = 0
    d = delta + 1

    while (k < nStats and d > delta):
        print('Iteration no.' + str(k + 1))
        # Initializing vector of utilities
        utility = np.zeros(nStats)
        # Sample from p_ABC(theta | y, F)
        gamma_vector_current = f.sample_gamma(queried_idx, feedbacks, nGSamples, nStats, pi, rho)
        samples_current = f.sample_ABC_Posterior(s_obs, param, s_sim, p, gamma_vector_current)

        for j in range(nStats):
            if j not in queried_idx:  # Skipping the utility computation if the jth statistic has already been queried
                # Sample from p_ABC(theta | y, F, fj = 1)
                gamma_vector_new1 = f.sample_gamma(queried_idx + [j], feedbacks + [1], nGSamples, nStats, pi, rho)
                samples_future_f1 = f.sample_ABC_Posterior(s_obs, param, s_sim, p, gamma_vector_new1)
                # Sample from p_ABC(theta | y, F, fj = 0)
                gamma_vector_new0 = f.sample_gamma(queried_idx + [j], feedbacks + [0], nGSamples, nStats, pi, rho)
                samples_future_f0 = f.sample_ABC_Posterior(s_obs, param, s_sim, p, gamma_vector_new0)
                # Computing KL divergence
                KL_estimate_f1 = f.KLdivergence(samples_future_f1, samples_current)
                KL_estimate_f0 = f.KLdivergence(samples_future_f0, samples_current)
                # Computing utility
                utility[j] = predProb_f1 * KL_estimate_f1 + predProb_f0 * KL_estimate_f0

        d = utility.max()
        if d > delta: # Stopping criterion
            if criterion == 'normal':
                print('The queried statistic is no.' + str(np.argmax(utility) + 1))
                queried_idx.append(np.argmax(utility))
                feedbacks.append(f.generate_feedback(gamma_true, pi, np.argmax(utility)))
                print('Obtained feedback: ' + str(feedbacks[-1]) + '\n')
            elif criterion == 'random':
                idx_tmp = list(np.arange(0,nStats))
                for z in range(0,len(queried_idx)):
                    idx_tmp.remove(queried_idx[z])
                idx = np.random.choice(idx_tmp)
                print('The queried statistic is no.' + str(idx + 1))
                queried_idx.append(idx)
                feedbacks.append(f.generate_feedback(gamma_true, pi, idx))
                print('Obtained feedback: ' + str(feedbacks[-1]) + '\n')
        else:
            print('We do not need to query another feedback from the expert! \n')
        k = k + 1

    gamma_hat = np.zeros(nStats)
    gamma_hat[queried_idx] = feedbacks
    print('Generating final ABC results...')
    samples_final = f.regression_ABC(s_obs, param, s_sim, p, gamma_hat.astype('bool'))

    return k-1, samples_final


if __name__ == "__main__":
    gamma_true = config_gk.gamma_true
    nR = config_gk.nR

    rho = config_ABC.rho
    pi = config_ABC.pi
    delta = config_ABC.delta
    nGSamples = config_ABC.nGSamples
    p = config_ABC.p
    
    nSim = int(sys.argv[1])
    criterion = sys.argv[2]
    
    print('Running ' + str(nR) + ' replications of the HITL g-and-k experiment with ' + str(nSim) + ' simulations.')
    print('Criterion = ' + criterion)
    print('----------')

    param = np.load('gk_param.npy') # Parameters
    nParam = param.shape[2]
    s_sim = np.load('gk_ssim.npy')  # Simulated sum. stats.
    s_obs = np.load('gk_sobs.npy')  # Observed sum. stats.
    
    k_tab = np.zeros(nR,)
    samples_tab = np.zeros((nR, int(nSim*p), nParam))
    
    for i in range(0,nR):
        print('Replication n.' + str(i+1))
        k_tab[i], samples_tab[i,:,:] = gk_hitl(param[i, :nSim, :], s_sim[i, :nSim, :], s_obs[[i], :], rho, pi, delta, nGSamples, p, gamma_true, criterion, i+1)
        print('----------')

    np.save('k_HITL_' + criterion + '_' + str(nSim), k_tab)
    np.save('samples_HITL_' + criterion + '_' + str(nSim), samples_tab)