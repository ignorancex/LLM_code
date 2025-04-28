# Computes autocorrelation time as a function
# of linear size of square Ising lattice

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import importlib
import emcee
# import ising_analysis
# importlib.reload(ising_analysis)

''' ------------------------- Define Functions (Start) ------------------------- '''

def autocorrelation(data):
    '''Computes normalized autocorrelation function of sample data for each time'''
    N = data.shape[0]
    _autocorrelation = np.zeros(N)
    for Δt in range(N-1): # let the time separation be all possible distances
        c0 = np.mean(data[:N - Δt]**2) - np.mean(data[:N - Δt])**2 #Variance at t0
        ct = np.mean(data[:N - Δt]*data[Δt:]) - np.mean(data[:N - Δt])*np.mean(data[Δt:]) # unnormalized autocorrelation fn.
        _autocorrelation[Δt] = ct/c0 # normalized autocorrelation function for this 'radius' (actually time separation)
    return _autocorrelation

def autocorrelation_function(time,scale,autocorrelation_time):
    '''exponential form of the autocorrelation function'''
    return scale*np.exp(-time/autocorrelation_time)

''' ------------------------- Define Functions (End) ------------------------- '''

# Set how much initial samples to discard
throwaway = 0

U = 3.3
t = 1.0
L = 12
N = L
l = L//2
num_replicas = 1
estimator = "energies"

# Open file for writing
filename = "/home/ecasiano/Desktop/papers-code-truncExponSampling/processed_data/1D_%d_%d_%d_%.6f_%.6f_betas_%d_%s_square_autocorrs_truncated.dat"%(L,N,L//2,U,t,num_replicas,estimator)
file = open(filename, "w")
need_header = True

for beta in [2.0,4.0,8.0,16.0,32.0]:
    print("β = %.2f"%beta)
    # Set physical parameters
    # N = L
    # l = L/2
    # beta = L/2
    Uot = U/t
    
    if need_header:
        header = "#L=%d, N=%d, l=%d, U/t=%.6f, num_replicas=%d \n# betas    tau_K         tau_K_err          tau_V          tau_V_err \n"%(L,N,l,Uot,num_replicas)
        file.write(header)
        need_header = False

    seeds = list(range(80))
    # seeds.remove(20)
    # seeds.remove(21)
    # seeds.remove(22)
    # seeds.remove(24)

#     # Open file for writing
#     filename = "/home/ecasiano/Desktop/papers-code-truncExponSampling/ProcessedData/1D_%d_%d_%d_%.6f_%.6f_%.6f_%d_%s_seeds_square_autocorrs_truncated.dat"%(L,N,l,U,t,beta,num_replicas,estimator)
#     file = open(filename, "w")
#     header = "#L=%d, N=%d, l=%d, U/t=%.6f, beta=%.6f, num_replicas=%d \n# seed    tau_K          tau_V \n"%(L,N,l,Uot,beta,num_replicas)
#     file.write(header)

    tau_Ks = np.zeros_like(seeds)
    tau_Vs = np.zeros_like(seeds)

    for i,seed in enumerate(seeds):
        # Load data 
        estimator = "K"
        data_correlated_K = np.loadtxt("/home/ecasiano/Desktop/TruncExponData/data_truncated/1D_%d_%d_%d_%.6f_%.6f_%.6f_%d_%s_%d_square.dat"%(L,N,l,U,t,beta,num_replicas,estimator,seed))
        K_data = data_correlated_K[throwaway:]

        estimator = "V"
        data_correlated_V = np.loadtxt("/home/ecasiano/Desktop/TruncExponData/data_truncated/1D_%d_%d_%d_%.6f_%.6f_%.6f_%d_%s_%d_square.dat"%(L,N,l,U,t,beta,num_replicas,estimator,seed))
        V_data = data_correlated_V[throwaway:]

        tau_K = emcee.autocorr.integrated_time(K_data)
        tau_V = emcee.autocorr.integrated_time(V_data)

        tau_Ks[i] = tau_K
        tau_Vs[i] = tau_V

#         file.write('%d %.8f %.8f\n'%(seed,tau_K,tau_V))

        print("seed: {}, τ_K: {}, τ_V: {}".format(seed,tau_K,tau_V)) # from fitting

    tau_K_mean = np.mean(tau_Ks)
    tau_K_err = np.std(tau_Ks, ddof=1) / np.sqrt(np.size(tau_Ks))

    tau_V_mean = np.mean(tau_Vs)
    tau_V_err = np.std(tau_Vs, ddof=1) / np.sqrt(np.size(tau_Vs))

    print("Average τ_K: %.4f +/- %.4f"%(tau_K_mean,tau_K_err))
    print("Average τ_V: %.4f +/- %.4f"%(tau_V_mean,tau_V_err))
    
    file.write('%.6f %.8f %.8f %.8f %.8f\n'%(beta,tau_K_mean,tau_K_err,tau_V_mean,tau_V_err))
    
# Close file if finished sampling
file.close()
