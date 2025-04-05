"""
This module generates all plots for the different examples
"""
###############################################################################
# preample
import numpy as np
import matplotlib.pyplot as plt 
from qubitmlmodel import qubitMLmodel
from utilities import Pauli_operators, generate_noise
from scipy.signal import periodogram
###############################################################################
# define simulation parameters
T                 = 1
M                 = 1024
K                 = 2000
Omega             = 12                                # qubit energy gap
static_operators  = [0.5*Pauli_operators[3]*Omega]    # drift Hamiltonian
dynamic_operators = [0.5*Pauli_operators[1]]          # control Hamiltonian 
noise_operators   = [0.5*Pauli_operators[3]]          # noise Hamiltonian
initial_states    = [
                     np.array([[0.5,0.5],[0.5,0.5]]),
                     np.array([[0.5,-0.5j],[0.5j,0.5]]),
                     np.array([[1,0],[0,0]])
                    ]                                 # intial state of qubit 
measurement_operators = Pauli_operators[1:2]          # measurement operators
n_max             = 5                                 # number of pulses per sequence
time_range        = [(0.5*T/M) + (j*T/M) for j in range(M)] 
# define dataset names
models            = ["G_1q_X_Z_N%d"%idx for idx in [0,1,2,3,4,5]]
###############################################################################
# Generate and analyze different noise profiles
for profile in [0,1,2,3,4,5]:
    beta = generate_noise(T=T, M=M, K=K, profile=profile)
    # plot noise
    if profile in [2,3,4]:
        # estimate the correlation matrix of the noise
        correlation = 0
        for k in range(K):
            correlation = correlation + beta[0,:,k:k+1,0]@beta[0,:,k:k+1,0].T
        correlation = correlation/K
        # plot correlation matrix
        plt.figure()
        plt.matshow(correlation,0)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.colorbar()
    elif profile in [1,5]:
        # estimate the PSD of the noise
        psd = 0
        for k in range(K):
            f, Pxx = periodogram(beta[0,:,k,0], M/T)            
            psd = psd + Pxx
        psd = psd/K
        plt.figure()
        plt.plot(f[f>0], psd[1:])
        plt.xlabel('f',fontsize=11)
        plt.ylabel('psd',fontsize=11)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid()
    
    plt.savefig("Noise%d.pdf"%profile, bbox_inches='tight')
###############################################################################
# define the qubit ML models
mlmodels   = [qubitMLmodel(T, M, dynamic_operators, static_operators, noise_operators, measurement_operators, initial_states, K=4) for _ in models]

# load the pretrained models and plot the MSE
for idx_m , m in enumerate(models):
    mlmodels[idx_m].load_model("%s"%m)
    plt.figure()
    plt.loglog(mlmodels[idx_m].training_history, label='training')
    plt.loglog(mlmodels[idx_m].val_history, label='testing')
    plt.grid()
    plt.xlabel('iteration',fontsize=11)
    plt.ylabel('MSE',fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11)
    plt.savefig("MSE%s.pdf"%m, bbox_inches='tight')
###############################################################################
for idx_ex in [0,1,2]:  
    #plot the optimal pulses
    optimal_pulse = np.load("Example%d_pulse.npy"%idx_ex)
    plt.figure()
    plt.plot(time_range, optimal_pulse[0,:,0,0])
    plt.xlabel('t',fontsize=11)
    plt.ylabel(r'$f_x(t)$',fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid()
    plt.savefig("pulse%d.pdf"%idx_ex, bbox_inches='tight')
###############################################################################
# plot the confuction matrices
for idx_ex in [0,1,2]:
    confusion = np.load("confusion%d.npy"%idx_ex)
    for idx in range(5):
        confusion[idx,:] = 100*confusion[idx,:]/sum(confusion[idx,:]) 
    plt.figure(figsize=[1, 1])
    plt.matshow(confusion)
    plt.xlabel('Predicted Noise Profile', fontsize=11)
    plt.ylabel('True Noise Profile', fontsize=11)
    plt.colorbar()
    x,y = np.meshgrid([0,1,2,3,4],[0,1,2,3,4])
    for xx,yy in zip(x.flatten(), y.flatten()):
        plt.text(xx, yy, "%.2f"%(confusion[yy, xx]),va='center', ha='center')
    if idx_ex==1:
        plt.xticks([0,1,2,3,4],[5,1,2,3,4], fontsize=11)
        plt.yticks([0,1,2,3,4],[5,1,2,3,4], fontsize=11)
    else:
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
    plt.savefig("confusion%d.pdf"%idx_ex, format='pdf', bbox_inches='tight')
