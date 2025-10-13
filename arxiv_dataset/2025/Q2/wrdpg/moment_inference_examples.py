'''
Created on Apr 21, 2025

@author: bernardo
'''

import numpy as np
from graspologic.simulations import sbm
from graspologic.embed import AdjacencySpectralEmbed as ASE
import matplotlib.pyplot as plt
from utils.embedding_utils import align_Xs
from utils.moments_utils import normal_moments

plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['lines.markersize'] = 15
plt.rcParams['axes.grid'] = True



######### 2-block SBM + N(1, 1) for all nodes ###########
ratio = 0.7

p_c1 = 0.7
p_c2 = 0.5
q = 0.3

mu = 1.0
sigma = 0.5

P = [[p_c1, q],
     [q, p_c2]]


# Normal distribution moments
K = 6

moments_normal = normal_moments(mu, sigma, K)
moments_seq = moments_normal[1:]


X_seq = []

# Compute theoretical latent positions
for idx,moment in enumerate(moments_seq):
    X_c1 = np.array([np.sqrt(p_c1*moment),0])
    X_c2 = np.array([q*np.sqrt(moment/p_c1),np.sqrt(moment*(p_c2-np.square(q)/p_c1))])
    X_seq.append(np.vstack((X_c1,X_c2)))

w_norm = np.random.normal
w_norm_args = dict(loc = mu, scale = sigma)

ase = ASE(n_components=2, diag_aug=True, algorithm='full')

N_list = [200, 2000]

for N in N_list:
    
    n = [int(N*ratio),int(N*(1-ratio))]
    # Sample graph from base model
    sbm_graph = sbm(n=n, p=P, wt=w_norm, wtargs=w_norm_args, directed=False)
    
    Xhats_sbm = []
    
    for k in np.arange(1,K+1):
        # Embed A^(k)
        Xhat = ase.fit_transform(np.power(sbm_graph,k))
        # Align with theoretical latent positions
        X = np.empty_like(Xhat)
        X[:n[0]] = X_seq[k-1][0,:]
        X[n[0]:] = X_seq[k-1][1,:]
        rotated_Xhat = align_Xs(Xhat, X)
        Xhats_sbm.append(rotated_Xhat)
        
    # Plot
    fig, axs = plt.subplots(figsize=(10,10),nrows=2, ncols=2, layout='constrained')
    colors = n[0]*['maroon'] + n[1]*['royalblue']
    
    which_k = [0,3]
    for i,k in enumerate(which_k):
        
        # Plot recovered latent positions
        axs[i,0].scatter(Xhats_sbm[k][:,0],Xhats_sbm[k][:,1],c=colors,alpha=0.1)
        
        # Plot reconstructed moments
        axs[i,1].hist((Xhats_sbm[k]@Xhats_sbm[k].T).flatten(), bins='auto', density=True, rwidth=0.8)
        axs[i,1].axvline(moments_seq[k]*p_c1, color = 'maroon', linestyle = '--')
        axs[i,1].axvline(moments_seq[k]*p_c2, color = 'maroon', linestyle = '--')
        axs[i,1].axvline(moments_seq[k]*q, color = 'black', linestyle = '--')
        
        axs[i,0].set_title(f'$\hat{{\mathbf{{X}}}}[{k+1}]$')
        axs[i,1].set_title(f'$\hat{{\mathbf{{M}}}}[{k+1}]$')
        
        if k==3:
            axs[i,0].set_xlim((0.24,2))
            axs[i,0].set_ylim((-1,1.8))
            axs[i,1].set_xlim((0,3))

plt.show()
