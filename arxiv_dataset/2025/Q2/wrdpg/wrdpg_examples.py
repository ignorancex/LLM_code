'''
Created on Jul 28, 2023

@author: bernardo
'''

from graspologic.embed import AdjacencySpectralEmbed as ASE
from graspologic.simulations import er_np, sbm
from scipy.stats import chi2, norm

import matplotlib.pyplot as plt
import numpy as np
from utils.embedding_utils import align_Xs
from utils.moments_utils import normal_moments, poisson_moments


plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['lines.markersize'] = 15
plt.rcParams['axes.grid'] = True

#np.random.seed(42)

######### ER + N(1, 1) for all nodes ###########
N = 1000
p = 0.5
mu = 1
sigma = 0.1
sigma_sq = np.square(sigma)

K = 6

# Normal distribution moments
moments_normal = normal_moments(mu,sigma,2*(K+1))
    
X_seq = []

# Compute theoretical latent positions
for moment in moments_normal[1:K+1]:
    X_seq.append(np.sqrt(p*moment))
    
w_norm = np.random.normal
w_norm_args = dict(loc = mu, scale = sigma)

er_graph = er_np(N, p, wt=w_norm, wtargs=w_norm_args)

ase = ASE(n_components=1, diag_aug=True, algorithm='truncated')

Xhats = []

# Embed A^(k)
for k in np.arange(1,K+1):
    Xhat = ase.fit_transform(np.power(er_graph,k))
    if np.mean(Xhat) < 0:
        Xhat = -Xhat 
    Xhats.append(Xhat)

fig, axs = plt.subplots(figsize=(12,7),nrows=2, ncols=3, layout='constrained')
axs = axs.flatten()

for idx, ax in enumerate(axs):
    ax.hist(Xhats[idx], bins=20, density=True, rwidth=0.8)
    ax.axvline(X_seq[idx], color = 'maroon', linestyle = '--')
    ax.set_title(f'$\hat{{\mathbf{{X}}}}[{idx+1}]$')
    # Plot limiting gaussians
    sigma_sq_lim = (p*moments_normal[2*(idx+1)] - np.square(p*moments_normal[idx+1]))/(p*moments_normal[idx+1])
    sigma_sq_lim = sigma_sq_lim/N
    sigma_lim = np.sqrt(sigma_sq_lim)
    x = np.linspace(X_seq[idx] - 4*sigma_lim, X_seq[idx] + 4*sigma_lim, 100)
    ax.plot(x, norm.pdf(x, X_seq[idx], sigma_lim),linestyle='dashed', color='black')
    

######### 2-block SBM + N(1, 1) for all nodes ###########
K = 6
N = 1000
ratio = 0.7
n = [int(N*ratio),int(N*(1-ratio))]

p_c1 = 0.7
p_c2 = 0.3
q = 0.1

mu = 1
sigma = 0.1
sigma_sq = np.square(sigma)

P = [[p_c1, q],
     [q, p_c2]]


# Normal distribution moments
moments_normal = normal_moments(mu,sigma,2*(K+1))

X_seq = []

# Compute theoretical latent positions
for moment in moments_normal[1:K+1]:
    X_c1 = np.array([np.sqrt(p_c1*moment),0])
    X_c2 = np.array([q*np.sqrt(moment/p_c1),np.sqrt(moment*(p_c2-np.square(q)/p_c1))])
    X_seq.append(np.vstack((X_c1,X_c2)))

w_norm_args = dict(loc = mu, scale = sigma)
sbm_graph = sbm(n=n, p=P, wt=w_norm, wtargs=w_norm_args)

ase = ASE(n_components=2, diag_aug=True, algorithm='truncated')

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

color_list = ['maroon', 'royalblue']

fig, axs = plt.subplots(figsize=(10,16),nrows=2, ncols=3, layout='constrained')
axs = axs.flatten() 
colors = n[0]*[color_list[0]] + n[1]*[color_list[1]]

for idx, ax in enumerate(axs):
    ax.scatter(Xhats_sbm[idx][:,0],Xhats_sbm[idx][:,1],c=colors,alpha=0.1)
    ax.scatter(X_seq[idx][:,0],X_seq[idx][:,1],c='black',marker='x')
    ax.set_title(f'$\hat{{\mathbf{{X}}}}[{idx+1}]$')

# Plot 95% confidence levels for limiting gaussians
conf_level = 0.95

# Parametrize the unit circle
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.stack((np.cos(theta), np.sin(theta)))  # shape (2, N)

# Probabilities of belonging to each sbm block
pi_1 = ratio
pi_2 = 1-pi_1

# Compute the squared Mahalanobis radius (quantile of chi-squared with 2 dof)
r2 = chi2.ppf(conf_level, df=2)
r = np.sqrt(r2)

for idx,X in enumerate(X_seq):
    means = [X_seq[idx][0,:],X_seq[idx][1,:]]
    
    outer_block1 = np.outer(means[0],means[0])
    outer_block2 = np.outer(means[1],means[1])

    delta_k = pi_1*outer_block1 + pi_2*outer_block2
    delta_inv = np.linalg.inv(delta_k)

    E_k_normal = moments_normal[idx+1]
    E_2k_normal = moments_normal[2*(idx+1)]
        
    cov_1 = pi_1*outer_block1*(p_c1*E_2k_normal-np.square(p_c1*E_k_normal)) + pi_2*outer_block2*(q*E_2k_normal-np.square(q*E_k_normal))
    cov_1 = delta_inv@cov_1@delta_inv
        
    cov_2 = pi_1*outer_block1*(q*E_2k_normal-np.square(q*E_k_normal)) + pi_2*outer_block2*(p_c2*E_2k_normal-np.square(p_c2*E_k_normal)) 
    cov_2 = delta_inv@cov_2@delta_inv
            
    cov_1 = cov_1/N
    cov_2 = cov_2/N
    covs = [cov_1,cov_2]
    
    for comm, (mean,cov) in enumerate(zip(means,covs)):
        # Cholesky decomposition for transforming the unit circle
        L = np.linalg.cholesky(cov)
        ellipsoid = mean[:, None] + r * L @ circle
        
        axs[idx].plot(ellipsoid[0], ellipsoid[1], color='black', linestyle='dashed')


######### 2-block SBM + N(5, 0.1) or Poisson(5) ###########
mu = 5
sigma = 0.1
sigma_sq = np.square(sigma)
lam = mu + 0.1

N = 2000
ratio = 0.5
n = [int(N*ratio),int(N*(1-ratio))]
p = 0.5

P = [[p, p],
     [p, p]]

wt = [[np.random.normal, np.random.normal],
      [np.random.normal, np.random.poisson]]

w_norm_args = dict(loc = mu, scale = sigma)
w_poisson_args = dict(lam=lam)

wtargs = [[w_norm_args, w_norm_args],
           [w_norm_args, w_poisson_args]]


sbm_graph = sbm(n=n, p=P, wt=wt, wtargs=wtargs)

K = 3

# Normal distribution moments
moments_normal = normal_moments(mu, sigma, 2*(K+1))
# Poisson distribution moments
moments_poisson = poisson_moments(lam, 2*(K+1))

X_seq = []

# Compute theoretical latent positions
for moment_normal, moment_poisson in zip(moments_normal[1:K+1],moments_poisson[1:K+1]):
    X_c1 = np.array([np.sqrt(p*moment_normal),0])
    X_c2 = np.array([np.sqrt(p*moment_normal),np.sqrt(p*(moment_poisson-moment_normal))])
    X_seq.append(np.vstack((X_c1,X_c2)))
    
Xhats_sbm = []

ase = ASE(n_components=2, diag_aug=True, algorithm='truncated')

for k in np.arange(1,K+1):
    # Embed A^(k)
    Xhat = ase.fit_transform(np.power(sbm_graph,k))
    # Align with theoretical latent positions
    X = np.empty_like(Xhat)
    X[:n[0]] = X_seq[k-1][0,:]
    X[n[0]:] = X_seq[k-1][1,:]
    rotated_Xhat = align_Xs(Xhat, X)
    Xhats_sbm.append(rotated_Xhat)

fig, axs = plt.subplots(figsize=(18,6),nrows=1, ncols=3, layout='constrained')
axs = axs.flatten() 
colors = n[0]*[color_list[0]] + n[1]*[color_list[1]]

for idx, ax in enumerate(axs):
    ax.scatter(Xhats_sbm[idx][:,0],Xhats_sbm[idx][:,1],c=colors,alpha=0.3)
    ax.scatter(X_seq[idx][:,0],X_seq[idx][:,1],c='black',marker='x')
    ax.set_title(f'$\hat{{\mathbf{{X}}}}[{idx+1}]$')


# Probabilities of belonging to each sbm block
pi_1 = ratio
pi_2 = 1-pi_1

for idx,X in enumerate(X_seq):
    means = [X_seq[idx][0,:],X_seq[idx][1,:]]
    
    outer_block1 = np.outer(means[0],means[0])
    outer_block2 = np.outer(means[1],means[1])

    delta_k = pi_1*outer_block1 + pi_2*outer_block2
    delta_inv = np.linalg.inv(delta_k)
    
    E_k_normal = moments_normal[idx+1]
    E_2k_normal = moments_normal[2*(idx+1)]
    
    cov1 =(p*E_2k_normal-np.square(p*E_k_normal))*(pi_1*outer_block1+pi_2*outer_block2)
    
    cov1 = delta_inv@cov1@delta_inv 
    
    E_k_poisson = moments_poisson[idx+1]
    E_2k_poisson = moments_poisson[2*(idx+1)]
    
    cov2 = pi_1*(p*E_2k_normal-np.square(p*E_k_normal))*outer_block1
    cov2 += pi_2*(p*E_2k_poisson-np.square(p*E_k_poisson))*outer_block2
    
    cov2 = delta_inv@cov2@delta_inv
    
    cov1 = cov1/N
    cov2 = cov2/N
    
    covs = [cov1,cov2]
    
    for comm, (mean,cov) in enumerate(zip(means,covs)):
        # Cholesky decomposition for transforming the unit circle
        L = np.linalg.cholesky(cov)
        ellipsoid = mean[:, None] + r * L @ circle
        
        axs[idx].plot(ellipsoid[0], ellipsoid[1], color='black', linestyle='dashed')
        
plt.show()