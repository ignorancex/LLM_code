'''
Created on Jun 27, 2024

@author: bernardo
'''
from scipy.stats import probplot
from tqdm import tqdm

import  matplotlib.pyplot as plt
import maxent
import numpy as np
import pymaxent as pmnt
from utils.moments_utils import exponential_moments


# plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.markersize'] = 15


# Exponential distribution parameters
scale = 1/2
lam = 1/scale

# Maximum moment order
K = 3

# Exponential distribution moments 
moments_exp = exponential_moments(scale, K)
# Lagrange multipliers for exponential distribution
lambda_exp = np.zeros_like(moments_exp)
lambda_exp[0] = -np.log(lam)
lambda_exp[1] = lam

# Support for exponential distribution
bnds=[0,10]


n_reps = 100

lambdas_gd = np.zeros((n_reps,K+1))
lambdas_pmnt = np.zeros((n_reps,K+1))



for i in tqdm(range(n_reps),desc='Estimating exponential distribution', total=n_reps):
    # Random starting point
    lambguess = np.random.rand(len(lambda_exp))
    
    # Reconstruct using gradient descent on dual space
    lambdas_gd_exp = maxent.solve_dual_continuous_log(moments_exp, Omega=bnds, lambdas0=lambguess)
    lambdas_gd[i,:] = lambdas_gd_exp
    
    # Reconstruct using PyMaxEnt
    _, lambdas_pmnt_exp = pmnt.reconstruct(moments_exp, bnds=bnds, mix_density=False, lambguess=lambguess)
    lambdas_pmnt[i,:] = lambdas_pmnt_exp
    

fig, ax = plt.subplots(figsize=(18,9),nrows=1, ncols=1, layout="constrained")
moments_idxs = np.arange(K+1)
tick_labels = [f'$\lambda_{i}$' for i in moments_idxs]

medianprops = dict(linewidth=3, color='darkgreen')
boxprops = dict(linewidth=3, color='darkgreen')
flierprops = dict(marker='o', markerfacecolor='None', markeredgecolor='darkgreen')
whiskerprops = dict(linewidth=3, color='darkgreen',alpha=0.8)
capprops = whiskerprops

ax.scatter(moments_idxs,lambda_exp, c='royalblue', label='True $\lambda_i$', marker='x', s=250)

ax.boxplot(lambdas_gd,tick_labels=tick_labels, positions=moments_idxs, label='Gradient descent',
               medianprops = dict(linewidth=3, color='maroon'), whis=(5,95))

ax.boxplot(lambdas_pmnt,tick_labels=tick_labels, positions=moments_idxs, label='PyMaxEnt',
               medianprops=medianprops, boxprops=boxprops, flierprops=flierprops, whis=(5,95),
               whiskerprops=whiskerprops, capprops=capprops)

ax.set_ylabel('$\lambda_i$ values')
ax.set_xlabel('Lagrange multipliers')
ax.legend()

fig.suptitle('Exponential pdf estimation with maximum entropy')


exp_solution_rv = maxent.maximum_entropy_rv(lambdas=lambdas_gd_exp, momtype=1, a=bnds[0],b=bnds[1])
exp_solution_rv_pmnt = maxent.maximum_entropy_rv(lambdas=lambdas_pmnt_exp, momtype=1, a=bnds[0],b=bnds[1])

fig_rvs, axs_rvs= plt.subplots(figsize=(10,10),nrows=1, ncols=2, layout='constrained')

n_samples = 1000
probplot(exp_solution_rv.rvs(size=n_samples), sparams=(0,scale), dist='expon', plot=axs_rvs[0],rvalue=True)
probplot(exp_solution_rv_pmnt.rvs(size=n_samples), sparams=(0,scale), dist='expon', plot=axs_rvs[1],rvalue=True)
fig_rvs.suptitle('Estimation of exponential dist')
axs_rvs[0].set_title('GD')
axs_rvs[1].set_title('PyMaxEnt')


plt.show()
