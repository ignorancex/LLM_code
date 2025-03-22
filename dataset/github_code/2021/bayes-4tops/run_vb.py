# here i take the data and run one gibbs sampling procedure
# inputs are: data_dir output_dir Number of events considered Number of saved samples Burnin Space between samples

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, bernoulli, expon, uniform, beta, gamma, multinomial
from scipy.special import digamma
import random
from scipy.special import gamma as gamma_function
from scipy.special import gammaln
from scipy.special import factorial
from scipy.special import beta as beta_function
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import dirichlet
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import corner

import necessary_functions as nf

data_dir = sys.argv[1]
N=int(sys.argv[2])
Nprior=10
Ninit=10

data = np.loadtxt(data_dir+'/processed_data.dat')
data_smeared = np.loadtxt(data_dir+'/processed_data_smeared.dat')

labels=data[:,2]
f1=np.sum(labels==1)/len(labels)

ohe_nj=OneHotEncoder(handle_unknown='error')
ohe_nb=OneHotEncoder(handle_unknown='error')
Y1=ohe_nj.fit_transform(data[:,0].reshape(-1,1)).toarray()
Y2=ohe_nb.fit_transform(data[:,1].reshape(-1,1)).toarray()

X=[]
for n in range(Y1.shape[0]):
  X.append([Y1[n],Y2[n]])

true_alphas=np.zeros((2,Y1.shape[1]))
true_betas=np.zeros((2,Y2.shape[1]))
for k in range(2):
  true_alphas[k]=np.mean(Y1[labels==k],axis=0)
  true_betas[k]=np.mean(Y2[labels==k],axis=0)
  
Y1_smeared=ohe_nj.transform(data_smeared[:,0].reshape(-1,1)).toarray()
Y2_smeared=ohe_nb.transform(data_smeared[:,1].reshape(-1,1)).toarray()
fake_alphas=np.zeros((2,Y1.shape[1]))
fake_betas=np.zeros((2,Y2.shape[1]))
for k in range(2):
  fake_alphas[k]=np.mean(Y1_smeared[data_smeared[:,2]==k],axis=0)
  fake_betas[k]=np.mean(Y2_smeared[data_smeared[:,2]==k],axis=0)
  

K=true_alphas.shape[0]
dj=true_alphas.shape[1]
db=true_betas.shape[1]


eta_pie, eta_alpha, eta_beta =np.ones(2), np.where(Nprior*fake_alphas>=1.01,Nprior*fake_alphas,1.01), np.where(Nprior*fake_betas>=1.01,Nprior*fake_betas,1.01)
#eta_alpha, eta_beta = np.array([eta_alpha[k]*Nprior/np.sum(eta_alpha[k]) for k in range(K)], np.array([eta_beta[k]*Nprior/np.sum(eta_beta[k]) for k in range(K)]

gamma_pie_0, gamma_alpha_0, gamma_beta_0 = np.ones(2), Ninit*fake_alphas, Ninit*fake_betas

gamma_pie_VB, gamma_alpha_VB, gamma_beta_VB, rmatrix_VB, probs_VB = nf.do_VB_algorithm(N,2,200,0.0000001,gamma_pie_0,gamma_alpha_0,gamma_beta_0,eta_pie,eta_alpha,eta_beta,X)

last_index = np.argmin(gamma_pie_VB[:,0])-1

indexes=[i for i in range(200)]
plt.plot([i for i in range(201)],probs_VB, color='blue')
plt.axvline(indexes[last_index],linestyle='dashed',color='black')
plt.xlabel('Iterations')
plt.ylabel('ELBO')
plt.savefig(data_dir+'/elbo_vb.pdf')
plt.savefig(data_dir+'/elbo_vb.png')

np.savetxt(data_dir+'/rmatrix_VB.dat',rmatrix_VB[last_index-1])
np.savetxt(data_dir+'/gamma_pie_VB.dat',gamma_pie_VB[last_index])
np.savetxt(data_dir+'/gamma_alpha_VB.dat',gamma_alpha_VB[last_index])
np.savetxt(data_dir+'/gamma_beta_VB.dat',gamma_beta_VB[last_index])
