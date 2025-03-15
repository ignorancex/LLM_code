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
output_dir = sys.argv[2]
N=int(sys.argv[3])
T=int(sys.argv[4])
burnout=int(sys.argv[5])
keep_every=int(sys.argv[6])

data = np.loadtxt(data_dir+'/processed_data.dat')
data_smeared = np.loadtxt(data_dir+'/processed_data_smeared.dat')

labels=data[:,2]
f1=np.sum(labels==1)/len(labels)

ohe_nj=OneHotEncoder(handle_unknown='error')
ohe_nb=OneHotEncoder(handle_unknown='error')
Y1=ohe_nj.fit_transform(data[:,0].reshape(-1,1)).toarray()
Y2=ohe_nb.fit_transform(data[:,1].reshape(-1,1)).toarray()

# X=[]
# for n in range(Y1.shape[0]):
#   X.append([Y1[n],Y2[n]])

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

# Z_init=multinomial.rvs(p=[0.5,0.5],n=1,size=N)
# Z_list=np.zeros((T,N,K))
# pie_list=np.zeros((T,K))
# alphas_list=np.zeros((T,K,dj))
# betas_list=np.zeros((T,K,db))

# Nprior=10
# eta_pie, eta_alpha, eta_beta =np.ones(2), Nprior*fake_alphas, Nprior*fake_betas

# Z_list, pie_list, alphas_list, betas_list = nf.do_homemade_Gibbs_sampling(Z_init,X[:N], eta_pie,eta_alpha,eta_beta,T,burnout,keep_every)

#### for optimized version of Gibbs sampler

binsj=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0)
binsb=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0)

print(max(data[:,0]),binsj, binsb)
njb, edgesx, edgesy = np.histogram2d(data[:N,0],data[:N,1],bins=[binsj,binsb])
np.save(data_dir+'/njb.npy',njb)
print(njb.shape)

if njb.shape[0]!=dj or njb.shape[1]!=db:
  print("Wrong observable length")

Z_init=np.zeros((K,dj,db))
for j in range(dj):
  for b in range(db):
    Z_init[:,j,b]= multinomial.rvs(p=[0.5,0.5],n=njb[j,b])
Z_list=np.zeros((T,K,dj,db))
pie_list=np.zeros((T,K))
alphas_list=np.zeros((T,K,dj))
betas_list=np.zeros((T,K,db))

Nprior=10

# ensure no parameters of the prior are too small

eta_pie, eta_alpha, eta_beta =np.ones(2), np.where(Nprior*fake_alphas>=1.01,Nprior*fake_alphas,1.01), np.where(Nprior*fake_betas>=1.01,Nprior*fake_betas,1.01)
#eta_alpha, eta_beta = np.array([eta_alpha[k]*Nprior/np.sum(eta_alpha[k]) for k in range(K)], np.array([eta_beta[k]*Nprior/np.sum(eta_beta[k]) for k in range(K)]

Z_list, pie_list, alphas_list, betas_list = nf.do_homemade_Gibbs_sampling_optimized(Z_init,njb, eta_pie,eta_alpha,eta_beta,T,burnout,keep_every)


np.save(output_dir+'/Z_list.npy',Z_list)
np.save(output_dir+'/pie_list.npy',pie_list)
np.save(output_dir+'/alphas_list.npy',alphas_list)
np.save(output_dir+'/betas_list.npy',betas_list)
