# here i make the data
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

import necessary_functions

np.random.seed(42)

file_dir = sys.argv[1]
data_dir = sys.argv[2]

data=[]
with open(file_dir,'r') as f:
  for nline, line in enumerate(f):
    if(nline>0):
      data.append([float(line.split(' ')[i]) for i in range(3)])
data=np.asarray(data)
np.random.shuffle(data)

data_smeared=data.copy()
nj_movement=np.argmax(multinomial(n=1,p=[0.1,0.7,0.2]).rvs(len(data)),axis=1)
nb_movement=np.argmax(multinomial(n=1,p=[0.95,0.05]).rvs(len(data)),axis=1)
print(data_smeared.shape, nj_movement.shape)
data_smeared[:,0]+=nj_movement
data_smeared[:,1]+=nb_movement

data=data[np.where(data[:,0]>3.0)[0]]
#data=data[np.where(data[:,0]>2.0)[0]]
data=data[np.where(data[:,0]<10)[0]]
#data=data[np.where(data[:,0]<9)[0]]
data=data[np.where(data[:,1]<=5)[0]]
#data=data[np.where(data[:,1]<=3)[0]]
data=data[np.where(data[:,1]>=2)[0]]
#data[np.where(data[:,1]>=3)[0],1]=3

np.savetxt(data_dir+'/processed_data.dat',data)

data_smeared=data_smeared[np.where(data_smeared[:,0]>3.0)[0]]
#data_smeared=data_smeared[np.where(data_smeared[:,0]>2.0)[0]]
data_smeared=data_smeared[np.where(data_smeared[:,0]<10)[0]]
#data_smeared=data_smeared[np.where(data_smeared[:,0]<9)[0]]
data_smeared=data_smeared[np.where(data_smeared[:,1]<=5)[0]]
#data_smeared=data_smeared[np.where(data_smeared[:,1]<=3)[0]]
data_smeared=data_smeared[np.where(data_smeared[:,1]>=2)[0]]
data_smeared=data_smeared[np.where(data_smeared[:,0]>=data_smeared[:,1])[0]]
#data_smeared[np.where(data_smeared[:,1]>=3)[0],1]=3

np.savetxt(data_dir+'/processed_data_smeared.dat',data_smeared)

binsj=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0)
binsb=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0)


fig, (ax1, ax2) = plt.subplots(2,2,figsize=(12,9))

hist = ax1[0].hist2d(data[np.where(data[:,2]==0)[0],0],data[np.where(data[:,2]==0)[0],1],bins=[binsj,binsb],cmap='gist_heat_r')
ax1[0].set_xlabel(r'$N_j$')
ax1[0].set_ylabel(r'$N_b$')
ax1[0].set_title(r'$t\overline{t}W^{\pm}$ Data')
fig.colorbar(hist[3],ax=ax1[0])

hist = ax1[1].hist2d(data[np.where(data[:,2]==1)[0],0],data[np.where(data[:,2]==1)[0],1],bins=[binsj,binsb],cmap='gist_heat_r')
ax1[1].set_xlabel(r'$N_j$')
ax1[1].set_ylabel(r'$N_b$')
ax1[1].set_title(r'4-top Data')
fig.colorbar(hist[3],ax=ax1[1])

hist = ax2[0].hist2d(data_smeared[np.where(data_smeared[:,2]==0)[0],0],data_smeared[np.where(data_smeared[:,2]==0)[0],1],bins=[binsj,binsb],cmap='gist_heat_r')
ax2[0].set_xlabel(r'$N_j$')
ax2[0].set_ylabel(r'$N_b$')
ax2[0].set_title(r'$t\overline{t}W^{\pm}$ MC')
fig.colorbar(hist[3],ax=ax2[0])

hist = ax2[1].hist2d(data_smeared[np.where(data_smeared[:,2]==1)[0],0],data_smeared[np.where(data_smeared[:,2]==1)[0],1],bins=[binsj,binsb],cmap='gist_heat_r')
ax2[1].set_xlabel(r'$N_j$')
ax2[1].set_ylabel(r'$N_b$')
ax2[1].set_title(r'4-top MC')
fig.colorbar(hist[3],ax=ax2[1])


fig.tight_layout()
plt.savefig(data_dir+'/data_vs_mc_2d.pdf')

ohe_nj=OneHotEncoder(handle_unknown='error')
ohe_nb=OneHotEncoder(handle_unknown='error')

labels=data[:,2]
f1=np.sum(labels==1)/len(labels)

f1_mc=np.sum(data_smeared[:,2]==1)/len(data_smeared[:,2])

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
  
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4.5))

#ax1.set_title('True AUC = '+str(round(auc_true,3))+', Prior AUC = '+str(round(auc_prior,3)))
ax1.plot(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),true_alphas[0],'bo',label='true ttW')
ax1.plot(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),true_alphas[0],'b-')
ax1.plot(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),fake_alphas[0],'bx',label='fake ttW')
ax1.plot(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),fake_alphas[0],'b--')
ax1.plot(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),true_alphas[1],'ro',label='true 4top, $f_{1}$ = '+str(round(f1,2)))
ax1.plot(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),true_alphas[1],'r-')
ax1.plot(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),fake_alphas[1],'rx',label='fake 4top, $f_{1}$ = '+str(round(f1_mc,2)))
ax1.plot(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),fake_alphas[1],'r--')
ax1.legend(loc='upper right')
ax1.set_xlabel('$N_{j}$')
#plt.show()

ax2.plot(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),true_betas[0],'bo',label='true ttW')
ax2.plot(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),true_betas[0],'b-')
ax2.plot(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),fake_betas[0],'bx',label='fake ttW')
ax2.plot(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),fake_betas[0],'b--')
ax2.plot(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),true_betas[1],'ro',label='true 4top, $f_{1}$ = '+str(round(f1,2)))
ax2.plot(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),true_betas[1],'r-')
ax2.plot(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),fake_betas[1],'rx',label='fake 4top, $f_{1}$ = '+str(round(f1_mc,2)))
ax2.plot(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),fake_betas[1],'r--')
ax2.legend(loc='upper right')
ax2.set_xlabel('$N_{b}$')

fig.tight_layout()

plt.savefig(data_dir+'/data_vs_mc_1d.pdf')
