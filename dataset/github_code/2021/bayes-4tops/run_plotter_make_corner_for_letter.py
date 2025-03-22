# here i take all walkers and do trace plots, corner plots and histograms
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.labelsize': 14})
from scipy.stats import poisson, norm, bernoulli, expon, uniform, beta, gamma, multinomial, multivariate_normal
from scipy.stats import rv_histogram
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
nwalkers = int(sys.argv[2])
N=int(sys.argv[3])	
T=int(sys.argv[4])
burnout=int(sys.argv[5])
keep_every=int(sys.argv[6])
Nprior=10

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
  
# now I do need to use Nprior
fake_alphas, fake_betas = np.where(fake_alphas>=1.01/Nprior,fake_alphas,1.01/Nprior), np.where(fake_betas>=1.01/Nprior,fake_betas,1.01/Nprior)
fake_alphas, fake_betas = np.array([fake_alphas[k]/np.sum(fake_alphas[k]) for k in range(K)], np.array([fake_betas[k]/np.sum(fake_betas[k]) for k in range(K)]


K=true_alphas.shape[0]
dj=true_alphas.shape[1]
db=true_betas.shape[1]


pie_list=np.zeros((nwalkers,T,K))
alphas_list=np.zeros((nwalkers,T,K,dj))
betas_list=np.zeros((nwalkers,T,K,db))

for walker in range(nwalkers):
  pie_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/pie_list.npy')
  alphas_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/alphas_list.npy')
  betas_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/betas_list.npy')
  

f1vals=np.arange(0.01,1.0,0.01)

pie_list_all_walkers=np.zeros((nwalkers*T,K))
alphas_list_all_walkers=np.zeros((nwalkers*T,K,dj))
betas_list_all_walkers=np.zeros((nwalkers*T,K,db))
for walker in range(nwalkers):
  pie_list_all_walkers[walker*T:(walker+1)*T]=pie_list[walker]
  alphas_list_all_walkers[walker*T:(walker+1)*T]=alphas_list[walker]
  betas_list_all_walkers[walker*T:(walker+1)*T]=betas_list[walker]

dim=(K-1)+K*(dj+db)
  
var_corner=np.zeros((nwalkers*T,dim))
var_names=[r'$\pi_1$']
var_corner_truth = np.zeros(dim)

var_corner[:,0]=pie_list_all_walkers[:,1].T
var_corner_truth[0]=f1

for i in range(dj):
  var_names.append(r'$\alpha_{0,'+str(i)+'}$')
  var_corner[:,i+1]=alphas_list_all_walkers[:,0,i]
  var_corner_truth[i+1]=true_alphas[0,i]
for i in range(dj):
  var_names.append(r'$\alpha_{1,'+str(i)+'}$')
  var_corner[:,dj+1+i]=alphas_list_all_walkers[:,1,i]
  var_corner_truth[1+dj+i]=true_alphas[1,i]
for i in range(db):
  var_names.append(r'$\beta_{0,'+str(i)+'}$')
  var_corner[:,1+2*dj+i]=betas_list_all_walkers[:,0,i]
  var_corner_truth[1+2*dj+i]=true_betas[0,i]
for i in range(db):
  var_names.append(r'$\beta_{1,'+str(i)+'}$')
  var_corner[:,1+2*dj+db+i]=betas_list_all_walkers[:,1,i]
  var_corner_truth[1+2*dj+db+i]=true_betas[1,i]
  
var_corner_prior=np.zeros((nwalkers*T,dim))
var_corner_prior[:,0]=dirichlet.rvs(size=nwalkers*T,alpha=np.ones(2))[:,1]
var_corner_prior[:,1:1+dj]=dirichlet.rvs(size=nwalkers*T,alpha=Nprior*fake_alphas[0])
var_corner_prior[:,1+dj:1+2*dj]=dirichlet.rvs(size=nwalkers*T,alpha=Nprior*fake_alphas[1])
var_corner_prior[:,1+2*dj:1+2*dj+db]=dirichlet.rvs(size=nwalkers*T,alpha=Nprior*fake_betas[0])
var_corner_prior[:,1+2*dj+db:1+2*dj+2*db]=dirichlet.rvs(size=nwalkers*T,alpha=Nprior*fake_betas[1])

# VB stuff, need to run run_vb.py first

gamma_pie_VB = np.loadtxt(data_dir+'/gamma_pie_VB.dat')
gamma_alpha_VB = np.loadtxt(data_dir+'/gamma_alpha_VB.dat')
gamma_beta_VB = np.loadtxt(data_dir+'/gamma_beta_VB.dat')

var_corner_VB=np.zeros((nwalkers*T,dim))
var_corner_VB[:,0]=dirichlet.rvs(size=nwalkers*T,alpha=gamma_pie_VB)[:,1]
var_corner_VB[:,1:1+dj]=dirichlet.rvs(size=nwalkers*T,alpha=gamma_alpha_VB[0])
var_corner_VB[:,1+dj:1+2*dj]=dirichlet.rvs(size=nwalkers*T,alpha=gamma_alpha_VB[1])
var_corner_VB[:,1+2*dj:1+2*dj+db]=dirichlet.rvs(size=nwalkers*T,alpha=gamma_beta_VB[0])
var_corner_VB[:,1+2*dj+db:1+2*dj+2*db]=dirichlet.rvs(size=nwalkers*T,alpha=gamma_beta_VB[1])


llr_diffs=np.zeros(dim)
llr_diffs_VB=np.zeros(dim)
# Loop over the diagonal
for i in range(dim):
    llr_diffs[i]=2*(nf.do_log_likelihood_estimate(var_corner[:,i],var_corner_truth[i])-nf.do_log_likelihood_estimate(var_corner_prior[:,i],var_corner_truth[i]))
    llr_diffs_VB[i]=2*(nf.do_log_likelihood_estimate(var_corner_VB[:,i],var_corner_truth[i])-nf.do_log_likelihood_estimate(var_corner_prior[:,i],var_corner_truth[i]))


Nbins=20
lower_limit=np.zeros(dim)
upper_limit=np.ones(dim)
lower_limit=np.quantile(var_corner,0.01,axis=0)
upper_limit=np.quantile(var_corner,0.99,axis=0)
range_corner=[[lower_limit[d],upper_limit[d]] for d in range(dim)]

lower_limit_prior=np.zeros(dim)
upper_limit_prior=np.ones(dim)
lower_limit_prior=np.quantile(var_corner_prior,0.01,axis=0)
upper_limit_prior=np.quantile(var_corner_prior,0.99,axis=0)
range_corner_prior=[[lower_limit_prior[d],upper_limit_prior[d]] for d in range(dim)]


lower_limit_VB=np.zeros(dim)
upper_limit_VB=np.ones(dim)
lower_limit_VB=np.quantile(var_corner_VB,0.01,axis=0)
upper_limit_VB=np.quantile(var_corner_VB,0.99,axis=0)
range_corner_VB=[[lower_limit_VB[d],upper_limit_VB[d]] for d in range(dim)]


total_range=[ [min(lower_limit[d],lower_limit_prior[d],lower_limit_VB[d]),max(upper_limit[d],upper_limit_prior[d],upper_limit_VB[d])] for d in range(dim)]
#total_range=[ [0.0,1.0] for d in range(dim)]


fig, ax = plt.subplots(2,2,figsize=(8,8))
#alpha10, beta10
index_1=1+dj
index_2=1+2*dj+db
ax[0,0].hist(var_corner[:,index_1],bins=np.linspace(total_range[index_1][0],total_range[index_1][1],Nbins),histtype='step',color='black',density=True)
ax[0,0].hist(var_corner_prior[:,index_1],bins=np.linspace(total_range[index_1][0],total_range[index_1][1],Nbins),histtype='step',color='red',density=True)
ax[0,0].hist(var_corner_VB[:,index_1],bins=np.linspace(total_range[index_1][0],total_range[index_1][1],Nbins),histtype='step',color='magenta',density=True)
ax[0,0].axvline(var_corner_truth[index_1],color='#4682b4')
ax[0,0].set_xlabel(var_names[index_1])
ax[0,0].set_ylabel('PDF')
ax[0,0].set_xlim(total_range[index_1])
ax[0,0].set_title('LLR = '+str(round(llr_diffs[index_1],2))+' ('+str(round(llr_diffs_VB[index_1],2))+')')

ax[1,1].hist(var_corner[:,index_2],bins=np.linspace(total_range[index_2][0],total_range[index_2][1],Nbins),histtype='step',color='black',density=True)
ax[1,1].hist(var_corner_prior[:,index_2],bins=np.linspace(total_range[index_2][0],total_range[index_2][1],Nbins),histtype='step',color='red',density=True)
ax[1,1].hist(var_corner_VB[:,index_2],bins=np.linspace(total_range[index_2][0],total_range[index_2][1],Nbins),histtype='step',color='magenta',density=True)
ax[1,1].axvline(var_corner_truth[index_2],color='#4682b4')
ax[1,1].set_xlabel(var_names[index_2])
ax[1,1].set_xlim(total_range[index_2])
ax[1,1].set_title('LLR = '+str(round(llr_diffs[index_2],2))+' ('+str(round(llr_diffs_VB[index_2],2))+')')

corner.hist2d(var_corner[:,index_1],var_corner[:,index_2],ax=ax[1,0],new_fig=False,bins=Nbins,color='black')
corner.hist2d(var_corner_prior[:,index_1],var_corner_prior[:,index_2],ax=ax[1,0],new_fig=False,bins=Nbins,color='red')
corner.hist2d(var_corner_VB[:,index_1],var_corner_VB[:,index_2],ax=ax[1,0],new_fig=False,bins=Nbins,color='magenta')
ax[1,0].set_xlabel(var_names[index_1])
ax[1,0].set_ylabel(var_names[index_2])
ax[1,0].scatter(var_corner_truth[index_1],var_corner_truth[index_2],color='#4682b4')
ax[1,0].axvline(var_corner_truth[index_1],color='#4682b4')
ax[1,0].axhline(var_corner_truth[index_2],color='#4682b4')
ax[1,0].set_xlim(total_range[index_1])
ax[1,0].set_ylim(total_range[index_2])


columns=['Log-Likelihood Ratio']
rows=[r'$\pi_{1}$',r'$\alpha_{0,j}$',r'$\alpha_{1,j}$',r'$\beta_{0,b}$',r'$\beta_{1,b}$','All']
cell_text=np.asarray([str(round(llr_diffs[0],2))+' ('+str(round(llr_diffs_VB[0],2))+')',str(round(np.sum(llr_diffs[1:1+dj]),2))+' ('+str(round(np.sum(llr_diffs_VB[1:1+dj]),2))+')',str(round(np.sum(llr_diffs[1+dj:1+2*dj]),2))+' ('+str(round(np.sum(llr_diffs_VB[1+dj:1+2*dj]),2))+')',str(round(np.sum(llr_diffs[1+2*dj:1+2*dj+db]),2))+' ('+str(round(np.sum(llr_diffs_VB[1+2*dj:1+2*dj+db]),2))+')',str(round(np.sum(llr_diffs[1+2*dj+db:1+2*dj+2*db]),2))+' ('+str(round(np.sum(llr_diffs_VB[1+2*dj+db:1+2*dj+2*db]),2))+')',str(round(np.sum(llr_diffs),2))+' ('+str(round(np.sum(llr_diffs_VB),2))+')']).reshape(-1,1)

this_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      colWidths=[1.0],
                      bbox=(0.0,1.115,1.0,1.0),
                      fontsize=14                    
                      #loc='center'
                      )
this_table.set_fontsize(14)
ax[0,1].set_axis_off()
ax[0,0].label_outer()
ax[0,1].label_outer()
ax[1,0].label_outer()
#ax[1,1].label_outer()

fig.tight_layout()

plt.savefig(data_dir+'/corner_for_letter.pdf')
plt.savefig(data_dir+'/corner_for_letter.png')
