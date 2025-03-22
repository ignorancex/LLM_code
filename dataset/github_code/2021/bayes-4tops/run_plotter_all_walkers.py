# here i take all walkers and do trace plots, corner plots and histograms
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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


# Z_list=np.zeros((nwalkers,T,N,K))

### optimized version
Z_list=np.zeros((nwalkers,T,K,dj,db))
pie_list=np.zeros((nwalkers,T,K))
alphas_list=np.zeros((nwalkers,T,K,dj))
betas_list=np.zeros((nwalkers,T,K,db))

for walker in range(nwalkers):
  # Z_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/Z_list.npy')
  pie_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/pie_list.npy')
  alphas_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/alphas_list.npy')
  betas_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/betas_list.npy')
  
pie_list_all_walkers=np.zeros((nwalkers*T,K))
alphas_list_all_walkers=np.zeros((nwalkers*T,K,dj))
betas_list_all_walkers=np.zeros((nwalkers*T,K,db))
for walker in range(nwalkers):
  pie_list_all_walkers[walker*T:(walker+1)*T]=pie_list[walker]
  alphas_list_all_walkers[walker*T:(walker+1)*T]=alphas_list[walker]
  betas_list_all_walkers[walker*T:(walker+1)*T]=betas_list[walker]

f1vals=np.arange(0.01,1.0,0.01)


fig, ax = plt.subplots(1,2,figsize=(8,4))
for walker in range(nwalkers):

  ax[0].plot(pie_list[walker][:,0],'b.')
  ax[1].plot(pie_list[walker][:,1],'r.')
  ax[0].set_ylabel(r'$\pi_0$')
  ax[1].set_ylabel(r'$\pi_1$')

ax[0].set_xlabel('Iterations')
ax[1].set_xlabel('Iterations')
fig.tight_layout()

plt.savefig(data_dir+'/pies_trace_plot.png')
plt.savefig(data_dir+'/pies_trace_plot.pdf')

fig, ax = plt.subplots(1,dj,figsize=(4*dj,4))
f1vals=np.arange(0.01,1.0,0.01)
for walker in range(nwalkers):

  for j in range(dj):
    ax[j].plot(alphas_list[walker][:,0,j],'b.')
    ax[j].plot(alphas_list[walker][:,1,j],'r.')
    ax[j].set_ylabel(r'$\alpha_{k'+str(j)+'}$')
    ax[j].set_xlabel('Iterations')
fig.tight_layout()

plt.savefig(data_dir+'/alphas_trace_plot.png')
plt.savefig(data_dir+'/alphas_trace_plot.pdf')

fig, ax = plt.subplots(1,db,figsize=(4*db,4))
f1vals=np.arange(0.01,1.0,0.01)
for walker in range(nwalkers):

  for b in range(db):
    ax[b].plot(betas_list[walker][:,0,b],'b.')
    ax[b].plot(betas_list[walker][:,1,b],'r.')
    ax[b].set_ylabel(r'$\beta_{k'+str(b)+'}$')
    ax[b].set_xlabel('Iterations')
fig.tight_layout()

plt.savefig(data_dir+'/betas_trace_plot.png')
plt.savefig(data_dir+'/betas_trace_plot.pdf')

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


fig = corner.corner(
    var_corner, labels=var_names,truths=var_corner_truth,bins=Nbins,range=total_range);

plt.savefig(data_dir+'/corner_plot.png')
plt.savefig(data_dir+'/corner_plot.pdf')


corner.corner(
    var_corner_prior,fig=fig, color='red',bins=Nbins,range=total_range);


# Extract the axes
axes = np.array(fig.axes).reshape((dim, dim))
llr_diffs=np.zeros(dim)
# Loop over the diagonal
for i in range(dim):
    ax = axes[i, i]
    llr_diffs[i]=2*(nf.do_log_likelihood_estimate(var_corner[:,i],var_corner_truth[i])-nf.do_log_likelihood_estimate(var_corner_prior[:,i],var_corner_truth[i]))
    ax.set_title('LLR = '+str(round(llr_diffs[i],2)))

prior_truth=dirichlet.logpdf(x=[1-f1,f1],alpha=np.ones(2))+dirichlet.logpdf(x=true_alphas[0],alpha=Nprior*fake_alphas[0])+dirichlet.logpdf(x=true_alphas[1],alpha=Nprior*fake_alphas[1])+dirichlet.logpdf(x=true_betas[0],alpha=Nprior*fake_betas[0])+dirichlet.logpdf(x=true_betas[1],alpha=Nprior*fake_betas[1])


plt.text(0.05, 0.95, r'Sum of independent Gibbs LLR = '+str(round(np.sum(llr_diffs),3)), transform=axes[0,1].transAxes, fontsize=14,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


        
plt.savefig(data_dir+'/corner_plot_bis.png')
plt.savefig(data_dir+'/corner_plot_bis.pdf')


corner.corner(
    var_corner_VB,fig=fig, color='magenta',bins=Nbins,range=total_range);

axes = np.array(fig.axes).reshape((dim, dim))
llr_diffs_VB=np.zeros(dim)
# Loop over the diagonal
for i in range(dim):
    ax = axes[i, i]
    llr_diffs_VB[i]=2*(nf.do_log_likelihood_estimate(var_corner_VB[:,i],var_corner_truth[i])-nf.do_log_likelihood_estimate(var_corner_prior[:,i],var_corner_truth[i]))
    ax.set_title('LLR = '+str(round(llr_diffs[i],2))+' ('+str(round(llr_diffs_VB[i],2))+')')

plt.text(0.05, 0.75, r'Sum of independent VI LLR = '+str(round(np.sum(llr_diffs_VB),3)), transform=axes[0,1].transAxes, fontsize=14,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


plt.savefig(data_dir+'/corner_plot_tris.png')
plt.savefig(data_dir+'/corner_plot_tris.pdf')


#non dependent indexes
mask_index = np.ones(dim,dtype=bool)
mask_index[1+dj-1]=0
mask_index[1+2*dj-1]=0
mask_index[1+2*dj+db-1]=0
mask_index[1+2*dj+2*db-1]=0

var_means=np.mean(var_corner[:,mask_index],axis=0)
var_cov=np.cov(var_corner[:,mask_index].T)
print("Mean length")
print(var_means.shape)
print("Cov shape")
print(var_cov.shape)

gaussian_approx=multivariate_normal(mean=var_means,cov=var_cov)
gaussian_samples=gaussian_approx.rvs(size=nwalkers*T)

var_gaussian_samples=np.zeros((nwalkers*T,dim))
var_gaussian_samples[:,0:1+dj-1]=gaussian_samples[:,0:1+dj-1]
var_gaussian_samples[:,1+dj-1]=np.ones(nwalkers*T)-np.sum(gaussian_samples[:,1:1+dj-1],axis=1)
var_gaussian_samples[:,1+dj:1+2*dj-1]=gaussian_samples[:,1+dj-1:1+2*dj-1-1]
var_gaussian_samples[:,1+2*dj-1]=np.ones(nwalkers*T)-np.sum(gaussian_samples[:,1+dj-1:1+2*dj-1-1],axis=1)
var_gaussian_samples[:,1+2*dj:1+2*dj+db-1]=gaussian_samples[:,1+2*dj-2:1+2*dj+db-1-2]
var_gaussian_samples[:,1+2*dj+db-1]=np.ones(nwalkers*T)-np.sum(gaussian_samples[:,1+2*dj-2:1+2*dj+db-1-2],axis=1)
var_gaussian_samples[:,1+2*dj+db:1+2*dj+2*db-1]=gaussian_samples[:,1+2*dj+db-3:1+2*dj+2*db-1-3]
var_gaussian_samples[:,1+2*dj+2*db-1]=np.ones(nwalkers*T)-np.sum(gaussian_samples[:,1+2*dj+db-3:1+2*dj+2*db-1-3],axis=1)

corner.corner(
    var_gaussian_samples,fig=fig, color='limegreen',bins=Nbins,range=total_range);

gauss_approx_truth=gaussian_approx.logpdf(var_corner_truth[mask_index])

plt.text(0.05, 0.75, r"Log Prior on the true values = "+str(round(prior_truth,3)), transform=axes[0,1].transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.text(0.05, 0.55, r"Log Posterior on the true values = "+str(round(gauss_approx_truth,3)), transform=axes[0,1].transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.text(0.05, 0.35, r"2*(Log Posterior - Log Prior) = "+str(round(2*(gauss_approx_truth-prior_truth),3)), transform=axes[0,1].transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


plt.savefig(data_dir+'/corner_plot_fourth.png')
plt.savefig(data_dir+'/corner_plot_fourth.pdf')


alpha_mean=np.mean(alphas_list_all_walkers,axis=0)
alpha_err=np.std(alphas_list_all_walkers,axis=0)


beta_mean=np.mean(betas_list_all_walkers,axis=0)
beta_err=np.std(betas_list_all_walkers,axis=0)



# histogram plots

fig, ax = plt.subplots(2,5,figsize=(20,8))

# alpha ttW
# data
ax[0,0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=true_alphas[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step',color='blue', label='True ttW')
ax[1,0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=true_alphas[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='blue',label='True ttW')

# prior
nprior, bprior, pprior = ax[0,0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='red',label='Prior ttW')
nprior_up, bprior_up, pprior_up = ax[0,0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[0]+np.sqrt(dirichlet.var(alpha=Nprior*fake_alphas[0])),bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='red')
nprior_down, bprior_down, pprior_down = ax[0,0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[0]-np.sqrt(dirichlet.var(alpha=Nprior*fake_alphas[0])),bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='red')
ax[0,0].bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

#posterior 
nposterior, bposterior, pposterior = ax[1,0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='black',label='Posterior ttW')
nposterior_up, bposterior_up, posterior_up = ax[1,0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[0]+alpha_err[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='black')
nposterior_down, bposterior_down, pposterior_down = ax[1,0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[0]-alpha_err[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='black')
ax[1,0].bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=0, color='grey', alpha=0.25, zorder=-1)

ax[0,0].set_ylim(0.0,1.0)
ax[1,0].set_ylim(0.0,1.0)
ax[0,0].legend(loc='upper right')
ax[1,0].legend(loc='upper right')


# alpha 4top
# data
ax[0,1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=true_alphas[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step',color='blue', label='True 4-top')
ax[1,1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=true_alphas[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='blue',label='True 4-top')

# prior
nprior, bprior, pprior = ax[0,1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='red',label='Prior 4-top')
nprior_up, bprior_up, pprior_up = ax[0,1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[1]+np.sqrt(dirichlet.var(alpha=Nprior*fake_alphas[1])),bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='red')
nprior_down, bprior_down, pprior_down = ax[0,1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[1]-np.sqrt(dirichlet.var(alpha=Nprior*fake_alphas[1])),bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='red')
ax[0,1].bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

#posterior 
nposterior, bposterior, pposterior = ax[1,1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='black',label='Posterior 4-top')
nposterior_up, bposterior_up, posterior_up = ax[1,1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[1]+alpha_err[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='black')
nposterior_down, bposterior_down, pposterior_down = ax[1,1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[1]-alpha_err[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='black')
ax[1,1].bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=0, color='grey', alpha=0.25, zorder=-1)

ax[0,1].set_ylim(0.0,1.0)
ax[1,1].set_ylim(0.0,1.0)
ax[0,1].legend(loc='upper right')
ax[1,1].legend(loc='upper right')


# beta ttW
# data
ax[0,2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=true_betas[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step',color='blue', label='True ttW')
ax[1,2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=true_betas[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='blue',label='True ttW')

# prior
nprior, bprior, pprior = ax[0,2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='red',label='Prior ttW')
nprior_up, bprior_up, pprior_up = ax[0,2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[0]+np.sqrt(dirichlet.var(alpha=Nprior*fake_betas[0])),bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='red')
nprior_down, bprior_down, pprior_down = ax[0,2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[0]-np.sqrt(dirichlet.var(alpha=Nprior*fake_betas[0])),bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='red')
ax[0,2].bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

#posterior 
nposterior, bposterior, pposterior = ax[1,2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='black',label='Posterior ttW')
nposterior_up, bposterior_up, posterior_up = ax[1,2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[0]+beta_err[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='black')
nposterior_down, bposterior_down, pposterior_down = ax[1,2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[0]-beta_err[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='black')
ax[1,2].bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=0, color='grey', alpha=0.25, zorder=-1)

ax[0,2].set_ylim(0.0,1.0)
ax[1,2].set_ylim(0.0,1.0)
ax[0,2].legend(loc='upper right')
ax[1,2].legend(loc='upper right')


# beta 4top
# data
ax[0,3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=true_betas[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step',color='blue', label='True 4-top')
ax[1,3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=true_betas[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='blue',label='True 4-top')

# prior
nprior, bprior, pprior = ax[0,3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='red',label='Prior 4-top')
nprior_up, bprior_up, pprior_up = ax[0,3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[1]+np.sqrt(dirichlet.var(alpha=Nprior*fake_betas[1])),bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='red')
nprior_down, bprior_down, pprior_down = ax[0,3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[1]-np.sqrt(dirichlet.var(alpha=Nprior*fake_betas[1])),bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='red')
ax[0,3].bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

#posterior 
nposterior, bposterior, pposterior = ax[1,3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='black',label='Posterior 4-top')
nposterior_up, bposterior_up, posterior_up = ax[1,3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[1]+beta_err[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='black')
nposterior_down, bposterior_down, pposterior_down = ax[1,3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[1]-beta_err[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='black')
ax[1,3].bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=0, color='grey', alpha=0.25, zorder=-1)

ax[0,3].set_ylim(0.0,1.0)
ax[1,3].set_ylim(0.0,1.0)
ax[0,3].legend(loc='upper right')
ax[1,3].legend(loc='upper right')

# pies

ax[0,4].axvline(f1,color='blue',label='Prior $\pi_1$')
ax[1,4].axvline(f1,color='blue',label='True $\pi_1$')
ax[0,4].plot(f1vals,dirichlet.pdf([1-f1vals,f1vals],alpha=[1.0,1.0]),'r--',label='Prior $\pi_1$')
ax[1,4].hist(pie_list_all_walkers[:,1],color='black',label='Posterior 4-top',alpha=0.2,density=True)
#ax[4].fill_between(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0), beta_mean[1]-beta_err[1],beta_mean[1]+beta_err[1],alpha=0.2,color='red')

ax[0,4].legend(loc='upper left')
ax[1,4].legend(loc='upper left')

ax[1,0].set_xlabel('$N_j$')
ax[1,1].set_xlabel('$N_j$')
ax[1,2].set_xlabel('$N_b$')
ax[1,3].set_xlabel('$N_b$')
ax[1,4].set_xlabel(r'$\pi_{1}$')
fig.tight_layout()

plt.savefig(data_dir+'/histogram.png')
plt.savefig(data_dir+'/histogram.pdf')

# condensed histograms

fig, ax = plt.subplots(1,5,figsize=(20,4))

# alpha ttW
# data
ax[0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=true_alphas[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step',color='blue', label='True ttW')

# prior
nprior, bprior, pprior = ax[0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='red',label='Prior ttW')
nprior_up, bprior_up, pprior_up = ax[0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[0]+np.sqrt(dirichlet.var(alpha=Nprior*fake_alphas[0])),bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='red')
nprior_down, bprior_down, pprior_down = ax[0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[0]-np.sqrt(dirichlet.var(alpha=Nprior*fake_alphas[0])),bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='red')
ax[0].bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

#posterior 
nposterior, bposterior, pposterior = ax[0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='black',label='Posterior ttW')
nposterior_up, bposterior_up, posterior_up = ax[0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[0]+alpha_err[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='black')
nposterior_down, bposterior_down, pposterior_down = ax[0].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[0]-alpha_err[0],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='black')
ax[0].bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=0, color='grey', alpha=0.25, zorder=-1)

ax[0].set_ylim(0.0,1.0)
ax[0].legend(loc='upper right')

# alpha 4top
# data
ax[1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=true_alphas[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step',color='blue', label='True 4-top')

# prior
nprior, bprior, pprior = ax[1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='red',label='Prior 4-top')
nprior_up, bprior_up, pprior_up = ax[1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[1]+np.sqrt(dirichlet.var(alpha=Nprior*fake_alphas[1])),bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='red')
nprior_down, bprior_down, pprior_down = ax[1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=fake_alphas[1]-np.sqrt(dirichlet.var(alpha=Nprior*fake_alphas[1])),bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='red')
ax[1].bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

#posterior 
nposterior, bposterior, pposterior = ax[1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', color='black',label='Posterior 4-top')
nposterior_up, bposterior_up, posterior_up = ax[1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[1]+alpha_err[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='black')
nposterior_down, bposterior_down, pposterior_down = ax[1].hist(np.arange(min(data[:,0]),max(data[:,0])+1.0,1.0),weights=alpha_mean[1]-alpha_err[1],bins=np.arange(min(data[:,0])-0.5,max(data[:,0])+1.5,1.0), histtype='step', linestyle='--', color='black')
ax[1].bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=0, color='grey', alpha=0.25, zorder=-1)

ax[1].set_ylim(0.0,1.0)
ax[1].legend(loc='upper right')

# beta ttW
# data
ax[2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=true_betas[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step',color='blue', label='True ttW')

# prior
nprior, bprior, pprior = ax[2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='red',label='Prior ttW')
nprior_up, bprior_up, pprior_up = ax[2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[0]+np.sqrt(dirichlet.var(alpha=Nprior*fake_betas[0])),bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='red')
nprior_down, bprior_down, pprior_down = ax[2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[0]-np.sqrt(dirichlet.var(alpha=Nprior*fake_betas[0])),bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='red')
ax[2].bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

#posterior 
nposterior, bposterior, pposterior = ax[2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='black',label='Posterior ttW')
nposterior_up, bposterior_up, posterior_up = ax[2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[0]+beta_err[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='black')
nposterior_down, bposterior_down, pposterior_down = ax[2].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[0]-beta_err[0],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='black')
ax[2].bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=0, color='grey', alpha=0.25, zorder=-1)

ax[2].set_ylim(0.0,1.0)
ax[2].legend(loc='upper right')


# beta 4top
# data
ax[3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=true_betas[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='blue',label='True 4-top')

# prior
nprior, bprior, pprior = ax[3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='red',label='Prior 4-top')
nprior_up, bprior_up, pprior_up = ax[3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[1]+np.sqrt(dirichlet.var(alpha=Nprior*fake_betas[1])),bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='red')
nprior_down, bprior_down, pprior_down = ax[3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=fake_betas[1]-np.sqrt(dirichlet.var(alpha=Nprior*fake_betas[1])),bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='red')
ax[3].bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

#posterior 
nposterior, bposterior, pposterior = ax[3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', color='black',label='Posterior 4-top')
nposterior_up, bposterior_up, posterior_up = ax[3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[1]+beta_err[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='black')
nposterior_down, bposterior_down, pposterior_down = ax[3].hist(np.arange(min(data[:,1]),max(data[:,1])+1.0,1.0),weights=beta_mean[1]-beta_err[1],bins=np.arange(min(data[:,1])-0.5,max(data[:,1])+1.5,1.0), histtype='step', linestyle='--', color='black')
ax[3].bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=0, color='grey', alpha=0.25, zorder=-1)

ax[3].set_ylim(0.0,1.0)
ax[3].legend(loc='upper right')

# pies

ax[4].axvline(f1,color='blue',label='True')
ax[4].plot(f1vals,dirichlet.pdf([1-f1vals,f1vals],alpha=[1.0,1.0]),'r--',label='Prior')
ax[4].hist(pie_list_all_walkers[:,1],color='black',label='Posterior',alpha=0.2,density=True)

ax[4].legend(loc='upper left')

ax[0].set_xlabel('$N_j$')
ax[1].set_xlabel('$N_j$')
ax[2].set_xlabel('$N_b$')
ax[3].set_xlabel('$N_b$')
ax[4].set_xlabel(r'$\pi_{1}$')
fig.tight_layout()

plt.savefig(data_dir+'/condensed_histogram.png')
plt.savefig(data_dir+'/condensed_histogram.pdf')


# Z_list_all_walkers=np.zeros((nwalkers*T,N,K))
# for walker in range(nwalkers):
#   Z_list_all_walkers[walker*T:(walker+1)*T]=Z_list[walker]

# Z_list_average_over_walkers=np.mean(Z_list_all_walkers,axis=0)

# fig = plt.figure(figsize=(8,6))

# bins_z = np.linspace(0.0,1.0,10)
# plt.hist(Z_list_average_over_walkers[labels[:N]==0.0,1], bins=bins_z,color='blue',alpha=0.4, label='ttW')
# plt.hist(Z_list_average_over_walkers[labels[:N]==1.0,1], bins=bins_z,color='red',alpha=0.4, label='4-top')
# plt.xlabel(r'$\mathbb{E}[z_{1}]$')
# plt.ylabel('Events')
# plt.legend(loc='upper left')
# plt.savefig(data_dir+'/average_Z_assignments.pdf')
# plt.savefig(data_dir+'/average_Z_assignments.png')

