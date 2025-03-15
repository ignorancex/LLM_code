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

from scipy.stats import dirichlet


def beta_function_k(alphas):
  return np.prod([gamma_function(alpha1) for alpha1 in alphas])/gamma_function(np.sum(alphas))

def log_beta_function_k(alphas):
  return np.sum([gammaln(alpha1) for alpha1 in alphas])-gammaln(np.sum(alphas))

# test for pies

def logprob_pie(N,pie,alpha,beta,data):
  K=len(pie)
  dj=alpha.shape[1]
  db=beta.shape[1]
  logprob=0.0
  for n in range(N):
    jn=np.argmax(data[n][0])
    bn=np.argmax(data[n][1])
    logprob+=np.log(pie[0]*alpha[0,jn]*beta[0,bn]+pie[1]*alpha[1,jn]*beta[1,bn])#np.log(alpha[0,jn]*beta[0,bn])-np.log(pie[0]*alpha[0,jn]*beta[0,bn]+pie[1]*alpha[1,jn]*beta[1,bn])
  return logprob
  
def test_for_pies(N,pie_vals,alpha,beta,data):
  logprobs=np.zeros(len(pie_vals))
  for nval, pie_val in enumerate(pie_vals):
    logprobs[nval]=logprob_pie(N,pie_val,alpha,beta,data)
  return logprobs

# parameter estimations

# benchmark

def benchmark(data,alphas_MC,betas_MC,f1_MC):
  N=len(data)
  K=alphas_MC.shape[0]
  dj=alphas_MC.shape[1]
  db=betas_MC.shape[1]
  alphas_tuned=np.zeros(alphas_MC.shape)
  betas_tuned=np.zeros(betas_MC.shape)
  for j in range(dj):
    alphas_tuned[:,j]=(np.sum(data[:,0]==j+min(data[:,0]))/(N*(1-f1_MC)*alphas_MC[0,j]+N*f1_MC*alphas_MC[1,j]))*alphas_MC[:,j]
  for b in range(db):
    betas_tuned[:,b]=(np.sum(data[:,1]==b+min(data[:,1]))/(N*(1-f1_MC)*betas_MC[0,b]+N*f1_MC*betas_MC[1,b]))*betas_MC[:,b]
  return alphas_tuned, betas_tuned
  
# EM

def do_E_step_EM(pies,alphas,betas,data):
  N=len(data)
  K=len(pies)
  gammas=np.zeros((N,K))
  for n in range(N):
    gammas_aux=np.zeros(K)
    for k in range(K):
      gammas_aux[k]=pies[k]*multinomial(n=1,p=alphas[k]).pmf(data[n][0])*multinomial(n=1,p=betas[k]).pmf(data[n][1])
    gammas[n]=gammas_aux/np.sum(gammas_aux)
  return np.asarray(gammas)
  
def do_M_step_EM(gammas,data):
  N=gammas.shape[0]
  K=gammas.shape[1]
  pies=np.zeros(K)
  Nk=np.sum(gammas,axis=0)
  pies=Nk/N
  dj=len(data[0][0])
  db=len(data[0][1])
  Nkj=np.zeros((K,dj))
  Nkb=np.zeros((K,db))
  for n in range(N):
    Nkj[:,np.argmax(data[n][0])]+=gammas[n,:]
    Nkb[:,np.argmax(data[n][1])]+=gammas[n,:]
  alphas=np.zeros((K,dj))
  betas=np.zeros((K,db))
  for k in range(K):
    alphas[k]=Nkj[k,:]/Nk[k]
    betas[k]=Nkb[k,:]/Nk[k]
  return pies, alphas, betas
  
def logprob_EM(pies,alphas,betas,data):
  N=len(data)
  K=len(pies)
  logprob=0.0
  for n in range(N):
    aux=0.0
    for k in range(K):
      aux+=pies[k]*multinomial(n=1,p=alphas[k]).pmf(data[n][0])*multinomial(n=1,p=betas[k]).pmf(data[n][1])
    logprob+=np.log(aux)/N #normalizo por dato
  return logprob

def do_EM_algorithm(N,K,T, thresh,pie_0,alpha_0,beta_0, data):
  pies=np.zeros((T+1,K))
  dj=alpha_0.shape[1]
  db=beta_0.shape[1]
  alphas=np.zeros((T+1,K,dj))
  betas=np.zeros((T+1,K,db))
  gammas=np.zeros((T,N,K))
  probs=np.zeros(T+1)
  pies[0]=pie_0
  alphas[0]=alpha_0
  betas[0]=beta_0
  probs[0]=logprob_EM(pies[0],alphas[0],betas[0],data[:N])

  for t in range(T):
    #print("Starting step ", t+1)
    #print("Do E Step")
    gammas[t]=do_E_step_EM(pies[t],alphas[t],betas[t],data[:N])
    #print("Do M Step")
    pies_aux, alphas_aux, betas_aux = do_M_step_EM(gammas[t],data[:N])
    #print("Getting new Likelihood")
    logprob_aux=logprob_EM(pies_aux,alphas_aux,betas_aux,data[:N])
    #print("Checking convergence")
    if(logprob_aux-probs[t]>-thresh):
      probs[t+1]=logprob_aux
      pies[t+1]=pies_aux
      alphas[t+1]=alphas_aux
      betas[t+1]=betas_aux
    else:
      break
  return pies, alphas, betas, gammas, probs
  

# EM + priors

def do_E_step_EM_priors(pies,alphas,betas,data):
  N=len(data)
  K=len(pies)
  gammas=np.zeros((N,K))
  for n in range(N):
    gammas_aux=np.zeros(K)
    for k in range(K):
      gammas_aux[k]=pies[k]*multinomial(n=1,p=alphas[k]).pmf(data[n][0])*multinomial(n=1,p=betas[k]).pmf(data[n][1])
    gammas[n]=gammas_aux/np.sum(gammas_aux)
  return np.asarray(gammas)
  
def do_M_step_EM_priors(gammas,eta_pie,eta_alpha,eta_beta,data):
  N=gammas.shape[0]
  K=gammas.shape[1]
  pies=np.zeros(K)
  Nk=np.sum(gammas,axis=0)
  eta_pie_0=np.sum(eta_pie)
  eta_alpha_0=np.sum(eta_alpha,axis=1)
  eta_beta_0=np.sum(eta_beta,axis=1)
  pies=np.asarray(list(map(lambda k: (Nk[k]+eta_pie[k]-1.0)/(N+eta_pie_0-K), range(K))))
  dj=len(data[0][0])
  db=len(data[0][1])
  Nkj=np.zeros((K,dj))
  Nkb=np.zeros((K,db))
  for n in range(N):
    Nkj[:,np.argmax(data[n][0])]+=gammas[n,:]
    Nkb[:,np.argmax(data[n][1])]+=gammas[n,:]
  alphas=np.zeros((K,dj))
  betas=np.zeros((K,db))
  for k in range(K):
    alphas[k]=np.asarray(list(map(lambda j: (Nkj[k,j]+eta_alpha[k,j]-1.0)/(Nk[k]+eta_alpha_0[k]-dj), range(dj))))
    betas[k]=np.asarray(list(map(lambda b: (Nkb[k,b]+eta_beta[k,b]-1.0)/(Nk[k]+eta_beta_0[k]-db), range(db))))
  return pies, alphas, betas
  
def logprob_EM_priors(pies,alphas,betas,data):
  N=len(data)
  K=len(pies)
  logprob=0.0
  for n in range(N):
    aux=0.0
    for k in range(K):
      aux+=pies[k]*multinomial(n=1,p=alphas[k]).pmf(data[n][0])*multinomial(n=1,p=betas[k]).pmf(data[n][1])
    logprob+=np.log(aux)/N #normalizo por dato
  return logprob
  
def do_EM_priors_algorithm(N,K,T, thresh,pie_0,alpha_0,beta_0, eta_pie, eta_alpha, eta_beta, data):
  pies=np.zeros((T+1,K))
  dj=alpha_0.shape[1]
  db=beta_0.shape[1]
  alphas=np.zeros((T+1,K,dj))
  betas=np.zeros((T+1,K,db))
  gammas=np.zeros((T,N,K))
  probs=np.zeros(T+1)
  pies[0]=pie_0
  alphas[0]=alpha_0
  betas[0]=beta_0
  probs[0]=logprob_EM_priors(pies[0],alphas[0],betas[0],data[:N])

  for t in range(T):
    #print("Starting step ", t+1)
    #print("Do E Step")
    gammas[t]=do_E_step_EM_priors(pies[t],alphas[t],betas[t],data[:N])
    #print("Do M Step")
    pies_aux, alphas_aux, betas_aux = do_M_step_EM_priors(gammas[t],eta_pie, eta_alpha, eta_beta, data[:N])
    #print("Getting new Likelihood")
    logprob_aux=logprob_EM_priors(pies_aux,alphas_aux,betas_aux,data[:N])
    #print("Checking convergence")
    if(logprob_aux-probs[t]>-thresh):
      probs[t+1]=logprob_aux
      pies[t+1]=pies_aux
      alphas[t+1]=alphas_aux
      betas[t+1]=betas_aux
    else:
      break
  return pies, alphas, betas, gammas, probs
  
# VB

def do_E_step_VB(gamma_pie,gamma_alpha,gamma_beta,data):
  N=len(data)
  K=len(gamma_pie)
  rho=np.zeros((N,K))
  r=np.zeros((N,K))
  for n in range(N):
    jn=np.argmax(data[n][0])
    bn=np.argmax(data[n][1])
    for k in range(K):
      rho[n,k]=np.exp(digamma(gamma_pie[k])-digamma(np.sum(gamma_pie))+digamma(gamma_alpha[k,jn])-digamma(np.sum(gamma_alpha[k]))+digamma(gamma_beta[k,bn])-digamma(np.sum(gamma_beta[k])))
    r[n,:]=rho[n,:]/(np.sum(rho[n]))
  return r
  
def do_M_step_VB(eta_pie,eta_alpha,eta_beta,r,data):
  N=len(data)
  K=len(eta_pie)
  dj=eta_alpha.shape[1]
  db=eta_beta.shape[1]
  Nkj=np.zeros((K,dj))
  Nkb=np.zeros((K,db))
  Nk=np.sum(r,axis=0)
  for n in range(N):
    jn=np.argmax(data[n][0])
    bn=np.argmax(data[n][1])
    Nkj[:,jn]+=r[n,:]
    Nkb[:,bn]+=r[n,:]
  if(np.allclose(np.sum(Nkj,axis=1),Nk) and np.allclose(np.sum(Nkb,axis=1),Nk)):
    return eta_pie+Nk, eta_alpha + Nkj, eta_beta + Nkb
  else:
    return "Error"
    
def ELBO(eta_pie,eta_alpha,eta_beta,gamma_pie, gamma_alpha, gamma_beta, r,data):
  score = 0
  N=len(data)
  K=len(eta_pie)
  dj=eta_alpha.shape[1]
  db=eta_beta.shape[1]
  Nkj=np.zeros((K,dj))
  Nkb=np.zeros((K,db))
  Nk=np.sum(r,axis=0)
  for n in range(N):
    jn=np.argmax(data[n][0])
    bn=np.argmax(data[n][1])
    Nkj[:,jn]+=r[n,:]
    Nkb[:,bn]+=r[n,:]
  
  #E[log(p(X|Z,alpha,beta))]
  for k in range(K):
    #tmp1=Nk[k]*(digamma(gamma_pie[k])-digamma(np.sum(gamma_pie)))
    tmp1=0.0
    tmp2=0.0
    for j in range(dj):
      tmp2+=Nkj[k,j]*(digamma(gamma_alpha[k,j])-digamma(np.sum(gamma_alpha[k])))
    tmp3=0.0
    for b in range(db):
      tmp3+=Nkb[k,b]*(digamma(gamma_beta[k,b])-digamma(np.sum(gamma_beta[k])))
    score+=tmp1+tmp2+tmp3

  #E[log(p(z|eta)-log(q(Z|gamma))]
  tmp1=0.0
  for n in range(N):
    for k in range(K):
      tmp1+=r[n,k]*(digamma(gamma_pie[k])-digamma(np.sum(gamma_pie))-np.log(r[n,k]))
  score+=tmp1

  #compensate
  #score=score/N

  #E[log(p(pi|eta))-log(q(pi|gamma))]
  #tmp1=0.0
  #tmp1=np.log(beta_function_k(gamma_pie))-np.log(beta_function_k(eta_pie))
  tmp1=log_beta_function_k(gamma_pie)-log_beta_function_k(eta_pie)
  for k in range(K):
    tmp1+=(eta_pie[k]-gamma_pie[k])*(digamma(gamma_pie[k])-digamma(np.sum(gamma_pie)))
  score+=tmp1

  #E[log(p(alpha|eta))-log(q(alpha|gamma))]
  tmp1=0.0
  for k in range(K):
    #tmp2=0.0
    #tmp2=np.log(beta_function_k(gamma_alpha[k]))-np.log(beta_function_k(eta_alpha[k]))
    tmp2=log_beta_function_k(gamma_alpha[k])-log_beta_function_k(eta_alpha[k])
    for j in range(dj):
      tmp2+=(eta_alpha[k,j]-gamma_alpha[k,j])*(digamma(gamma_alpha[k,j])-digamma(np.sum(gamma_alpha[k])))
    tmp1+=tmp2
  score+=tmp1

  #E[log(p(beta|eta))-log(q(beta|gamma))]
  tmp1=0.0
  for k in range(K):
    #tmp2=0.0
    #tmp2=np.log(beta_function_k(gamma_beta[k]))-np.log(beta_function_k(eta_beta[k]))
    tmp2=log_beta_function_k(gamma_beta[k])-log_beta_function_k(eta_beta[k])
    for b in range(db):
      tmp2+=(eta_beta[k,b]-gamma_beta[k,b])*(digamma(gamma_beta[k,b])-digamma(np.sum(gamma_beta[k])))
    tmp1+=tmp2
  score+=tmp1
    
  return score
  
def do_VB_algorithm(N,K,T, thresh,gamma_pie_0,gamma_alpha_0,gamma_beta_0, eta_pie, eta_alpha, eta_beta, X):
  pies=np.zeros((T+1,K))
  dj=gamma_alpha_0.shape[1]
  db=gamma_beta_0.shape[1]
## posterior definition
  gamma_pie=np.zeros((T+1,K))
  gamma_alpha=np.zeros((T+1,K,dj))
  gamma_beta=np.zeros((T+1,K,db))
  rmatrix=np.zeros((T,N,K))
  probs=np.zeros(T+1)

  ## initialize gammas
  gamma_pie[0]=gamma_pie_0
  gamma_alpha[0]=gamma_alpha_0
  gamma_beta[0]=gamma_beta_0

  for t in range(T):
    #print("Starting step ", t+1)
    #print("Do E Step")
    rmatrix[t]=do_E_step_VB(gamma_pie[t],gamma_alpha[t],gamma_beta[t],X[:N])
    #print("Calculatin logprob from E step")
    probs[t]=ELBO(eta_pie,eta_alpha,eta_beta,gamma_pie[t],gamma_alpha[t],gamma_beta[t],rmatrix[t],X[:N])
    #print("Do M Step")
    gamma_pie_aux, gamma_alpha_aux, gamma_beta_aux = do_M_step_VB(eta_pie,eta_alpha,eta_beta,rmatrix[t],X[:N])
    #print("Getting new Likelihood")
    logprob_aux=ELBO(eta_pie,eta_alpha,eta_beta,gamma_pie_aux,gamma_alpha_aux,gamma_beta_aux,rmatrix[t],X[:N])
    #print("Checking convergence")
    #if(1.0>0.0):
    #if(np.allclose(gamma_pie[t],gamma_pie_aux)==False or np.allclose(gamma_alpha[t],gamma_alpha_aux)==False or np.allclose(gamma_beta[t],gamma_beta_aux)==False or logprob_aux >= probs[t] ):
    if(abs(logprob_aux) <= (1.0-thresh)*abs(probs[t]) ):
      probs[t+1]=logprob_aux
      gamma_pie[t+1]=gamma_pie_aux
      gamma_alpha[t+1]=gamma_alpha_aux
      gamma_beta[t+1]=gamma_beta_aux
    else:
      break
  return gamma_pie, gamma_alpha, gamma_beta, rmatrix, probs
  
# Gibbs sampler

def one_Gibbs_step(Zini,data,eta_pie,eta_alpha,eta_beta):
  N=Zini.shape[0]
  K=Zini.shape[1]
  dj=len(data[0][0])
  db=len(data[0][1])
  Nk=np.sum(Zini,axis=0)
  Nkj=np.zeros((K,dj))
  Nkb=np.zeros((K,db))
  for n in range(N):
    jn=np.argmax(data[n][0])
    bn=np.argmax(data[n][1])
    Nkj[:,jn]+=Zini[n]
    Nkb[:,bn]+=Zini[n]
  pie=dirichlet.rvs(alpha=eta_pie+Nk,size=1)[0]
  alpha0=dirichlet.rvs(alpha=eta_alpha[0]+Nkj[0],size=1)[0]
  alpha1=dirichlet.rvs(alpha=eta_alpha[1]+Nkj[1],size=1)[0]
  beta0=dirichlet.rvs(alpha=eta_beta[0]+Nkb[0],size=1)[0]
  beta1=dirichlet.rvs(alpha=eta_beta[1]+Nkb[1],size=1)[0]
  alpha=np.hstack([alpha0.reshape(-1,1),alpha1.reshape(-1,1)]).T
  beta=np.hstack([beta0.reshape(-1,1),beta1.reshape(-1,1)]).T
  Zfin=np.zeros((N,K))
  for n in range(N):
    jn=np.argmax(data[n][0])
    bn=np.argmax(data[n][1])
    Zfin[n]=multinomial.rvs(p=[pie[0]*alpha[0,jn]*beta[0,bn]/(pie[0]*alpha[0,jn]*beta[0,bn]+pie[1]*alpha[1,jn]*beta[1,bn]),pie[1]*alpha[1,jn]*beta[1,bn]/(pie[0]*alpha[0,jn]*beta[0,bn]+pie[1]*alpha[1,jn]*beta[1,bn])],n=1,size=1)[0]
  return Zfin, pie, alpha, beta

def one_Gibbs_step_optimized(Zini,Njb,eta_pie,eta_alpha,eta_beta):
  N=np.sum(Njb)
  K=Zini.shape[0]
  dj=Zini.shape[1]
  db=Zini.shape[2]
  Nkj = np.sum(Zini,axis=2)
  Nkb = np.sum(Zini,axis=1)
  Nk=np.sum(Nkj,axis=1)

  pie=dirichlet.rvs(alpha=eta_pie+Nk,size=1)[0]
  alpha0=dirichlet.rvs(alpha=eta_alpha[0]+Nkj[0],size=1)[0]
  alpha1=dirichlet.rvs(alpha=eta_alpha[1]+Nkj[1],size=1)[0]
  beta0=dirichlet.rvs(alpha=eta_beta[0]+Nkb[0],size=1)[0]
  beta1=dirichlet.rvs(alpha=eta_beta[1]+Nkb[1],size=1)[0]
  alpha=np.hstack([alpha0.reshape(-1,1),alpha1.reshape(-1,1)]).T
  beta=np.hstack([beta0.reshape(-1,1),beta1.reshape(-1,1)]).T
  Zfin=np.zeros((K,dj,db))
  for j in range(dj):
    for b in range(db):
      Zfin[:,j,b]=multinomial.rvs(p=[pie[0]*alpha[0,j]*beta[0,b]/(pie[0]*alpha[0,j]*beta[0,b]+pie[1]*alpha[1,j]*beta[1,b]),pie[1]*alpha[1,j]*beta[1,b]/(pie[0]*alpha[0,j]*beta[0,b]+pie[1]*alpha[1,j]*beta[1,b])],n=Njb[j,b],size=1)[0]
  return Zfin, pie, alpha, beta
  
def do_homemade_Gibbs_sampling(Zini,data, eta_pie,eta_alpha,eta_beta,T,burnout,keep_every):
  N=Zini.shape[0]
  K=Zini.shape[1]
  dj=len(data[0][0])
  db=len(data[0][1])
  Z_list=np.zeros((T,N,K))
  pie_list=np.zeros((T,K))
  alpha_list=np.zeros((T,K,dj))
  beta_list=np.zeros((T,K,db))
  Z_aux = Zini
  pie_aux=np.zeros(K)
  alpha_aux=np.zeros((K,dj))
  beta_aux=np.zeros((K,db))
  ## burnout to erase dependence on initial step
  for ind_burn in range(burnout):
    Z_aux, pie_aux, alpha_aux, beta_aux = one_Gibbs_step(Zini,data,eta_pie,eta_alpha,eta_beta)
  ## now lets save every keep_every until I have T samples
  Z_list[0]=Z_aux
  for n in range(T*keep_every):
    Z_aux, pie_aux, alpha_aux, beta_aux = one_Gibbs_step(Z_aux,data,eta_pie,eta_alpha,eta_beta)
    if(float(n)%float(keep_every) == 0):
      Z_list[int(n/keep_every)]=Z_aux
      pie_list[int(n/keep_every)]=pie_aux
      alpha_list[int(n/keep_every)]=alpha_aux
      beta_list[int(n/keep_every)]=beta_aux
  return Z_list, pie_list, alpha_list, beta_list

def do_homemade_Gibbs_sampling_optimized(Zini,Njb, eta_pie,eta_alpha,eta_beta,T,burnout,keep_every):
  N=np.sum(Njb)
  K=Zini.shape[0]
  dj=Zini.shape[1]
  db=Zini.shape[2]

  Z_list=np.zeros((T,K,dj,db))
  pie_list=np.zeros((T,K))
  alpha_list=np.zeros((T,K,dj))
  beta_list=np.zeros((T,K,db))
  Z_aux = Zini
  pie_aux=np.zeros(K)
  alpha_aux=np.zeros((K,dj))
  beta_aux=np.zeros((K,db))
  ## burnout to erase dependence on initial step
  for ind_burn in range(burnout):
    Z_aux, pie_aux, alpha_aux, beta_aux = one_Gibbs_step_optimized(Zini,Njb,eta_pie,eta_alpha,eta_beta)
  ## now lets save every keep_every until I have T samples
  # Z_list[0]=Z_aux
  for n in range(T*keep_every):
    Z_aux, pie_aux, alpha_aux, beta_aux = one_Gibbs_step_optimized(Z_aux,Njb,eta_pie,eta_alpha,eta_beta)
    if(float(n)%float(keep_every) == 0):
      Z_list[int(n/keep_every)]=Z_aux
      pie_list[int(n/keep_every)]=pie_aux
      alpha_list[int(n/keep_every)]=alpha_aux
      beta_list[int(n/keep_every)]=beta_aux
  return Z_list, pie_list, alpha_list, beta_list
  
def thin_a_sample(sample,keep_every):
  nwalker=sample.shape[0]
  T=sample.shape[1]
  final_length= int(float(T)/float(keep_every))
  return sample[:,0:T:keep_every]

def do_log_likelihood_estimate(samples,val,window=0.01,thresh=0.001):
  nk = np.sum(np.where(samples<=val+0.5*window,1.0,0.0)*np.where(samples>=val-0.5*window,1.0,0.0))
  if(nk>thresh*len(samples)):
    return np.log(nk/(len(samples)*window))
  else:
    return np.log(thresh)
