# -*- coding: utf-8 -*-
"""
AdaBoost and robust one-bit compressed sensing

Geoffrey Chinot, Felix Kuchelmeister, Matthias Löffler and Sara van de Geer
"""

###############################################################################
# Packages
###############################################################################

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

###############################################################################
# Functions
###############################################################################

#creates a random s-sparse vector with dimension p and length 1
def sparse_rademacher_prior_beta(s,p):
    tmp = np.random.randint(0,2,s)*2-1 #Rademacher sample
    tmp = tmp/np.sqrt(s) #normalize
    tmp = np.append(tmp,[0]*(p-s))
    np.random.shuffle(tmp)
    return(tmp)

#sign-flips (noise)
def sign_flip(n,corr): #corr is the number of corrupted observations
    #find elements with largest margin
    tmp = np.repeat(1,n)
    tmp[np.random.choice(range(0,n),corr,replace = False)] = -1
    return(tmp)

#how the prediction error is simulated
def mc_loss(beta_0,beta_hat,n_sim):
    p = np.size(beta_0)
    X_mc = np.random.normal(size = (n_sim,p)) #simulate features
    y_0 = np.sign(np.dot(X_mc,beta_0)) #simulate correct labels
    y_hat = np.sign(np.dot(X_mc,beta_hat)) #simulate predicted labels
    return(sum(y_0!=y_hat)/n_sim) #report average 0/1-loss over n_sim simulations

#how p is calculated
def f_1(n):
    return(10*n)

#how T is calculated
def f_T(n,p,s,corr,eps):
    T = (n*np.sqrt(s+corr))**(2/3)*np.log(p)*eps**(-2)
    return(int(T))

def max_margin(n,p,yX):
    C = np.concatenate((-yX,yX),axis = 1) 
    d = np.repeat(-1,n)
    c = np.repeat(1,2*p)
    res = linprog(c, A_ub= C , b_ub= d, bounds = (0,None), method='interior-point', options = {'maxiter': 500000})
    hat_beta = res.x[0:p]-res.x[p:2*p]
    print("interpolation:", all(np.dot(yX,hat_beta)>=-1e-11))
    return(res)

def max_margin_beta(res,p):
    tmp = res.x[0:p] - res.x[p:2*p]
    tmp = tmp/np.linalg.norm(tmp,2)
    return(tmp)

#use this, as to obtain the margin, we do not divide by the ell_2_norm.
def max_margin_margin(res,p):
    tmp = res.x[0:p] - res.x[p:2*p]
    return(1/np.linalg.norm(tmp,1))

def Ada(n,p,yX,T,eps): #T is run time, eps is learning rate
    #initialize
    bet_tilde = np.zeros(p, dtype = float) 
    W = np.zeros(n, dtype = float)
    
    #rescale
    X_infty = np.max(np.abs(yX))
    yX_r = yX/X_infty
        
    for t in range(0,T):
        #calculate weights
        tmp = np.exp(-np.dot(yX_r,bet_tilde)) #auxiliary function
        if(any(tmp == np.inf)): #test if any value is infinite
            tmp_2 = np.zeros(n)
            tmp_2[tmp == np.inf] = 1

        W = tmp/np.sum(tmp)
    
        #find update direction
        candidate_index = 0
        candidate_value = 0
        for v in range(0,p):
            tmp = np.dot(W,yX_r[:,v])
            if(np.abs(tmp) > np.abs(candidate_value)):
                candidate_value = tmp
                candidate_index = v
                
        #update
        direction = np.zeros(p, dtype = float)
        direction[candidate_index] = 1
        bet_tilde = bet_tilde + eps*candidate_value*direction

    print("interpolation:", all(np.dot(yX,bet_tilde)>=-1e-11))
    return(bet_tilde/np.linalg.norm(bet_tilde,2))

###############################################################################
# Plot 1/3/4: Prediction error as number of observations n grows
###############################################################################

#initialize prediciton errors
    #prediction errors for max margin
ERROR_MAX_NORMAL  = np.ones([10], dtype = float)
ERROR_MAX_STUDENT = np.ones([10], dtype = float)
ERROR_MAX_UNIFORM = np.ones([10], dtype = float)
ERROR_MAX_LAPLACE = np.ones([10], dtype = float)

    #prediction errors for adaboost
ERROR_ADA_NORMAL  = np.ones([10], dtype = float)
ERROR_ADA_STUDENT = np.ones([10], dtype = float)
ERROR_ADA_UNIFORM = np.ones([10], dtype = float)
ERROR_ADA_LAPLACE = np.ones([10], dtype = float)

#initialize margins
    #margin for max margin
MARGIN_MAX_NORMAL  = np.ones([10], dtype = float)
MARGIN_MAX_STUDENT = np.ones([10], dtype = float)
MARGIN_MAX_UNIFORM = np.ones([10], dtype = float)
MARGIN_MAX_LAPLACE = np.ones([10], dtype = float)

    #margin for adaboost
MARGIN_ADA_NORMAL  = np.ones([10], dtype = float)
MARGIN_ADA_STUDENT = np.ones([10], dtype = float)
MARGIN_ADA_UNIFORM = np.ones([10], dtype = float)
MARGIN_ADA_LAPLACE = np.ones([10], dtype = float)

s = 5  # sparsity
eps = 0.2 #step size adaboost
n_mc = 10000 #number of simulations for approximation of prediction error
sequence_ = np.arange(100,1100,100) # number of observations

# number of corrupted observations
corr = 40 #for plot 1
#corr = 0 #for plots 3 and 4

np.random.seed(0) 

for i in range(0,10):   
    n = sequence_[i] # number of observations
    print('Iteration:', i+1, ', n = ', n)
    p = f_1(n) # dimension of observations
    T = f_T(n,p,s,corr,eps) # number of steps for adaboost. CHOOSE LOWER T TO SPEED UP COMPUTATION
    RAD = sign_flip(n, corr) #Corrputions of the labels
    beta = sparse_rademacher_prior_beta(s, p) # generating \beta^*
    
    #NORMAL
    X = np.random.normal(size = (n,p)) # design matrix normal
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*RAD
    yX = np.dot(np.diag(Y),X) 
          
        #MAX MARGIN
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    ERROR_MAX_NORMAL[i] = mc_loss(beta,beta_max_margin,n_mc) #estimating prediction error
    
    tmp = np.dot(yX, beta_max_margin)
    MARGIN_MAX_NORMAL[i] = np.min(tmp)/np.linalg.norm(beta_max_margin,1)
    
        #ADABOOST
    beta_Ada = Ada(n,p,yX,T,eps)
    ERROR_ADA_NORMAL[i] = mc_loss(beta,beta_Ada,n_mc) #estimating prediction error
    
    tmp = np.dot(yX, beta_Ada)
    MARGIN_ADA_NORMAL[i] = np.min(tmp)/np.linalg.norm(beta_Ada,1)

    #STUDENT
    X = np.random.standard_t(df = int(np.log(p))+1, size=(n,p)) # design matrix student
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*RAD
    yX = np.dot(np.diag(Y),X) 
    
        #MAX MARGIN
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    ERROR_MAX_STUDENT[i] = mc_loss(beta,beta_max_margin,n_mc) #estimating prediction error
    
    tmp = np.dot(yX, beta_max_margin)
    MARGIN_MAX_STUDENT[i] = np.min(tmp)/np.linalg.norm(beta_max_margin,1)

        #ADABOOST
    beta_Ada = Ada(n,p,yX,T,eps)
    ERROR_ADA_STUDENT[i] = mc_loss(beta,beta_Ada,n_mc) #estimating prediction error
    
    tmp = np.dot(yX, beta_Ada)
    MARGIN_ADA_STUDENT[i] = np.min(tmp)/np.linalg.norm(beta_Ada,1)
    
    #UNIFORM
    X = np.random.uniform(low = -(3/2)**(1/3), high = (3/2)**(1/3), size=(n,p))
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*RAD
    yX = np.dot(np.diag(Y),X) 
      
        #MAX MARGIN
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    ERROR_MAX_UNIFORM[i] = mc_loss(beta,beta_max_margin,n_mc) #estimating prediction error    
    
    tmp = np.dot(yX, beta_max_margin)
    MARGIN_MAX_UNIFORM[i] = np.min(tmp)/np.linalg.norm(beta_max_margin,1)
      
        #ADABOOST
    beta_Ada = Ada(n,p,yX,T,eps)
    ERROR_ADA_UNIFORM[i] = mc_loss(beta,beta_Ada,n_mc) #estimating prediction error
    
    tmp = np.dot(yX, beta_Ada)
    MARGIN_ADA_UNIFORM[i] = np.min(tmp)/np.linalg.norm(beta_Ada,1)
    
    #LAPLACE
    X = np.random.laplace(loc=0, scale = 1, size = (n,p))
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*RAD
    yX = np.dot(np.diag(Y),X) 
      
        #MAX MARGIN
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    ERROR_MAX_LAPLACE[i] = mc_loss(beta,beta_max_margin,n_mc) #estimating prediction error
    
    tmp = np.dot(yX, beta_max_margin)
    MARGIN_MAX_LAPLACE[i] = np.min(tmp)/np.linalg.norm(beta_max_margin,1)
    
        #ADABOOST
    beta_Ada = Ada(n,p,yX,T,eps)
    ERROR_ADA_LAPLACE[i] = mc_loss(beta,beta_Ada,n_mc) #estimating prediction error
    
    tmp = np.dot(yX, beta_Ada)
    MARGIN_ADA_LAPLACE[i] = np.min(tmp)/np.linalg.norm(beta_Ada,1)

#Plot 1 & 3 (Prediction error)
xlabel_ = '$n$'
ylabel_ = 'Prediction error'
legend_ = ["Normal","Student","Uniform","Laplace"]

plt.plot(sequence_,ERROR_MAX_NORMAL, "blue", linestyle = "solid")
plt.plot(sequence_,ERROR_MAX_STUDENT, "purple", linestyle = "solid")
plt.plot(sequence_,ERROR_MAX_UNIFORM, "orange", linestyle = "solid")
plt.plot(sequence_,ERROR_MAX_LAPLACE, "green", linestyle = "solid")

plt.plot(sequence_,ERROR_ADA_NORMAL, "blue", linestyle = "dashdot")
plt.plot(sequence_,ERROR_ADA_STUDENT, "purple", linestyle = "dashdot")
plt.plot(sequence_,ERROR_ADA_UNIFORM, "orange", linestyle = "dashdot")
plt.plot(sequence_,ERROR_ADA_LAPLACE, "green", linestyle = "dashdot")

plt.xlabel(xlabel_)
plt.ylabel(ylabel_)
plt.legend(legend_, frameon = False)

#Plot 4 (Margin)
xlabel_ = '$n$'
ylabel_ = 'Margin'
legend_ = ["Normal","Student","Uniform","Laplace"]

plt.plot(sequence_,MARGIN_MAX_NORMAL, "blue", linestyle = "solid")
plt.plot(sequence_,MARGIN_MAX_STUDENT, "purple", linestyle = "solid")
plt.plot(sequence_,MARGIN_MAX_UNIFORM, "orange", linestyle = "solid")
plt.plot(sequence_,MARGIN_MAX_LAPLACE, "green", linestyle = "solid")

plt.plot(sequence_,MARGIN_ADA_NORMAL, "blue", linestyle = "dashdot")
plt.plot(sequence_,MARGIN_ADA_STUDENT, "purple", linestyle = "dashdot")
plt.plot(sequence_,MARGIN_ADA_UNIFORM, "orange", linestyle = "dashdot")
plt.plot(sequence_,MARGIN_ADA_LAPLACE, "green", linestyle = "dashdot")

plt.xlabel(xlabel_)
plt.ylabel(ylabel_)
plt.legend(legend_, frameon = False)
    
###############################################################################
# Plot 2: Prediction error as contamination |O| grows
###############################################################################

#initialize prediciton errors
    #prediction errors for max margin
ERROR_MAX_NORMAL  = np.ones([10], dtype = float)
ERROR_MAX_STUDENT = np.ones([10], dtype = float)
ERROR_MAX_UNIFORM = np.ones([10], dtype = float)
ERROR_MAX_LAPLACE = np.ones([10], dtype = float)

    #prediction errors for adaboost
ERROR_ADA_NORMAL  = np.ones([10], dtype = float)
ERROR_ADA_STUDENT = np.ones([10], dtype = float)
ERROR_ADA_UNIFORM = np.ones([10], dtype = float)
ERROR_ADA_LAPLACE = np.ones([10], dtype = float)

n = 500 # number of observations
p = f_1(n) # dimension of observations
s = 5 # sparsity
eps = 0.2 # step size adaboost
n_mc = 10000 #number of simulations for approximation of prediction error

np.random.seed(0)    
    
for i in range(0,10):
    corr = 5*i
    print("#corruptions =",corr)
    T = f_T(n,p,s,corr,eps) # number of iterations for adaboost. CHOOSE LOWER T TO SPEED UP COMPUTATION
    RAD = sign_flip(n, corr) #Noise
    beta = sparse_rademacher_prior_beta(s, p) #generate beta
    
    #NORMAL
    X = np.random.normal(size = (n,p)) # design matrix normal
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*RAD
    yX = np.dot(np.diag(Y),X) 
    
        #MAX MARGIN
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    ERROR_MAX_NORMAL[i] = mc_loss(beta,beta_max_margin,n_mc) #estimating prediction error
    
        #ADABOOST
    beta_Ada = Ada(n,p,yX,T,eps)
    ERROR_ADA_NORMAL[i] = mc_loss(beta,beta_Ada,n_mc) #estimating prediction error
    
    #STUDENT
    X = np.random.standard_t(df = int(np.log(p))+1, size=(n,p)) # design matrix student
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*RAD
    yX = np.dot(np.diag(Y),X) 
    
        #MAX MARGIN
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    ERROR_MAX_STUDENT[i] = mc_loss(beta,beta_max_margin,n_mc) #estimating prediction error

        #ADABOOST
    beta_Ada = Ada(n,p,yX,T,eps)
    ERROR_ADA_STUDENT[i] = mc_loss(beta,beta_Ada,n_mc) #estimating prediction error

    #UNIFORM
    X = np.random.uniform(low = -(3/2)**(1/3), high = (3/2)**(1/3), size=(n,p))
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*RAD
    yX = np.dot(np.diag(Y),X) 
      
        #MAX MARGIN
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    ERROR_MAX_UNIFORM[i] = mc_loss(beta,beta_max_margin,n_mc) #estimating prediction error    

        #ADABOOST
    beta_Ada = Ada(n,p,yX,T,eps)
    ERROR_ADA_UNIFORM[i] = mc_loss(beta,beta_Ada,n_mc) #estimating prediction error
    
    #LAPLACE
    X = np.random.laplace(loc=0, scale = 1, size = (n,p))
    Y = np.sign(np.dot(X,beta)) # Y in the noiseless case
    Y = Y*RAD
    yX = np.dot(np.diag(Y),X) 
      
        #MAX MARGIN
    sol_max_margin = max_margin(n,p,yX) # finding the max-margin solution
    beta_max_margin = max_margin_beta(sol_max_margin,p) # extracting \hat{\beta}
    ERROR_MAX_LAPLACE[i] = mc_loss(beta,beta_max_margin,n_mc) #estimating prediction error
    
        #ADABOOST
    beta_Ada = Ada(n,p,yX,T,eps)
    ERROR_ADA_LAPLACE[i] = mc_loss(beta,beta_Ada,n_mc) #estimating prediction error

#Plot
xlabel_ = '# Corruptions'
ylabel_ = 'Prediction error'
legend_ = ["Normal","Student","Uniform","Laplace"]

plt.plot(sequence_,ERROR_MAX_NORMAL, "blue", linestyle = "solid")
plt.plot(sequence_,ERROR_MAX_STUDENT, "purple", linestyle = "solid")
plt.plot(sequence_,ERROR_MAX_UNIFORM, "orange", linestyle = "solid")
plt.plot(sequence_,ERROR_MAX_LAPLACE, "green", linestyle = "solid")

plt.plot(sequence_,ERROR_ADA_NORMAL, "blue", linestyle = "dashdot")
plt.plot(sequence_,ERROR_ADA_STUDENT, "purple", linestyle = "dashdot")
plt.plot(sequence_,ERROR_ADA_UNIFORM, "orange", linestyle = "dashdot")
plt.plot(sequence_,ERROR_ADA_LAPLACE, "green", linestyle = "dashdot")

plt.xlabel(xlabel_)
plt.ylabel(ylabel_)
plt.legend(legend_, frameon = False)