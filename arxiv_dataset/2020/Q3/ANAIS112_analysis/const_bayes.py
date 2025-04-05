# BAYES FACTOR
# w free , constant bg
# uses the residual rate


import numpy as np 
from scipy import optimize , stats
from scipy.special import ndtri
import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
import nestle
import time
from multiprocessing import Pool
from contextlib import closing
tol=0.1



f1="fig2_16.dat"
data1 = np.loadtxt(f1)                    
data1=np.transpose(data1)

f2="fig2_26.dat"
data2 = np.loadtxt(f2)                    
data2=np.transpose(data2)

#-----------------MODULATION-------------------

def fit_cosine(x,S,w,t_0):
    	return S*np.cos(w*(x+t_0))

def log_likelihood_cosine(P,DATA):
    y_fit=fit_cosine(DATA[0],P[0],P[1],P[2])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))


def chi2_val_cosine(P,DATA):
        sigma=DATA[2]
        y_fit=fit_cosine(DATA[0],P[0],P[1],P[2])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

#degrees of freedom
def dof_val(P,DATA):
  return len(DATA[0]) - len(P)

#chi squared likelihood function
def chi2L_cosine(P,DATA):
  chi2 = chi2_val_cosine(P,DATA)
  dof = dof_val(P,DATA)
  return stats.chi2(dof).pdf(chi2)


#=========================constant background function==========================
def fit_const(x,k):
    	return k[0]*x**0

def log_likelihood_const(k,DATA):
    y_fit=fit_const(DATA[0],k)
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))


def chi2_val_const(k,DATA):
        sigma=DATA[2]
        y_fit=fit_const(DATA[0],k)
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

#chi squared likelihood function
def chi2L_const(k,DATA):
  chi2 = chi2_val_const(k,DATA)
  dof = dof_val(k,DATA)
  return stats.chi2(dof).pdf(chi2)


def prior_transform_const(k,data):

    A_lim=np.max(np.abs(data[1]))
    return -A_lim+2*A_lim*k

def nestle_const(k,DATA):
    f = lambda k: log_likelihood_const(k, DATA)
    prior = lambda k: prior_transform_const(k,DATA)
    res = nestle.sample(f, prior, 1, method='multi',npoints=2000)
    print (res.summary())
    return res.logz


def prior_transform_cosine(P,data):
        B_lim=np.max(np.abs(data[1]))
        return np.array([-B_lim*P[0]+2*B_lim*P[0] , P[1]*0.61686+0.01153 , P[2]*365])
#        return np.array([-B_lim+2*B_lim*P[0] ,0.017214+1.36e-05*ndtri(P[2]),P[2]*365]) # Gaussian prior around DAMA's best-fit period 
#        return  np.array([-B_lim+2*B_lim*P[0] ,P[1]*0.61686+0.01153 ,145+5*ndtri(P[2])]) # Gaussian prior around DAMA's best-fit phase



def nestle_cosine(P,DATA):
    f1 = lambda P: log_likelihood_cosine(P, DATA)

    prior = lambda P: prior_transform_cosine(P,DATA)
    with closing(Pool(processes=16)) as pool:
	    sampler = NestedSampler(log_likelihood_cosine, prior_transform_cosine, 3, 
                                        bound='multi',nlive=1024,sample='rwalk',logl_args=[DATA],ptform_args=[DATA],pool=pool,queue_size=16)
	    t0 = time.time()
	    sampler.run_nested(dlogz=tol, print_progress=False) # don't output progress bar
	    t0 = time.time()
    res=sampler.results
    print (res.summary())
    print (res.logz)
    
    return res.logz[-1]
  
def bayesian(P,k,data):
    Zc=nestle_cosine(P,data)
    Zk=nestle_const(k,data)
    Z= np.exp(Zc-Zk)
    print('Cosine logz=',Zc,', Const logz=',Zk,'\nBayes Factor: ',Z)
    

g=[0]

k1 = optimize.minimize(chi2_val_const, g,args=data1,method='SLSQP')
k2 = optimize.minimize(chi2_val_const, g,args=data2,method='SLSQP')

guess=[0,0.0172,0]
bnd = [(None,None),(2.*np.pi/545.0 , 2.*np.pi/10.0), (None,None)]

cos1 = optimize.minimize(chi2_val_cosine, guess,args=data1,method='SLSQP',bounds=bnd)
cos2 = optimize.minimize(chi2_val_cosine, guess,args=data2,method='SLSQP',bounds=bnd)    

print("H_0 : No modulation, constant fit")
print("H_1 : Cosine modulation")

print("Gaussian prior on period")
print("\n1-6keV")
bayesian(cos1.x, k1.x, data1)

print("\n2-6keV")
bayesian(cos2.x, k2.x, data2)
