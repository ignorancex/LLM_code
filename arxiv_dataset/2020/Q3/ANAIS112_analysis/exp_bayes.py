# BAYES FACTOR
# w free, exp bg
# uses the total Cosine-100 event rate


import numpy as np 
from scipy import optimize , stats
from scipy.special import ndtri
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
import nestle
from multiprocessing import Pool
from contextlib import closing
tol=0.1

f="data16.dat"
data1 = np.loadtxt(f)                    
data1=np.transpose(data1)

#a1=100*np.max(data1[1])

f2="data26.dat"
data2 = np.loadtxt(f2)                    
data2=np.transpose(data2)




def fit_bg(x,R0,R1,tau):
    	return R0 + R1*np.exp(-x/tau)

#background-only best fit (for each data set separately)
guess=[2.0,1.0,1000]

def chi2_bg(P,DATA):
        sigma=DATA[2]
        y_fit=fit_bg(DATA[0],P[0],P[1],P[2])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

bnd=((1e-10,None),(1e-10,None),(1e-10,None))
bg1 = optimize.minimize(chi2_bg, guess,args=data1,method='SLSQP',bounds=bnd)
bg2= optimize.minimize(chi2_bg, guess,args=data2,method='SLSQP',bounds=bnd)


def log_likelihood_bg(P,DATA):
        y_fit=fit_bg(DATA[0],P[0],P[1],P[2])
        return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))


#-----------------MODULATION-------------------

def fit_cosine(x,S,w,t_0,R0,R1,tau):
    	return R0 + R1*np.exp(-x/tau) + S*np.cos(w*(x+t_0))

def chi2_cosine(P,DATA):
    y_fit=fit_cosine(DATA[0],P[0],P[1],P[2],P[3],P[4],P[5])
    sigma=DATA[2]
    r = (DATA[1] - y_fit)/sigma
    return np.sum(r**2)

def log_likelihood_cosine(P,DATA):
    y_fit=fit_cosine(DATA[0],P[0],P[1],P[2],P[3],P[4],P[5])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))

guess1=[0,0,0,bg1.x[0],bg1.x[1],bg1.x[2]]
guess2=[0,0,0,bg2.x[0],bg2.x[1],bg2.x[2]]

bnd = [(None,None),(2.*np.pi/545.0 , 2.*np.pi/10.0),(None,None),(1e-10,None),(1e-10,None),(1e-10,None)]

cos1 = optimize.minimize(chi2_cosine, guess1,args=data1,method='SLSQP',bounds=bnd)
cos2 = optimize.minimize(chi2_cosine, guess2,args=data2,method='SLSQP',bounds=bnd)    


def dof_val(P,DATA):
  return len(DATA[0]) - len(P)

#chi squared likelihood function
def chi2L_cosine(P,DATA):
  chi2 = chi2_cosine(P,DATA)
  dof = dof_val(P,DATA)
  return stats.chi2(dof).pdf(chi2)

def chi2L_bg(k,DATA):
  chi2 = chi2_bg(k,DATA)
  dof = dof_val(k,DATA)
  return stats.chi2(dof).pdf(chi2)




def prior_transform_bg(k,DATA):
	a1=np.max(np.abs(DATA[1]))
	return np.array([a1*k[0],a1*k[1],550*k[2]+0.1])


def nestle_bg1(k=bg1.x,DATA=data1):
    with closing(Pool(processes=16)) as pool:
	    sampler = NestedSampler(log_likelihood_bg, prior_transform_bg, 3, 
                                        bound='multi',nlive=1024,sample='rwalk',logl_args=[DATA],ptform_args=[DATA],pool=pool,queue_size=16)
	    sampler.run_nested(dlogz=0.1, print_progress=False) # don't output progress bar
    res=sampler.results
    print (res.summary())
    print (res.logz)

    return res.logz[-1]

def nestle_bg2(P=bg2.x,DATA=data2):
        with closing(Pool(processes=16)) as pool:
                sampler = NestedSampler(log_likelihood_bg, prior_transform_bg, 3,
                                        bound='multi',nlive=1024,sample='rwalk',logl_args=[DATA],ptform_args=[DATA],pool=pool,queue_size=16)
                sampler.run_nested(dlogz=0.1, print_progress=False) # don't output progress bar                                               
        res=sampler.results
        print (res.summary())
        print (res.logz)
        return res.logz[-1]

def prior_transform_cos(P,DATA):
	a1=np.max(np.abs(DATA[1]))
	return np.array([a1*P[0],a1*P[1],550*P[2]+0.1,a1*P[3],P[4]*0.61686+0.01153 , P[5]*365]  )
#	return np.array([a1*P[0],a1*P[1],550*P[2]+0.1,a1*P[3],0.017214+1.36e-05*ndtri(P[4]), P[5]*365]  ) Gaussian prior around DAMA period
#	return np.array([a1*P[0],a1*P[1],550*P[2]+0.1,a1*P[3],P[4]*0.61686+0.01153, 145+5*ndtri(P[5])]  ) Gaussian prior around DAMA phase 



def nestle_cos1(P=cos1.x,DATA=data1):
    with closing(Pool(processes=16)) as pool:
            sampler = NestedSampler(log_likelihood_cosine, prior_transform_cos, 6,
                                        bound='multi',nlive=1024,sample='rwalk',logl_args=[DATA],ptform_args=[DATA],pool=pool,queue_size=16)
            sampler.run_nested(dlogz=0.1, print_progress=False) # don't output progress bar                                        
    res=sampler.results
    print (res.summary())
    return res.logz[-1]

def nestle_cos2(P=cos2.x,DATA=data2):
    with closing(Pool(processes=16)) as pool:
            sampler = NestedSampler(log_likelihood_cosine, prior_transform_cos, 6,
                                        bound='multi',nlive=1024,sample='rwalk',logl_args=[DATA],ptform_args=[DATA],pool=pool,queue_size=16)
            sampler.run_nested(dlogz=0.1, print_progress=False) # don't output progress bar                                                       
    res=sampler.results
    print (res.summary())
    print (res.logz)
    return res.logz[-1]
 
def bayesian1():
    Zc=nestle_cos1()
    Zk=nestle_bg1()
    Z= np.exp(Zc-Zk)
    print('Cosine logz=',Zc,', Const logz=',Zk,'\nBayes Factor: ',Z)
   
def bayesian2():
    Zc=nestle_cos2()
    Zk=nestle_bg2()
    Z= np.exp(Zc-Zk)
    print('Cosine logz=',Zc,', Const logz=',Zk,'\nBayes Factor: ',Z)
    
    

print("\n1-6keV")

bayesian1()

print("\n2-6keV")

bayesian2()


#==============================================================================
