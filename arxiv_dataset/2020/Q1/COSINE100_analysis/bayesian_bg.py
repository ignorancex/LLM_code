
# this code calculates evidence for background only hypothesis
 

import numpy as np 
from scipy import optimize , stats
import os
import dynesty
import time
from dynesty import NestedSampler
from multiprocessing import Pool
from contextlib import closing



#f=os.path.expanduser("~")+"/Desktop/COSINE100/data/c2_data.txt"
data1 = np.loadtxt("crystal2.txt",delimiter=',')                    

#g=os.path.expanduser("~")+"/Desktop/COSINE100/data/c3_data.txt"
data2 = np.loadtxt("crystal3.txt",delimiter=',')                    

#h=os.path.expanduser("~")+"/Desktop/COSINE100/data/c4_data.txt"
data3 = np.loadtxt("crystal4.txt",delimiter=',')                    

#k=os.path.expanduser("~")+"/Desktop/COSINE100/data/c6_data.txt"
data4 = np.loadtxt("crystal6.txt",delimiter=',')                    

#l=os.path.expanduser("~")+"/Desktop/COSINE100/data/c7_data.txt"
data5 = np.loadtxt("crystal7.txt",delimiter=',')                    

data=np.hstack((data1,data2,data3,data4,data5))
tol=0.1

def fit_bg(x,c,p0,p1):
    	return c + p0*np.exp(-np.log(2)*x/p1)


def log_likelihood_bg(P):
    
        sigma1=[ data1[3,i] for i in range(len(data1[0])) ]
        y_fit1=fit_bg(data1[0],P[0],P[1],P[2])
        
        sigma2=[  data2[3,i]  for i in range(len(data2[0])) ]
        y_fit2=fit_bg(data2[0],P[3],P[4],P[5])
        
        sigma3=[ data3[3,i]  for i in range(len(data3[0])) ]
        y_fit3=fit_bg(data3[0],P[6],P[7],P[8])
        
        sigma4=[  data4[3,i]  for i in range(len(data4[0])) ]
        y_fit4=fit_bg(data4[0],P[9],P[10],P[11])
        
        sigma5=[ data5[3,i]  for i in range(len(data5[0])) ]
        y_fit5=fit_bg(data5[0],P[12],P[13],P[14])    

        y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
        sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))

        return sum(stats.norm.logpdf(*args) for args in zip(data[1],y_fit,sigma))


a1=100.0*np.max(data[1])
b1=30000.0


def prior_transform_bgnoprior(P):
        return np.array([a1*P[0],a1*P[1],b1*P[2],a1*P[3],a1*P[4],P[5]*b1,a1*P[6],P[7]*a1,P[8]*b1,P[9]*a1,P[10]*a1,P[11]*b1,P[12]*a1,P[13]*a1,P[14]*b1])




def nestle_multi_bg():
        with closing(Pool(processes=24)) as pool:
    # Run nested sampling
                sampler = NestedSampler(log_likelihood_bg, prior_transform_bgnoprior, 15, 
                                        bound='balls', nlive=1024,sample='rwalk',pool=pool,queue_size=24)
                t0 = time.time()
                sampler.run_nested(dlogz=tol, print_progress=False) # don't output progress bar
                t1 = time.time()
                pool.terminate
        res=sampler.results
        print(res.summary())
        return res.logz[-1],res.logzerr[-1]

#----------------------------------------------------------------------------

print ("Evidence for Background")
Z1,Z1err = nestle_multi_bg()
print (Z1)
print (Z1err)
