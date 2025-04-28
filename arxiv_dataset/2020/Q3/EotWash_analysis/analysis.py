import nestle
import numpy as np 
from scipy import optimize , stats
import matplotlib.pyplot as plt


d=np.loadtxt('Data.txt')
data=d.transpose()

x=data[0]
y=data[1]
yerr=data[2]


# the 3 hypotheses
def hypo1(x,a):
    	return a*x**0

def hypo2(x,a,m):
    	return a*np.exp(-m*x)
        
def hypo3(x,a,m,ph):
    	return a*np.cos(m*x + ph)

# log of likelihood
def log_likelihood1(P,DATA):
    y_fit=hypo1(DATA[0],P[0])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))
def log_likelihood2(P,DATA):
    y_fit=hypo2(DATA[0],P[0],P[1])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))
def log_likelihood3(P,DATA):
    y_fit=hypo3(DATA[0],P[0],P[1],P[2])
    return sum(stats.norm.logpdf(*args) for args    in zip(DATA[1],y_fit,DATA[2]))

# chi^2 values
def chi2_val1(P,DATA):
        sigma=DATA[2]
        y_fit=hypo1(DATA[0],P[0])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)
def chi2_val2(P,DATA):
        sigma=DATA[2]
        y_fit=hypo2(DATA[0],P[0],P[1])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)
def chi2_val3(P,DATA):
        sigma=DATA[2]
        y_fit=hypo3(DATA[0],P[0],P[1],P[2])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

#degrees of freedom
def dof_val(P,DATA):
  return len(DATA[0]) - len(P)


#chi squared likelihood function
def chi2L(chi2,dof):
  return stats.chi2(dof).pdf(chi2)


#bayesian priors
def prior_transform_1(P1,data):
    return 0.1*P1 -0.05
 
def prior_transform_2(P2,data):    
    return np.array([0.1*P2[0] -0.05, P2[1]*50.0] )

def prior_transform_3(P3,data):    
    return np.array([P3[0]*0.01, 60+P3[1]*10.0, 2*np.pi*P3[2]] )

# bayesian: NESTED SAMPLING
def nestle1(DATA):
    f = lambda P1: log_likelihood1(P1, DATA)
    prior = lambda P1: prior_transform_1(P1,DATA)
    res = nestle.sample(f, prior, 1, method='multi',
                    npoints=2000)
    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)
    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
    return res, samples_nestle

def nestle2(DATA):
    f = lambda P2: log_likelihood2(P2, DATA)
    prior = lambda P2: prior_transform_2(P2,DATA)
    res = nestle.sample(f, prior, 2, method='multi',
                    npoints=2000)
    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)
    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
    return res, samples_nestle

def nestle3(DATA):
    f = lambda P3: log_likelihood3(P3, DATA)
    prior = lambda P3: prior_transform_3(P3,DATA)
    res = nestle.sample(f, prior, 3, method='multi',
                    npoints=2000)
    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)
    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
    return res, samples_nestle


# get posterior samples using nested sampling
def get_posterior():
    z1,S1=nestle1(data)
    z2,S2=nestle2(data)
    z3,S3=nestle3(data)
    return z1,z2,z3,S1,S2,S3


# 1sigma errors in the parameters
def get_err(samples,ndim):
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(q)

# likelihood values
def likelihood1(P,DATA):
    y_fit=hypo1(DATA[0],P)
    return stats.norm.pdf(DATA[1],y_fit,DATA[2])

def likelihood2(P,DATA):
        y_fit=[hypo2(DATA[0],p[0],p[1]) for p in P]
        return stats.norm.pdf(DATA[1],y_fit,DATA[2])
def likelihood3(P,DATA):
    y_fit=[hypo3(DATA[0],p[0],p[1],p[2]) for p in P]
    return stats.norm.pdf(DATA[1],y_fit,DATA[2])

# WAIC calculation
def WAIC(z1,z2,z3,S1,S2,S3):
    
    # number of posterior samples 
    #s1=np.size(z1.samples, 0)
    #s2=np.size(z2.samples, 0)
    #s3=np.size(z3.samples, 0)
    
    #print(s1,s2,s3)
    S=6000  # number of samples to take from posterior
    P=S1
    P1=P[:S]
    lppd=0
    pwaic=0

    for d1 in data.T:
            lk=likelihood1(P1,d1)
#            l1=np.sum(lk)
            lppd+=np.log(np.mean(lk))
            pwaic+=np.var(np.log(lk))

    waic1=-2.0*(lppd-pwaic)
    print ("hypothesis 1")
    print (waic1)
    P=S2
    P2=P[:S]
    lppd=0
    pwaic=0

    for d1 in data.T:
            lk=likelihood2(P2,d1)
#            l1=np.sum(lk)
            lppd+=np.log(np.mean(lk))
            pwaic+=np.var(np.log(lk))
#            avgL=np.mean(np.log(lk))
#            lnP=np.sum(np.power(lk-avgL,2))
#            pwaic+=lnP/(S-1.0)

    waic2=-2.0*(lppd-pwaic)
    print ("hypothesis 2")
    print(waic2)

    lppd=0
    pwaic=0
    P=S3
    len(P)
    P3=P[:S]
    for d1 in data.T:
            lk=likelihood3(P3,d1)
#            l1=np.sum(lk)
            lppd+=np.log(np.mean(lk))
            pwaic+=np.var(np.log(lk))
    waic3=-2.0*(lppd-pwaic)
    print ("hypothesis 3")
    print(waic3)
        

#freq,aic,bic tests
def analysis(P1,P2,P3,DATA):
    chi2_1,chi2_2,chi2_3= chi2_val1(P1,DATA),chi2_val2(P2,DATA),chi2_val3(P3,DATA)
    dof1,dof2,dof3 =  dof_val(P1,DATA), dof_val(P2,DATA), dof_val(P3,DATA)
    
    aic1=-2*log_likelihood1(P1,DATA) + 2*1
    aic2=-2*log_likelihood2(P2,DATA) + 2*2
    aic3=-2*log_likelihood3(P3,DATA) + 2*3

    bic1=-2*log_likelihood1(P1,DATA) + 1*np.log(len(DATA[0]))
    bic2=-2*log_likelihood2(P2,DATA) + 2*np.log(len(DATA[0]))
    bic3=-2*log_likelihood3(P3,DATA) + 3*np.log(len(DATA[0]))
    del_bic2= bic1-bic2
    del_bic3= bic1-bic3   
    N=len(DATA[0])

    aicc1=(aic1 + 2.0*1*(1+1)/(N-1-1))
    aicc2=(aic2 + 2.0*2*(2+1)/(N-2-1))
    aicc3=(aic3 + 2.0*3*(3+1)/(N-3-1))
    del_aicc2= aicc1-aicc2
    del_aicc3= aicc1-aicc3   
    
    print("newtonian : alpha=",'%.4f'%P1[0])
    print("yukawa : alpha= ",'%.4f'%P2[0]," , m= ",'%.4f'%P2[1])
    print("oscillating : alpha= ",'%.4f'%P3[0]," , m= ",'%.4f'%P3[1]," , phase= ",'%.4f'%P3[2])

    print("\nchi2 value/dof\nnewtonian :", '%.4f'%chi2_1,'/',dof1,"\nyukawa :", '%.4f'%chi2_2,'/',dof2,"\noscillating:",'%.4f'%chi2_3,'/',dof3)
    print("\nchi2 likelihoods\nnewtonian :", '%.4f'%chi2L(chi2_1,dof1),"\nyukawa :", '%.4f'%chi2L(chi2_2,dof2),"\noscillating:", '%.4f'%chi2L(chi2_3,dof3))
    
    print("\nAICc values","\nnewtonian:",'%.2f'%aicc1,"\nyukawa:",'%.2f'%aicc2,"\noscillating:",'%.2f'%aicc3)
    print("\nBIC values","\nnewtonian:",'%.2f'%bic1,"\nyukawa:",'%.2f'%bic2,"\noscillating:",'%.2f'%bic3)
    
    print("\nyukawa vs newtonian")
    d=np.abs(chi2_2-chi2_1)
    print("difference in chi square values = ",'%.4f'%d)
    p=stats.chi2(dof1-dof2).sf(d)
    print ("p value=",'%.2f'%p)
    print("Confidence level : ",'%.2f'%stats.norm.isf(p),'\u03C3')
    print("delta AICc = ",'%.2f'%del_aicc2)
    print("delta BIC = ",'%.2f'%del_bic2)
    
    print("\noscillating vs newtonian")
    d=np.abs(chi2_3-chi2_1)
    print("difference in chi square values = ",'%.2f'%d)
    p=stats.chi2(dof1-dof3).sf(d)
    print ("p value=",'%.4f'%p)
    print("Confidence level : ",'%.2f'%stats.norm.isf(p),'\u03C3')
    print("delta AICc = ",'%.2f'%del_aicc3)
    print("delta BIC = ",'%.2f'%del_bic3)
    

#bayesian test
def bayesian(z1,z2,z3):

    print("\nbayesian")
    print("newtonian ","logz=",z1.logz,"; a=",np.mean(z1.samples))
    print("yukawa ","logz=",z2.logz,"; a=",np.mean(z2.samples[:,0]),"; m=",np.mean(z2.samples[:,1]))
    print("oscillating ","logz=",z3.logz,"; a=",np.mean(z3.samples[:,0]),"; m=",np.mean(z3.samples[:,1]))
    print("bayes factor yukawa-newtonian=",np.exp(z2.logz-z1.logz))
    print("bayes factor osc-newt",np.exp(z3.logz-z1.logz))
    #print("bayes factor osc-yukawa",np.exp(z3.logz-z2.logz))

def plot(P1,P2,P3):
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(15,8))
    
    
    plt.subplot(111)
    plt.xscale("log")
    p1 = np.linspace(data[0].min(),data[0].max(),10000)   
    plt.plot(p1, hypo1(p1,P1[0]),label='Newtonian',color='black',linewidth=2)
    plt.plot(p1, hypo2(p1,P2[0],P2[1]),label='Yukawa',color='dodgerblue',linewidth=2)
    plt.plot(p1, hypo3(p1,P3[0],P3[1],P3[2]),linewidth=1.5,label='Oscillating',color='indianred')
    plt.grid(color='w')
    plt.errorbar(data[0],data[1],yerr = data[2],fmt='none',alpha=0.6,c='black')
    plt.scatter(data[0],data[1],c='black',s=25)
    plt.legend(loc='upper right',title='Parametrization',fontsize=20,title_fontsize=20,frameon=False)
    plt.tick_params(axis='both',labelsize=20)
    plt.xlabel("r (mm)",fontsize=20,fontweight='bold')
    plt.ylabel("Residual torque (fN$\cdot$m)",fontsize=20,fontweight='bold')
    plt.savefig("fig.png")

#-==========================================================================-======

# getting best fits by chi^2 minimization


g1=np.array([0])
g2=np.array([0,20])
g3=np.array([0,65,3*np.pi/4])

P1 = optimize.minimize(chi2_val1, g1,args=data, method='Nelder-Mead')
P2 = optimize.minimize(chi2_val2, g2,args=data,  method='Nelder-Mead')
P3 = optimize.minimize(chi2_val3, g3,args=data, method='Nelder-Mead')


# plotting the best fit models
#plot(P1.x,P2.x,P3.x)

# Frequentist, AIC, BIC
analysis(P1.x,P2.x,P3.x,data)

# posterior samples
z1,z2,z3,S1,S2,S3 = get_posterior()

# WAIC calculation
WAIC(z1,z2,z3,S1,S2,S3)
   
#bayes factor 
bayesian(z1,z2,z3)

