#!/usr/bin/env python
# coding: utf-8

# In[25]:


from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
from matplotlib.pylab import plt
import nestle 
from scipy.optimize import curve_fit
from scipy.optimize import minimize
np.random.seed(0)


# In[28]:


data24 = np.genfromtxt('2012_2-4kev.txt').T
data25 = np.genfromtxt('2012_2-5kev.txt').T
data13 = np.genfromtxt('2018_1-3kev.txt').T
data16 = np.genfromtxt('2018_1-6kev.txt').T
data26 = np.genfromtxt('2-6KeV.txt').T
data_wNaI =  np.genfromtxt('2-6KeV_wNaI_again.csv', delimiter = ',').T

data_arr = [data24, data25, data13, data16, data26, data_wNaI]
data_titles = ['DAMA/LIBRA Phase - I 2-4KeV', 'DAMA/LIBRA Phase - I 2-5KeV', 'DAMA/LIBRA Phase -II 1-3KeV', 'DAMA/LIBRA Phase - II 1-6KeV', 'DAMA/LIBRA Phase - II and Phase -II 2-6KeV', 'DAMA/NaI, DAMA/LIBRA Phase -I and Phase - II 2-6KeV']
t1 = data24[0]
xerr1 = data24[2]
erry1 = data24[3]
y1 = data24[1]

t2 = data25[0]
xerr2 = data25[2]
erry2 = data25[3]
y2 = data25[1]

t3 = data13[0]
xerr3 = data13[2]
erry3 = data13[3]
y3 = data13[1]

t4 = data16[0]
xerr4 = data16[2]
erry4 = data16[3]
y4 = data16[1]

t5 = data26[0]
xerr5 = data26[2]
erry5 = data26[3]
y5 = data26[1]

t6 = data_wNaI[0]
xerr6 = data_wNaI[2]
erry6 = data_wNaI[3]
y6 = data_wNaI[1]

x_arr = [t1, t2, t3, t4, t5, t6]
y_arr = [y1, y2, y3, y4, y5, y6]
erry_arr = [erry1, erry2, erry3, erry4, erry5, erry6]


# In[29]:


difference = []
for i in range(len(t6)-1):
    difference.append((t6[i+1] - t6[i]))

#print(difference)
print(np.median(difference))
print(t6[len(t6)-1])


# In[30]:


def cosine(x,theta):
    A = theta[0]
    w = theta[1]
    t0= theta[2]
    return A*np.cos(w*(x -t0))

# The likelihood function:
def loglike_cos(theta, *args):
        A = theta[0]
        w = theta[1]
        t0= theta[2]
        #print("Check in loglike, ie, first datapoint", data[0][0])
        x = data[0]
        y = data[1]
        sigma=np.zeros(len(data[0]))
        for i in range(len(data[0])):
            sigma[i] = data[3, i]
            #sigma[i]=np.sqrt((data[3,i]**2)+((A*w*np.sin(w*(x[i]-t_0)))*data[2,i])**2)
        yM= cosine(x,theta)
        return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2)
                         + (y - yM) ** 2 / sigma ** 2)

# Defines a flat prior for each parameter:
def prior_transform_cos(theta):
    A = theta[0]
    w = theta[1]
    t0 = theta[2]
    
    A_lim = 0.05
    return np.array([A_lim*(2*A - 1) , 0.1281389*w +  0.0015, t0*2*np.pi/w])

def nestle_multi_cos():
    # Run nested sampling
    res = nestle.sample(loglike_cos, prior_transform_cos, 3, method='multi',
                    npoints=2000, args = data)
    print(res.summary())

    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
   
    print("For A: \n")
    print(np.mean(samples_nestle[:,0]))      # mean of A samples
    print(np.median(samples_nestle[:, 0]))
    print(np.std(samples_nestle[:,0])      )# standard deviation of A samples
    print("For w: \n")
    print( np.mean(samples_nestle[:,1])    )  # mean of w samples
    print(np.median(samples_nestle[:, 1]))
    print( np.std(samples_nestle[:,1])    )  # standard deviation of w samples
    print("For t0: \n")
    print( np.mean(samples_nestle[:,2])    )  # mean of t0 samples
    print(np.median(samples_nestle[:, 2]))
    print( np.std(samples_nestle[:,2])    )  # standard deviation of t0 samples
    print(len(samples_nestle))              # number of posterior samples
    
    fin = samples_nestle[np.argmax(res.logl[keepidx])]
    
    return res.logz, pm, covm
    



Z_cos = []
params_cos = []
covm_cos = []
for i in range(len(data_arr)):
    print(data_titles[i], ':')
    data = data_arr[i]
    #print('The first datapoint is: ', data[0][0])
    Z2, pm, covm = nestle_multi_cos()
    params_cos.append(pm)
    covm_cos.append(covm)
    Z_cos.append(Z2)
    print("A0: ", pm[0], " +/- ", np.sqrt(covm[0, 0]))
    print("w: ", pm[1], " +/- ", np.sqrt(covm[1, 1]))
    print("t0: ", pm[2], " +/- ", np.sqrt(covm[2, 2]))
    


# In[23]:


def linear(x,theta):
    m = theta[0]
    c = theta[1]
    w = theta[2]
    t0 = theta[3]
    return (m*x + c)*np.cos(w*(x - t0))

# The likelihood function:
def loglike_line(theta, *args):
        m = theta[0]
        c = theta[1]
        w = theta[2]
        t0 = theta[3]
        #print("Check in loglike, ie, first datapoint", data[0][0])
        x = data[0]
        y = data[1]
        sigma = np.zeros(len(data[0]))
        for i in range(len(data[0])):
            sigma[i] = np.sqrt(data[3, i]**2)
            #sigma[i]=np.sqrt((data[3,i]**2)+(((m*x[i] + c)*w*np.sin(w*(x[i]-t_0)) 
            #                                  + m*np.cos(w*(x[i]-t_0)))*data[2,i])**2)
        yM= linear(x,theta)
        return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2)
                         + (y - yM) ** 2 / sigma ** 2)

# Defines a flat prior for each parameter:
def prior_transform_line(theta):
    m = theta[0]
    c = theta[1]
    w = theta[2]
    t0 = theta[3]
    
    m_lim = 0.05
    c_lim = 0.05
    return np.array([m_lim*(2*m -1), c_lim*(2*c - 1) ,  0.1281389*w +  0.0015 , t0*2*np.pi/w])

def nestle_multi_line():
    # Run nested sampling
    res = nestle.sample(loglike_line, prior_transform_line, 4, method='multi',
                    npoints=2000, args = data)
    print(res.summary())

    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
   
    print("For m: \n")
    print(np.mean(samples_nestle[:,0]))      # mean of m samples
    print(np.median(samples_nestle[:,0]))
    print(np.std(samples_nestle[:,0])      )# standard deviation of m samples
    print("For c: \n")
    print( np.mean(samples_nestle[:,1])    )  # mean of c samples
    print(np.median(samples_nestle[:,1]))
    print( np.std(samples_nestle[:,1])    )  # standard deviation of c samples
    
    print("For w: \n")
    print( np.mean(samples_nestle[:,2])    )  # mean of w samples
    print(np.median(samples_nestle[:,2]))
    print( np.std(samples_nestle[:,2])    )  # standard deviation of w samples
    print("For t0: \n")
    print( np.mean(samples_nestle[:,3])    )  # mean of t0 samples
    print(np.median(samples_nestle[:,3]))
    print( np.std(samples_nestle[:,3])    )  # standard deviation of t0 samples
    
    
    print(len(samples_nestle))              # number of posterior samples
    
    fin = samples_nestle[np.argmax(res.logl[keepidx])]
    
    return res.logz, pm, covm
    


# In[ ]:


Z_line = []
params_line = []
covm_line = []
for i in range(len(data_arr)):
    print(data_titles[i], ':')
    data = data_arr[i]
    #print('The first datapoint is: ', data[0][0])
    
    Z2, pm, covm = nestle_multi_line()
    params_line.append(pm)
    covm_line.append(covm)
    Z_line.append(Z2)
    
    print("m: ", pm[0], " +/- ", np.sqrt(covm[0, 0]))
    print("c: ", pm[1], " +/- ", np.sqrt(covm[1, 1]))
    print("w: ", pm[2], " +/- ", np.sqrt(covm[2, 2]))
    print("t0: ", pm[3], " +/- ", np.sqrt(covm[3, 3]))


# In[15]:


def exponential(x,theta):
    C = theta[0]
    D = theta[1]
    w = theta[2]
    t0 = theta[3]
    return C*np.exp(-D*(x))*np.cos(w*(x - t0))

# The likelihood function:
def loglike_exp(theta, *args):
    C = theta[0]
    D = theta[1]
    w = theta[2]
    t0 = theta[3]

    x = data[0]
    y = data[1]
    sigma=np.zeros(len(data[0]))
    for i in range(len(data[0])):
        sigma[i] = np.sqrt((data[3,i]**2))
        #sigma[i]=np.sqrt((data[3,i]**2)+(((C*np.exp(-D*(x[i])))*w*np.sin(w*(x[i]-t_0)) 
        #                                  - (D*C*np.exp(-D*(x[i])))*np.cos(w*(x[i]-t_0)))*data[2,i])**2)
    yM = exponential(x,theta)
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2)
                         + (y - yM) ** 2 / sigma ** 2)

# Defines a flat prior for each parameter:
def prior_transform_exp(theta):
    C = theta[0]
    D = theta[1]
    w = theta[2]
    t0 = theta[3]

    C_lim = 0.05
    D_lim = 0.01
    return np.array([C_lim*C, D_lim*(2*D - 1) , 0.1281389*w +  0.0015 , t0*2*np.pi/w])

def nestle_multi_exp():
    # Run nested sampling
    res = nestle.sample(loglike_exp, prior_transform_exp, 4, method='multi',
                    npoints=2000, args = data)
    print(res.summary())

    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
   
    print("For C: ")
    print(np.mean(samples_nestle[:,0]))      # mean of C samples
    print(np.median(samples_nestle[:,0])) 
    print(np.std(samples_nestle[:,0])      )# standard deviation of C samples
    print("For D: ")
    print(np.mean(samples_nestle[:,1]))      # mean of d samples
    print(np.median(samples_nestle[:,1])) 
    print(np.std(samples_nestle[:,1])      )# standard deviation of D samples
    print("For w: ")
    print( np.mean(samples_nestle[:,2])    )  # mean of w samples
    print(np.median(samples_nestle[:,2])) 
    print( np.std(samples_nestle[:,2])    )  # standard deviation of w samples
    print("For t0: ")
    print( np.mean(samples_nestle[:,3])    )  # mean of t0 samples
    print(np.median(samples_nestle[:,3])) 
    print( np.std(samples_nestle[:,3])    )  # standard deviation of t0 samples
    print(len(samples_nestle))              # number of posterior samples
    
    fin = samples_nestle[np.argmax(res.logl[keepidx])]
    
    return res.logz, pm, covm
    


# In[16]:


Z_exp = []
params_exp = []
covm_exp = []
for i in range(len(data_arr)):
    print(data_titles[i], ':')
    data = data_arr[i]
    #print('The first datapoint is: ', data[0][0])
        
    Z2, pm, covm = nestle_multi_exp()
    params_exp.append(pm)
    covm_exp.append(covm)
    Z_exp.append(Z2)
    
    print("C: ", pm[0], " +/- ", np.sqrt(covm[0, 0]))
    print("D: ", pm[1], " +/- ", np.sqrt(covm[1, 1]))
    print("w: ", pm[2], " +/- ", np.sqrt(covm[2, 2]))
    print("t0: ", pm[3], " +/- ", np.sqrt(covm[3, 3]))


# In[24]:


#BAYES FACTOR CALCULATIONS
def bayesfactor(Z1, nf):
    Z1 = np.array(Z1)
    nf = np.array(nf)
    return np.exp(Z1 - nf)

#Line and Constant
print(bayesfactor(Z_line, Z_cos))

#Exponential and Constant
print(bayesfactor(Z_exp, Z_cos))


# In[21]:


import pandas as pd

DF1 = pd.DataFrame(params_cos)
DF1.to_csv("best_params_cos.csv")


DF2 = pd.DataFrame(params_line)
DF2.to_csv("best_params_line.csv")


DF5 = pd.DataFrame(Z_cos)
DF5.to_csv("logz_cos.csv")


DF6 = pd.DataFrame(Z_line)
DF6.to_csv("logz_line.csv")



DF3 = pd.DataFrame(params_exp)
DF3.to_csv("best_params_exp.csv")

DF7 = pd.DataFrame(Z_exp)
DF7.to_csv("logz_exp.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




