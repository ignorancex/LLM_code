#!/usr/bin/env python
# coding: utf-8

# In[12]:


from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
from matplotlib.pylab import plt
import nestle 
from scipy.optimize import curve_fit
from scipy.optimize import minimize
np.random.seed(0)


# In[20]:


data23 = np.genfromtxt('mean_23keV.csv', delimiter = ',').T
data34 = np.genfromtxt('mean_34keV.csv', delimiter = ',').T
data45 = np.genfromtxt('mean_45keV.csv', delimiter = ',').T
data56 = np.genfromtxt('mean_56keV.csv', delimiter = ',').T
data12 = np.genfromtxt('mean_12keV.csv', delimiter = ',').T

example = np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])
#print(data23[1])
y1 = data23[1]
x1 = data23[0]
erry1 = np.flip(np.genfromtxt('errmean23.csv', delimiter = ',').T[1]/2)

y2 = data34[1]
x2 = data34[0]
erry2 = np.flip(np.genfromtxt('errmean34.csv', delimiter = ',').T[1]/2)

y3 = data45[1]
x3 = data45[0]
erry3 = np.flip(np.genfromtxt('errmean45.csv', delimiter = ',').T[1]/2)

y4 = data56[1]
x4 = data56[0]
erry4 = np.flip(np.genfromtxt('errmean56.csv', delimiter = ',').T[1]/2)

y5 = data12[1]
x5 = data12[0]
erry5 = np.flip(np.genfromtxt('errmean12.csv', delimiter = ',').T[1]/2)
print(y5, erry5)

erry_arr = [erry5, erry1, erry2, erry3, erry4]
y_arr = [y5, y1, y2, y3, y4]
x_arr = [x5, x1, x2, x3, x4]


# In[21]:


#######CONSTANT COSINE##############
def cosine(x,theta2):
    A=theta2[0]
    return A

def loglike_cosine1(theta2):
    yM=cosine(x1,theta2)
    return -0.5 * np.sum(np.log(2 * np.pi * erry1 ** 2)
                         + (y1 - yM) ** 2 / erry1 ** 2)
def loglike_cosine2(theta2):
    yM=cosine(x2,theta2)
    return -0.5 * np.sum(np.log(2 * np.pi * erry2 ** 2)
                         + (y2 - yM) ** 2 / erry2 ** 2)
def loglike_cosine3(theta2):
    yM=cosine(x3,theta2)
    return -0.5 * np.sum(np.log(2 * np.pi * erry3 ** 2)
                         + (y3 - yM) ** 2 / erry3 ** 2)
def loglike_cosine4(theta2):
    yM=cosine(x4,theta2)
    return -0.5 * np.sum(np.log(2 * np.pi * erry4 ** 2)
                         + (y4 - yM) ** 2 / erry4 ** 2)
def loglike_cosine5(theta2):
    yM=cosine(x5,theta2)
    return -0.5 * np.sum(np.log(2 * np.pi * erry5 ** 2)
                         + (y5 - yM) ** 2 / erry5 ** 2)


#A is taken between -1 and 1
def prior_transform2(theta2):
    A = theta2[0]
    
    
    
    return np.array([0.05*(2*A - 1)])

def nestle_multi_cosine(loglike):
    
    
    # Run nested sampling
    res = nestle.sample(loglike, prior_transform2, 1, method='multi',
                    npoints=2000)
    #print(res)
    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
    

    return res.logz, pm, covm


# In[22]:


##############LINE COSINE############
def linecosine(x, theta3):
    m = theta3[0]
    c = theta3[1]
    
    return((m*(x) + c))

def loglike_line1(theta3):
    yM1 = linecosine(x1, theta3)
    return -0.5 * np.sum(np.log(2 * np.pi * erry1 ** 2)
                         + (y1 - yM1) ** 2 / erry1 ** 2)

def loglike_line2(theta3):
    yM2 = linecosine(x2, theta3)
    return -0.5 * np.sum(np.log(2 * np.pi * erry2 ** 2)
                         + (y2 - yM2) ** 2 / erry2 ** 2)

def loglike_line3(theta3):
    yM3 = linecosine(x3, theta3)
    return -0.5 * np.sum(np.log(2 * np.pi * erry3 ** 2)
                         + (y3 - yM3) ** 2 / erry3 ** 2)

def loglike_line4(theta3):
    yM4 = linecosine(x4, theta3)
    return -0.5 * np.sum(np.log(2 * np.pi * erry4 ** 2)
                         + (y4 - yM4) ** 2 / erry4 ** 2)

def loglike_line5(theta3):
    yM5 = linecosine(x5, theta3)
    return -0.5 * np.sum(np.log(2 * np.pi * erry5 ** 2)
                         + (y5 - yM5) ** 2 / erry5 ** 2)


#m is taken between -1 and 1; c is taken between -100 and 100
def prior_transform3(theta3):
    m = theta3[0]
    c = theta3[1]

    
    m_lim = 0.05
    c_lim = 0.05
    return np.array([m_lim*(2*m - 1), c_lim*(2*c - 1)])

    
def nestle_multi_line(loglike):
    # Run nested sampling
    res = nestle.sample(loglike, prior_transform3, 2, method='multi',
                    npoints=2000)
    #print(res)
    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
    #print(len(res.logl[keepidx]))
    #print(np.argmax(res.logl[keepidx]))
    fin = samples_nestle[np.argmax(res.logl[keepidx])]


    return res.logz, pm, covm


# In[23]:


##############EXPONENTIAL############
def expcosine(x, theta4):
    A = theta4[0]
    B = theta4[1]
    return((A*np.exp(-B*(x ))))

def loglike_exp1(theta4):
    yM = expcosine(x1, theta4)
    return -0.5 * np.sum(np.log(2 * np.pi * erry1 ** 2)
                         + (y1 - yM) ** 2 / erry1 ** 2)

def loglike_exp2(theta4):
    yM = expcosine(x2, theta4)
    return -0.5 * np.sum(np.log(2 * np.pi * erry2 ** 2)
                         + (y2 - yM) ** 2 / erry2 ** 2)

def loglike_exp3(theta4):
    yM = expcosine(x3, theta4)
    return -0.5 * np.sum(np.log(2 * np.pi * erry3 ** 2)
                         + (y3 - yM) ** 2 / erry3 ** 2)

def loglike_exp4(theta4):
    yM = expcosine(x4, theta4)
    return -0.5 * np.sum(np.log(2 * np.pi * erry4 ** 2)
                         + (y4 - yM) ** 2 / erry4 ** 2)

def loglike_exp5(theta4):
    yM = expcosine(x5, theta4)
    return -0.5 * np.sum(np.log(2 * np.pi * erry5 ** 2)
                         + (y5 - yM) ** 2 / erry5 ** 2)


#A is taken between -3 and 3; B is taken between 0 and 1
def prior_transform4(theta4):
    A = theta4[0]
    B = theta4[1]

    A_lim = 0.05
    B_lim = 0.1
    return np.array([A_lim*(2*A - 1), B_lim*(2*B - 1)])


    
def nestle_multi_exp(loglike):
    # Run nested sampling
    res = nestle.sample(loglike, prior_transform4, 2, method='multi',
                    npoints=2000)
    #print(res)
    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)

    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
    
    #print(len(res.logl[keepidx]))
    #print(np.argmax(res.logl[keepidx]))
    fin = samples_nestle[np.argmax(res.logl[keepidx])]

    return res.logz, pm, covm


# In[24]:


def evaluateresults(func, nestlefunc, loglikearr):
    Z_arr = []
    params = []
    covariances = []
    fig = plt.figure(figsize = (14, 20))
    for i in range(len(loglikearr)):
        
        if np.abs(np.min(y_arr[i]))>np.abs(np.max(y_arr[i])): A_lim=np.abs(np.min(y_arr[i]))
        else: A_lim=np.abs(np.max(y_arr[i]))
        
        Z, pm, covm = nestlefunc(loglikearr[i])
        #print("Log of evidence for  data_titles[i]  is ", Z)
        print("Estimated parameters for is ", pm)
        print("covariances: ", covm)
        
        Z_arr.append(Z)
        params.append(pm)
        covariances.append(covm)
        
    return Z_arr, params, covariances


# In[25]:


#CONSTANT AMPLITUDE
loglike_cos = np.array([loglike_cosine5, loglike_cosine1, loglike_cosine2, loglike_cosine3, loglike_cosine4])

Z_cos, params_cos, cov_cos = evaluateresults(cosine, nestle_multi_cosine, loglike_cos)
print(Z_cos)


# In[26]:


#LINEAR
loglike_line = [loglike_line5, loglike_line1, loglike_line2, loglike_line3, loglike_line4]
Z_line, params_line, cov_line = evaluateresults(linecosine, nestle_multi_line, loglike_line)
print(Z_line)


# In[27]:


#EXPONENTIAL
loglike_exp = [loglike_exp5, loglike_exp1, loglike_exp2, loglike_exp3, loglike_exp4]
Z_exp, params_exp, cov_exp = evaluateresults(expcosine, nestle_multi_exp, loglike_exp)
print(Z_exp)

import pandas as pd

DF1 = pd.DataFrame(params_exp)
DF1.to_csv("best_params_exp_mean_0.1.csv")


# In[28]:


#BAYES FACTOR CALCULATIONS
def bayesfactor(Z1, nf):
    Z1 = np.array(Z1)
    nf = np.array(nf)
    return np.exp(Z1 - nf)

print('For energy ranges in keVee: [2-3, 3-4, 4-5, 5-6]')
#Line and Constant
print(bayesfactor(Z_line, Z_cos))

#Exponential and Constant
print(bayesfactor(Z_exp, Z_cos))


# In[29]:


print("For constant fits: ")
for i in range(len(params_cos)):
    print("number ", i)
    print("A: ", params_cos[i][0], "+/-", np.sqrt(cov_cos[i][0, 0]))
    
print("\n")
print("For linear fits: ")
print(cov_line[1][0, 0], cov_line[1][1, 1])
for i in range(len(params_line)):
    print("number ", i)
    print("m: ", params_line[i][0], "+/-", np.sqrt(cov_line[i][0, 0]))
    print("c: ", params_line[i][1], "+/-", np.sqrt(cov_line[i][1, 1]))
    
print("\n")
print("For exponential fits: ")
for i in range(len(params_exp)):
    print("number ", i)
    print("A: ", params_exp[i][0], "+/-", np.sqrt(cov_exp[i][0, 0]))
    print("B: ", params_exp[i][1], "+/-", np.sqrt(cov_exp[i][1, 1]))


# In[30]:





#print(params_line)

data_titles = ['DAMA/LIBRA 1-2 keVee', 'DAMA/LIBRA 2-3 keVee', 'DAMA/LIBRA 3-4 keVee', 'DAMA/LIBRA 4-5 keVee', 'DAMA/LIBRA 5-6 keVee']
fig = plt.figure(figsize = (24, 20))

fig.subplots_adjust(wspace=0.45, right=0.95,
                    hspace=0.3, top=0.95)

plt.style.use('seaborn-colorblind')
plt.rcParams['text.usetex'] = True
#plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 33
plt.rcParams['legend.fontsize'] = 36
plt.rcParams['axes.linewidth'] = 1.5

plt.rcParams['font.weight'] = 1000
#fig.patch.set_facecolor('white')
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.ylabel("A $(cpd/kg/keV)$", fontsize = 50, color = 'black', fontweight=1000, labelpad = 50)

x_plot = np.arange(0.5, 14.5, 0.01)
print(x_plot)
for i in range(len(loglike_exp)):
    ax = fig.add_subplot(5, 1, i+1)
    #ax.set_title(data_titles[i])
    ax.errorbar(x_arr[i], y_arr[i], erry_arr[i], fmt = 'ko')
    ax.plot(x_plot, cosine(x_plot, params_cos[i])*(x_plot/x_plot), color = 'r', linewidth = 2.5, label = 'Constant $S_m$')
    ax.plot(x_plot, linecosine(x_plot, params_line[i]), color = 'b', linestyle = 'dashed', linewidth = 2.5, label = 'Linear $S_m$')
    ax.plot(x_plot, expcosine(x_plot, params_exp[i]), color = 'k', linestyle = 'dashdot', linewidth = 2.5, label = 'Exponential $S_m$')
    #plt.ylabel('A(cpd/kg/keV)')
    ax.tick_params(labelsize=38)
    if i == 0:
        ax.legend(bbox_to_anchor=(1.3, 1.025)).get_frame().set_edgecolor('black')
        #ax.set_xlim(0, 15)
    
    #ax.set_ylim(-0.01, 0.03)
    if i == len(loglike_exp) - 1:
        ax.set_xlabel("Annual Cycle $(years)$", fontsize = 50, color = 'black', fontweight=1000, labelpad = 30)
    

  
    
    plt.locator_params(axis='y', nbins=4)
    ax.text(0.98, 0.95, data_titles[i],
            ha='right', va='top', transform=ax.transAxes)
    
plt.savefig('bestfitkelso_w2018_oth.jpg')


# In[ ]:





# In[ ]:




