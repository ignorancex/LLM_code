#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
from scipy import optimize , stats
from matplotlib.pylab import plt


# In[4]:


data23 = np.genfromtxt('mean_23keV.csv', delimiter = ',').T
data34 = np.genfromtxt('mean_34keV.csv', delimiter = ',').T
data45 = np.genfromtxt('mean_45keV.csv', delimiter = ',').T
data56 = np.genfromtxt('mean_56keV.csv', delimiter = ',').T
data12 = np.genfromtxt('mean_12keV.csv', delimiter = ',').T

time = np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])
#print(data23)
y1 = data23[1]
x1 = data23[0]
#x1 = time
erry1 = np.flip(np.genfromtxt('errmean23.csv', delimiter = ',').T[1]/2)

y2 = data34[1]
x2 = data34[0]
#x2 = time
erry2 = np.flip(np.genfromtxt('errmean34.csv', delimiter = ',').T[1]/2)

y3 = data45[1]
#x3 = data45[0]
x3 = time
erry3 = np.flip(np.genfromtxt('errmean45.csv', delimiter = ',').T[1]/2)

y4 = data56[1]
x4 = data56[0]
#x4 = time
erry4 = np.flip(np.genfromtxt('errmean56.csv', delimiter = ',').T[1]/2)

y5 = data12[1]
x5 = data12[0]
#x4 = time
erry5 = np.flip(np.genfromtxt('errmean12.csv', delimiter = ',').T[1]/2)


erry_arr = [erry1, erry2, erry3, erry4, erry5]
y_arr = [y1, y2, y3, y4, y5]
x_arr = [x1, x2, x3, x4, x5]
data_arr = [data23, data34, data45, data56, data12]

i=4
data = data_arr[i]
x = x_arr[i]
y = y_arr[i]
yerr = erry_arr[i]


# In[39]:


#cosine model
def fit_cosine(x,A):
	return A*x**0

def cosine(x,theta):
	A = theta[0]
	return A*x**0

#linear model
def fit_linear(x, a, b):
	return (a*x + b)

def linear(x, theta):
    a = theta[0]
    b =theta[1]
    return (a*x + b)    

#exponential model
def fit_exp(x, C, D):
    return C*np.exp(-D*(x))

def exponential(x, theta):
    C = theta[0]
    D = theta[1]
    return C*np.exp(-D*(x))
    


# In[40]:


def logL(theta, model, data=data):
    if model==cosine or model==fit_cosine:
        A = theta[0]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((yerr[i]**2))
    if model == fit_linear or model == linear:
        a = theta[0]
        b = theta[1]

        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((yerr[i]**2))
    if model == fit_exp or model == exponential:
        C = theta[0]
        D = theta[1]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((yerr[i]**2))
            
    
    y_fit = model(x,theta)
    return sum(stats.norm.logpdf(*args) for args in zip(y, y_fit, sigma))

#chi square value
def chi2_val(theta,model,data=data):
    if model==cosine or model==fit_cosine:
        A = theta[0]

        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((yerr[i]**2))
    if model == fit_linear or model == linear:
        a = theta[0]
        b = theta[1]

        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((yerr[i]**2))
    if model == fit_exp or model == exponential:
        C = theta[0]
        D = theta[1]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((yerr[i]**2))
    
    
    y_fit=model(x,theta)
    r = (y - y_fit)/sigma
    return np.sum(r**2)

#degrees of freedom
def dof_val(theta,data=data):
  return len(x) - len(theta)

#chi squared likelihood function
def chi2L(theta,model,data=data):
  chi2 = chi2_val(theta,model)
  dof = dof_val(theta)
  return stats.chi2(dof).pdf(chi2)


# In[41]:


#negative log likelihood function
cos_neg_logL = lambda theta: -logL(theta, cosine, data)
lin_neg_logL = lambda thetaL: -logL(thetaL, linear, data)
exp_neg_logL = lambda thetaE: -logL(thetaE, exponential, data)

#initial guess
parE_init = [0.0, 0.0]
parL_init = [0.0, 0.0]
par_init =[0.00]


# In[42]:



#negative likelihood minimization
exp_fin = optimize.fmin_bfgs(exp_neg_logL, parE_init, disp=False)
lin_fin = optimize.fmin_bfgs(lin_neg_logL, parL_init, disp=False)
par_fin = optimize.fmin_bfgs(cos_neg_logL, par_init, disp=False)

A = par_fin[0]
#w = par_fin[1]
#t = par_fin[2]

c1=chi2_val(par_fin,cosine)
c2=chi2_val(lin_fin,linear)
c3=chi2_val(exp_fin, exponential)
d1=dof_val(par_fin)
d2=dof_val(lin_fin)
d3=dof_val(exp_fin)

print("\nCosine : Amplitude= ",A)
print("Linear : a= ",lin_fin[0], " b = ", lin_fin[1])
print("Exponential : C= ",exp_fin[0], " D = ", exp_fin[1])

print("\nCosine :  Chi-Square likelihood:" , chi2L(par_fin,cosine)," ; Chi square value=",c1,"\nLinear :  Chi-Square likelihood:" , chi2L(lin_fin,linear)," ; Chi square value=",c2, "\nExponential : Chi-Square likelihood: ", chi2L(exp_fin,exponential)," ; Chi square value=",c3)


# In[43]:


p1=stats.chi2(d1).sf(c1)
print("cosine ",'dof',d1,'pval',p1,'sigma' ,stats.norm.isf(p1))
p2=stats.chi2(d2).sf(c2)
print("Linear ",'dof',d2,'pval',p2,'sigma',stats.norm.isf(p2))
p3=stats.chi2(d3).sf(c3)
print("Exponential ",'dof',d3,'pval',p3,'sigma',stats.norm.isf(p3))


dL=np.abs(c2-c1)
print("difference in chi square values = ",dL)
dE=np.abs(c3-c1)
print("difference in chi square values = ",dE)
pL=stats.chi2(1).sf(dL)
print ("p value for linear=",pL)
print("Linear: Confidence level : ",stats.norm.isf(pL),'\u03C3','\n')
pE=stats.chi2(1).sf(dE)
print ("p value for exponential=",pE)
print("Exponential: Confidence level : ",stats.norm.isf(pE),'\u03C3','\n')


# In[36]:


fig = plt.figure(figsize = (16, 4))
x_plot = np.arange(x[0], x[len(x) - 1], 1)
plt.title("blah")
plt.errorbar(x, y, yerr, fmt = '.')
plt.plot(x_plot, cosine(x_plot, par_fin)*(x_plot/x_plot), 'b-', label = 'constant amplitude')
plt.plot(x_plot, linear(x_plot, lin_fin), 'r-', label = 'linear amplitude')
plt.plot(x_plot, exponential(x_plot, exp_fin), 'g-', label = 'exponential amplitude')
plt.xlabel("Time (days)")
plt.ylabel("Residuals (cpd/kg/keV)")
plt.legend()


# In[ ]:





# In[ ]:




