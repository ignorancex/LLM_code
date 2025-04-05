#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np 
from scipy import optimize , stats
from matplotlib.pylab import plt


# In[51]:


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


i =5

data = data_arr[i]
data_title = data_titles[i]
x=data[0]
y=data[1]
xerr=data[2]
yerr=data[3]


# In[52]:


#cosine model
def fit_cosine(x,A,w,t_0):
	return A*np.cos(w*(x-t_0))

def cosine(x,theta):
	A = theta[0]
	w = theta[1]
	t_0= theta[2]
	return A*np.cos(w*(x-t_0))

#linear model
def fit_linear(x, a, b, w, t_0):
	return (a*x + b)*np.cos(w*(x-t_0))

def linear(x, theta):
    a = theta[0]
    b =theta[1]
    w = theta[2]
    t_0 = theta[3]
    return (a*x + b)*np.cos(w*(x-t_0))    

#exponential model
def fit_exp(x, C, D, w, t_0):
    return C*np.exp(-D*(x))*np.cos(w*(x-t_0))

def exponential(x, theta):
    C = theta[0]
    D = theta[1]
    w = theta[2]
    t_0 = theta[3]
    return C*np.exp(-D*(x))*np.cos(w*(x-t_0))


# In[53]:


def logL(theta, model, data=data):
    if model==cosine or model==fit_cosine:
        A = theta[0]
        w = theta[1]
        t_0= theta[2]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2))
    if model == fit_linear or model == linear:
        a = theta[0]
        b = theta[1]
        w = theta[2]
        t_0 = theta[3]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2))
    if model == fit_exp or model == exponential:
        C = theta[0]
        D = theta[1]
        w = theta[2]
        t_0 = theta[3]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2))
            
    
    y_fit = model(x,theta)
    return sum(stats.norm.logpdf(*args) for args in zip(y, y_fit, sigma))

#chi square value
def chi2_val(theta,model,data=data):
    if model==cosine or model==fit_cosine:
        A = theta[0]
        w = theta[1]
        t_0= theta[2]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2))
    if model == fit_linear or model == linear:
        a = theta[0]
        b = theta[1]
        w = theta[2]
        t_0 = theta[3]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2))
    if model == fit_exp or model == exponential:
        C = theta[0]
        D = theta[1]
        w = theta[2]
        t_0 = theta[3]
        sigma=np.zeros(len(x))
        for i in range(len(x)):
            sigma[i]=np.sqrt((data[3,i]**2))
    
    
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


# In[54]:


#negative log likelihood function
cos_neg_logL = lambda theta: -logL(theta, cosine, data)
lin_neg_logL = lambda thetaL: -logL(thetaL, linear, data)
exp_neg_logL = lambda thetaE: -logL(thetaE, exponential, data)

#initial guess
parE_init = [0.0, 0.0, 0.0172, 152.5]
parL_init = [0.0, 0.01, 0.0172, 152.5]
par_init =[0.0, 0.0172, 152.5]


# In[55]:



#negative likelihood minimization
exp_fin = optimize.fmin_bfgs(exp_neg_logL, parE_init, disp=False)
lin_fin = optimize.fmin_bfgs(lin_neg_logL, parL_init, disp=False)
par_fin = optimize.fmin_bfgs(cos_neg_logL, par_init, disp=False)

A = par_fin[0]
w = par_fin[1]
t = par_fin[2]

c1=chi2_val(par_fin,cosine)
c2=chi2_val(lin_fin,linear)
c3=chi2_val(exp_fin, exponential)
d1=dof_val(par_fin)
d2=dof_val(lin_fin)
d3=dof_val(exp_fin)

print("\nCosine : Amplitude= ",A,"  w= ",w," /days ","  initial phase= ",t," days")
print("Linear : a= ",lin_fin[0], " b = ", lin_fin[1], " w = ", lin_fin[2], " t_0 = ", lin_fin[3])
print("Exponential : C= ",exp_fin[0], " D = ", exp_fin[1], " w = ", exp_fin[2], " t_0 = ", exp_fin[3])

print("\nCosine :  Chi-Square likelihood:" , chi2L(par_fin,cosine)," ; Chi square value=",c1,"\nLinear :  Chi-Square likelihood:" , chi2L(lin_fin,linear)," ; Chi square value=",c2, "\nExponential : Chi-Square likelihood: ", chi2L(exp_fin,exponential)," ; Chi square value=",c3)


# In[56]:


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


# In[57]:


fig = plt.figure(figsize = (16, 4))
x_plot = np.arange(x[0], x[len(x) - 1], 1)
plt.title(data_title)
plt.errorbar(x, y, yerr, fmt = '.')
plt.plot(x_plot, cosine(x_plot, par_fin)*(x_plot/x_plot), 'b-', label = 'constant amplitude')
plt.plot(x_plot, linear(x_plot, lin_fin), 'r-', label = 'linear amplitude')
plt.plot(x_plot, exponential(x_plot, exp_fin), 'g-', label = 'exponential amplitude')
plt.xlabel("Time (days)")
plt.ylabel("Residuals (cpd/kg/keV)")
plt.legend()


# In[ ]:





# In[ ]:




