# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:38:18 2020

@author: alankar


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.lines import Line2D
import pickle

#Constants
kB = 1.3807e-16 #Boltzman's Constant in CGS
mp = 1.6726231e-24 #Mass of a Proton in CGS
GAMMA = 5./3 #Specific Heat Ratio for an Ideal Gas

fig = plt.figure(figsize=(30,30))

CHI = np.linspace(1.0,1000, 100000)

M1=0.5
M2=1.0
M3=1.5
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])    
    
#Problem Constants
mu = 0.672442
Tcl = 1.e4 #K
ncl = 0.1 # particles per cm^3
T_hot = CHI*Tcl
LAMBDA_HOT= LAMBDA(T_hot) #erg cm3 s-1    #LAMBDA at T_hot #GET IT FROM COOLTABLE.DAT
Tmix= np.sqrt(Tcl*T_hot) #K
LAMBDA_MIX = LAMBDA(Tmix) #erg cm3 s-1    #LAMBDA at T_mix #GET IT FROM COOLTABLE.DAT
ALPHA = 1.
n_hot=ncl/CHI
     
#Normalized Quantities
Tcl_4 = Tcl/1e4 #K
P3 = (ncl*Tcl)/1e3 #cm-3 K 
CHI_100=(CHI)/100
LAMBDA_HOT_N23 = LAMBDA_HOT/1e-23 #erg cm3 s-1
LAMBDA_MIX_N21_4 = LAMBDA_MIX/(10**-21.4) #erg cm3 s-1
    
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))   
    
R1= (2 * (Tcl_4**(5/2)) * M1 * CHI_100 )/(P3*LAMBDA_MIX_N21_4*ALPHA)
R2= (2 * (Tcl_4**(5/2)) * M2 * CHI_100 )/(P3*LAMBDA_MIX_N21_4*ALPHA)
R3= (2 * (Tcl_4**(5/2)) * M3 * CHI_100 )/(P3*LAMBDA_MIX_N21_4*ALPHA)

pc=3.098e18
tcc1= (np.sqrt(CHI)*R1*pc)/(M1*cs_hot)
tcc2= (np.sqrt(CHI)*R2*pc)/(M2*cs_hot)
tcc3= (np.sqrt(CHI)*R3*pc)/(M3*cs_hot)

f1=0.9*((2*R1*(n_hot/0.01))**0.3)*((M1*(cs_hot/1.e7))**0.6)
f2=0.9*((2*R2*(n_hot/0.01))**0.3)*((M2*(cs_hot/1.e7))**0.6)
f3=0.9*((2*R3*(n_hot/0.01))**0.3)*((M3*(cs_hot/1.e7))**0.6)

t_life_pred1=10*tcc1*f1
t_life_pred2=10*tcc2*f2
t_life_pred3=10*tcc3*f3

t_cool_hot=((1/(GAMMA-1))*kB*T_hot)/(n_hot*LAMBDA_HOT)

Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)

Y1=np.log10(t_life_pred1/Myr)
Y2=np.log10(t_life_pred2/Myr)
Y3=np.log10(t_life_pred3/Myr)
    
plt.plot(X,Y1,label='Gronke-Oh Criterion for $\mathrm{\mathcal{M}=0.5}$',linewidth=4.5)
plt.plot(X,Y2,label='Gronke-Oh Criterion for $\mathrm{\mathcal{M}=1.0}$',linewidth=4.5, color='red')
plt.plot(X,Y3,label='Gronke-Oh Criterion for $\mathrm{\mathcal{M}=1.5}$',linewidth=4.5, color='green')

############################################

data1=np.loadtxt('Li_pt_dest.dat')
X1=data1[:,0]  
Y1=data1[:,1]
plt.plot(X1,Y1,'o', color='gray', markersize=30, label='Li Destroyed Clouds',alpha=0.5)

data1=np.loadtxt('Li_pt_grth.dat')
X1=data1[:,0]  
Y1=data1[:,1]
plt.plot(X1,Y1,'^', color='gray', markersize=30, label='Li Growing Clouds', alpha=0.5)

#######################################################
############################################

M=0.5
R= [10.36,3.49]
pc=3.098e18
T_hot=1.e6
n_hot=0.001
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])
LAMBDA_HOT=LAMBDA(T_hot)
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))
tcc= (10*np.asarray(R)*pc)/(M*cs_hot)
f=0.9*((2*np.asarray(R)*(n_hot/0.01))**0.3)*((M*(cs_hot/1.e7))**0.6)

t_life_pred=10*tcc*f
t_cool_hot=(1.5*kB*T_hot)/(n_hot*LAMBDA_HOT)
    
Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)
Y=np.log10(t_life_pred/Myr)
X,Y=np.meshgrid(X,Y)
marker_style = dict(color='tab:blue', linestyle='None', marker='^',
                    markersize=30, markerfacecoloralt='tab:red', markeredgewidth=5)
filling = Line2D.fillStyles[-1]
plt.plot(X,Y,label=r'Growing Clouds in Our Simulations for $\mathrm{\mathcal{M}=0.5}$',fillstyle=filling, **marker_style)

#######################################################

M=1.0
R= [14.0,5.47]
pc=3.098e18
T_hot=1.e6
n_hot=0.001
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])
LAMBDA_HOT=LAMBDA(T_hot)
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))
tcc= (10*np.asarray(R)*pc)/(M*cs_hot)
f=0.9*((2*np.asarray(R)*(n_hot/0.01))**0.3)*((M*(cs_hot/1.e7))**0.6)

t_life_pred=10*tcc*f
t_cool_hot=(1.5*kB*T_hot)/(n_hot*LAMBDA_HOT)
    
Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)
Y=np.log10(t_life_pred/Myr)
X,Y=np.meshgrid(X,Y)
marker_style = dict(color='tab:red', linestyle='None', marker='^',
                    markersize=30, markerfacecoloralt='tab:red', markeredgewidth=5)
filling = Line2D.fillStyles[-1]
plt.plot(X,Y,label=r'Growing Clouds in Our Simulations for $\mathrm{\mathcal{M}=1.0}$',fillstyle=filling, **marker_style)

#############################################################

M=1.5
R= [17.0,7.16]
pc=3.098e18
T_hot=1.e6
n_hot=0.001
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])
LAMBDA_HOT=LAMBDA(T_hot)
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))
tcc= (10*np.asarray(R)*pc)/(M*cs_hot)
f=0.9*((2*np.asarray(R)*(n_hot/0.01))**0.3)*((M*(cs_hot/1.e7))**0.6)

t_life_pred=10*tcc*f
t_cool_hot=(1.5*kB*T_hot)/(n_hot*LAMBDA_HOT)

Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)
Y=np.log10(t_life_pred/Myr)
X,Y=np.meshgrid(X,Y)
marker_style = dict(color='tab:green', linestyle='None', marker='^',
                    markersize=30, markerfacecoloralt='tab:red', markeredgewidth=5)
filling = Line2D.fillStyles[-1]
plt.plot(X,Y,label=r'Growing Clouds in Our Simulations for $\mathrm{\mathcal{M}=1.5}$',fillstyle=filling, **marker_style)

#######################################################


M=0.5
R=[23.92,124.06]
pc=3.098e18
T_hot=3.e6
n_hot=0.1/300
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])
LAMBDA_HOT=LAMBDA(T_hot)
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))
tcc= (17.32*np.asarray(R)*pc)/(M*cs_hot)
f=0.9*((2*np.asarray(R)*(n_hot/0.01))**0.3)*((M*(cs_hot/1.e7))**0.6)

t_life_pred=10*tcc*f
t_cool_hot=(1.5*kB*T_hot)/(n_hot*LAMBDA_HOT)
    
Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)
Y=np.log10(t_life_pred/Myr)
X,Y=np.meshgrid(X,Y)
marker_style = dict(color='tab:blue', linestyle='None', marker='^',
                    markersize=30, markerfacecoloralt='tab:red', markeredgewidth=5)
filling = Line2D.fillStyles[-1]
plt.plot(X,Y,fillstyle=filling, **marker_style)

##############################################################
M=1.0
R=[37.64,169.02]
pc=3.098e18
T_hot=3.e6
n_hot=0.1/300
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])
LAMBDA_HOT=LAMBDA(T_hot)
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))
tcc= (17.32*np.asarray(R)*pc)/(M*cs_hot)
f=0.9*((2*np.asarray(R)*(n_hot/0.01))**0.3)*((M*(cs_hot/1.e7))**0.6)

t_life_pred=10*tcc*f
t_cool_hot=(1.5*kB*T_hot)/(n_hot*LAMBDA_HOT)
    
Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)
Y=np.log10(t_life_pred/Myr)
X,Y=np.meshgrid(X,Y)
marker_style = dict(color='tab:red', linestyle='None', marker='^',
                    markersize=30, markerfacecoloralt='tab:red', markeredgewidth=5)
filling = Line2D.fillStyles[-1]
plt.plot(X,Y,fillstyle=filling, **marker_style)

#############################################################
M=1.5
R=[49.01,202.45]
pc=3.098e18
T_hot=3.e6
n_hot=0.1/300
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])
LAMBDA_HOT=LAMBDA(T_hot)
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))
tcc= (17.32*np.asarray(R)*pc)/(M*cs_hot)
f=0.9*((2*np.asarray(R)*(n_hot/0.01))**0.3)*((M*(cs_hot/1.e7))**0.6)

t_life_pred=10*tcc*f
t_cool_hot=(1.5*kB*T_hot)/(n_hot*LAMBDA_HOT)
    
Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)
Y=np.log10(t_life_pred/Myr)
X,Y=np.meshgrid(X,Y)
marker_style = dict(color='tab:green', linestyle='None', marker='^',
                    markersize=30, markerfacecoloralt='tab:red', markeredgewidth=5)
filling = Line2D.fillStyles[-1]
plt.plot(X,Y,fillstyle=filling, **marker_style)

#######################################################

M=1.0
R= [1.0,0.5]
pc=3.098e18
T_hot=1.e6
n_hot=0.001
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])
LAMBDA_HOT=LAMBDA(T_hot)
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))
tcc= (10*np.asarray(R)*pc)/(M*cs_hot)
f=0.9*((2*np.asarray(R)*(n_hot/0.01))**0.3)*((M*(cs_hot/1.e7))**0.6)

t_life_pred=10*tcc*f
t_cool_hot=(1.5*kB*T_hot)/(n_hot*LAMBDA_HOT)
    
Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)
Y=np.log10(t_life_pred/Myr)
X,Y=np.meshgrid(X,Y)
marker_style = dict(color='tab:red', linestyle='None', marker='o',
                    markersize=30, markerfacecoloralt='tab:red', markeredgewidth=5)
filling = Line2D.fillStyles[-1]
plt.plot(X,Y,label=r'Destroyed Clouds in Our Simulations for $\mathrm{\mathcal{M}=1.0}$',fillstyle=filling, **marker_style)

#######################################################3
#######################################################

M=1.0
R= [2.8,1.5]
pc=3.098e18
T_hot=3.e6
n_hot=0.1/300
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])
LAMBDA_HOT=LAMBDA(T_hot)
cs_hot=np.sqrt((GAMMA*kB*T_hot)/(mu*mp))
tcc= (17.32*np.asarray(R)*pc)/(M*cs_hot)
f=0.9*((2*np.asarray(R)*(n_hot/0.01))**0.3)*((M*(cs_hot/1.e7))**0.6)

t_life_pred=10*tcc*f
t_cool_hot=(1.5*kB*T_hot)/(n_hot*LAMBDA_HOT)
    
Myr=365*24*60*60*1.e6
X=np.log10(t_cool_hot/Myr)
Y=np.log10(t_life_pred/Myr)
X,Y=np.meshgrid(X,Y)
marker_style = dict(color='tab:red', linestyle='None', marker='o',
                    markersize=30, markerfacecoloralt='tab:red', markeredgewidth=5)
filling = Line2D.fillStyles[-1]
plt.plot(X,Y,fillstyle=filling, **marker_style)

#######################################################3

x1=np.linspace(-2,6,10000)
y1=np.linspace(-2,6,10000)
plt.plot(x1,y1,label="Li Criterion",linestyle='--',color='black',linewidth=4.5)

plt.grid()
plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
plt.ylim((-4,5))
plt.xlim((-2,5))
plt.xlabel(r'$log_{10}\left(t_{cool,hot}\right)$ [Myr]',fontsize=70)
plt.ylabel(r'$log_{10}\left(t_{life,pred}\right)$ [Myr]',fontsize=70)
plt.arrow(4.2, 0.2, 0.0, 1.3, head_width=0.2, head_length=0.2, fc='k', ec='k',width=0.02)
plt.text(4.40, 0.7, r'$\mathrm{\mathcal{M}}$', horizontalalignment='center', verticalalignment='center', fontsize=60)
plt.legend(loc='best', prop={'size': 30}, bbox_to_anchor=(0.53, 0.69),framealpha=0.3)
pickle.dump(fig, open('myplot.pickle', 'wb'))
plt.savefig('Overlay.pdf',transparent =True, bbox_inches='tight')