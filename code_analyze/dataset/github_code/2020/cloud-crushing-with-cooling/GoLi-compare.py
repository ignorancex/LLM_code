# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:29:52 2020

@author: alankar

Plots Figure 1 from the paper. 
It compares the Gronke-Oh and Li criterion in terms of the run prameters of our simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

CHI = np.logspace(1.5, 2.5, 100)
M = np.linspace(0.5, 1.5, 100)
cooling = np.loadtxt('cooltable.dat') #solar metallicity
LAMBDA = interpolate.interp1d(cooling[:,0], cooling[:,1])


X1, X2 = np.meshgrid(CHI, M)

#Constants
kB = 1.3807e-16 #Boltzman's Constant in CGS
mp = 1.6726231e-24 #Mass of a Proton in CGS
GAMMA = 5./3 #Specific Heat Ratio for an Ideal Gas

#Problem Constants
mu = 0.672442
Tcl = 1.e4 #K
ncl = 0.1 # particles per cm^3
T_hot = CHI*Tcl
LAMBDA_HOT= LAMBDA(T_hot) #erg cm3 s-1    #LAMBDA at T_hot #GET IT FROM COOLTABLE.DAT
Tmix= np.sqrt(Tcl*T_hot) #K
LAMBDA_MIX = LAMBDA(Tmix) #erg cm3 s-1    #LAMBDA at T_mix #GET IT FROM COOLTABLE.DAT
ALPHA = 1
n_hot=ncl/CHI

#Normalized Quantities
Tcl_4 = Tcl/1e4 #K
P3 = (ncl*Tcl)/1e3 #cm-3 K 
CHI_100=(X1)/100
LAMBDA_HOT_N21_4 = LAMBDA_HOT/(10**-21.4) #erg cm3 s-1
LAMBDA_MIX_N21_4 = LAMBDA_MIX/(10**-21.4) #erg cm3 s-1

cs = np.sqrt(GAMMA*kB/(mu*mp))*np.sqrt(X1*Tcl)/1e5 #kms^-1
vCl=X2*cs
R_Li = (15.4*(Tcl_4**(12/13)))*(X2**(4/13))*((CHI_100)**(20/13))/((ncl/0.1)*(LAMBDA_HOT_N21_4**(10/13)))

Rgo = (2 * (Tcl_4**(5/2)) * X2 * CHI_100 )/(P3*LAMBDA_MIX_N21_4*ALPHA)

#Plotting Area
fig = plt.figure(figsize=(60,20))
ax1 = plt.subplot2grid((20,20), (0,0), rowspan=3, colspan=1)
ax2 = plt.subplot2grid((20,20), (0,1), rowspan=3, colspan=1, sharey=ax1, sharex=ax1)
ax3 = plt.subplot2grid((20,20), (0,2), rowspan=3, colspan=1, sharey=ax1, sharex=ax1)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.025, hspace=0.)
    
#Gronke Oh et. al.
CS10 = ax1.contour(X1,X2,np.log10(Rgo),8,colors='w',linestyles='dotted')
CS11 = ax1.pcolormesh(X1,X2,np.log10(Rgo),shading='auto')#,10)
axins1 = inset_axes(ax1, width="100%", height="5%", loc='upper left', 
                    bbox_transform=ax1.transAxes, bbox_to_anchor=(0., 0.1, 1.0, 1.0),
                    borderpad=0)
cb=plt.colorbar(CS11, cax=axins1, format='%.1f', orientation='horizontal', pad=-25 )
axins1.xaxis.set_ticks_position('top')
cb.set_label(label='Gronke-Oh Radius (pc) [log]',size=12, labelpad=-38, y=0.5)
cb.ax.tick_params(labelsize=8)
ax1.clabel(CS10,inline=10,fontsize=10,colors='w')
ax1.set_xlabel(r'$\chi$',fontsize=20)
ax1.set_ylabel(r'$\rm\mathcal{M}$',fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=18, direction="out", pad=3)
ax1.tick_params(axis='both', which='minor', labelsize=15, direction="out", pad=3)

#Li et. al.
CS10 = ax2.contour(X1,X2,np.log10(R_Li),5,colors='w',linestyles='dotted')
CS11 = ax2.pcolormesh(X1,X2,np.log10(R_Li),shading='auto')#,10)
axins2 = inset_axes(ax2, width="100%", height="5%", loc='upper left', 
                    bbox_transform=ax2.transAxes, bbox_to_anchor=(0., 0.1, 1.0, 1.0),
                    borderpad=0)
cb=plt.colorbar(CS11, cax=axins2, format='%.1f', orientation='horizontal', pad=-25 )
axins2.xaxis.set_ticks_position('top')
cb.set_label(label='Li Radius (pc) [log]',size=12, labelpad=-38, y=0.5)
cb.ax.tick_params(labelsize=8)
ax2.clabel(CS10,inline=10,fontsize=10,colors='w')
ax2.set_xlabel(r'$\chi$',fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=18, direction="out", pad=3)
ax2.tick_params(axis='both', which='minor', labelsize=15, direction="out", pad=3)

#Ratio 
CS10 = ax3.contour(X1,X2,np.log10(Rgo/R_Li),5,colors='w',linestyles='dotted')
CS11 = ax3.pcolormesh(X1,X2,np.log10(Rgo/R_Li), shading='auto')
axins3 = inset_axes(ax3, width="100%", height="5%", loc='upper left', 
                    bbox_transform=ax3.transAxes, bbox_to_anchor=(0., 0.1, 1.0, 1.0),
                    borderpad=0)
cb=plt.colorbar(CS11, cax=axins3, format='%.1f', orientation='horizontal', pad=-25 )
axins3.xaxis.set_ticks_position('top')
cb.set_label(label=r'log$_{10}(R_{GO}/R_{Li})$',size=12, labelpad=-38, y=0.5)
cb.ax.tick_params(labelsize=8)
ax3.clabel(CS10,inline=10,fontsize=10,colors='w')
ax3.set_xlabel(r'$\chi$',fontsize=20)
ax3.tick_params(axis='both', which='major', labelsize=18, direction="out", pad=3)
ax3.tick_params(axis='both', which='minor', labelsize=15, direction="out", pad=3)

'''
There is an issue when saved to vector graphics like pdf resulting in unwanted
white streak lines in the plot. Better to have vector graphics, we plot it eps format.
Then open the file in a text editor and 
change "fill" to "gsave fill grestore stroke" and save the file.
See: https://tex.stackexchange.com/questions/418308/why-does-texstudio-internal-pdf-viewer-shows-streak-white-lines-on-image
'''

plt.savefig('GOLi-comp.eps', transparent=True, bbox_inches='tight')
plt.show()
plt.close()
