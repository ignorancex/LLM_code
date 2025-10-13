"""
Created on Wed Jul  2 15:46:00 2025

@author: jwn0027
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

# Setup the style
params = Table.read('./src/data/rcParams.txt', format='csv')
for i in range(len(params)):
    plt.rcParams[params['name'][i]] = params['value'][i]
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True

#FTN Color points from Maxime Devogle
wave_data = [4670,6550,7480,9100]
ref_data = [0.82,1.2,1.37,1.6]
ref_error = [0.02,0.01,0.03,0.08]

#Load data files
pholus = np.genfromtxt('./src/data/pholus_binzel_1992.csv',delimiter=',')[1:]
oumuamua = np.genfromtxt('./src/data/oumuamua_ye_2017.csv',delimiter=',')[1:]
d_types = np.genfromtxt('./src/data/d_types.csv',delimiter=',')[1:]
borisov = np.genfromtxt('./src/data/borisov_deleon_2019.csv',delimiter=',')[1:]
#ATLAS spectrum from Karen Meech
atlas_3i = np.genfromtxt('./src/data/3I-Reflectivity_JN_edit.csv',delimiter=',')

def d_type_err(wl,d_types):
    #To match approximately the errors in DeMeo et al. 2009
    d_type_err = d_types*(0.1/10000*wl)
    return d_type_err

#Normalize to values at 5500 Angstroms
pholus_normref = pholus[:,1]/0.970
oumuamua_normref = oumuamua[:,1]/0.970
borisov_normref = borisov[:,1] #already normalized!
d_type_normref = d_types[:,1]/0.853
atlas_normref = atlas_3i[:,1]/1.03

#Plot data
d_type_sig = d_type_err(d_types[:,0],d_type_normref)
fig = plt.figure(figsize=(8,6))
#fig.set_facecolor('w')

alpha = 0.25
lw = 0.

plt.errorbar(wave_data, ref_data, yerr=ref_error,fmt='o',c='k',ls='-', ms=10, lw=2,label='3I/ATLAS MuSCAT3 4-Color',
             zorder=10)
plt.step(atlas_3i[:,0]*1e4,atlas_normref,c='#6434E9',alpha=alpha+0.2,label='3I/ATLAS SNIFS Spectrum', zorder=10)

plt.fill_between(pholus[:,0], pholus_normref-pholus_normref*0.1, pholus_normref+pholus_normref*0.1,
                 interpolate=True,color='#49CC5C',alpha=alpha,label='Pholus - Binzel 1992', lw=lw)

plt.fill_between(oumuamua[:,0], oumuamua_normref-oumuamua_normref*0.4, oumuamua_normref+oumuamua_normref*0.4,
                 interpolate=True,color='#F8C421',alpha=alpha,label="1I/'Oumuamua - Ye et al. 2017", lw=lw)

plt.fill_between(borisov[:,0], borisov_normref-borisov_normref*0.12, borisov_normref+borisov_normref*0.12,
                 interpolate=True,color='#2C7CE5',alpha=alpha,label="2I/Borisov - DeLeon et al. 2020", lw=lw)

plt.fill_between(d_types[:,0], d_type_normref-d_type_sig, d_type_normref+d_type_sig, interpolate=True,
                 color='#F82553', alpha=alpha,label="D Types - DeMeo et al. 2009", lw=lw)

plt.fill_betweenx(np.linspace(0,0.1,10),4900,5300 ,alpha=0.3,color='red', lw=lw)


plt.text(4400, 0.15,'SNIFS Dichroic Mask')

plt.xlabel('Wavelength ($\AA$)', fontsize=18)
plt.ylabel('Normalized Reflectance', fontsize=18)
plt.legend(loc=2)#, facecolor='w')
plt.grid(alpha=0.4)
plt.xlim(3000,9500)
plt.ylim(0,3)
plt.tight_layout()
plt.savefig('./src/tex/figures/3I_color_comp_v3.png', dpi=300, transparent=True)
