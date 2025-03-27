#===========================AtmNu_Recoils.py===================================#
# Created by Ciaran O'Hare 2020

# Script for doing the Monte Carlo sim of atmopsheric neutrino recoils
#==============================================================================#


import os
import sys
sys.path.append('../src')
from numpy import *
from numpy import random
from Params import *
from NeutrinoFuncs import *
from LabFuncs import *

### Possible variables
ngen = 10000000
Nuc = eval(sys.argv[1])
print('Nucleus = ',Nuc.Name)
fname = 'AtmNu_GranSasso_SolarMin.d'
E_th = 0.01
E_max = 200.0
recoildat_fname = 'AtmNu_Recoils_'+Nuc.Name+'.txt'


#### Load high energy data
Phi_tot,E_high,cosZ,phi_Az = GetAtmNuFluxes(fname)
Phi_high = squeeze(sum(sum(Phi_tot,0),0))

###### Load low energy FLUKA data
dat1 = loadtxt(nufile_dir+'/atmospheric/FLUKA/AtmNu_mubar.txt',delimiter=',')
dat2 = loadtxt(nufile_dir+'/atmospheric/FLUKA/AtmNu_mu.txt',delimiter=',')
dat3 = loadtxt(nufile_dir+'/atmospheric/FLUKA/AtmNu_e.txt',delimiter=',')
dat4 = loadtxt(nufile_dir+'/atmospheric/FLUKA/AtmNu_ebar.txt',delimiter=',')
E_low = dat1[:,0]
Phi_low = dat1[:,1]+dat2[:,1]+dat3[:,1]+dat4[:,1]

###### Join the two
E_join = append(E_low[0:260],E_high[9:])
Phi_join = append(Phi_low[0:260],Phi_high[9:])

##### Interpolate to create new array
nfine = 1000
E_nu_max = 1.0e4
E_fine = linspace(E_join[0],E_nu_max,nfine)
Phi_fine = interp(E_fine,E_join,Phi_join)

# Generate ngen initial energies and directions
E_gen,phi_nu_gen,costh_nu_gen,E_r_gen =\
    GenerateAtmNuDirections(ngen,E_fine,Phi_fine,E_high,Phi_tot,cosZ,phi_Az,Nuc)

# Scatter each neutrino
E_r_gen,phi_r_gen,costh_r_gen =\
    ScatterNeutrinos(Nuc,E_gen,phi_nu_gen,costh_nu_gen,E_r_gen)

# Window
mask_window = (E_r_gen<E_max)*(E_r_gen>E_th)
E_r_gen = E_r_gen[mask_window]
phi_r_gen = phi_r_gen[mask_window]
costh_r_gen = costh_r_gen[mask_window]
nleft = size(costh_r_gen)
print('Number of samples left = ',nleft)

t_gen = random.uniform(size=nleft)
DAT = column_stack((E_r_gen,phi_r_gen,costh_r_gen,t_gen,zeros(shape=nleft)))

print('Generating Cygnus angles')
for i in range(0,nleft):
    v_lab = LabVelocity(Jan1+67+t_gen[i])
    v_lab = v_lab/sqrt(sum(v_lab**2.0))
    x_rec = array([cos(phi_r_gen[i])*sqrt(1-costh_r_gen[i]**2.0),
                    sin(phi_r_gen[i])*sqrt(1-costh_r_gen[i]**2.0),
                    costh_r_gen[i]])
    DAT[i,4] = sum(v_lab*x_rec)

hdr = 'E_r [keV] \t phi \t costh \t t \t costh_cyg'
fmt = '%1.3f'

file_exists = os.path.exists(recoil_dir+recoildat_fname)
if file_exists:
    DAT_prev = loadtxt(recoil_dir+recoildat_fname)
    DAT = vstack((DAT,DAT_prev))
    savetxt(recoil_dir+recoildat_fname,DAT,header=hdr,fmt=fmt)
    print('merged')
else:
    savetxt(recoil_dir+recoildat_fname,DAT,header=hdr,fmt=fmt)
    print('first write')
