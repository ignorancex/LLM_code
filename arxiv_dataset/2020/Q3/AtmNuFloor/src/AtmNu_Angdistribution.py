import os
import sys
sys.path.append('../src')
from numpy import *
from numpy import random
from Params import *
from NeutrinoFuncs import *

# Script for creating the nice distributions of CEvNS recoils (Fig. 1)

# This file doesn't save all its recoils because we need a large number to
# make a nice plot of energy/phi/costh distribution. So everytime this file
# is run the new distribution is merged with a previous one to make it smoother
# each time.

### Possible variables
ngen = 10000000
Nuc = Ar40
fname = 'AtmNu_GranSasso_SolarMin.d'
nc = 20
np = 20
ne = 50
recoildat_fname = 'AtmNu_Ar_RecoilDist_3D.txt'


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
E_th = 2.0
E_max = 100.0
mask_window = (E_r_gen<E_max)*(E_r_gen>E_th)
E_r_gen = E_r_gen[mask_window]
phi_r_gen = phi_r_gen[mask_window]
costh_r_gen = costh_r_gen[mask_window]
print(size(costh_r_gen))

# Make histogram
E_edges = linspace(E_th,E_max,ne+1)
Ei = digitize(E_r_gen,E_edges,right=True)-1
R_3D = zeros(shape=(ne,np*nc))
for i in range(0,ne):
    H,ce,pe = histogram2d(costh_r_gen[(Ei==i)],phi_r_gen[(Ei==i)],\
                        bins=(nc,np),range=[[-1.0,1.0],[-pi,pi]])
    R_3D[i,:] = H.reshape(nc*np)
R_3D = 1.0*R_3D/(1.0*sum(R_3D))
cc = (ce[1:]+ce[0:-1])/2.0
pp = (pe[1:]+pe[0:-1])/2.0



# Save data (merge files if already exists)
[C,P] = meshgrid(cc,pp)
C = C.reshape(nc*np)
P = P.reshape(nc*np)
C = append(nc,C)
P = append(np,P)
E_centers = (E_edges[1:]+E_edges[0:-1])/2.0
DAT = zeros(shape=(ne+2,nc*np+1))
DAT[0,:] = 1.0*C
DAT[1,:] = 1.0*P
DAT[2:,0] = 1.0*E_centers
DAT[2:,1:] = 1.0*R_3D

file_exists = os.path.exists(recoil_dir+recoildat_fname)
if file_exists:
    DAT_prev = loadtxt(recoil_dir+recoildat_fname)
    if (shape(DAT_prev)[0]==shape(DAT)[0])&(shape(DAT_prev)[1]==shape(DAT)[1]):
        DAT[2:,1:] += DAT_prev[2:,1:]
        DAT[2:,1:] /= sum(DAT[2:,1:])
        savetxt(recoil_dir+recoildat_fname,DAT)
        print('merged')
    else:
        savetxt(recoil_dir+recoildat_fname,DAT)
        print('overwritten')
else:
    savetxt(recoil_dir+recoildat_fname,DAT)
    print('first write')
