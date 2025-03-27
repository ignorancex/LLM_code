import os
import sys
sys.path.append('../src')
from numpy import *
from numpy import random
from Params import *
from NeutrinoFuncs import *
from LabFuncs import *

# This file doesn't save all its recoils because we need a large number to
# make a nice plot of energy/phi/costh distribution. So everytime this file
# is run the new distribution is merged with a previous one to make it smoother
# each time.

# I needed to run this file (both Xe131 and Ar40) around 10 times to get
# a nice distribution, but the recoil file ends up being huge (which is why
# it's not in the git repository)

#==============================================================================#
# Input
Nuc = eval(sys.argv[1])
print('Nucleus = ',Nuc.Name)
if Nuc.Name=='Xe':
    E_min = 2.0
    E_max = 200.0
elif Nuc.Name=='Ar':
    E_min = 20.0
    E_max = 400.0
#==============================================================================#

ngen = 1000000
fname = 'AtmNu_GranSasso_SolarMin.d'
ne = 20


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

# Window and get angles
mask_window = (E_r_gen<=E_max)*(E_r_gen>=E_min)
E_r_gen = E_r_gen[mask_window]
phi_r_gen = phi_r_gen[mask_window]
costh_r_gen = costh_r_gen[mask_window]
nleft = size(costh_r_gen)
print('nleft=',size(costh_r_gen))
print('Generating Cygnus angles')
costh_r_gen_2 = zeros(shape=nleft)
t_gen = random.uniform(size=nleft)
for i in range(0,nleft):
    v_lab = LabVelocity(Jan1+67+t_gen[i])
    v_lab = v_lab/sqrt(sum(v_lab**2.0))
    x_rec = array([cos(phi_r_gen[i])*sqrt(1-costh_r_gen[i]**2.0),
                    sin(phi_r_gen[i])*sqrt(1-costh_r_gen[i]**2.0),
                    costh_r_gen[i]])
    costh_r_gen_2[i] = sum(v_lab*x_rec)




# Binning
costhmin = 0.0
costh_edges = sqrt(linspace(0.0,1.0,ne+1))
costh_centers = (costh_edges[1:]+costh_edges[0:-1])/2.0
E_r_edges = logspace(log10(E_min),log10(E_max),ne+1)
E_r_centers = (E_r_edges[1:]+E_r_edges[0:-1])/2.0
[E,C] = meshgrid(E_r_centers,costh_centers)
eff2 = efficiency(Nuc,E)


# Atmospheric neutrino rate
R_Atm = R_AtmNu(E_min,E_max,Nuc=Nuc,eff_on=False)

R1,ce,ee = histogram2d(abs(costh_r_gen),log10(E_r_gen),bins=(ne,ne),\
                    range=[[0.0,1.0],[log10(E_min),log10(E_max)]])
R1 = R_Atm*R1/sum(sum(R1))

R2,ce,ee = histogram2d(abs(costh_r_gen_2),log10(E_r_gen),bins=(ne,ne),\
                    range=[[0.0,1.0],[log10(E_min),log10(E_max)]])
R2 = R_Atm*R2/sum(sum(R2))

DAT1 = vstack((costh_centers,E_r_centers,R1))
DAT2 = vstack((costh_centers,E_r_centers,R2))
recoildat_fname1 = recoil_dir+'AtmNu_Ecosth_'+Nuc.Name+'_Stationary.txt'
recoildat_fname2 = recoil_dir+'AtmNu_Ecosth_'+Nuc.Name+'_CygnusTracking.txt'

file_exists = os.path.exists(recoildat_fname1)
if file_exists:
    DAT_prev1 = loadtxt(recoildat_fname1)
    DAT_prev2 = loadtxt(recoildat_fname2)
    if (shape(DAT_prev1)[0]==shape(DAT1)[0])&(shape(DAT_prev1)[1]==shape(DAT1)[1]):
        DAT1[2:,:] = (DAT_prev1[2:,:]+DAT1[2:,:])/2.0
        DAT2[2:,:] = (DAT_prev2[2:,:]+DAT2[2:,:])/2.0

        savetxt(recoildat_fname1,DAT1)
        savetxt(recoildat_fname2,DAT2)
        print('merged')
    else:
        savetxt(recoildat_fname1,DAT1)
        savetxt(recoildat_fname2,DAT2)
        print('overwritten')
else:
    savetxt(recoildat_fname1,DAT1)
    savetxt(recoildat_fname2,DAT2)
    print('first write')
