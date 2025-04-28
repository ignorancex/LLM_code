import numpy as np
import constants as c
import pickle
from pdf_trans import *
from collect_CF_results import *

'''
H1743-322 (Steiner et al. 2012a)
- DataThief ASCII table contains {rCF, f_rCF}
- {M, D, i} normally and independently distributed
- Adopt the Ozel mass distribution for f_M
'''
#====================================================================================================
# Collect several results from the published disk continnum fitting measurements and output them in a pickle file.
# CGS units: { aCF [-], rCF [Rg], fcolCF [-], M [g], D [cm], i [rad], gGR(r [Rg],i [rad]), gNT(r [Rg]) }
#     f_aCF  - Marginal density of the black hole spin measured from continuum fitting
#     f_rCF  - Marginal density of the inner disk radius measured from continuum fitting
#     fcolCF - Color correction factor adopted when measuring the BH spin from continuum fitting
#     f_MDi  - Joint density of (M, D, inc) used for the black hole spin error budget
#     f_M    - Marginal density of the black hole mass
#     f_D    - Marginal density of the source distance
#     f_i    - Marginal density of the inner disk inclination
#     gGR    - Photon propagation relativistic disk flux correction factor gGR(r,i)
#     gNT    - Disk structure     relativistic disk flux correction factor gNT(r)
#
#     aCF_min, aCF_max - Minimum, Maximum black hole spin [-] measured from continuum fitting
#     rCF_min, rCF_max - Minimum, Maximum inner disk radius [Rg] measured from continuum fitting
#       M_min,   M_max - Minimum, Maximum black hole mass [g]
#       D_min,   D_max - Minimum, Maximum source distance [cm]
#       i_min,   i_max - Minimum, Maximum inner disk inclination [rad]
#
# NOTE: These probability densitites are Python objects with a .pdf atttribute.
#       f_x.pdf(x) <-- This accesses the probability density of f_x @ x

fdtf  = '../data/CF/rCF_H1743.txt'
fGR   = '../data/GR/gGR_gNT_H1743.h5'
fout  = '../data/CF/CF_results_H1743.p'

# Number of standard deviations for setting the Min/Max of each marginal density
Nstd = 4

#----------------------------------------------------------------------------------------------------
# Collect: f_aCF, f_rCF

# Inner disk radius, rCF [Rg]
rCF, f_rCF = get_DataThief_fx(file=fdtf)
rCF_min, rCF_max = np.min(rCF), np.max(rCF)

# Black hole spin, aCF [-]
f_aCF   = transform_frisco2fspin(f_risco=f_rCF, Na=1000)
aCF_min = risco2spin(risco=rCF_max)
aCF_max = risco2spin(risco=rCF_min)

#----------------------------------------------------------------------------------------------------
# Collect: fcolCF, f_M, f_D, f_i, f_MDi

# Color correction factor, fcol [-]
fcolCF = 1.6

# Mass, M [g]
M_min, M_max = 5.0 * c.sun2g, 15.0 * c.sun2g
f_M = get_fM_Ozel()  # [g]^-1

# Distance, D [cm]
#D_val, D_err = 8.5 * c.kpc2cm, 0.8 * c.kpc2cm
#D_min, D_max = get_minmax(x_val=D_val, x_err=D_err, Nstd=Nstd)
#f_D = get_truncnorm(x_val=D_val, x_err=D_err, x_min=D_min, x_max=D_max)
fdtf_D         = droot + 'data/H1743/D_H1743.txt'
D_kpc, f_D_kpc = get_DataThief_fx(file=fdtf_D)
D_cm           = np.array(D_kpc) * c.kpc2cm
f_D_cm         = f_D_kpc.pdf(D_kpc) / c.kpc2cm
f_D_interp     = interp1d(x=D_cm, y=f_D_cm, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
f_D            = generate_f_x(f_x_interp=f_D_interp)
D_min, D_max   = np.min(D_cm), np.max(D_cm)

# Inner disk inclination, i [rad]
#i_val, i_err = 75.0 * c.deg2rad, 3.0 * c.deg2rad
#i_min, i_max = get_minmax(x_val=i_val, x_err=i_err, Nstd=Nstd)
#f_i = get_truncnorm(x_val=i_val, x_err=i_err, x_min=i_min, x_max=i_max)
fdtf_i = droot + 'data/H1743/i_H1743.txt'
i_deg, f_i_deg = get_DataThief_fx(file=fdtf_i)
i_rad          = np.array(i_deg) * c.deg2rad
f_i_rad        = f_i_deg.pdf(i_deg) / c.deg2rad
f_i_interp     = interp1d(x=i_rad, y=f_i_rad, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
f_i            = generate_f_x(f_x_interp=f_i_interp)
i_min, i_max   = np.min(i_rad), np.max(i_rad)

# Joint density of {M, D, i}
f_MDi = get_fMDi_indep(f_M=f_M, f_D=f_D, f_i=f_i)

#----------------------------------------------------------------------------------------------------
# Collect: gGR(r,i), gNT(r)
f        = h5py.File(fGR, 'r')
r_grid   = f.get('r_grid')[:]              # [Rg]
i_grid   = f.get('i_grid')[:] * c.deg2rad  # [rad]
gGR_grid = f.get('gGR')[:,:]               # gGR(r,i)
gNT_grid = f.get('gNT')[:]                 # gNT(r)
f.close()

#----------------------------------------------------------------------------------------------------
# Approximate the location of the K-peak in the unknown f_K(K) marginal distribution
M     = np.linspace(M_min, M_max, 1000)
M_val = M[np.argmax(f_M.pdf(M))]
D     = np.linspace(D_min, D_max, 1000)
D_val = D[np.argmax(f_D.pdf(D))]
i     = np.linspace(i_min, i_max, 1000)
i_val = i[np.argmax(f_i.pdf(i))]
gGR   = RectBivariateSpline(x=r_grid, y=i_grid, z=gGR_grid, kx=3, ky=3, s=0)
gNT   = interp1d(x=r_grid, y=gNT_grid, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
K_val = calc_K_peak(f_r=f_rCF, fcol=fcolCF, M=M_val, D=D_val, i=i_val, gGR=gGR, gNT=gNT)
print("\nApproximate K-peak in f_K(K): ", K_val)

#----------------------------------------------------------------------------------------------------
# Write out the results to a pickle file
dumpdict = {'f_aCF':f_aCF, 'f_rCF':f_rCF, 'fcolCF':fcolCF, 'f_M':f_M, 'f_D':f_D, 'f_i':f_i, 'f_MDi':f_MDi, \
            'gGR':gGR_grid, 'gNT':gNT_grid, 'r_grid':r_grid, 'i_grid':i_grid, \
            'aCF_min':aCF_min, 'aCF_max':aCF_max, 'rCF_min':rCF_min, 'rCF_max':rCF_max, \
            'M_min':M_min, 'M_max':M_max, 'D_min':D_min, 'D_max':D_max, 'i_min':i_min, 'i_max':i_max, \
            'K_val':K_val}
pickle.dump(dumpdict, open(fout, 'wb'))

#====================================================================================================

# Example of calculating the joint density w/ broadcasting
#f_MDi_vals = f_MDi.pdf(M=M[:,None,None], D=D[None,:,None], i=i[None,None,:])
