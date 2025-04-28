### execute at line 90 of calc_obs 
### to compare original and interpolated parameters vs. EEP

import sys, os, time, pickle
# PARS imports
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
sys.path.append('..')
from pa.lib import surface as sf
# cluster parameters imports
from lib import mist_util as mu
import config as cf
# Python imports
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# plt.rcParams.update({
# 	"text.usetex": True,
# 	"font.family": "serif", 
# 	"font.serif": "Computer Modern",
#     "font.size": 20
# })

mpl.rcParams['font.size'] = 16

### This part is similar to calc_obs.py ###

# pre-compute Roche model volume versus PARS's omega
# and PARS's omega versus MESA's omega
sf.calcVA()
sf.calcom()

# Load and filter MIST models
print('Loading MIST...', end='')
start = time.time()
st = mu.Set('../../data/mist_grid.npy')
print('%.2f' % (time.time() - start) + ' seconds.')

st.select_MS() # select main sequence
st.select_Z(cf.Z) # select metallicity
st.select_valid_rotation() # select rotation with omega < 1
st.set_omega0() # set omega from omega_M; ignore the L_edd factor

# choose isochrone ages so that the space of age prior parameters with appreciable likelihoods
# is covered sufficiently finely
nt = 11
it = 102
tM = np.unique(st.t)[it : it + nt] # MIST ages around 9.159
st.select(np.isin(st.t, tM)) # select the ages in the model set
# split each interval between MIST ages into 4 equal parts
ts = [np.linspace(tM[i], tM[i+1], 5) for i in range(nt - 1)] # this is a list of ndarrays
# also split the first 5 intervals [t_M, t_M + delta_t], such that t_M is an original MIST age,
# put the new age a fourth of the way from t_M to t_M + delta_t 
for i in range(5):
	ts_new = (3./4) * ts[i][0] + (1./4) * ts[i][1]
	ts[i] = np.insert(ts[i], 1, ts_new) 
t = np.unique(np.concatenate(ts)) # refined ages
is_tM = np.isin(t, tM) # whether the refined age is an original MIST age

print('Loading PARS...', end='', flush=True)
start = time.time()
with open('../../data/pars_grid_ZM' + str(cf.Z).replace('-', 'm').replace('.', 'p') + '.pkl', 'rb') as f: 
	pars = pickle.load(f)
print('%.2f' % (time.time() - start) + ' seconds.')
mu.Grid.pars = pars # give a PARS grid reference to the grid class
# apply the lower mass cut-off for the primaries according the region of interest on the CMD 
st1 = st.copy(); st1.select_age( t[-1] ) # pick the highest age
Mmin = mu.Mlim(st1)
st.select_mass(Mmin=Mmin)

### This part is different from calc_obs.py, 
### here we plot variables for interpolated and non-interpolated isochrones
eeps = []
rdiffs = []
ldiffs = []
t1 = tM[5] # 9.154362416107382
for oM0 in np.around(np.linspace(0, 0.7, 8), 1):
	# original model grid at this age, at highest rotational speed used
	st1 = st.copy() 
	st1.select_age( t1 ) 
	st1.select(st1.oM0 == oM0)
	# new, interpolated model grid at this age
	st2 = st.copy()
	st2.select(np.isin(st2.t, np.delete(tM, 5))) 
	st2.select_age( t1 ) # now interpolate to get this age
	st2.select(st2.oM0 == oM0)
	# EEPs
	EEP1 = np.around(st1.EEP).astype(int)
	EEP2 = np.around(st2.EEP).astype(int)
	EEP = np.intersect1d(EEP1, EEP2)
	eeps.append(EEP)
	mask1 = np.isin(EEP1, EEP)
	mask2 = np.isin(EEP2, EEP)
	# radii
	R1 = st1.R[mask1]
	R2 = st2.R[mask2]
	Rdiff = 2 * (R1 - R2) / (R1 + R2)
	rdiffs.append(Rdiff)
	# luminosities
	L1 = 10**st1.logL[mask1]
	L2 = 10**st2.logL[mask2]
	Ldiff = 2 * (L1 - L2) / (L1 + L2)
	ldiffs.append(Ldiff)
# # model grids at the neighboring ages
# # left neighbor
# st1_left = st.copy() 
# st1_left.select_age( tM[4] ) 
# st1_left.select(st1_left.oM0 == 0.7)
# # right neighbor
# st1_right = st.copy() 
# st1_right.select_age( tM[6] ) 
# st1_right.select(st1_right.oM0 == 0.7)

### plot ###

# put a box with age and omega_M information, maybe do several omega_M
textstr = '\n'.join((
    r'${\rm [M/H]}_{\rm M} = ' + str(cf.Z) + '$',
    r'$\log{\,t} = ' + cf.fstr(tM[5], 4) + '$',
	r'$\log{\,t}_{\rm neighbors} = \{' + cf.fstr(tM[4], 4) + ', ' + cf.fstr(tM[6], 4) + '\}$'))

for i in range(len(eeps)):
	plt.scatter(eeps[i], ldiffs[i])
plt.xlabel("EEP")
plt.ylabel(r"$(L_{\rm orig} - L_{\rm interp})\,/\,L_{\rm avg}$")
plt.text(0.25, 0.25, textstr, transform=plt.gca().transAxes, horizontalalignment='left',
        verticalalignment='top', bbox=dict(facecolor='w', alpha=0.0, edgecolor='w'))
plt.tight_layout()
plt.savefig('../../data/plots/L_EEP_interp_age.pdf', dpi=300)
plt.close()

for i in range(len(eeps)):
	plt.scatter(eeps[i], rdiffs[i])
plt.xlabel("EEP")
plt.ylabel(r'$(R_{\rm orig} - R_{\rm interp})\,/\,R_{\rm avg}}$')
plt.text(0.25, 0.25 , textstr, transform=plt.gca().transAxes, horizontalalignment='left',
        verticalalignment='top', bbox=dict(facecolor='w', alpha=0.0, edgecolor='w'))
plt.tight_layout()
plt.savefig('../../data/plots/R_EEP_interp_age.pdf', dpi=300)
plt.close()