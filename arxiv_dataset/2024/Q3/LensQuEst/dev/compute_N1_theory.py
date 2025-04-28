from lensingbiases._lensing_biases import lensingbiases as bias

#first we need ClTT
import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))

from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *
import pickle
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib
from tqdm import trange, tqdm


oup_fname = '../data/input/universe_Planck15/camb/CAMB_outputs.pkl'
print(oup_fname)
f = open(oup_fname, 'rb') 
powers,cl,c_lensed,c_lens_response = pickle.load(f)
f.close()

totCL=powers['total']
unlensedCL=powers['unlensed_scalar']

#unlensed
L = np.arange(unlensedCL.shape[0])

unlensedTT = unlensedCL[:,0]
F = unlensedTT
funlensedTT = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

unlensedEE = unlensedCL[:,1]
F = unlensedEE
funlensedEE = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

unlensedBB = unlensedCL[:,2]
F = unlensedBB
funlensedBB = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

unlensedTE = unlensedCL[:,3]
F = unlensedTE
funlensedTE = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

#lens potential
L = np.arange(cl.shape[0])
PP = cl[:,0]
fPP = interp1d(L, PP, kind='linear', bounds_error=False, fill_value=0.)

PT = cl[:,1]
fPT = interp1d(L, PT, kind='linear', bounds_error=False, fill_value=0.)

PE = cl[:,2]
fPE = interp1d(L, PE, kind='linear', bounds_error=False, fill_value=0.)

#lensed maps
L = np.arange(totCL.shape[0])

lensedTT = totCL[:,0]
F = lensedTT
flensedTT = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

lensedEE = totCL[:,1]
F = lensedEE
flensedEE = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

lensedBB = totCL[:,2]
F = lensedBB
flensedBB = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

lensedTE = totCL[:,3]
F = lensedTE
flensedTE = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)



#TgradT
L = np.arange(c_lens_response.shape[0])

cTgradT = c_lens_response.T[0]

fTgradT_log = interp1d(L, np.log(cTgradT), kind='linear', bounds_error=False, fill_value=0.)
fTgradT = lambda L:np.exp(fTgradT_log(L))


print("Map properties")

# number of pixels for the flat map
nX = 1200
nY = 1200

# map dimensions in degrees
sizeX = 20.
sizeY = 20.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 51  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S4/SO specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)


#compute totalTT
lensedTT = totCL[:,0]/(L*(L+1))*2*np.pi
F = lensedTT
flensedTT = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

ftot = lambda l : flensedTT(l) + cmb.fForeground(l) + cmb.fdetectorNoise(l)



lRange = [1., 7e3]
lensedcmbfile = "n1_data/lensedCls.dat"
f = open(lensedcmbfile, "w") 
print('# %15s %15s %15s %15s %15s'%('L', 'TT', 'EE', 'BB', 'TE'), file=f)
for l in range(int(lRange[0])+1, int(lRange[1])):
    print('  %15s %15e %15e %15e %15e'%(l,
                                        ftot(l)*l*(l+1)/(2*np.pi),
                                        flensedEE(l),
                                        flensedBB(l), 
                                        flensedTE(l)), file=f)
f.close()

phifile = "n1_data/lenspotentialCls.dat"
f = open(phifile, "w") 
print('# %15s %15s %15s %15s %15s %15s %15s %15s'%('L', 'TT', 'EE', 'BB', 'TE', 'PP', 'TP', 'EP'), file=f)


for l in range(int(lRange[0])+1, int(lRange[1])):

    print('  %15s %15e %15e %15e %15e %15e %15e %15e'%(l,
                                                       funlensedTT(l),
                                                       funlensedEE(l),
                                                       funlensedBB(l),
                                                       funlensedTE(l),
                                                       fPP(l),
                                                       fPT(l),
                                                       fPE(l)), file=f)
f.close()

FWHM = 1.4 #arcmin
noise_level = 7 #muK*arcmin

lmin = 2
lmax = 3.5e3
lmaxout = lmax
lmax_TT = lmax
tmp_output = 'n1_data'
print('computing N1')

#### TEST WITH SAMPLE DATA
# phifile = 'n1_data/test_data_set_lenspotentialCls.dat'
# lensedcmbfile = 'n1_data/test_data_set_lensedCls.dat'
# FWHM = 3.5 #arcmin
# noise_level = 17.72 #muK*arcmin
####

bias.compute_n0(phifile, lensedcmbfile, FWHM/60,
                noise_level,
                lmin,
                lmaxout,
                lmax,
                lmax_TT,
                0,
                tmp_output)

bias.compute_n1(phifile, lensedcmbfile, FWHM/60,
                noise_level,
                lmin,
                lmaxout,
                lmax,
                lmax_TT,
                0,
                tmp_output)