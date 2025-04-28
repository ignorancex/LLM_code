import pickle
import warnings

import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))
from tqdm import tqdm,trange

from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *
import pickle
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np


FILE_IDX = int(sys.argv[1])
#######
IN_DATA_FNAMES = ['/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_%d.pkl'%(i) for i in range(1,51)]


pairs = [
#   [0,0], #N0
#   [0,1], #kappa
#   [1,0], #kappa
#   [0,2], #N1
#   [1,1], #N1
#   [2,0], #N1
#    [0,3], #should vanish
#    [1,2], #should vanish
#    [2,1], #should vanish
#    [3,0], #should vanish
#    [0,4], #N2 
#    [1,3], #N2
#    [2,2], #N2
#    [3,1], #N2
#    [4,0], #N2
   [-1, -1], #QE
   [-2, -2], #unlensed
]



warnings.filterwarnings("ignore")
#####

oup_fname = '../data/input/universe_Planck15/camb/CAMB_outputs.pkl'
print(oup_fname)
f = open(oup_fname, 'rb') 
powers,cl,c_lensed,c_lens_response = pickle.load(f)
f.close()

totCL=powers['total']
unlensedCL=powers['unlensed_scalar']

L = np.arange(unlensedCL.shape[0])

unlensedTT = unlensedCL[:,0]/(L*(L+1))*2*np.pi
F = unlensedTT
funlensedTT = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

L = np.arange(cl.shape[0])
PP = cl[:,0]
rawPP = PP*2*np.pi/((L*(L+1))**2)
rawKK = L**4/4 * rawPP

fKK = interp1d(L, rawKK, kind='linear', bounds_error=False, fill_value=0.)

L = np.arange(totCL.shape[0])

lensedTT = totCL[:,0]/(L*(L+1))*2*np.pi
F = lensedTT
flensedTT = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


ftot = lambda l : flensedTT(l) + cmb.fForeground(l) + cmb.fdetectorNoise(l)


L = np.arange(c_lens_response.shape[0])

cTgradT = c_lens_response.T[0]/(L*(L+1))*2*np.pi

fTgradT = interp1d(L, cTgradT, kind='linear', bounds_error=False, fill_value=0.)

# In[3]:



# In[4]:




# In[5]:


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


# In[6]:


print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: ftot(l) 

# # reinterpolate: gain factor 10 in speed
# L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
# F = np.array(list(map(forCtotal, L)))
# cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)
cmb.fCtotal = ftot # no longer reinterpolating since it seems like it leads to errors?


in_data = {}

fname = IN_DATA_FNAMES[FILE_IDX]
f = open(fname, 'rb') 
c_in_data = pickle.load(f) 
f.close()
for key in c_in_data:
    if(key not in in_data.keys()):
        in_data[key] = np.array(c_in_data[key])
    else:
        in_data[key] = np.vstack( (in_data[key],np.array(c_in_data[key])) )


for key in in_data:
    print(key, np.shape(in_data[key]))
    
    
pairs = [
    [0,0], #N0
    [0,1], #kappa
    [1,0], #kappa
    [1,1], #N1
    [0,2], #N1
    [2,0], #N1
    [-1, -1], #QE
    [-2, -2], #unlensed
]

data_names = {
    0: 'cmb0F_1',
    1: 'lCmbF_o1_1',
    2: 'lCmbF_o2_1',
    -1: 'lCmbF_1',
    -2: 'totalF_0',
}


ps_data = {}

print("Gets a theoretical prediction for the noise")
fNqCmb_fft = baseMap.forecastN0Kappa(funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
Ntheory = lambda l: fNqCmb_fft(l) 






TkTkps = []
TkpTks = []
TTps = []
TpTs = []

kappa_fixed = baseMap.genGRF(fKK, test=False)

for s_idx in trange(1):
    #Qu+23 E10
    mapsF = [in_data['cmb0F_1'][s_idx], 
            in_data['cmb0F_1'][s_idx+100], 
            in_data['cmb0F_1'][s_idx+200],
            in_data['cmb0F_1'][s_idx+300]] 
    maps = list(map(baseMap.inverseFourier, mapsF))
  
    kappasF = [kappa_fixed,
            kappa_fixed,
            in_data['kCmbF_1'][s_idx],
            in_data['kCmbF_1'][s_idx+100]]
  
    #do lensing
    maps = [baseMap.doLensing(cmb0, kappaFourier=kCmbFourier) for cmb0, kCmbFourier in zip(maps, kappasF)]
    mapsF = list(map(baseMap.fourier, maps))
  
    #apply foreground and noise
    fgsF  = [in_data['fgF_1'][s_idx], 
            in_data['fgF_1'][s_idx+100], 
            in_data['fgF_1'][s_idx+200],
            in_data['fgF_1'][s_idx+300]] 
  
    noisesF =  [in_data['noiseF_1'][s_idx], 
            in_data['noiseF_1'][s_idx+100], 
            in_data['noiseF_1'][s_idx+200],
            in_data['noiseF_1'][s_idx+300]] 
  
  
    mapsF = [mapF + fgF + noiseF for mapF, fgF, noiseF in zip(mapsF, fgsF, noisesF)]
    maps = list(map(baseMap.inverseFourier, mapsF))
  
  
    TkTkp = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=mapsF[0],
                                         dataFourier2=mapsF[1])
  
    TkpTk = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=mapsF[1],
                                         dataFourier2=mapsF[0])
  
    TTp   = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=mapsF[2],
                                         dataFourier2=mapsF[3])
  
    TpT   = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=mapsF[3],
                                         dataFourier2=mapsF[2])
  

    if(len(TkTkps)==0):
        TkTkps = np.array([TkTkp])
    else:
        TkTkps = np.vstack((TkTkps, np.array([TkTkp])))


    if(len(TkpTks)==0):
        TkpTks = np.array([TkpTk])
    else:
        TkpTks = np.vstack((TkpTks, np.array([TkpTk])))



    if(len(TTps)==0):
        TTps = np.array([TTp])
    else:
        TTps = np.vstack((TTps, np.array([TTp])))
      

    if(len(TpTs)==0):
        TpTs = np.array([TpT])
    else:
        TpTs = np.vstack((TpTs, np.array([TpT])))
