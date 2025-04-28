import pickle
import sys
import warnings
DATA_IDX = eval(sys.argv[1])
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
cmb.fCtotal = ftot # no longer reinterpolating since it seems like it leads to errors?


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


# In[2]:




# In[3]:


## kappa_fixed = baseMap.genGRF(fKK, test=False)


TkTkps = []
TkpTks = []
TTps = []
TpTs = []






def gen_clFourier(fcur):
    f = lambda l: np.sqrt(fcur(l))
    clFourier = np.array(list(map(f, baseMap.l.flatten())))
    clFourier = np.nan_to_num(clFourier)
    clFourier = clFourier.reshape(np.shape(baseMap.l))
    return clFourier

print('precomputing Cls')
clFourier_ftot = gen_clFourier(ftot)
clFourier_funlensedTT = gen_clFourier(funlensedTT)
clFourier_fKK = gen_clFourier(fKK)
clFourier_fForeground = gen_clFourier(cmb.fForeground)
clFourier_fdetectorNoise = gen_clFourier(cmb.fdetectorNoise)


for s_idx in trange(100):
    np.random.seed(DATA_IDX*9742322 + s_idx )

    #Qu+23 E10
    mapsF = [baseMap.genGRF(funlensedTT, clFourier=clFourier_funlensedTT, test=False),
             baseMap.genGRF(funlensedTT, clFourier=clFourier_funlensedTT, test=False),
             baseMap.genGRF(funlensedTT, clFourier=clFourier_funlensedTT, test=False),
             baseMap.genGRF(funlensedTT, clFourier=clFourier_funlensedTT, test=False)]

    maps = list(map(baseMap.inverseFourier, mapsF))

    common_lensing = baseMap.genGRF(fKK, clFourier=clFourier_fKK, test=False)

    kappasF = [common_lensing,
               common_lensing,
               baseMap.genGRF(fKK, clFourier=clFourier_fKK, test=False),
               baseMap.genGRF(fKK, clFourier=clFourier_fKK, test=False)]



    #do lensing
    maps = [baseMap.doLensing(cmb0, kappaFourier=kappaF) 
            for cmb0, kappaF in zip(maps, kappasF)]
    mapsF = list(map(baseMap.fourier, maps))
    
    #apply foreground and noise
    fgsF  = [baseMap.genGRF(cmb.fForeground, clFourier=clFourier_fForeground, test=False),
             baseMap.genGRF(cmb.fForeground, clFourier=clFourier_fForeground, test=False),
             baseMap.genGRF(cmb.fForeground, clFourier=clFourier_fForeground, test=False),
             baseMap.genGRF(cmb.fForeground, clFourier=clFourier_fForeground, test=False)]



    
    noisesF =  [baseMap.genGRF(cmb.fdetectorNoise, clFourier=clFourier_fdetectorNoise, test=False),
                baseMap.genGRF(cmb.fdetectorNoise, clFourier=clFourier_fdetectorNoise, test=False),
                baseMap.genGRF(cmb.fdetectorNoise, clFourier=clFourier_fdetectorNoise, test=False),
                baseMap.genGRF(cmb.fdetectorNoise, clFourier=clFourier_fdetectorNoise, test=False)]




    
    
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


# In[4]:


np.array(TkTkps).shape

with open('/oak/stanford/orgs/kipac/users/delon/LensQuEst/N1-mcmc-morestats-Ts%d.pkl'%(DATA_IDX), "wb") as f:
    pickle.dump((TkTkps, TkpTks, TTps, TpTs), f)
