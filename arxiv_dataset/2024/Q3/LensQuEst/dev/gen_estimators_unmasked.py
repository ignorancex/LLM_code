import pickle
import warnings

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
import numpy as np
#######
IN_DATA_FNAMES = ['/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_%d.pkl'%(i) for i in range(1,51)]



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
funlensedTT_log = interp1d(L, np.log(F), kind='linear', bounds_error=False, fill_value=0.)
funlensedTT = lambda L:np.exp(funlensedTT_log(L))


L = np.arange(cl.shape[0])
PP = cl[:,0]
rawPP = PP*2*np.pi/((L*(L+1))**2)
rawKK = L**4/4 * rawPP

fKK_log = interp1d(L, np.log(rawKK), kind='linear', bounds_error=False, fill_value=0.)
fKK = lambda L:np.exp(fKK_log(L))


L = np.arange(totCL.shape[0])

lensedTT = totCL[:,0]/(L*(L+1))*2*np.pi
F = lensedTT
flensedTT_log = interp1d(L, np.log(F), kind='linear', bounds_error=False, fill_value=0.)
flensedTT = lambda L:np.exp(flensedTT_log(L))


ftot = lambda l : flensedTT(l) + cmb.fForeground(l) + cmb.fdetectorNoise(l)


L = np.arange(c_lens_response.shape[0])

cTgradT = c_lens_response.T[0]/(L*(L+1))*2*np.pi

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


# In[12]:
in_data = {}
file_idx = eval(sys.argv[3])
fname = IN_DATA_FNAMES[file_idx-1]
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

# In[30]:



data_names = {
    0: 'cmb0F_1',
    1: 'lCmbF_o1_1',
    2: 'lCmbF_o2_1',
    3: 'lCmbF_o3_1',
    4: 'lCmbF_o4_1',
    -1: 'totalF_1',
    -2: 'totalF_0',
    -3: 'totalF_randomized_0',
}






# In[ ]:


from tqdm import trange, tqdm



# In[43]:


fgFourier = in_data['fgF_1']
noiseFourier = in_data['noiseF_1']


pair = [eval(sys.argv[1]), eval(sys.argv[2])]
print(pair)

data = {}

pair_key = '%d%d'%(pair[0],pair[1])
keys = [data_names[p] for p in pair]
print(pair, keys)
N_data = min(len(in_data[keys[0]]), len(in_data[keys[1]]))


c_data = []
c_data_sqrtN = []
c_data_kR  = []

for data_idx in trange(N_data):
    dataF0 = in_data[keys[0]][data_idx]
    dataF1 = in_data[keys[1]][data_idx]        
    QE = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal, 
                                         lMin=lMin, lMax=lMax, 
                                         dataFourier=dataF0,
                                         dataFourier2=dataF1)
    sqrtNhat = []
    kR = []
    if(pair[0]==pair[1]):
        sqrtNhat = baseMap.computeQuadEstKappaAutoCorrectionMap(fTgradT,
                                                                cmb.fCtotal, 
                                                                lMin=lMin, lMax=lMax, 
                                                                dataFourier=dataF0)

        if(len(c_data_sqrtN)==0):
            c_data_sqrtN = np.array([sqrtNhat])
        else:
            c_data_sqrtN = np.vstack((c_data_sqrtN, np.array([sqrtNhat])))


    if(len(c_data)==0):
        c_data = np.array([QE])
    else:
        c_data = np.vstack((c_data, np.array([QE])))
        
        
    assert(len(c_data)==data_idx+1)
    
data[pair_key] = c_data
data[pair_key+'_sqrtN'] = c_data_sqrtN
f = open('/oak/stanford/orgs/kipac/users/delon/LensQuEst/estimators_FILE%d_pair_%d_%d.pkl'%(file_idx, pair[0], pair[1]), 'wb') 
pickle.dump(data, f)
f.close()

