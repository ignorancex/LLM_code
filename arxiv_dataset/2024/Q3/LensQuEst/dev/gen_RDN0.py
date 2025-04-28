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
#######
MASKING=False
ANISO=False

WORST_CASE = True
if(len(sys.argv)>2):
    if(sys.argv[2] == 'masking'):
        MASKING = True
        print('masked')
    if(sys.argv[2] == 'aniso'):
        ANISO = True
        print('anisotropic noise')

mask_file = 'mask_simple1200x1200.png'
psfile = 'point_sources_1200x1200.png'
psapod = 3

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
if(ANISO):
    with open('f_aniso_ftot.pkl', 'rb') as f:
        ftot = pickle.load(f)
    if(WORST_CASE):
        with open('f_aniso_ftot_worst_case.pkl', 'rb') as f:
            ftot = pickle.load(f)

    print('loaded estimated ftot')


L = np.arange(c_lens_response.shape[0])

cTgradT = c_lens_response.T[0]/(L*(L+1))*2*np.pi

fTgradT = interp1d(L, cTgradT, kind='linear', bounds_error=False, fill_value=0.)

# In[3]:



# In[4]:


errmap = None
with open('anisotropic_noise_map.pkl', 'rb') as f:
    errmap = pickle.load(f)
if(WORST_CASE):
    with open('anisotropic_noise_map_worst_case.pkl', 'rb') as f:
        errmap = pickle.load(f)
# In[5]:


print("Map properties")

# number of pixels for the flat map
nX = 1200
nY =1200

# map dimensions in degrees
sizeX = 20.
sizeY = 20.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra


# In[6]:


print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
cmb.fCtotal = ftot


in_data = {}
# In[9]:

#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
from scipy.ndimage import gaussian_filter 
from scipy.fft import fft2

mask = rgb2gray(plt.imread(mask_file))
apodized_mask = gaussian_filter(mask, 3)
point_sources = rgb2gray(plt.imread(psfile))
point_sources = gaussian_filter(point_sources, psapod) 
apodized_mask += point_sources
nPos = np.where(apodized_mask>1)
apodized_mask[nPos] = 1
mask = 1-mask
apodized_mask = 1 - apodized_mask

for a in apodized_mask:
    for b in a:
        assert(b<=1 and b>=0)

plt.imshow(apodized_mask)
plt.savefig('figures/apodized_masked_%dx%d.pdf'%(nX, nY),bbox_inches='tight')


s1_IN_DATA_FNAMES = ['/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_%d.pkl'%(i) for i in range(1,6)]
s2_IN_DATA_FNAMES = ['/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_%d.pkl'%(i) for i in range(46,51)]


for fname in tqdm(s1_IN_DATA_FNAMES):
    f = open(fname, 'rb') 
    c_in_data = pickle.load(f) 
    f.close()
    for key in c_in_data:
        if('totalF_0' not in key):
            continue
        if(key not in in_data.keys()):
            in_data[key] = np.array(c_in_data[key])
        else:
            in_data[key] = np.vstack( (in_data[key],np.array(c_in_data[key])) )

for fname in tqdm(s2_IN_DATA_FNAMES):
    f = open(fname, 'rb') 
    c_in_data = pickle.load(f) 
    f.close()
    for key in c_in_data:
        if('totalF_0' not in key):
            continue
        if(key not in in_data.keys()):
            in_data[key] = np.array(c_in_data[key])
        else:
            in_data[key] = np.vstack( (in_data[key],np.array(c_in_data[key])) )

            
if(ANISO):
    in_data = {}
    print('loading sims for anisotropinc noise')
    DATA_FNAME = '/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_for_aniso_rdn0.pkl'
    if(WORST_CASE):
        DATA_FNAME = '/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_for_aniso_rdn0_worst_case.pkl'

    with open(DATA_FNAME, 'rb') as f:
        in_data = pickle.load(f)


def get_file_and_index(sim_index, num_files=50, sims_per_file=10):
    file_index = sim_index // sims_per_file + 1
    inner_index = sim_index % sims_per_file
    return file_index, inner_index


d_idx = eval(sys.argv[1])
file_index, inner_index = get_file_and_index(d_idx)
d_IN_DATA_FNAME = '/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_%d.pkl'%(file_index)
print('Data %d is in %s index %d'%(d_idx, d_IN_DATA_FNAME, inner_index))

d = None
with open(d_IN_DATA_FNAME, 'rb') as f:
    c_in_data = pickle.load(f)
    d = c_in_data['totalF_1'][inner_index]
    if(ANISO):
        d = c_in_data['lCmbF_1'][inner_index]
        #first we'll renormalize error map so that when we apply the error map, 
        #at best, anisotropic_noise = isotropic noise
        print('applying anisotropic detector noise')
        aniso_noise = baseMap.inverseFourier(c_in_data['noiseF_1'][inner_index]) / np.min(errmap)
        aniso_noise = aniso_noise * errmap # apply errmap
        aniso_noise_fourier = baseMap.fourier(aniso_noise)

        d = d + c_in_data['fgF_1'][inner_index] + aniso_noise_fourier

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


#RDN0

RDN0_data = {}


ds1s = []
s1ds = []
s1s2s= []
s2s1s= []


tmp_idx = 0

if(MASKING):
    print('masking data')
    d = baseMap.fourier(baseMap.inverseFourier(d) * apodized_mask)
    
for s_idx in trange(50):
    s1 = in_data['totalF_0'][s_idx]
    if(MASKING):
        s1 = baseMap.fourier(baseMap.inverseFourier(s1) * apodized_mask)

    print('CURR', s_idx)
    for s2_idx in range(50):
        s2 = in_data['totalF_0'][s2_idx+len(in_data['totalF_0'])//2]

        if(MASKING):
            s2 = baseMap.fourier(baseMap.inverseFourier(s2) * apodized_mask)
           
        ds1 = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal,
                                             lMin=lMin, lMax=lMax,
                                             dataFourier=d,
                                             dataFourier2=s1)
        s1d = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal,
                                             lMin=lMin, lMax=lMax,
                                             dataFourier=s1,
                                             dataFourier2=d)
        s1s2 = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal,
                                             lMin=lMin, lMax=lMax,
                                             dataFourier=s1,
                                             dataFourier2=s2)
        s2s1 = baseMap.computeQuadEstKappaNorm(fTgradT, cmb.fCtotal,
                                             lMin=lMin, lMax=lMax,
                                             dataFourier=s2,
                                             dataFourier2=s1)

        RDN0_data = {
            'ds1' : ds1,
            's1d' : s1d,
            's1s2': s1s2,
            's2s1': s2s1
        }


        oup_fname = '/scratch/users/delon/LensQuEst/RDN0-in_data-%d-%d.pkl'%(d_idx,tmp_idx)
        if(MASKING):
            oup_fname = '/scratch/groups/risahw/delon/LensQuEst/RDN0-in_data-%d-%d.pkl'%(d_idx,tmp_idx)
        if(ANISO):
            oup_fname = '/scratch/users/delon/LensQuEst/RDN0-in_data-%d-%d-aniso.pkl'%(d_idx,tmp_idx)
            if(WORST_CASE):
                oup_fname = '/scratch/users/delon/LensQuEst/RDN0-in_data-%d-%d-aniso_worst_case.pkl'%(d_idx,tmp_idx)


        print(oup_fname)
        f = open(oup_fname, 'wb') 
        pickle.dump(RDN0_data, f)
        f.close()
        tmp_idx += 1

        del ds1
        del s1d
        del s1s2
        del s2s1
