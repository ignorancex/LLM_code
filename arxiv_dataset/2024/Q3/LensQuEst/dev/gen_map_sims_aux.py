import sys

#######
DATA_FNAME = '/oak/stanford/orgs/kipac/users/delon/LensQuEst/map_sims_800x800_20x20_fixed_k_diff_CMB_%d.pkl'%(eval(sys.argv[1]))
print(DATA_FNAME)

preload=False
N_RUNS = 10
import warnings
warnings.filterwarnings("ignore")
#####


# In[2]:


import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))


# In[3]:


from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *
import pickle
import seaborn as sns
from scipy.stats import spearmanr


# In[4]:


print("Map properties")

# number of pixels for the flat map
nX = 800
nY = 800

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


# In[5]:


print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: cmb.ftotal(l) 

# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


# In[6]:


print("CMB lensing power spectrum")
u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


# In[7]:


print("Gets a theoretical prediction for the noise")
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
Ntheory = lambda l: fNqCmb_fft(l) 


# In[9]:


#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#
#from scipy.ndimage import gaussian_filter 
#from scipy.fft import fft2
#
#mask = rgb2gray(plt.imread('mask_simple%dx%d.png'%(nX, nY)))
#apodized_mask = gaussian_filter(mask, 3)
#point_sources = rgb2gray(plt.imread('point_sources_bigger.png'))
#point_sources = gaussian_filter(point_sources, 1.5) 
#apodized_mask += point_sources
#nPos = np.where(apodized_mask>1)
#apodized_mask[nPos] = 1
#mask = 1-mask
#apodized_mask = 1 - apodized_mask
#
#for a in apodized_mask:
#    for b in a:
#        assert(b<=1 and b>=0)
## plt.imshow(apodized_mask)


# In[17]:


from tqdm import trange,tqdm 
import pickle

from itertools import product

poss = list(product([True], range(N_RUNS)))


oup_fname = '../data/input/universe_Planck15/camb/CAMB_outputs.pkl'
print(oup_fname)
f = open(oup_fname, 'rb') 
powers,cl,c_lensed,c_lens_response = pickle.load(f)
f.close()

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

data = {}
if(preload):
    f = open(DATA_FNAME, 'rb') 
    data = pickle.load(f) 
    f.close()
    for key in data:
        print(key, np.shape(data[key]))

        
        

#fixed lensing potential
kCmbFourier = baseMap.genGRF(fKK, test=False)
kCmb = baseMap.inverseFourier(kCmbFourier)


for LENSED, run_n in tqdm(poss):
    post_fix = '_%d'%(LENSED)

    c_Data = {}
    c_Data['cmb0F'+post_fix] = None
    c_Data['kCmbF'+post_fix] = None
    c_Data['lCmbF'+post_fix] = None
    for i in range(1,5):
        c_Data['lCmbF_o%d'%(i)+post_fix] = None
    c_Data['fgF'+post_fix] = None
    c_Data['noiseF'+post_fix] = None
    c_Data['totalF'+post_fix] = None
#    c_Data['totalF_M'+post_fix] = None

    
    totalCmbFourier, totalCmb = None, None
    
    if(not LENSED):
        totalCmbFourier = baseMap.genGRF(cmb.ftotal)
        totalCmb = baseMap.inverseFourier(totalCmbFourier)
        
    elif(LENSED):
        cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
        cmb0 = baseMap.inverseFourier(cmb0Fourier)
        c_Data['cmb0F'+post_fix] = cmb0Fourier
        
        c_Data['kCmbF'+post_fix] = kCmbFourier
        
#        for i in range(1,5):
#            lensedCmb = baseMap.doLensingTaylor(unlensed=cmb0, kappaFourier=kCmbFourier, order=i)
#            lensedCmbFourier = baseMap.fourier(lensedCmb)
#            c_Data['lCmbF_o%d'%(i)+post_fix] = lensedCmbFourier
            
        lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
        lensedCmbFourier = baseMap.fourier(lensedCmb)
        c_Data['lCmbF'+post_fix] = lensedCmbFourier
        
        fgFourier = baseMap.genGRF(cmb.fForeground, test=False)
        lensedCmbFourier = lensedCmbFourier + fgFourier
        lensedCmb = baseMap.inverseFourier(lensedCmbFourier)
        c_Data['fgF'+post_fix] = fgFourier

        noiseFourier = baseMap.genGRF(cmb.fdetectorNoise, test=False)
        totalCmbFourier = lensedCmbFourier + noiseFourier
        totalCmb = baseMap.inverseFourier(totalCmbFourier)
        c_Data['noiseF'+post_fix] = noiseFourier
        
    c_Data['totalF'+post_fix] = totalCmbFourier

#    totalCmb = apodized_mask*totalCmb
#    totalCmbFourier = baseMap.fourier(totalCmb)
#        
#    c_Data['totalF_M'+post_fix] = totalCmbFourier


    for key in c_Data:
        if(c_Data[key] is None):
            continue
        if(key not in data.keys()):
            data[key] = np.array([c_Data[key]])
        else:
            data[key] = np.vstack((np.array([c_Data[key]]), data[key]))  
            

    f = open(DATA_FNAME, 'wb') 
    p = pickle.Pickler(f) 
    p.fast = True 
    p.dump(data)
    f.close()
