
# coding: utf-8

# In[1]:



# In[2]:


import os, sys
WORKING_DIR = os.path.dirname(os.path.abspath(''))
sys.path.insert(1, os.path.join(WORKING_DIR,'LensQuEst'))

#to get latex to work, shoulldn't be necessary for most ppl
os.environ['PATH'] = "%s:/usr/local/cuda-11.2/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/anaconda/bin:/home/delon/texlive/bin/x86_64-linux:/home/delon/.local/bin:/home/delon/bin"%os.environ['PATH']


# In[3]:


from universe import *
from halo_fit import *
from cmb import *
from flat_map import *
from weight import *
from pn_2d import *
import pickle
from tqdm import trange,tqdm 

from itertools import product



# In[22]:


#######
DATA_FNAME = '/data/delon/LensQuEst/N0_Ni.pkl'
mean_field_file = 'mask_simple400x400_point_sources_bigger.pkl'
preload=True
import warnings
warnings.filterwarnings("ignore")
#####


# In[5]:


print("Map properties")

# number of pixels for the flat map
nX = 400 # 1200
nY = 400 #1200

# map dimensions in degrees
sizeX = 10.
sizeY = 10.

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
cmb = StageIVCMB(beam=1., noise=1., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
# basiscally gets what we theoretically expect the
# power spectrum will look like
forCtotal = lambda l: cmb.ftotal(l)

# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


# In[7]:


print("CMB lensing power spectrum")
u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


# In[8]:


print("Gets a theoretical prediction for the noise")
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)


# In[9]:


#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# In[10]:


from scipy.ndimage import gaussian_filter 
from scipy.fft import fft2

mask = rgb2gray(plt.imread('mask_simple%dx%d.png'%(nX, nY)))
apodized_mask = gaussian_filter(mask, 3)
point_sources = rgb2gray(plt.imread('point_sources_bigger.png'))
point_sources = gaussian_filter(point_sources, 1.5) 
apodized_mask += point_sources
nPos = np.where(apodized_mask>1)
apodized_mask[nPos] = 1
mask = 1-mask
apodized_mask = 1 - apodized_mask

for a in apodized_mask:
    for b in a:
        assert(b<=1 and b>=0)


# In[12]:


from tqdm import trange,tqdm 
import pickle

from itertools import product

N_RUNS = 95
poss = list(product([True, False],[True, False], range(N_RUNS)))
fsky = np.sum(apodized_mask)/(nX*nY)
c1 = fsky
c2 = fsky**2

data = {}
if(preload):
    f = open(DATA_FNAME, 'rb') 
    data = pickle.load(f) 
    f.close()
    for key in data:
        print(key, np.shape(data[key]))


for MASKED, LENSED, run_n in tqdm(poss):
    post_fix = '%d_%d'%(MASKED,LENSED)

    mean_field = pickle.load(open(mean_field_file, 'rb'))
    if(not MASKED):
        mean_field = np.zeros_like(mean_field)

    totalCmbFourier, totalCmb = 0, 0
    if(not LENSED):
        totalCmbFourier = baseMap.genGRF(cmb.ftotal)
        totalCmb = baseMap.inverseFourier(totalCmbFourier)
    elif(LENSED):
        cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
        cmb0 = baseMap.inverseFourier(cmb0Fourier)

        kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
        kCmb = baseMap.inverseFourier(kCmbFourier)

        lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
        lensedCmbFourier = baseMap.fourier(lensedCmb)

        fgFourier = baseMap.genGRF(cmb.fForeground, test=False)
        lensedCmbFourier = lensedCmbFourier + fgFourier
        lensedCmb = baseMap.inverseFourier(lensedCmbFourier)

        noiseFourier = baseMap.genGRF(cmb.fdetectorNoise, test=False)
        totalCmbFourier = lensedCmbFourier + noiseFourier
        totalCmb = baseMap.inverseFourier(totalCmbFourier)

    if(MASKED):
        totalCmb = apodized_mask*totalCmb
        totalCmbFourier = baseMap.fourier(totalCmb)

    c_Data = {}
    
    #kappa
    c_Data['k'+post_fix] = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, 
                                                           cmb.fCtotal, 
                                                           lMin=lMin, lMax=lMax,
                                                           dataFourier=totalCmbFourier)
    if(MASKED):
        c_Data['k'+post_fix] -= mean_field

    #Nhat
    c_Data['sqrtNhat'+post_fix] = baseMap.computeQuadEstKappaAutoCorrectionMap(cmb.funlensedTT, 
                                                                               cmb.fCtotal, 
                                                                               lMin=lMin, lMax=lMax, 
                                                                               dataFourier=totalCmbFourier)

    #kappaR
    totalCmbFourierRandomized = baseMap.randomizePhases(totalCmbFourier)
    totalCmbRandomized = baseMap.inverseFourier(totalCmb)
    c_Data['kR'+post_fix] = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, 
                                                            cmb.fCtotal, 
                                                            lMin=lMin, lMax=lMax,
                                                            dataFourier=totalCmbFourierRandomized)

    c_Data['Nhat'+post_fix] = [0,0,0]
    c_Data['kRkR'+post_fix] = [0,0,0]
    c_Data['kk'+post_fix]   = [0,0,0]
    
    c_Data['kk'+post_fix][0],   c_Data['kk'+post_fix][1],   c_Data['kk'+post_fix][2]   = baseMap.powerSpectrum(dataFourier=c_Data['k'+post_fix])
    c_Data['Nhat'+post_fix][0], c_Data['Nhat'+post_fix][1], c_Data['Nhat'+post_fix][2] = baseMap.powerSpectrum(dataFourier=c_Data['sqrtNhat'+post_fix])
    c_Data['kRkR'+post_fix][0], c_Data['kRkR'+post_fix][1], c_Data['kRkR'+post_fix][2] = baseMap.powerSpectrum(dataFourier=c_Data['kR'+post_fix])

    if(MASKED):
        #fsky corrections
        c_Data['kk'+post_fix][1] /= c1
        c_Data['kk'+post_fix][2] /= c1
        
        c_Data['Nhat'+post_fix][1] /= c2
        c_Data['Nhat'+post_fix][2] /= c2

        c_Data['kRkR'+post_fix][1] /= c2
        c_Data['kRkR'+post_fix][2] /= c2

    for key in c_Data:
        if(key not in data.keys()):
            data[key] = np.array([c_Data[key]])
        else:
            data[key] = np.vstack((np.array([c_Data[key]]), data[key]))  
            
    f = open(DATA_FNAME, 'wb') 
    pickle.dump(data, f)
    f.close()


