import os
import torch
import torch.distributions as tdist
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from datetime import datetime
import numpy as np
from numpy import log, sqrt, exp
from scipy.io import FortranFile
from scipy.stats import pearsonr
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

# load data from file

f = FortranFile('./newz3taured.dat', 'r')
nlos=int(np.asscalar(f.read_ints()))
print("nlos = %d" %nlos)
npix=int(np.asscalar(f.read_ints()))
zred=np.asscalar(f.read_record('f4'))
blenkms=np.asscalar(f.read_record('f4'))
blen=np.asscalar(f.read_record('f4'))*0.001  #back into mpc/h
print(nlos,npix,zred,blenkms,blen)
taured=[]
nstep=1 #skipping through in steps of 1
for i in range(0,nlos,nstep):       
    tauredin=f.read_record('f4')
    taured.extend(tauredin)
f.close()
print('len(taured)=',len(taured))

nlos=int(nlos/nstep)

print ('nlos,npix=',nlos,npix)

taured=np.array(taured)

taured=np.reshape(taured,(nlos,1,npix))
reshapedTaured = np.reshape(taured, (256,256,1,npix))

print('shape of taured=',taured.shape)

# Split test, train, and validation data sets

tauValidate = np.reshape(reshapedTaured[0:114,0:114],(114*114,1,512))
tauTest = np.reshape(reshapedTaured[-114:,-114:],(114*114,1,512))
tauTrain = np.concatenate((
               np.reshape(reshapedTaured[:114,114:],(114*(256-114),1,512)), \
               np.reshape(reshapedTaured[114:-114],(256*(256-2*114),1,512)), \
               np.reshape(reshapedTaured[-114:,:-114],(114*(256-114),1,512))))

# Smooth tau in each dataset

tauTest = gaussian_filter1d(tauTest, 6, axis=-1, mode="wrap")
tauValidate = gaussian_filter1d(tauValidate, 6, axis=-1, mode="wrap")
tauTrain = gaussian_filter1d(tauTrain, 6, axis=-1, mode="wrap")

# remove outliers

mintaured=taured.min()
maxtaured=taured.max()
taured[taured > 0.5e9] = -1.e10

tauTest[tauTest > 0.5e9] = -1.e10
tauValidate[tauValidate > 0.5e9] = -1.e10
tauTrain[tauTrain > 0.5e9] = -1.e10

maxtaured=taured.max()
print (maxtaured)
# set to realistic max:

tauTest[tauTest < -0.5e9] = maxtaured
tauValidate[tauValidate < -0.5e9] = maxtaured
tauTrain[tauTrain < -0.5e9] = maxtaured

# calculate flux

flux=np.exp(-1.*taured)
fluxTest = np.exp(-1.*tauTest)
fluxValidate = np.exp(-1.*tauValidate)
fluxTrain = np.exp(-1.*tauTrain)

signalRMS = np.sqrt(np.mean(flux**2))
print("signalRMS", signalRMS)

# Move arrays to GPU
if torch.cuda.is_available():
    tauTest = torch.from_numpy(tauTest).cuda()
    tauValidate = torch.from_numpy(tauValidate).cuda()
    tauTrain = torch.from_numpy(tauTrain).cuda()


    fluxTest = torch.from_numpy(fluxTest).cuda()
    fluxValidate = torch.from_numpy(fluxValidate).cuda()
    fluxTrain = torch.from_numpy(fluxTrain).cuda()
    
else:
    tauTest = torch.from_numpy(tauTest)
    tauValidate = torch.from_numpy(tauValidate)
    tauTrain = torch.from_numpy(tauTrain)


    fluxTest = torch.from_numpy(fluxTest)
    fluxValidate = torch.from_numpy(fluxValidate)
    fluxTrain = torch.from_numpy(fluxTrain)
    

tauredzero=taured[0,0,...]

fluxzero=flux[0,0,...]

plt.plot(tauredzero)
plt.plot(fluxzero)
plt.show()


npHistResult = np.histogram(taured, bins=[0,2,10000])
binProportions = npHistResult[0]/(2**25)

if torch.cuda.is_available():
    binProportions = torch.from_numpy(binProportions).cuda()

del reshapedTaured