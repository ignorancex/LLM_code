import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "STIX"
rcParams["mathtext.fontset"] = "stix"
rcParams['figure.figsize']=(10,7)
rcParams['axes.labelsize'] = 18
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize']=20

from scipy.interpolate import interp1d

eos_data_dir = '../eos_data/nsat'
interp_cache = {}
radius_samples = []

fig,ax=plt.subplots()
for e in range(1,2001):
    eos_file = '{}/{}.dat'.format(eos_data_dir, int(e))
    rdat, mdat, _ = np.loadtxt(eos_file, unpack=True)
    plt.plot(rdat,mdat,color='silver',alpha=0.1)
    interp = scipy.interpolate.interp1d(mdat, rdat)
    interp_cache[e] = interp
    radius_samples.append(interp_cache[e](1.4))
    if(int(e)==477):
        print(interp_cache[e](1.4))
    if(int(e)==895):
        print(interp_cache[e](1.4))
    if(int(e)==1280):
        print(interp_cache[e](1.4))
eos_file = '{}/{}.dat'.format(eos_data_dir, 477)
rdat, mdat, _ = np.loadtxt(eos_file, unpack=True)
plt.plot(rdat,mdat,color='blue',label='soft')
eos_file = '{}/{}.dat'.format(eos_data_dir, 895)
rdat, mdat, _ = np.loadtxt(eos_file, unpack=True)
plt.plot(rdat,mdat,color='orangered',label='medium')
eos_file = '{}/{}.dat'.format(eos_data_dir, 1280)
rdat, mdat, _ = np.loadtxt(eos_file, unpack=True)
plt.plot(rdat,mdat,color='green',label='stiff')
plt.xlim(7,18)
plt.grid()
ax.tick_params(direction='in',which='both')
plt.ylabel(r'Mass $[\mathrm{M}_{\odot}]$')
plt.xlabel(r'Radius $[\mathrm{km}]$')
plt.legend()
plt.tight_layout()
fig.savefig('Figure1-mass-radius-curves.png',dpi=200)