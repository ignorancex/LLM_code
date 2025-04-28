import os, sys
import pickle
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.patches import Arrow
from matplotlib.gridspec import GridSpec
from astropy.table import Table, Column, vstack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from utils import (load_plt_params, convolve_model, convolve_model_xy,
                   load_parula, ibin)

# Load limb depths:
w, evening, evening_error, morning, morning_error, covariances = np.loadtxt('catwoman_res100_priorlds.txt', unpack = True, usecols = (0,1,2,3,4,5))
w3, evening3, evening_error3, morning3, morning_error3, covariances3 = np.loadtxt('catwoman_res30_priorlds.txt', unpack = True, usecols = (0,1,2,3,4,5))

idx = np.where(w>=2)[0]
w, evening, evening_error, morning, morning_error, covariances = w[idx], evening[idx], evening_error[idx], morning[idx], morning_error[idx], covariances[idx]

# Load tdepth:
wtspec, dtspec, dtspec_err = np.loadtxt('catwoman_tspec_res100_priorlds.txt', unpack = True)

# Load model:
models = pickle.load(open('chimera_models.pkl', 'rb'))#pickle.load(open('goyal-bf-model.pkl','rb'))
cmodels = pickle.load(open('chimera_models_met.pkl','rb'))
evening_model, morning_model = models['models']
cevening_model, cmorning_model = cmodels['models']
bemodel, bmmodel = models['binned models']

total_model = models['total model']
# Smooth models for display purposes:
#from scipy.ndimage import gaussian_filter
#evening_model = gaussian_filter(evening_model, 1)
#morning_model = gaussian_filter(morning_model, 1)

#cevening_model = gaussian_filter(cevening_model, 1)
#cmorning_model = gaussian_filter(cmorning_model, 1)

#print(cevening_model)
# set the matplotlib parameters
pltparams = load_plt_params()
parula = load_parula()
COLOR = pltparams[pltparams['name']=='text.color']['value'][0]

fig = plt.figure(figsize=(10,12))
fig.set_facecolor('w')


gs = GridSpec(420+70+25, 1, figure=fig)#, height_ratios=[2,2,1])#, wspace=0.05, hspace=0)

cut = 150 
hspace = 10
ax1 = fig.add_subplot(gs[:150, 0]) 

print(total_model)
ax1.plot(models['wavelengths'], total_model, color = 'black', lw = 3)
ax1.errorbar(wtspec, dtspec, dtspec_err, fmt = '.', ms = 9, color = 'grey')#, alpha = 0.5)

ax1.set_xlim(2,np.max(w)+0.05)
ax1.set_ylim(20750,22850)

ax1.set_ylabel('Total depth (ppm)')

ax2 = fig.add_subplot(gs[150+25:300+25,0])

ax2.plot(models['wavelengths'], evening_model, color = 'orangered', lw = 3)
ax2.plot(models['wavelengths'], morning_model, color = 'cornflowerblue', lw = 3)

fout = open('models.dat', 'w')
for i in range(len(models['wavelengths'])):

    fout.write('{0:.10f} {1:.10f} {2:.10f}'.format(models['wavelengths'][i], evening_model[i], morning_model[i]))

fout.close()
#ax2.plot(models['wavelengths'], cevening_model, '--', color = 'orangered', lw = 3)
#ax2.plot(models['wavelengths'], cmorning_model, '--', color = 'cornflowerblue', lw = 3)

ax2.errorbar(w, evening, evening_error, fmt = '.', ms = 9, color = 'tomato', elinewidth = 2, alpha = 0.4)
ax2.errorbar(w, morning, morning_error, fmt = '.', ms = 9, color = 'cornflowerblue', elinewidth = 2, alpha = 0.4)

#ax2.plot(w, bemodel, 's', color = 'tomato', mec = 'white', ms = 8, zorder = 2)
#ax2.plot(w, bmmodel, 's', color = 'cornflowerblue', mec = 'white', ms = 8, zorder = 2)

ax2.errorbar(w3, evening3, evening_error3, fmt = 'o', color = 'tomato', ecolor = 'tomato', ms = 10, elinewidth = 3)#, label = 'Evening')
ax2.errorbar(w3, morning3, morning_error3, fmt = 'o', color = 'cornflowerblue', ecolor = 'cornflowerblue', ms = 10, elinewidth = 3)#, label = 'Morning')

ax2.set_ylabel('Limb depth (ppm)')
ax2.set_xlim(2,np.max(w)+0.05)
ax2.set_ylim(9500,12300)

ax3 = fig.add_subplot(gs[325 + 25:420])

ax3.plot([2,5.5], [0., 0], '--', color = 'tomato', lw = 3)
eresiduals = evening - bemodel
ax3.errorbar(w, eresiduals, evening_error, fmt = '.', ms = 9, color = 'tomato', elinewidth = 2, alpha = 0.4)

# Bin evening residuals to w3:
ebin, ebin_err = ibin(w3, w, eresiduals, evening_error)
ax3.errorbar(w3, ebin, ebin_err, fmt = 'o', color = 'tomato', ecolor = 'tomato', ms = 10, elinewidth = 3)

ax3.set_ylabel('O-C (ppm)')
ax3.set_xlim(2,np.max(w)+0.05)
ax3.set_ylim(-800,800)

ax4 = fig.add_subplot(gs[420 + 25:])

ax4.set_ylabel('O-C (ppm)')
ax4.plot([2,5.5], [0., 0], '--', color = 'cornflowerblue', lw = 3)

mresiduals = morning - bmmodel
ax4.errorbar(w, mresiduals, morning_error, fmt = '.', ms = 9, color = 'cornflowerblue', elinewidth = 2, alpha = 0.4)

mbin, mbin_err = ibin(w3, w, mresiduals, morning_error)
ax4.errorbar(w3, mbin, mbin_err, fmt = 'o', color = 'cornflowerblue', ecolor = 'cornflowerblue', ms = 10, elinewidth = 3)


ax4.set_xlabel('Wavelength ($\mu$m)')

ax4.set_ylim(-800,800)
ax4.set_xlim(2,np.max(w)+0.05)

# Deactivate ticks for first three plots:
for ax in [ax1, ax2, ax3]:

    ax.tick_params(labelbottom=False)

plt.savefig('model.pdf',
            dpi=250,#, rasterize=True,
            #transparent=True,
            bbox_inches='tight')
