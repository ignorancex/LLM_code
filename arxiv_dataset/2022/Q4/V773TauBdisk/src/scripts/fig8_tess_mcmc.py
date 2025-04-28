# Dario Gonzalez Picos
# picos@mail.strw.leidenuniv.nl
# Last update June 2022


'''
--- Script to generate figure 8 of the paper ---
Load TESS data, computed MCMC fit and Lomb-Scargle variability model
and display results
'''
import paths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

#data_dir = '../data/'
#out_dir = '../plots/'

plt.style.use(paths.data / 'texstyle') # my tex template

# =============================================================================
#                           functions
# =============================================================================
def svm_flux(time, model):
    nsines = len(model)
    print('Building stellar variability model with {:} sine-terms'.format(nsines))
    sv =  sines(time, model.amplitude, model.period, model.phase)
    return 1. + sv

def sines(time, amplitudes, periods, phases):
    '''
    function that returns the sum of several sinusoids
    '''
    trend = 0
    for amplitude, period, phase in zip(amplitudes, periods, phases):
        sine = amplitude * np.sin(2 * np.pi * time / period + phase)
        trend += sine
    return trend

def wavefunc(x, theta):
    p1, p2, a11, a12, a21, a22, b11, b12, b21, b22, c = theta
    w11 = a11*np.sin(2*np.pi*x/p1) + b11*np.cos(2*np.pi*x/p1)
    w12 = a12*np.sin(2*np.pi*x*2/p1) + b12*np.cos(2*np.pi*x*2/p1)
    w21 = a21*np.sin(2*np.pi*x/p2) + b21*np.cos(2*np.pi*x/p2)
    w22 = a22*np.sin(2*np.pi*x*2/p2) + b22*np.cos(2*np.pi*x*2/p2)
    
    return c + w11 + w12 + w21 + w22

def sample_walkers(x,nsamples,flattened_chain, model=wavefunc):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = model(x,i)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread




# =============================================================================
#                            main script
# =============================================================================
# TESS data: eleanor output with time=(HJD-2450000+7000) and flux=pca_flux (sector=19)   
df = pd.read_csv(paths.data / 'v773tau_eleanor.csv')
df['time'] = df.time + 7000
x, y, yerr = df.time, df.flux, df.flux_err

# MCMC model and parameters
samples = np.load(paths.data / 'TESS_MCMC-samples.npy')
best_fit_model = np.load(paths.data / 'TESS_MCMC-best_fit_model.npy')
theta_max = np.load(paths.data / 'TESS_MCMC-best_parameters.npy')

# Lomb-Scarlge stellar variability model
tess_svm = pd.read_csv(paths.data / 'tess_svmodel.csv')



lsc_model = svm_flux(x, tess_svm)

####
fig = plt.figure(figsize=(7,5))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0)
ax = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
plt.subplots_adjust(wspace=None, hspace=None)
plt.setp(ax.get_xticklabels(), visible=False)
print('Plotting MODEL against DATA...')
ax.errorbar(x,y,yerr, fmt='.', color='brown', alpha=0.6, ms=4., label='TESS', zorder=10)
ax.plot(x, lsc_model,'-', color='blue', alpha=0.5, lw=2., 
        label='LS')

ax.plot(x, best_fit_model,'-', color='green', alpha=0.8, lw=2., 
        label='MCMC')

med_model, spread = sample_walkers(x,500,samples)
ax.fill_between(x,med_model-spread,med_model+spread,color='green',alpha=0.3) #label=r'$1\sigma$ MCMC')
#ax.fill_between(x,med_model-2*spread,med_model+2*spread,color='green',alpha=0.3) #,label=r'$2\sigma$ MCMC')
ax.legend(ncol=3)

# COmpute and plot residuals
lsc_res =  y - lsc_model
mcmc_res = y - best_fit_model


ax1.plot(x, lsc_res, '-', color='blue', label='LombScargle', alpha=0.8)  
ax1.plot(x, mcmc_res, '-', color='green', label='MCMC', alpha=0.8)  

ax1.axhline(y=0, c='k', alpha=0.3)
ax.set(ylabel='Relative flux [-]')

ax1.set(xlabel='HJD-2450000', ylabel='Residuals[-]')

# plt.show()
fig.savefig(paths.figures / 'fig8_tess_stellar_var.pdf', dpi=300, bbox_inches='tight')









