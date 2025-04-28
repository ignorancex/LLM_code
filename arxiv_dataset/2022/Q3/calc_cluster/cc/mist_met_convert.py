# interpolate magnitudes from a PARS grid with many metallicities 
# to a grid corresponding to a single MIST metallicity
from lib import mist_util as mu
import config as cf
from pa.lib import util as ut

import time, pickle
import numpy as np
from scipy.interpolate import interp1d

iodir = '../data/'

## this code is for interpolating at one specific MIST metallicity value on the PARS grid
print("solar metallicity is " + str(ut.Zsun))
# Load the original PARS grid
print('Loading PARS...')
start = time.time()
with open(iodir + 'pars_grid.pkl', 'rb') as f: 
	pars = pickle.load(f)
print('\t' + str(time.time() - start) + ' seconds.\n')

# construct the PARS metallicity array from the one MIST metallicity
logZp = np.array([ ut.logZp_from_logZm(cf.Z) ])

# interpolate
print('Interpolating on the metallicity grid...')
start = time.time()
sh = list(pars.Mag.shape)
sh[4] = len(logZp) # change the metallicity dimension in the shape of the PARS grid
Mag2 = np.empty(sh)
for i in range(len(pars.omega)):
	mag = np.take(pars.Mag, i, axis=1)
	f = interp1d(pars.Z, mag, kind='linear', axis=3)
	Mag2[:, i, ...] = f(logZp)
pars.Mag = Mag2
pars.Z = logZp
print('\t' + str(time.time() - start) + ' seconds.\n')

### Pickle the grid
with open(iodir + 'pars_grid_ZM' + str(cf.Z).replace('-', 'm').replace('.', 'p') + '.pkl', 'wb') as f:
	pickle.dump(pars, f)