import sys
sys.path.append('../')
import numpy as np

from schmactions import schmactions
from schmactions import compute_offset_list

import astropy.units as u
import pickle

from joblib import Parallel, delayed
from tqdm import tqdm

nproc = 40

init_pos = [8, 0, 0] * u.kpc
init_vel_list = [ [0, -190, v] * u.km/u.s for v in np.linspace(0, 150, 100) ]

zoffset_list = [[0, 0, 10] * u.pc,
                [0, 0, 50] * u.pc,
                [0, 0, 100] * u.pc]
voffset = [0, 0, 0] * u.km/u.s

names = ['10', '50', '100']

def _helper_(init_vel, zoffset):
    try:
        s = schmactions(init_pos, init_vel)
        true_actions = s.res['actions'].to_value(u.kpc * u.km/u.s)
        zmax = s.zmax.to_value(u.kpc)
    
        r = s.compute_actions_offset(zoffset, voffset)
        act = s.extract_actions(r)
    except:
        return [np.nan, np.nan, np.nan]

    return [true_actions, act, zmax]

for zoffset, n in zip(zoffset_list, names):
    out = Parallel(n_jobs=nproc) (delayed(_helper_)(init_vel, zoffset) for init_vel in tqdm(init_vel_list))

    pickle.dump(out, open('dJz_fun_of_Jz_'+n+'pc.p', 'wb'))
