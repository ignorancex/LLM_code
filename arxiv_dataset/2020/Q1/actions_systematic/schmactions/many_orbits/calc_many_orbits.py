import sys
sys.path.append('../')
import numpy as np

from schmactions import schmactions
from schmactions import compute_offset_list

import astropy.units as u
import pickle

nproc = 40
init_pos = [8, 0, 0] * u.kpc
init_vel_list = [[0, -190, 10] * u.km/u.s,
                 [0, -190, 50] * u.km/u.s,
                 [0, -190, 190] * u.km/u.s]

name_list = ['thin', 'thick', 'halo']

zoffset_list = [[0, 0, z] * u.pc for z in np.arange(0, 500, 10)]
xoffset_list = [[x, 0, 0] * u.pc for x in np.arange(0, 500, 10)]

voffset = [0, 0, 0] * u.km/u.s

for init_vel, name in zip(init_vel_list, name_list):
    s = schmactions(init_pos, init_vel)

    pickle.dump(s.res, open('true_res_'+name+'.p', 'wb')) 

    zout = compute_offset_list(s, zoffset_list, nproc=nproc)
    xout = compute_offset_list(s, xoffset_list, nproc=nproc)

    pickle.dump(zout, open('zout_'+name+'.p', 'wb'), protocol=4)
    pickle.dump(xout, open('xout_'+name+'.p', 'wb'), protocol=4)
