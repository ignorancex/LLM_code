import sys
sys.path.append('../')

from schmactions import schmactions

import astropy.units as u
import pickle

init_pos = [8, 0, 0] * u.kpc
init_vel = [0, -190, 50] * u.km/u.s
init_vel_thin = [0, -190, 10] * u.km/u.s

zoffset = [0, 0, 100] * u.pc
xoffset = [100, 0, 0] * u.pc

voffset = [0, 0, 0] * u.km/u.s

s = schmactions(init_pos, init_vel)
sthin = schmactions(init_pos, init_vel_thin)

pickle.dump(s.res, open('true_res.p', 'wb')) 
pickle.dump(sthin.res, open('true_res_thin.p', 'wb'))

zout = s.compute_actions_offset(zoffset, voffset, cadence=1)
xout = s.compute_actions_offset(xoffset, voffset, cadence=1)
zout_thin = sthin.compute_actions_offset(zoffset, voffset, cadence=1)
xout_thin = sthin.compute_actions_offset(xoffset, voffset, cadence=1)

pickle.dump(zout, open('zout.p', 'wb'), protocol=4)
pickle.dump(xout, open('xout.p', 'wb'), protocol=4)

pickle.dump(zout_thin, open('zout_thin.p', 'wb'), protocol=4)
pickle.dump(xout_thin, open('xout_thin.p', 'wb'), protocol=4)
