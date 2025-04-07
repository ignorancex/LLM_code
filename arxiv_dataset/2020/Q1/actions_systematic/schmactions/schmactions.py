import numpy as np

import astropy.units as u

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp

from tqdm import tqdm
import warnings

from joblib import Parallel, delayed

class schmactions(object):
    def __init__(self, init_pos, init_vel, mw=None,
                 t1=0*u.Gyr, t2=5*u.Gyr, dt=1*u.Myr, N_max=8, save_orbit=False):
        
        if mw is None:
            self.mw = gp.MilkyWayPotential()
        else:
            self.mw = mw

        self.t1 = t1
        self.t2 = t2
        self.dt = dt
        self.N_max = 8
        
        self.q = gd.PhaseSpacePosition(pos=init_pos, vel=init_vel)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            orbit = self.mw.integrate_orbit(self.q, dt=self.dt,
                                                 t1=self.t1, t2=self.t2)
            self.res = gd.find_actions(orbit, N_max=self.N_max)
        
        self.zmax = orbit.zmax()

        if save_orbit:
            #optional since orbit breaks pickling
            self.orbit = orbit

    def compute_actions_offset(self, pos_offset, vel_offset, 
                               cadence=10, end=1000):
        orbit = self.mw.integrate_orbit(self.q, dt=self.dt,
                                        t1=self.t1, t2=self.t2)
        pos = np.transpose(orbit.pos.xyz)
        pos = np.add(pos, pos_offset)
        vel = np.transpose(orbit.vel.d_xyz)
        vel = np.add(vel, vel_offset)

        pos = np.transpose(pos)
        vel = np.transpose(vel)

        all_q = gd.PhaseSpacePosition(pos=pos, vel=vel)
        out = []
        time = orbit.t
        with warnings.catch_warnings(record=True):
            for q,t in tqdm(zip(all_q[0:end:cadence],time[0:end:cadence])):
                try:
                    orbit = self.mw.integrate_orbit(q, dt=self.dt,
                                                t1=self.t1, t2=self.t2)
                    res = gd.find_actions(orbit, N_max=self.N_max)
                    res['time'] = t
                    out.append(res)
                except:
                    np.nan
        return out

    def extract_actions(self, out, units=u.kpc * u.km/u.s):
        return np.array([r['actions'].to_value(units) for r in out])

    def extract_time(self, out, units=u.Myr):
        return np.array([r['time'].to_value(units) for r in out])

def _helper_compute_offsets_(s, poff, voff, cd=10, end=1000):
    return s.compute_actions_offset(poff, voff, cd, end)

def compute_offset_list(s, offset_list, nproc=1, vel=False,
                        cadence=10, end=1000, fname=None):
    # s is a schmactions object
    # vel means the offset list is a vel offset
    if vel:
        aux = [0, 0, 0] * u.kpc
    else:
        aux = [0, 0, 0] * u.km/u.s

    # act_result = []
    # for off in tqdm(offset_list):
    #     if vel:
    #         out = s.compute_actions_offset(aux, off, cadence, end)
    #     else:
    #         out = s.compute_actions_offset(off, aux, cadence, end)
    #     act = s.extract_actions(out)
    #     act_result.append(act)
    # act_result = np.array(act_result)

    if vel:
        act_result = Parallel(n_jobs=nproc) (delayed(_helper_compute_offsets_)(s, aux, off, cadence, end) for off in tqdm(offset_list))
    else:
        act_result = Parallel(n_jobs=nproc) (delayed(_helper_compute_offsets_)(s, off, aux, cadence, end) for off in tqdm(offset_list))


    out = {'act_result': act_result,
           'offset_list': offset_list,
           'vel': vel}
    return out


if __name__ == '__main__':
    s = schmactions([8, 0, 0] * u.kpc, [0, -190, 50] * u.km/u.s)
    # out = s.compute_actions_offset([0, 0, 100] * u.pc, [0, 0, 0] * u.km/u.s)
