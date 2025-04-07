import gizmo_analysis as gizmo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from astropy.constants import G as G_astropy
import astropy.units as u

import sys
from joblib import Parallel, delayed

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

nbootstrap = 1000
np.random.seed(162)

rcut = 0.5
zcut = 1.0
nspoke = 50

Rsolar = 8.2
Rmin = 7.2
Rmax = 9.2
dR = 0.1

nproc = int(sys.argv[1])

def read_snap(gal):
    # takes in galaxy (string = m12i, m12f, m12m, etc.)
    # reads in and sets fiducial coordinates
    # returns snap
    gal_info = 'fiducial_coord/' + gal + '_res7100_center.txt'
    root = '/mnt/ceph/users/firesims/fire2/metaldiff/'
    #root = '/Users/abeane/scratch/actions_systematic/data/fire2/metaldiff/'
    sim_directory = root + gal + '_res7100'
    snap = gizmo.io.Read.read_snapshots(['star'], 'index', 600,
                                        properties=['position', 'id',
                                                    'mass', 'velocity'],
                                        assign_center=False,
                                        simulation_directory=sim_directory)
    ref = np.genfromtxt(gal_info, comments='#', delimiter=',')
    snap.center_position = ref[0]
    snap.center_velocity = ref[1]
    snap.principal_axes_vectors = ref[2:]
    for k in snap.keys():
        for attr in ['center_position', 'center_velocity', 
                     'principal_axes_vectors']:
            setattr(snap[k], attr, getattr(snap, attr))

    return snap

def gen_pos():
    theta = np.linspace(0, 2.*np.pi, nspoke)

    posx = Rsolar * np.cos(theta)
    posy = Rsolar * np.sin(theta)
    posz = np.zeros(len(posx))
    pos = np.transpose([posx, posy, posz])
    return theta, pos

def get_midplane_with_error(pos, star_pos, star_vel):
    
    def get_init_keys(p, star_pos):
        pos_diff = np.subtract(star_pos, p)
        rmag = np.linalg.norm(pos_diff[:,:2], axis=1)
        rbool = rmag < rcut
        zbool = np.abs(pos_diff[:,2]) < 2.0 * zcut
        keys = np.where(np.logical_and(rbool, zbool))[0]
        return keys

    def get_keys(p, part):
        pos_diff = np.subtract(part, p)
        rmag = np.linalg.norm(pos_diff[:,:2], axis=1)
        rbool = rmag < rcut
        zbool = np.abs(pos_diff[:,2]) < zcut
        keys = np.where(np.logical_and(rbool, zbool))[0]
        return keys
    
    def midplane(pos, init_pos, init_vel):
        mid_pos = pos.copy()
        for _ in range(10):
            keys = get_keys(mid_pos, init_pos)
            mid_pos[2] = np.median(init_pos[:,2][keys])
        mid_vel = np.median(init_vel[:,2][keys])
        return mid_pos[2], mid_vel

    # get all particles within 2x zheight
    init_keys = get_init_keys(pos, star_pos)
    init_pos = star_pos[init_keys]
    init_vel = star_vel[init_keys]

    # calculate midplane using all particles
    midplane_central, midplane_vel = midplane(pos, init_pos, init_vel)
    
    # prepare to bootstrap
    keys_to_choose = list(range(len(init_pos)))
    rand_choice = np.random.choice(keys_to_choose, len(init_pos)*nbootstrap)
    rand_choice = np.reshape(rand_choice, (nbootstrap, len(init_pos)))
    init_pos_rand = init_pos[rand_choice]
    init_vel_rand = init_vel[rand_choice]
    med_rand = np.array([ midplane(pos, ipos, ivel) for ipos,ivel in zip(init_pos_rand,init_vel_rand) ])
    dist_pos = np.subtract(med_rand[:,0], midplane_central)
    dist_vel = np.subtract(med_rand[:,1], midplane_vel)
    up_pos = np.percentile(dist_pos, 50+68/2)
    low_pos = np.percentile(dist_pos, 50-68/2)
    up_vel = np.percentile(dist_vel, 50+68/2)
    low_vel = np.percentile(dist_vel, 50-68/2)
    l = midplane_central - up_pos
    h = midplane_central - low_pos
    l_v = midplane_vel - up_vel
    h_v = midplane_vel - low_vel
    
    return midplane_central, l, h, midplane_vel, l_v, h_v

def fit(x, theta):
    return x[0]*np.cos(theta+x[1]) + x[2]

def chisq(x, theta, midplane_est):
    return np.sum(np.square(np.subtract(fit(x, theta), midplane_est)))

def main(gal):
    snap = read_snap(gal)
    
    star_pos = snap['star'].prop('host.distance.principal')
    star_vel = snap['star'].prop('host.velocity.principal')
    
    theta, pos = gen_pos()

    result = Parallel(n_jobs=nproc) (delayed(get_midplane_with_error)(p, star_pos, star_vel) for p in tqdm(pos))
    result = np.array(result)

    midplane_est = result[:,0]

    res = minimize(chisq, np.array([0.1, 0, 0]), args=(theta, midplane_est), method='Nelder-Mead')
    A = res.x[0]
    B = res.x[1]
    C = res.x[2]
    print('A=', A, 'B=', B, 'C=', C)
    fit = A*np.cos(theta + B) + C

    out = np.concatenate((theta.reshape(nspoke, 1), result, fit.reshape(nspoke,1)), axis=1)

    np.save('output/out_'+gal+'.npy', out)

if __name__ == '__main__':
    # glist = ['m12i', 'm12f', 'm12m']
    glist = ['m12m']
    for gal in glist:
        main(gal)

