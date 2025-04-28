import os, sys, warnings, argparse, h5py, numpy as np, time, itertools, random, shutil, datetime

# tensorflow imports and settings
import tensorflow as tf

# other packages import
from sklearn.neighbors import KNeighborsClassifier

# fastlaue imports
from collections import OrderedDict
from laueotx.utils import logging as utils_logging
from laueotx.utils import io as utils_io
from laueotx.utils import config as utils_config
from laueotx import laue_math, laue_math_tensorised, laue_math_graph
from laueotx.detector import Detector
from laueotx.beamline import Beamline
from laueotx.peaks_data import PeaksData
from laueotx.filenames import *
from laueotx.grain import Grain
from tqdm.auto import trange, tqdm
from laueotx.laue_rotation import Rotation

# verbosity
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def get_detectors_from_config(conf):

    detectors = []
    for d in conf['detectors']:
        det = Detector(detector_type=d['type'],
                       side_length=d['side_length'],
                       position=d['position'],
                       tilt=d['tilt'],
                       tol_position=d['position_err'],
                       tol_tilt=d['tilt_err'],
                       hole_diameter=d['hole_diameter'])
        detectors.append(det)
    
    return detectors

def get_beam_spectrum(conf):

    if 'spectrum' in conf['beam'].keys():
        
        spec = np.loadtxt(conf['beam']['spectrum']).T
        spec = (spec[0], spec[1])

    else: # mock flat spectrum

        fake_lam = tf.constant([0,1e12], dtype=tf.float64)
        fake_spec = tf.constant([1, 1], dtype=tf.float64)
        spec = (fake_lam, fake_spec)

    return spec

    



class polycrystalline_sample():

    def __init__(self, conf, hkl_sign_redundancy=False, precision=64, rotation_type='rp', restrict_fitting_area=False):

        self.conf = conf

        if restrict_fitting_area:
            fitting_area_fraction = conf['solver']['fitting_area']
            for d in self.conf['detectors']:
                d['side_length'] *= fitting_area_fraction
                d['hole_diameter'] *= 1/fitting_area_fraction
            conf['beam']['lambda_min'] /= fitting_area_fraction
            conf['beam']['lambda_max'] *= fitting_area_fraction

        
        self.omegas = utils_config.get_angles(conf['angles'])

        self.dt = get_detectors_from_config(conf)

        self.bl = Beamline(omegas=self.omegas, 
                           detectors=self.dt,
                           lambda_lim=[conf['beam']['lambda_min'], conf['beam']['lambda_max']])

        self.gr = Grain(material=self.conf['sample']['material'], 
                        file_hkl=self.conf['sample']['file_hkl'], 
                        beamline=self.bl)
        
        self.sample_size = self.conf['sample']['size']

        self.max_grain_rot = 0.4

        self.hkl_sign_redundancy = hkl_sign_redundancy

        if precision==32:
            self.precision_np = np.float32
            self.precision_tf = tf.float32
        elif precision==64:
            self.precision_np = np.float64
            self.precision_tf = tf.float64

        self.rotation_type = rotation_type

        self.beam_spectrum = get_beam_spectrum(conf)


    def get_grains(self, n_grains, grain_pos, grain_rot):

        self.n_grains = n_grains
        self.set_tensor_variables(grain_rot, grain_pos)

    def get_grains_voxelized(self, n_voxels, n_grains, grain_pos, grain_rot):

        self.n_grains = n_grains
        max_grain_pos = self.sample_size/2
        voxel_pos, voxel_rot, voxel_grn = voxelize(grain_pos, grain_rot, max_grain_pos, n_voxels)
        self.voxel_g = voxel_grn
        self.voxel_r = voxel_rot
        self.voxel_x = voxel_pos
        self.set_tensor_variables(voxel_rot, voxel_pos)


    def get_random_grains(self, n_grains, rule='latin_hypercube'):

        max_grain_pos = self.sample_size/2
        max_det_pos = self.gr.beamline.detectors.tol_position[0]
        max_det_rot = np.radians(self.gr.beamline.detectors.tol_tilt[0])

        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=12)
        seq = sampler.random(n=n_grains)*2-1

        # import chaospy
        # seq = chaospy.J(*list([chaospy.Uniform(lower=-1, upper=1) for _ in range(12)])).sample(n_grains, rule=rule).T

        rot = seq[:,:3] * self.max_grain_rot
        pos = seq[:,3:6] * max_grain_pos

        return pos, rot

    def random_sample(self, n_grains, rule='latin_hypercube'):

        grain_pos, grain_rot = self.get_random_grains(n_grains=n_grains, rule=rule)
        self.set_tensor_variables(grain_rot, grain_pos)


    def random_sample_voxelized(self, n_voxels, n_grains, rule='latin_hypercube'):

        grain_pos, grain_rot = self.get_random_grains(n_grains=n_grains, rule=rule)
        max_grain_pos = self.sample_size/2
        voxel_pos, voxel_rot, voxel_grn = voxelize(grain_pos, grain_rot, max_grain_pos, n_voxels)
        self.voxel_g = voxel_grn
        self.voxel_r = voxel_rot
        self.voxel_x = voxel_pos
        self.set_tensor_variables(voxel_rot, voxel_pos)

    def get_rotation_matrix(self, rot):

        if self.rotation_type == 'rp':
            U = laue_math.r_to_U(rot).astype(self.precision_np)
        elif self.rotation_type == 'mrp':
            U =  Rotation.from_mrp(rot).as_matrix().astype(self.precision_np)

        return U


    def set_tensor_variables(self, rot, pos, d0=None, dr=None):

        self.grain_rot = rot
        self.grain_pos = pos

        
        # get basic variables
        Gamma =  self.gr.beamline.O.astype(self.precision_np)
        x0 =     pos.astype(self.precision_np)
        B =      self.gr.B.astype(self.precision_np)
        h =      laue_math.remove_hkl_sign_redundancy(self.gr.hkl_planes) if not self.hkl_sign_redundancy else np.array(self.gr.hkl_planes, dtype=self.precision_np)
        h =      np.expand_dims(h, axis=-1).astype(self.precision_np)
        v =      np.moveaxis(np.dot(B, h), 1, 0).astype(self.precision_np)
        dl =     tf.constant(self.gr.beamline.detectors.side_length, dtype=self.precision_tf)
        dh =     tf.constant(self.gr.beamline.detectors.hole_diameter, dtype=self.precision_tf)
        lam =    tf.constant(np.array([self.gr.beamline.lambda_min, self.gr.beamline.lambda_max]), dtype=self.precision_tf)
        U =      self.get_rotation_matrix(rot)

        # get indices
        n_ang = Gamma.shape[0]
        n_hkl = v.shape[0] 
        n_det = 2 
        n_dim = 3
        n_pos = len(pos)

        I = tf.eye(n_dim, dtype=Gamma.dtype)

        # for now, fix the detector params 

        dn = np.tile(np.expand_dims(self.gr.beamline.detectors.normal, 0),   reps=[rot.shape[0],1,1]).astype(self.precision_np)
        d0 = np.tile(np.expand_dims(self.gr.beamline.detectors.position, 0), reps=[rot.shape[0],1,1]).astype(self.precision_np) if d0 is None else d0.astype(self.precision_np)
        dr = np.tile(np.expand_dims(self.gr.beamline.detectors.tilt, 0),     reps=[rot.shape[0],1,1]).astype(self.precision_np) if dr is None else dr.astype(self.precision_np)
        
        self.U     = tf.constant(U)
        self.x0    = tf.constant(x0)
        self.Gamma = tf.constant(Gamma)
        self.v     = tf.constant(v)
        self.dn    = tf.constant(dn)
        self.d0    = tf.constant(d0)
        self.dr    = tf.constant(dr)
        self.dl    = tf.constant(dl)
        self.dh    = tf.constant(dh)
        self.lam   = tf.constant(lam)
        self.I     = tf.constant(I)

    def update_tensor_variables(self, rot, pos):

        U =      self.get_rotation_matrix(rot)
        x0 =     pos.astype(self.precision_np)
        self.x0    = tf.constant(x0)
        self.U     = tf.constant(U)
        
    def set_batch_indices(self):

        n_grn, n_ang, n_hkl, n_det = len(self.U), len(self.Gamma), len(self.v), len(self.dl)
        i_grn, i_ang, i_det = get_batch_indices(n_grn, n_ang, n_hkl, n_det)
        self.i_ang = tf.constant(i_ang)
        self.i_det = tf.constant(i_det)
        self.i_grn = tf.constant(i_grn)

    def get_batch_indices(self, n_grn=1):

        n_ang, n_hkl, n_det = len(self.Gamma), len(self.v), len(self.dl)
        i_grn, i_ang, i_det = get_batch_indices(n_grn, n_ang, n_hkl, n_det)
        i_ang = tf.constant(i_ang)
        i_det = tf.constant(i_det)
        i_grn = tf.constant(i_grn)

        return i_grn, i_ang, i_det

    def get_batch_indices_full(self, n_grn=1):
        
        n_ang, n_hkl, n_det = len(self.Gamma), len(self.v), len(self.dl)   
        i_grn, i_ang, i_hkl, i_det, i_all = get_batch_indices_full(n_grn, n_ang, n_hkl, n_det)
        i_grn = tf.expand_dims(i_grn, axis=-1)
        i_ang = tf.expand_dims(i_ang, axis=-1)
        i_hkl = tf.expand_dims(i_hkl, axis=-1)
        i_det = tf.expand_dims(i_det, axis=-1)
        i_all = tf.expand_dims(i_all, axis=-1)
        return i_grn, i_ang, i_hkl, i_det, i_all

    def get_spots(self):

        # import ipdb; ipdb.set_trace(); 
        # pass

        # def fast_full_forward(U, x0, Gamma, v, dn, d0, dr, I):
        s_spot, p_spot, p_lam = laue_math_tensorised.fast_full_forward(self.U, self.x0, self.Gamma, self.v, self.dn, self.d0, self.dr, self.I)

        # def fast_spot_select(p_spot, s_spot, p_lam, lam, dn, dl, dh):
        select = laue_math_tensorised.fast_spot_select(p_spot, s_spot, p_lam, self.lam, self.dn, self.dl, self.dh)

        return s_spot, p_spot, p_lam, select

    def get_spots_batch(self, n_per_batch=100, reference_frame='detector'):

        if reference_frame=='laboratory':

            s_spot, p_spot, p_lam, select = laue_math_graph.batch_spots_lab(n_per_batch, self.U, self.x0, self.Gamma, self.v, self.dn, self.d0, self.dr, self.dl, self.dh, self.lam, self.I)

        elif reference_frame=='detector':

            s_spot, p_spot, p_lam, select = laue_math_graph.batch_spots(n_per_batch, self.U, self.x0, self.Gamma, self.v, self.dn, self.d0, self.dr, self.dl, self.dh, self.lam, self.I)

        p_lam = tf.tile(tf.expand_dims(p_lam, axis=3), multiples=[1,1,1,2])

        return s_spot, p_spot, p_lam, select

    def get_image_batch(self, pixel_size, angle_size, n_pix, n_ang, n_det, n_per_batch=100):

        n_per_batch_max = min(self.U.shape[0], n_per_batch)

        img = np.zeros((n_ang, n_det, n_pix, n_pix), dtype=self.precision_np)

        # get indices for batch (for spot selection later)
        i_grn, i_ang, i_det = self.get_batch_indices(n_grn=n_per_batch_max)
        i_ang = tf.cast(i_ang, tf.float32)

        # render in batch mode
        img = laue_math_graph.render_image_cube_batch(img, n_per_batch_max, pixel_size, angle_size, i_ang, i_det, self.beam_spectrum, self.U, self.x0, self.Gamma, self.v, self.dn, self.d0, self.dr, self.dl, self.dh, self.lam, self.I) 

        return img

    # def get_segmentation_image_batch(self, pixel_size, angle_size, n_pix, n_ang, n_det, n_per_batch=100):

    #     img = np.zeros((n_ang, n_det, n_pix, n_pix), dtype=self.precision_np)

    #     # get indices for batch (for spot selection later)
    #     i_grn, i_ang, i_det = self.get_batch_indices(n_grn=1)
    #     i_ang = tf.cast(i_ang, tf.float32)

    #     # render in batch mode
    #     # img = laue_math_graph.render_image_cube_batch(img, n_per_batch_max, pixel_size, angle_size, i_ang, i_det, self.U, self.x0, self.Gamma, self.v, self.dn, self.d0, self.dr, self.dl, self.dh, self.lam, self.I) 
    #     img = laue_math_graph.render_segmentation_image_cube_batch(img, pixel_size, angle_size, i_ang, i_det, i_grn, self.U, self.x0, self.Gamma, self.v, self.dn, self.d0, self.dr, self.dl, self.dh, self.lam, self.I) 

    #     return img
        

def get_batch_indices_full(n_grn, n_ang, n_hkl, n_det, variable_dtype=True):

    i_grn, i_ang, i_hkl, i_det = tf.meshgrid(np.arange(n_grn, dtype=np.int32), 
                                             np.arange(n_ang, dtype=np.int32), 
                                             np.arange(n_hkl, dtype=np.int32), 
                                             np.arange(n_det, dtype=np.int32), 
                                             indexing='ij')

    i_grn = tf.cast(tf.constant(i_grn), dtype=tf.int32)
    i_ang = tf.cast(tf.constant(i_ang), dtype=tf.int32)
    i_hkl = tf.cast(tf.constant(i_hkl), dtype=tf.int32)
    i_det = tf.cast(tf.constant(i_det), dtype=tf.int32)
    i_all = tf.reshape(tf.constant(np.arange(len(i_grn.numpy().ravel())), dtype=tf.int32), shape=i_grn.shape)

    return i_grn, i_ang, i_hkl, i_det, i_all


def get_batch_indices(n_grn, n_ang, n_hkl, n_det):

    i_grn, i_ang, i_det = tf.meshgrid(np.arange(n_grn, dtype=np.int32), np.arange(n_ang, dtype=np.int32), np.arange(n_det, dtype=np.int32))
    i_grn = tf.transpose(i_grn, perm=[1, 0, 2])
    i_ang = tf.transpose(i_ang, perm=[1, 0, 2])
    i_det = tf.transpose(i_det, perm=[1, 0, 2])
    i_grn = tf.reshape(i_grn, shape=(n_grn, n_ang, 1, n_det, 1))
    i_ang = tf.reshape(i_ang, shape=(n_grn, n_ang, 1, n_det, 1))
    i_det = tf.reshape(i_det, shape=(n_grn, n_ang, 1, n_det, 1))
    i_grn = tf.tile(i_grn, multiples=(1, 1, n_hkl, 1, 1))
    i_ang = tf.tile(i_ang, multiples=(1, 1, n_hkl, 1, 1))
    i_det = tf.tile(i_det, multiples=(1, 1, n_hkl, 1, 1))

    return i_grn, i_ang, i_det

def voxelize(grain_pos, grain_rot, max_grain_pos, n, g_voxels=None, method='voronoi'):

    # create voxel grid
    x_side = np.linspace(-max_grain_pos, max_grain_pos, n)
    x0, x1, x2 = np.meshgrid(x_side, x_side, x_side)
    x_voxels = np.concatenate([x0.reshape(-1,1), x1.reshape(-1,1), x2.reshape(-1,1)], axis=-1)

    if g_voxels == None:

        if method == 'voronoi':
            
            # classify voxels to grains
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(grain_pos, y=np.arange(len(grain_pos)))
            g_voxels = knn.predict(x_voxels)


        else:

            raise Exception(f'voxelization method {method} unknown')
    
    # assign rotations
    r_voxels = grain_rot[g_voxels]

    return x_voxels, r_voxels, g_voxels

        
# @profile
def merge_duplicated_spots(s, i_grn, i_ang, i_hkl, i_det, i_all, i_ray=None, p_lam=None, decimals=4, return_index=False, split_by_grain=True, verb=True):

    sr = np.round(s.numpy(), decimals)

    if i_ray is not None:
        
        i_hash = (1e8*i_ang[:,0].numpy()).astype(np.uint64) + (1e4*i_det[:,0].numpy()).astype(np.uint64) + (i_ray[:,0].numpy()).astype(np.uint64)
        
        if split_by_grain:
            i_hash += (1e12*i_grn[:,0].numpy()).astype(np.uint64)                    

        uv, ui = np.unique(i_hash, return_index=True)

    else:

        if split_by_grain:
            s_spot_concat = np.concatenate([i_grn.numpy(), i_ang.numpy(), i_det.numpy(), s.numpy()], axis=1)
        else:
            s_spot_concat = np.concatenate([i_ang.numpy(), i_det.numpy(), s.numpy()], axis=1)

        s_spot_concat = np.round(s_spot_concat, decimals)
        uv, ui = np.unique(s_spot_concat, axis=0, return_index=True)

    s_ = np.array(s)[ui]
    i_grn_ = np.array(i_grn)[ui]
    i_ang_ = np.array(i_ang)[ui]
    i_hkl_ = np.array(i_hkl)[ui]
    i_det_ = np.array(i_det)[ui]
    i_all_ = np.array(i_all)[ui]

    if verb:        
        LOGGER.info(f'merged duplicated spots {len(s)} -> {len(s_)} (split={split_by_grain})')

    if return_index:
        return tf.constant(s_), tf.constant(i_grn_), tf.constant(i_ang_), tf.constant(i_hkl_), tf.constant(i_det_), tf.constant(i_all_), ui
    else:
        return tf.constant(s_), tf.constant(i_grn_), tf.constant(i_ang_), tf.constant(i_hkl_), tf.constant(i_det_), tf.constant(i_all_)


def select_indices(select, s_spot, i_grn, i_ang, i_hkl, i_det, i_all):

    return s_spot[select], i_grn[select], i_ang[select], i_hkl[select], i_det[select], i_all[select]




def get_sobol_grid_laue(n_grains, max_grain_pos=5./2., laue_group='c', sig_rot=0.01, sig_pos=0.02, index=0, seed=12312312):

    from fastlaue3dnd.laue_rotation import get_rotation_constraint_rf, Rotation

    if laue_group.lower() == 'c':

        max_grain_rot = np.max(np.abs(Rotation.from_rodrigues_frank([0.414, 0, 0]).as_mrp()))

    else: 

        raise Exception(f'laue group {laue_group_} not implemented')

    if n_grains>1e7:
        LOGGER.warning(f'generating large number of sobol samples {n_grains}, this may cause memory problems')
    

    # oversample to pass the later selection
    sobol_samples = tf.math.sobol_sample(dim=6, num_results=n_grains*2, skip=index*n_grains, dtype=tf.float64).numpy()*2 -1 # numbers between -1 and 1
    grain_rot = sobol_samples[:,0:3] * max_grain_rot
    grain_pos = sobol_samples[:,3:6] * max_grain_pos

    # add noise
    np.random.seed(seed*index)
    if sig_rot>0:
        grain_rot += np.random.normal(size=grain_rot.shape, scale=sig_rot)
    if sig_pos>0:
        grain_pos += np.random.normal(size=grain_pos.shape, scale=sig_pos)

    # apply the constraints
    grain_rot_tf = Rotation.as_rodrigues_frank(Rotation.from_mrp(grain_rot))
    select = get_rotation_constraint_rf(grain_rot_tf)
    grain_rot_tf = grain_rot_tf[select][:n_grains]
    grain_rot = Rotation.from_rodrigues_frank(grain_rot_tf).as_mrp()
    grain_pos = grain_pos[select][:n_grains]

    # clip so that all grains are within the parameters
    grain_rot = np.clip(grain_rot, -max_grain_rot, max_grain_rot)
    grain_pos = np.clip(grain_pos, -max_grain_pos, max_grain_pos)

    return grain_rot, grain_pos

def apply_selections(select, *args):

    if len(args)==0:
        raise Exception('must provide at least one array to select')

    list_out = []
    for a in args:
        list_out += [a[select]]

    return list_out



