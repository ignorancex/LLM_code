import os, sys, warnings, argparse, h5py, numpy as np, time, itertools, random, shutil, datetime

# tensorflow imports and settings
import tensorflow as tf

from collections import OrderedDict
from laueotx.utils import logging as utils_logging
# from laueotx.utils import io as utils_io
# from laueotx.utils import config as utils_config
# from laueotx import laue_math, laue_math_tensorised, rodrigues_space
# from laueotx.detector import Detector
# from laueotx.beamline import Beamline
# from laueotx.peaks_data import PeaksData
# from laueotx.filenames import *
# from laueotx.grain import Grain
# from laueotx.laue_math_tensorised import fast_full_forward, fast_spot_select
from laueotx.config import TF_FUNCTION_JIT_COMPILE

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)

LOGGER = utils_logging.get_logger(__file__)



def get_spot_hist(angles, detectors, s_sample, i_ang, i_det, det_side_length, kernel_n_pix):

    img_cube = render_image_cube(s_spot=s_sample,
                             i_ang=i_ang,
                             i_det=i_det,
                             n_ang=len(angles),
                             n_det=len(detectors),
                             n_pix=kernel_n_pix,
                             pixel_size=det_side_length/kernel_n_pix)

    img_cube = img_cube.numpy()
    return img_cube


@tf.function
def render_image_cube(s_spot, i_ang, i_det, n_ang, n_det, n_pix, pixel_size):

    img_cube = np.zeros((n_ang, n_det, n_pix, n_pix), dtype=np.float32)
    
    indices = s_spot[:,1:]/pixel_size
    indices = tf.cast(indices, tf.int32) + n_pix//2
    i_ang = tf.cast(i_ang, tf.int32)
    i_det = tf.cast(i_det, tf.int32)
    indices = tf.concat([i_ang, i_det, indices], axis=-1)
    updates = tf.ones(indices.shape[0], dtype=tf.float32)
    img_cube = tf.tensor_scatter_nd_add(img_cube, indices, updates)

    return img_cube


@tf.function
def nn_distance(nn_lookup_pos, s_spot, i_ang, i_det, i_grn, pixel_size, n_pix):

    s_nn = nn_lookup(nn_lookup_pos, s_spot, i_ang, i_det, i_grn, pixel_size, n_pix)
    dist = tf.reduce_sum( (s_nn - s_spot[:,1:])**2, axis=1)

    return tf.math.segment_mean(dist, segment_ids=i_grn[:,0])

@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def nn_lookup_all(nn_lookup_arr, s_target, s_spot, i_ang, i_det, i_grn, pixel_size, n_pix):

    i_ang = tf.cast(i_ang, tf.int64)
    i_det = tf.cast(i_det, tf.int64)

    indices = s_spot[:,1:]/pixel_size

    indices = tf.cast(indices, tf.int64) + tf.cast(n_pix, tf.int64)//2

    indices = tf.concat([i_ang, i_det, indices], axis=-1)
    i_nn = tf.gather_nd(nn_lookup_arr, indices)

    return i_nn


@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def nn_lookup(nn_lookup_arr, s_target, s_spot, i_ang, i_det, i_grn, pixel_size, n_pix):

    i_ang = tf.cast(i_ang, tf.int64)
    i_det = tf.cast(i_det, tf.int64)

    indices = s_spot[:,1:]/pixel_size
    indices = tf.cast(indices, tf.int64) + tf.cast(n_pix, tf.int64)//2
    indices = tf.concat([i_ang, i_det, indices], axis=-1)

    i_nn = tf.gather_nd(nn_lookup_arr, indices)
    s_nn = tf.gather(s_target[:,1:], i_nn)
    diff = tf.reduce_sum((s_nn - tf.expand_dims(s_spot[:,1:], axis=1))**2, axis=-1)
    min_ind = tf.math.argmin(diff, axis=1)
    nn = tf.gather(i_nn, min_ind, batch_dims=1)

    s_match = tf.gather(s_target, nn)
    diff = s_spot - s_match

    return nn

@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def nn_lookup_dist(nn_lookup_arr, s_target, s_spot, i_ang, i_det, i_grn, pixel_size, n_pix):

    i_ang = tf.cast(i_ang, tf.int64)
    i_det = tf.cast(i_det, tf.int64)

    indices = s_spot[:,1:]/pixel_size
    indices = tf.cast(indices, tf.int64) + tf.cast(n_pix, tf.int64)//2
    indices = tf.concat([i_ang, i_det, indices], axis=-1)

    i_nn = tf.gather_nd(nn_lookup_arr, indices)
    s_nn = tf.gather(s_target[:,1:], i_nn)
    diff = tf.reduce_sum((s_nn - tf.expand_dims(s_spot[:,1:], axis=1))**2, axis=-1)
    diff_min = tf.math.reduce_min(diff, axis=1)
    
    return diff_min

@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def nn_lookup_ind_dist(nn_lookup_arr, s_target, s_spot, i_ang, i_det, i_grn, pixel_size, n_pix):

    i_ang = tf.cast(i_ang, tf.int64)
    i_det = tf.cast(i_det, tf.int64)

    indices = s_spot[:,1:]/pixel_size
    indices = tf.cast(indices, tf.int64) + tf.cast(n_pix, tf.int64)//2
    indices = tf.concat([i_ang, i_det, indices], axis=-1)

    i_nn = tf.gather_nd(nn_lookup_arr, indices)
    s_nn = tf.gather(s_target[:,1:], i_nn)
    diff = tf.reduce_sum((s_nn - tf.expand_dims(s_spot[:,1:], axis=1))**2, axis=-1)
    min_ind = tf.math.argmin(diff, axis=1)
    nn = tf.gather(i_nn, min_ind, batch_dims=1)
    diff_min = tf.math.reduce_min(diff, axis=1)

    return diff_min, nn
    


@tf.function
def batch_distance(n_per_batch, nn_lookup_pos, i_grn, i_ang, i_det, detector_side_length, lookup_n_pix, U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I):

    dataset = tf.data.Dataset.from_tensor_slices((U, x0, dn, d0, dr))
    dataset = dataset.batch(batch_size=n_per_batch).prefetch(buffer_size=2)
    dataset = iter(dataset)

    assert U.shape[0] % n_per_batch == 0, 'number of grid points must be a multiple of a n points per batch'

    n_batches = int(np.ceil(U.shape[0]/n_per_batch))


    dist = []

    for i in range(n_batches):

        U_, x0_, dn_, d0_, dr_ = next(dataset)

        s_spot, p_spot, p_lam = fast_full_forward(U_, x0_, Gamma, v, dn_, d0_, dr_, I)
        select = fast_spot_select(p_spot, s_spot, p_lam, lam, dn_, dl, dh)
        try:
            s_spot_sample = s_spot[select]
            i_grn_sample = i_grn[select]
            i_ang_sample = i_ang[select]
            i_det_sample = i_det[select]
        except Exception as err:
            import ipdb; ipdb.set_trace(); 
            pass


        d_ = nn_distance(nn_lookup_pos, s_spot_sample, i_ang_sample, i_det_sample, i_grn_sample, pixel_size=detector_side_length/lookup_n_pix, n_pix=lookup_n_pix)
        dist.append(d_)
    
    dist = tf.concat(dist, axis=0)
    # dist = tf.clip_by_value(dist, 0, 1)

    return dist

class spot_neighbor_lookup():

    def __init__(self, conf, s_spot, i_ang, i_det, lookup_n_pix, detector_side_length, n_neighbours=2, compute_pos_lookup=False, dtype=tf.uint32):

        self.conf = conf
        self.s_spot = s_spot
        self.i_spot = np.arange(len(self.s_spot))
        self.i_ang = i_ang
        self.i_det = i_det
        self.angles = np.arange(np.max(i_ang)+1)
        self.detectors = np.arange(np.max(i_det)+1)
        self.lookup_n_pix = lookup_n_pix
        self.detector_side_length = detector_side_length
        self.n_dim = 2
            
        # pre-compute lookup tables
        self.n_neighbours = n_neighbours
        self.dtype = dtype
        self.compute_nearest_neighbour_lookup()

    def compute_nearest_neighbour_lookup(self):

        pix_size = self.detector_side_length/self.lookup_n_pix
        grid_x = np.linspace(-self.detector_side_length/2, self.detector_side_length/2, self.lookup_n_pix)
        x, y = np.meshgrid(grid_x, grid_x)
        X = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
        self.s_ind = np.arange(len(self.s_spot), dtype=np.int32)

        nn_lookup_ind = np.zeros((len(self.angles), len(self.detectors), self.lookup_n_pix, self.lookup_n_pix, self.n_neighbours), dtype=np.int32)
        min_k = self.n_neighbours
        from sklearn.neighbors import KDTree
        for ai in LOGGER.progressbar(self.angles, desc=f'creating spot lookup table with n_pix={self.lookup_n_pix} n_neighbours_max={self.n_neighbours}', at_level='info'):
            for di in self.detectors:

                select = (self.i_ang == ai) & (self.i_det == di)
                s_spot = np.array(self.s_spot[select[:,0]])
                s_ind = self.s_ind[select[:,0]]
                LOGGER.debug(f'ai={ai} di={di} n_spots={len(s_spot)}')

                if len(s_spot)>1:
                    tree = KDTree(s_spot[:,1:])

                    k = min(self.n_neighbours, len(s_spot))
                    dist, inds = tree.query(X, k=k)

                    ind = np.reshape(s_ind[inds], (self.lookup_n_pix, self.lookup_n_pix, k))
                    nn_lookup_ind[ai, di][:,:,:k] = ind

                    min_k = min(min_k, k)

        # reshape tensors
        nn_lookup_ind = tf.constant(nn_lookup_ind, dtype=self.dtype)
        nn_lookup_ind = tf.transpose(nn_lookup_ind, perm=[0, 1, 3, 2, 4])
        self.nn_lookup_ind = nn_lookup_ind[...,:min_k]

        LOGGER.info(f'created nearest neighbor lookup n_spots={len(self.s_spot)} nn_lookup_ind={self.nn_lookup_ind.shape} k_nn={min_k}')

    def distance_params(self, r, x):

        from laueotx.polycrystalline_sample import polycrystalline_sample
        sample = polycrystalline_sample(self.conf)
        sample.set_tensor_variables(rot=r, pos=x)
        i_grn, i_ang, i_det = sample.get_batch_indices(n_grn=len(r))
        s_spot, p_spot, p_lam, select = sample.get_spots_batch()
        s_spot_sample = s_spot[select]
        i_grn_sample = i_grn[select]
        i_ang_sample = i_ang[select]
        i_det_sample = i_det[select]

        return self.distance_spots(s_spot_sample, i_ang_sample, i_det_sample, i_grn_sample)

    def distance_params_batch(self, r, x, n_per_batch=1000):

        n_per_batch = min(n_per_batch, len(r))

        from laueotx.polycrystalline_sample import polycrystalline_sample
        sample = polycrystalline_sample(self.conf, precision=32)
        sample.set_tensor_variables(rot=r, pos=x)

        i_grn, i_ang, i_det = sample.get_batch_indices(n_grn=n_per_batch)

        dist = batch_distance(n_per_batch, self.nn_lookup_pos, i_grn, i_ang, i_det, self.detector_side_length, self.lookup_n_pix, sample.U, sample.x0, sample.Gamma, sample.v, sample.dn, sample.d0, sample.dr, sample.dl, sample.dh, sample.lam, sample.I)
        return dist

    def get_nn_position(self, s_det, i_ang, i_det, i_grn):

        return nn_lookup(self.nn_lookup_pos, s_det, i_ang, i_det, i_grn, 
                         pixel_size=self.detector_side_length/self.lookup_n_pix, 
                         n_pix=self.lookup_n_pix)






