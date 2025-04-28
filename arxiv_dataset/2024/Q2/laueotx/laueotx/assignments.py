import os, sys, warnings, argparse, h5py, numpy as np, time, itertools, random, shutil, datetime
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
from laueotx import laue_math, laue_math_tensorised, laue_math_graph, assignments
from laueotx.utils import logging as utils_logging
from laueotx.utils import io as utils_io
from laueotx.utils import config as utils_config
from laueotx.laue_rotation import Rotation
from laueotx.spot_neighbor_lookup import spot_neighbor_lookup, nn_lookup, nn_lookup_all, nn_lookup_ind_dist, nn_lookup_all
from laueotx.polycrystalline_sample import polycrystalline_sample, merge_duplicated_spots, apply_selections
from laueotx import spot_prototype_selection

FLAG_OUTLIER = -999999
LOGGER = utils_logging.get_logger(__file__)

def jaccard_distance(a, b):

    ni = len(np.intersect1d(a, b))
    nu = len(a)+len(b)-ni
    return 1.-ni/nu


def get_ray_index(v, decimals=4):

    vv = v[...,0]
    vv = vv/np.linalg.norm(vv, axis=1, keepdims=True)
    uval, uind, uinv = np.unique(np.round(vv, decimals=4), axis=0, return_index=True, return_inverse=True)
    return tf.cast(uinv, dtype=tf.int32)


def replace_grain_ids(spot_assign, grain_accept):

    spot_assign_ = spot_assign.copy()
    for i, gi in enumerate(grain_accept):
        select = spot_assign==gi
        spot_assign_[select] = i
    return spot_assign_

def get_neighbors_remove_outliers(s_best, s_target, max_l2_inlier, inds_best, nn_lookup_data, batch_size=100):

    # unpack args
    nn_lookup_ind, lookup_pixel_size, lookup_n_pix = nn_lookup_data
    i_ang_best, i_det_best, i_grn_best = inds_best

    # get nearest neighbor and translate indices for segment_sums, remove outliers
    
    n_batches = int(np.ceil(len(s_best)/batch_size))
    i_target_in, dist_target_in, i_grn_in = [], [], []
    # for i in LOGGER.progressbar(range(n_batches), at_level='info', desc=f'getting neighbours for {len(s_best)} spots, removing outliers with distance > {max_l2_inlier:4.2f} [mm]'):
    for i in range(n_batches):

        si, ei = i*batch_size, (i+1)*batch_size

        # get the nearest neighbors
        i_target_ = nn_lookup_all(nn_lookup_ind, s_target, s_best[si:ei], i_ang_best[si:ei], i_det_best[si:ei], i_grn_best[si:ei], lookup_pixel_size, tf.cast(lookup_n_pix, tf.int32))

        # calculate distances
        s_target_ = tf.gather(s_target, i_target_)
        dist_target_ = tf.reduce_sum((s_target_-tf.expand_dims(s_best[si:ei], axis=1))**2, axis=-1)
        l2_target = tf.math.sqrt(dist_target_)

        # remove outliers
        select_inliers = l2_target<max_l2_inlier
        i_target_in_, dist_target_in_, i_grn_in_ = apply_selections(select_inliers, i_target_, dist_target_, tf.repeat(i_grn_best[si:ei], l2_target.shape[1], axis=1))

        # store
        i_target_in.append(i_target_in_)
        dist_target_in.append(dist_target_in_)
        i_grn_in.append(i_grn_in_)

    # merge lists
    i_target_in = np.concatenate(i_target_in, axis=0)
    dist_target_in = np.concatenate(dist_target_in, axis=0)
    i_grn_in = np.concatenate(i_grn_in, axis=0)

    # verb
    LOGGER.debug(f'got {len(i_target_in)} neighbor spots')

    return i_target_in, dist_target_in, i_grn_in



def prototype_selection_spot_subset(sample, a_est, x_est):

    n_candidates = len(a_est)
    sample.set_tensor_variables(a_est, x_est) # this is only so that the next line runs
    i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_candidates)
    # i_ray = tf.gather(get_ray_index(sample.v, decimals=4), i_hkl)
    
    # get spots    
    LOGGER.info(f'generating spots for {n_candidates} candidate grains')
    s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=batch_size)
    s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all)
    s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, decimals=4, return_index=False, split_by_grain=True, verb=False)
    _, uc = np.unique(i_grn_best, return_counts=True)
    min_spot_loss_delta = np.mean(uc)

    # get nearest neighbor and translate indices for segment_sums, remove outliers
    i_target_in, dist_target_in, i_grn_in = get_neighbors_remove_outliers(s_best, s_target, max_l2_inlier, 
                                                                          inds_best=(i_ang_best, i_det_best, i_grn_best), 
                                                                          nn_lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), 
                                                                          batch_size=batch_size)
    
    # convert indices for unsorted_segment_sum
    i_target_nn_inds, i_target_nn = tf.unique(i_target_in)
    n_target_nn = len(i_target_nn_inds)

    # compute SPOT allocation   

    # get the spot-to-nn_spot cost matrix
    eps = 2*max(noise_sig, pixel_size)**2
    C = tf.math.exp(-dist_target_in/eps) + 1e-20
    ot_b = 1.0
    i_gts_uv, segment_ids_gts = tf.unique( tf.cast(i_target_nn, tf.int64)*10000000000 + tf.cast(i_grn_in, tf.int64)) # joint grain-to-spot hash
    segment_ids_tgt = tf.gather(i_target_nn, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    segment_ids_grn = tf.gather(i_grn_in, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    Cgs_seg = tf.math.unsorted_segment_min(C, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv))
    segment_ids_grn = tf.cast(tf.math.unsorted_segment_mean(i_grn_in, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32) # this works because i_grn and i_tgt are the same for multiple hash matches
    segment_ids_tgt = tf.cast(tf.math.unsorted_segment_mean(i_target_nn, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32)



    ot_b = 1.0
    select_prototype, spot_loss = spot_prototype_selection.tf_sparse_SPOT_GreedySubsetSelection(C=Cgs_seg,
                                                                                                segs_src=segment_ids_grn,
                                                                                                segs_tgt=segment_ids_tgt,
                                                                                                n_src=n_candidates, 
                                                                                                n_tgt=n_target_nn,
                                                                                                q=ot_b, 
                                                                                                k=n_max_grains_select)

    # use the epsilon criterion to remove prototypes that are not contributing much more to the SPOT objective

    spot_loss = np.array(spot_loss)
    # max_spot_loss_diff = 1e-2
    # select = (spot_loss/spot_loss.max()) > max_spot_loss_diff
    # spot_loss = spot_loss[select]
    # LOGGER.info(f'restricting grains with spot_loss/spot_loss.max() > {max_spot_loss_diff:2.4f} {len(spot_loss)}, {len(select)}')
    spot_loss_frac = np.argmin(spot_loss[1:]/spot_loss[:-1])

    if spot_loss_frac == 0:
        n_grains_accept = 1
    else:
        n_grains_accept = spot_loss_frac+2

    # y = spot_loss - np.min(spot_loss)
    # y = y/np.max(y)
    # n_grains_accept = np.nonzero(np.abs(np.diff(y, n=1))<1e-3)[0][0]+1
    # n_grains_accept = 120

    # n_grains_accept = np.argmax(np.diff(spot_loss, n=2)[1:])+3
    grain_accept = select_prototype[:n_grains_accept]
    LOGGER.info(f'SelectionOfPrototypesOT checked {len(select_prototype)} grains, selected {n_grains_accept}')

    return grain_accept


def prototype_selection_spot(conf, s_target, a_est, x_est, l_est, lookup_data, noise_sig, batch_size=1000, n_sig_out=3, pixel_size=0.1, n_max_grains_select=1200, test=False, **kwargs):

    # def get_spot_to_grain_cost(sample, a_est, x_est, batch_size):

    #     # get spots    
    #     LOGGER.info(f'generating spots for {n_candidates} candidate grains')


    #     eps = 2*max(noise_sig, pixel_size)**2
    #     n_batches = int(np.ceil(n_candidates/batch_size))
    #     list_Cgs_seg = []
    #     list_segment_ids_grn = []
    #     list_segment_ids_tgt = []

    #     # run this in a loop to limit memory usage
    #     for i in LOGGER.progressbar(range(n_batches), desc='getting grain-to-spot cost matrix', at_level='info'):


    #         si, ei  = i*batch_size, (i+1)*batch_size
    #         a_batch, x_batch = a_est[si:ei], x_est[si:ei]
    #         n_grains_batch = len(a_batch)
    #         i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_grains_batch)
    #         i_grn = i_grn + i*batch_size

    #         sample.set_tensor_variables(a_batch, x_batch) # this is only so that the next line runs
    #         s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=batch_size)
    #         s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all)
    #         s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, decimals=4, return_index=False, split_by_grain=True, verb=False)
    #         _, uc = np.unique(i_grn_best, return_counts=True)

    #         # get nearest neighbor and translate indices for segment_sums, remove outliers
    #         i_target_in, dist_target_in, i_grn_in = get_neighbors_remove_outliers(s_best, s_target, max_l2_inlier, 
    #                                                                               inds_best=(i_ang_best, i_det_best, i_grn_best), 
    #                                                                               nn_lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), 
    #                                                                               batch_size=batch_size)
    #         i_target_nn_inds, i_target_nn = tf.unique(i_target_in)

    #         # get the spot-to-nn_spot cost matrix
    #         C = tf.math.exp(-dist_target_in/eps) + 1e-20
    #         i_gts_uv, segment_ids_gts = tf.unique( tf.cast(i_target_nn, tf.int64)*10000000000 + tf.cast(i_grn_in, tf.int64)) # joint grain-to-spot hash
    #         segment_ids_tgt = tf.gather(i_target_nn, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    #         segment_ids_grn = tf.gather(i_grn_in, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    #         Cgs_seg = tf.math.unsorted_segment_min(C, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv))
    #         segment_ids_grn = tf.cast(tf.math.unsorted_segment_mean(i_grn_in, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32) # this works because i_grn and i_tgt are the same for multiple hash matches
    #         segment_ids_tgt = tf.cast(tf.math.unsorted_segment_mean(i_target_nn, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32)

    #         list_Cgs_seg.append(Cgs_seg)
    #         list_segment_ids_grn.append(segment_ids_grn)
    #         # list_segment_ids_tgt.append(i_target_in)
    #         list_segment_ids_tgt.append(segment_ids_tgt)

    #     Cgs_seg = tf.concat(list_Cgs_seg, axis=0)
    #     segment_ids_grn = tf.concat(list_segment_ids_grn, axis=0)
    #     segment_ids_tgt = tf.concat(list_segment_ids_tgt, axis=0)
    #     # convert indices for unsorted_segment_sum
    #     # i_target_nn_inds, i_target_nn = tf.unique(segment_ids_tgt_in)
    #     # i_target_nn_inds, segment_ids_tgt = tf.unique(tf.concat(list_segment_ids_tgt, axis=0))
    #     n_target_nn = len(segment_ids_tgt)

    #     return Cgs_seg, segment_ids_grn, segment_ids_tgt, n_target_nn

    def prototype_selection_spot_subset(a_est, x_est, select_best=False):

        n_candidates = len(a_est)
        sample.set_tensor_variables(a_est, x_est) # this is only so that the next line runs
        i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_candidates)
        # i_ray = tf.gather(get_ray_index(sample.v, decimals=4), i_hkl)
        
        # get spots    
        # LOGGER.info(f'generating spots for {n_candidates} candidate grains')
        s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=n_candidates)
        s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all)
        s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, decimals=4, return_index=False, split_by_grain=True, verb=False)
        _, uc = np.unique(i_grn_best, return_counts=True)
        min_spot_loss_delta = np.mean(uc)

        # get nearest neighbor and translate indices for segment_sums, remove outliers
        i_target_in, dist_target_in, i_grn_in = get_neighbors_remove_outliers(s_best, s_target, max_l2_inlier, 
                                                                              inds_best=(i_ang_best, i_det_best, i_grn_best), 
                                                                              nn_lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), 
                                                                              batch_size=n_candidates)
        
        # convert indices for unsorted_segment_sum
        i_target_nn_inds, i_target_nn = tf.unique(i_target_in)
        n_target_nn = len(i_target_nn_inds)

        # compute SPOT allocation   

        # get the spot-to-nn_spot cost matrix
        eps = 2*max(noise_sig, pixel_size)**2
        C = tf.math.exp(-dist_target_in/eps) + 1e-20
        ot_b = 1.0
        i_gts_uv, segment_ids_gts = tf.unique( tf.cast(i_target_nn, tf.int64)*10000000000 + tf.cast(i_grn_in, tf.int64)) # joint grain-to-spot hash
        segment_ids_tgt = tf.gather(i_target_nn, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
        segment_ids_grn = tf.gather(i_grn_in, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
        Cgs_seg = tf.math.unsorted_segment_min(C, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv))
        segment_ids_grn = tf.cast(tf.math.unsorted_segment_mean(i_grn_in, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32) # this works because i_grn and i_tgt are the same for multiple hash matches
        segment_ids_tgt = tf.cast(tf.math.unsorted_segment_mean(i_target_nn, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32)


        select_prototype, spot_loss = spot_prototype_selection.tf_sparse_SPOT_GreedySubsetSelection(C=Cgs_seg,
                                                                                                    segs_src=segment_ids_grn,
                                                                                                    segs_tgt=segment_ids_tgt,
                                                                                                    n_src=n_candidates, 
                                                                                                    n_tgt=n_target_nn,
                                                                                                    q=ot_b, 
                                                                                                    k=n_candidates)

        # use the epsilon criterion to remove prototypes that are not contributing much more to the SPOT objective

        spot_loss = np.array(spot_loss)
        # max_spot_loss_diff = 1e-2
        # select = (spot_loss/spot_loss.max()) > max_spot_loss_diff
        # spot_loss = spot_loss[select]
        # LOGGER.info(f'restricting grains with spot_loss/spot_loss.max() > {max_spot_loss_diff:2.4f} {len(spot_loss)}, {len(select)}')

        # y = spot_loss - np.min(spot_loss)
        # y = y/np.max(y)
        # n_grains_accept = np.nonzero(np.abs(np.diff(y, n=1))<1e-3)[0][0]+1
        # n_grains_accept = 120

        # n_grains_accept = np.argmax(np.diff(spot_loss, n=2)[1:])+3

        if select_best:
            
            spot_loss_frac = np.argmin(spot_loss[1:]/spot_loss[:-1])
            n_grains_accept = 1 if spot_loss_frac == 0 else spot_loss_frac+2
            grain_accept = select_prototype[:n_grains_accept]
            LOGGER.info(f'Final SelectionOfPrototypesOT checked {len(select_prototype)} grains, selected {n_grains_accept}')

        else:

            n_grains_accept = batch_size//n_batches
            grain_accept = select_prototype[:n_grains_accept]
            LOGGER.info(f'--- Part SelectionOfPrototypesOT checked {len(select_prototype)} grains, selected {n_grains_accept}')

        return grain_accept, spot_loss


    tf.config.run_functions_eagerly(True)

    # get number of candidates and exit if needed    
    n_candidates = len(a_est)
    if n_candidates==0:
        raise Exception('did not find any grains, aborting')
    elif n_candidates==1:
        raise Exception('need more than one grain for this function to make sense')

    # unpack the lookup
    nn_lookup_ind, lookup_pixel_size, lookup_n_pix = lookup_data

    # settings
    max_l2_inlier = max(noise_sig, pixel_size)*n_sig_out

    # sort candidates according to the loss
    best_ids = np.argsort(l_est)
    a_est = np.array(a_est)[best_ids]
    x_est = np.array(x_est)[best_ids]
    l_est = np.array(l_est)[best_ids]

    # initialize grain
    sample = polycrystalline_sample(conf, hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=False)

    batch_size = 4000
    n_batches = int(np.ceil(n_candidates/batch_size))
    grain_ids = []

    LOGGER.info(f'Prototype selection loss with batch_size={batch_size} n_batches={n_batches}')
    if n_batches == 1:

        grain_accept, spot_loss = prototype_selection_spot_subset(a_est, x_est, select_best=True)

    else:
        
        for i in range(n_batches):

            si, ei  = i*batch_size, (i+1)*batch_size        
            subset_grain_ids, _ = prototype_selection_spot_subset(a_est[si:ei], x_est[si:ei], select_best=False)
            grain_ids.append( subset_grain_ids + si)
        grain_ids = np.concatenate(grain_ids)

        inds_grain_accept, spot_loss = prototype_selection_spot_subset(a_est[grain_ids], x_est[grain_ids], select_best=True)
        grain_accept = grain_ids[inds_grain_accept]



    # sample.set_tensor_variables(a_est, x_est) # this is only so that the next line runs
    # LOGGER.info(f'getting batch indices for {len(a_est)} grains')
    # i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_candidates)
    # i_ray = tf.gather(get_ray_index(sample.v, decimals=4), i_hkl)

    # Cgs_seg, segment_ids_grn, segment_ids_tgt, n_target_nn = get_spot_to_grain_cost(sample, a_est, x_est, batch_size)
    # new_Cgs_seg, new_segment_ids_grn, new_segment_ids_tgt, new_n_target_nn = Cgs_seg, segment_ids_grn, segment_ids_tgt, n_target_nn

    # # get spots    
    # LOGGER.info(f'generating spots for {n_candidates} candidate grains')

    # eps = 2*max(noise_sig, pixel_size)**2
    # ot_b = 1.0
    # n_batches = int(np.ceil(n_candidates/batch_size))
    # list_Cgs_seg = []
    # list_segment_ids_grn = []
    # list_segment_ids_tgt_in = []

    # # run this in a loop to limit memory usage
    # for i in LOGGER.progressbar(range(n_batches), desc='getting grain-to-spot cost matrix', at_level='info'):

    #     si, ei  = i*batch_size, (i+1)*batch_size
    #     sample.set_tensor_variables(a_est[si:ei], x_est[si:ei]) # this is only so that the next line runs
    #     s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=batch_size)
    #     s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best = apply_selections(select_sample, s_sample, i_grn[si:ei], i_ang[si:ei], i_det[si:ei], i_hkl[si:ei], i_all[si:ei])
    #     s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, decimals=4, return_index=False, split_by_grain=True, verb=False)
    #     _, uc = np.unique(i_grn_best, return_counts=True)
    #     min_spot_loss_delta = np.mean(uc)

    #     # get nearest neighbor and translate indices for segment_sums, remove outliers
    #     i_target_in, dist_target_in, i_grn_in = get_neighbors_remove_outliers(s_best, s_target, max_l2_inlier, 
    #                                                                           inds_best=(i_ang_best, i_det_best, i_grn_best), 
    #                                                                           nn_lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), 
    #                                                                           batch_size=batch_size)
    #     i_target_nn_inds, i_target_nn = tf.unique(i_target_in)

    #     # get the spot-to-nn_spot cost matrix
    #     C = tf.math.exp(-dist_target_in/eps) + 1e-20
    #     i_gts_uv, segment_ids_gts = tf.unique( tf.cast(i_target_nn, tf.int64)*10000000000 + tf.cast(i_grn_in, tf.int64)) # joint grain-to-spot hash
    #     segment_ids_tgt = tf.gather(i_target_nn, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    #     segment_ids_grn = tf.gather(i_grn_in, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    #     Cgs_seg = tf.math.unsorted_segment_min(C, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv))
    #     segment_ids_grn = tf.cast(tf.math.unsorted_segment_mean(i_grn_in, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32) # this works because i_grn and i_tgt are the same for multiple hash matches
    #     segment_ids_tgt = tf.cast(tf.math.unsorted_segment_mean(i_target_nn, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32)

    #     list_Cgs_seg.append(Cgs_seg)
    #     list_segment_ids_grn.append(segment_ids_grn)
    #     list_segment_ids_tgt_in.append(i_target_in)
    

    # Cgs_seg = tf.concat(list_Cgs_seg, axis=0)
    # segment_ids_grn = tf.concat(list_segment_ids_grn, axis=0)
    # segment_ids_tgt_in = tf.concat(list_segment_ids_tgt_in, axis=0)
    # # convert indices for unsorted_segment_sum
    # i_target_nn_inds, i_target_nn = tf.unique(segment_ids_tgt_in)
    # i_target_nn_inds, segment_ids_tgt = tf.unique(tf.concat(list_segment_ids_tgt_in, axis=0))
    # n_target_nn = len(i_target_nn_inds)


    # import pudb; pudb.set_trace();
    # pass
    
    # # initialize grain
    # sample = polycrystalline_sample(conf, hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=False)
    # sample.set_tensor_variables(a_est, x_est) # this is only so that the next line runs
    # i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_candidates)
    # # i_ray = tf.gather(get_ray_index(sample.v, decimals=4), i_hkl)
    
    # # get spots    
    # LOGGER.info(f'generating spots for {n_candidates} candidate grains')
    # s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=batch_size)
    # s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all)
    # s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, decimals=4, return_index=False, split_by_grain=True, verb=False)
    # _, uc = np.unique(i_grn_best, return_counts=True)
    # min_spot_loss_delta = np.mean(uc)

    # # get nearest neighbor and translate indices for segment_sums, remove outliers
    # i_target_in, dist_target_in, i_grn_in = get_neighbors_remove_outliers(s_best, s_target, max_l2_inlier, 
    #                                                                       inds_best=(i_ang_best, i_det_best, i_grn_best), 
    #                                                                       nn_lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), 
    #                                                                       batch_size=batch_size)
    
    # # convert indices for unsorted_segment_sum
    # i_target_nn_inds, i_target_nn = tf.unique(i_target_in)
    # n_target_nn = len(i_target_nn_inds)

    # # compute SPOT allocation   

    # # get the spot-to-nn_spot cost matrix
    # eps = 2*max(noise_sig, pixel_size)**2
    # C = tf.math.exp(-dist_target_in/eps) + 1e-20
    # ot_b = 1.0
    # i_gts_uv, segment_ids_gts = tf.unique( tf.cast(i_target_nn, tf.int64)*10000000000 + tf.cast(i_grn_in, tf.int64)) # joint grain-to-spot hash
    # segment_ids_tgt = tf.gather(i_target_nn, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    # segment_ids_grn = tf.gather(i_grn_in, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    # Cgs_seg = tf.math.unsorted_segment_min(C, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv))
    # segment_ids_grn = tf.cast(tf.math.unsorted_segment_mean(i_grn_in, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32) # this works because i_grn and i_tgt are the same for multiple hash matches
    # segment_ids_tgt = tf.cast(tf.math.unsorted_segment_mean(i_target_nn, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv)), tf.int32)



    # ot_b = 1.0
    # select_prototype, spot_loss = spot_prototype_selection.tf_sparse_SPOT_GreedySubsetSelection(C=Cgs_seg,
    #                                                                                             segs_src=segment_ids_grn,
    #                                                                                             segs_tgt=segment_ids_tgt,
    #                                                                                             n_src=n_candidates, 
    #                                                                                             n_tgt=n_target_nn,
    #                                                                                             q=ot_b, 
    #                                                                                             k=n_max_grains_select)

    # # use the epsilon criterion to remove prototypes that are not contributing much more to the SPOT objective

    # spot_loss = np.array(spot_loss)
    # # max_spot_loss_diff = 1e-2
    # # select = (spot_loss/spot_loss.max()) > max_spot_loss_diff
    # # spot_loss = spot_loss[select]
    # # LOGGER.info(f'restricting grains with spot_loss/spot_loss.max() > {max_spot_loss_diff:2.4f} {len(spot_loss)}, {len(select)}')
    # spot_loss_frac = np.argmin(spot_loss[1:]/spot_loss[:-1])

    # if spot_loss_frac == 0:
    #     n_grains_accept = 1
    # else:
    #     n_grains_accept = spot_loss_frac+2

    # # y = spot_loss - np.min(spot_loss)
    # # y = y/np.max(y)
    # # n_grains_accept = np.nonzero(np.abs(np.diff(y, n=1))<1e-3)[0][0]+1
    # # n_grains_accept = 120

    # # n_grains_accept = np.argmax(np.diff(spot_loss, n=2)[1:])+3
    # grain_accept = select_prototype[:n_grains_accept]
    # LOGGER.info(f'SelectionOfPrototypesOT checked {len(select_prototype)} grains, selected {n_grains_accept}')
    # print('full method', grain_accept)

    # sparse version
    if test:
        ot_b = tf.ones(n_target_nn, dtype=C.dtype)/n_target_nn
        spot_prototype_selection.test_sparse_vs_dense(C, n_candidates, n_max_grains_select, i_grn_in, i_target_nn, ot_b)

    # get the final params
    n_accept = len(grain_accept)
    a_accept, x_accept, l_accept = apply_selections(np.array(grain_accept).ravel(), np.array(a_est), np.array(x_est), np.array(l_est))

    # render final spots
    i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_accept)
    sample.set_tensor_variables(a_accept, x_accept) # this is only so that the next line runs
    s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=n_accept)
    s_final, i_grn_final, i_ang_final, i_det_final, i_hkl_final, i_all_final = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all)
    s_final, i_grn_final, i_ang_final, i_hkl_final, i_det_final, i_all_final = merge_duplicated_spots(s_final, i_grn_final, i_ang_final, i_hkl_final, i_det_final, i_all_final, decimals=4, return_index=False, split_by_grain=True, verb=False)

    # assign observed spots to grains
    dist_nn, spotind_nn = nn_lookup_ind_dist(nn_lookup_ind, s_target, s_final, i_ang_final, i_det_final, i_grn_final, lookup_pixel_size, tf.cast(lookup_n_pix, tf.int32)) # nearest neighbours and distances squared
    l2_nn = tf.math.sqrt(dist_nn) # get l2 norm
    select_inliers =  l2_nn<max_l2_inlier # check if inlier according to noise level
    spotind_nn_inliers = tf.where(select_inliers, spotind_nn, -1) # replace outliers with -1
    ordering = np.searchsorted(i_grn_final[:,0], np.unique(i_grn_final)) # find ordering
    ordering = np.append(ordering, len(i_grn_final)) # add last element 
    spot_obs_assign = np.ones(len(s_target), dtype=int)*FLAG_OUTLIER # init with outlier flag
    spot_mod_assign = []
    for gi in range(n_accept): # loop to assign spots to each grain
        
        si, ei = ordering[gi], ordering[gi+1] 
        spotind_obs = spotind_nn_inliers[si:ei]
        select_inliers = spotind_obs!=-1
        spotind_obs = spotind_obs[select_inliers] # remove outliers
        spot_obs_assign[spotind_obs] = gi
        mod_assign = np.ones(ei-si, dtype=int)*FLAG_OUTLIER
        mod_assign[select_inliers] = spotind_obs
        spot_mod_assign.append(mod_assign)

    # wrap up
    grain_accept = np.sort(np.int32(grain_accept))
    spot_obs_assign = replace_grain_ids(spot_obs_assign, grain_accept)
    spot_mod_assign = replace_grain_ids(np.concatenate(spot_mod_assign), grain_accept)
    n_accept = len(grain_accept)
    a_accept = a_est[grain_accept]
    x_accept = x_est[grain_accept]
    l_accept = l_est[grain_accept]

    # swich back to graph mode
    tf.config.run_functions_eagerly(False)

    n_spots_assigned = np.count_nonzero(spot_obs_assign>=0)
    LOGGER.info(f'final selection of {len(grain_accept)} grains with {n_spots_assigned}/{len(spot_obs_assign)} ({n_spots_assigned/len(spot_obs_assign)*100:4.2f}%) assigned spots')

    return a_accept, x_accept, l_accept, grain_accept, spot_obs_assign, spot_mod_assign, spot_loss
    



def prune_candidates_lasso(conf, s_target, a_est, x_est, l_est, lookup_data, noise_sig, batch_size=100, n_sig_out=3, pixel_size=0.1, **kwargs):


    from laueotx.polycrystalline_sample import polycrystalline_sample, merge_duplicated_spots, apply_selections
    from laueotx.spot_neighbor_lookup import spot_neighbor_lookup, nn_lookup, nn_lookup_all, nn_lookup_ind_dist, nn_lookup_all
    tf.config.run_functions_eagerly(True)


    # get number of candidates and exit if needed    
    n_candidates = len(a_est)
    if n_candidates==0:
        raise Exception('did not find any grains, aborting')
    elif n_candidates==1:
        return Exception('need more than one grain for this function to make sense')

    # unpack the lookup
    nn_lookup_ind, lookup_pixel_size, lookup_n_pix = lookup_data

    # init output
    spot_obs_assign = np.ones(len(s_target), dtype=int)*FLAG_OUTLIER
    spot_mod_assign = []

    ot_eps = 2*min(noise_sig*3, 0.1)**2
    ot_kap = 0.1
    ot_lam = 0.1
    k_nn = 8

    # initialize grain
    
    sample = polycrystalline_sample(conf, hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=False)
    sample.set_tensor_variables(a_est, x_est) # this is only so that the next line runs
    i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_candidates)
    # i_ray = tf.gather(get_ray_index(sample.v, decimals=4), i_hkl)
    
    # get spots    
    s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=batch_size)
    # s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best, i_ray_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all, i_ray)
    s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all)
    # s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, i_ray=i_ray_best, decimals=4, return_index=False, split_by_grain=True, verb=False)
    s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, decimals=4, return_index=False, split_by_grain=True, verb=False)

    def get_assignment(s, i_ang, i_det, i_grn, eps, kap, lam, k_nn=None):
    
    
        i_target = nn_lookup_all(nn_lookup_ind, s_target, s, i_ang, i_det, i_grn, lookup_pixel_size, tf.cast(lookup_n_pix, tf.int32))
        s_nn = tf.gather(s_target, i_target[:,:k_nn])
        dist = tf.reduce_sum((s_nn-tf.expand_dims(s[:,:k_nn], axis=1))**2, axis=-1)
        K_sparse = tf.math.exp(-dist**2/eps) + 1e-20
        i_target_flat = tf.reshape(i_target[:,:k_nn], (-1,))
        i_target_nn_inds, i_target_nn_flat = tf.unique(i_target_flat)
        i_target_nn = tf.reshape(i_target_nn_flat, shape=i_target[:,:k_nn].shape)
        n_target_nn = len(i_target_nn_inds)
        ot_a=tf.ones(len(s), dtype=tf.float64)
        ot_b=tf.ones(n_target_nn, dtype=tf.float64)
        from laueotx import optimal_transport
        Q = optimal_transport.sinkhorn_knopp_unbalanced_sparse(a=ot_a, b=ot_b, K_sparse=K_sparse, i_target=i_target_nn, reg=eps, reg_ma=lam, reg_mb=kap, n_iter_max=1000, err_threshold=1e-6)
        Q += 1e-20

        # full loss calculation

        l2 = tf.math.sqrt(dist)
        l2_sq_q = tf.reduce_sum(Q*l2**2)
        H = tf.reduce_sum(tf.math.log(Q)*Q)
            
        Q_sum_kap = tf.reduce_sum(Q, axis=1)
        D1_kappa = tf.reduce_sum(Q_sum_kap*tf.math.log(Q_sum_kap/ot_a))
        D2_kappa = tf.reduce_sum(Q_sum_kap)

        Q_flat = tf.reshape(Q, (-1,1))
        Q_sum_lam = tf.math.unsorted_segment_sum(Q_flat, segment_ids=i_target_nn_flat, num_segments=len(i_target_nn_inds))
        D1_lambda = tf.reduce_sum(Q_sum_lam[:,0]*tf.math.log(Q_sum_lam[:,0]/ot_b))
        D2_lambda = tf.reduce_sum(Q_sum_lam)

        loss_full = 0.5*l2_sq_q + eps*H + kap*(D1_kappa - D2_kappa + tf.reduce_sum(ot_a)) + lam*(D1_lambda - D2_lambda + tf.reduce_sum(ot_b))

        return Q, loss_full

    i_remaining_candidates = np.arange(n_candidates)
    history_candidates = []
    history_loss = []
    for i in range(n_candidates-1):

        # select the remaining model spots
        select = np.in1d(i_grn_best, i_remaining_candidates)

        # convert indices
        i_target_nn_inds, i_target_nn_flat = tf.unique(i_grn_best[:,0][select])

        # get assignment for this grain configuration
        Q, loss_full = get_assignment(s_best[select], i_ang_best[select], i_det_best[select], i_grn=i_target_nn_flat, kap=ot_kap, lam=ot_lam, eps=ot_eps, k_nn=k_nn)

        Q_per_grain = tf.math.unsorted_segment_mean(tf.reduce_sum(Q, axis=1), segment_ids=i_target_nn_flat, num_segments=len(i_target_nn_inds));
        # i_del = [np.argmin(Q_per_grain)]
        i_del = [np.argmin(Q_per_grain)]
        i_remaining_candidates = np.delete(i_remaining_candidates, i_del)
        LOGGER.info(f'deleting {len(i_del):>6d}/{n_candidates} candidates, remaining {len(i_remaining_candidates):>6d}, min_Q={np.min(Q_per_grain):6.4e}, max_Q={np.max(Q_per_grain):6.4e} loss={np.min(loss_full): 6.4e}')
        history_candidates.append(i_remaining_candidates)
        history_loss.append(loss_full)

        if len(i_remaining_candidates)==0:
            break

    history_loss = np.array(history_loss)
    id_best = np.argmin(history_loss)
    grain_accept = np.array(history_candidates[id_best])
    select_final = np.in1d(i_grn_best, grain_accept)
    Q, loss_full = get_assignment(s_best[select_final], i_ang_best[select_final], i_det_best[select_final], i_grn_best[select_final], kap=ot_kap, lam=ot_lam, eps=ot_eps, k_nn=k_nn)
    LOGGER.info(f'best loss for {len(grain_accept)} grain, loss={loss_full: 4.4e}')

    # i_best_candidates remove zeros
    select_final = np.in1d(i_grn_best, grain_accept)
    LOGGER.info(f'best configuration with {len(grain_accept)} grain, final loss={loss_full: 4.4e}')

    # get the final params
    n_accept = len(grain_accept)
    a_accept = a_est[grain_accept]
    x_accept = x_est[grain_accept]
    l_accept = l_est[grain_accept]
    s_final = s_best[select_final]
    i_ang_final = i_ang_best[select_final]
    i_det_final = i_det_best[select_final]
    i_grn_final = i_grn_best[select_final]


    # assign observed spots to grains
    dist_nn, spotind_nn = nn_lookup_ind_dist(nn_lookup_ind, s_target, s_final, i_ang_final, i_det_final, i_grn_final, lookup_pixel_size, tf.cast(lookup_n_pix, tf.int32))
    max_diff = max(noise_sig, pixel_size)*n_sig_out
    spotind_nn_inliers = tf.where(dist_nn<max_diff, spotind_nn, -1)
    ordering = np.searchsorted(i_grn_final[:,0], np.unique(i_grn_final))
    ordering = np.append(ordering, len(i_grn_final)) # add last element 
    spot_obs_assign = np.ones(len(s_target), dtype=int)*FLAG_OUTLIER
    spot_mod_assign = []
    for gi in range(n_accept):
        
        si, ei = ordering[gi], ordering[gi+1]
        spotind_obs = spotind_nn_inliers[si:ei]
        select_inliers = spotind_obs!=-1
        spotind_obs = spotind_obs[select_inliers] # remove outliers
        spot_obs_assign[spotind_obs] = gi
        mod_assign = np.ones(ei-si, dtype=int)*FLAG_OUTLIER
        mod_assign[select_inliers] = spotind_obs
        spot_mod_assign.append(mod_assign)

    # wrap up
    grain_accept = np.int32(grain_accept)
    spot_obs_assign = replace_grain_ids(spot_obs_assign, grain_accept)
    spot_mod_assign = replace_grain_ids(np.concatenate(spot_mod_assign), grain_accept)
    n_accept = len(grain_accept)
    a_accept = a_est[grain_accept]
    x_accept = x_est[grain_accept]
    l_accept = l_est[grain_accept]

    # swich back to graph mode
    tf.config.run_functions_eagerly(False)

    return a_accept, x_accept, l_accept, grain_accept, spot_obs_assign, spot_mod_assign












    class spot_scatter(tf.keras.layers.Layer):
    
        def __init__(self, n_candidates, n_obs_spots, map_grain_to_model, map_spot_to_nn, s_model, regularizer, initializer=tf.keras.initializers.Ones()):

            super(spot_scatter, self).__init__()
            self.map_grain_to_model = map_grain_to_model
            self.map_spot_to_nn = map_spot_to_nn
            self.s_model = s_model
            self.n_obs_spots = n_obs_spots
            self.regularizer=regularizer
            self.initializer=initializer
            self.n_candidates = n_candidates

        def call(self, inputs):

            w_gather = tf.gather(self.w, self.map_grain_to_model)
            s_gather = tf.math.abs(w_gather) * inputs
            return tf.math.unsorted_segment_mean(s_gather, segment_ids=self.map_spot_to_nn, num_segments=self.n_obs_spots)

        def build(self, input_shape):

            self.w = self.add_weight("kernel",
                                    shape=(n_candidates,),
                                    regularizer=self.regularizer,
                                    initializer=self.initializer)

    import pudb; pudb.set_trace();
    pass

    # l1, l2 = 0.5, (1-0.5)/2.
    l1 = 100
    swm = spot_scatter(n_candidates=n_candidates, n_obs_spots=len(s_target), map_grain_to_model=i_grn_best, map_spot_to_nn=spotind_nn, s_model=s_best, regularizer=tf.keras.regularizers.L1(l1))
    swm.build(s_best.shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    from tensorflow.compat.v1.train import GradientDescentOptimizer
    # optimizer = GradientDescentOptimizer(learning_rate=0.0001)
    n_epochs = 10000
    for i in range(n_epochs):

        with tf.GradientTape() as tape:
                
            w = swm.trainable_weights[0]
            s_pred = swm(s_best, training=True)
            loss = tf.reduce_mean((s_pred-s_target)**2) +  l1*tf.reduce_mean(tf.math.abs(w))
            grads = tape.gradient(loss, swm.trainable_weights)
            optimizer.apply_gradients(zip(grads, swm.trainable_weights))

        if i % 100 == 0:
            print("epoch {} loss={:12.6e}".format(i,loss))

    import pudb; pudb.set_trace();
    pass


def prune_candidates_clustering(conf, s_target, a_est, x_est, l_est, lookup_data, noise_sig, batch_size=100, **kwargs):

    from laueotx.polycrystalline_sample import polycrystalline_sample, merge_duplicated_spots, apply_selections
    from laueotx.spot_neighbor_lookup import spot_neighbor_lookup, nn_lookup, nn_lookup_all, nn_lookup_ind_dist
    tf.config.run_functions_eagerly(True)
    
    # get number of candidates and exit if needed    
    n_candidates = len(a_est)
    if n_candidates==0:
        raise Exception('did not find any grains, aborting')
    elif n_candidates==1:
        return Exception('need more than one grain for this function to make sense')

    # unpack the lookup
    nn_lookup_ind, lookup_pixel_size, lookup_n_pix = lookup_data

    # init output
    spot_obs_assign = np.ones(len(s_target), dtype=int)*FLAG_OUTLIER
    spot_mod_assign = []

    # initialize grain
    
    sample = polycrystalline_sample(conf, hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=False)
    sample.set_tensor_variables(a_est, x_est) # this is only so that the next line runs
    i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_candidates)
    # i_ray = tf.gather(get_ray_index(sample.v, decimals=4), i_hkl)
    
    # get spots    
    s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=batch_size)
    # s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best, i_ray_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all, i_ray)
    s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all)
    # s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, i_ray=i_ray_best, decimals=4, return_index=False, split_by_grain=True, verb=False)
    s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, decimals=4, return_index=False, split_by_grain=True, verb=False)
    dist_nn, spotind_nn = nn_lookup_ind_dist(nn_lookup_ind, s_target, s_best, i_ang_best, i_det_best, i_grn_best, lookup_pixel_size, tf.cast(lookup_n_pix, tf.int32))

    # get problem variables
    ordering = np.searchsorted(i_grn_best[:,0], np.unique(i_grn_best))
    ordering = np.append(ordering, len(i_grn_best)) # add last element 
    pixel_size = 0.1
    n_sig_out = 3
    max_diff = max(noise_sig, pixel_size)*n_sig_out
    spotind_nn_inliers = tf.where(dist_nn<max_diff, spotind_nn, -1)

    # create candidate distance matrix based on overlap in assigned observed spots
    coo1, coo2, dist = [], [], []
    time_start = time.time()
    ranger = LOGGER.progressbar(np.vstack(np.triu_indices(n_candidates,1)).T, at_level='info', desc='computing solution correlations ...')

    i_spg = np.concatenate([np.arange(c) for c in np.unique(i_grn_best, return_counts=True)[1]])

    spotind_nn_inliers_np = np.array(spotind_nn_inliers)
    
    for (i1, i2) in ranger:

        si1, ei1 = ordering[i1], ordering[i1+1]
        si2, ei2 = ordering[i2], ordering[i2+1]

        # get jaccard index
        jd = jaccard_distance(spotind_nn_inliers_np[si1:ei1], spotind_nn_inliers_np[si2:ei2])
                                
        if jd<1:
            coo1.append(i1)
            coo2.append(i2)
            dist.append(jd)
            if i1%1000==0:
                ranger.set_description(f'computing jaccard distance between grain candidates  ... found {len(dist)} pairs with overlaps')

    # run clustering
    coo1 = np.array(coo1)
    coo2 = np.array(coo2)
    dist = np.array(dist)

    from scipy.sparse import coo_array, csr_array
    dist_matrix = coo_array((dist, (coo1, coo2)), shape=(n_candidates, n_candidates))

    def dist_to_dense(dist, coo1, coo2, n, max_dist=2):
        from scipy.sparse import coo_array, csr_array
        dist = dist+1
        dist_matrix = coo_array((np.concatenate([dist, dist]), (np.concatenate([coo1, coo2]), np.concatenate([coo2, coo1]))), shape=(n, n))
        dist_matrix = dist_matrix.todense()
        dist_matrix[dist_matrix==0] = max_dist+1
        dist_matrix -= 1 
        np.fill_diagonal(dist_matrix, 0.)
        return dist_matrix

    dist_matrix_dense = dist_to_dense(dist, coo1, coo2, n_candidates)

    clustering_method = 'optics'

    if clustering_method == 'optics':

        from sklearn.cluster import OPTICS
        optics_xi = 0.5
        optics_max_epx = 0.2
        LOGGER.info(f'pruning duplicate solutions using OPTICS optics_xi={optics_xi:2.4e}')
        labels = OPTICS(min_samples=2, metric='precomputed', xi=optics_xi, max_eps=optics_max_epx).fit_predict(dist_matrix_dense)
        n_noise = np.count_nonzero(labels == -1)
        # add noise points as separate candidates
        labels[labels == -1] = np.arange(n_noise)+1+np.max(labels)

    elif clustering_method == 'dbscan':

        from sklearn.cluster import DBSCAN
        eps_dbscan = 0.001
        LOGGER.info(f'pruning duplicate solutions using DBSCAN eps_dbscan={eps_dbscan:2.4e}')
        labels = DBSCAN(eps=eps_dbscan, min_samples=1, metric='precomputed').fit_predict(dist_matrix_dense)
        # add noise points as separate candidates
        n_noise = np.count_nonzero(labels == -1)
        labels[labels == -1] = np.arange(n_noise)+1+np.max(labels)

    elif clustering_method == 'affinity_propagation':
            
        from sklearn.cluster import AffinityPropagation
        LOGGER.info(f'pruning duplicate solutions using AffinityPropagation')
        labels = AffinityPropagation(affinity='precomputed').fit_predict(-dist_matrix_dense)

    elif clustering_method == 'agglomerative_clustering':

        from sklearn.cluster import AgglomerativeClustering
        LOGGER.info(f'pruning duplicate solutions using AgglomerativeClustering')
        labels = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.99).fit_predict(dist_matrix_dense)

    # sort labels according to the loss, so that we accept the best grain in the cluster
    sorting = np.argsort(l_est)
    labels_uv, grain_accept = np.unique(labels[sorting], return_index=True)
    grain_accept = sorting[grain_accept]

    # assign observed spots to grains
    for gi in grain_accept:
        
        si, ei = ordering[gi], ordering[gi+1]
        spotind_obs = spotind_nn_inliers[si:ei]
        select_inliers = spotind_obs!=-1
        spotind_obs = spotind_obs[select_inliers] # remove outliers
        spot_obs_assign[spotind_obs] = gi
        mod_assign = np.ones(ei-si, dtype=int)*FLAG_OUTLIER
        mod_assign[select_inliers] = spotind_obs
        spot_mod_assign.append(mod_assign)

    # wrap up
    grain_accept = np.sort(np.int32(grain_accept))
    spot_obs_assign = replace_grain_ids(spot_obs_assign, grain_accept)
    spot_mod_assign = replace_grain_ids(np.concatenate(spot_mod_assign), grain_accept)
    n_accept = len(grain_accept)
    a_accept = a_est[grain_accept]
    x_accept = x_est[grain_accept]
    l_accept = l_est[grain_accept]


    # swich back to graph mode
    tf.config.run_functions_eagerly(False)

    return a_accept, x_accept, l_accept, grain_accept, spot_obs_assign, spot_mod_assign




# @profile
def assign_grain_spot_heuristic_batch(conf, s_target, a_est, x_est, loss, lookup_data, noise_sig, n_grains_max=None, batch_size=1000, n_candidates=100000, sort=True, max_loss=-1, method='strict'):

    tf.config.run_functions_eagerly(True)

    from laueotx.polycrystalline_sample import polycrystalline_sample, merge_duplicated_spots, apply_selections
    from laueotx.spot_neighbor_lookup import spot_neighbor_lookup, nn_lookup, nn_lookup_all, nn_lookup_ind_dist

    # sort the parameters according to the loss
    if sort:
        best_ids = np.argsort(loss)
        a_est_candidate = np.array(a_est)[best_ids]
        x_est_candidate = np.array(x_est)[best_ids]
        l_est_candidate = np.array(loss)[best_ids]
    else:
        a_est_candidate = np.array(a_est)
        x_est_candidate = np.array(x_est)
        l_est_candidate = np.array(loss)

    # unpack the lookup

    nn_lookup_ind, lookup_pixel_size, lookup_n_pix = lookup_data

    # init output 

    spot_obs_assign = np.ones(len(s_target), dtype=int)*FLAG_OUTLIER
    spot_available = None
    grain_accept = []
    spot_mod_assign = []

    # start batches
                
    n_candidate = len(a_est_candidate)
    batch_size = min(n_candidates, batch_size)
    n_batches = int(n_candidates//batch_size)
    n_grains_found_previous = 0
    n_checked = 0

    # initialize grain

    sample = polycrystalline_sample(conf, hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=False)
    sample.set_tensor_variables(a_est_candidate[:batch_size], x_est_candidate[:batch_size]) # this is only so that the next line runs
    i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=batch_size)
    i_ray = tf.gather(get_ray_index(sample.v, decimals=4), i_hkl)
        
    LOGGER.info(f'searching {n_candidates} best candidate grains')
    ranger = LOGGER.progressbar(range(n_batches), at_level='info', desc='starting search')
    for i in ranger:

        si, ei = batch_size*i, batch_size*(i+1)
        
        # get the best solutions and their parameters

        a_best = a_est_candidate[si:ei]
        x_best = x_est_candidate[si:ei]
        l_best = l_est_candidate[si:ei]
        
  
        time_taken_total = 0
        sample.set_tensor_variables(a_best, x_best)
        s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=batch_size)
        s_best, i_grn_best, i_ang_best, i_det_best, i_hkl_best, i_all_best, i_ray_best = apply_selections(select_sample, s_sample, i_grn, i_ang, i_det, i_hkl, i_all, i_ray)
        i_grn_best = i_grn_best + si
        s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, select_merged = merge_duplicated_spots(s_best, i_grn_best, i_ang_best, i_hkl_best, i_det_best, i_all_best, i_ray=i_ray_best, decimals=4, return_index=True, split_by_grain=True, verb=False)
        dist_best, spotind_nn_best = nn_lookup_ind_dist(nn_lookup_ind, s_target, s_best, i_ang_best, i_det_best, i_grn_best, lookup_pixel_size, tf.cast(lookup_n_pix, tf.int32))

        if method == 'strict':

            assign_fun = assignments.assign_grain_spot_heuristic_with_outliers

        elif method == 'relaxed':

            assign_fun = assignments.assign_grain_spot_heuristic_relaxed

        # elif method == 'clustering':

        #     assign_fun = assignments.assign_grain_spot_heuristic_clustering

        else: 
            raise Exception(f'unknown assign function {method}')

        spot_available = assign_fun(spotind_nn=np.array(spotind_nn_best),
                                    diff_best=np.sqrt(dist_best),
                                    i_grn_best=np.array(i_grn_best),
                                    s_best=np.array(s_best),
                                    s_obs=np.array(s_target),
                                    l_best=l_best,
                                    noise_sig=noise_sig,
                                    n_sig_out=5,
                                    grain_accept=grain_accept,
                                    spot_available=spot_available, 
                                    spot_mod_assign=spot_mod_assign,
                                    spot_obs_assign=spot_obs_assign,
                                    n_checked_already=n_checked,
                                    n_candidates=n_candidates,
                                    ranger=ranger)

        n_grains_found_previous = len(grain_accept)
        n_checked += batch_size

        if len(grain_accept)==n_grains_max:
            break

        if n_checked > n_candidates:
            break
    
    # new line after the '\r' output
    # if LOGGER.level==20:
    #     print(' .. done')


    if len(grain_accept)==0:    

        raise Exception('did not find any grains, aborting')

    grain_accept = np.int32(grain_accept)
    spot_obs_assign = replace_grain_ids(spot_obs_assign, grain_accept)
    spot_mod_assign = replace_grain_ids(np.concatenate(spot_mod_assign), grain_accept)

    n_accept = len(grain_accept)
    a_accept = a_est_candidate[grain_accept]
    x_accept = x_est_candidate[grain_accept]
    l_accept = l_est_candidate[grain_accept]

    tf.config.run_functions_eagerly(False)

    return a_accept, x_accept, l_accept, grain_accept, spot_obs_assign, spot_mod_assign



# @profile
def assign_grain_spot_heuristic_relaxed(spotind_nn, diff_best, i_grn_best, s_obs, s_best, noise_sig, grain_accept=None, spot_available=None, spot_obs_assign=None, spot_mod_assign=None, i_grn_init=None, s_init=None, l_best=None, n_grains_max=1500, n_sig_out=3, n_checked_already=0, n_candidates=100000, ranger=None):

    n_spots_obs = len(s_obs) 
    spot_ids = np.arange(n_spots_obs)
    grain_ids = np.unique(i_grn_best)
    spot_current = np.zeros(n_spots_obs, dtype=bool)
    pixel_size = 0.1
    frac_max_outliers = 1
    frac_counts = 0.1
    mad_max = max(noise_sig, pixel_size)*n_sig_out
    max_diff = max(noise_sig, pixel_size)*n_sig_out
    l_best = np.zeros(len(s_best)) if l_best is None else l_best # init dummy
    ordering = np.searchsorted(i_grn_best[:,0], grain_ids)
    ordering = np.append(ordering, len(i_grn_best)) # add last element 

    time_start = time.time()

    n_max_spots_per_grain = k = 1000
    if spot_available is None:
        j = 0
        spot_available = -1*np.ones((n_grains_max, n_max_spots_per_grain), dtype=np.int32)
    else:
        j = np.nonzero(np.any(spot_available>=0, axis=1))[0]
        j = 0 if len(j)==0 else j[-1]+1
    # spot_available = -1*np.ones((n_grains_max, n_max_spots_per_grain), dtype=np.int32) if spot_available is None else spot_available

    n_spots_remaining = np.count_nonzero(spot_obs_assign<0)
    for i, gi in enumerate(grain_ids):

        select_best = np.s_[ordering[i]:ordering[i+1]]
        spotind_current = spotind_nn[select_best]
        diff_current = diff_best[select_best]
        select_inliers = diff_current<max_diff
        spotind_current_inliers = np.sort(spotind_current[select_inliers])
        frac_outliers = 1-np.count_nonzero(select_inliers)/len(select_inliers) if len(select_inliers)>0 else 1 
        condition_frac_outliers = frac_outliers<frac_max_outliers
        mad_current = np.median(diff_current)*1.4828
        condition_mad = True
        # condition_mad = mad_current < mad_max

        if j==0:
            i_max_overlap = n_max_overlap = 0 # so that verbosity works later

        else:

            set_corr = np.isin(spot_available[:j,:k], spotind_current_inliers, kind='table')
            n_overlap = np.sum(set_corr, axis=1)
            n_max_overlap = np.max(n_overlap)
            i_max_overlap = np.argmax(n_overlap)

        condition_spot_count = n_max_overlap/len(spotind_current) < frac_counts


        if LOGGER.level==10:
            LOGGER.debug(f'considering grain candidate {i+1:>4d}/{len(grain_ids)} {gi:>4d} loss={l_best[i]:8.4f} mad_current={mad_current: 4.2f}/{mad_max: 4.2f}, spots overlap with n_max_overlap={n_max_overlap:>4d}/{len(spotind_current)*frac_counts: 6.2f} (grain {i_max_overlap:>4d}), spots remaining {n_spots_remaining:>4d} frac_outliers={frac_outliers: 6.3f} / {frac_max_outliers: 6.3f}' )

        if condition_spot_count and condition_frac_outliers and condition_mad:

            spot_available[j,:len(spotind_current_inliers)] = spotind_current_inliers

            grain_accept.append(gi)
            n_grains_current = len(grain_accept)
            spot_obs_assign[spotind_current_inliers] = gi
            mod_assign = np.ones(len(spotind_current), dtype=int)*gi 
            mod_assign[~select_inliers] = FLAG_OUTLIER
            spot_mod_assign.append(mod_assign)

            # verbosity
            n_spots_remaining = np.count_nonzero(spot_obs_assign<0)
            msg = f'grain={n_grains_current+1:>5d} trial={n_checked_already+i+1:>6d}/{n_candidates:>6d} loss={l_best[i]:8.4f} mad={mad_current: 4.2f}/{mad_max: 4.2f} n_max_overlap={n_max_overlap:>4d}/{len(spotind_current)*frac_counts: 6.2f} n_spots_remaining={n_spots_remaining:>6d} {n_spots_remaining/len(spot_obs_assign)*100:4.2f}%  frac_outliers={frac_outliers: 6.3f}' 
            LOGGER.debug(msg)
            if ranger is not None:
                ranger.set_description(msg)
            else:
                if LOGGER.level==20:
                    sys.stdout.write('\r--> ' + msg)

            # increase counters
            j+=1
            k = max(k, len(spotind_current_inliers))

            if j>=n_grains_max:
                raise Exception(f'too many grains {j}')
            


    return spot_available

# @profile
def assign_grain_spot_heuristic_with_outliers(spotind_nn, diff_best, i_grn_best, s_obs, s_best, noise_sig, grain_accept=None, spot_available=None, spot_obs_assign=None, spot_mod_assign=None, i_grn_init=None, s_init=None, l_best=None, n_sig_out=3, patience=0, n_checked_already=0):

    n_spots_obs = len(s_obs) 
    spot_ids = np.arange(n_spots_obs)
    grain_ids = np.unique(i_grn_best)
    spot_available = np.ones(n_spots_obs, dtype=bool) if spot_available is None else spot_available
    spot_current = np.zeros(n_spots_obs, dtype=bool)
    pixel_size = 0.1
    frac_max_outliers = 1./3.
    frac_counts = 0.55
    max_diff = max(noise_sig, pixel_size)*n_sig_out
    l_best = np.zeros(len(s_best)) if l_best is None else l_best # init dummy
    ordering = np.searchsorted(i_grn_best[:,0], grain_ids)
    ordering = np.append(ordering, len(i_grn_best)) # add last element 

    time_start = time.time()
    for i, gi in enumerate(grain_ids):

        spot_current[:] = False
        select_best = np.s_[ordering[i]:ordering[i+1]]
        spotind_current = spotind_nn[select_best]
        # s_best_current = s_best[select_best]
        # s_obs_current = s_obs[spotind_current]
        # rms_current = np.sqrt(np.mean((s_best_current-s_obs_current)**2, axis=1))
        # dist_current = np.sum((s_best_current-s_obs_current)**2, axis=1)
        # select_inliers = rms_current<max_diff
        diff_current = diff_best[select_best]
        select_inliers = diff_current<max_diff
        spotind_current_inliers = spotind_current[select_inliers]
        frac_outliers = 1-np.count_nonzero(select_inliers)/len(select_inliers) if len(select_inliers)>0 else 1
        spot_current[spotind_current_inliers] = True
        spot_mark = spot_available & spot_current 
        n_spots_required = len(spotind_current)*frac_counts
        condition_frac_outliers = frac_outliers<frac_max_outliers
        condition_spot_count = np.count_nonzero(spot_mark) > n_spots_required

        if LOGGER.level==10:
            LOGGER.debug(f'considering grain candidate {i:>4d}/{len(grain_ids)} {gi:>4d} loss={l_best[i]:8.4f} spots marked {np.count_nonzero(spot_mark):>4d}/{int(n_spots_required):>4d} spots remaining {np.count_nonzero(spot_available):>4d} frac_outliers={frac_outliers: 6.3f} / {frac_max_outliers: 6.3f}' )

        if condition_spot_count and condition_frac_outliers:
            
            grain_accept.append(gi)
            
            spot_available[spot_mark] = False
            spot_obs_assign[spot_mark] = gi
            mod_assign = np.ones(len(spotind_current), dtype=int)*gi 
            mod_assign[~select_inliers] = FLAG_OUTLIER
            spot_mod_assign.append(mod_assign)

            msg = f'--> grain={len(grain_accept):>5d} trial={n_checked_already+i:>5d} loss={l_best[i]:8.4f} n_spots_marked={np.count_nonzero(spot_mark):>4d}/{n_spots_required:>4.1f} spots remaining {np.count_nonzero(spot_available):>8d} frac_outliers={frac_outliers: 6.3f} / {frac_max_outliers: 6.3f}' 
            LOGGER.debug(msg)
            if LOGGER.level==20:
                sys.stdout.write('\r' + msg)

    return spot_available

def assign_refine_sinkhorn(conf, a_init, x_init, s_target, nn_lookup_data, noise_sig, method='sinkhorn_unbalanced_sparse', test=False):


    from laueotx.polycrystalline_sample import polycrystalline_sample 
    from laueotx.spot_neighbor_lookup import spot_neighbor_lookup, nn_lookup
    from laueotx.deterministic_annealing import get_neighbour_stats, get_sinkhorn_optimization_control_params

    # some constants
    n_grn_sample = len(a_init)
    det_side_length =  conf['detectors'][0]['side_length']
    n_iter_annealing = 1000

    LOGGER.info(f'=============> refining the solution using double assignment with method={method}')
    LOGGER.info(f'generating model spots for {n_grn_sample} initialization grains')
    sample_trials = polycrystalline_sample(conf=conf, hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=False)
    sample_trials.set_tensor_variables(a_init, x_init)
    i_grn, i_ang, i_hkl, i_det, i_all = sample_trials.get_batch_indices_full(n_grn=n_grn_sample)

    max_nn_dist, n_spots_per_grain = get_neighbour_stats(sample_trials, 
                                                         inds_trials=(i_grn[...,0], i_ang[...,0], i_det[...,0]), 
                                                         lookup_data=nn_lookup_data, 
                                                         s_target=s_target)

    n_iter_annealing, control_params = get_sinkhorn_optimization_control_params(conf=conf, 
                                                                                max_nn_dist=max_nn_dist,
                                                                                n_target=len(s_target),
                                                                                n_model=len(a_init)*n_spots_per_grain)

    a_fit_global, x_fit_global, q_fit, inds_model, loss_global, s_global, aux_info = laue_math_graph.optimize_global(s_target=s_target,
                                                                                                                      nn_lookup_data=nn_lookup_data,
                                                                                                                      indices_mesh=(i_grn, i_ang, i_hkl, i_det, i_all),
                                                                                                                      params_grain=(sample_trials.U, sample_trials.x0, sample_trials.v),
                                                                                                                      params_experiment=(sample_trials.Gamma, sample_trials.dn, sample_trials.d0, sample_trials.dr, sample_trials.dl, sample_trials.dh, sample_trials.lam),
                                                                                                                      n_iter=n_iter_annealing, 
                                                                                                                      verb=True, 
                                                                                                                      test=test,
                                                                                                                      control_params=control_params,
                                                                                                                      assignment_method=method)

    if len(a_fit_global)==0:
        raise Exception('Did not find any grains, aborting')
    else:
        LOGGER.info(f'final number of grains {len(a_fit_global)}')

    s_global_2d = intersect_detectors(s_global, i_det=inds_model[2], detectors_specs=conf['detectors'])

    loss_global = np.array(loss_global).T
    a_fit_global = Rotation.from_matrix(a_fit_global.numpy()).as_mrp()

    loss_init = np.mean(loss_global[:,0])
    loss_opt = np.mean(loss_global[:,-2])
    loss_init_median = np.median(loss_global[:,0])
    loss_opt_median = np.median(loss_global[:,-2])
    LOGGER.info('loss (mean)     = {: 12.6f} -> {: 12.6f}'.format(loss_init, loss_opt))
    LOGGER.info('loss (median)   = {: 12.6f} -> {: 12.6f}'.format(loss_init_median, loss_opt_median))

    dict_out = {'i_target': aux_info[0],
                'a_fit_global': a_fit_global,
                'x_fit_global': x_fit_global,
                'q_fit':q_fit,
                'loss_global': loss_global,
                's_global': s_global,
                's_global_2d': s_global_2d,
                'inds_model': inds_model,
                'l2_fit': aux_info[-1],
                'aux_info': aux_info}

    return dict_out


def intersect_detectors(s, i_det, detectors_specs):
    
    s = np.array(s)
    tau = np.linalg.norm(s, axis=-1)
    s_scale = np.array(s).copy()

    for d in detectors_specs:

        select = d['id'] == i_det[0,:,0]
        det_distance = d['position'][0]
        scale = det_distance/s[select,:,0]
        tau_scale = np.abs(scale*tau[select])
        s_scale[select] = s[select] * tf.expand_dims(tau_scale/tau[select], axis=-1)

    return s_scale


def array_neighbours_to_sparse(Q, spot_ind, n_target):

    Q = np.array(Q)
    spot_ind = np.array(spot_ind)
    i_rep = np.repeat(np.arange(len(Q)), spot_ind.shape[1])

    # add extra entry in case it's not present in spot_ind
    Q_extend = np.concatenate([Q.ravel(), [0]])
    i_extend = np.concatenate([i_rep, [len(Q)-1]])
    j_extend = np.concatenate([spot_ind.ravel(), [n_target-1]])
    
    from scipy.sparse import coo_array
    Q_coo = coo_array( (Q_extend, (i_extend, j_extend)) )

    return Q_coo


def get_softassign_spot_matching(Qs, i_grn_mod, l2, noise_sig=1, n_sig_out=3):
    
    FLAG_OUTLIER = -999999
    # spotind_accept_i = np.array(np.argmax(Qs, axis=1))[:,0]
    # spotind_accept_q = np.array(np.max(Qs, axis=1).todense())[:,0]

    # thresh= np.quantile(Qs.data, 1-np.mean(Qs.shape)/len(Qs.data))
    # try:
    #     thresh= np.quantile(Qs.data, 1-np.sum(Qs.shape)/len(Qs.data))
    # except:
    #     thresh = np.max(Qs.data)/1e10
    thresh = n_sig_out*noise_sig

    uc = np.array(np.argmax(Qs, axis=0))
    ur = np.array(np.argmax(Qs, axis=1))[:,0]

    l2_csr = l2.tocsr()
    l2_csr.eliminate_zeros()

    select_thresh_r = l2_csr[np.arange(Qs.shape[0]), ur] < thresh
    select_thresh_c = l2_csr[uc, np.arange(Qs.shape[1])] < thresh

    
    # qc = np.array(np.max(Qs, axis=0).todense())[0,:]
    # qr = np.array(np.max(Qs, axis=1).todense())[:,0]
    select_unique_matches_r = (uc[ur] == np.arange(len(ur))) #& (qr<thresh)
    select_unique_matches_c = (ur[uc] == np.arange(len(uc))) #& (qc<thresh)
    LOGGER.info('softassign matching, Qs.shape={} thresh={:2.4e} found assignment matrix entries with unique matches: mod={}/{} obs={}/{}'.format(Qs.shape, thresh, np.count_nonzero(select_unique_matches_r),len(select_unique_matches_r), np.count_nonzero(select_unique_matches_c), len(select_unique_matches_c)))

    spot_obs_assign = i_grn_mod[uc]
    spot_mod_assign = np.array(i_grn_mod).copy()
    spot_obs_assign[ (~select_unique_matches_c) | (~select_thresh_c) ] = FLAG_OUTLIER
    spot_mod_assign[ (~select_unique_matches_r) | (~select_thresh_r) ] = FLAG_OUTLIER

    s2s_obs_assign = uc.copy()
    s2s_mod_assign = ur.copy()
    s2s_obs_assign[ (~select_unique_matches_c) | (~select_thresh_c) ] = FLAG_OUTLIER
    s2s_mod_assign[ (~select_unique_matches_r) | (~select_thresh_r) ] = FLAG_OUTLIER

    delta_inliers = l2_csr[np.arange(Qs.shape[0]), ur][select_thresh_r]
    LOGGER.info('Fitted inliers spots: std    delta {:2.4f} mm'.format(np.std(delta_inliers)))
    LOGGER.info('Fitted inliers spots: median delta {:2.4f} mm'.format(np.median(delta_inliers)))

    return spot_obs_assign, spot_mod_assign, s2s_obs_assign, s2s_mod_assign


def assign_grain_spot_heuristic(spotind_nn, i_grn_best, s_obs, s_best, i_grn_init=None, s_init=None):

    n_spots = len(s_obs) 
    spot_ids = np.arange(n_spots)
    grain_ids = np.unique(i_grn_best)
    spot_current = np.zeros(n_spots, dtype=bool)
    n_marked_ignore = 10
    grain_accept = []
    spot_assign = np.ones(n_spots, dtype=int)*FLAG_OUTLIER
    max_diff = 2

    for i in range(len(grain_ids)):

        spot_current[:] = False
        select_best = i_grn_best == i
        # select_init = i_grn_init == i
        spotind_current = spotind_nn[select_best[:,0]]
        spot_current[spotind_current] = True
        spot_mark = spot_available & spot_current

        s_best_current = s_best[select_best[:,0]]
        # s_init_current = s_init[select_init[:,0]]
        s_obs_current = s_obs[spotind_current]

        max_diff_current = np.max(np.sqrt(np.mean((s_best_current-s_obs_current)**2, axis=1)))

        # s_init_unique = np.unique(np.round(s_init_current[:,1:], 3), axis=0)
        # s_best_unique = np.unique(np.round(s_best_current[:,1:], 3), axis=0)

        condition_spot_count = np.count_nonzero(spot_mark) > (np.count_nonzero(spot_current) - n_marked_ignore)
        condition_max_diff = max_diff_current < max_diff
        # condition_nspots_start_end = len(s_init_unique) == len(s_best_unique)

        LOGGER.debug(f'considering grain candidate {i:>4d} spots marked {len(spotind_current):>4d} spots remaining {np.count_nonzero(spot_available):>4d} max_diff={max_diff_current: 6.3f} n_best={len(s_best_current):>4d}' )
        # LOGGER.debug(f'considering grain candidate {i:>4d} spots marked {len(spotind_current):>4d} spots remaining {np.count_nonzero(spot_available):>4d} max_diff={max_diff_current: 6.3f} n_init={len(s_init_unique):>4d} n_best={len(s_best_unique):>4d}' )

        # if condition_spot_count and condition_max_diff and condition_nspots_start_end:
        if condition_spot_count and condition_max_diff:
            
            grain_accept.append(i)
            
            spot_available[spot_current] = False
            spot_assign[spot_current] = i

            LOGGER.info(f'--> accepted grain={len(grain_accept):>5d} trial={i:>5d} n_spots_marked={len(spotind_current):>8d} n_spots_remaining={np.count_nonzero(spot_available):>8d} max_diff={max_diff_current: 2.3f}' )

    grain_accept = np.array(grain_accept)
    return grain_accept, spot_assign
