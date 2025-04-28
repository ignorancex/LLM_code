import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
import numpy as np

from laueotx.laue_math_tensorised import fast_full_forward, fast_spot_select, fast_full_forward_lab
from laueotx.spot_neighbor_lookup import nn_lookup, nn_lookup_all
from laueotx.laue_coordinate_descent import batch_get_u, batch_get_r 
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_baseline
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_streamlined
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_hardassign
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_softassign
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_softlowmem
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_softent
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_softpart
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_softslacks
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_sinkhorn_balanced_full
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_sinkhorn_balanced_sparse
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_sinkhorn_unbalanced_full
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_sinkhorn_unbalanced_sparse
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_sinkhorn_partial_full
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_sinkhorn_partial_sparse
from laueotx.laue_coordinate_descent import batch_optimize_coordinate_descent_sinkhorn_slacks_sparse


global_optimizer = {'balanced':          batch_optimize_coordinate_descent_sinkhorn_balanced_full, 
                    'balanced_full':     batch_optimize_coordinate_descent_sinkhorn_balanced_sparse,
                    'unbalanced_full':   batch_optimize_coordinate_descent_sinkhorn_unbalanced_full,
                    'unbalanced':        batch_optimize_coordinate_descent_sinkhorn_unbalanced_sparse,        
                    'partial_full':      batch_optimize_coordinate_descent_sinkhorn_partial_full,
                    'partial':           batch_optimize_coordinate_descent_sinkhorn_partial_sparse,
                    'softslacks':        batch_optimize_coordinate_descent_sinkhorn_slacks_sparse}

local_optimizer = {'baseline'    : batch_optimize_coordinate_descent_baseline,
                   'streamlined' : batch_optimize_coordinate_descent_streamlined,
                   'hardassign'  : batch_optimize_coordinate_descent_hardassign,
                   'softassign'  : batch_optimize_coordinate_descent_softassign,
                   'softlowmem'  : batch_optimize_coordinate_descent_softlowmem,
                   'softent'     : batch_optimize_coordinate_descent_softent,
                   'softslacks'  : batch_optimize_coordinate_descent_softslacks,
                   'softpart'    : batch_optimize_coordinate_descent_softpart}


@tf.function
def pad_zeros_along_axis(x, n, axis=0):

    return tf.concat([x, tf.zeros([n, x.shape[1]],  dtype=x.dtype)], axis=axis)


@tf.function
def pad_arrays_to_equal_size(x, y):

    if len(x) > len(y):

        n = len(x)-len(y)
        y = pad_zeros_along_axis(y, n, axis=0)
        print(f'padded y with {n} zeros x.shape={x.shape} y.shape={y.shape}')

    elif len(x) < len(y):

        n = len(y)-len(x)
        x = pad_zeros_along_axis(x, n, axis=0)
        print(f'padded x with {n} zeros x.shape={x.shape} y.shape={y.shape}')

    return x, y


def optimize_global(s_target, nn_lookup_data, indices_mesh, params_grain, params_experiment, assignment_method, control_params, i_join=None, n_iter=2, verb=False, test=False):

    from laueotx.polycrystalline_sample import merge_duplicated_spots, apply_selections

    # unpack input
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = nn_lookup_data
    i_grn, i_ang, i_hkl, i_det, i_all = indices_mesh
    U, x0, v = params_grain
    Gamma, dn, d0, dr, dl, dh, lam = params_experiment
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0], dtype=tf.float64)
    n_grn, n_ang, n_hkl, n_det, _ = i_grn.shape

    # convert the basis into unit vectors
    v_unit = v/tf.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v_unit[...,0]

    # generate the full model spot set
    s_lab, p_lab, p_lam, select = fast_full_forward_lab(U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I_eye)

    # cheat for now - use only joined set
    if i_join is not None:

        print(f'using spot set intersection for testing, len(i_join)={len(i_join)}')
        select = np.in1d(i_all, i_join)
        s_lab_select = tf.reshape(s_lab, (-1, 3))[select]
        i_grn_select = tf.expand_dims(tf.reshape(i_grn, (-1,))[select], axis=-1)
        i_ang_select = tf.expand_dims(tf.reshape(i_ang, (-1,))[select], axis=-1)
        i_det_select = tf.expand_dims(tf.reshape(i_det, (-1,))[select], axis=-1)
        i_hkl_select = tf.reshape(i_hkl, (-1,1))[select]
        i_all_select = tf.reshape(i_all, (-1,1))[select]
        v_lab_select = tf.gather(v_unit, i_hkl_select[:,0])
        i_gpl_select = i_grn_select*n_hkl + i_hkl_select # grain-plane unique index


    else:

        s_lab_select, i_grn_select, i_ang_select, i_det_select, i_hkl_select, i_all_select = apply_selections(select, s_lab, i_grn, i_ang, i_det, i_hkl, i_all)
        v_lab_select = tf.gather(v_unit, i_hkl_select[:,0])
        s_lab_select, i_grn_select, i_ang_select, i_hkl_select, i_det_select, i_all_select, select_merged = merge_duplicated_spots(s_lab_select, i_grn_select, i_ang_select, i_hkl_select, i_det_select, i_all_select, decimals=4, return_index=True)
        v_lab_select = tf.gather(v_lab_select, select_merged)
        i_gpl_select = i_grn_select*n_hkl + i_hkl_select # grain-plane unique index


    # get the nearest neighbour in the measured data
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s_lab_select, i_ang_select, i_det_select, i_grn_select, nn_lookup_pix_size, nn_lookup_n_pix)
    s_match = tf.gather(s_target, i_target)

    # get initial varibles for optimization
    g_select = tf.gather(Gamma, i_ang_select[:,0])
    a_init = tf.gather(U, i_grn_select[:,0])
    x_init = tf.gather(x0, i_grn_select[:,0])

    # run optimization
    fun_opt = global_optimizer[assignment_method]
    print(f'Using global optimization method={assignment_method} function={fun_opt}')
    loss, x_fit, a_fit, q_fit, select_valid, s_full, aux_info = fun_opt(p=s_match,
                                                                        a=a_init,
                                                                        x=x_init,
                                                                        s=s_lab_select,
                                                                        v=v_lab_select,
                                                                        g=g_select,
                                                                        segment_ids=i_grn_select[:,0], 
                                                                        s_target=s_target,
                                                                        inds_model=(i_grn_select, i_ang_select, i_hkl_select, i_det_select, i_gpl_select, n_hkl),
                                                                        lookup_data=(nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix), 
                                                                        n_iter=n_iter, 
                                                                        control_params=control_params,
                                                                        verb=False,
                                                                        test=test)

    # get l2 loss between spots
    l2_nn = tf.math.sqrt(tf.reduce_sum((s_match - s_full)**2, axis=-1))
    aux_info += (l2_nn,)

    # get the final grain selection
    uv, i_grn_valid = np.unique(i_grn_select[select_valid], return_inverse=True)
    inds_model = tf.constant(i_grn_valid, dtype=tf.int32)[:,tf.newaxis], i_ang_select[select_valid], i_det_select[select_valid], i_hkl_select[select_valid], i_all_select[select_valid], v_lab_select[select_valid]

    return a_fit, x_fit, q_fit, inds_model, loss, s_full, aux_info


@tf.function
def batch_optimize_grain(n_per_batch, s_target, nn_lookup_data, indices_mesh, params_grain, params_experiment, n_iter=256, control_params=None, verb=False, opt_fun='baseline'):
    # unpack input
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = nn_lookup_data
    i_grn, i_ang, i_hkl, i_det = indices_mesh
    U, x0, v = params_grain
    Gamma, dn, d0, dr, dl, dh, lam, noise_sig = params_experiment
    v_unit = v/tf.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v_unit[...,0]
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0.], dtype=tf.float64)

    dataset = tf.data.Dataset.from_tensor_slices((U, x0, dn, d0, dr))
    dataset = dataset.batch(batch_size=n_per_batch).prefetch(buffer_size=2)
    dataset = dataset.repeat()
    dataset = iter(dataset)

    n_batches = int(np.ceil(U.shape[0]/n_per_batch))
   
    list_x = []
    list_a = []
    list_loss = []
    list_nspot = []
    list_fracin = []

    # for softassign_lowmem
    # import pudb; pudb.set_trace();
    # pass
    # already selected ...
    # i_unique_grn_hkl, segment_ids_unique_grn_hkl = np.unique(np.vstack([segment_ids, i_hkl]).T, return_inverse=True, axis=0)
    # seg_ones = tf.ones(len(p), dtype=p.dtype)
    # v_red = tf.math.unsorted_segment_mean(v, segment_ids=segment_ids_unique_grn_hkl, num_segments=len(i_unique_grn_hkl))
    # 
    # for softassign_lowmem
    n_grn, n_ang, n_hkl, n_det = i_grn.shape
    i_gpl = i_grn*n_hkl + i_hkl # grain-plane unique index
    # i_unique_grn_hkl, segment_ids_unique_grn_hkl = np.unique(, return_inverse=True, axis=0)
    # v_red = tf.gather(v_unit, i_unique_grn_hkl[:,1])




    # i_unique_grn_hkl, segment_ids_unique_grn_hkl = np.unique(np.vstack([segment_ids, i_hkl]).T, return_inverse=True, axis=0)
    # i_unique_grn_hkl, segment_ids_unique_grn_hkl = np.unique(i_grn_hkl, return_inverse=True, axis=0)
    # v_red = tf.gather(v_unit, i_unique_grn_hkl[:,1])


    for i in range(n_batches):

        if verb:
            print(f'=================> batch {i}/{n_batches}')

        # get the next batch of parameters
        U_, x0_, dn_, d0_, dr_ = next(dataset)

        # get the forward model prediction
        s_lab, p_lab, p_lam, select = fast_full_forward_lab(U_, x0_, Gamma, v, dn_, d0_, dr_, dl, dh, lam, I_eye)

        # select spots in the detector and wavelength range
        s_lab_select = s_lab[select]

        # get indices of these spots
        i_grn_select = tf.expand_dims(i_grn[select], axis=-1)
        i_ang_select = tf.expand_dims(i_ang[select], axis=-1)
        i_det_select = tf.expand_dims(i_det[select], axis=-1)
        i_hkl_select = i_hkl[select]
        i_gpl_select = i_gpl[select]

        # get the lattice vectors for each selected spot

        # get the nearest neighbour in the measured data
        # i_target = nn_lookup(nn_lookup_spot_ind, s_target, s_lab_select, i_ang_select, i_det_select, i_grn_select, nn_lookup_pix_size, nn_lookup_n_pix)
        # s_match = tf.gather(s_target, i_target)

        # if i == 0:
        #     from fastlaue3dnd import utils_io
        #     p_lam_select = np.tile(np.expand_dims(p_lam, axis=-1), [1, 1, 1, 2])[select]
        #     utils_io.write_arrays(f'info_batch{i}.h5', 'w', s_lab_select, i_ang_select, i_det_select, i_grn_select, p_lam_select)
        #     raise Exception('debugging, goodbye!')

        # if verb:
        #     n_target = len(s_target)
        #     n_compare = min(len(s_target), len(i_target))
        #     n_match = np.count_nonzero(i_target.numpy()[:n_compare]==np.arange(n_target)[:n_compare])
        #     print(f'match n_target={len(s_target)} n_model={len(s_lab_select)} {n_match}/{n_target} {n_match/n_target}')

        # get initial variales for optimization
        g_select = tf.gather(Gamma, i_ang_select[:,0])
        a_init = tf.gather(U_, i_grn_select[:,0])
        x_init = tf.gather(x0_, i_grn_select[:,0])
        v_lab_select = tf.gather(v_unit, i_hkl_select)
        w_init = tf.einsum('bij, bj -> bi', a_init, v_lab_select)
        r_init = batch_get_r(s_lab_select, w_init, x_init, g_select, e, I_eye)

        if opt_fun == 'baseline':

            i_target = nn_lookup(nn_lookup_spot_ind, s_target, s_lab_select, i_ang_select, i_det_select, i_grn_select, nn_lookup_pix_size, nn_lookup_n_pix)
            s_match = tf.gather(s_target, i_target)
            loss, fracin, x_fit, a_fit = batch_optimize_coordinate_descent(p=s_match,
                                                a=a_init,
                                                x=x_init,
                                                r=r_init,
                                                v=v_lab_select,
                                                g=g_select,
                                                e=e, 
                                                I=I_eye, 
                                                segment_ids=i_grn_select[:,0], 
                                                n_iter=n_iter, 
                                                verb=verb)

        elif opt_fun == 'streamlined':

            i_target = nn_lookup(nn_lookup_spot_ind, s_target, s_lab_select, i_ang_select, i_det_select, i_grn_select, nn_lookup_pix_size, nn_lookup_n_pix)
            s_match = tf.gather(s_target, i_target)
            loss, fracin, x_fit, a_fit = batch_optimize_coordinate_descent_streamlined(p=s_match,
                                                a=a_init,
                                                x=x_init,
                                                r=r_init,
                                                v=v_lab_select,
                                                g=g_select,
                                                e=e, 
                                                I=I_eye, 
                                                segment_ids=i_grn_select[:,0], 
                                                n_iter=n_iter, 
                                                verb=verb)
            
        elif opt_fun == 'hardassign':

            i_target = nn_lookup(nn_lookup_spot_ind, s_target, s_lab_select, i_ang_select, i_det_select, i_grn_select, nn_lookup_pix_size, nn_lookup_n_pix)
            s_match = tf.gather(s_target, i_target)
            loss, fracin, x_fit, a_fit = batch_optimize_coordinate_descent_hardassign(p=s_match,
                                                a=a_init,
                                                x=x_init,
                                                r=r_init,
                                                v=v_lab_select,
                                                g=g_select,
                                                e=e, 
                                                I=I_eye, 
                                                segment_ids=i_grn_select[:,0], 
                                                n_iter=n_iter, 
                                                verb=verb,
                                                nn_lookup_spot_ind=nn_lookup_spot_ind,
                                                s_target=s_target,
                                                i_ang=i_ang_select,
                                                i_det=i_det_select,
                                                i_grn=i_grn_select,
                                                nn_lookup_pix_size=nn_lookup_pix_size,
                                                nn_lookup_n_pix=nn_lookup_n_pix)

        elif 'soft' in opt_fun:

            loss, fracin, x_fit, a_fit = local_optimizer[opt_fun](a=a_init,
                                                                  x=x_init,
                                                                  s=s_lab_select,
                                                                  v=v_lab_select,
                                                                  g=g_select,
                                                                  s_target=s_target,
                                                                  consts=(e, I_eye),
                                                                  n_iter=n_iter, 
                                                                  verb=verb,
                                                                  inds_model=(i_grn_select, i_ang_select, i_hkl_select, i_det_select, i_gpl_select, n_hkl),
                                                                  lookup_data=(nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix),
                                                                  control_params=control_params)            

        nspot = tf.math.segment_sum(tf.ones(len(i_grn_select)), segment_ids=i_grn_select[:,0])


        # s_lab, p_lab, p_lam, select = fast_full_forward_lab(a_fit, x_fit, Gamma, v, dn_, d0_, dr_, dl, dh, lam, I_eye)
        # s_lab_select = s_lab[select]
        # i_grn_select = tf.expand_dims(i_grn[select], axis=-1)
        # i_ang_select = tf.expand_dims(i_ang[select], axis=-1)
        # i_det_select = tf.expand_dims(i_det[select], axis=-1)
        # i_hkl_select = i_hkl[select]
        # i_gpl_select = i_gpl[select]
        # i_target = nn_lookup(nn_lookup_spot_ind, s_target, s_lab_select, i_ang_select, i_det_select, i_grn_select, nn_lookup_pix_size, nn_lookup_n_pix)
        # s_match = tf.gather(s_target, i_target)
        # l2 = tf.linalg.norm(s_lab_select-s_match, axis=1)
        # list_l2 = []
        # for i in range(n_grn):
        #     select = i_grn_select[:,0]==i
        #     list_l2.append(np.median(l2[select]))
        # list_l2 = np.array(list_l2)
        # l2_seg = tf.math.segment_mean(l2, i_grn_select[:,0])

        list_x.append(x_fit)
        list_a.append(a_fit)
        list_loss.append(loss)
        list_nspot.append(nspot)
        list_fracin.append(fracin)

    if len(list_a)>1:
        a_fit = tf.concat(list_a, axis=0)
        x_fit = tf.concat(list_x, axis=0)
        loss = tf.concat(list_loss, axis=0)
        nspot = tf.concat(list_nspot, axis=0) 
        fracin = tf.concat(list_fracin, axis=0) 


    return a_fit, x_fit, loss, nspot, fracin


@tf.function
def batch_spots_select_lab(n_per_batch, U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I):

    dataset = tf.data.Dataset.from_tensor_slices((U, x0, dn, d0, dr))
    dataset = dataset.batch(batch_size=n_per_batch).prefetch(buffer_size=2)
    dataset = iter(dataset)

    n_grains = U.shape[0]
    n_batches = int(np.ceil(n_grains/n_per_batch))

    list_s_lab = []
    list_p_lab = []
    list_p_lam = []

    for i in range(n_batches):

        U_, x0_, dn_, d0_, dr_ = next(dataset)

        s_lab, p_lab, p_lam, select = fast_full_forward_lab(U_, x0_, Gamma, v, dn_, d0_, dr_, dl, dh, lam, I)

        list_s_lab.append(s_lab[select])
        list_p_lab.append(p_lab[select])
        list_p_lam.append(p_lam[select])

    s_lab = tf.concat(list_s_lab, axis=0)
    p_lab = tf.concat(list_p_lab, axis=0)
    p_lam =  tf.concat(list_p_lam, axis=0)

    return s_lab, p_lab, p_lam

@tf.function
def batch_spots_lab(n_per_batch, U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I):

    dataset = tf.data.Dataset.from_tensor_slices((U, x0, dn, d0, dr))
    dataset = dataset.batch(batch_size=n_per_batch).prefetch(buffer_size=2)
    dataset = iter(dataset)

    n_grains = U.shape[0]
    n_batches = int(np.ceil(n_grains/n_per_batch))

    list_s_lab = []
    list_p_lab = []
    list_p_lam = []
    list_select = []

    for i in range(n_batches):

        U_, x0_, dn_, d0_, dr_ = next(dataset)

        s_lab, p_lab, p_lam, select = fast_full_forward_lab(U_, x0_, Gamma, v, dn_, d0_, dr_, dl, dh, lam, I)

        list_s_lab.append(s_lab)
        list_p_lab.append(p_lab)
        list_p_lam.append(p_lam)
        list_select.append(select)

    s_lab = tf.concat(list_s_lab, axis=0)
    p_lab = tf.concat(list_p_lab, axis=0)
    p_lam =  tf.concat(list_p_lam, axis=0)
    select = tf.concat(list_select, axis=0)

    return s_lab, p_lab, p_lam, select



@tf.function
def batch_spots(n_per_batch, U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I):

    dataset = tf.data.Dataset.from_tensor_slices((U, x0, dn, d0, dr))
    dataset = dataset.batch(batch_size=n_per_batch).prefetch(buffer_size=2)
    dataset = iter(dataset)

    n_grains = U.shape[0]
    n_batches = int(np.ceil(n_grains/n_per_batch))

    list_s_spot = []
    list_p_spot = []
    list_p_lam = []
    list_select = []

    for i in range(n_batches):

        U_, x0_, dn_, d0_, dr_ = next(dataset)

        s_spot, p_spot, p_lam = fast_full_forward(U_, x0_, Gamma, v, dn_, d0_, dr_, I)
        select = fast_spot_select(p_spot, s_spot, p_lam, lam, dn_, dl, dh)

        list_s_spot.append(s_spot)
        list_p_spot.append(p_spot)
        list_p_lam.append(p_lam)
        list_select.append(select)

    s_spot = tf.concat(list_s_spot, axis=0)
    p_spot = tf.concat(list_p_spot, axis=0)
    p_lam =  tf.concat(list_p_lam, axis=0)
    select = tf.concat(list_select, axis=0)

    return s_spot, p_spot, p_lam, select


@tf.function
def render_image_cube(s_spot, select, i_ang, i_det, n_ang, n_det, n_pix, pixel_size):

    img_cube = np.zeros((n_ang, n_det, n_pix, n_pix), dtype=np.float32)
    s_spot_select = s_spot[select]
    i_det_select = i_det[select] 
    i_ang_select = i_ang[select]

    indices = s_spot_select[:,1:]/pixel_size
    indices = tf.cast(indices, tf.int32) + n_pix//2
    i_ang_select = tf.cast(i_ang_select, tf.int32)
    i_det_select = tf.cast(i_det_select, tf.int32)
    indices = tf.concat([i_ang_select, i_det_select, indices], axis=-1)
    updates = tf.ones(indices.shape[0], dtype=tf.float32)
    img_cube = tf.tensor_scatter_nd_add(img_cube, indices, updates)

    return img_cube



@tf.function
def add_spots_to_image(img, s_spot, select, i_ang, i_det, n_pix, pixel_size, angle_size, s_weight):

    # select the spots to histogram
    s_spot_select = s_spot[select]
    i_det_select = i_det[select] 
    i_ang_select = i_ang[select]
    

    s_weight_select = s_weight[select]

    # calculate pixel index
    i_pos_select = s_spot_select[:,1:]/pixel_size
    i_pos_select = tf.cast(i_pos_select, tf.int64) + n_pix//2


    # calculate angle index
    i_ang_select = i_ang_select/angle_size
    i_ang_select = tf.cast(i_ang_select, tf.int64)

    # get detector index
    i_det_select = tf.cast(i_det_select, tf.int64)

    # variables to scatter
    indices = tf.concat([i_ang_select, i_det_select, i_pos_select], axis=-1)
    updates = tf.ones(indices.shape[0], dtype=tf.float64)*s_weight_select
    
    img = tf.tensor_scatter_nd_add(img, indices, updates)

    return img
    
@tf.function
def render_image_cube_batch(img, n_per_batch, pixel_size, angle_size, i_ang, i_det, beam_spectrum, U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I):

    spec_lam, spec_val = beam_spectrum
    delta_lam = spec_lam[1]-spec_lam[0]
    n_ang, n_det, n_pix = img.shape[:-1]

    dataset = tf.data.Dataset.from_tensor_slices((U, x0, dn, d0, dr))
    dataset = dataset.batch(batch_size=n_per_batch).prefetch(buffer_size=2)
    dataset = iter(dataset)

    n_batches = int(np.ceil(U.shape[0]/n_per_batch))

    # from tqdm.auto import trange
    for i in range(n_batches):

        U_, x0_, dn_, d0_, dr_ = next(dataset)

        s_spot, p_spot, p_lam = fast_full_forward(U_, x0_, Gamma, v, dn_, d0_, dr_, I)
        select = fast_spot_select(p_spot, s_spot, p_lam, lam, dn_, dl, dh)

        spec_lam = tf.gather(spec_val, indices=tf.cast(tf.math.abs(p_lam)/delta_lam, tf.int32)) # nearest neighbour interp
        s_weight = tf.tile(tf.expand_dims(spec_lam * p_lam**4.0, axis=-1), multiples=[1,1,1,2])

        img = add_spots_to_image(img, s_spot, select, i_ang, i_det, n_pix, pixel_size, angle_size, s_weight=s_weight)

    return img



# def render_segmentation_image_cube_batch(img, pixel_size, angle_size, i_ang, i_det, i_grn, U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I):

#     # img_cube = np.zeros((n_ang, n_det, n_pix, n_pix), dtype=np.float32)
#     n_ang, n_det, n_pix = img.shape[:-1]


#     dataset = tf.data.Dataset.from_tensor_slices((U, x0, dn, d0, dr))
#     dataset = dataset.batch(batch_size=1).prefetch(buffer_size=2)
#     dataset = iter(dataset)

#     from tqdm.auto import trange
#     for i in trange(U.shape[0]):

#         img_temp = tf.zeros_like(img)

#         U_, x0_, dn_, d0_, dr_ = next(dataset)

#         s_spot, p_spot, p_lam = fast_full_forward(U_, x0_, Gamma, v, dn_, d0_, dr_, I)
#         select = fast_spot_select(p_spot, s_spot, p_lam, lam, dn_, dl, dh)
#         s_weight = tf.zeros_like(s_spot) + i + 1
#         img_temp = add_spots_to_image(img_temp, s_spot, select, i_ang, i_det, n_pix, pixel_size, angle_size, s_weight=s_weight)

#         # img = tf.where((img>0)  & (img_temp>0), -1, img)
#         # img = tf.where((img==0) & (img_temp>0), img_temp, img)

#     return img

