import numpy as np
from tqdm.auto import trange
import tensorflow as tf
from laueotx.spot_neighbor_lookup import nn_lookup, nn_lookup_dist, nn_lookup_all
from laueotx import optimal_transport
from laueotx.config import TF_FUNCTION_JIT_COMPILE
from laueotx.laue_ops import *
import time

astr = lambda x: np.array2string(np.array(x), max_line_width=1000, precision=5, formatter={'all': lambda x: '{: 2.6f}'.format(x)})

##############################################
#
# Tensorflow version of functions 
#
##############################################

@tf.function
def batch_robust_sign(a):
    
    s = tf.math.sign(a)
    s = tf.where(s==0, 1, s)
    s = tf.where(tf.math.abs(a)<1e-10, 1, s)
    return s

@tf.function()
def batch_align_sign(u, w):

    u = tf.expand_dims(u, axis=0)
    u = tf.concat([u, -u], axis=0)
    d = tf.expand_dims(tf.reduce_sum((u - w)**2, axis=-1), axis=-1)
    u = tf.where(d[0]<d[1], u[0], u[1])

    return u


@tf.function()
def batch_get_u_signfix(p, x, w, g, e):

    # specular reflection solution
    u = batch_get_u(p, x, g, e)
    
    # select the right sign with respect to the basis w
    u = batch_align_sign(u, w)
    
    return u

@tf.function
def batch_get_u(p, x, g, e):
    """Summary
    
    Parameters
    ----------
    p : TYPE
        spots from multiple grains
    x : TYPE
        grain position (per spot)
    g : TYPE
        rotation matrices for sample tomographic rotation (per spot)
    e : TYPE
        incident beam vector [1, 0, 0]
    """

    x_rot = tf.einsum('sij, sj -> si', g, x)
    p_bar = p - x_rot

    # eigenvalue solution
    # M = tf.einsum('sji, sj, k, skn -> sin', g, p_bar, e, g )
    # Mt = (M + tf.transpose(M, perm=[0,2,1]))/2.
    # Lam, Q = tf.linalg.eigh(Mt)
    # u = Q[...,0]

    # Rhombohedron solution
    p_norm = p_bar/tf.linalg.norm(p_bar, axis=1, keepdims=True)
    u = p_norm - e
    u /= tf.linalg.norm(u, axis=1, keepdims=True)
    u = tf.einsum('sji, sj -> si', g, u) # inverse rotation

    # test
    # M_np = []
    # Q_np = []
    # for i in range(len(p)):
    #     M_ = np.linalg.multi_dot([g[i].numpy().T, np.atleast_2d(p_bar[i]).T, np.atleast_2d(e), g[i].numpy()])
    #     Mt = (M_+M_.T)/2
    #     Lam_, Q_ = np.linalg.eigh(Mt)
    #     M_np.append(M_)
    #     Q_np.append(Q_)
    # M_np = np.array(M_np)
    # Q_np = np.array(Q_np)

    return u


@tf.function
def batch_get_r(p, u, x, g, e, I):
    """Summary
    
    Parameters
    ----------
    p : TYPE
        Description
    u : TYPE
        Description
    x : TYPE
        Description
    g : TYPE
        Description
    e : TYPE
        Description
    I : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    x_rot = tf.einsum('sij, sj -> si', g, x)
    p_bar = p - x_rot
    w = tf.einsum('sij, sj -> si', g, u)
    H = I - 2*tf.einsum('si,sj->sij', w, w) # Householder matrix, specular reflection
    b = tf.einsum('sij,j -> si', H, e)
    r = tf.einsum('si, si -> s', p_bar, b)/tf.einsum('si, si -> s', b, b)

    return r

@tf.function
def batch_get_x(p, u, r, g, e, I_eye, i):
    """Summary
    
    Parameters
    ----------
    p : TYPE
        Description
    u : TYPE
        Description
    r : TYPE
        Description
    g : TYPE
        Description
    e : TYPE
        Description
    I : TYPE
        Description
    i : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    w = tf.einsum('sij, sj -> si', g, u)
    H = I_eye - 2*tf.einsum('si,sj->sij', w, w) # Householder matrix, specular reflection
    l = tf.einsum('sij, j -> si', H, e) * tf.expand_dims(r, axis=-1)
    x = tf.einsum('sji, sj -> si', g, p - l)
    x = tf.math.segment_mean(x, segment_ids=i)
    x = tf.gather(x, i)

    return x

@tf.function
def batch_get_s(u, x, r, g, e, I):
    """Summary
    
    Parameters
    ----------
    u : TYPE
        Description
    x : TYPE
        Description
    r : TYPE
        Description
    g : TYPE
        Description
    e : TYPE
        Description
    I : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    w = tf.einsum('sij, sj -> si', g, u)
    H = I - 2*tf.einsum('si,sj->sij', w, w) # Householder matrix, specular reflection
    l = tf.einsum('sij, j -> si', H, e) * tf.expand_dims(r, axis=-1)
    s = tf.einsum('sij, sj -> si', g, x) + l
    return s

@tf.function
def batch_optimize_coordinate_descent_u(p, u, x, r, g, e, I_eye, segment_ids, n_iter=10): 

    loss = []
    for j in range(n_iter): 

        r = batch_get_r(p, u, x, g, e, I_eye)
        x = batch_get_x(p, u, r, g, e, I_eye, segment_ids)
        u = batch_get_u(p, x, g, e)
        s = batch_get_s(u, x, r, g, e, I_eye)

        mse = tf.math.segment_mean(tf.reduce_sum((s - p)**2, axis=1), segment_ids=segment_ids)

        loss.append(mse)

    loss = tf.concat(loss, axis=0)
        
    return loss, u, x, r

@tf.function
def batch_get_a(u, v, segment_ids):

    return segment_wahba_svd(u, v, segment_ids)

@tf.function
def batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label, i, verb=False):

    s = batch_get_s(w, x, r, g, e, I_eye)
    mse = tf.math.segment_mean(tf.reduce_sum((s - p)**2, axis=1), segment_ids=segment_ids)
    mse = tf.expand_dims(mse, axis=-1)
    
    if verb:    
        print(f'loss (first grain) {mse.numpy().ravel()[0]:2.8f} iter={i} {label}')

    return mse


@tf.function
def batch_optimize_coordinate_descent_baseline(p, x, r, a, v, g, e, I_eye, segment_ids, n_iter=10, verb=False): 


    from scipy.spatial.transform import Rotation

    loss = []

    for j in range(n_iter): 

        if verb:
            print(f'--------> iter={j}')

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)
        if verb:
            mse = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='update a', i=j, verb=verb)

        # starting loss
        if j == 0:
            mse = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='update a', i=j, verb=False)
            loss.append(mse)

        # update r
        r = batch_get_r(p, w, x, g, e, I)
        if verb:
            mse = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='update r', i=j, verb=verb)

        # update x
        x = batch_get_x(p, w, r, g, e, I, segment_ids)
        if verb:
            mse = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='update x', i=j, verb=verb)

        # update a
        u = batch_get_u_signfix(p, x, w, g, e)
        a = batch_get_a(u, v, segment_ids)
        a = tf.gather(a, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)
        mse = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='update a', i=j, verb=verb)

        loss.append(mse)

    loss = tf.concat(loss, axis=1)

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)
        
    return loss, x, a



@tf.function
def batch_optimize_coordinate_descent_streamlined(p, x, r, a, v, g, e, I_eye, segment_ids, n_iter=10, verb=False): 

    loss = []

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    mse = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='initial loss', i=0)
    loss.append(mse)

    for j in range(n_iter): 


        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar = p - x_rot
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('si, si -> s', p_bar, b)/tf.einsum('si, si -> s', b, b)

        # update x

        l = b * tf.expand_dims(r, axis=-1)
        x = tf.einsum('sji, sj -> si', g, p - l)
        x = tf.math.segment_mean(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
    
        # calculate q
        q = coord_descent_update_q(g, x, p, e)
        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - x_rot

        # eigenvalue solution
        # M = tf.einsum('sji, sj, k, skn -> sin', g, p_bar, e, g )
        # Mt = (M + tf.transpose(M, perm=[0,2,1]))/2.
        # Lam, Q = tf.linalg.eigh(Mt)
        # q = Q[...,0]
        # sign fix
        # q = tf.expand_dims(q, axis=0)
        # q = tf.concat([q, -q], axis=0)
        # d = tf.expand_dims(tf.reduce_sum((q - w)**2, axis=-1), axis=-1)
        # q = tf.where(d[0]<d[1], q[0], q[1])

        # Rhombohedron solution
        # p_norm = p_bar/tf.linalg.norm(p_bar, axis=1, keepdims=True)
        # q = p_norm - e
        # q /= tf.linalg.norm(q, axis=1, keepdims=True)
        # q = tf.einsum('sji, sj -> si', g, q)

        a = segment_wahba_svd(q, v, segment_ids)
        a = tf.gather(a, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)
    
    mse = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='final loss', i=-1)
    loss.append(mse)

    loss = tf.concat(loss, axis=1)

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    return loss, x, a





@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def batch_optimize_coordinate_descent_hardassign(p, x, r, a, v, g, e, I_eye, segment_ids, nn_lookup_spot_ind, s_target, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix, n_iter=10, verb=False): 

    loss = []

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    mse_init = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='initial loss', i=0)
    loss.append(mse_init)

    for j in range(n_iter): 


        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar = p - x_rot
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('si, si -> s', p_bar, b)/tf.einsum('si, si -> s', b, b)

        # update x

        l = b * tf.expand_dims(r, axis=-1)
        x = tf.einsum('sji, sj -> si', g, p - l)
        x = tf.math.segment_mean(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
    
        # calculate q
        q = coord_descent_update_q(g, x, p, e)
        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - x_rot

        # eigenvalue solution
        # M = tf.einsum('sji, sj, k, skn -> sin', g, p_bar, e, g )
        # Mt = (M + tf.transpose(M, perm=[0,2,1]))/2.
        # Lam, Q = tf.linalg.eigh(Mt)
        # q = Q[...,0]
        # sign fix
        # q = tf.expand_dims(q, axis=0)
        # q = tf.concat([q, -q], axis=0)
        # d = tf.expand_dims(tf.reduce_sum((q - w)**2, axis=-1), axis=-1)
        # q = tf.where(d[0]<d[1], q[0], q[1])

        # Rhombohedron solution
        # p_norm = p_bar/tf.linalg.norm(p_bar, axis=1, keepdims=True)
        # q = p_norm - e
        # q /= tf.linalg.norm(q, axis=1, keepdims=True)
        # q = tf.einsum('sji, sj -> si', g, q)

        a = segment_wahba_svd(q, v, segment_ids)
        a = tf.gather(a, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

        # hard assign using lookup, update p
        s = tf.einsum('sij, sj -> si', g, x) + l
        i_target = nn_lookup(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
        p = tf.gather(s_target, i_target)

        if verb:

            r = batch_get_r(p, w, x, g, e, I_eye)
            mse = batch_loss(p, w, x, r, g, e, I, segment_ids, label='step loss', i=j, verb=True)

            if j>0:
                grains = np.unique(i_grn)
                for ig in grains:
                    select = i_grn == ig
                    select = select[:,0]
                    print(f'number of consistent spots grain {ig:<4d}/{len(grains)}     loss= {mse_init[ig].numpy().ravel()[0]: 12.6f} -> {mse[ig].numpy().ravel()[0]: 12.6f}', np.count_nonzero(i_target[select] == i_prev[select]), len(i_prev[select]))

            i_prev = np.array(i_target).copy()

    mse = batch_loss(p, w, x, r, g, e, I_eye, segment_ids, label='final loss', i=-1)
    loss.append(mse)

    loss = tf.concat(loss, axis=1)

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    return loss, x, a













@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def batch_optimize_coordinate_descent_softassign(a, x, s, v, g, s_target, consts, inds_model, lookup_data, control_params=(100, 1e-4, 0.99), n_iter=10, verb=False): 

    e, I_eye = consts
    i_grn, i_ang, i_hkl, i_det, i_gpl, n_hkl = inds_model
    segment_ids = i_grn[:,0]
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data

    # init q
    # d_target, i_target = nn_lookup_dist(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)
    k = p.shape[1]
    v_flat = tf.repeat(v, k, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k, axis=0)
    seg_ones = tf.ones(len(p), dtype=p.dtype)
    i_gpl_uv, i_gpl_ui = tf.unique(i_gpl)
    v_agg = tf.math.unsorted_segment_mean(v, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    loss = []
    beta_init, beta_min, beta_decrease = control_params
    beta = beta_init

    # print(f'=============> new batch')

    for j in range(n_iter): 

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar = p - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        s = tf.expand_dims(x_rot, axis=1) + l

        # update Q

        sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
        logQ = -sig_sq/beta
        Q = tf.math.exp( logQ - tf.reduce_max(logQ, axis=1, keepdims=True) )
        Qn = tf.reduce_sum(Q, axis=1, keepdims=True)
        Q = tf.math.divide(Q, Qn)

        if j == 0:

            loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
            loss_q = tf.reduce_mean(loss_q, axis=(1))
            mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)
            mse = tf.expand_dims(mse, axis=-1)
            loss.append(mse)


        # if True:

        #     loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
        #     loss_q = tf.reduce_mean(loss_q, axis=(1))
        #     mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)

        #     # grains = np.unique(i_grn)
        #     # for ig in grains:
        #          # print(f'grain {ig+1}/{len(grains)} loss={mse[ig]: 12.6e}')

        #     print(f'iter {j+1}/{n_iter} loss={np.mean(mse): 12.6e} beta={beta} maxQ={np.mean(np.max(Q, axis=1)):8.4e}')

        #     mse = tf.expand_dims(mse, axis=-1)
        #     loss.append(mse)


        # update x

        x = tf.einsum('sn, sji, snj -> si', Q, g, p - l)
        x = tf.math.segment_mean(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
    
        # calculate q
        q = coord_descent_update_q(g, x, p, e)
        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - tf.expand_dims(x_rot, axis=1)

        # # Rhombohedron solution
        # p_norm = p_bar/tf.linalg.norm(p_bar, axis=-1, keepdims=True)
        # q = p_norm - e
        # q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        # q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q, shape=(-1,))

        # a = segment_wahba_svd_weighted(q_flat, v_flat, Q_flat, segment_ids_flat)
        # a = tf.gather(a, segment_ids)
        a_agg = segment_aggregate_wahba_svd_weighted(v_agg, q, Q, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones)
        a = tf.gather(a_agg, segment_ids)


        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

        # break loop?
        # reduce beta, annealing
        beta *= beta_decrease



    x_rot = tf.einsum('sij, sj -> si', g, x)
    p_bar = p - tf.expand_dims(x_rot, axis=1)
    u = tf.einsum('sij, sj -> si', g, w)
    H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
    b = tf.einsum('sij, j -> si', H, e)
    r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
    l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
    s = tf.expand_dims(x_rot, axis=1) + l

    sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
    logQ = -sig_sq/beta
    Q = tf.math.exp( logQ - tf.reduce_max(logQ, axis=1, keepdims=True) )
    Qn = tf.reduce_sum(Q, axis=1, keepdims=True)
    Q = tf.math.divide(Q, Qn)
        
    loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
    loss_q = tf.reduce_mean(loss_q, axis=(1))
    mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)
    mse = tf.expand_dims(mse, axis=-1)
    loss.append(mse)

    loss = tf.concat(loss, axis=1)

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    return loss, x, a







@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def batch_optimize_coordinate_descent_softlowmem(a, x, s, v, g, s_target, consts, inds_model, lookup_data, control_params=(100, 1e-4, 0.99), n_iter=10, verb=False): 
    
    e, I_eye = consts
    i_grn, i_ang, i_hkl, i_det, i_gpl, n_hkl = inds_model
    segment_ids = i_grn[:,0]
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data

    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)
    k = p.shape[1]
    v_flat = tf.repeat(v, k, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k, axis=0)
    seg_ones = tf.ones(len(p), dtype=p.dtype)
    i_gpl_uv, i_gpl_ui = tf.unique(i_gpl)
    v_agg = tf.math.unsorted_segment_mean(v, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))

    w = tf.einsum('bij, bj -> bi', a, v)

    loss = []
    beta_init, beta_min, beta_decrease = control_params
    beta = beta_init

    # print(f'=============> new batch')

    for j in range(n_iter): 

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)

        p_bar = p - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        s = tf.expand_dims(x_rot, axis=1) + l

        # update Q

        sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
        logQ = -sig_sq/beta
        Q = tf.math.exp( logQ - tf.reduce_max(logQ, axis=1, keepdims=True) ) + 1e-20
        Qn = tf.reduce_sum(Q, axis=1, keepdims=True)
        Q = tf.math.divide(Q, Qn)

        if j == 0:

            loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
            loss_q = tf.reduce_mean(loss_q, axis=(1))
            mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)
            mse = tf.expand_dims(mse, axis=-1)
            loss.append(mse)


        # if True:
            
        #     # print(f'=============> iter {j+1}/{n_iter}')

        #     loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
        #     loss_q = tf.reduce_mean(loss_q, axis=(1))
        #     mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)

        #     # grains = np.unique(i_grn)
        #     # for ig in grains:
        #          # print(f'grain {ig+1}/{len(grains)} loss={mse[ig]: 12.6e}')

        #     print(f'iter  iter {j+1}/{n_iter} loss={np.mean(mse): 12.6e} beta={beta} maxQ={np.mean(np.max(Q, axis=1)):8.4e}')

        #     mse = tf.expand_dims(mse, axis=-1)
        #     loss.append(mse)


        # update x

        x = tf.einsum('sn, sji, snj -> si', Q, g, p - l)
        x = tf.math.segment_mean(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
    
        # calculate q
        q = coord_descent_update_q(g, x, p, e)
        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - tf.expand_dims(x_rot, axis=1)

        # # Rhombohedron solution
        # p_norm = p_bar/tf.linalg.norm(p_bar, axis=-1, keepdims=True)
        # q = p_norm - e
        # q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        # q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q, shape=(-1,))

        a_agg = segment_aggregate_wahba_svd_weighted(v_agg, q, Q, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones)
        a = tf.gather(a_agg, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

        # break loop?
        # reduce beta, annealing
        beta *= beta_decrease


    x_rot = tf.einsum('sij, sj -> si', g, x)
    p_bar = p - tf.expand_dims(x_rot, axis=1)
    u = tf.einsum('sij, sj -> si', g, w)
    H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
    b = tf.einsum('sij, j -> si', H, e)
    r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
    l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
    s = tf.expand_dims(x_rot, axis=1) + l

    sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
    logQ = -sig_sq/beta
    Q = tf.math.exp( logQ - tf.reduce_max(logQ, axis=1, keepdims=True) ) + 1e-20
        
    loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
    loss_q = tf.reduce_mean(loss_q, axis=(1))
    mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)
    mse = tf.expand_dims(mse, axis=-1)
    loss.append(mse)

    loss = tf.concat(loss, axis=1)

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    return loss, x, a





# @tf.function(jit_compile=False)
def batch_optimize_coordinate_descent_softent(a, x, s, v, g, s_target, consts, inds_model, lookup_data, control_params=(100, 1e-4, 0.99), n_iter=10, verb=False): 

    e, I_eye = consts
    i_grn, i_ang, i_hkl, i_det, i_gpl, n_hkl = inds_model
    segment_ids = i_grn[:,0]
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data

    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)
    n_model, k_nn, n_dims = p.shape
    v_flat = tf.repeat(v, k_nn, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k_nn, axis=0)
    seg_ones = tf.ones(n_model, dtype=p.dtype)
    seg_count = tf.math.segment_sum(seg_ones, segment_ids)
    i_gpl_uv, i_gpl_ui = tf.unique(i_gpl)
    v_agg = tf.math.unsorted_segment_mean(v, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    loss = []
    eps_init, lam_mod, eps_decrease = control_params
    
    ot_err_thresh = 1e-3 # use low threshold for speed
    for j in range(n_iter): 

        eps = eps_init * eps_decrease**tf.cast(j, tf.float64)
        fia = lam_mod/(lam_mod+eps)

        # update r    
        s, l = coord_descent_update_r(g, x, p, w, e, I_eye)

        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - tf.expand_dims(x_rot, axis=1)
        # u = tf.einsum('sij, sj -> si', g, w)
        # H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        # b = tf.einsum('sij, j -> si', H, e)
        # r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        # l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        # s = tf.expand_dims(x_rot, axis=1) + l

        # update Q
        # sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
        # logQ = -sig_sq/eps

        # # safe exp and re-normalize
        # # logQ_max = tf.gather(tf.math.segment_max(logQ, segment_ids), segment_ids)
        # # Q = tf.math.exp( logQ - logQ_max) 
        # # Qun = tf.gather(tf.math.segment_sum(tf.reduce_sum(Q, axis=1), segment_ids=segment_ids), segment_ids)
        # # Q = tf.math.divide_no_nan(Q, tf.expand_dims(Qun, axis=-1))
        # Q = tf.math.exp(logQ) + 1e-20

        # # run softassign with entropic regularization
        # Q = tf.math.divide_no_nan(Q, tf.reduce_sum(Q, axis=1, keepdims=True)**fia)
        Q = coord_descent_update_assignment_softent(p, s, eps, fia)

        # Q_sorted = np.sort(Q, axis=1)
        # print('Q stats  median=', astr(np.median(Q_sorted, axis=0)), '              std=', astr(np.std(Q_sorted, axis=0)))

        # re-normalize weights for each model grain
        Qun = tf.reduce_sum(Q, axis=1)
        Qun = tf.math.segment_sum(Qun, segment_ids=segment_ids)
        Qun = tf.gather(Qun, segment_ids)
        Q = tf.math.divide_no_nan(Q, tf.expand_dims(Qun, axis=-1))

        # verb
        # if True:
        #     Qun_ = tf.reduce_sum(Q, axis=1)
        #     Qun_max = tf.gather(tf.math.segment_max(Qun_, segment_ids=segment_ids), segment_ids)
        #     n_out = np.count_nonzero(Qun_<Qun_max*1e-3)
        #     frac_outliers = n_out/len(Qun_max)
        #     print(f'iter={j} eps={eps:2.4e} fia={fia:2.4e} frac_outliers={frac_outliers:8.8f} {n_out}')


        if j == 0:

            loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=1)
            loss_q = tf.reduce_mean(loss_q, axis=1)
            mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids) * seg_count
            mse = tf.expand_dims(mse, axis=-1)
            loss.append(mse)


        # if True:
            
        #     # print(f'=============> iter {j+1}/{n_iter}')

        #     loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
        #     loss_q = tf.reduce_mean(loss_q, axis=(1))
        #     mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)

        #     # grains = np.unique(i_grn)
        #     # for ig in grains:
        #          # print(f'grain {ig+1}/{len(grains)} loss={mse[ig]: 12.6e}')

        #     print(f'iter  iter {j+1}/{n_iter} loss={np.mean(mse): 12.6e} beta={beta} maxQ={np.mean(np.max(Q, axis=1)):8.4e}')

        #     mse = tf.expand_dims(mse, axis=-1)
        #     loss.append(mse)


        # update x

        x = tf.einsum('sn, sji, snj -> si', Q, g, p - l)
        x = tf.math.segment_sum(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
    
        # calculate q
        q = coord_descent_update_q(g, x, p, e)

        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - tf.expand_dims(x_rot, axis=1)

        # # Rhombohedron solution
        # p_norm = p_bar/tf.linalg.norm(p_bar, axis=-1, keepdims=True)
        # q = p_norm - e
        # q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        # q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q, shape=(-1,))

        # a = segment_wahba_svd_weighted(q_flat, v_flat, Q_flat, segment_ids_flat)
        # a = tf.gather(a, segment_ids)

        a_agg = segment_aggregate_wahba_svd_weighted(v_agg, q, Q, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones)
        a = tf.gather(a_agg, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

    s, l = coord_descent_update_r(g, x, p, w, e, I_eye)

    # x_rot = tf.einsum('sij, sj -> si', g, x)
    # p_bar = p - tf.expand_dims(x_rot, axis=1)
    # u = tf.einsum('sij, sj -> si', g, w)
    # H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
    # b = tf.einsum('sij, j -> si', H, e)
    # r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
    # l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
    # s = tf.expand_dims(x_rot, axis=1) + l

    # update Q
    # sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
    # logQ = -sig_sq/eps

    # # safe exp and re-normalize
    # Q = tf.exp(logQ) + 1e-20

    # # entropic sinkhorn scaling
    # Qs = tf.math.divide_no_nan(Q, tf.reduce_sum(Q, axis=1, keepdims=True)**fia)
    Qs = coord_descent_update_assignment_softent(p, s, eps, fia)

    
    # re-normalize weights for each model grain
    Qun = tf.reduce_sum(Qs, axis=1)
    Qun = tf.math.segment_sum(Qun, segment_ids=segment_ids)
    Qun = tf.gather(Qun, segment_ids)
    Qsn = tf.math.divide_no_nan(Qs, tf.expand_dims(Qun, axis=-1))
            
    # calculate various losses
    l2 = tf.math.sqrt(tf.reduce_sum((s-p)**2, axis=-1))
    loss_q = tf.reduce_sum(tf.expand_dims(Qsn, axis=-1)*(s-p)**2, axis=1)
    loss_q = tf.reduce_mean(loss_q, axis=1)

    # old buggy loss
    mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids) * seg_count

    # full loss with entropic terms
    l2_nn = tf.math.segment_mean(tf.reduce_min(l2, axis=1), segment_ids)
    l2_qsn = tf.math.segment_sum(tf.reduce_sum(Qsn*l2, axis=1), segment_ids)/tf.math.segment_sum(tf.reduce_sum(Qsn, axis=1), segment_ids)
    l2_sq_q = tf.math.segment_sum(tf.reduce_sum(Qs*l2**2, axis=1), segment_ids)
    H = tf.math.segment_sum(tf.reduce_sum(tf.math.log(Qs)*Qs, axis=1), segment_ids)
    D1 = tf.math.segment_sum(tf.reduce_sum(Qs, axis=1)*tf.reduce_sum(tf.math.log(Qs), axis=1), segment_ids)
    D2 = tf.math.segment_sum(tf.reduce_sum(Qs, axis=1), segment_ids)
    D = D1 - D2
    loss_full = 0.5*l2_sq_q + eps*H + lam_mod*D
    loss.extend([tf.expand_dims(l2_nn, axis=-1), tf.expand_dims(mse, axis=-1), tf.expand_dims(loss_full, axis=-1)])

    # chi2 statistic forf inliers 
    inlier_q_min = 5e-2
    is_inlier = tf.cast(tf.reduce_max(Qs, axis=-1) > inlier_q_min, tf.float64)
    n_inliers = tf.math.segment_sum(is_inlier, segment_ids)
    chi2_nn = tf.math.segment_sum(tf.reduce_min(l2**2, axis=-1)*is_inlier, segment_ids)/(eps/2.)
    chi2_red = tf.math.divide_no_nan(chi2_nn, n_inliers)
    frac_inliers = tf.expand_dims(n_inliers/seg_count, axis=-1)
    chi2_red = tf.expand_dims(chi2_red, axis=-1) 
    loss.append(chi2_red)

    loss = tf.concat(loss, axis=1)

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    return loss, frac_inliers, x, a








@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def batch_optimize_coordinate_descent_softpart(a, x, s, v, g, s_target, consts, inds_model, lookup_data, control_params=(100, 1e-4, 0.99), n_iter=10, verb=False): 

    e, I_eye = consts
    i_grn, i_ang, i_hkl, i_det, i_gpl, n_hkl = inds_model
    segment_ids = i_grn[:,0]
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data
    eps_init, eps_decrease, outliers_m = control_params

    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)
    n, k = i_target.shape
    v_flat = tf.repeat(v, k, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k, axis=0)

    seg_ones = tf.ones(len(p), dtype=p.dtype)
    i_gpl_uv, i_gpl_ui = tf.unique(i_gpl)
    v_agg = tf.math.unsorted_segment_mean(v, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))

    seg_count = tf.math.segment_sum(seg_ones, segment_ids)
    ot_a = tf.expand_dims(1./tf.gather(seg_count, segment_ids), axis=-1)[:,0]
    ot_b = tf.expand_dims(1./tf.gather(seg_count, segment_ids), axis=-1)[:,0]
    ot_m = tf.constant(outliers_m, dtype=tf.float64)
    ot_dx = tf.constant(1, dtype=tf.float64)
    ot_dy = tf.constant(1, dtype=tf.float64)

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    loss = []
    
    ot_err_thresh = 1e-4 # use high threshold for speed
    for j in range(n_iter): 

        eps = eps_init * eps_decrease**tf.cast(j, tf.float64)

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar = p - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        s = tf.expand_dims(x_rot, axis=1) + l

        # update Q

        sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
        logQ = -sig_sq/eps

        # safe exp and re-normalize
        # logQ_max = tf.gather(tf.math.segment_max(logQ, segment_ids), segment_ids)
        # Q = tf.math.exp( logQ - logQ_max) 
        # Qun = tf.gather(tf.math.segment_sum(tf.reduce_sum(Q, axis=1), segment_ids), segment_ids)
        # Q = tf.math.divide_no_nan(Q, tf.expand_dims(Qun, axis=-1))
        Q = tf.math.exp(logQ) + 1e-20

        # entropic sinkhorn scaling

        # # scale1 = tf.math.minimum(tf.math.divide_no_nan(ot_a, tf.reduce_sum(Q, axis=1)), dx)
        # a_norm = tf.math.divide_no_nan(ot_a, tf.reduce_sum(Q, axis=1))
        # scale1 = tf.where(a_norm<dx, a_norm, dx)
        # # scale1 = tf.math.minimum(tf.math.divide_no_nan(ot_a, tf.reduce_sum(Q, axis=1)), dx)
        # Q1 = scale1[:,tf.newaxis] * Q
        # # scale2 = tf.math.minimum(tf.math.divide_no_nan(ot_b[:,tf.newaxis], Q1), dy[:,tf.newaxis])
        # b_norm = tf.math.divide_no_nan(ot_b[:,tf.newaxis], Q1)
        # scale2 = tf.where(b_norm<dy[:,tf.newaxis], b_norm, dy[:,tf.newaxis])
        # Q2  = scale2 * Q1
        # m_norm = tf.math.divide_no_nan(ot_m, tf.math.segment_sum(tf.reduce_sum(Q2, axis=1), segment_ids))
        # Q = Q2 * tf.expand_dims(tf.gather(m_norm, segment_ids), axis=-1)

        Q = optimal_transport.sinkhorn_knopp_oneside_partial(Q, ot_a, ot_b, ot_dx, ot_dy, ot_m, n, segment_ids, err_thresh=ot_err_thresh)

        # re-normalize weights for each model grain
        Qun = tf.reduce_sum(Q, axis=1)
        Qun = tf.math.segment_sum(Qun, segment_ids=segment_ids)
        Qun = tf.gather(Qun, segment_ids)
        Q = tf.math.divide_no_nan(Q, tf.expand_dims(Qun, axis=-1))

        # Q_max = np.sort(Q, axis=1)[:,::-1]
        # print(f'iter {j:>6d}/{n_iter} mse={mse[0,0]:8.4e} sum_Q={np.sum(Q):2.6e} mean_Q={astr(np.mean(Q_max, axis=0))}')

        # verb
        # if True:
        #     Qun_ = tf.reduce_sum(Q, axis=1)
        #     Qun_max = tf.gather(tf.math.segment_max(Qun_, segment_ids=segment_ids), segment_ids)
        #     n_out = np.count_nonzero(Qun_<Qun_max*1e-3)
        #     frac_outliers = n_out/len(Qun_max)
        #     print(f'iter={j} eps={eps:2.4e} fia={fia:2.4e} frac_outliers={frac_outliers:8.8f} {n_out}')

        # Qua = np.sum(Q, axis=1)[segment_ids==0]
        # print('Q stats', astr([eps, fia, np.min(Qua), np.max(Qua), np.mean(Qua), np.std(Qua)/np.mean(Qua)]))

        if j == 0:
        # if True:

            loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
            loss_q = tf.reduce_mean(loss_q, axis=(1))
            mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids) * seg_count
            mse = tf.expand_dims(mse, axis=-1)
            loss.append(mse)
        

        # if True:
            
        #     # print(f'=============> iter {j+1}/{n_iter}')

        #     loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
        #     loss_q = tf.reduce_mean(loss_q, axis=(1))
        #     mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)

        #     # grains = np.unique(i_grn)
        #     # for ig in grains:
        #          # print(f'grain {ig+1}/{len(grains)} loss={mse[ig]: 12.6e}')

        #     print(f'iter  iter {j+1}/{n_iter} loss={np.mean(mse): 12.6e} beta={beta} maxQ={np.mean(np.max(Q, axis=1)):8.4e}')

        #     mse = tf.expand_dims(mse, axis=-1)
        #     loss.append(mse)


        # update x

        x = tf.einsum('sn, sji, snj -> si', Q, g, p - l)
        x = tf.math.segment_sum(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
    
        # calculate q
        q = coord_descent_update_q(g, x, p, e)
        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - tf.expand_dims(x_rot, axis=1)

        # # Rhombohedron solution
        # p_norm = p_bar/tf.linalg.norm(p_bar, axis=-1, keepdims=True)
        # q = p_norm - e
        # q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        # q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q, shape=(-1,))

        # a = segment_wahba_svd_weighted(q_flat, v_flat, Q_flat, segment_ids_flat)
        # a = tf.gather(a, segment_ids)

        a_agg = segment_aggregate_wahba_svd_weighted(v_agg, q, Q, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones)
        a = tf.gather(a_agg, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

    x_rot = tf.einsum('sij, sj -> si', g, x)
    p_bar = p - tf.expand_dims(x_rot, axis=1)
    u = tf.einsum('sij, sj -> si', g, w)
    H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
    b = tf.einsum('sij, j -> si', H, e)
    r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
    l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
    s = tf.expand_dims(x_rot, axis=1) + l

    sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
    logQ = -sig_sq/eps
    Q = tf.math.exp(logQ) + 1e-20
    Q = optimal_transport.sinkhorn_knopp_oneside_partial(Q, ot_a, ot_b, ot_dx, ot_dy, ot_m, n, segment_ids, err_thresh=ot_err_thresh)


    # # safe exp and re-normalize
    # logQ_max = tf.gather(tf.math.segment_max(logQ, segment_ids), segment_ids)
    # Q = tf.math.exp( logQ - logQ_max) 
    # Qun = tf.gather(tf.math.segment_sum(tf.reduce_sum(Q, axis=1), segment_ids=segment_ids), segment_ids)
    # Q = tf.math.divide_no_nan(Q, tf.expand_dims(Qun, axis=-1))

    # # entropic sinkhorn scaling

    # # scale1 = tf.math.minimum(tf.math.divide_no_nan(ot_a, tf.reduce_sum(Q, axis=1)), dx)
    # a_norm = tf.math.divide_no_nan(ot_a, tf.reduce_sum(Q, axis=1))
    # scale1 = tf.where(a_norm<dx, a_norm, dx)
    # # scale1 = tf.math.minimum(tf.math.divide_no_nan(ot_a, tf.reduce_sum(Q, axis=1)), dx)
    # Q1 = scale1[:,tf.newaxis] * Q
    # # scale2 = tf.math.minimum(tf.math.divide_no_nan(ot_b[:,tf.newaxis], Q1), dy[:,tf.newaxis])
    # b_norm = tf.math.divide_no_nan(ot_b[:,tf.newaxis], Q1)
    # scale2 = tf.where(b_norm<dy[:,tf.newaxis], b_norm, dy[:,tf.newaxis])
    # Q2  = scale2 * Q1
    # m_norm = tf.math.divide_no_nan(ot_m, tf.math.segment_sum(tf.reduce_sum(Q2, axis=1), segment_ids))
    # Q = Q2 * tf.expand_dims(tf.gather(m_norm, segment_ids), axis=-1)
    
    loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
    loss_q = tf.reduce_mean(loss_q, axis=(1))
    mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids) * seg_count

    mse = tf.expand_dims(mse, axis=-1) 
    loss.append(mse)

    loss = tf.concat(loss, axis=1)

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    return loss, x, a

@tf.function(jit_compile=False)
def batch_optimize_coordinate_descent_softslacks(a, x, s, v, g, s_target, consts, inds_model, lookup_data, control_params=(100, 1e-4, 0.99), n_iter=10, verb=False): 

    # unpack args
    e, I_eye = consts
    i_grn, i_ang, i_hkl, i_det, i_gpl, n_hkl = inds_model
    segment_ids = i_grn[:,0]
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data
    eps_init, eps_decrease, n_sig_outlier = control_params

    # get neighbours and other constants
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)
    n_model, k_nn = i_target.shape
    v_flat = tf.repeat(v, k_nn, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k_nn, axis=0)
    seg_ones = tf.ones(len(p), dtype=p.dtype)
    i_gpl_uv, i_gpl_ui = tf.unique(i_gpl)
    v_agg = tf.math.unsorted_segment_mean(v, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))
    seg_count = tf.math.segment_sum(seg_ones, segment_ids)
    
    # init variables
    w = tf.einsum('bij, bj -> bi', a, v)
    alpha_slacks = -0.5*(n_sig_outlier)**2

    logC_outlier = tf.exp(tf.zeros((n_model, 1), dtype=tf.float64))
        
    # run loop
    loss = []
    for j in range(n_iter): 

        # decrease temperature
        eps = eps_init * eps_decrease**tf.cast(j, tf.float64)

        # update r    
        s, l = coord_descent_update_r(g, x, p, w, e, I_eye)

        # update Q
        Q_slacks = coord_descent_update_assignment_softslacks(p, s, eps, alpha_slacks, logC_outlier)

        # re-normalize weights for each model grain for weigthed average
        Q = Q_slacks[:,:-1] # cut out slack variables
        Qun = tf.reduce_sum(Q, axis=1)
        Qun = tf.math.segment_sum(Qun, segment_ids=segment_ids)
        Qun = tf.gather(Qun, segment_ids)
        Q = tf.math.divide_no_nan(Q, tf.expand_dims(Qun, axis=-1))

        if j == 0:

            loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=1)
            loss_q = tf.reduce_mean(loss_q, axis=1)
            mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids) * seg_count
            mse = tf.expand_dims(mse, axis=-1)
            loss.append(mse)

        # update x
        x = tf.einsum('sn, sji, snj -> si', Q, g, p - l)
        x = tf.math.segment_sum(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
        q = coord_descent_update_q(g, x, p, e)
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q, shape=(-1,))

        # procrustes
        a_agg = segment_aggregate_wahba_svd_weighted(v_agg, q, Q, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones)
        a = tf.gather(a_agg, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

    # final updates
    s, l = coord_descent_update_r(g, x, p, w, e, I_eye)

    # update Q
    Q_slacks = coord_descent_update_assignment_softslacks(p, s, eps, alpha_slacks, logC_outlier)
    Q_slacks = tf.math.divide_no_nan(Q_slacks, tf.reduce_sum(Q_slacks, axis=1, keepdims=True))
    
    # losses calculation

    # full loss
    C_in = tf.reduce_sum((s-p)**2, axis=-1)
    C_out = tf.ones((n_model, 1), dtype=tf.float64)*alpha_slacks
    C = tf.concat([C_in, C_out], axis=-1)
    loss_q = tf.reduce_sum(Q_slacks*C, axis=1)
    mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)
    mse = tf.expand_dims(mse, axis=-1) 
    loss.append(mse)

    # chi2 statistic for inliers 
    id_outlier = k_nn
    is_inlier = tf.cast(tf.math.argmax(Q_slacks, axis=-1) < id_outlier, tf.float64)
    n_inliers = tf.math.segment_sum(is_inlier, segment_ids)
    chi2_nn = tf.math.segment_sum(tf.reduce_min(C_in, axis=-1)*is_inlier, segment_ids)/(eps/2.)
    chi2_red = tf.math.divide_no_nan(chi2_nn, n_inliers)
    frac_inliers = tf.expand_dims(n_inliers/seg_count, axis=-1)
    chi2_red = tf.expand_dims(chi2_red, axis=-1) 
    loss.append(chi2_red)

    # merge losses
    loss = tf.concat(loss, axis=1)

    # average per grain
    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    return loss, frac_inliers, x, a



def batch_optimize_coordinate_descent_sinkhorn_balanced_sparse(p, x, s, a, v, g, segment_ids, s_target, inds_model, lookup_data, n_iter=10, control_params=(1e3, 0.99, 0.1, 64), verb=False, test=False): 

    # unpack params

    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data 
    eps_init, eps_decrease, eps_min, n_iter_sinkhorn = control_params
    i_grn, i_ang, i_hkl, i_det = inds_model

    # define constant
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0.], dtype=tf.float64)

    # init indices and constants
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    i_target_ = np.array(i_target)
    p = tf.gather(s_target, i_target)
    i_target_flat = tf.reshape(i_target, (-1,))
    i_target_unique_ids, i_target_unique_flat = tf.unique(i_target_flat)
    i_target_unique = tf.reshape(i_target_unique_flat, shape=i_target.shape)
    n_target_unique = len(i_target_unique_ids)
    n_target = len(s_target)
    k_nn = p.shape[1]
    v_flat = tf.repeat(v, k, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k, axis=0)

    # init w and loss
    w = tf.einsum('bij, bj -> bi', a, v)
    loss = []
    list_Qm = []
    list_nperfect = []


    # optimizer steps
    for j in range(n_iter): 

        ## -------------- seg version

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar = p - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        s = tf.expand_dims(x_rot, axis=1) + l

        # update Q

        sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
        logQ = -beta*sig_sq
        Q_init = tf.math.exp( logQ - tf.reduce_max(logQ, axis=1, keepdims=True) ) 
        Q = tf.constant(Q_init)

        if j == 0:

            Q_first = np.array(Q_init)
            s_first = np.array(s)


        # Sinkhorn

        for h in range(n_iter_sinkhorn):

            # segment sum sinkhorn

            # update rows
            Qr = tf.reduce_sum(Q, axis=1, keepdims=True)
            Q_ = tf.math.divide(Q, Qr)

            # update columns using segment sums
            Qc = tf.math.unsorted_segment_sum(data=tf.reshape(Q_, shape=(-1,)), segment_ids=i_target_unique, num_segments=n_target_unique)
            Qc = tf.gather(Qc, i_target_flat)
            Qc = tf.reshape(Qc, shape=i_target.shape)
            Q_ = tf.math.divide(Q_, Qc)

            Q = Q_

            if tf.math.reduce_any(tf.math.is_nan(Q)):

                import ipdb; ipdb.set_trace(); 
                pass

        # fake Qs to correct assignment
        # Q = np.zeros(Q.shape)
        # Q[:,0] = 1
        # Q = tf.constant(Q)

        if verb:

            Qm = np.argmax(Q, axis=1)
            s_match = np.array([s[i,Qm[i]] for i in range(len(s))])
            p_match = np.array([p[i,Qm[i]] for i in range(len(p))])
            i_match = np.array([i_target_[i,Qm[i]] for i in range(len(i_target_))])
            sorting = np.argsort(i_match)
            s_match_sorted = s_match[sorting]
            i_match_sorted = i_match[sorting]
            loss_hard = np.mean((s_target-s_match_sorted)**2)
            n_matched_perfect = np.count_nonzero(i_match_sorted==np.arange(len(i_match_sorted)))
            list_nperfect.append(n_matched_perfect)



        loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
        loss_q = tf.reduce_mean(loss_q, axis=(1))
        mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)
        mse = tf.expand_dims(mse, axis=-1)
        loss.append(mse)
        if verb:
            Qm = np.max(Q, axis=1)
            list_Qm.append(Qm)
            print(f'======> iter {j: 5d}   loss_q={tf.reduce_mean(loss_q): 12.6f}     mean Q_max={np.mean(Qm): 12.6e}      Q_diff={np.max(np.abs(Q_-Q)): 12.6e} ')


        # update x

        x = tf.einsum('sn, sji, snj -> si', Q, g, p - l)
        x = tf.math.segment_mean(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
    
        # calculate q
        q = coord_descent_update_q(g, x, p, e)
        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - tf.expand_dims(x_rot, axis=1)


        # # Rhombohedron solution
        # p_norm = p_bar/tf.linalg.norm(p_bar, axis=-1, keepdims=True)
        # q = p_norm - e
        # q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        # q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q, shape=(-1,))

        a = segment_wahba_svd_weighted(q_flat, v_flat, Q_flat, segment_ids_flat)
        a = tf.gather(a, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

        # reduce beta, annealing
        beta *= beta_increase


    x_rot = tf.einsum('sij, sj -> si', g, x)
    p_bar = p - tf.expand_dims(x_rot, axis=1)
    u = tf.einsum('sij, sj -> si', g, w)
    H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
    b = tf.einsum('sij, j -> si', H, e)
    r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
    l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
    s = tf.expand_dims(x_rot, axis=1) + l

    sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
    logQ = -beta*sig_sq
    Q = tf.math.exp( logQ - tf.reduce_max(logQ, axis=1, keepdims=True) )

    loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2, axis=(1))
    loss_q = tf.reduce_mean(loss_q, axis=(1))
    mse = tf.math.segment_mean( loss_q, segment_ids=segment_ids)
    mse = tf.expand_dims(mse, axis=-1)
    loss.append(mse)

    loss = tf.concat(loss, axis=1)

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    # debugging section

    i_target_ = np.array(i_target)

    # calculate the first hard assignment and hard loss

    Qm_first = np.argmax(Q_first, axis=1)
    s_first = np.array([s_first[i,Qm_first[i]] for i in range(len(s_first))])
    p_first = np.array([p[i,Qm_first[i]] for i in range(len(p))])
    i_first = np.array([i_target_[i,Qm_first[i]] for i in range(len(i_target_))])

    sorting = np.argsort(i_first)
    s_first_sorted = s_first[sorting]
    i_match_first = i_first[sorting]
    loss_hard = np.mean((s_target-s_first_sorted)**2)


    # calculate the final hard assignment and hard loss

    Qm = np.argmax(Q, axis=1)
    i_target_ = np.array(i_target)
    s_best = np.array([s[i,Qm[i]] for i in range(len(s))])
    p_best = np.array([p[i,Qm[i]] for i in range(len(p))])
    i_best = np.array([i_target_[i,Qm[i]] for i in range(len(i_target_))])

    sorting = np.argsort(i_best)
    s_best_sorted = s_best[sorting]
    i_match = i_best[sorting]
    loss_hard = np.mean((s_target-s_best_sorted)**2)


    print(f'loss_hard = {loss_hard:12.6f}')

    if verb:
        fname = 'opt_vars.npy'
        np.save(fname, [loss, list_Qm, i_match_first, i_match, list_nperfect])
        print(f'saved {fname}')

    return loss, x, a, Q, i_match, None, None


def batch_optimize_coordinate_descent_sinkhorn_balanced_full(p, x, s, a, v, g, segment_ids, nn_lookup_spot_ind, s_target, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix, inds_target=None, n_iter=10, n_iter_sinkhorn=64, verb=False, test=False, beta=0.01, beta_increase=1.0): 

    # define constant
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0.], dtype=tf.float64)


    # init q
    # d_target, i_target = nn_lookup_dist(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)

    # i_target_uv, i_target_ui = tf.unique(tf.reshape(s_target, (-1,)))
    # i_target = tf.reshape(tf.gather(i_target_uv, i_target_ui), shape=i_target.shape)
    i_target_flat = tf.reshape(i_target, (-1,))
    n_target = len(np.unique(i_target_flat))

    # full version (no neighbours)
    i_target_full = np.repeat(np.arange(s_target.shape[0]).reshape(1,-1), s_target.shape[0], axis=0)
    p_full = tf.gather(s_target, i_target_full)
    k = p.shape[1]
    v_flat = tf.repeat(v, k, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k, axis=0)

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    alpha = 1e-6
    eps = 1e-20
    beta = 0.01
    beta_increase = 1.00

    loss = []

    for j in range(n_iter): 

        ## -------------- seg version

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar = p - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        s = tf.expand_dims(x_rot, axis=1) + l

        # update Q

        sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
        logQ = -beta*sig_sq
        Q_init = tf.math.exp( logQ - tf.reduce_max(logQ, axis=1, keepdims=True) ) 
        Q = tf.constant(Q_init)

        k_ind = i_target_flat
        i_ind = tf.repeat(np.arange(s.shape[0], dtype=np.int32), s.shape[1], axis=0)
        indices = tf.concat([tf.expand_dims(i_ind, axis=-1), tf.expand_dims(k_ind, axis=-1)], axis=-1)

        Q_scatter = tf.zeros((s.shape[0], s_target.shape[0]), dtype=tf.float64) #+ eps
        Q_scatter = tf.tensor_scatter_nd_add(Q_scatter, indices, tf.reshape(Q, (-1,)))
        # Q_scatter = Q_scatter/tf.reduce_sum(Q_scatter)

        ## -------------- full version

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar_full = p_full - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r_full = tf.einsum('sni, si -> sn', p_bar_full, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l_full = tf.expand_dims(b, axis=1) * tf.expand_dims(r_full, axis=-1)
        s_full = tf.expand_dims(x_rot, axis=1) + l_full

        # update Q

        sig_sq_full = tf.reduce_sum((p_full - s_full)**2, axis=-1)
        logQ_full = -beta*sig_sq_full

        ang_mask = i_ang == tf.transpose(i_ang, [1,0])
        det_mask = i_det == tf.transpose(i_det, [1,0])
        mask = det_mask & ang_mask
        logQ_full = tf.where(mask, logQ_full, -1e20)
        Q_init_full = tf.math.exp( logQ_full - tf.reduce_max(logQ_full, axis=1, keepdims=True) ) # + eps
        Q_full = tf.constant(Q_init_full)
        # Q_full = Q_full/tf.reduce_sum(Q_full)

        from laueotx.utils import io as utils_io
        utils_io.write_arrays('Qs_init.h5', Q_full=Q_full, Q_scatter=Q_scatter)

        nn = 0
        diff =  p[:,nn,:]-s_target
        diff_tot = np.sum(np.abs(diff), axis=1)
        select = diff_tot>1e-8
        print('bad ids',np.nonzero(select))
        print('s_target bad', s_target[select].numpy())
        print('p[:,0,:] bad', p[:,0,:][select].numpy())
        print('i_ang bad', i_ang[select].numpy().ravel())
        print('i_det bad', i_det[select].numpy().ravel())
        print('i_grn bad', i_grn[select].numpy().ravel())


        # Sinkhorn

        for h in range(n_iter_sinkhorn):

            # full matrix sinkhorn
            Q_full_ = Q_full
            Qr_full = tf.reduce_sum(Q_full_, axis=1, keepdims=True)
            Q_full_ = Q_full_ / Qr_full
            Qc_full = tf.reduce_sum(Q_full_, axis=0, keepdims=True)
            Q_full_ = Q_full_ / Qc_full

            # segment sum sinkhorn

            # update rows
            Qr = tf.reduce_sum(Q, axis=1, keepdims=True)
            Q_ = tf.math.divide(Q, Qr)

            # update columns using segment sums
            Qc = tf.math.unsorted_segment_sum(data=tf.reshape(Q_, shape=(-1,)), segment_ids=i_target_flat, num_segments=n_target)
            Qc = tf.gather(Qc, i_target_flat)
            Qc = tf.reshape(Qc, shape=i_target.shape)
            Q_ = tf.math.divide(Q_, Qc)

            Q = Q_
            Q_full = Q_full_

            if tf.math.reduce_any(tf.math.is_nan(Q)):

                import ipdb; ipdb.set_trace(); 
                pass

        Q_scatter = tf.zeros((s.shape[0], s_target.shape[0]), dtype=tf.float64) 
        Q_scatter = tf.tensor_scatter_nd_add(Q_scatter, indices, tf.reshape(Q, (-1,)))

        store_arrays('Qs.h5', Q_full=Q_full, Q_scatter=Q_scatter)

        loss_q = tf.reduce_sum(tf.expand_dims(Q, axis=-1)*(s-p)**2)
        loss.append(loss_q)
        print(f'======> iter {j}.   loss_q={loss_q: 12.6f}    Q_diff={np.max(np.abs(Q_-Q)): 12.6e}')

        # update x

        x = tf.einsum('sn, sji, snj -> si', Q, g, p - l)
        x = tf.math.segment_mean(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update a
    
        # calculate q
        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar = p - tf.expand_dims(x_rot, axis=1)


        # Rhombohedron solution
        p_norm = p_bar/tf.linalg.norm(p_bar, axis=-1, keepdims=True)
        q = p_norm - e
        q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q, shape=(-1,))

        a = segment_wahba_svd_weighted(q_flat, v_flat, Q_flat, segment_ids_flat)
        a = tf.gather(a, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

        # reduce beta, annealing
        beta *= beta_increase

    max_diff = np.max(np.abs(Q_full-Q_scatter))
    print(f'testing sparse vs full assignment matrix, Q_scatter vs Q_full, max diff = {max_diff}')
    
    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    return loss, x, a, (Q_full, Q_scatter, Q), s_full, None



def batch_optimize_coordinate_descent_sinkhorn_unbalanced_full(p, x, s, a, v, g, segment_ids, nn_lookup_spot_ind, s_target, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix, inds_target, n_iter=10, n_iter_sinkhorn=64, verb=False, test=False, beta=0.01, beta_increase=1.0): 
    
    # define constant
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0.], dtype=tf.float64)

    # unpack target indices
    i_grn_target, i_ang_target, i_hkl_target, i_det_target, i_all_target = inds_target

    # init q
    # d_target, i_target = nn_lookup_dist(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)

    # i_target_uv, i_target_ui = tf.unique(tf.reshape(s_target, (-1,)))
    # i_target = tf.reshape(tf.gather(i_target_uv, i_target_ui), shape=i_target.shape)
    i_target_flat = tf.reshape(i_target, (-1,))
    n_target = len(np.unique(i_target_flat))

    # full version (no neighbours)
    i_target_full = np.repeat(np.arange(s_target.shape[0]).reshape(1,-1), p.shape[0], axis=0)
    p_full = tf.gather(s_target, i_target_full)
    k = p.shape[1]
    v_flat = tf.repeat(v, k, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k, axis=0)


    v_flat_full = tf.repeat(v, p_full.shape[1], axis=0)
    segment_ids_flat_full = tf.repeat(segment_ids, p_full.shape[1], axis=0)


    # init w
    w = tf.einsum('bij, bj -> bi', a, v)
    w_sparse = tf.einsum('bij, bj -> bi', a, v)

    alpha = 1e-6
    eps = 1e-20
    beta = 0.01
    beta_increase = 1.00

    loss = []
    loss_sparse = []

    lam_init = 1000
    eps_init = 1e-3
    lam_decrease = 0.97

    # init sparse solution
    x_sparse = x
    a_sparse = a

    for j in range(n_iter): 

        lam_current = lam_current*lam_decrease if j > 0 else lam_init

        ot_reg=lam_current
        ot_reg_m_kl=eps_init



        ## -------------- seg version

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x_sparse)
        p_bar = p - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w_sparse)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        s = tf.expand_dims(x_rot, axis=1) + l

        # update Q

        sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
        Q_sparse = tf.math.exp( -sig_sq/lam_current ) 

        k_ind = i_target_flat
        i_ind = tf.repeat(np.arange(s.shape[0], dtype=np.int32), s.shape[1], axis=0)
        indices = tf.concat([tf.expand_dims(i_ind, axis=-1), tf.expand_dims(k_ind, axis=-1)], axis=-1)

        Q_scatter = tf.zeros((s.shape[0], s_target.shape[0]), dtype=tf.float64) #+ eps
        Q_scatter = tf.tensor_scatter_nd_add(Q_scatter, indices, tf.reshape(Q_sparse, (-1,)))
        

        ## -------------- full version

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar_full = p_full - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r_full = tf.einsum('sni, si -> sn', p_bar_full, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l_full = tf.expand_dims(b, axis=1) * tf.expand_dims(r_full, axis=-1)
        s_full = tf.expand_dims(x_rot, axis=1) + l_full

        # update Q

        sig_sq_full = tf.reduce_sum((p_full - s_full)**2, axis=-1)
        Q_full = tf.math.exp( -sig_sq_full/lam_current )
        
        ang_mask = i_ang == tf.transpose(i_ang_target, [1,0])
        det_mask = i_det == tf.transpose(i_det_target, [1,0])
        mask = det_mask & ang_mask

        # test different uot settings

        # if test:

        #     grid_reg = [0.0001, 0.001, 0.01, 0.1, 1]
        #     grid_eps = [0, 0.0001, 0.001, 0.01, 0.1, 1]
        #     for i, reg_lam in enumerate(grid_reg):
        #         for j, reg_eps in enumerate(grid_reg):

        #             ot_a=np.ones(len(p))/len(p)
        #             ot_b=np.ones(len(s_target))/len(s_target)
        #             ot_M=logQ_full.numpy()
        #             ot_reg=reg_lam
        #             ot_reg_m_kl=reg_eps
        #             entropic_kl_uot, log_uot = optimal_transport.sinkhorn_knopp_unbalanced(ot_a, ot_b, ot_M, ot_reg, ot_reg_m_kl, K_mask=mask, log=True)

        #             store_arrays(f'uot_eps{j}_lam{i}.h5', Q_full=logQ_full, entropic_kl_uot=entropic_kl_uot, log_uot_err=log_uot['err'], log_uot=np.array(log_uot['err']), reg=np.array([ot_reg, ot_reg_m_kl]))

        # nn = 0
        # diff =  p[:,nn,:]-s_target
        # diff_tot = np.sum(np.abs(diff), axis=1)
        # select = diff_tot>1e-8
        # print('bad ids',np.nonzero(select))
        # print('s_target bad', s_target[select].numpy())
        # print('p[:,0,:] bad', p[:,0,:][select].numpy())
        # print('i_ang bad', i_ang[select].numpy().ravel())
        # print('i_det bad', i_det[select].numpy().ravel())
        # print('i_grn bad', i_grn[select].numpy().ravel())


        # full matrix Sinkhorn-Knopp using POT

        ot_a=np.ones(len(p))/len(p)
        ot_b=np.ones(len(s_target))/len(s_target)
        M_full=sig_sq_full.numpy()

        time_start = time.time()
        
        Q_sinkhorn_full, ot_log = optimal_transport.sinkhorn_knopp_unbalanced(ot_a, ot_b, np.array(M_full), ot_reg, ot_reg_m_kl, K_mask=mask, log=True)
        Q_sinkhorn_sparse, Q_sinkhorn_scatter = optimal_transport.sinkhorn_knopp_unbalanced_full_vs_sparse(ot_a, ot_b, np.array(Q_scatter), np.array(Q_sparse), n_target, i_target, ot_reg, ot_reg_m_kl, K_mask=mask, log=True)

        time_elapsed = time.time()-time_start

        Q_norm = tf.reduce_sum(Q_sinkhorn_full, axis=1)
        Q_norm = tf.math.segment_sum(Q_norm, segment_ids=segment_ids)
        Q_norm = tf.gather(Q_norm, segment_ids)
        Qn = Q_sinkhorn_full/tf.expand_dims(Q_norm, axis=-1)

        Q_norm_sparse = tf.reduce_sum(Q_sinkhorn_sparse, axis=1)
        Q_norm_sparse = tf.math.segment_sum(Q_norm_sparse, segment_ids=segment_ids)
        Q_norm_sparse = tf.gather(Q_norm_sparse, segment_ids)
        Qn_sparse = Q_sinkhorn_sparse/tf.expand_dims(Q_norm_sparse, axis=-1)


        if tf.math.reduce_any(tf.math.is_nan(Qn)):

            import ipdb; ipdb.set_trace(); 
            pass

        diff = tf.math.sqrt(tf.reduce_mean((s_full-p_full)**2, axis=-1))
        loss_q = tf.reduce_sum(Qn*diff, axis=-1)
        loss_q = tf.math.segment_sum( loss_q, segment_ids=segment_ids)
        loss.append(loss_q)

        diff_sparse = tf.math.sqrt(tf.reduce_mean((s-p)**2, axis=-1))
        loss_q_sparse = tf.reduce_sum(Qn_sparse*diff_sparse, axis=-1)
        loss_q_sparse = tf.math.segment_sum( loss_q_sparse, segment_ids=segment_ids)
        loss_sparse.append(loss_q_sparse)

        if verb:
                
            Qna = np.array(Qn)
            Qn_nodiag = Qna.copy()
            np.fill_diagonal(Qn_nodiag, 0)
            if Qn.shape[0] == Qn.shape[1]:
                diagonality = np.mean(np.diag(Qna))/((np.mean(np.sum(Qn_nodiag, axis=0))+np.mean(np.sum(Qn_nodiag, axis=1)))/2)
            else:
                diagonality = np.nan

            print(f'======> iter {j:>10d}   loss_q={np.sum(loss_q): 12.6e}      loss_q_sparse={np.sum(loss_q_sparse): 12.6e}   Sinkhorn done in {time_elapsed:2.4f} sec, sum(Q_full)={np.sum(Q_sinkhorn_full):e}  sum(Q_sparse)={np.sum(Q_sinkhorn_sparse):e}    lam={lam_current:6.4e}      diagonality={diagonality:12.6e}')

        # update x

        # x = tf.einsum('sn, sji, snj -> si', Q, g, p_full - l_full)
        x = tf.einsum('sn, sji, snj -> si', Qn, g, p_full - l_full)
        x = tf.math.segment_sum(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)


        x_sparse = tf.einsum('sn, sji, snj -> si', Qn_sparse, g, p - l)
        x_sparse = tf.math.segment_sum(x_sparse, segment_ids=segment_ids)
        x_sparse = tf.gather(x_sparse, segment_ids)

        # update a
    
        # calculate q
        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar_full = p_full - tf.expand_dims(x_rot, axis=1)

        x_rot = tf.einsum('sij, sj -> si', g, x_sparse)
        p_bar_sparse = p - tf.expand_dims(x_rot, axis=1)

        # Rhombohedron solution
        p_norm_full = p_bar_full/tf.linalg.norm(p_bar_full, axis=-1, keepdims=True)
        q = p_norm_full - e
        q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q_full, shape=(-1,))
        Qn_flat = tf.reshape(Qn, shape=(-1,))

        a = segment_wahba_svd_weighted(q_flat, v_flat_full, Qn_flat, segment_ids_flat_full)
        a = tf.gather(a, segment_ids)

        # Rhombohedron solution
        p_norm_sparse = p_bar_sparse/tf.linalg.norm(p_bar_sparse, axis=-1, keepdims=True)
        q = p_norm_sparse - e
        q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q_sparse, shape=(-1,))
        Qn_flat = tf.reshape(Qn_sparse, shape=(-1,))

        a_sparse = segment_wahba_svd_weighted(q_flat, v_flat, Qn_flat, segment_ids_flat)
        a_sparse = tf.gather(a_sparse, segment_ids)

        # rotate basis
        w = tf.einsum('bij, bj -> bi', a, v)

        w_sparse = tf.einsum('bij, bj -> bi', a_sparse, v)


        # reduce beta, annealing
        beta *= beta_increase


    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)


    x_sparse = tf.math.segment_mean(x_sparse, segment_ids=segment_ids)
    a_sparse = tf.math.segment_mean(a_sparse, segment_ids=segment_ids)

    return loss, x, a, Q_sinkhorn_full, None, s_full, (loss_sparse, x_sparse, a_sparse, Q_sinkhorn_sparse, s)



def batch_optimize_coordinate_descent_sinkhorn_unbalanced_sparse(p, x, s, a, v, g, segment_ids, s_target, inds_model, lookup_data, n_iter=10, control_params=(1e3, 0.99, 0.1, 1e-3, 1e-3, 64), verb=False, test=False): 

    from laueotx.utils import logging as utils_logging
    LOGGER = utils_logging.get_logger(__file__)

    # unpack params
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data 
    eps_init, eps_decrease, eps_min, lam_obs, lam_mod, n_iter_sinkhorn = control_params
    i_grn, i_ang, i_hkl, i_det, i_gpl, n_hkl = inds_model
    
    # define constant
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0.], dtype=tf.float64)

    # get neighbours
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)
    i_target_flat = tf.reshape(i_target, (-1,))
    i_target_nn_inds, i_target_nn_flat = tf.unique(i_target_flat)
    i_target_nn = tf.reshape(i_target_nn_flat, shape=i_target.shape)
    n_target_nn = len(i_target_nn_inds)
    n_target = len(s_target)

    # other useful variables
    n_model, k_nn, n_dims = p.shape
    v_flat = tf.repeat(v, k_nn, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k_nn, axis=0)
    seg_ones = tf.ones(n_model, dtype=p.dtype)
    i_model = tf.repeat(tf.expand_dims(tf.range(len(s), dtype=tf.int32), axis=-1), k_nn, axis=0)
    i_gpl_uv, i_gpl_ui = tf.unique(i_gpl[:,0])
    v_agg = tf.math.unsorted_segment_mean(v, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))
    ot_a=tf.ones(len(p), dtype=tf.float64)
    ot_b=tf.ones(n_target_nn, dtype=tf.float64)

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    # init optimization
    eps_current = eps_init
    loss_sparse = []
    ranger = range if verb else lambda n: LOGGER.progressbar(range(n), at_level='info', desc='running double assignment coordinate descent')
    for j in ranger(n_iter): 

        # update r    
        s, l = coord_descent_update_r(g, x, p, w, e, I_eye)

        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - tf.expand_dims(x_rot, axis=1)
        # u = tf.einsum('sij, sj -> si', g, w)
        # H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        # b = tf.einsum('sij, j -> si', H, e)
        # r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        # l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        # s = tf.expand_dims(x_rot, axis=1) + l

        # update Q with unbalanced entropic Sinkhorn 

        l2_sq = tf.reduce_sum((p-s)**2, axis=-1)
        Q_sparse = tf.math.exp( -l2_sq/eps_current ) 

        Q_sinkhorn_sparse = optimal_transport.sinkhorn_knopp_unbalanced_sparse(ot_a, ot_b, Q_sparse, i_target=i_target_nn, i_model=i_model, reg=eps_current, reg_ma=lam_mod, reg_mb=lam_obs)

        # re-normalize weights for each model grain

        Q_norm_sparse = tf.reduce_sum(Q_sinkhorn_sparse, axis=1)
        Q_norm_sparse = tf.math.segment_sum(Q_norm_sparse, segment_ids=segment_ids)
        Q_norm_sparse = tf.gather(Q_norm_sparse, segment_ids)
        Qn_sparse = Q_sinkhorn_sparse/tf.expand_dims(Q_norm_sparse, axis=-1)

        if j==0:
            Q_init_sparse = Q_sparse

        # compute loss

        diff_sparse = tf.math.sqrt(tf.reduce_mean((s-p)**2, axis=-1))
        loss_q_sparse = tf.reduce_sum(Qn_sparse*diff_sparse, axis=-1)
        loss_q_sparse = tf.math.segment_sum( loss_q_sparse, segment_ids=segment_ids)
        loss_sparse.append(loss_q_sparse)

        # report

        if verb:
            print(f'======> iter {j:>10d}   loss_q_sparse={np.sum(loss_q_sparse): 12.6e}   Sinkhorn done in {time_elapsed:2.4f} sec, sum(Q_sparse)={np.sum(Q_sinkhorn_sparse):e}    eps={eps_current:6.4e} ')

        # update grain position

        x = tf.einsum('sn, sji, snj -> si', Qn_sparse, g, p - l)
        x = tf.math.segment_sum(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update grain orientation
        q = coord_descent_update_q(g, x, p, e)

        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar_sparse = p - tf.expand_dims(x_rot, axis=1)

        # # Rhombohedron solution
        # p_norm_sparse = p_bar_sparse/tf.linalg.norm(p_bar_sparse, axis=-1, keepdims=True)
        # q = p_norm_sparse - e
        # q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        # q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q_sparse, shape=(-1,))
        Qn_flat = tf.reshape(Qn_sparse, shape=(-1,))

        a_agg = segment_aggregate_wahba_svd_weighted(v_agg, q, Qn_sparse, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones)
        a = tf.gather(a_agg, segment_ids)

        # rotate basis to the updated grain orientation

        w = tf.einsum('bij, bj -> bi', a, v)

        # decrease tempreature

        eps_current = eps_current*eps_decrease

        # if eps_current<eps_min:

        #     break

    # calculate the final grain position and orientation
    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    # select only grains which have received at least half of the spot weights
    # frac_weight_accept = 0.5
    # n_spots_per_grain = tf.math.segment_sum(tf.ones(len(segment_ids), dtype=tf.float64), segment_ids)
    # Q_grain = tf.math.segment_sum(tf.reduce_sum(Q_sinkhorn_sparse, axis=1), segment_ids)
    # select = Q_grain > (n_spots_per_grain * frac_weight_accept)
    # x = x[select]
    # a = a[select]
    # select = tf.gather(select, segment_ids)
    # Q_sinkhorn_sparse = Q_sinkhorn_sparse[select]
    # i_target = i_target[select]
    # Q_init_sparse = Q_init_sparse[select]
    # s = s[select]
    # segment_ids = segment_ids[select]


    return loss_sparse, x, a, Q_sinkhorn_sparse, None, s, (i_target, Q_init_sparse)





def batch_optimize_coordinate_descent_sinkhorn_partial_full(p, x, s, a, v, g, segment_ids, nn_lookup_spot_ind, s_target, inds_model, inds_target, nn_lookup_pix_size, nn_lookup_n_pix, n_iter=10, n_iter_sinkhorn=64, verb=False, test=False, lam_init=1e3, lam_decrease=0.99, lam_min=0.1, eps_obs=1e-3, eps_mod=1e-3): 

    # define constant
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0.], dtype=tf.float64)

    s_init = np.array(s).copy()
    i_grn_target, i_ang_target, i_hkl_target, i_det_target, i_all_target = inds_target
    i_grn, i_ang, i_hkl, i_det = inds_model


    # init q
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    p = tf.gather(s_target, i_target)

    i_target_flat = tf.reshape(i_target, (-1,))
    n_target = len(np.unique(i_target_flat))

    # full version (no neighbours)
    k = p.shape[1]
    v_flat = tf.repeat(v, k, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k, axis=0)

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)


    loss_sparse = []

    # get lambda
    lam_current = lam_init

    ranger = range if verb else trange
    for j in ranger(n_iter): 

        # update r    

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar = p - tf.expand_dims(x_rot, axis=1)
        u = tf.einsum('sij, sj -> si', g, w)
        H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        b = tf.einsum('sij, j -> si', H, e)
        r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        s = tf.expand_dims(x_rot, axis=1) + l

        # update Q with unbalanced entropic Sinkhorn 

        l2_sq = tf.reduce_sum((p-s)**2, axis=-1)
        Q_nn = tf.math.exp( -l2_sq/lam_current ) 
        time_start = time.time()


        # make full matrix from neighbours

        k_ind = i_target_flat
        i_ind = tf.repeat(np.arange(s.shape[0], dtype=np.int32), s.shape[1], axis=0)
        indices = tf.cast(tf.concat([tf.expand_dims(i_ind, axis=-1), tf.expand_dims(k_ind, axis=-1)], axis=-1), tf.int64)

        Q_scatter = tf.zeros((s.shape[0], s_target.shape[0]), dtype=tf.float64) #+ eps
        Q_scatter = tf.tensor_scatter_nd_add(Q_scatter, indices, tf.reshape(Q_nn, (-1,)))

        # grain-to-spot cost
        ang_mask = i_ang == tf.transpose(i_ang_target, [1,0])
        det_mask = i_det == tf.transpose(i_det_target, [1,0])
        mask = det_mask & ang_mask
        Q_scatter = np.where(mask, Q_scatter, 0)
        Q_gs = tf.math.segment_sum(Q_scatter, segment_ids)
    
        ot_a = tf.math.segment_sum(tf.ones(len(segment_ids), dtype=tf.float64), segment_ids)/n_target
        ot_b = tf.ones(len(s_target), dtype=tf.float64)/n_target

        Q_sinkhorn_full, Q_sinkhorn_sparse = optimal_transport.entropic_partial_wasserstein_full_vs_sparse(ot_a, ot_b, K=Q_gs, K_sparse=Q_nn, i_target=i_target, segment_ids=segment_ids, reg=1e-3, K_mask=mask, m=1, verbose=True)

        if j==0:
            Q_init_sparse = Q_sparse

        # re-normalize weights for each model grain

        Q_norm_sparse = tf.reduce_sum(Q_sinkhorn_sparse, axis=1)
        Q_norm_sparse = tf.math.segment_sum(Q_norm_sparse, segment_ids=segment_ids)
        Q_norm_sparse = tf.gather(Q_norm_sparse, segment_ids)
        Qn_sparse = Q_sinkhorn_sparse/tf.expand_dims(Q_norm_sparse, axis=-1)

        # compute loss

        diff_sparse = tf.math.sqrt(tf.reduce_mean((s-p)**2, axis=-1))
        loss_q_sparse = tf.reduce_sum(Qn_sparse*diff_sparse, axis=-1)
        loss_q_sparse = tf.math.segment_sum( loss_q_sparse, segment_ids=segment_ids)
        loss_sparse.append(loss_q_sparse)

        # update grain position

        x = tf.einsum('sn, sji, snj -> si', Qn_sparse, g, p - l)
        x = tf.math.segment_sum(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update grain orientation

        x_rot = tf.einsum('sij, sj -> si', g, x)
        p_bar_sparse = p - tf.expand_dims(x_rot, axis=1)

        # Rhombohedron solution
        p_norm_sparse = p_bar_sparse/tf.linalg.norm(p_bar_sparse, axis=-1, keepdims=True)
        q = p_norm_sparse - e
        q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q_sparse, shape=(-1,))
        Qn_flat = tf.reshape(Qn_sparse, shape=(-1,))

        a = segment_wahba_svd_weighted(q_flat, v_flat, Qn_flat, segment_ids_flat)
        a = tf.gather(a, segment_ids)

        # rotate basis to the updated grain orientation

        w = tf.einsum('bij, bj -> bi', a, v)

        # decrease tempreature

        lam_current = lam_current*lam_decrease

        if lam_current<lam_min:

            break

    # calculate final loss

    s, l = coord_descent_update_r(g, x, p, w, e, I_eye)
    diff_sparse = tf.math.sqrt(tf.reduce_mean((s-p)**2, axis=-1))
    loss_q_sparse = tf.reduce_sum(Qn_sparse*diff_sparse, axis=-1)
    loss_q_sparse = tf.math.segment_sum( loss_q_sparse, segment_ids=segment_ids)
    loss_sparse.append(loss_q_sparse)

    # calculate the final grain position and orientation

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    # select all here, no cuts
    select = np.ones(len(p), dtype=tf.bool)
    return loss_sparse, x, a, Q_sinkhorn_sparse, select, s, (i_target, Q_init_sparse)




def get_tf_sparse_cost_matrix_with_slacks():

    i_tgt_flat = tf.reshape(i_target, (-1,))
    i_mod_flat = tf.repeat(np.arange(K_sparse.shape[0], dtype=np.int32), K_sparse.shape[1], axis=0)
    i_tgt_slacks = tf.concat([tf.ones(len(a)-1, dtype=tf.int32)*(len(b)-1), tf.range(len(b)-1, dtype=tf.int32)], axis=0)
    i_mod_slacks = tf.concat([tf.range(len(a)-1, dtype=tf.int32),           tf.ones(len(b)-1, dtype=tf.int32)*(len(a)-1)], axis=0)
    i_tgt_flat = tf.concat([i_tgt_flat, i_tgt_slacks, tf.constant(len(b)-1, shape=(1,))], axis=0)
    i_mod_flat = tf.concat([i_mod_flat, i_mod_slacks, tf.constant(len(a)-1, shape=(1,))], axis=0)
    i_sparse = tf.cast(tf.concat([tf.expand_dims(i_mod_flat, axis=-1), tf.expand_dims(i_tgt_flat, axis=-1)], axis=-1), tf.int64)
    K_slacks = tf.concat([tf.ones(len(a)-1, dtype=tf.float64)*tf.math.exp(-slack_cost_a), tf.ones(len(b)-1, dtype=tf.float64)*tf.math.exp(-slack_cost_b)], axis=0)
    K_sparse_flat = tf.concat([tf.reshape(K_sparse, (-1,)), K_slacks, tf.constant(slack_cost_corner, shape=(1,), dtype=tf.float64)], axis=0)
    Ks = tf.sparse.SparseTensor(values=K_sparse_flat, indices=i_sparse, dense_shape=(aa.shape[0], bb.shape[0]))
    Ks = tf.sparse.reorder(Ks)
    Ksd = tf.sparse.to_dense(Ks)








def batch_optimize_coordinate_descent_sinkhorn_slacks_sparse(p, x, s, a, v, g, segment_ids, s_target, inds_model, lookup_data, n_iter=10, control_params=(1e3, 0.99, 0.1, 0.75, 0, 0, 1000), n_iter_sinkhorn=64, verb=False, test=False):

    from laueotx.utils import logging as utils_logging
    LOGGER = utils_logging.get_logger(__file__)

    # unpack params
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data 
    eps_init, eps_decrease, eps_min, c_outlier, n_max_unmatched, n_max_outliers, n_iter_sinkhorn = control_params
    i_grn, i_ang, i_hkl, i_det, i_gpl, n_hkl = inds_model

    # get neighbours
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    i_target_flat = tf.reshape(i_target, (-1,))
    i_target_nn_inds, i_target_nn_flat = tf.unique(i_target_flat)
    i_target_nn = tf.reshape(i_target_nn_flat, shape=i_target.shape)
    n_target_nn = len(i_target_nn_inds)

    p = tf.gather(s_target, i_target)
    n_model, k_nn, n_dims = p.shape
    n_target = len(s_target)
    v_flat = tf.repeat(v, k_nn, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k_nn, axis=0)
    seg_ones = tf.ones(n_model, dtype=p.dtype)
    i_gpl_uv, i_gpl_ui = tf.unique(i_gpl[:,0])
    v_agg = tf.math.unsorted_segment_mean(v, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))
    i_model = tf.repeat(tf.expand_dims(tf.range(len(s), dtype=tf.int32), axis=-1), k_nn, axis=0)

    # define constant
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0.], dtype=tf.float64)
    s_init = np.array(s).copy()
    ot_a=np.ones(len(p)+1, dtype=np.float64)
    ot_b=np.ones(n_target_nn+1, dtype=np.float64)

    # fix the number of outliers and unmatched rays so that the sum or rows and columns agree
    ot_b[-1] = n_max_unmatched 
    ot_a[-1] = ot_b[-1] + n_target_nn - n_model
    if ot_b[-1]<0 or ot_a[-1]<0:
        raise Exception(f"maximum number of outliers vs maximum number of unmatched rays does not agree, n_model_spots={n_model} n_target_spots={n_target_nn} n_max_unmatched={n_max_unmatched} n_max_outliers={n_max_outliers}")

    ot_a = tf.constant(ot_a, dtype=tf.float64)
    ot_b = tf.constant(ot_b, dtype=tf.float64)
    ot_outlier_cost_a = tf.constant(c_outlier, dtype=tf.float64)
    ot_outlier_cost_b = tf.constant(0., dtype=tf.float64)
    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    # import pudb; pudb.set_trace();
    # pass

    # coo_matrix((data, (i, j)), [shape=(M, N)])
    # M = scipy.sparse.coo_matrix(            np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]])       )
    # In [32]: Mcsr = M.tocsr() 
    # In [33]: np.split(Mcsr.indices, Mcsr.indptr)    
    # In [34]: Mcsc = M.tocsc() 
    # In [35]: np.split(Mcsc.indices, Mcsc.indptr
    # from scipy.sparse import coo_matrix 
    # Mcoo = coo_matrix((np.reshape(Q_nn, (-1,)), (tf.reshape(i_model, (-1,)), tf.reshape(i_target, (-1,)))), shape=(n_model, n_target_nn))
    # Mcsc = Mcoo.tocsc()
    # list_c = np.split(Mcsc.indices, Mcsc.indptr)    
    # Mcsr = Mcoo.tocsr()
    # list_r = np.split(Mcsr.indices, Mcsr.indptr)    



    # sort the indices to enable sparse tensor ops without memory re-writing
    
    # i_tgt_flat = tf.reshape(i_target_nn, (-1,))
    # i_model_flat = tf.repeat(np.arange(n_model, dtype=np.int32), k_nn, axis=0)
    # i_tgt_slacks = tf.concat([tf.ones(len(ot_a)-1, dtype=tf.int32)*(len(ot_b)-1), tf.range(len(ot_b)-1, dtype=tf.int32)], axis=0)
    # i_mod_slacks = tf.concat([tf.range(len(ot_a)-1, dtype=tf.int32), tf.ones(len(ot_b)-1, dtype=tf.int32)*(len(ot_a)-1)], axis=0)
    # i_tgt_flat = tf.concat([i_tgt_flat, i_tgt_slacks, tf.constant(len(ot_b)-1, shape=(1,))], axis=0)
    # i_mod_flat = tf.concat([i_mod_flat, i_mod_slacks, tf.constant(len(ot_a)-1, shape=(1,))], axis=0)
    # indices_sparse = tf.cast(tf.concat([tf.expand_dims(i_mod_flat, axis=-1), tf.expand_dims(i_tgt_flat, axis=-1)], axis=-1), tf.int64)
    # reorder_sorting = tf.constant(np.argsort(indices_sparse[:,0]*10000000 + indices_sparse[:,1]))
    # inverse_sorting = tf.constant(np.argsort(reorder_sorting))

    # start optimization
    eps_current = eps_init
    loss_sparse = []
    ranger = range if verb else lambda n: LOGGER.progressbar(range(n), at_level='info', desc='running double assignment coordinate descent')
    for j in ranger(n_iter): 

        # update r    

        s, l = coord_descent_update_r(g, x, p, w, e, I_eye)

        # update Q with unbalanced entropic Sinkhorn 

        # l2_sq = tf.reduce_sum((p-s)**2, axis=-1)
        # Q_nn = tf.math.exp( -l2_sq/eps_current ) + 1e-20
        Q_nn = coord_descent_Qnn(p, s, eps_current)
        # Q_sinkhorn_sparse = optimal_transport.sinkhorn_knopp_slacks_sparse_v2(ot_a, ot_b, ot_outlier_cost_a, ot_outlier_cost_b, K_nn=Q_nn, indices_sparse=indices_sparse, reorder_sorting=reorder_sorting, inverse_sorting=inverse_sorting)
        # Q_sinkhorn_sparse = optimal_transport.sinkhorn_knopp_slacks_full_vs_sparse(ot_a, ot_b, ot_outlier_cost_a, ot_outlier_cost_b, K_sparse=Q_nn, i_target=i_target_nn)
        Q_sinkhorn_sparse = optimal_transport.sinkhorn_knopp_slacks_sparse(ot_a, ot_b, ot_outlier_cost_a, ot_outlier_cost_b, K_sparse=Q_nn, i_target=i_target_nn, i_model=i_model)
        

        if j == 0:
            Q_init_nn = Q_nn

        # re-normalize weights for each model grain

        Q_norm_sparse = tf.reduce_sum(Q_sinkhorn_sparse, axis=1)
        Q_norm_sparse = tf.math.segment_sum(Q_norm_sparse, segment_ids=segment_ids)
        Q_norm_sparse = tf.gather(Q_norm_sparse, segment_ids)
        Qn_sparse = Q_sinkhorn_sparse/tf.expand_dims(Q_norm_sparse, axis=-1)

        # compute loss
        if j == 0:
            diff_sparse = tf.math.sqrt(tf.reduce_mean((s-p)**2, axis=-1))
            loss_q_sparse = tf.reduce_sum(Qn_sparse*diff_sparse, axis=-1)
            loss_q_sparse = tf.math.segment_sum( loss_q_sparse, segment_ids=segment_ids)
            loss_sparse.append(loss_q_sparse)

        # report

        if verb:
            print(f'======> iter {j:>10d}   loss_q_sparse={np.sum(loss_q_sparse): 12.6e}   Sinkhorn done in {time_elapsed:2.4f} sec, sum(Q_sparse)={np.sum(Q_sinkhorn_sparse):e}    lam={eps_current:6.4e} ')

        # update grain position

        x = tf.einsum('sn, sji, snj -> si', Qn_sparse, g, p - l)
        x = tf.math.segment_sum(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update grain orientation

        q = coord_descent_update_q(g, x, p, e)
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q_nn, shape=(-1,))
        Qn_flat = tf.reshape(Qn_sparse, shape=(-1,))
        a_agg = segment_aggregate_wahba_svd_weighted(v_agg, q, Qn_sparse, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones)
        a = tf.gather(a_agg, segment_ids)

        # rotate basis to the updated grain orientation

        w = tf.einsum('bij, bj -> bi', a, v)

        # decrease tempreature

        eps_current = eps_current*eps_decrease

        # if eps_current<eps_min:
        #     break

    # calculate final loss

    s, l = coord_descent_update_r(g, x, p, w, e, I_eye)
    diff_sparse = tf.math.sqrt(tf.reduce_mean((s-p)**2, axis=-1))
    loss_q_sparse = tf.reduce_sum(Qn_sparse*diff_sparse, axis=-1)
    loss_q_sparse = tf.math.segment_sum( loss_q_sparse, segment_ids=segment_ids)
    loss_sparse.append(loss_q_sparse)

    # calculate the final grain position and orientation

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    # select only grains which have received at least half of the spot weights
    # seg_count = tf.math.segment_sum(seg_ones, segment_ids)
    # frac_weight_accept = 0.5
    # Q_grain = tf.math.segment_sum(tf.reduce_sum(Q_sinkhorn_sparse, axis=1), segment_ids)
    # select = Q_grain > (seg_count * frac_weight_accept)
    # x = x[select]
    # a = a[select]
    # select = tf.gather(select, segment_ids)
    # Q_sinkhorn_sparse = Q_sinkhorn_sparse[select]
    # i_target = i_target[select]
    # Q_init_nn = Q_init_nn[select]
    # s = s[select]
    # segment_ids = segment_ids[select]
    # if verb:
    #     print('fraction of matched spots', np.sum(Q_grain)/ np.sum(seg_count))

    return loss_sparse, x, a, Q_sinkhorn_sparse, None, s, (i_target, Q_init_nn)



def batch_optimize_coordinate_descent_sinkhorn_partial_sparse(p, x, s, a, v, g, segment_ids, s_target, inds_model, lookup_data, n_iter=10, control_params=(1e3, 0.99, 0.1, 0.75, 64), n_iter_sinkhorn=64, verb=False, test=False): 

    from laueotx.utils import logging as utils_logging
    LOGGER = utils_logging.get_logger(__file__)

    # unpack params
    nn_lookup_spot_ind, nn_lookup_pix_size, nn_lookup_n_pix = lookup_data 
    eps_init, eps_decrease, eps_min, ot_m, n_iter_sinkhorn = control_params
    i_grn, i_ang, i_hkl, i_det, i_gpl, n_hkl = inds_model

    # define constant
    I_eye = tf.eye(3, dtype=tf.float64)
    e = tf.constant([1.,0.,0.], dtype=tf.float64)
    n_target = len(s_target)
    n_model = len(p)
    seg_ones = tf.ones(n_model, dtype=p.dtype)
    s_init = np.array(s).copy()

    # get neighbours
    i_target = nn_lookup_all(nn_lookup_spot_ind, s_target, s, i_ang, i_det, i_grn, nn_lookup_pix_size, nn_lookup_n_pix)
    i_target_flat = tf.reshape(i_target, (-1,))
    i_target_nn_inds, i_target_nn_flat = tf.unique(i_target_flat)
    i_target_nn = tf.reshape(i_target_nn_flat, shape=i_target.shape)
    n_target_nn = len(i_target_nn_inds)

    p = tf.gather(s_target, i_target)
    n_model, k_nn, n_dims = p.shape
    v_flat = tf.repeat(v, k_nn, axis=0)
    segment_ids_flat = tf.repeat(segment_ids, k_nn, axis=0)
    i_model = tf.repeat(tf.expand_dims(tf.range(len(s), dtype=tf.int32), axis=-1), k_nn, axis=0)
    i_gpl_uv, i_gpl_ui = tf.unique(i_gpl[:,0])
    v_agg = tf.math.unsorted_segment_mean(v, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))
    ot_a = tf.math.segment_sum(seg_ones, segment_ids)/n_target_nn
    ot_b = tf.ones(n_target_nn, dtype=tf.float64)/n_target_nn
    ot_m = tf.constant(ot_m, dtype=tf.float64)

    # init w
    w = tf.einsum('bij, bj -> bi', a, v)

    # start optimization
    eps_current = eps_init
    loss_sparse = []
    ranger = range if verb else lambda n: LOGGER.progressbar(range(n), at_level='info', desc='running double assignment coordinate descent')
    for j in ranger(n_iter): 

        # update r    

        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar = p - tf.expand_dims(x_rot, axis=1)
        # u = tf.einsum('sij, sj -> si', g, w)
        # H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
        # b = tf.einsum('sij, j -> si', H, e)
        # r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
        # l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
        # s = tf.expand_dims(x_rot, axis=1) + l

        s, l = coord_descent_update_r(g, x, p, w, e, I_eye)


        # update Q with unbalanced entropic Sinkhorn 

        # l2_sq = tf.reduce_sum((p-s)**2, axis=-1)
        # Q_nn = tf.math.exp( -l2_sq/eps_current )
        Q_nn = coord_descent_Qnn(p, s, eps_current)
        # time_start = time.time() 
        Q_sinkhorn_sparse = optimal_transport.entropic_partial_wasserstein_sparse(ot_a, ot_b, K_sparse=Q_nn, i_target=i_target_nn, i_model=i_model, i_grains=segment_ids, m=ot_m)
        # time_elapsed = time.time() - time_start

        if j == 0:

            Q_init_nn = Q_nn

        # re-normalize weights for each model grain

        Q_norm_sparse = tf.reduce_sum(Q_sinkhorn_sparse, axis=1)
        Q_norm_sparse = tf.math.segment_sum(Q_norm_sparse, segment_ids=segment_ids)
        Q_norm_sparse = tf.gather(Q_norm_sparse, segment_ids)
        Qn_sparse = Q_sinkhorn_sparse/tf.expand_dims(Q_norm_sparse, axis=-1)

        # compute loss

        if j == 0:

            diff_sparse = tf.math.sqrt(tf.reduce_mean((s-p)**2, axis=-1))
            loss_q_sparse = tf.reduce_sum(Qn_sparse*diff_sparse, axis=-1)
            loss_q_sparse = tf.math.segment_sum( loss_q_sparse, segment_ids=segment_ids)
            loss_sparse.append(loss_q_sparse)

        # report

        if verb:
            print(f'======> iter {j:>10d}   loss_q_sparse={np.sum(loss_q_sparse): 12.6e}   Sinkhorn done in {time_elapsed:2.4f} sec, sum(Q_sparse)={np.sum(Q_sinkhorn_sparse):e}    lam={eps_current:6.4e} ')

        # update grain position

        x = tf.einsum('sn, sji, snj -> si', Qn_sparse, g, p - l)
        x = tf.math.segment_sum(x, segment_ids=segment_ids)
        x = tf.gather(x, segment_ids)

        # update grain orientation
        q = coord_descent_update_q(g, x, p, e)

        # x_rot = tf.einsum('sij, sj -> si', g, x)
        # p_bar_sparse = p - tf.expand_dims(x_rot, axis=1)

        # # Rhombohedron solution
        # p_norm_sparse = p_bar_sparse/tf.linalg.norm(p_bar_sparse, axis=-1, keepdims=True)
        # q = p_norm_sparse - e
        # q = q/tf.linalg.norm(q, axis=-1, keepdims=True)
        # q = tf.einsum('sji, snj -> sni', g, q) # derotate q
        q_flat = tf.reshape(q, shape=(-1, q.shape[-1]))
        Q_flat = tf.reshape(Q_nn, shape=(-1,))
        Qn_flat = tf.reshape(Qn_sparse, shape=(-1,))

        # a = segment_wahba_svd_weighted(q_flat, v_flat, Qn_flat, segment_ids_flat)
        # a = tf.gather(a, segment_ids)

        a_agg = segment_aggregate_wahba_svd_weighted(v_agg, q, Qn_sparse, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones)
        a = tf.gather(a_agg, segment_ids)

        # rotate basis to the updated grain orientation

        w = tf.einsum('bij, bj -> bi', a, v)

        # decrease tempreature

        eps_current = eps_current*eps_decrease

        # if eps_current<eps_min:

        #     break


    # calculate final loss
    
    s, l = coord_descent_update_r(g, x, p, w, e, I_eye)
    diff_sparse = tf.math.sqrt(tf.reduce_mean((s-p)**2, axis=-1))
    loss_q_sparse = tf.reduce_sum(Qn_sparse*diff_sparse, axis=-1)
    loss_q_sparse = tf.math.segment_sum( loss_q_sparse, segment_ids=segment_ids)
    loss_sparse.append(loss_q_sparse)

    # calculate the final grain position and orientation

    x = tf.math.segment_mean(x, segment_ids=segment_ids)
    a = tf.math.segment_mean(a, segment_ids=segment_ids)

    # select only grains which have received at least half of the spot weights
    # frac_weight_accept = 0.5
    # Q_grain = tf.math.segment_sum(tf.reduce_sum(Q_sinkhorn_sparse, axis=1), segment_ids)
    # select = Q_grain > (ot_a * ot_m * frac_weight_accept)
    # x = x[select]
    # a = a[select]
    # select = tf.gather(select, segment_ids)
    # Q_sinkhorn_sparse = Q_sinkhorn_sparse[select]
    # i_target = i_target[select]
    # Q_init_nn = Q_init_nn[select]
    # s = s[select]
    # segment_ids = segment_ids[select]

    return loss_sparse, x, a, Q_sinkhorn_sparse, None, s, (i_target, Q_init_nn)




