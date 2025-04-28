# import ot
import warnings
import tensorflow as tf
import numpy as np
from laueotx.config import TF_FUNCTION_JIT_COMPILE

#################################################################################################################
##
## Common function
##
#################################################################################################################

@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def sinkhorn_error(u, v, u_prev_, v_prev_, i_model, i_target):

    u_ = tf.gather(u, tf.reshape(i_model, (-1,)))
    v_ = tf.gather(v, tf.reshape(i_target, (-1,)))
    err =  tf.sqrt(tf.reduce_sum((u_*v_ - u_prev_*v_prev_)**2)) / tf.sqrt(tf.reduce_sum((u_prev_*v_prev_)**2))

    return err, u_, v_



#################################################################################################################
##
## Unbalanced
##
#################################################################################################################



@tf.function()
def sinkhorn_knopp_unbalanced_sparse(a, b, K_sparse, i_target, i_model, reg, reg_ma, reg_mb, eps=1e-40, n_iter_max=1000, err_threshold=1e-8):

    err_threshold = tf.constant(err_threshold, dtype=tf.float64)
    n_iter_max = tf.constant(n_iter_max, dtype=tf.int32)

    dim_a, dim_b = len(a), len(b)
    u = tf.ones(dim_a, dtype=tf.float64) #/ dim_a
    v = tf.ones(dim_b, dtype=tf.float64) #/ dim_b

    fia = reg_ma / (reg_ma + reg)
    fib = reg_mb / (reg_mb + reg)
    K_sparse += eps

    u_ = tf.gather(u, indices=tf.reshape(i_target, shape=(-1,)))
    v_ = tf.gather(v, indices=tf.reshape(i_model, shape=(-1,)))

    err = tf.constant(1e10, dtype=tf.float64)
    i = 0
    while (i < n_iter_max) and (err > err_threshold):
        
        u_prev_ = u_
        v_prev_ = v_
        Kv_sparse = tf.reduce_sum(K_sparse * tf.gather(v, i_target), axis=1)
        u = tf.math.divide_no_nan(a, Kv_sparse)**fia
        Ku_sparse = K_sparse * tf.expand_dims(u, axis=-1)
        Ktu_sparse = tf.math.unsorted_segment_sum(data=tf.reshape(Ku_sparse, shape=(-1,)), segment_ids=tf.reshape(i_target, shape=(-1,)), num_segments=dim_b)
        v = tf.math.divide_no_nan(b, Ktu_sparse)**fib

        err, u_, v_ = sinkhorn_error(u, v, u_prev_, v_prev_, i_model, i_target)

    Q_sparse = tf.expand_dims(u, axis=-1) * K_sparse * tf.gather(v, i_target)

    return Q_sparse


#################################################################################################################
## Unbalanced - TESTS
#################################################################################################################


def sinkhorn_knopp_unbalanced(a, b, M, reg, reg_ma, reg_mb=None, K_mask=None, numItermax=1000, stopThr=1e-6, verbose=False, log=False, **kwargs):

    import ot
    from ot.utils import list_to_array
    from ot.backend import get_backend
    
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    if reg_mb is None:
        reg_mb = reg_ma

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, 1), type_as=M) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = nx.ones(dim_a, type_as=M) / dim_a
        v = nx.ones(dim_b, type_as=M) / dim_b

    K = nx.exp(M / (-reg))

    if K_mask is not None:
        K = nx.where(K_mask, K, 0)

    fia = reg_ma / (reg_ma + reg)
    fib = reg_mb / (reg_mb + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fia
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fib

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.
        )
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = nx.log(u + 1e-300)
        log['logv'] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u[:, None] * K * v[None, :], log
        else:
            return u[:, None] * K * v[None, :]


def sinkhorn_knopp_unbalanced_full_vs_sparse(a, b, K, K_sparse, n_target, i_target, reg, reg_ma, reg_mb=None, K_mask=None, numItermax=1000, stopThr=1e-6, verbose=False, log=False, **kwargs):

    import ot
    from ot.utils import list_to_array
    from ot.backend import get_backend

    K, a, b = list_to_array(K, a, b)
    nx = get_backend(K, a, b)

    i_target_flat = tf.reshape(i_target, shape=(-1,))

    if reg_mb is None:
        reg_mb = reg_ma

    dim_a, dim_b = K.shape
    dim_a, dim_b = K_sparse.shape[0], n_target

    n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    u = nx.ones(dim_a, type_as=K) / dim_a
    v = nx.ones(dim_b, type_as=K) / dim_b
    # M_reg = M / (-reg)
    # K = nx.exp(M_reg)
    # K = nx.where(K_mask, K, 0)

    u_sparse = tf.ones(dim_a, dtype=tf.float64) / dim_a
    v_sparse = tf.ones(dim_b, dtype=tf.float64) / dim_b

    # if K_mask is not None:
    #     K = nx.where(K_mask, K, 0)

    fia = reg_ma / (reg_ma + reg)
    fib = reg_mb / (reg_mb + reg)

    err = 1.

    # M_sparse = tf.constant(M_sparse)
    # M_sparse_reg = M_sparse / (-reg)
    # K_sparse = tf.math.exp(M_sparse_reg)

    K_sparse += 1e-20
    K += 1e-20


    for i in range(numItermax):
        
        uprev = u
        vprev = v
        u_sparse_prev = u_sparse
        v_sparse_prev = v_sparse

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fia
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fib

        Kv_sparse = tf.reduce_sum(K_sparse*tf.gather(v_sparse, i_target), axis=1)
        u_sparse = (a/Kv_sparse)**fia

        Ku_sparse = K_sparse*tf.expand_dims(u_sparse, axis=-1)
        Ku_sparse_flat = tf.reshape(Ku_sparse, shape=(-1,))
        Ktu_sparse = tf.math.unsorted_segment_sum(data=Ku_sparse_flat, segment_ids=i_target_flat, num_segments=n_target)
        v_sparse = (b / Ktu_sparse) ** fib


        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.)
        err_v = nx.max(nx.abs(v - vprev)) / max(nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.)
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break


        if tf.reduce_any(Ktu_sparse==0.) or tf.reduce_any(~tf.math.is_finite(u_sparse)) or tf.reduce_any(~tf.math.is_finite(v_sparse)):

            u_sparse = u_sparse_prev
            v_sparse = v_sparse_prev

            break

        err_u_sparse = tf.reduce_max(tf.math.abs(u_sparse - u_sparse_prev)) / tf.reduce_max( tf.concat([tf.reduce_max(tf.math.abs(u_sparse)), tf.reduce_max(tf.math.abs(u_sparse_prev)), 1.], axis=0))
        err_v_sparse = tf.reduce_max(tf.math.abs(v_sparse - v_sparse_prev)) / tf.reduce_max( tf.concat([tf.reduce_max(tf.math.abs(v_sparse)), tf.reduce_max(tf.math.abs(v_sparse_prev)), 1.], axis=0))    
        err_sparse = 0.5 * (err_u_sparse + err_v_sparse)

        if err_sparse < stopThr:

            break

    Q = u[:, None] * K * v[None, :]

    Q_sparse = tf.expand_dims(u_sparse, axis=-1) * K_sparse * tf.gather(v_sparse, i_target)

    if tf.reduce_any(tf.math.is_nan(Q_sparse)):

        import ipdb; ipdb.set_trace(); 
        pass

    # assert np.allclose(np.sum(Q), np.sum(Q_sparse)), f'sum(Q_dense)={np.sum(Q)} sum(Q_sparse)={np.sum(Q_sparse)}'

    return Q_sparse, Q



#################################################################################################################
##
## Softslacks
##
#################################################################################################################



@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def sinkhorn_slacks_update_u(K_sparse, a, v, Ka_slack, Kb_slack, i_target):

    Kv_sparse = tf.reduce_sum(K_sparse * tf.gather(v[:-1], i_target), axis=1)
    Kv_sparse = Kv_sparse + v[-1] * Ka_slack[:-1]
    Kv_slack = tf.reduce_sum(Kb_slack * v, keepdims=True)
    Kv_sparse_slack = tf.concat([Kv_sparse, Kv_slack], axis=0)

    u_sparse = tf.math.divide_no_nan(a, Kv_sparse_slack)

    return u_sparse


@tf.function(jit_compile=False)
def sinkhorn_slacks_update_v(K_sparse, b, u, Ka_slack, Kb_slack, i_target):

    Ku_sparse = tf.math.unsorted_segment_sum(data=tf.reshape(K_sparse * tf.expand_dims(u[:-1], axis=-1), shape=(-1,)), segment_ids=tf.reshape(i_target, (-1,)), num_segments=len(b)-1)
    v = accel_sinkhorn_update_v_part(Ku_sparse, b, u, Ka_slack, Kb_slack)

    return v

@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def accel_sinkhorn_update_v_part(Ku_sparse, b, u, Ka_slack, Kb_slack):
    
    Ku_sparse = Ku_sparse + u[-1] * Kb_slack[:-1]
    Ku_slack = tf.reduce_sum(Ka_slack * u, keepdims=True)
    Ku_sparse_slack = tf.concat([Ku_sparse, Ku_slack], axis=0)

    v_sparse = tf.math.divide_no_nan(b, Ku_sparse_slack)

    return v_sparse

@tf.function()
def sinkhorn_knopp_slacks_sparse(a, b, slack_cost_a, slack_cost_b, K_sparse, i_target, i_model, n_iter_max=1000, err_threshold=1e-3):


    slack_cost_corner = tf.constant(0., dtype=tf.float64, shape=(1,))
    err_threshold = tf.constant(err_threshold, dtype=tf.float64)
    n_iter_max = tf.constant(n_iter_max, dtype=tf.int32)

    dim_a, dim_b = len(a), len(b)
    u = tf.ones(dim_a, dtype=tf.float64) #/ dim_a
    v = tf.ones(dim_b, dtype=tf.float64) #/ dim_b
    K_sparse = K_sparse + 1e-20

    Ka_slack  = tf.ones(dim_a-1, dtype=tf.float64) * tf.math.exp(-slack_cost_a)
    Kb_slack  = tf.ones(dim_b-1, dtype=tf.float64) * tf.math.exp(-slack_cost_b)
    Kab_slack = tf.math.exp(-slack_cost_corner)
    Ka_slack = tf.concat([Ka_slack, Kab_slack], axis=0)
    Kb_slack = tf.concat([Kb_slack, Kab_slack], axis=0)

    u_ = tf.gather(u, tf.reshape(i_model, shape=(-1,)))
    v_ = tf.gather(v, tf.reshape(i_target, shape=(-1,)))

    i = 0
    err = tf.constant(1e10, dtype=tf.float64)
    while (i < n_iter_max) and (err > err_threshold):

        u_prev_ = u_
        v_prev_ = v_

        u = sinkhorn_slacks_update_u(K_sparse, a, v, Ka_slack, Kb_slack, i_target)
        v = sinkhorn_slacks_update_v(K_sparse, b, u, Ka_slack, Kb_slack, i_target)

        err, u_, v_ = sinkhorn_error(u, v, u_prev_, v_prev_, i_model, i_target)

        # print(f'iter {i:>6d}/{n_iter_max} err={np.array(err):8.4e}')
        i+=1

    Q_sparse = tf.expand_dims(u[:-1], axis=-1) * K_sparse * tf.gather(v[:-1], i_target)

    # Q_row = tf.reduce_sum(Q_sparse, axis=1)
    # print(f'finished slack sinkhorn min_row={Q_row.min()} n_rows_large={np.count_nonzero(Q_row>1e-3)}')

    return Q_sparse


#################################################################################################################
# Softslacks - TESTS
#################################################################################################################


@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def update_Ku_Kv_part1(K_sparse, v_sparse, u_sparse, Ka_slack, Kb_slack, v_slack, a, i_target):

    Kv_sparse = tf.reduce_sum(K_sparse * tf.gather(v_sparse, i_target), axis=1)
    Kv_sparse = Kv_sparse + v_slack*Ka_slack[:-1]

    Kv_slack = tf.reduce_sum(Kb_slack * v_slack, keepdims=True)
    Kv_sparse_slack = tf.concat([Kv_sparse, Kv_slack], axis=0)

    u_sparse = tf.math.divide_no_nan(a, Kv_sparse_slack)
    Ku_sparse = K_sparse * tf.expand_dims(u_sparse[:-1], axis=-1)

    return Ku_sparse, Kv_sparse_slack, u_sparse


@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def update_Ku_Kv_part2(Ku_sparse, u_slack, Kb_slack, b):

    Ku_sparse = Ku_sparse + u_slack*Kb_slack[:-1]
    Ku_slack = tf.reduce_sum(Kb_slack * u_slack, keepdims=True)
    Ku_sparse_slack = tf.concat([Ku_sparse, Ku_slack], axis=0)
    v_sparse = tf.math.divide_no_nan(b, Ku_sparse_slack)

    return Ku_sparse, v_sparse

@tf.function(jit_compile=False)
def sinkhorn_knopp_slacks_sparse_v2(a, b, slack_cost_a, slack_cost_b, K_nn, indices_sparse, reorder_sorting, inverse_sorting, n_iter_max=1000, err_threshold=1e-4):

    slack_cost_corner = 1.0

    K_slacks = tf.concat([tf.ones(len(a)-1, dtype=tf.float64)*tf.math.exp(-slack_cost_a), tf.ones(len(b)-1, dtype=tf.float64)*tf.math.exp(-slack_cost_b)], axis=0)
    K_nn_slacks_flat = tf.concat([tf.reshape(K_nn, (-1,)), K_slacks, tf.constant(slack_cost_corner, shape=(1,), dtype=tf.float64)], axis=0)
    Ks = tf.sparse.SparseTensor(values=tf.gather(K_nn_slacks_flat, reorder_sorting), indices=tf.gather(indices_sparse, reorder_sorting), dense_shape=(a.shape[0], b.shape[0]))
    
    u = tf.ones((a.shape[0], 1), tf.float64)
    v = tf.ones((b.shape[0], 1), tf.float64) 
    u_ = tf.gather(u, Ks.indices[:,0])[:,0]
    v_ = tf.gather(v, Ks.indices[:,1])[:,0]
    a_ = tf.expand_dims(a, axis=-1)
    b_ = tf.expand_dims(b, axis=-1)
    i = 0
    err = tf.constant(1e20, tf.float64)

    while i<n_iter_max and err>err_threshold :
    # for j in range(100):

        u_prev = u_
        v_prev = v_
        Kv = tf.sparse.sparse_dense_matmul(Ks, v)
        u = tf.math.divide_no_nan(a_, Kv)
        Ku = tf.sparse.sparse_dense_matmul(Ks, u, adjoint_a=True)
        v = tf.math.divide_no_nan(b_, Ku)
        u_ = tf.gather(u, Ks.indices[:,0])[:,0]
        v_ = tf.gather(v, Ks.indices[:,1])[:,0]
        err = tf.math.sqrt(tf.reduce_sum((v_*u_-v_prev*u_prev)**2)/tf.reduce_sum((v_prev*u_prev)**2))
        i = i + 1
    
    
    Q_nn_slacks_flat = Ks.values*v_*u_ # otherwise use Qs = tf.sparse.SparseTensor(values=Ks.values*v_*u_, indices=Ks.indices, dense_shape=Ks.shape)

    # cut out slacks
    Q_nn_slacks_flat = tf.gather(Q_nn_slacks_flat, inverse_sorting)
    Q_nn = tf.reshape(Q_nn_slacks_flat[:K_nn.shape[0]*K_nn.shape[1]], shape=K_nn.shape)

    # # test against full

    # K = tf.zeros((a.shape[0], b.shape[0]), dtype=tf.float64) #+ eps
    # K = tf.tensor_scatter_nd_add(K, indices_sparse, tf.reshape(K_nn_slacks_flat, (-1,)))
    # Q = K
    # for i in range(100):
    #     Q_prev = Q
    #     Kv = tf.matmul(K, v)
    #     u = tf.math.divide_no_nan(a_, Kv)
    #     Ktu = tf.matmul(tf.transpose(K, (1,0)), u)
    #     v = tf.math.divide_no_nan(b_, Ktu)
    # Q2 = u * K * tf.transpose(v, (1,0))
    # Q2_sparse_flat = tf.gather_nd(Q2, indices_sparse)
    # Q2_nn =  tf.reshape(Q2_sparse_flat[:K_nn.shape[0]*K_nn.shape[1]], shape=K_nn.shape)
    # print(np.max(np.abs(Q2_nn-Q_nn)))

    return Q_nn


def sinkhorn_knopp_slacks_full_vs_sparse(a, b, slack_cost_a, slack_cost_b, K_sparse, i_target, n_iter_max=1000, err_threshold=1e-6):


    i_tgt_flat = tf.reshape(i_target, (-1,))
    i_mod_flat = tf.repeat(np.arange(K_sparse.shape[0], dtype=np.int32), K_sparse.shape[1], axis=0)
    i_sparse = tf.cast(tf.concat([tf.expand_dims(i_mod_flat, axis=-1), tf.expand_dims(i_tgt_flat, axis=-1)], axis=-1), tf.int64)
    K_sparse_flat = tf.reshape(K_sparse, (-1,))

    K = tf.zeros((a.shape[0]-1, b.shape[0]-1), dtype=tf.float64) #+ eps
    K = tf.tensor_scatter_nd_add(K, i_sparse, tf.reshape(K_sparse, (-1,)))
    K = tf.concat([K, tf.ones((K.shape[0],1), tf.float64)*tf.math.exp(-slack_cost_a)], axis=1)
    K = tf.concat([K, tf.ones((1,K.shape[1]), tf.float64)*tf.math.exp(-slack_cost_b)], axis=0)
    K = np.array(K)
    K[-1,-1] = 1e9
    K = tf.constant(K)

    u = tf.ones((a.shape[0], 1), tf.float64)
    v = tf.ones((b.shape[0], 1), tf.float64) 
    aa = a[:,tf.newaxis]
    bb = b[:,tf.newaxis]
    import pudb; pudb.set_trace();
    pass


    Q = K
    for i in range(100):
        Q_prev = Q
        Kv = tf.matmul(K, v)
        u = tf.math.divide_no_nan(aa, Kv)
        Ktu = tf.matmul(tf.transpose(K, (1,0)), u)
        v = tf.math.divide_no_nan(bb, Ktu)
        Q = u * K * tf.transpose(v, (1,0))
        err = tf.math.reduce_euclidean_norm(Q-Q_prev)/tf.math.reduce_euclidean_norm(Q_prev)
        print(f'i={i:>4d} err={err:2.4e}')
        if err<err_threshold:
            break
    Q1 = Q

    Q = K
    for i in range(100):
        Q_prev = Q
        Q = tf.math.divide_no_nan(Q, tf.reduce_sum(Q, axis=1, keepdims=True)/aa)
        Q = tf.math.divide_no_nan(Q, tf.reduce_sum(Q, axis=0, keepdims=True)/tf.transpose(bb, (1,0)))
        err = tf.math.reduce_euclidean_norm(Q-Q_prev)/tf.math.reduce_euclidean_norm(Q_prev)  
        print(f'i={i:>4d} err={err:2.4e}')
        if err<err_threshold:
            break
    Q2 = Q

    aa = a[:-1,tf.newaxis]
    bb = b[:-1,tf.newaxis]
    u = tf.ones((aa.shape[0], 1), tf.float64)
    v = tf.ones((bb.shape[0], 1), tf.float64) 
    Ks = tf.sparse.SparseTensor(values=K_sparse_flat, indices=i_sparse, dense_shape=(aa.shape[0], bb.shape[0]))
    Ks = tf.sparse.reorder(Ks)
    nuv = min(len(u), len(v))
    Q = Ks
    for i in range(100):

        u_prev = u_
        v_prev = v_
        Kv = tf.sparse.sparse_dense_matmul(Ks, v)
        u = tf.math.divide_no_nan(aa, Kv)
        Ku = tf.sparse.sparse_dense_matmul(Ks, u, adjoint_a=True)
        v = tf.math.divide_no_nan(bb, Ku)
        u_ = tf.gather(u, Ks.indices[:,0])[:,0]
        v_ = tf.gather(v, Ks.indices[:,1])[:,0]
        err = tf.math.reduce_euclidean_norm(v_*u_-v_prev*u_prev)/tf.math.reduce_euclidean_norm(v_prev*u_prev)
        print(f'i={i:>4d} err={err:2.4e}')
        if err<err_threshold:
            break
    Q3 = tf.sparse.SparseTensor(values=Ks.values*v_*u_, indices=Ks.indices, dense_shape=(Ks.shape))
    Q3 = tf.sparse.to_dense(Q3)


    K_ = K[:-1,:-1]
    aa = a[:-1,tf.newaxis]
    bb = b[:-1,tf.newaxis]
    u = tf.ones((aa.shape[0], 1), tf.float64)
    v = tf.ones((bb.shape[0], 1), tf.float64) 
    Q = K_
    for i in range(100):
        Q_prev = Q
        Kv = tf.matmul(K_, v)
        u = tf.math.divide_no_nan(aa, Kv)
        Ktu = tf.matmul(tf.transpose(K_, (1,0)), u)
        v = tf.math.divide_no_nan(bb, Ktu)
    Q4 = u * K_ * tf.transpose(v, (1,0))


    slack_cost_corner = 1

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


    aa = a[:,tf.newaxis]
    bb = b[:,tf.newaxis]
    u = tf.ones((aa.shape[0], 1), tf.float64)
    v = tf.ones((bb.shape[0], 1), tf.float64) 
    for i in range(100):

        u_prev = u_
        v_prev = v_
        Kv = tf.sparse.sparse_dense_matmul(Ks, v)
        u = tf.math.divide_no_nan(aa, Kv)
        Ku = tf.sparse.sparse_dense_matmul(Ks, u, adjoint_a=True)
        v = tf.math.divide_no_nan(bb, Ku)
        u_ = tf.gather(u, Ks.indices[:,0])[:,0]
        v_ = tf.gather(v, Ks.indices[:,1])[:,0]
        # err = tf.math.reduce_euclidean_norm(v_*u_-v_prev*u_prev)/tf.math.reduce_euclidean_norm(v_prev*u_prev)
        # print(f'i={i:>4d} err={err:2.4e}')
        # if err<err_threshold:
        #     break
    Q5 = tf.sparse.SparseTensor(values=Ks.values*v_*u_, indices=Ks.indices, dense_shape=(Ks.shape))
    Q5 = tf.sparse.to_dense(Q5)

    K = np.array(K)
    K[-1,-1] = 1
    K = tf.constant(K)
    aa = a[:,tf.newaxis]
    bb = b[:,tf.newaxis]
    u = tf.ones((aa.shape[0], 1), tf.float64)
    v = tf.ones((bb.shape[0], 1), tf.float64) 
    Q = K
    for i in range(100):
        Q_prev = Q
        Kv = tf.matmul(K, v)
        u = tf.math.divide_no_nan(aa, Kv)
        Ktu = tf.matmul(tf.transpose(K, (1,0)), u)
        v = tf.math.divide_no_nan(bb, Ktu)
    Q6 = u * K * tf.transpose(v, (1,0))

    print(np.max(np.abs(Q5-Q6)))


    import pudb; pudb.set_trace();
    pass



#################################################################################################################
##
## Partial
##
#################################################################################################################



@tf.function()
def entropic_partial_wasserstein_sparse(a, b, K_sparse, i_target, i_model, i_grains, m=None, n_iter_max=200, err_threshold=1e-3):


    K_sparse = K_sparse + 1e-20
    dim_a, dim_b = len(a), len(b)
    dx_sparse = tf.ones(dim_a, dtype=tf.float64)
    dy_sparse = tf.ones(dim_b, dtype=tf.float64)
    K_sparse = K_sparse * m / tf.reduce_sum(K_sparse)
    q1_sparse = tf.ones(K_sparse.shape, dtype=tf.float64)
    q2_sparse = tf.ones(K_sparse.shape, dtype=tf.float64)
    q3_sparse = tf.ones(K_sparse.shape, dtype=tf.float64)

    i = 0
    err = tf.constant(1e20, dtype=tf.float64)
    while i < n_iter_max and err > err_threshold:
 
        Kprev_sparse = K_sparse
        scale1_sparse = tf.math.minimum( tf.math.divide_no_nan(a, tf.reduce_sum(tf.math.segment_sum(K_sparse, segment_ids=i_grains), axis=1)), dx_sparse)
        scale1_sparse = tf.gather(scale1_sparse, i_grains)
        K1_sparse = scale1_sparse[:,tf.newaxis] * K_sparse
        q1_sparse = q1_sparse * tf.math.divide_no_nan(Kprev_sparse, K1_sparse)
        K1prev_sparse = K1_sparse
        K1_sparse = K1_sparse * q2_sparse

        K1_sparse_flat = tf.reshape(K1_sparse, shape=(-1,))
        scale2_sparse = tf.math.minimum(tf.math.divide_no_nan(b, tf.math.unsorted_segment_sum(K1_sparse_flat, segment_ids=tf.reshape(i_target, shape=(-1,)), num_segments=dim_b)), dy_sparse)
        K2_sparse  = tf.reshape(tf.gather(scale2_sparse, indices=tf.reshape(i_target, shape=(-1,))) * K1_sparse_flat, shape=K_sparse.shape)
        K2prev_sparse = K2_sparse
        K2_sparse = K2_sparse * q3_sparse
        K_sparse = K2_sparse * tf.math.divide_no_nan(m, tf.reduce_sum(K2_sparse))
        q3_sparse = q3_sparse * tf.math.divide_no_nan(K2prev_sparse, K_sparse)
        
        err = tf.math.reduce_euclidean_norm(Kprev_sparse - K_sparse) / tf.math.reduce_euclidean_norm(Kprev_sparse)
        i = i + 1
        

    return K_sparse


#################################################################################################################
## Partial - TESTS
#################################################################################################################

def entropic_partial_wasserstein(a, b, K, m=None, numItermax=1000, stopThr=1e-100, verbose=False, log=False):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    K_input = K.copy()

    K = K + 1e-20

    dim_a, dim_b = K.shape
    dx = np.ones(dim_a, dtype=np.float64)
    dy = np.ones(dim_b, dtype=np.float64)

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    if m is None:
        m = np.min((np.sum(a), np.sum(b))) * 1.0
    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    if m > np.min((np.sum(a), np.sum(b))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    log_e = {'err': []}

    K = np.multiply(K, m / np.sum(K))

    err, cpt = 1, 0
    q1 = np.ones(K.shape)
    q2 = np.ones(K.shape)
    q3 = np.ones(K.shape)

    while (err > stopThr and cpt < numItermax):

        Kprev = K
        K = K * q1
        K1 = np.dot(np.diag(np.minimum(a / np.sum(K, axis=1), dx)), K)
        q1 = q1 * Kprev / K1
        K1prev = K1
        K1 = K1 * q2
        K2 = np.dot(K1, np.diag(np.minimum(b / np.sum(K1, axis=0), dy)))
        q2 = q2 * K1prev / K2
        K2prev = K2
        K2 = K2 * q3
        K = K2 * (m / np.sum(K2))
        q3 = q3 * K2prev / K

        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            print('Warning: numerical errors at iteration', cpt)
            break

        if cpt % 10 == 0:
            err = np.linalg.norm(Kprev - K)
            if log:
                log_e['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 11)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt = cpt + 1
    # log_e['partial_w_dist'] = np.sum(M * K)
    if log:
        return K, log_e
    else:
        return K


def entropic_partial_wasserstein_full_vs_sparse(a, b, K, K_sparse, i_target, reg, K_mask, segment_ids, m=None, numItermax=1000, stopThr=1e-100, verbose=False, log=False):

    i_target_flat = tf.reshape(i_target, shape=(-1,))

    K = K + 1e-20
    K_sparse = K_sparse + 1e-20
    K_init = K
    K_sparse_init = K_sparse

    dim_a, dim_b = len(a), len(b)
    dx = np.ones(dim_a, dtype=np.float64)
    dy = np.ones(dim_b, dtype=np.float64)

    dx_sparse = tf.ones(dim_a, dtype=tf.float64)
    dy_sparse = tf.ones(dim_b, dtype=tf.float64)

    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    if m > tf.reduce_min((tf.reduce_sum(a), tf.reduce_sum(b))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    K = np.multiply(K, m / np.sum(K))
    K_sparse = K_sparse * m / tf.reduce_sum(K_sparse)

    err, cpt = 1, 0
    q1 = np.ones(K.shape)
    q2 = np.ones(K.shape)
    q3 = np.ones(K.shape)

    q1_sparse = tf.ones(K_sparse.shape, dtype=tf.float64)
    q2_sparse = tf.ones(K_sparse.shape, dtype=tf.float64)
    q3_sparse = tf.ones(K_sparse.shape, dtype=tf.float64)


    while (err > stopThr and cpt < numItermax):
 
        Kprev = K
        K = K * q1
        scale1 = np.minimum(a / np.sum(K, axis=1), dx)
        K1 = np.dot(np.diag(scale1), K)
        q1 = q1 * Kprev / K1
        K1prev = K1
        K1 = K1 * q2
        scale2 = np.minimum(b / np.sum(K1, axis=0), dy)
        K2 = np.dot(K1, np.diag(scale2))
        q2 = q2 * K1prev / K2
        K2prev = K2
        K2 = K2 * q3
        K = K2 * (m / np.sum(K2))
        q3 = q3 * K2prev / K

        Kprev_sparse = K_sparse
        scale1_sparse = tf.math.minimum(a/tf.reduce_sum(tf.math.segment_sum(K_sparse, segment_ids=segment_ids), axis=1), dx)
        scale1_sparse = tf.gather(scale1_sparse, segment_ids)
        K1_sparse = scale1_sparse[:,tf.newaxis] * K_sparse
        q1_sparse = q1_sparse * Kprev_sparse / K1_sparse
        K1prev_sparse = K1_sparse
        K1_sparse = K1_sparse * q2_sparse

        K1_sparse_flat = tf.reshape(K1_sparse, shape=(-1,))
        scale2_sparse = tf.math.minimum(b / tf.math.unsorted_segment_sum(K1_sparse_flat, segment_ids=i_target_flat, num_segments=dim_b), dy)
        K2_sparse  = tf.reshape(tf.gather(scale2_sparse, i_target_flat) * K1_sparse_flat, shape=K_sparse.shape)
        K2prev_sparse = K2_sparse
        K2_sparse = K2_sparse * q3_sparse
        K_sparse = K2_sparse * (m / tf.reduce_sum(K2_sparse))
        q3_sparse = q3_sparse * K2prev_sparse / K_sparse


        if tf.reduce_any(~tf.math.is_finite(K_sparse)):

            break
    
        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            print('Warning: numerical errors at iteration', cpt)
            break

        err = tf.math.reduce_euclidean_norm(Kprev_sparse - K_sparse)

        if cpt % 10 == 0:
            err = np.linalg.norm(Kprev - K)
            if log:
                log_e['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 11)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt = cpt + 1

    return K, K_sparse



