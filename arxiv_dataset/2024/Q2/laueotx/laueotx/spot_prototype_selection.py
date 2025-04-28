import numpy as np
import scipy
from scipy import sparse
import time
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)

from laueotx.utils import logging as utils_logging
LOGGER = utils_logging.get_logger(__file__)


def test_sparse_vs_dense(C, n_candidates, n_max_grains_select, i_grn_in, i_target_nn, ot_b):

    def print_spot_results(select, loss, tag=''):

        print('results for {}'.format(tag))
        for i in range(len(select)):
            print('grain {:>5d} {:2.4e}'.format(select[i], loss[i]))

    # get the grain-to-spot cost matrix
    block_val = tf.reduce_max(C)*1e20
    Cgs = []
    for i in LOGGER.progressbar(range(n_candidates), at_level='info', desc='creating grain-to-spot cost matrix'):
        select = (i_grn_in[:,0] == i) 
        C_ = tf.math.unsorted_segment_sum(C[select], segment_ids=i_target_nn[select], num_segments=n_target_nn)
        C_ = tf.where(C_==0, block_val, C_)
        C_ = tf.cast(C_, tf.float64)
        Cgs.append(tf.expand_dims(C_, axis=0))
    Cgs = tf.concat(Cgs, axis=0)

    # original version 
    select_prototype_orig, spot_loss_orig = spot_prototype_selection.SPOT_GreedySubsetSelection(Cgs.numpy(), ot_b.numpy(), m=n_max_grains_select)
    print_spot_results(select_prototype_orig, spot_loss_orig, tag='DENSE (numpy)')

    # tensorflow version
    select_prototype_tf, spot_loss_tf = spot_prototype_selection.tf_SPOT_GreedySubsetSelection(Cgs, ot_b, m=n_max_grains_select)
    print_spot_results(select_prototype_tf, spot_loss_tf, tag='DENSE (tensorflow)')
    
    # sparse version
    i_gts_uv, segment_ids_gts = tf.unique(i_target_nn*100000 + i_grn_in[:,0]) # joint grain-to-spot hash
    segment_ids_tgt = tf.gather(i_target_nn, segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    segment_ids_grn = tf.gather(i_grn_in[:,0], segment_ids_gts) # there should not be multiple assignments of the same target to a single model grain, but just in case
    Cgs_seg = tf.math.unsorted_segment_sum(C, segment_ids=segment_ids_gts, num_segments=len(i_gts_uv))
    select_prototype_sparse, spot_loss_sparse = spot_prototype_selection.tf_sparse_vs_dense_SPOT_GreedySubsetSelection(C=Cgs_seg,
                                                                                                                       segs_s2t=segment_ids_gts,
                                                                                                                       segs_src=segment_ids_grn,
                                                                                                                       segs_tgt=segment_ids_tgt,
                                                                                                                       n_src=n_candidates, 
                                                                                                                       n_tgt=n_target_nn,
                                                                                                                       n_tgt_per_src=n_spots_per_grain,
                                                                                                                       q=ot_b[0], 
                                                                                                                       k=n_max_grains_select,
                                                                                                                       C_full=Cgs,
                                                                                                                       pdb=True)
    print_spot_results(select_prototype_sparse, spot_loss_sparse, tag='SPARSE (tensorflow, comparison)')


    select_prototype, spot_loss = spot_prototype_selection.tf_sparse_SPOT_GreedySubsetSelection(C=Cgs_seg,
                                                                                                segs_src=segment_ids_grn,
                                                                                                segs_tgt=segment_ids_tgt,
                                                                                                n_src=n_candidates, 
                                                                                                n_tgt=n_target_nn,
                                                                                                q=ot_b[0], 
                                                                                                k=n_max_grains_select)

    print_spot_results(select_prototype, spot_loss, tag='SPARSE (tensorflow, final)')

    

def tf_sparse_SPOT_GreedySubsetSelection(C, segs_src, segs_tgt, n_src, n_tgt, q, k):
    """Summary
    
    Parameters
    ----------
    C : TYPE
        Description
    segs_src : TYPE
        Description
    segs_tgt : TYPE
        Description
    n_src : TYPE
        Description
    n_tgt : TYPE
        Description
    q : float
        Weight of the target sampkes, assuming all target samples have the same weight
    k : TYPE
        Number of prototypes to choose
    
    Returns
    -------
    TYPE
        Description
    """

    a = tf.newaxis

    cost_min_vals = tf.ones(n_tgt, dtype=tf.float64) * 1e3
    cost_min_inds = tf.ones(n_tgt, dtype=tf.int32)*-1
    remaining_src = tf.range(n_src)
    max_increment_ind = tf.constant(n_src, dtype=tf.int32) # init to a large value
    prototype_set = []
    prototype_cost = []

    for i in range(k):
        
        remaining_src = tf.gather(remaining_src, tf.where(remaining_src!=max_increment_ind)[:,0])
        increment_vals = tf.maximum(tf.gather(cost_min_vals, segs_tgt) - C, 0.)
        increment_vals = tf.math.segment_sum(increment_vals, segs_src)*q
        increment_vals = tf.gather(increment_vals, remaining_src)
        max_increment_ind = tf.gather(remaining_src, tf.math.argmax(increment_vals))
        max_increment_ind = tf.cast(max_increment_ind, tf.int32)
        inds_chosen   = tf.where(segs_src==max_increment_ind)[:,0]
        C_chosen   = tf.gather(C, indices=inds_chosen)
        indices_tgt   = tf.gather(segs_tgt, indices=inds_chosen)
        cost_min_vals_chosen = tf.gather(cost_min_vals, indices=indices_tgt)
        indices_chosen = tf.where((cost_min_vals_chosen - C_chosen) > 0)
        update_vals = tf.gather(C_chosen, indices=indices_chosen)[:,0]
        indices = tf.gather(indices_tgt, indices=indices_chosen)
        cost_min_vals = tf.tensor_scatter_nd_update(cost_min_vals, indices=indices, updates=update_vals)
        update_inds   = tf.ones(update_vals.shape[0], dtype=np.int32)*i
        cost_min_inds = tf.tensor_scatter_nd_update(cost_min_inds, indices=indices, updates=update_inds)
        cost = tf.reduce_sum(cost_min_vals*q)
        prototype_set.append(max_increment_ind)
        prototype_cost.append(cost)

    prototype_set = tf.stack(prototype_set)
    prototype_cost = tf.stack(prototype_cost)
    return prototype_set, prototype_cost


def tf_sparse_vs_dense_SPOT_GreedySubsetSelection(C, segs_s2t, segs_src, segs_tgt, n_src, n_tgt, n_tgt_per_src, q, k, C_full=None, pdb=False):
    """Summary
    
    Parameters
    ----------
    C : TYPE
        Description
    segs_s2t : TYPE
        Description
    segs_src : TYPE
        Description
    segs_tgt : TYPE
        Description
    n_src : TYPE
        Description
    n_tgt : TYPE
        Description
    n_tgt_per_src : TYPE
        Description
    q : float
        Weight of the target sampkes, assuming all target samples have the same weight
    k : TYPE
        Number of prototypes to choose
    """

    def summary(x): print(x.shape, np.min(x), np.max(x), np.sum(x), np.mean(x), np.median(x), np.unique(x).shape, np.count_nonzero(x))

    a = tf.newaxis

    cost_min_vals = tf.ones(n_tgt, dtype=tf.float64) * 1e3
    cost_min_inds = tf.ones(n_tgt, dtype=tf.int32)*-1
    remaining_src = tf.range(n_src)
    prototype_set = []
    prototype_cost = []

    # comparison
    tf_C = tf.constant(C_full)
    tf_targetMarginal = tf.ones(C_full.shape[1], dtype=C_full.dtype)*q
    tf_numY = tf_C.shape[0]
    tf_numX = tf_C.shape[1]
    tf_allY = tf.range(tf_numY)
    tf_targetMarginal = tf.reshape(tf_targetMarginal, (1, tf_numX))
    tf_currMinCostValues = tf.ones((1, tf_numX), dtype=tf_C.dtype) * 1e3
    tf_currMinSourceIndex = tf.zeros((1, tf_numX), dtype=tf.int32)
    tf_remainingElements = tf_allY
    tf_chosenElements = []
    tf_S = []
    tf_setValues = []


    max_increment_ind = tf.constant(n_src, dtype=tf.int32) # init to a large value
    for i in range(k):

        ############ ------------------ SPARSE

        remaining_src = tf.gather(remaining_src, tf.where(remaining_src!=max_increment_ind)[:,0])
        increment_vals = tf.maximum(tf.gather(cost_min_vals, segs_tgt) - C, 0.)
        increment_vals = tf.math.segment_sum(increment_vals, segs_src)*q
        increment_vals = tf.gather(increment_vals, remaining_src)
        # max_increment_ind = tf.cast(tf.math.argmax(increment_vals), tf.int32)
        max_increment_ind = tf.gather(remaining_src, tf.math.argmax(increment_vals))
        max_increment_ind = tf.cast(max_increment_ind, tf.int32)

        ############ ------------------ DENSE

        # comparison
        sizeS = i
        tf_remainingElements = tf_remainingElements[~np.in1d(np.array(tf_remainingElements), np.array(tf_chosenElements))]
        tf_temp1 = tf.math.maximum(tf_currMinCostValues-tf_C, 0.)
        tf_temp1 = tf.matmul(tf_temp1, tf.transpose(tf_targetMarginal, (1,0)))
        tf_incrementValues = tf.gather(tf_temp1, tf_remainingElements)
        tf_maxIncrementIndex = tf.math.argmax(tf_incrementValues)

        ############ ------------------ SPARSE

        inds_chosen   = tf.where(segs_src==max_increment_ind)[:,0]
        C_chosen   = tf.gather(C, indices=inds_chosen)
        indices_tgt   = tf.gather(segs_tgt, indices=inds_chosen)
        cost_min_vals_chosen = tf.gather(cost_min_vals, indices=indices_tgt)
        indices_chosen = tf.where((cost_min_vals_chosen - C_chosen) > 0)
        update_vals = tf.gather(C_chosen, indices=indices_chosen)[:,0]
        indices = tf.gather(indices_tgt, indices=indices_chosen)
        cost_min_vals = tf.tensor_scatter_nd_update(cost_min_vals, indices=indices, updates=update_vals)
        update_inds   = tf.ones(update_vals.shape[0], dtype=np.int32)*i
        cost_min_inds = tf.tensor_scatter_nd_update(cost_min_inds, indices=indices, updates=update_inds)

        cost = tf.reduce_sum(cost_min_vals*q)

        prototype_set.append(max_increment_ind)
        prototype_cost.append(cost)

        print('SPOT sparse: {:>4d}/{:d} max_increment_ind={:>3d} cost={:4.2e}'.format(i+1, k, max_increment_ind, cost))

        ############ ------------------ DENSE

        tf_chosenElements = tf.gather(tf_remainingElements, tf_maxIncrementIndex)
        tf_tempIndex = (tf_currMinCostValues - tf.gather(tf_C, tf_chosenElements)) > 0
        tf_D = tf.gather(tf_C, tf_chosenElements)
        indices_dense = tf.where(tf_tempIndex[0])
        updates_vals_dense = tf.gather(tf_D[0], indices_dense[:,0])
        tf_currMinCostValues = tf.tensor_scatter_nd_update(tf_currMinCostValues[0], indices_dense, updates_vals_dense)[a, :]
        updates_inds_dense = tf.ones(indices_dense.shape[0], dtype=np.int32)*sizeS
        tf_currMinSourceIndex = tf.tensor_scatter_nd_update(tf_currMinSourceIndex[0], indices_dense, updates_inds_dense)[a, :]
        tf_currObjectiveValue =tf.matmul(tf_currMinCostValues, tf.transpose(tf_targetMarginal, (1,0)))

        tf_S.append(tf_chosenElements)
        tf_setValues.append(tf_currObjectiveValue)

        print('SPOT tf: i={:>4d}/{} chosen={:>4d} obj={:4.2e} len(tf_remainingElements)={}'.format(sizeS+1, k, np.array(tf_chosenElements)[0], np.array(tf_currObjectiveValue)[0,0], len(tf_remainingElements)))

        if pdb:
            import pudb; pudb.set_trace();
            pass

    prototype_set = tf.concat(prototype_set, axis=0)
    prototype_cost = tf.concat(prototype_cost, axis=0)
    return prototype_set, prototype_cost

def tf_SPOT_GreedySubsetSelection(C, targetMarginal, m):
    # https://github.com/royparijat/SPOT/blob/main/python/SPOTgreedy.py
    # Assumes one source point selected at a time, which simplifies the code.
    # C: Cost matrix of OT: number of source x number of target points {[numY * numX]}
    # targetMarginal: 1 x number of target (row-vector) size histogram of target distribution. Non negative entries summing to 1 {[1*numX]}
    # m: number of prototypes to be selected.

    a = tf.newaxis
    tf_C = tf.constant(C)
    # C = np.array(C)
    tf_targetMarginal = tf.constant(targetMarginal)
    # targetMarginal = np.array(targetMarginal)

    targetMarginal = targetMarginal / np.sum(targetMarginal)
    tf_targetMarginal = tf_targetMarginal / tf.reduce_sum(tf_targetMarginal)
    
    # numY = C.shape[0]
    # numX = C.shape[1]
    tf_numY = tf_C.shape[0]
    tf_numX = tf_C.shape[1]

    # allY = np.arange(numY)
    tf_allY = tf.range(tf_numY)
    
    # just to make sure we have a row vector.
    # targetMarginal = targetMarginal.reshape(1, numX)
    tf_targetMarginal = tf.reshape(tf_targetMarginal, (1, tf_numX))

    # Intialization
    # S = np.zeros((1, m), dtype=int)
    tf_S = []
    # timeTaken = np.zeros((1, m), dtype=int)
    # setValues = np.zeros((1, m), dtype=float)
    tf_setValues = []
    sizeS = 0
    # currOptw = []
    # currMinCostValues = np.ones((1, numX), dtype=float) * 1000000
    # currMinSourceIndex = np.zeros((1, numX), dtype=int)
    tf_currMinCostValues = tf.ones((1, tf_numX), dtype=tf_C.dtype) * 1e3
    tf_currMinSourceIndex = tf.zeros((1, tf_numX), dtype=tf.int32)

    # remainingElements = allY
    tf_remainingElements = tf_allY
    # chosenElements = []
    tf_chosenElements = []

    iterNum = 0
    start = time.time()

    while sizeS < m:
        
        iterNum = iterNum + 1

        # remainingElements = remainingElements[~np.in1d(np.array(remainingElements), np.array(chosenElements))]
        tf_remainingElements = tf_remainingElements[~np.in1d(np.array(tf_remainingElements), np.array(tf_chosenElements))]
        
        # temp1 = np.maximum(currMinCostValues - C, 0)
        # temp1 = np.matmul(temp1, targetMarginal.T)

        tf_temp1 = tf.math.maximum(tf_currMinCostValues-tf_C, 0)
        tf_temp1 = tf.matmul(tf_temp1, tf.transpose(tf_targetMarginal, (1,0)))

        # incrementValues = temp1[remainingElements]
        # maxIncrementIndex = np.argmax(np.array(incrementValues))

        tf_incrementValues = tf.gather(tf_temp1, tf_remainingElements)
        tf_maxIncrementIndex = tf.math.argmax(tf_incrementValues)

        # Chosing the best element
        # chosenElements = remainingElements[maxIncrementIndex]
        # S[0][sizeS] = chosenElements;

        tf_chosenElements = tf.gather(tf_remainingElements, tf_maxIncrementIndex)
        tf_S.append(tf_chosenElements)

        # Updating currMinCostValues and currMinSourceIndex vectors
        # tempIndex = (currMinCostValues - C[chosenElements, :]) > 0
        # D = C[chosenElements]
        # currMinCostValues[tempIndex] = D[tempIndex[0]]
        # currMinSourceIndex[tempIndex] = sizeS

        tf_tempIndex = (tf_currMinCostValues - tf.gather(tf_C, tf_chosenElements)) > 0
        tf_D = tf.gather(tf_C, tf_chosenElements)
        indices = tf.where(tf_tempIndex[0])
        updates = tf.gather(tf_D[0], indices[:,0])

        tf_currMinCostValues = tf.tensor_scatter_nd_update(tf_currMinCostValues[0], indices, updates)[a, :]
        updates = tf.ones(indices.shape[0], dtype=np.int32)*sizeS
        tf_currMinSourceIndex = tf.tensor_scatter_nd_update(tf_currMinSourceIndex[0], indices, updates)[a, :]

        # Current objective and other booking
        # currObjectiveValue = np.sum(np.dot(currMinCostValues, targetMarginal.T))
        # setValues[0][sizeS] = currObjectiveValue

        tf_currObjectiveValue =tf.matmul(tf_currMinCostValues, tf.transpose(tf_targetMarginal, (1,0)))
        tf_setValues.append(tf_currObjectiveValue)

        # print('SPOT np: i={:>4d}/{} chosen={:>4d} obj={:4.2e}'.format(sizeS+1, m, chosenElements, currObjectiveValue))
        print('SPOT tf: i={:>4d}/{} chosen={:>4d} obj={:4.2e} len(tf_remainingElements)={}'.format(sizeS+1, m, np.array(tf_chosenElements)[0], np.array(tf_currObjectiveValue)[0,0], len(tf_remainingElements)))

        sizeS = sizeS + 1

    tf_setValues = tf.concat(tf_setValues, axis=0)[:,0]
    tf_S = tf.concat(tf_S, axis=0)

    return tf_S, tf_setValues

def SPOT_GreedySubsetSelection(C, targetMarginal, m):
    # https://github.com/royparijat/SPOT/blob/main/python/SPOTgreedy.py
    # Assumes one source point selected at a time, which simplifies the code.
    # C: Cost matrix of OT: number of source x number of target points {[numY * numX]}
    # targetMarginal: 1 x number of target (row-vector) size histogram of target distribution. Non negative entries summing to 1 {[1*numX]}
    # m: number of prototypes to be selected.

    from scipy.sparse import csr_matrix

    targetMarginal = targetMarginal / np.sum(targetMarginal)
    numY = C.shape[0]
    numX = C.shape[1]
    allY = np.arange(numY)
    # just to make sure we have a row vector.
    targetMarginal = targetMarginal.reshape(1, numX)

    # Intialization
    S = np.zeros((1, m), dtype=int)
    timeTaken = np.zeros((1, m), dtype=int)
    setValues = np.zeros((1, m), dtype=int)
    sizeS = 0
    currOptw = []
    currMinCostValues = np.ones((1, numX), dtype=int) * 1000000
    currMinSourceIndex = np.zeros((1, numX), dtype=int)
    remainingElements = allY
    chosenElements = []
    iterNum = 0
    start = time.time()
    while sizeS < m:
        iterNum = iterNum + 1
        remainingElements = remainingElements[~np.in1d(np.array(remainingElements), np.array(chosenElements))]
        temp1 = np.maximum(currMinCostValues - C, 0)
        temp1 = np.matmul(temp1, targetMarginal.T)
        incrementValues = temp1[remainingElements]
        maxIncrementIndex = np.argmax(np.array(incrementValues))
        # Chosing the best element
        chosenElements = remainingElements[maxIncrementIndex]
        S[0][sizeS] = chosenElements;
        # Updating currMinCostValues and currMinSourceIndex vectors
        tempIndex = (currMinCostValues - C[chosenElements, :]) > 0
        D = C[chosenElements]
        currMinCostValues[tempIndex] = D[tempIndex[0]]
        # currMinSourceIndex reflects index in set S
        currMinSourceIndex[tempIndex] = sizeS
        # Current objective and other booking
        currObjectiveValue = np.sum(np.dot(currMinCostValues, targetMarginal.T))
        setValues[0][sizeS] = currObjectiveValue
        if sizeS == m-1 :
            print("targetMarginal", targetMarginal);
            gammaOpt = csr_matrix((targetMarginal[0], (currMinSourceIndex[0], range(0, numX))), shape=(m, numX));
            print("gammaOpt \n", gammaOpt);
            currOptw = np.sum(gammaOpt, axis=1).flatten();
            print("currOptw \n", currOptw);
        sizeS = sizeS + 1
    end = time.time()
    print("S : ", S)
    print("Time : ", end - start)
    return S[0], setValues[0]



def remove_candidates_with_bad_fits(chi2_red, frac_inliers, apply_to, max_chi2_red=1.5, min_frac_inliers=0.8):

    # remove bad fits
    n_trials = len(chi2_red)
    select_fracin = frac_inliers>min_frac_inliers 
    select_chi2 = chi2_red<max_chi2_red
    select_candidate = select_fracin & select_chi2
    n_candidates = np.count_nonzero(select_candidate)
    LOGGER.info(f'selected {n_candidates}/{n_trials}, by reduced chi2 < {max_chi2_red:2.2f}: {np.count_nonzero(select_chi2):4d}/{n_trials}, by fraction of outliers < {1.-min_frac_inliers:4.2f}: {np.count_nonzero(select_fracin)}/{n_trials}')

    if n_candidates<1:
        raise Exception('did not find any prototypes that meet the criteria')

    output = [x[select_candidate] for x in apply_to]

    return output

def prune_similar_solutions(a, x, apply_to, precision_rot=0.0001, precision_pos=0.001):

    quantize =  lambda x, q: q*np.round(x/q)

    from sklearn.preprocessing import MinMaxScaler
    x_round = quantize(x, precision_pos)
    a_round = quantize(a, precision_pos)
    theta = np.concatenate([a_round, x_round], axis=1)
    uval, uind =  np.unique(theta, axis=0, return_index=True)
    LOGGER.info(f'removed duplicate solution in grain parameter space, rounded to with precision: orientation {precision_rot:2.2e}, position {precision_pos:2.2e} , selected {len(uind)}/{len(theta)}')
    output = [x[uind] for x in apply_to]

    return output
