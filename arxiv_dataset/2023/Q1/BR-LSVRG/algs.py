import numpy as np
import random
import time
import pickle
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.stats import norm as norm_d
from scipy.stats import randint
from scipy.stats import bernoulli
import scipy
from functions import *
from utils import *
from copy import deepcopy
import math
from itertools import permutations
from scipy.spatial.distance import cdist, euclidean

def br_l_svrg(filename, x_init, A, y, gamma, num_of_byz, p, num_of_workers, attack, agg,
              l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
              batch_size=1, save_info_period=100, x_star=None, f_star=None):
    # m -- total number of datasamples
    # n -- dimension of the problem
    m, n = A.shape
    assert (len(x_init) == n)
    assert (len(y) == m)
    
    #batch_size = 8*L/l2

    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)

    # this array below is needed to reduce the time of sampling stochastic gradients
    indices_arr = randint.rvs(low=0, high=m, size=1000)
    num_of_indices = len(indices_arr)
    for i in range(num_of_workers - 1):
        indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=m, size=num_of_indices)))
    indices_counter = 0

    # it is needed for l-svrg updates
    bernoulli_arr = bernoulli.rvs(p, size=num_of_workers * 1000)
    bernoulli_size = len(bernoulli_arr)
    w_vectors = np.tile(deepcopy(x), [num_of_workers, 1])
    grads_w = logreg_grad(x, [A, y, l2, sparse_full])
    shape = (num_of_workers, len(grads_w))
    G_w = np.zeros(shape)
    for i in range(num_of_workers - 1):
        grads_w = np.vstack((grads_w, logreg_grad(x, [A, y, l2, sparse_full])))

    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0]) - f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])

    t_start = time.time()
    num_of_data_passes = 0.0

    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()

    indices_counter = 0
    bernoulli_counter = 0

    #method
    for it in range(int(S * m / batch_size)):
        if indices_counter == num_of_indices:
            indices_arr = randint.rvs(low=0, high=m, size=num_of_indices)
            num_of_indices = len(indices_arr)
            for i in range(num_of_workers - 1):
                indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=m, size=num_of_indices)))
            indices_counter = 0

        if bernoulli_counter >= bernoulli_size - num_of_workers + num_of_byz:
            bernoulli_arr = bernoulli.rvs(p, size=bernoulli_size)
            bernoulli_counter = 0

        # below we emulate the workers behavior and aggregate their updates on-the-fly
        bits_sum_temp = 0
        
        for i in range(num_of_workers - num_of_byz):
            A_i = A_for_batch
            y_i = y
            g_i = logreg_grad(x, [A_i[indices_arr[i][indices_counter:indices_counter + batch_size]],
                                  y_i[indices_arr[i][indices_counter:indices_counter + batch_size]], l2,
                                 sparse_stoch]) - logreg_grad(w_vectors[i], [
            A_i[indices_arr[i][indices_counter:indices_counter + batch_size]],
         y_i[indices_arr[i][indices_counter:indices_counter + batch_size]], l2, sparse_stoch]) + grads_w[i]
            #g_i = logreg_grad(x, [A_i, y_i, l2, sparse_stoch])
            G_w[i] = g_i
            if (bernoulli_arr[bernoulli_counter] == 1):
                w_vectors[i] = deepcopy(x)
                grads_w[i] = logreg_grad(w_vectors[i], [A_i, y_i, l2, sparse_stoch])
                num_of_data_passes += 1.0 / num_of_workers
            bernoulli_counter += 1
            
        #attack
        if attack == "BF":
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                A_i = A_for_batch
                y_i = y
                g_i = -1 * logreg_grad(x, [A_i[indices_arr[i][indices_counter:indices_counter+batch_size]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+batch_size]], l2, sparse_stoch]) - logreg_grad(w_vectors[i], [A_i[indices_arr[i][indices_counter:indices_counter+batch_size]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+batch_size]], l2, sparse_stoch]) + grads_w[i]
                G_w[i] = g_i
                if (bernoulli_arr[bernoulli_counter] == 1):
                    w_vectors[i] = deepcopy(x)
                    grads_w[i] = logreg_grad(w_vectors[i], [A_i, y_i, l2, sparse_stoch])
                    num_of_data_passes += 1.0 / num_of_workers
                bernoulli_counter += 1
            
        if attack == "LF":
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                A_i = A_for_batch
                y_i = -1 * y
                g_i = logreg_grad(x, [A_i[indices_arr[i][indices_counter:indices_counter+batch_size]], 
                                 y_i[indices_arr[i][indices_counter:indices_counter+batch_size]], l2, sparse_stoch]) - logreg_grad(w_vectors[i], [A_i[indices_arr[i][indices_counter:indices_counter+batch_size]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+batch_size]], l2, sparse_stoch]) + grads_w[i]
                #g_i = logreg_grad(x, [A_i, y_i, l2, sparse_stoch])
                G_w[i] = g_i
                if (bernoulli_arr[bernoulli_counter] == 1):
                    w_vectors[i] = deepcopy(x)
                    grads_w[i] = logreg_grad(w_vectors[i], [A_i, y_i, l2, sparse_stoch])
                    num_of_data_passes += 1.0/num_of_workers
                bernoulli_counter += 1
            
        if attack == "IPM":
            sum_of_good = sum(G_w)
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                G_w[i] = -0.1 * sum_of_good / (num_of_workers - num_of_byz)
                
        if attack == "ALIE":
            exp_of_good = sum(G_w) / (num_of_workers - num_of_byz)
            var_of_good = (sum(G_w * G_w)) / (num_of_workers - num_of_byz) - exp_of_good * exp_of_good
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                G_w[i] = exp_of_good - 1.06 * var_of_good
        
                
        if agg == "GM":
            perm = np.random.permutation(num_of_workers)
            x = x - gamma * GM(perm, 2, G_w)
            
        if agg == "M":
            perm = np.random.permutation(num_of_workers)
            x = x - gamma * np.mean(G_w, axis=0)
            
        if agg == "CM":
            perm = np.random.permutation(num_of_workers)
            x = x - gamma * CM(perm, 2, G_w)
            
            
        indices_counter += 1
        num_of_data_passes += 2.0 * batch_size / m
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0]) - f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break

    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0]) - f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)

    res = {'last_iter': x, 'func_vals': func_val, 'iters': its, 'time': tim, 'data_passes': data_passes,
           'squared_distances': sq_distances, 'num_of_workers': num_of_workers, 'num_of_byz': num_of_byz}

    with open("dump/" + filename + "_BR_L_SVRG_" + attack + "_" + agg + "_gamma_" + str(gamma) + "_l2_" + str(l2) + "_p_" + str(p) + "_epochs_" + str(S) + "_workers_" + str(num_of_workers) + "_batch_" + str(batch_size) + "_byz_" + str(num_of_byz) + ".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def byrd_saga(filename, x_init, A, y, gamma, num_of_byz, num_of_workers, attack, agg,
         l2=0, sparse_full=True, sparse_stoch=False, l1=0, S=50, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=m * S)
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    ref_point = np.array(x_star) 
    x = np.array(x_init)
    
    points_table = np.tile(copy.deepcopy(x), ((m * num_of_workers), 1))
    
    # this array below is needed to reduce the time of sampling stochastic gradients
    indices_arr = randint.rvs(low=0, high=m, size=1000)
    num_of_indices = len(indices_arr)
    for i in range(num_of_workers - 1):
        indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=m, size=num_of_indices)))
    indices_counter = 0
    
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, l1]) - f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    grad_sum = logreg_grad(x, [A, y, l2, sparse_full])
    num_of_data_passes = 1.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    
   
    shape1 = (num_of_workers, len(grad_sum))
    G_w = np.zeros(shape1)
    shape2 = (m, len(x))
    G = np.zeros(shape2)
    grad = np.zeros(shape1)
    g = np.zeros(shape1)
    old_batch_grad_sum = np.zeros(shape1)
    for w in range(num_of_workers):
        G_w[w] = logreg_grad(x, [A, y, l2, sparse_full])
            
    for it in range(int(m * S / batch_size)):
        for w in range(num_of_workers - num_of_byz):
            if indices_counter >= indices_size - batch_size:
                indices_counter = 0
                indices = randint.rvs(low=0, high=m, size=indices_size)
            
            batch_ind = indices[indices_counter:(indices_counter+batch_size)]
            indices_counter += batch_size
        
            A_w = A
            y_w = y
            grad[w] = logreg_grad(x, [A_w[batch_ind], y_w[batch_ind], l2, sparse_stoch])
            old_batch_grad_sum[w] = 0.0
        
            for idx in batch_ind:
                old_batch_grad_sum[w] += logreg_grad(points_table[w * m + idx], [A_w[idx:idx+1], y_w[idx:idx+1], l2, sparse_stoch]) * 1.0 / batch_size
            
            g[w] = grad[w] - old_batch_grad_sum[w] + G_w[w]
            G_w[w] += (grad[w] - old_batch_grad_sum[w]) * 1.0 * batch_size / m
            
            points_table[w * m + batch_ind] = x
        
        #attack    
        if attack == "BF":
            for w in range(num_of_workers - num_of_byz, num_of_workers):
                if indices_counter >= indices_size - batch_size:
                    indices_counter = 0
                    indices = randint.rvs(low=0, high=m, size=indices_size)
            
                batch_ind = indices[indices_counter:(indices_counter+batch_size)]
                indices_counter += batch_size
        
                grad[w] = logreg_grad(x, [A[batch_ind], y[batch_ind], l2, sparse_stoch])
                old_batch_grad_sum[w] = 0.0
        
                for idx in batch_ind:
                    old_batch_grad_sum[w] += logreg_grad(points_table[w*m+idx], [A[idx:idx+1], y[idx:idx+1], l2, sparse_stoch]) * 1.0 / batch_size
            
                g[w] = (-1) * (grad[w] - old_batch_grad_sum[w] + G_w[w])
                G_w[w] += (grad[w] - old_batch_grad_sum[w]) * 1.0 * batch_size / m
            
                points_table[w*m + batch_ind] = x
            
        if attack == "LF":
            for w in range(num_of_workers - num_of_byz, num_of_workers):
                if indices_counter >= indices_size - batch_size:
                    indices_counter = 0
                    indices = randint.rvs(low=0, high=m, size=indices_size)
            
                batch_ind = indices[indices_counter:(indices_counter+batch_size)]
                indices_counter += batch_size
        
                grad[w] = logreg_grad(x, [A[batch_ind], -1 * y[batch_ind], l2, sparse_stoch])
                old_batch_grad_sum[w] = 0.0
        
                for idx in batch_ind:
                    old_batch_grad_sum[w] += logreg_grad(points_table[w*m+idx], [A[idx:idx+1], (-1) * y[idx:idx+1], l2, sparse_stoch]) * 1.0 / batch_size
            
                g[w] = (grad[w] - old_batch_grad_sum[w] + G_w[w])
                G_w[w] += (grad[w] - old_batch_grad_sum[w])*1.0*batch_size / m
            
                points_table[w*m+batch_ind] = x
                
        if attack == "IPM":
            sum_of_good = sum(G_w)
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                G_w[i] = -0.1 * sum_of_good / (num_of_workers - num_of_byz)
        
        if attack == "ALIE":
            exp_of_good = sum(G_w) / (num_of_workers - num_of_byz)
            var_of_good = (sum(G_w * G_w)) / (num_of_workers - num_of_byz) - exp_of_good * exp_of_good
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                G_w[i] = exp_of_good - 1.06 * var_of_good
            
        if agg == "GM":
            x = x - gamma * geometric_median(g)
            
        if agg == "CM":
            perm = np.random.permutation(num_of_workers)
            x = x - gamma * CM(perm, 2, g)
        
        num_of_data_passes += 2.0 * batch_size / m
                
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1]) - f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    
    
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+ filename + "_Byrd_SAGA_" + attack + "_" + agg + "_gamma_" + str(gamma) + "_l2_" + str(l2) + "_epochs_" + str(S)
              + "_workers_" + str(num_of_workers) + "_batch_" + str(batch_size) + "_byz_" + str(num_of_byz) + ".txt", 'wb') as file:
        pickle.dump(res, file)
    return res



def byz_vr_marina(filename, x_init, A, y, gamma, num_of_byz, p, num_of_workers, attack, agg,
              l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
              batch_size=1, save_info_period=100, x_star=None, f_star=None):
    # m -- total number of datasamples
    # n -- dimension of the problem
    m, n = A.shape
    assert (len(x_init) == n)
    assert (len(y) == m)
    
    #batch_size = 8*L/l2

    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)

    # this array below is needed to reduce the time of sampling stochastic gradients
    indices_arr = randint.rvs(low=0, high=m, size=1000)
    num_of_indices = len(indices_arr)
    for i in range(num_of_workers - 1):
        indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=m, size=num_of_indices)))
    indices_counter = 0

    # it is needed for l-svrg updates
    bernoulli_arr = bernoulli.rvs(p, size=num_of_workers * 1000)
    bernoulli_size = len(bernoulli_arr)
    w_vectors = np.tile(deepcopy(x), [num_of_workers, 1])
    grads_w = logreg_grad(x, [A, y, l2, sparse_full])
    shape = (num_of_workers, len(grads_w))
    G_w = np.zeros(shape)
    G_w1 = np.zeros(len(grads_w))
    x1 = np.zeros(n)
    for i in range(num_of_workers - 1):
        grads_w = np.vstack((grads_w, logreg_grad(x, [A, y, l2, sparse_full])))

    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0]) - f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])

    t_start = time.time()
    num_of_data_passes = 0.0

    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()

    indices_counter = 0
    bernoulli_counter = 0

    #method
    for it in range(int(S * m / batch_size)):
        if indices_counter == num_of_indices:
            indices_arr = randint.rvs(low=0, high=m, size=num_of_indices)
            num_of_indices = len(indices_arr)
            for i in range(num_of_workers - 1):
                indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=m, size=num_of_indices)))
            indices_counter = 0

        if bernoulli_counter >= bernoulli_size - num_of_workers + num_of_byz:
            bernoulli_arr = bernoulli.rvs(p, size=bernoulli_size)
            bernoulli_counter = 0

        # below we emulate the workers behavior and aggregate their updates on-the-fly
        bits_sum_temp = 0
        
        for i in range(num_of_workers - num_of_byz):
            A_i = A_for_batch
            y_i = y
            if (bernoulli_arr[bernoulli_counter] == 1):
                g_i = logreg_grad(deepcopy(x), [A_i, y_i, l2, sparse_stoch])
                num_of_data_passes += 1.0 / num_of_workers
            else: 
                g_i = G_w1 + logreg_grad(deepcopy(x), [A_i[indices_arr[i][indices_counter:indices_counter + batch_size]],
                                  y_i[indices_arr[i][indices_counter:indices_counter + batch_size]], l2,
                                 sparse_stoch]) - logreg_grad(x1, [A_i[indices_arr[i][indices_counter:indices_counter + batch_size]],
                                  y_i[indices_arr[i][indices_counter:indices_counter + batch_size]], l2,
                                 sparse_stoch]) 
            #g_i = logreg_grad(x, [A_i, y_i, l2, sparse_stoch])
            G_w[i] = g_i
            
        #attack
        if attack == "BF":
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                A_i = A_for_batch
                y_i = y
                g_i = -1 * logreg_grad(deepcopy(x), [A_i[indices_arr[i][indices_counter:indices_counter+batch_size]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+batch_size]], l2, sparse_stoch]) + logreg_grad(x1, [A_i[indices_arr[i][indices_counter:indices_counter+batch_size]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+batch_size]], l2, sparse_stoch]) + G_w1
                G_w[i] = g_i
                if (bernoulli_arr[bernoulli_counter] == 1):
                    G_w1 = logreg_grad(deepcopy(x), [A_i, y_i, l2, sparse_stoch])
            
        if attack == "LF":
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                A_i = A_for_batch
                y_i = -1 * y
                g_i = logreg_grad(deepcopy(x), [A_i[indices_arr[i][indices_counter:indices_counter+batch_size]], 
                                 y_i[indices_arr[i][indices_counter:indices_counter+batch_size]], l2, sparse_stoch]) - logreg_grad(x1, [A_i[indices_arr[i][indices_counter:indices_counter+batch_size]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+batch_size]], l2, sparse_stoch]) + G_w1
                #g_i = logreg_grad(x, [A_i, y_i, l2, sparse_stoch])
                G_w[i] = g_i
                if (bernoulli_arr[bernoulli_counter] == 1):
                    G_w1 = logreg_grad(deepcopy(x), [A_i, -1 * y_i, l2, sparse_stoch])
            
        if attack == "IPM":
            if (bernoulli_arr[bernoulli_counter] == 1):
                G_w[i] = logreg_grad(deepcopy(x), [A_i, y_i, l2, sparse_stoch])
            else:
                sum_of_good = sum(G_w - G_w1)
                for i in range(num_of_workers - num_of_byz, num_of_workers):
                    G_w[i] = -0.1 * sum_of_good / (num_of_workers - num_of_byz) + G_w1
                
        if attack == "ALIE":
            if (bernoulli_arr[bernoulli_counter] == 1):
                G_w[i] = logreg_grad(deepcopy(x), [A_i, y_i, l2, sparse_stoch])
            else:    
                exp_of_good = sum(G_w - G_w1) / (num_of_workers - num_of_byz)
                var_of_good = (sum((G_w-G_w1) * (G_w - G_w1))) / (num_of_workers - num_of_byz) - exp_of_good * exp_of_good
                for i in range(num_of_workers - num_of_byz, num_of_workers):
                    G_w[i] = exp_of_good - 1.06 * var_of_good + G_w1
        
                
        if (bernoulli_arr[bernoulli_counter] == 1):
            x1 = deepcopy(x)
            G_w1 = np.mean(G_w, axis=0)
            x = x - gamma * G_w1
        else:
            if agg == "GM":
                perm = np.random.permutation(num_of_workers)
                x1 = deepcopy(x)
                G_w1 = GM(perm, 2, G_w)            
                x = x - gamma * G_w1
            
            if agg == "M":
                perm = np.random.permutation(num_of_workers)
                x1 = deepcopy(x)
                G_w1 = np.mean(G_w, axis=0)
                x = x - gamma * G_w1
            
            if agg == "CM":
                perm = np.random.permutation(num_of_workers)
                x1 = deepcopy(x)
                G_w1 = CM(perm, 2, G_w)
                x = x - gamma * G_w1
        
        bernoulli_counter += 1
            
        indices_counter += 1
        num_of_data_passes += 2.0 * batch_size / m
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0]) - f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break

    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0]) - f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)

    res = {'last_iter': x, 'func_vals': func_val, 'iters': its, 'time': tim, 'data_passes': data_passes,
           'squared_distances': sq_distances, 'num_of_workers': num_of_workers, 'num_of_byz': num_of_byz}

    with open("dump/" + filename + "_Byz_VR_MARINA_" + attack + "_" + agg + "_gamma_" + str(gamma) + "_l2_" + str(l2) + "_p_" + str(p) + "_epochs_" + str(S) + "_workers_" + str(num_of_workers) + "_batch_" + str(batch_size) + "_byz_" + str(num_of_byz) + ".txt", 'wb') as file:
        pickle.dump(res, file)
    return res