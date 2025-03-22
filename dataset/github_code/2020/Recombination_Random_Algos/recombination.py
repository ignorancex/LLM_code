# THIS REPOSITORY CONTATINS THE ALGORITHMS EXPLAINED IN THE WORK
# Cosentino, Oberhauser, Abate - "A randomized algorithm to reduce the support of discrete measures " 
# NeurIPS 2020


####################################################################################
# 
# All the functions recomb_* take as argument the matrix of the points X 
# Nxn containing the N points in R^n. All the functions return a convex combination 
# of points and weights of the origin, therefore they require that E[X]=0 respect
# to the desired probability measure mu, except recomb_log, recomb_combined. Moroever, all the functions 
# recomb_* require a maximum number of iterations and there is a possibility to choose
# an initial basis for the cone with the parameter idx; the only exception again is 
# represented by the function recomb_log, recomb_combined.
# All the functions recomb_* return w_star, idx_star, x_star, t, ERR, iterations, 
# eliminated_points. w_star is the vector containtng the weights of the points
# x_star which represent the soltuion of the recombination problem. idx_star tells us the indices 
# of x_star in X. t is the running time of the functions, ERR returns 0 if no Errors 
# have been recognised, iterations returns the number of iterations necessary to find the solution.
# eliminated points represents the points the algorithm was able to eliminate.
# As already mentioned the only function that works slightly different is recomb_log, recomb_combined.
# 
# We have partially rewritten the algorithm presented in Tchernychova Lyons, 
# "Caratheodory cubature measures", PhD thesis, University of Oxford, 2016.
# Partially because we do not consider their studies relative to different trees, 
# due to numerical instability in some cases as declared in the work by the authours 
# and also because there the analysis was condiucted in specific cases.
# See the reference for more details. 
# 
# The name of the variables are as close as possible to the name of the vairables 
# of the cited work.
# 
####################################################################################

import numpy as np
import copy, timeit

from numpy.core.records import get_remaining_size

######################################################################################
# recomb_basic represents the not-optimized algorithm.
######################################################################################

def recomb_basic(X, max_iter, idx=[]):
    # It takes X (N x n) and returns the weights w_star and the n+1 points
    # x_star.

    # X must have mean 0

    tic = timeit.default_timer()
    N, n = np.shape(X)
    iterations, ERR = 1, 0
    eliminated_points = np.array([], dtype=int)
    
    # print("max_iterations", max_iterations)
    w_star = np.zeros([1, n+1])
    x_star = np.zeros([1, n+1])
    idx_star = np.nan*np.ones([1, n+1])

    while True:

        if N<=n+1:
            w_star = np.linalg.solve(np.append(np.transpose(X),np.ones([1,n+1]), axis=0),
                                     np.append(np.zeros([1,n]),1))
            x_star, ERR = X, 0

            if any(w_star<0) or any(w_star>1) or np.round(np.sum(w_star),6)!=1.:
                print("Warning weights found no contraint, N<=n+1")
                ERR, w_star, x_star = 3, w_star*np.nan, x_star*np.nan
            
            t = timeit.default_timer()-tic
            idx_star = np.arange(n)
            return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)
        
        if iterations > max_iter:
            print("ERROR NO convergence")
            ERR, w_star, x_star = 2, w_star*np.nan, x_star*np.nan
            t = timeit.default_timer()-tic
            return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)

        idx = np.random.choice(N, n, replace = False)
        # idx = choose_initial_points(X)
        cone_basis = X[idx,:]
        A = np.linalg.inv(np.transpose(cone_basis))
        AX = np.matmul(A, np.transpose(X))

        # tmp_1 = indices of the points inside (if any) the inverse cone defined via cone_basis
        tmp_1 = np.transpose(np.round(AX,6)<=0)
        tmp_1 = np.arange(N)[np.all(tmp_1,1)]

        if np.any(tmp_1):
            # Compute weights
            x_star = X[tmp_1[0],:]
            x_star = np.append(cone_basis, x_star[np.newaxis], axis=0)
            x_star = np.transpose(x_star)
            w_star = np.linalg.solve(np.append(x_star,np.ones([1,n+1]), axis=0),
                                     np.append(np.zeros([1,n]),1))

            if any(w_star<0) or any(w_star>1) or np.round(sum(w_star),6)!=1.:
                print("Warning weights found no contraint", w_star, sum(w_star))
                ERR, w_star, x_star = 1, w_star, x_star
            
            t = timeit.default_timer()-tic
            idx_star = np.append(idx,tmp_1[0])
            return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)
        
        iterations += 1

######################################################################################
# recomb_NOMor_NOreset represents the optimized algorithm without the use of the  
# Sherman–Morrison formula. Moreover, it does not use the reset 
# strategy neither.
######################################################################################

def recomb_NOMor_NOreset(X, max_iter, idx=[], DEBUG=False):
    # It takes X (N x n) and returns the weights w_star and the n+1 points
    # x_star.

    # X must have mean 0

    tic = timeit.default_timer()
    N, n = np.shape(X)
    
    if len(idx)!=n:
        idx = np.random.choice(N, n, replace = False)
        # idx = choose_initial_points(X)

    iterations, ERR = 1, 0
    eliminated_points = np.array([], dtype=int)
    
    w_star = np.zeros([1, n+1])
    x_star = np.zeros([1, n+1])
    idx_star = np.nan*np.ones([1, n+1])
    
    while True:

        if N<=n+1:
            w_star = np.linalg.solve(np.append(np.transpose(X),np.ones([1,n+1]), axis=0),
                                     np.append(np.zeros([1,n]),1))
            x_star, ERR = X, 0

            if any(w_star<0) or any(w_star>1) or np.round(np.sum(w_star),6)!=1.:
                print("Warning weights found, no constraint, data<=d+1")
                ERR, w_star, x_star = 3, w_star*np.nan, x_star*np.nan
            
            t = timeit.default_timer()-tic
            idx_star = np.arange(n)
            return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)
        
        if iterations > max_iter:
            if DEBUG: print("ERROR NO convergence")
            ERR, w_star, x_star = 2, w_star*np.nan, x_star*np.nan
            t = timeit.default_timer()-tic
            return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)
        
        cone_basis = X[idx,:]

        # add numerical stability to the method
        if iterations==1:    
            while np.linalg.matrix_rank(cone_basis)<n:
                idx = np.random.choice(N, n, replace = False)
                cone_basis = X[idx,:]

        A = np.linalg.inv(np.transpose(cone_basis))
        AX = np.matmul(A, np.transpose(X))
        
        # tmp_1 = indices of the points inside (if any) the inverse cone defined via cone_basis
        tmp_1 = np.transpose(np.round(AX,6)<=0)
        tmp_1 = np.arange(N)[np.all(tmp_1,1)]

        if len(tmp_1)>0:        
            # Compute weights
            x_star = X[tmp_1[0],:]
            x_star = np.append(cone_basis, x_star[np.newaxis], axis=0)
            w_star = np.linalg.solve(np.append(np.transpose(x_star),np.ones([1,n+1]), axis=0),
                                     np.append(np.zeros([1,n]),1))

            if any(w_star<0) or any(w_star>1) or np.round(sum(w_star),6)!=1.:
                print("Warning weights found, no constraint")
                ERR, w_star, x_star = 1, w_star, x_star
            
            t = timeit.default_timer()-tic
            idx_star = np.append(idx,tmp_1[0])
            return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)
        
        # delete useless points
        tmp_2 = np.transpose(np.round(AX,6)>0)
        tmp_2 = np.arange(N)[np.all(tmp_2,1)]

        tmp2_in_idx = np.in1d(idx,tmp_2)
        if any(tmp2_in_idx):  
            if DEBUG:
                print("ERROR wrong elimination") 
            # means numerical problem during the inversion (see matrix A)
            tmp_2 = []
            A = np.linalg.inv(np.transpose(cone_basis))
            AX = np.matmul(A, np.transpose(X))

            # ERR, w_star, x_star = 4, w_star*np.nan, x_star*np.nan
            # t = timeit.default_timer()-tic
            # return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)
        elif len(tmp_2)>0:
            if DEBUG:
                print("I am deleting ", len(tmp_2), "point/s")
            # delete useless points
            X[tmp_2,:] = np.nan
            AX[:,tmp_2] = np.nan
            if iterations>1:
                normX[tmp_2] = np.nan
                X_sphere[tmp_2,:] = np.nan
            eliminated_points = np.append(eliminated_points,
                                        tmp_2).reshape(-1)

        # compute norm:
        # given the searching srategy of the algorithm we compute the projection of the
        # points on the sphere once for ever
        if iterations == 1: 
            normX = np.sqrt(np.sum(X*X,1))
            X_sphere = np.divide(X,normX[:,np.newaxis])
            
        idx_tobechanged = (iterations-1) % n
        idx_max = new_point_meanmax(X_sphere, idx, idx_tobechanged)
        idx[idx_tobechanged] = idx_max

        iterations += 1

        if iterations % 200 == 0:
            print("Recombination procedure iteration = ", iterations)

####################################################################
# recomb_Mor_NOreset represents the optimized algorithm using the  
# Sherman–Morrison formula. 
# recomb_Mor_reset add the reset strategy to recomb_Mor_NOreset
####################################################################

def recomb_1(X):
    # special case when n==1

    # It takes X (N x 1) and returns the weights w_star and the 1+1 points
    # x_star.

    # X must have mean 0
    
    N, n = np.shape(X)
    tic = timeit.default_timer()
    idx = [0,0]

    sign = X>=0
    sign = sign[:,0]
    idx[0] = np.arange(N)[sign][0]
    idx[1] = np.arange(N)[np.logical_not(sign)][0]

    x_star = X[idx]

    w_star = np.zeros(2)
    w_star[1] = x_star[0]/(x_star[0]-x_star[1])
    w_star[0] = 1 - w_star[1]

    iterations, ERR, eliminated_points = 1, 0, 0
    t = timeit.default_timer()-tic

    return w_star, idx, x_star, t, ERR, iterations, eliminated_points

def recomb_Mor_NOreset(X, max_iter, idx=[], X_sphere = [], DEBUG=False, HC_paradigm=False):
    # It takes X (N x n) and returns the weights w_star and the n+1 points
    # x_star.
    # X_sphere takes the coordinate of the projection of the points in 
    # X on the sphere

    # X must have mean 0 
    # HC_paradigm is a boolean varialbe that says if this function has been called by the 
    # Hierarchical Clustering paradigm function recomb_log, it just says if this function can 
    # print or not some of the output 

    tic = timeit.default_timer()
    N, n = np.shape(X)

    if n==1:
        return recomb_1(X)

    if len(idx) != n:
        idx = np.random.choice(N, min(N,n), replace = False)
        
        # idx = choose_initial_points(X)
        # idx = choose_initial_points_tens_sq(X)
    
    iterations, ERR = 1, 0
    eliminated_points = np.array([], dtype=int)
    remaing_points = np.shape(X)[0]

    idx_story = np.arange(N)
    w_star = np.zeros([1, n+1])
    x_star = np.zeros([1, n+1])
    idx_star = np.nan*np.ones([1, n+1])

    while True:
        
        if remaing_points<=n+1:
            # idx_tmp = np.logical_not(np.isnan(X[:,0]))
            x_star = X[idx_story,:]
            # w_star = np.linalg.solve(np.append(np.transpose(x_star),np.ones([1,remaing_points]), axis=0),
            #                          np.append(np.zeros([1,remaing_points-1]),1))
            w_star = np.linalg.lstsq(np.append(np.transpose(x_star),np.ones([1,remaing_points]), axis=0).T, 
                            np.append(np.zeros([1,remaing_points-1]),1))[0]
            ERR = 0

            if any(w_star<0) or any(w_star>1) or np.round(np.sum(w_star),6)!=1.:
                if DEBUG==True:
                    print("Warning weights found no contraint, data<=d+1")
                ERR, w_star, x_star = 3, w_star*np.nan, x_star*np.nan
            
            t = timeit.default_timer()-tic
            idx_star = idx_story
            return w_star, idx_star, x_star, t, ERR, iterations, eliminated_points

        if iterations > max_iter:
            ERR, w_star, x_star = 2, w_star*np.nan, x_star*np.nan
            if not HC_paradigm:
                if DEBUG: print("ERROR: NO convergence")
                t = timeit.default_timer()-tic
                return w_star, idx_star, x_star, t, ERR, iterations, eliminated_points
            
            t = timeit.default_timer()-tic
            return w_star, idx_star*np.nan, x_star, t, ERR, iterations, eliminated_points, X_sphere
        
        if iterations==1:
            cone_basis = X[idx,:]
            
            ii = 0
            # add numerical stability to the method
            while np.linalg.matrix_rank(cone_basis)<n:
                idx = np.random.choice(N, n, replace = False)
                cone_basis = X[idx,:]
                ii += 1
                if ii==5:
                    if not HC_paradigm:
                        if DEBUG == True:
                            print("ERROR: singular cone basis")
                    ERR, w_star, x_star = 6, w_star*np.nan, x_star*np.nan
                    t = timeit.default_timer()-tic
                    return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)
                
            A = np.linalg.inv(np.transpose(cone_basis))
            
            # DEBUG
            # test_inv = np.matmul(A,np.transpose(cone_basis))
            # if not np.allclose(test_inv, np.eye(d)):
            #     print("test_inv", np.allclose(test_inv, np.eye(d)))
            
            AX = np.zeros((n,N))
            AX[:,idx_story] = np.matmul(A, np.transpose(X[idx_story]))

        if iterations>1:
            idx_changed = (iterations-2) % n

            A, prod, const = update_inverse(A,X[idx,:],
                                idx_changed,
                                X[idx_max,:])
              
            idx[idx_changed] = idx_max
            
            # add numerical stability to the method
            test_inv = np.matmul(A,np.transpose(X[idx,:]))
            if not np.allclose(test_inv, np.eye(n)):
                cone_basis = X[idx,:]
                A = np.linalg.inv(np.transpose(cone_basis))

                test_inv = np.matmul(A,np.transpose(X[idx,:]))
                if not np.allclose(test_inv, np.eye(n)):
                    if DEBUG == True:
                        print("WARNING: numerical instability")
                    ERR, w_star, x_star = 1, w_star*np.nan, x_star*np.nan
                    idx_star = idx_star*np.nan
                    t = timeit.default_timer()-tic
                    return w_star, idx_star, x_star, t, ERR, iterations, eliminated_points
                
                AX = np.zeros((n,N))
                AX[:,idx_story] = np.matmul(A, np.transpose(X[idx_story]))
            else:
                AX -= np.matmul(prod,AX[idx_changed,:][np.newaxis])/const
                cone_basis = X[idx,:]
            
            # DEBUG
            # AX_2 = np.matmul(A,np.transpose(X))
            # if not np.allclose(AX, AX_2):
            #     print("false")
        
        # tmp_1 = indices of the points inside (if any) the inverse cone defined via cone_basis
        tmp_1 = np.transpose(AX[:,idx_story]<=0)
        tmp_1 = np.arange(remaing_points)[np.all(tmp_1,1)]

        if len(tmp_1)>0:
            
            x_star = X[idx_story[tmp_1[0]],:]
            # Compute weights
            w_star = solve_given_inverse(cone_basis,x_star,A)
            x_star = np.append(cone_basis, x_star[np.newaxis], axis=0)
            idx_star = np.append(idx,idx_story[tmp_1[0]])

            if any(w_star<0) or any(w_star>1) or np.round(sum(w_star),6)!=1.:
                if DEBUG == True:
                    print("Warning weights found, no constraint")
                ERR, w_star, x_star = 1, w_star*np.nan, x_star*np.nan
                idx_star = idx_star*np.nan
            elif not np.allclose(np.sum(np.multiply(x_star,w_star[:,np.newaxis]),0),np.zeros(n)):
                if DEBUG == True:
                    print("Warning, low precison in the solution")
                ERR, w_star, x_star = 1, w_star*np.nan, x_star*np.nan
                idx_star = idx_star*np.nan

            t = timeit.default_timer()-tic
            
            return w_star, idx_star, x_star, t, ERR, iterations, eliminated_points
        
        # delete useless points
        tmp_2 = np.transpose(AX[:,idx_story]>0)
        tmp_2 = np.arange(remaing_points)[np.all(tmp_2,1)]

        idx_in_tmp2 = np.in1d(idx,idx_story[tmp_2])
        if any(idx_in_tmp2):
            # means numerical problem during the inversion (see matrix A)
            if DEBUG:
                print("wrong elimination, numerical instability")
            A = np.linalg.inv(np.transpose(cone_basis))
            # AX = np.matmul(A, np.transpose(X[idx_story]))
            AX = np.zeros((n,N))
            AX[:,idx_story] = np.matmul(A, np.transpose(X[idx_story]))
            tmp_2 = []

            # ERR, w_star, x_star = 6, w_star*np.nan, x_star*np.nan
            # t = timeit.default_timer()-tic
            # return(w_star, idx_star, x_star, t, ERR, iterations, eliminated_points)    
        elif len(tmp_2)>0:
            if DEBUG:
                print("I am deleting ", len(tmp_2), "point/s")
            
            eliminated_points = np.append(eliminated_points,
                                        idx_story[tmp_2]).reshape(-1)
            # delete useless points
            idx_story = np.delete(idx_story, tmp_2)
            # AX = AX[:,idx_story]
            # X[tmp_2,:] = np.nan
            # AX[:,tmp_2] = np.nan
            # if iterations>1:
            #     X_sphere[tmp_2,:] = np.nan
            
            remaing_points -= len(tmp_2)

        if remaing_points <= n+1:
            continue
        # compute norm
        # given the searching srategy of the algorithm we compute the projection of the
        # points on the sphere once for ever

        if iterations == 1 and X_sphere == []: 
            normX = np.sqrt(np.sum(np.multiply(X,X),1))
            # X_sphere = np.zeros((N,n))
            X_sphere = np.divide(X,normX[:,np.newaxis])       
        # elif iterations == 1:
        #     idx_tmp = np.arange(N)[np.all(X_sphere==0,1)]
        #     idx_tmp = idx_tmp[np.isin(idx_tmp,idx_story)]
        #     normX = np.sqrt(np.sum(np.multiply(X[idx_tmp],X[idx_tmp]),1))
        #     X_sphere[idx_tmp] = np.divide(X[idx_tmp],normX[:,np.newaxis]) 
        
        idx_tobechanged = (iterations-1) % n
        
        idx_in_tmp3 = np.arange(remaing_points)[np.isin(idx_story,idx)]
        idx_tobechanged_tmp = np.arange(len(idx))[np.argsort(idx)==idx_tobechanged][0]
        
        idx_max = new_point_meanmax(X_sphere[idx_story], idx_in_tmp3, idx_tobechanged_tmp)
        idx_max = idx_story[idx_max]

        iterations += 1
        
        if iterations % 200 == 0 and (not HC_paradigm):
            if DEBUG == True:
                print("Recombination procedure iteration = ", iterations)

def recomb_Mor_reset(X, max_iter, idx=[], reset_factor=0, X_sphere = [], DEBUG=False, HC_paradigm=False):
    # this functions add a reset strategy to recomb_Mor_NOreset
    # using a strategy defined in the work.

    # X must have mean 0 
    # HC_paradigm is a boolean varialbe that says if this function has been called by the 
    # Hierarchical Clustering paradigm function recomb_log, it just says if this function can 
    # print or not some of the output 
    
    tic = timeit.default_timer()
    N, n = X.shape

    if n == 1:
        return recomb_1(X)
    
    
    if reset_factor == 0:
        # reset_factor = n**2
        reset_factor = 2*(n+1)
        # reset_factor = n+1
    
    total_iteration = 0
    eliminated_points = np.array([])

    i = 0
    while True:
        
        if i <=1:
            max_iter_LV = reset_factor
        else:
            max_iter_LV = reset_factor * LasVegas_strat(i)
        
        results = recomb_Mor_NOreset(X, max_iter_LV, idx, X_sphere, DEBUG, HC_paradigm)
        
        results = [results for results in results]
        eliminated_points = np.append(eliminated_points,results[6]).reshape(-1)
        
        ERR = results[4]
        
        if ERR == 0:
            results[3] = timeit.default_timer() - tic
            results[5] = total_iteration + results[5]
            results[6] = eliminated_points
            # results.append(X_sphere)
            return results[:]
        
        total_iteration += results[5]
        
        if total_iteration >= max_iter:
            w_star = np.empty(n+1)*np.nan
            x_star = np.empty(n+1)*np.nan
            idx_star = np.empty(n+1)*np.nan
            if not HC_paradigm:
                if DEBUG: print("ERROR: NO convergence")
            ERR = 2
            t = timeit.default_timer()-tic
            return(w_star, idx_star, x_star, t, ERR, total_iteration, eliminated_points)
        
        if ERR != 6 and len(results)>7:
            X_sphere = results[7]

        i += 1
        if DEBUG:
            print("reset")

####################################################################
# recomb_log uses the clustering paradigm using recomb_Mor_reset
####################################################################

def recomb_log(X, max_iter=0, mu=0, fact=0,DEBUG=False):
    # It takes X (N x n) and returns the weights w_star and the n+1 points
    # x_star.
    # mu represents the weights of the points in X, while fact is 
    # a parameter related with the clustering paradigm

    # Note that this function does not need the 
    # barycenter of the point in X (relatively to mu) to be 0

    tic = timeit.default_timer()
    N, n = X.shape
    
    if max_iter == 0:
        max_iter = n**4

    if fact == 0:
        fact = 50
    number_of_sets = fact*(n+1)
    
    if np.all(mu==0) or len(mu)!=N or np.any(mu<0):
        mu = np.ones(N)/N

    com = np.zeros(n) # Center Of Mass
    
    idx_story = np.arange(N)
    idx_story = idx_story[mu!=0]
    remaining_points = len(idx_story)

    while True:
        
        # remaining points at the next step are = to remaining_points/number_of_sets*(n+1)

        numb_points_next_step = int(remaining_points/fact)
        
        if numb_points_next_step >= number_of_sets: 
            number_of_el = int(remaining_points/number_of_sets)
            idx_next_step = []
        else:
            threshold = remaining_points - (number_of_sets - numb_points_next_step) > number_of_sets
            if threshold:
                # remaining_points = int(number_of_sets*number_of_sets/(n+1))
                idx_next_step = idx_story[- (number_of_sets - numb_points_next_step):]
                idx_story = idx_story[:- (number_of_sets - numb_points_next_step)]
                remaining_points = len(idx_story)
                number_of_el = int(remaining_points/number_of_sets)
                # numb_points_next_step = int(remaining_points/number_of_sets)*(n+1)
            else:
                com = np.sum(np.multiply(X[idx_story],mu[idx_story,np.newaxis]),0)
                w_star, idx_star, x_star, _, ERR, _, _ = recomb_Mor_reset(X[idx_story]-com, max_iter, 
                                                                            [], 0, [], DEBUG, True)
                if ERR != 0:
                    # w_star, x_star = np.nan, np.nan
                    t = timeit.default_timer()-tic
                    return w_star, idx_star, x_star, t, ERR, np.nan, np.nan
                idx_star = idx_story[idx_star]
                toc = timeit.default_timer()-tic
                return w_star, idx_star, x_star, toc, ERR, np.nan, np.nan

        X_tmp = np.empty((number_of_sets,n))
        # mu_tmp = np.empty(number_of_sets)

        idx = idx_story[:number_of_el*number_of_sets].reshape(number_of_el,-1)
        X_tmp = np.multiply(X[idx],mu[idx,np.newaxis]).sum(axis=0)
        tot_weights = np.sum(mu[idx],0)

        idx_last_part = idx_story[number_of_el*number_of_sets:]
        X_tmp[-1] += np.multiply(X[idx_last_part],mu[idx_last_part,np.newaxis]).sum(axis=0)
        tot_weights[-1] += np.sum(mu[idx_last_part],0)

        X_tmp = np.divide(X_tmp,tot_weights[np.newaxis].T)
        
        com = np.sum(np.multiply(X_tmp,tot_weights[:,np.newaxis]),0)/np.sum(tot_weights)

        w_star, idx_star, _, _, ERR, _, _ = recomb_Mor_reset(X_tmp-com, max_iter, [], 0, [], DEBUG, True)
        
        if ERR != 0:
            print("ERROR: recombined measure not found by the speicified number of iterations")
            w_star, x_star, idx_star = np.nan, np.nan, np.nan
            t = timeit.default_timer()-tic
            return w_star, idx_star, x_star, t, ERR, np.nan, np.nan

        idx_tomaintain = idx[:,idx_star].reshape(-1)
        idx_tocancel = np.ones(idx.shape[1]).astype(bool)
        idx_tocancel[idx_star] = False
        idx_tocancel = idx[:,idx_tocancel].reshape(-1)

        mu[idx_tocancel] = 0.
        mu_tmp = np.multiply(mu[idx[:,idx_star]],w_star)
        mu_tmp = np.divide(mu_tmp,tot_weights[idx_star])
        mu[idx_tomaintain] = mu_tmp.reshape(-1)*np.sum(tot_weights)

        idx_tmp = np.equal(idx_star,number_of_sets-1)
        idx_tmp = np.arange(len(idx_tmp))[idx_tmp!=0]
        #if idx_star contains the last barycenter, whose set could have more points
        if len(idx_tmp)>0:    
            mu_tmp = np.multiply(mu[idx_last_part],w_star[idx_tmp])
            mu_tmp = np.divide(mu_tmp,tot_weights[idx_star[idx_tmp]])
            mu[idx_last_part] = mu_tmp*np.sum(tot_weights)
            idx_tomaintain = np.append(idx_tomaintain,idx_last_part)
        else:
            idx_tocancel = np.append(idx_tocancel,idx_last_part)
            mu[idx_last_part] = 0.
        
        idx_story = np.copy(idx_tomaintain)
        idx_story = np.append(idx_story,idx_next_step).astype(int)
        remaining_points = len(idx_story)
        # idx_next_step = []
        # remaining_points = np.sum(mu>0)

        if remaining_points-len(idx_last_part)<=number_of_sets:
            
            com = np.sum(np.multiply(X[idx_story],mu[idx_story,np.newaxis]),0)
            w_star, idx_star, x_star, _, ERR, _, _ = recomb_Mor_reset(X[idx_story]-com, max_iter,
                                                                    [], 0, [], DEBUG, True)
            if ERR != 0:
                t = timeit.default_timer()-tic
                return w_star, idx_star, x_star, t, ERR, np.nan, np.nan
            idx_star = idx_story[idx_star]

            # DEBUG
            # com_recombined = np.sum(np.multiply(X[idx_star],w_star[:,np.newaxis]),0)
            # if not np.allclose(com, com_recombined):
            #     print("error")
            
            toc = timeit.default_timer()-tic
            return w_star, idx_star, x_star, toc, ERR, np.nan, np.nan

####################################################################
# recomb_combined uses the clustering paradigm using both the 
# Radnomized Algorithm recomb_Mor_NOreset both Tchernychova_Lyons
####################################################################

def recomb_combined(X, max_iter=0, mu=0, fact=0, DEBUG=False):
    # It takes X (N x n) and returns the weights w_star and the n+1 points
    # x_star.
    # mu represents the weights of the points in X, while fact is 
    # a parameter related with the clustering paradigm

    # Note that this is function does not need the 
    # barycenter of the point in X (relatively to mu) to be 0

    tic = timeit.default_timer()
    N, n = X.shape
    
    if max_iter == 0:
        max_iter = 2*n #n**4

    if fact == 0:
        fact = 50
    number_of_sets = fact*(n+1)
    
    if np.all(mu==0) or len(mu)!=N or np.any(mu<0):
        mu = np.ones(N)/N

    com = np.zeros(n) # Center Of Mass

    idx_story = np.arange(N)
    idx_story = idx_story[mu!=0]
    remaining_points = len(idx_story)

    while True:
        
        # remaining points at the next step are = to remaining_points/number_of_sets*(n+1)

        numb_points_next_step = int(remaining_points/fact)
        if numb_points_next_step >= number_of_sets: 
            number_of_el = int(remaining_points/number_of_sets)
            idx_next_step = []
        else:
            threshold = remaining_points - (number_of_sets - numb_points_next_step) > number_of_sets
            if threshold:
                # remaining_points = int(number_of_sets*number_of_sets/(n+1))
                idx_next_step = idx_story[- (number_of_sets - numb_points_next_step):]
                idx_story = idx_story[:- (number_of_sets - numb_points_next_step)]
                remaining_points = len(idx_story)
                number_of_el = int(remaining_points/number_of_sets)
                # numb_points_next_step = int(remaining_points/number_of_sets)*(n+1)
            else:
                com = np.sum(np.multiply(X[idx_story],mu[idx_story,np.newaxis]),0)
                w_star, idx_star, x_star, _, ERR, _, _ = recomb_Mor_NOreset(X[idx_story]-com, max_iter, 
                                                                            [], [], DEBUG)
                if ERR != 0:
                    #####################################################################
                    if DEBUG == True:
                        print("Using determiinistic Algorithm")
                    w_star, idx_star, x_star, _, ERR, _, _  = Tchernychova_Lyons(X[idx_story],mu[idx_story])
                    w_star = w_star/np.sum(mu[idx_story])
                    #####################################################################
                
                idx_star = idx_story[idx_star]

                toc = timeit.default_timer()-tic
                return w_star, idx_star, x_star, toc, ERR, np.nan, np.nan

        X_tmp = np.empty((number_of_sets,n))
        # mu_tmp = np.empty(number_of_sets)

        idx = idx_story[:number_of_el*number_of_sets].reshape(number_of_el,-1)
        X_tmp = np.multiply(X[idx],mu[idx,np.newaxis]).sum(axis=0)
        tot_weights = np.sum(mu[idx],0)

        idx_last_part = idx_story[number_of_el*number_of_sets:]
        X_tmp[-1] += np.multiply(X[idx_last_part],mu[idx_last_part,np.newaxis]).sum(axis=0)
        tot_weights[-1] += np.sum(mu[idx_last_part],0)

        X_tmp = np.divide(X_tmp,tot_weights[np.newaxis].T)
        
        com = np.sum(np.multiply(X_tmp,tot_weights[:,np.newaxis]),0)/np.sum(tot_weights)

        w_star, idx_star, _, _, ERR, _, _ = recomb_Mor_NOreset(X_tmp-com, max_iter, [], [], DEBUG)
        
        if ERR != 0:
            #####################################################################
            if DEBUG == True:
                print("Using determiinistic Algorithm")
            w_star, idx_star, _, _, ERR, _, _  = Tchernychova_Lyons(X_tmp,np.copy(tot_weights))
            w_star = w_star/np.sum(tot_weights)
            #####################################################################
        
        # np.allclose(np.sum(np.multiply(X_tmp[idx_star],w_star[:,np.newaxis]),0)/np.sum(tot_weights),com)
        # np.sum(tot_weights))

        idx_tomaintain = idx[:,idx_star].reshape(-1)
        idx_tocancel = np.ones(idx.shape[1]).astype(bool)
        idx_tocancel[idx_star] = False
        idx_tocancel = idx[:,idx_tocancel].reshape(-1)

        mu[idx_tocancel] = 0.
        mu_tmp = np.multiply(mu[idx[:,idx_star]],w_star)
        mu_tmp = np.divide(mu_tmp,tot_weights[idx_star])
        mu[idx_tomaintain] = mu_tmp.reshape(-1)*np.sum(tot_weights)

        idx_tmp = np.equal(idx_star,number_of_sets-1)
        idx_tmp = np.arange(len(idx_tmp))[idx_tmp!=0]
        #if idx_star contains the last barycenter, whose set could have more points
        if len(idx_tmp)>0:    
            mu_tmp = np.multiply(mu[idx_last_part],w_star[idx_tmp])
            mu_tmp = np.divide(mu_tmp,tot_weights[idx_star[idx_tmp]])
            mu[idx_last_part] = mu_tmp*np.sum(tot_weights)
            idx_tomaintain = np.append(idx_tomaintain,idx_last_part)
        else:
            idx_tocancel = np.append(idx_tocancel,idx_last_part)
            mu[idx_last_part] = 0.
        
        idx_story = np.copy(idx_tomaintain)
        idx_story = np.append(idx_story,idx_next_step).astype(int)
        remaining_points = len(idx_story)
        # idx_next_step = []
        # remaining_points = np.sum(mu>0)

        if remaining_points-len(idx_last_part)<=number_of_sets:
            
            com = np.sum(np.multiply(X[idx_story],mu[idx_story,np.newaxis]),0)
            w_star, idx_star, x_star, _, ERR, _, _ = recomb_Mor_NOreset(X[idx_story]-com, max_iter,
                                                                    [], [], DEBUG)
            if ERR != 0:
                #####################################################################
                if DEBUG == True:
                    print("Using determiinistic Algorithm")
                w_star, idx_star, x_star, _, ERR, _, _  = Tchernychova_Lyons(X[idx_story],np.copy(mu[idx_story]))
                w_star = w_star/np.sum(mu[idx_story])
                #####################################################################
            idx_star = idx_story[idx_star]

            # DEBUG
            # com_recombined = np.sum(np.multiply(X[idx_star],w_star[:,np.newaxis]),0)
            # if not np.allclose(com, com_recombined):
            #     print("error")
            
            toc = timeit.default_timer()-tic
            return w_star, idx_star, x_star, toc, ERR, np.nan, np.nan

####################################################################
# Tchernychova_Lyons* functions are the algortihms presented in 
# Tchernychova, Lyons - Caratheodory Cubature Measures, PhD Thesis, 
#                       Univeristy of Oxford, 2016
####################################################################

def Tchernychova_Lyons(X, mu=0,DEBUG=False):

    # It takes X (N x n) and returns the weights w_star and the n+1 points
    # x_star.
    # mu represents the weights of the points in X

    # This function does not need the 
    # barycenter of the point in X (relatively to mu) to be 0

    N, n = X.shape
    tic = timeit.default_timer()

    number_of_sets = 2*(n+1)
    
    if np.all(mu==0) or len(mu)!=N or np.any(mu<0):
        mu = np.ones(N)/N
    
    idx_story = np.arange(N)
    idx_story = idx_story[mu!=0]
    remaining_points = len(idx_story)

    while True:
        
        if remaining_points <= n+1:
            idx_star = np.arange(len(mu))[mu>0]
            w_star = mu[idx_star]
            x_star = X[idx_star]
            toc = timeit.default_timer()-tic
            return w_star, idx_star, X[idx_star], toc, ERR, np.nan, np.nan
        elif n+1 < remaining_points <= number_of_sets:
            w_star, idx_star, x_star, _, ERR, _, _ = Tchernychova_Lyons_CAR(X[idx_story], np.copy(mu[idx_story]),DEBUG)
            idx_story = idx_story[idx_star]
            mu[:] = 0.
            mu[idx_story] = w_star
            idx_star = idx_story
            x_star = X[idx_story]
            w_star = mu[mu>0]
            toc = timeit.default_timer()-tic
            return w_star, idx_star, x_star, toc, ERR, np.nan, np.nan
        
        # remaining points at the next step are = remaining_points/card*(n+1)
        
        # number of elements per set
        number_of_el = int(remaining_points/number_of_sets)
        # WHAT IF NUMBER OF EL == 0??????
        # IT SHOULD NOT GET TO THIS POINT GIVEN THAT AT THE END THERE IS A IF

        X_tmp = np.empty((number_of_sets,n))
        # mu_tmp = np.empty(number_of_sets)

        idx = idx_story[:number_of_el*number_of_sets].reshape(number_of_el,-1)
        X_tmp = np.multiply(X[idx],mu[idx,np.newaxis]).sum(axis=0)
        tot_weights = np.sum(mu[idx],0)

        idx_last_part = idx_story[number_of_el*number_of_sets:]
        X_tmp[-1] += np.multiply(X[idx_last_part],mu[idx_last_part,np.newaxis]).sum(axis=0)
        tot_weights[-1] += np.sum(mu[idx_last_part],0)

        X_tmp = np.divide(X_tmp,tot_weights[np.newaxis].T)

        w_star, idx_star, _, _, ERR, _, _ = Tchernychova_Lyons_CAR(X_tmp, np.copy(tot_weights))
        
        idx_tomaintain = idx[:,idx_star].reshape(-1)
        idx_tocancel = np.ones(idx.shape[1]).astype(bool)
        idx_tocancel[idx_star] = 0
        idx_tocancel = idx[:,idx_tocancel].reshape(-1)

        mu[idx_tocancel] = 0.
        mu_tmp = np.multiply(mu[idx[:,idx_star]],w_star)
        mu_tmp = np.divide(mu_tmp,tot_weights[idx_star])
        mu[idx_tomaintain] = mu_tmp.reshape(-1)

        idx_tmp = idx_star == number_of_sets-1
        idx_tmp = np.arange(len(idx_tmp))[idx_tmp!=0]
        #if idx_star contains the last barycenter, whose set could have more points
        if len(idx_tmp)>0:    
            mu_tmp = np.multiply(mu[idx_last_part],w_star[idx_tmp])
            mu_tmp = np.divide(mu_tmp,tot_weights[idx_star[idx_tmp]])
            mu[idx_last_part] = mu_tmp
            idx_tomaintain = np.append(idx_tomaintain,idx_last_part)
        else:
            idx_tocancel = np.append(idx_tocancel,idx_last_part)
            mu[idx_last_part] = 0.

        idx_story = np.copy(idx_tomaintain)
        remaining_points = len(idx_story)
        # remaining_points = np.sum(mu>0)

def Tchernychova_Lyons_CAR(X,mu,DEBUG=False):
    # this functions reduce X from N points to n+1

    # com = np.sum(np.multiply(X,mu[np.newaxis].T),0)
    X = np.insert(X,0,1.,axis=1)
    N, n = X.shape
    U, Sigma, V = np.linalg.svd(X.T)
    # np.allclose(U @ np.diag(Sigma) @ V, X.T)
    U = np.append(U, np.zeros((n,N-n)),1)
    Sigma = np.append(Sigma, np.zeros(N-n))
    Phi = V[-(N-n):,:].T
    cancelled = np.array([], dtype=int)

    for _ in range(N-n):
        
        alpha = mu/Phi[:,0]
        idx = np.arange(len(alpha))[Phi[:,0]>0]
        idx = idx[np.argmin(alpha[Phi[:,0]>0])]
        cancelled = np.append(cancelled, idx)
        mu[:] = mu-alpha[idx]*Phi[:,0]
        mu[idx] = 0.

        if DEBUG and (not np.allclose(np.sum(mu),1.)):
            # print("ERROR")
            print("sum ", np.sum(mu))
        
        Phi_tmp = Phi[:,0]
        Phi = np.delete(Phi,0,axis=1)
        Phi = Phi - np.matmul(Phi[idx,np.newaxis].T,Phi_tmp[:,np.newaxis].T).T/Phi_tmp[idx]
        Phi[idx,:] = 0.
    
    w_star = mu[mu>0]
    idx_star = np.arange(N)[mu>0]
    return w_star, idx_star, np.nan, np.nan, 0., np.nan, np.nan
    
############################
# Auxiliary functions
############################

def choose_initial_points(X):
    # X_cp = np.copy(X)
    N, n = X.shape
    idx = np.ones(n, dtype=int)*-1
    # idx_order = np.random.choice(n, n, replace = False)
    for i in range(n):
        if idx[i] != -1:
            continue
        prob = np.ones(N)
        prob[idx[:i]] = 0.
        prob = prob/np.sum(prob)
        tmp_bool = True
        while tmp_bool:
            idx[i] = np.random.choice(N,1,p=prob)
            tmp_bool = X[idx[i],0] == np.nan
        sign = np.sign(X[idx[i],:])
        idx_tmp = np.all(np.sign(X) == -sign,1)
        if len(idx_tmp[idx_tmp==True])>0 and i != n-1:
            idx[i+1] = np.arange(N)[idx_tmp][0]
    return idx

def choose_initial_points_tens_sq(X):
    # X_cp = np.copy(X)
    N, n = X.shape
    d = int((-3+np.sqrt(9+8*n))/2.)
    # idx = np.ones(n, dtype=int)*-1
    idx = np.random.choice(N, n, replace = False)
    second_dim = d-1
    for i in range(d):
        second_dim += i+1
        idx_tmp = X[:,i] <= X[:,second_dim]
        prob = np.ones(N)
        prob[idx_tmp] = 0.
        prob[idx] = 0.
        prob = prob/np.sum(prob)
        tmp_bool = True
        while tmp_bool:
            idx[i] = np.random.choice(N,1,p=prob)
            tmp_bool = X[idx[i],0] == np.nan
        
        idx_tmp = X[:,i] > X[:,second_dim]
        prob = np.ones(N)
        prob[idx_tmp] = 0.
        prob[idx] = 0.
        prob = prob/np.sum(prob)
        tmp_bool = True
        while tmp_bool:
            idx[second_dim] = np.random.choice(N,1,p=prob)
            tmp_bool = X[idx[second_dim],0] == np.nan
        
    return idx

def tens_sq(X):
    N, n = X.shape
    for i in range(n):
        for j in range(i+1):
            X = np.append(X, 
                          np.multiply(X[:,i],X[:,j])[np.newaxis].T,
                          1)
    return X

def new_point_meanmax(X,idx_old,idx_tobechanged):
    # It returns the indeces of the new points in the X set. 
    # X = data, 
    # idx_old = indeces of the old cone basis, 

    # n = X.shape[1]
    # idx_tmp = idx_old[idx_tobechanged]

    if idx_tobechanged==0:
        mean = np.mean(X[idx_old[1:],:],0)
        mean = mean/np.linalg.norm(mean)
    else:
        mean = np.mean(X[idx_old[:idx_tobechanged],:],0)
        mean = mean/np.linalg.norm(mean)

    distances = np.abs(np.matmul(X,mean[:,np.newaxis])-1)
    
    distances[idx_old] = np.nan 
    # distances[idx_tmp] = np.nan
    idx_max = np.nanargmax(distances)

    # DEBUG: add stability to the mehtod: if the condition below is satisfied it means that 
    # the points in the new basis will be "almost" dependent
     
    # if distances[idx_max]>1.99999 and np.all(np.sign(X[idx_max]) == -np.sign(mean)):
    #     print("distances = ", distances[idx_max])

    return idx_max

def update_inverse(A_old,old_basis,pos,new_vector):
    # using the Sherman–Morrison this function updates the inverse A

    e_pos = np.zeros([1,np.shape(A_old)[0]])   # row vector
    e_pos[0,pos] = 1

    vector = new_vector-old_basis[pos,:]
    vector = np.transpose(vector[np.newaxis])  # column vector
    
    # e^T * A_i = A_old[pos,:]
    prod_1 = np.matmul(A_old,vector)
    prod_2 = np.matmul(prod_1,A_old[pos,:][np.newaxis]) # numerator

    c = 1+prod_1[pos]
    A_new = A_old - prod_2/c

    # TEST INVERSE DEBUG
    # new_basis = copy.copy(old_basis)
    # new_basis[pos,:] = new_vector
    # test_inv = np.around(np.matmul(A_new,np.transpose(new_basis)),2)
    # d = np.shape(old_basis)[0]
    # if not np.allclose(test_inv, np.eye(d)):
    #     print(np.allclose(test_inv, np.eye(d)))

    return(A_new, prod_1, c)

def solve_given_inverse(cone_basis,x_star,A):
    # using the Sherman–Morrison this function solve a system since we have already 
    # computed the inverse of part of the martix

    prod_1 = np.matmul(A,x_star)
    const = 1-np.sum(prod_1)
    w = -1/const*np.append(prod_1,-1)
    return(w)

def LasVegas_strat(i=1):
    # it returns the number of
    # iterations before the next reset

    k = np.log2(i+1)
    k = np.floor(k)

    if i == 2.**k-1:
        return(int(2.**(k-1)))
    else:
        return(LasVegas_strat(i-2.**k+1))

