"""
Implementation of the algorithm, for finite-sample data. 

"""
import numpy as np
import numpy.linalg as LA
import networkx as nx
from networkx.algorithms.clique import find_cliques as find_maximal_cliques
import itertools
import time

flatten_list = lambda t: list(set([item for sublist in t for item in sublist]))

def compute_objective(S1,S2,Delta,lambda_l1,sym_loss=False):
    '''
    Parameters
    ----------
    S1, S2 : 2d array
        Sample covariance matrices.
    Delta : 2d array
        Parameters to compute gradient wrt.
    lambda_l1: scalar
        penalty parameter for l1 regularization
    sym_loss : Boolean, optional
        Use symmetric loss or not. The default is False.

    Returns
    -------
    scalar: loss with l1 regularization
    '''
    if sym_loss == False:
        return (Delta.T@S1@Delta@S2).trace()/2 - (Delta@(S1-S2)).trace() \
            + lambda_l1*np.sum(np.abs(Delta))
    elif sym_loss == True:
        return (Delta.T@S1@Delta@S2).trace()/4+ (Delta.T@S2@Delta@S1).trace()/4 \
            - (Delta@(S1-S2)).trace() + lambda_l1*np.sum(np.abs(Delta))        
    else:
        print('sym_loss input (False by default) should be either False or True')
        return None
    
def soft_thresholding(x,alpha):
    '''
    returns soft(x,alpha)
    '''
    return np.maximum((np.abs(x)-alpha),0) * np.sign(x)


def Delta_Theta_func(S1,S2,lambda_l1=0.1,rho=1.0,n_max_iter=500,stop_cond=1e-6,verbose=False,return_sym=True):
    '''
    Difference of inverse covariance estimation.
    A Direct Approach for Sparse Quadratic Discriminant Analysis (Jiang et al. 2018)

    Parameters
    ----------
    S1, S2 : 2d array
        Sample covariance matrices.
    lambda_l1 : float
        l1 norm parameter for Delta_Theta estimations over multiple nodes. The default is 0.1.
    rho : float
        penalty parameter for ADMM. No need change in most cases. The default is 1.0.
    n_max_iter : integer
        maximum number of iterations for ADMM. Does not need to be too large. The default is 500.
    stop_cond : float
        stopping condition for ADMM iterations. The default is 1e-6.
    verbose : Boolean
        The default is False.
    return_sym : Boolean
        Take symmetric (Delta + Delta.T)/2 in the end. The default is True.

    Returns
    -------
    Phi : 2d array
        Main output. Estimated Delta_Theta difference of inverse covariances.
    obj_hist: array
        history of objective over the iterations.

    '''
    p = len(S1)
    # initialize Delta, Phi, Lambda. Fix rho
    Delta = np.zeros([p,p])
    Phi = np.zeros([p,p])
    Lambda = np.zeros([p,p])
    
    # find the minimum and maximum eigenvalues of S1 kronecker S2, for rho heuristics
    eigen_max = LA.eigvals(S1)[0]*LA.eigvals(S2)[0]
    eigen_min = LA.eigvals(S1)[-1]*LA.eigvals(S2)[-1]
    # assign rho based on these values and penalty parameter
    if rho is None:
        if lambda_l1 <= eigen_min:
            rho = eigen_min
        elif lambda_l1 <= eigen_max:
            rho = eigen_max
        else:
            rho = lambda_l1
        
    # compute SVD's for sample covariance matrices
    [U1,D1,_] = LA.svd(S1)
    [U2,D2,_] = LA.svd(S2)
    B = 1/(D1[:,np.newaxis]*D2[np.newaxis,] + rho) 
    
    obj_hist = np.zeros(n_max_iter)
    obj_hist[0] = compute_objective(S1,S2,Delta,lambda_l1)
    # now update
    for it in range(n_max_iter):
        A = (S1-S2) - Lambda + rho*Phi
        # update Delta, notice the Hadamard product
        Delta = U1@(B*(U1.T@A@U2))@U2.T
        # update Phi
        Phi = soft_thresholding(Delta+Lambda/rho,lambda_l1/rho)
        # update Lambda
        Lambda += rho*(Delta-Phi)
    
        obj = compute_objective(S1,S2,Delta,lambda_l1)
        obj_hist[it] = obj
    
        # check stopping condition
        if np.abs(obj-obj_hist[it-1]) < stop_cond*(np.abs(obj)+1):
            sparsity = np.mean(Phi!=0)
            if verbose == True:
                print('Sparsity is %.3f, Converged in %d iterations, lambda:%.3f, rho:%.3f'%(sparsity,it,lambda_l1,rho))
            if return_sym == True:
                Phi = (Phi+Phi.T)/2
            return Phi, obj_hist[:it]
        
    if return_sym == True:
        Phi = (Phi+Phi.T)/2
        
    sparsity = np.mean(Phi!=0)
    if verbose == True:
        print('Sparsity is %.3f, Converged in %d iterations, lambda:%.3f, rho:%.3f'%(sparsity,it,lambda_l1,rho))
    return Phi, obj_hist        


def diff_marginal_noise_sample_direct(S1,S2):
    '''
    returns marginal variances
    '''
    diff_marginal_noise = []
    M = np.arange(len(S1))
    for i in M:
        a = S1[i,i] - S2[i,i]
        diff_marginal_noise.append(a)
        
    return np.squeeze(diff_marginal_noise)


def build_descendants_sample(S1,S2,M,S,lambda_l1=0.1,rho=None,n_max_iter=500,\
                     stop_cond=1e-6,verbose=True,tol=1e-9):
    '''
    given that elements of M is covered, i.e., elements of M has all their 
    Pa(An^I(j)) contained in M, compute pairwise ancestor-descendant relationship
    from M to S
    
    uses Delta_Theta_func repeatedly, as every part of the algorithm
    
    returns |M| x |S| binary matrix: M_ij = 1 if M_i is ancestor of S_j
    '''
    M_des = np.zeros([len(M),len(S)])
    for j_idx in range(len(M)):
        for s_idx in range(len(S)):
            pair = [M[j_idx],S[s_idx]]
            Delta_Theta_sj, obj_hist_sj = Delta_Theta_func(S1[pair][:,pair],S2[pair][:,pair],\
                                                   lambda_l1,rho,n_max_iter,stop_cond,verbose)
    
            M_des[j_idx,s_idx] = (np.abs(Delta_Theta_sj[0,1]) > tol)
            
    return M_des

def prune_sample(Ml,Al,S1,S2,N,lambda_l1=0.1,rho=None,n_max_iter=500,stop_cond=1e-6,\
                                 tol=1e-9,verbose=True):
    '''
    starting from 1 element subsets, consider all subsets of Al to form Ml \cup A, until we distinguish everything
    
    returns I_l and J_l in a group A_l.
    '''
    Il = []
    Jl = []
    m = len(Ml)
    for size in range(1,len(Al)+1):
        Al_size_subsets = list(itertools.combinations(Al,size))
        Al_size_subsets = [list(Al_size_subsets[i]) for i in range(len(Al_size_subsets))]
        for A in Al_size_subsets:
            # if all elements of A is already added to Jl, no need to continue
            if set(A).issubset(Jl):
                continue
            AM = Ml+A
            Delta_Theta_Ml_A, obj_hist_Ml_A = Delta_Theta_func(S1[AM][:,AM],S2[AM][:,AM],\
                                                       lambda_l1,rho,n_max_iter,stop_cond,verbose)
                                                               
            Delta_Theta_Ml_A = Delta_Theta_Ml_A[m:,m:]                
            identified_j = (np.where(~np.diag(np.abs(Delta_Theta_Ml_A) > tol))[0])
            # if there is some new j, add it to Jl list
            for id_j in identified_j:
                if A[id_j] not in Jl:
                    N[A[id_j]] = AM
                    Jl.append(A[id_j])
                
    Il = [i for i in Al if i not in Jl]
    return Il, Jl, N

def post_parent_sample(j,i,M_lists,A_groups,A_i,S1,S2,lambda_l1=0.1,rho=None,\
                       n_max_iter=500,stop_cond=1e-6,tol=1e-9,verbose=False):
    '''
    returns True if node j is a parent of node i and False otherwise
    '''
    Ai = A_groups[A_i[i]]
    M = M_lists[i]
    if j not in M:
        if j not in Ai:
            return False
    # run similar to PRUNE function
    for size in range(1,len(Ai)+1):
        Ai_size_subsets = list(itertools.combinations(Ai,size))
        Ai_size_subsets = [list(Ai_size_subsets[k]) for k in range(len(Ai_size_subsets))]        
        for A in Ai_size_subsets:
            AM = M+A
            if i not in AM or j not in AM:
                continue
            Delta_Theta_M_A, obj_hist_M_A = Delta_Theta_func(S1[AM][:,AM],S2[AM][:,AM],\
                                                             lambda_l1,rho,n_max_iter,stop_cond,verbose)
            Delta_Theta_M_A = np.abs(Delta_Theta_M_A) > tol
            i_index = AM.index(i)
            j_index = AM.index(j)
            if Delta_Theta_M_A[j_index,i_index] == False:
                # which means j is not a parent of i
                return False
        
    return True


def algorithm_sample(S1,S2,lambda_l1=0.1,rho=None,single_threshold=0.05,pair_l1=0.1,\
                     pair_threshold=1e-3,parent_l1=0.1,n_max_iter=500,stop_cond=1e-6,\
                         tol=1e-9,return_parents=True,verbose=True,Delta_hat_parent_check=False):
    '''
    finite-sample implementation of our algorithm.

    Parameters
    ----------
    S1, S2 : 2d array
        Sample covariance matrices.
    lambda_l1 : float
        l1 norm parameter for Delta_Theta estimations over multiple nodes. The default is 0.1.
    rho : float
        penalty parameter for ADMM. No need change in most cases. The default is 1.0.
    n_max_iter : integer
        maximum number of iterations for ADMM. Does not need to be too large. The default is 500.
    stop_cond : float
        stopping condition for ADMM iterations. The default is 1e-6.
    verbose : Boolean
        The default is False.
    return_sym : Boolean
        Take symmetric (Delta + Delta.T)/2 in the end. The default is True.
        
    single_threshold : float
        to form J_0 set, use a threshold to decide if two marginal noises are different. The default is 0.05.
    pair_l1 : float
        to form J_0^K sets, l1 norm parameter for pairwise Delta_Theta estimation. The default is 0.1.
    pair_threshold : float
        use a small threshold to eliminate small non-zero elements for pairwise Delta_Theta. The default is 1e-3.
    parent_l1 : TYPE, float
        l1 norm parameter for parent estimations. Can be adjusted to move along the ROC curve. The default is 0.1.


    return_parents : Boolean
        If True, return parent estimates of intervention targets as well. The default is True.

    Delta_hat_parent_check : Boolean
        If True, use the initial Delta_Theta estimation for final check on parent estimations. The default is False.

    Returns
    -------
    I_hat:
        estimated intervention targets.
    I_hat_parents: 
        estimated parents of I_hat nodes.
    N_lists: 
        neutralizing ancestor set for a non-intervened node (mostly for debugging)
    A_groups:
        estimated A_groups defined in the paper.
    t_past: float
        computation time in terms of seconds.
    '''    

    t0 = time.time()
    p = len(S1)
    # neutralizing ancestor set for j nodes
    N = [[] for i in range(p)]
    # estimate Delta_Theta for all nodes at first
    Delta_hat, obj_hist = Delta_Theta_func(S1,S2,lambda_l1,rho,n_max_iter,stop_cond,verbose)     
    # apply some small threshold to Delta_hat, use the same one as J0_k forming later
    Delta_hat = np.abs(Delta_hat) > pair_threshold
    # set of all I and Pa(I) nodes
    #S_Delta = np.where(np.diag(Delta_hat))[0]   
    S_Delta = list(set(np.where(Delta_hat)[0]))
    
    diff_marginal_noise = diff_marginal_noise_sample_direct(S1, S2)
    # use single_threshold while considering marginal noises
    J0 = np.intersect1d(np.where(np.abs(diff_marginal_noise)<single_threshold)[0],S_Delta)
    for j in J0:
        N[j] = [] # they have no intervened ancestor    
    
    S_d = np.setdiff1d(S_Delta,J0)
    
    'if J0 set is empty, skip some steps as all S_d belongs to same group'
    if len(J0) == 0:
        A_groups = [[i for i in S_d]]
        L = len(A_groups)
        JA_groups = [[]] 
    else:
        'build J0_ancestor sets'
        # for now, keep using Delta_Theta estimate for that
        # use pair_l1 for penalty parameter, and use pair_threshold for final threshold
        J0_anc = build_descendants_sample(S1,S2,J0,S_d,pair_l1,rho,\
                                          n_max_iter,stop_cond,verbose,pair_threshold).T
        J0_anc_lists = [J0[np.where(J0_anc[i])[0]] for i in range(len(J0_anc))]
        J0_anc_size = [len(J0_anc_lists[i]) for i in range(len(J0_anc_lists))]
    
        size_argsort_order = np.argsort(J0_anc_size)
        A_mat = np.zeros([len(J0_anc_size),len(J0_anc_size)])
        
        for i in range(len(J0_anc_size)):
            for j in range(len(J0_anc_size)):
                A_mat[i,j] = np.array_equal(J0_anc_lists[size_argsort_order[i]],J0_anc_lists[size_argsort_order[j]])
            
        A_groups = list(find_maximal_cliques(nx.from_numpy_matrix(A_mat)))
        A_groups = [np.sort([size_argsort_order[i] for i in A]) for A in A_groups]
    
        L = len(A_groups)
        # get the corresponding J0_Al subsets
        JA_groups = [list(J0_anc_lists[A_groups[i][0]]) for i in range(L)]
        # map to the correct indices in graph
        A_groups = [list(S_d[A_groups[i]]) for i in range(len(A_groups))]

    Al_sizes = [len(A) for A in A_groups]
    print('Al_sizes:',Al_sizes)
    # A_group memberships
    A_i = [[] for i in range(p)]
    for l in range(L):
        for a in A_groups[l]:
            A_i[a] = l

    B_mat = np.zeros([L,L])
    for i in range(L):
        for j in range(i):
            B_mat[i,j] = set(JA_groups[j]).issubset(JA_groups[i])
        
    M_lists = [[] for i in range(p)]
    N_lists = [[] for i in range(L)]
    J_all = []
    I_all = []

    for l in range(L):
        Al = A_groups[l]
        J0_Al = JA_groups[l]
        Bl = list(np.where(B_mat[l])[0])
        Ml = []; Ml.extend(J0_Al)
        for b in Bl:
            Ml.extend(J_all[b])
            Ml.extend(I_all[b])
            
        Ml = sorted(Ml)
        for a in Al:
            M_lists[a] = Ml
        Il, Jl, N = prune_sample(Ml,Al,S1,S2,N,lambda_l1,rho,n_max_iter,stop_cond,tol,verbose)
        N_lists[l] = N.copy()
        I_all.append(Il)
        J_all.append(Jl)
    
    I_hat = sorted(flatten_list(I_all))
    #J_hat = np.setdiff1d(S_Delta, I_hat)                
    
    if return_parents == True: 
        # now find j to i parent-child relationships
        I_hat_parents = [[] for i in range(len(I_hat))]
        
        for j in S_Delta:
            for i_idx in range(len(I_hat)):
                if j == I_hat[i_idx]:
                    continue
                if post_parent_sample(j,I_hat[i_idx],M_lists,A_groups,A_i,S1,S2,\
                                      parent_l1,rho,n_max_iter,stop_cond,tol,verbose):
                    # ok, now also check with Delta_hat
                    if Delta_hat_parent_check == True:
                        if Delta_hat[j,I_hat[i_idx]] == True:
                            I_hat_parents[i_idx].append(j)
                    else:
                        I_hat_parents[i_idx].append(j)
                        
                    
        t_past = time.time() - t0
        #print(JA_groups, A_groups)
        return I_hat, I_hat_parents, N_lists, A_groups, t_past
    
    else:
        t_past = time.time() - t0
        return I_hat, N_lists, A_groups, t_past
 
def algorithm_sample_multiple(setting_list,lambda_l1=0.1,single_threshold=0.05,pair_l1=0.05,pair_threshold=0.005,parent_l1=0.05,rho=1.0):
    '''
    when there exists multiple interventional settings, run algorithm_sample for each setting 
    and combine their results.
    
    Returns
    -------
    est_cpdag:
        estimated CPDAG by orienting possible edges with learned intervention parents
    est_skeleton:
        estimated skeleton 
    I_hat_all:
        estimated intervention targets for each setting
    I_hat_parents_all:
        estimated parents of I_hat nodes. (including i to i's which some can be non-oriented)
    Ij_hat_parents_all:
        estimated non-intervened parents of I_hat nodes.
    N_lists_all: 
        neutralizing ancestor set for a non-intervened node (mostly for debugging)
    A_groups_all:
        estimated A_groups defined in the paper.
    time_all: float
        computation time in terms of seconds.                            
    '''
    I_hat_all = {}
    I_hat_parents_all = {}
    Ij_hat_parents_all = {}
    N_lists_all = {}
    A_groups_all = {}
    time_all = {}
    
    S_obs = setting_list['setting_0']['S']
    nnodes = S_obs.shape[0]
    for idx_setting in range(1,len(setting_list)):
        I_hat, I_hat_parents, N_lists, A_groups, t_past = algorithm_sample(S_obs,setting_list['setting_%d'%idx_setting]['S'],\
                                                 lambda_l1,rho,single_threshold,pair_l1,pair_threshold,parent_l1,\
                                                 return_parents=True,verbose=False,Delta_hat_parent_check=False)
        Ij_hat_parents = [list(np.setdiff1d(I_hat_parents[i], I_hat)) for i in range(len(I_hat))]
        I_hat_all['setting_%d'%idx_setting] = I_hat
        I_hat_parents_all['setting_%d'%idx_setting] = I_hat_parents
        Ij_hat_parents_all['setting_%d'%idx_setting] = Ij_hat_parents
        N_lists_all['setting_%d'%idx_setting] = N_lists
        A_groups_all['setting_%d'%idx_setting] = A_groups
        time_all['setting_%d'%idx_setting] = t_past

    # now combine the learned information for final causal structure
    est_cpdag = np.zeros((nnodes,nnodes))
    for idx_setting in range(1,len(setting_list)):
        edges_current = list(zip(I_hat_all['setting_%d'%idx_setting],\
                                I_hat_parents_all['setting_%d'%idx_setting]))
        for edge in edges_current:
            est_cpdag[edge[1],edge[0]] = 1         
            
    est_skeleton = est_cpdag + est_cpdag.T
    est_skeleton[np.where(est_skeleton)] = 1

    return est_cpdag, est_skeleton, I_hat_all, I_hat_parents_all, Ij_hat_parents_all, N_lists_all, A_groups_all, time_all

 
def run_ours_real(S_obs,S_int,lambda_l1=0.1,single_threshold=0.05,pair_l1=0.05,pair_threshold=0.005,parent_l1=0.05,rho=1.0):
    '''
    Run our algorithm on real data with multiple itnerventional settings.
    
    Parameters
    ----------
    S_obs : 2d array
        observation sample covariance matrix.
    S_int : 
        sample covariance matrices for interventional settings.
    lambda_l1 : float
        l1 norm parameter for Delta_Theta estimations over multiple nodes. The default is 0.1.
    rho : float
        penalty parameter for ADMM. No need change in most cases. The default is 1.0.
    single_threshold : float
        to form J_0 set, use a threshold to decide if two marginal noises are different. The default is 0.05.
    pair_l1 : float
        to form J_0^K sets, l1 norm parameter for pairwise Delta_Theta estimation. The default is 0.05.
    pair_threshold : float
        use a small threshold to eliminate small non-zero elements for pairwise Delta_Theta. The default is 5e-3.
    parent_l1 : TYPE, float
        l1 norm parameter for parent estimations. Can be adjusted to move along the ROC curve. The default is 0.05.

    Returns
    -------
    est_cpdag:
        estimated CPDAG by orienting possible edges with learned intervention parents
    est_skeleton:
        estimated skeleton 
    I_hat_all:
        estimated intervention targets for each setting
    I_hat_parents_all:
        estimated parents of I_hat nodes. (including i to i's which some can be non-oriented)
    Ij_hat_parents_all:
        estimated non-intervened parents of I_hat nodes.
    N_lists_all: 
        neutralizing ancestor set for a non-intervened node (mostly for debugging)
    A_groups_all:
        estimated A_groups defined in the paper.
    time_all: float
        computation time in terms of seconds.          
    '''
    I_hat_all = {}
    I_hat_parents_all = {}
    Ij_hat_parents_all = {}
    N_lists_all = {}
    A_groups_all = {}
    time_all = {}
    nnodes = S_obs.shape[0]
    for idx_setting in range(len(S_int)):
        I_hat, I_hat_parents, t_past, N_lists, A_groups = algorithm_sample(S_obs,S_int['setting_%d'%idx_setting],\
                                                 lambda_l1,rho,single_threshold,pair_l1,pair_threshold,parent_l1,\
                                                 return_parents=True,verbose=False,Delta_hat_parent_check=False)
        Ij_hat_parents = [list(np.setdiff1d(I_hat_parents[i], I_hat)) for i in range(len(I_hat))]
        I_hat_all['setting_%d'%idx_setting] = I_hat
        I_hat_parents_all['setting_%d'%idx_setting] = I_hat_parents
        Ij_hat_parents_all['setting_%d'%idx_setting] = Ij_hat_parents
        N_lists_all['setting_%d'%idx_setting] = N_lists
        A_groups_all['setting_%d'%idx_setting] = A_groups
        time_all['setting_%d'%idx_setting] = t_past

    # now combine the learned information for final causal structure
    est_cpdag = np.zeros((nnodes,nnodes))
    for idx_setting in range(len(S_int)):
        edges_current = list(zip(I_hat_all['setting_%d'%idx_setting],\
                                I_hat_parents_all['setting_%d'%idx_setting]))
        for edge in edges_current:
            est_cpdag[edge[1],edge[0]] = 1         
            
    est_skeleton = est_cpdag + est_cpdag.T
    est_skeleton[np.where(est_skeleton)] = 1

    return est_cpdag, est_skeleton, I_hat_all, I_hat_parents_all, Ij_hat_parents_all, N_lists_all, A_groups_all, time_all
    

