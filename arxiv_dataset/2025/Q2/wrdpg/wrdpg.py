'''
Created on Apr 12, 2024

@author: bernardo
'''

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from graspologic.utils import symmetrize
import warnings
import scipy.linalg as la
from scipy.stats import norm
from scipy.special import comb
import maxent
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def normalize_support(v_support):
    """
    Center and scale the support to have mean 0 and standard deviation 1.
    Returns the normalized support, mean, and std.
    """
    mu = np.mean(v_support)
    sigma = np.std(v_support)
    normalized = (v_support - mu) / sigma
    return normalized, mu, sigma

def adjust_moments(moment_vector, mu, sigma):
    """
    Adjusts the moment vector to account for support normalization.
    Given original moments of x, returns moments of (x - mu)/sigma.
    """
    K = len(moment_vector) - 1
    adjusted = np.zeros_like(moment_vector)
    for k in range(K + 1):
        adjusted[k] = sum(
            comb(k, j) * (-mu) ** (k - j) * moment_vector[j] for j in range(k + 1)
        ) / sigma**k
    return adjusted

def project_to_a11(mat):
    proj_mat = mat.copy()
    proj_mat[0,0] = 1
    return proj_mat

def make_psd(symmetric_mat, tol=1e-6):
    eigvals, eigvecs = np.linalg.eigh(symmetric_mat)
    eigvals[(eigvals < 0) & (np.abs(eigvals) > tol)] = 0
    psd_mat = eigvecs@np.diag(eigvals)@eigvecs.T
    return psd_mat

def make_hankel(symmetric_mat):
    N = symmetric_mat.shape[0]
    proj_mat = np.zeros_like(symmetric_mat)
    for k in np.arange(-N+1,N):
        antidiag = np.diag(np.fliplr(symmetric_mat),k)
        antidiag_mean = np.mean(antidiag)*np.ones_like(antidiag)
        proj_mat += np.fliplr(np.diag(antidiag_mean,k))
    return proj_mat

def dykstra(symmetric_mat, max_iter=100, tol=1e-6):
    projected_mat = symmetric_mat.copy()
    last_jump = np.inf
    normal_a = 0 
    normal_psd = 0 
    i = 0
    while (i < max_iter) and (last_jump > tol):
        last_mat = projected_mat
        # Add normal component from last projection to A^{11} (matrices such that a_{11} = 1)
        new_mat = last_mat + normal_a
        # Project to A^{11}
        projected_mat = project_to_a11(new_mat)
        # Update normal vector from projection
        normal_a = new_mat-projected_mat
        # Add normal component from last projection to S_n^+ (psd symmetric matrices)
        new_mat = projected_mat + normal_psd
        # Project to S_n^+
        projected_mat = make_psd(new_mat)
        # Update normal vector from projection
        normal_psd = new_mat-projected_mat
        # Project to H_n (Hankel matrices subspace)
        projected_mat = make_hankel(projected_mat)
        # Since H_n is a subspace, it is not necessary to add a normal component
        
        # Update counter and jump
        last_jump = np.linalg.norm(last_mat-projected_mat)
        i+=1
        
    return projected_mat

def non_psd_idxs(symmetric_moments_mat, tol=1e-10):
    eigvals = np.linalg.eigvalsh(symmetric_moments_mat)
    return np.unique(np.nonzero( (eigvals < 0) & (np.abs(eigvals) > tol))[0])


def check_psd_moments_mat(edges_moments, directed=False):
    N = edges_moments.shape[0]
    if not directed:
        # can cut down on sampling by ~half
        triu_inds = np.triu_indices(N)
        moments = edges_moments[triu_inds]
        # Check that there are an odd number of moments
        p = moments.shape[-1]
        if (p%2 != 1):
            p = p-1
        
        # Check that moments form an admisible moments sequence
        symmetric_moments_mat = np.empty((moments.shape[0],(p+1)//2,(p+1)//2))
        for idx,mom in enumerate(moments):
            symmetric_moments_mat[idx,...] = la.hankel(mom[:(p+1)//2],mom[p//2:p])
        
        not_psd_idxs = non_psd_idxs(symmetric_moments_mat)
        non_psd_mats = symmetric_moments_mat[not_psd_idxs]
        
        for idx,mat in enumerate(non_psd_mats):
            psd_mat = dykstra(mat)
            symmetric_moments_mat[not_psd_idxs[idx]] = psd_mat
        
        # recover moments matrix
        psd_moments = np.concatenate((symmetric_moments_mat[:,0,:],symmetric_moments_mat[:,1:,-1]),axis=1)
        if (moments.shape[-1]%2 !=1):
            psd_moments = np.concatenate((psd_moments,moments[:,-1,None]),axis=1)
        psd_moments_mat = np.empty_like(edges_moments)
        psd_moments_mat[triu_inds] = psd_moments
        # Symmetrize
        tril_inds = np.tril_indices(N)
        psd_moments_mat[tril_inds] = psd_moments
            
    return psd_moments_mat

def discrete_distribution(values, probs, size=1):
    return np.random.choice(values,size=size,p=probs)

def estimate_discrete_distribution(symbols,moments,reg_strength=0.0, verbose=False):
    moments = np.asarray(moments)
    symbols = np.asarray(symbols)
    
    M = moments.shape[0]
    N = symbols.shape[0]
    
    vander_mat = np.vander(symbols, N=N, increasing=True)
    if M > N:
        if verbose:
            warnings.warn('You have specified more moments than symbols. Returning least-squares solution, which may not be optimal')
        p_hat,_,_,_ = np.linalg.lstsq(vander_mat.T,moments,rcond=-1)
    elif M == N:
                
        # Solve Vandermonde system using chebyshev polynomials
        cheb_mat = np.empty((N,N))
        cheb_mu =  np.empty_like(moments)
        domain = [np.amin(symbols), np.amax(symbols)]
        for i in range(N):
            cheb_poly_i = Chebyshev.basis(i,domain=domain)
            cheb_mat[i,:] = cheb_poly_i(symbols)
            cheb_poly_coefs = cheb_poly_i.convert(kind=np.polynomial.Polynomial).coef
            cheb_mu[i] = np.dot(cheb_poly_coefs,moments[:i+1])
        
        p_hat = np.linalg.solve(cheb_mat,cheb_mu)
         
    else:
        
        # Normalize support to prevent overflow
        norm_symbols, mu, sigma = normalize_support(symbols)
        norm_moments = adjust_moments(moments, mu, sigma)
        
        # Find maximum entropy by solving dual problem
        lambdas = maxent.solve_dual_discrete(norm_moments, norm_symbols,
                                             reg_strength=reg_strength,
                                             verbose=verbose)
        p_hat = maxent.lambdas_to_probabilities(lambdas, norm_symbols)
 
    # Check returned probabilities
    if np.isnan(np.sum(p_hat)):
        raise RuntimeError(f"There is a NaN in p_hat. Offending moments: {moments}")
    
    if np.sum(p_hat) == 0:
        p_hat = np.ones_like(p_hat)/len(p_hat)
    
    if np.sum(p_hat) != 1:
        p_hat = p_hat/np.sum(p_hat)
           
    return p_hat

def estimate_maxent_distribution(moments, Omega ,reg_strength=0.0,verbose=False):
    # Normalize support to prevent overflow
    norm_omega, mu, sigma = normalize_support(Omega)
    norm_moments = adjust_moments(moments, mu, sigma)
    lambdas_max_ent = maxent.solve_dual_continuous_log(norm_moments,norm_omega,reg_strength=reg_strength,verbose=verbose)
    
    return maxent.maximum_entropy_rv(lambdas=lambdas_max_ent, momtype=1, a=Omega[0],b=Omega[1])

def sample_discrete_edge(symbols, probs):
    return np.random.choice(symbols, p=probs)

def sample_continuous_edge(edge_rv):
    return edge_rv.rvs(1)

def moments_from_latent(X_seq, Q_seq=None, directed=False):
    
    if directed:
        X_l = X_seq[0,:,:,:]
        X_r = X_seq[1,:,:,:]
        (X_l,X_r) = X_seq
    else:
        X_l = X_seq
        X_r = X_seq
    
    #edges_moments = np.einsum('ijn,jkn->ikn', X_seq, np.transpose(X_seq,axes=(1,0,2)))   
    #this is faster
    Xlt = np.transpose(X_l, axes=(2,0,1))
    Xrt = np.transpose(X_r, axes=(2,1,0))
    if Q_seq is None:
        edges_moments_t = np.matmul(Xlt, Xrt)
    else:
        Qt = np.transpose(Q_seq, axes=(2,0,1))
        edges_moments_t = np.matmul(Xlt, np.matmul(Qt, Xrt))
        
    edges_moments = np.transpose(edges_moments_t,axes=(1,2,0))
    
    return edges_moments

def estimate_discrete_distribution_with_p_edge(val, moment_seq, P_edges, idx, edges_idxs, reg_strength=0.0, verbose=False):
    if idx in edges_idxs:
        # Estimate discrete distribution for nonzero values
        moment_seq[1:]/=P_edges[idx]
        p_hat = estimate_discrete_distribution(val, moment_seq,reg_strength=reg_strength,verbose=verbose)
    else:
        p_hat = np.zeros_like(val)

    return np.insert(p_hat*P_edges[idx],0,1-P_edges[idx])

def estimate_maxent_distribution_with_p_edge(moment_seq, Omega, idx, P_edges, edges_idxs, reg_strength=0.0, verbose=False):
    if idx in edges_idxs:
        # Estimate discrete distribution for nonzero values
        moment_seq[1:]/=P_edges[idx]
        edge_continuous_rv = estimate_maxent_distribution(moment_seq, Omega, reg_strength=reg_strength, verbose=verbose)
        edge_rv = maxent.mixed_rv(edge_continuous_rv,P_edges[idx])
    else:
        edge_rv = maxent.mixed_rv(norm(),P_edges[idx])

    return edge_rv
    
def estimate_edges_probabilities(edges_moments, values, directed=False, P_edges=None, parallel=True,reg_strength=0.0, verbose=False):
    
    # P_edge = matrix with edge's existance probability
    
    N = edges_moments.shape[0]
    
    if not directed:
        # can cut down on sampling by ~half
        triu_inds = np.triu_indices(N)
        moments = edges_moments[triu_inds]
        vals = values[triu_inds]
        if P_edges is not None:
            P_edges = P_edges[triu_inds]
    
    else:    
        moments = edges_moments.reshape((N**2,-1))
        vals = values.reshape((N**2,-1))
        if P_edges is not None:
            P_edges = P_edges.reshape((N**2,-1))
    
    if P_edges is None:

        if parallel:
            wrapped_func = partial(estimate_discrete_distribution, reg_strength=reg_strength, verbose=verbose)
            with Pool() as pool:
                p_hats = pool.starmap(wrapped_func, zip(vals, moments))
        else:
            p_hats = []
            for val,moment_seq in tqdm(zip(vals, moments), desc="Estimating discrete distribution", total = len(vals)):
                p_hats.append(estimate_discrete_distribution(val, moment_seq, reg_strength=reg_strength, verbose=verbose))
    else:
        # Estimate distribution only for edges with P_edge=1-p_0 \neq 0
        edges_idxs = np.nonzero(P_edges)[0]
        if parallel:
            # Add value 0 to vals
            wrapped_func = partial(estimate_discrete_distribution_with_p_edge, P_edges=P_edges, edges_idxs=edges_idxs, reg_strength=reg_strength, verbose=verbose)
            with Pool() as pool:
                p_hats = pool.starmap(wrapped_func,zip(vals, moments,np.arange(len(moments))))
        else:
            p_hats = []
            for idx, (val, moment_seq) in tqdm(enumerate(zip(vals,moments)),
                                               desc="Estimating discrete distribution",
                                               total = len(vals)):
                if idx in edges_idxs:
                    # Estimate discrete distribution for nonzero values
                    moment_seq[1:]/=P_edges[idx]
                    p_hat = estimate_discrete_distribution(val, moment_seq, reg_strength=reg_strength, verbose=verbose)
                else:
                    p_hat = np.zeros_like(val)
                # Add p_0
                p_hats.append(np.insert(p_hat*P_edges[idx],0,1-P_edges[idx]))
        
        vals = np.hstack((np.zeros((vals.shape[0],1)),vals))
    
    return p_hats, vals

def estimate_edges_distributions(edges_moments, supports, directed=False, P_edges=None, parallel=True,reg_strength=0.0, verbose=False):
    
    N = edges_moments.shape[0]
    
    if not directed:
        # can cut down on sampling by ~half
        triu_inds = np.triu_indices(N)
        moments = edges_moments[triu_inds]
        Omegas = supports[triu_inds]
        if P_edges is not None:
            P_edges = P_edges[triu_inds]
    
    else:    
        moments = edges_moments.reshape((N**2,-1))
        Omegas = supports.reshape((N**2,-1))
        if P_edges is not None:
            P_edges = P_edges.reshape((N**2,-1))
            
    if P_edges is None:

        if parallel:
            wrapped_func = partial(estimate_maxent_distribution, reg_strength=reg_strength, verbose=verbose)
            with Pool() as pool:
                edges_rvs = pool.starmap(wrapped_func, zip(moments, Omegas))
        else:
            edges_rvs = []
            for moment_seq, Omega in tqdm(zip(moments, Omegas),
                                          desc="Estimating edges distributions",
                                          total = len(Omegas)):
                edges_rvs.append(estimate_maxent_distribution(moment_seq,Omega,reg_strength,verbose))
    else:
        # Estimate distribution only for edges with P_edge=1-p_0 \neq 0
        edges_idxs = np.nonzero(P_edges)[0]
        if parallel:
            wrapped_func = partial(estimate_maxent_distribution_with_p_edge,
                                   P_edges=P_edges,
                                   edges_idxs=edges_idxs,
                                   reg_strength=reg_strength,
                                   verbose=verbose)
            with Pool() as pool:
                edges_rvs = pool.starmap(wrapped_func,
                                         zip(moments, Omegas, np.arange(len(moments)))
                                         )
        else:
            edges_rvs = []
            for idx, (moment_seq, Omega) in tqdm(enumerate(zip(moments,Omegas)),
                                                 desc="Estimating edges distributions",
                                                 total = len(Omegas)):
                if idx in edges_idxs:
                    # Estimate continuous distribution for nonzero values
                    moment_seq[1:]/=P_edges[idx]
                    edge_continuous_rv = estimate_maxent_distribution(moment_seq, Omega,reg_strength,verbose)
                    edge_rv = maxent.mixed_rv(edge_continuous_rv,P_edges[idx])
                else:
                    edge_rv = maxent.mixed_rv(norm(),P_edges[idx])
                    
                edges_rvs.append(edge_rv)
        
        Omegas  = np.hstack((np.zeros((Omegas.shape[0],1)),Omegas))
    
    return np.array(edges_rvs), Omegas
    

def sample_continuous_edges(edges_rvs, directed=False, loops=False, parallel=False, n_jobs=None):
    
    if parallel:
        n_jobs = n_jobs or cpu_count()
        with Pool(n_jobs) as pool:
            samples = pool.map(sample_continuous_edge, edges_rvs)
    else:
        samples = np.array([sample_continuous_edge(edge_rv) for edge_rv in edges_rvs])
    
    if not directed:
        N = int((-1+np.sqrt(1+8*edges_rvs.shape[0]))/2)
        triu_inds = np.triu_indices(N)
        
        A = np.zeros((N,N))
        A[triu_inds] = samples.squeeze()
        A = symmetrize(A, method="triu")
        
    else:
        N = int(np.sqrt(edges_rvs.shape[0]))
        A = samples.reshape((N,N))
        
        
    if loops:
        return A
    else:
        return A - np.diag(np.diag(A))


def sample_edges(p_hats, vals, directed=False, loops=False, parallel=False, n_jobs=None):
    # See graspologic.simulations.sample_edges
    
    if parallel:
        n_jobs = n_jobs or cpu_count()
        with Pool(n_jobs) as pool:
            samples = pool.starmap(sample_discrete_edge, zip(vals,p_hats))
    else:
        samples = np.array([sample_discrete_edge(val, p_hat) for val,p_hat in zip(vals,p_hats)])
    
    if not directed:
        N = int((-1+np.sqrt(1+8*vals.shape[0]))/2)
        triu_inds = np.triu_indices(N)
        
        A = np.zeros((N,N))
        A[triu_inds] = samples
        A = symmetrize(A, method="triu")
        
    else:
        N = int(np.sqrt(vals.shape[0]))
        A = samples.reshape((N,N))
        
        
    if loops:
        return A
    else:
        return A - np.diag(np.diag(A))

def discrete_wrdpg(X_seq, values, directed = False, loops = False):
    edges_moments = moments_from_latent(X_seq)
    p_hats, vals = estimate_edges_probabilities(edges_moments, values=values, directed=directed)
    A = sample_edges(p_hats, vals=vals, directed=directed, loops=loops)
        
    return A
