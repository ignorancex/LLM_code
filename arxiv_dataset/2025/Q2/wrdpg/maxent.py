'''
Created on Apr 12, 2025

@author: bernardo
'''
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.stats import rv_continuous, bernoulli
from scipy.special import logsumexp, softmax
from joblib import Parallel, delayed
import warnings


class maximum_entropy_rv(rv_continuous):
    def __init__(self,lambdas, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lambdas = lambdas
        self.stabilize_exp()
        self.set_bound()
        
    def set_bound(self):
        x = np.linspace(self.a,self.b,num=100)
        y = self.pdf(x)
        self.M = np.amax(y)*1.1 # Small safety factor
    
    def stabilize_exp(self, n_leggauss=200):
        # Gauss-Legendre quadrature points and weights
        nodes, weights = leggauss(n_leggauss)
        x_nodes = 0.5 * (self.b - self.a) * nodes + 0.5 * (self.b + self.a)  # Map to [a, b]

        # Evaluate the exponent at nodes
        exponents = eval_exponent(x_nodes, self.lambdas)
        self.max_exp = np.max(exponents)
        stabilized = exponents - self.max_exp
        f_vals = np.exp(stabilized)

        # Compute normalizing constant
        self.Z = 0.5 * (self.b - self.a) * np.dot(weights, f_vals) * np.exp(self.max_exp)     
    
    def _pdf(self, x):
        exponents = eval_exponent(x, self.lambdas)
        stabilized = exponents - self.max_exp
        return np.exp(stabilized) / self.Z
    
    def _rvs(self, size=1, random_state=None, num_points=1000):
        
        """
        Draw samples from the max-entropy PDF using inverse transform sampling.
    
        Args:
            size (int or np.ndarray): Size of the sampled points.
            num_points (int): Number of points to discretize the domain for CDF computation.
    
        Returns:
            np.ndarray: Samples drawn from g_alpha(x).
        """
    
        n_rvs = int(np.sum(size))
        x_grid = np.linspace(self.a, self.b, num_points)
        pdf_vals = self.pdf(x_grid)
    
        # Compute the CDF
        dx = x_grid[1] - x_grid[0]
        cdf_vals = np.cumsum(pdf_vals) * dx
        cdf_vals /= cdf_vals[-1]  # Normalize CDF to [0, 1]
    
        # Inverse transform sampling
        uniform_samples = random_state.uniform(size=n_rvs)
        
        sample_indices = np.searchsorted(cdf_vals, uniform_samples, side='right')
        samples = x_grid[np.clip(sample_indices, 0, n_rvs - 1)]
        
        return samples.reshape(size)
    
    
    def rejection_sampling(self, size=1, random_state=None):
        """Rejection sampling from the exponential family pdf."""
        
        n_rvs = int(np.sum(size))
        # Proposal: Uniform on Omega
        x_vals = random_state.uniform(self.a, self.b, size=5*n_rvs)
        y_vals = random_state.uniform(size=5*n_rvs)
    
        pdf_vals = self.pdf(x_vals)
        M = np.amax(pdf_vals)*1.1 # Small safety factor
        accepted = x_vals[y_vals * M <= pdf_vals]

        return accepted[:n_rvs].reshape(size)
        
    
class mixed_rv(rv_continuous):
    def __init__(self, continuous_rv, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.continuos_rv = continuous_rv
        # Edge existance probability
        self.p = p
    
    def _cdf(self, x):
        return (1-self.p)*bernoulli.cdf(0,self.p)+ self.p*self.continuos_rv.cdf(x)
    
    def _rvs(self, size=1, random_state=None):
        edges = random_state.binomial(n=1,p=self.p,size=size)
        variates = edges*self.continuos_rv.rvs(size=size)
        return variates
    
def eval_exponent(x, lambdas):
    return np.polynomial.polynomial.polyval(x, -lambdas)


def compute_log_moment(lambdas, order, x_vals=None, w_vals = None, Omega=None, n_leggauss=200):
    """
    Compute the r-th moment of the pdf defined by the exponential family,
    using Gauss-Legendre quadrature and log-domain integration.
    
    Parameters:
    - lambdas: array of optimal Lagrange multipliers
    - order: moment order (integer)
    - x_vals: array of nodes for Gauss-Legendre quadrature
    - x_vals: array of weights for Gauss-Legendre quadrature
    - Omega: list or set with integration bounds (support of the density)
    - n_leggauss: number of quadrature nodes
    
    Returns:
    - moment_r: estimated r-th moment
    """
    
    if all(arg is None for arg in (x_vals, w_vals, Omega)):
        raise ValueError("You should specify Omega or x_vals,w_vals")
    
    if Omega is not None:
        # Get Gauss-Legendre nodes and weights on [-1, 1]
        x_leg, w_leg = leggauss(n_leggauss)
        
        a, b = Omega
        
        # Rescale to [a, b]
        x_vals = 0.5 * (b - a) * x_leg + 0.5 * (b + a)
        w_vals = 0.5 * (b - a) * w_leg
    elif x_vals is None or w_vals is None:
        raise ValueError("You should specify both x_vals and w_vals")
            
    
    # Evaluate log of unnormalized density
    log_unnormalized = eval_exponent(x_vals, lambdas)
    max_log = np.max(log_unnormalized)  # for stabilization
    log_weights = log_unnormalized - max_log
    
    # Compute numerator (log-sum-exp of x^r * g(x))
    log_integrand_numerator = log_weights + np.log(x_vals**order + 1e-300)
    log_num = max_log + np.log(np.sum(np.exp(log_integrand_numerator) * w_vals))
    
    # Compute normalizing constant Z in log-domain
    logZ = max_log + np.log(np.sum(np.exp(log_weights) * w_vals))
    
    # Return the moment
    return np.exp(log_num - logZ)

def compute_moments(lambdas, K, x_vals=None, w_vals = None, Omega=None, n_leggauss=200):
    """Compute moments up to order K using Gauss–Legendre quadrature."""
    return np.array([compute_log_moment(lambdas,k,x_vals,w_vals,Omega,n_leggauss) for k in range(K+1)])
    
def log_integral_function(lambdas, x_vals, w_vals):
    """Compute log-integral function using Gauss–Legendre quadrature."""

    exponents = eval_exponent(x_vals, lambdas)
    
    log_integral = logsumexp(exponents,b=w_vals)
    
    return log_integral

def dual_objective_log_domain(lambdas, target_moments, x_vals, w_vals, reg_strength=0.0):
    log_integral = log_integral_function(lambdas, x_vals, w_vals)
    return np.dot(lambdas, target_moments) + log_integral - np.log(target_moments[0])+ reg_strength*np.square(np.linalg.norm(lambdas))

def dual_function_and_grad(lambdas, target_moments, x_vals, w_vals, reg_strength=0.0):
    est_moments = compute_moments(lambdas, len(lambdas)-1, x_vals=x_vals, w_vals=w_vals)
    grad = target_moments - est_moments + 2*reg_strength*lambdas
    dual_function = dual_objective_log_domain(lambdas, target_moments, x_vals, w_vals) + reg_strength*np.square(np.linalg.norm(lambdas))

    return dual_function, grad

def solve_dual_continuous_log(moment_vector, Omega, n_leggauss=200, reg_strength=0.0, verbose=False, lambdas0=None):
    """Solve the dual optimization problem in log-domain using BFGS."""
    
    if lambdas0 is None:
        lambdas0 = np.zeros_like(moment_vector)
        
    # Gauss-Legendre quadrature points and weights over [-1, 1]
    nodes, weights = leggauss(n_leggauss)

    # Change of variables to map to [Omega[0], Omega[1]]
    a, b = Omega
    x_vals = 0.5 * (nodes + 1) * (b - a) + a
    w_vals = 0.5 * (b - a) * weights
    
    result = minimize(dual_function_and_grad,
                      lambdas0,
                      args=(moment_vector,x_vals, w_vals, reg_strength),
                      jac = True,
                      method='BFGS',
                      options={'gtol': 1e-4, 'xrtol':1e-8, 'disp':verbose})
    
    # Correct lambda_0 since solution uses log-transform
    log_int = log_integral_function(result.x, x_vals, w_vals)
    result.x[0] = result.x[0]+log_int
    
    if (not result.success) and verbose:
        warnings.warn(f'Dual function minimization did not converge - Message: {result.message}', RuntimeWarning)
    return result.x

def solve_batch_log_continuous(moment_matrix, Omegas, n_jobs=-1):
    """
    Solve the dual problem for a batch of moment vectors in parallel.
    Each row in `moment_matrix` corresponds to an (i, j) pair's moment vector.
    """
    return Parallel(n_jobs=n_jobs)(
        delayed(solve_dual_continuous_log)(moment_vector, Omega)
        for moment_vector, Omega in zip(moment_matrix,Omegas)
    )

def compute_moment_discrete(lambdas, symbols, order):
    exponent = eval_exponent(symbols, lambdas)
    symbols_pow = np.power(symbols,order)
    logsum = logsumexp(exponent,b=symbols_pow)
    return np.exp(logsum)

def compute_moments_discrete(lambdas,symbols, K):
    return np.array([compute_moment_discrete(lambdas, symbols, k) for k in range(K+1)])
    
def dual_function_discrete(lambdas, target_moments, symbols, reg_strength=0.0):
    """
    Log-domain dual objective function for discrete max-entropy.
    """
    exponent = eval_exponent(symbols, lambdas)
    logsum = logsumexp(exponent)
    return np.dot(lambdas, target_moments) + logsum +  reg_strength*np.square(np.linalg.norm(lambdas))

def dual_function_and_grad_discrete(lambdas, target_moments, symbols, vander_mat, reg_strength=0.0):
    """
    Log-domain dual objective function and gradient for discrete max-entropy.
    
    Args:
        lambdas (np.ndarray): Lagrange multipliers for the discrete max-entropy problem, size K+1
        target_moments (np.ndarray): moments to fit
        symbols (np.ndarray): support of discrete distribution (size R+1)
        vander_mat (np.ndarray): vandermonde matrix of symbols, size (R+1,K+1). This is used to compute the gradient of the log-function

    Returns:
        np.ndarray: Samples drawn from g_alpha(x).
    """
    exponent = eval_exponent(symbols, lambdas)
    logsum = logsumexp(exponent)
    dual_fun = np.dot(lambdas, target_moments) + logsum - np.log(target_moments[0]) + reg_strength*np.square(np.linalg.norm(lambdas))
    grad = target_moments-softmax(exponent)@vander_mat + 2*reg_strength*lambdas
    return dual_fun, grad
    

def solve_dual_discrete(moment_vector, symbols, reg_strength=0.0, verbose=False):
    """Minimize the dual function to find the optimal lambda vector in the discrete case."""
    lambdas0 = np.zeros_like(moment_vector)
    lambdas0[0] = np.log(np.sqrt(2*np.pi))
    vander_mat = np.vander(symbols,N=len(moment_vector), increasing=True)
    result = minimize(dual_function_and_grad_discrete,
                      lambdas0,
                      args=(moment_vector, symbols, vander_mat, reg_strength), 
                      jac=True,
                      method='BFGS',
                      options={'gtol': 1e-4, 'xrtol':1e-8, 'disp':verbose})
    if (not result.success) and verbose:
        warnings.warn(f'Dual function minimization did not converge - Message: {result.message}', RuntimeWarning)
    return result.x
    
def solve_batch_discrete(moment_matrix, symbols_list, n_jobs=-1):
    """
    Solve the dual problem for a batch of moment vectors in parallel.
    Each row in `moment_matrix` corresponds to an (i, j) pair's moment vector.
    """
    return Parallel(n_jobs=n_jobs)(
        delayed(solve_dual_discrete)(moment_vector, symbols)
        for moment_vector, symbols in zip(moment_matrix,symbols_list)
    )
    
    
def lambdas_to_probabilities(lambdas,v_support):
    # Compute exponent terms in a stable way
    exponents = eval_exponent(v_support, lambdas)
    probs = softmax(exponents)
    return probs