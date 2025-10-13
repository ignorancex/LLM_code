import numpy as np
from scipy.optimize import minimize
from scipy import sparse
from otlm.models import OTLM
from scipy.special import lambertw


astr = lambda x: ' '.join([f'{x_:2.6e}' for x_ in x])


def get_toy_example():

    def gauss(a, mu, sig): 
        return a*np.exp(-(x-mu)**2/sig**2)

    def gauss_normed(a, mu, sig):
        y = gauss(1, mu, sig)
        y = a*y/np.sum(y)
        return y

    # problem parameters
    N = 100
    M = 3
    x = np.linspace(0, 4, N)
    x0 = [1, 2, 3.5]
    sig = 0.5

    # target
    y = gauss_normed(1, x0[0]+ sig*0.2, sig)  + gauss_normed(0.1, x0[1]+ sig*0.2, sig) + gauss_normed(0.05, x0[2], sig*0.4)
    y = y/np.sum(y)

    # basis
    b0 = gauss_normed(1., x0[0], sig) 
    b1 = gauss_normed(1., x0[1], sig)
    X = np.vstack([b for b in [b0, b1]]).T

    # transport cost
    x_ = np.arange(N)/N
    C = np.abs(x_[:,np.newaxis] - x_[np.newaxis,:]) + np.eye(len(x_))*1e-20
    C = C**2
    # C = C * 1000
        
    return X, y, C



###############################################################################
##
## Fix
##
###############################################################################



def kl_div(a, b):
    """
    Compute KL(a || b) = a * log(a/b) - a + b,
    safely handling cases where a or b could be zero-ish.
    Here we assume a, b > 0 in the domain.
    """
    # To avoid numerical issues if a or b are near zero, we add a tiny offset:
    eps = 1e-15
    a = np.maximum(a, eps)
    b = np.maximum(b, eps)
    return a * np.log(a / b) - a + b

def L(w, X, y):
    """
    Objective function L(w) = sum_i KL( sum_j X_{i,j} w_j  ||  y_i ).
    w : (M,) nonnegative
    X : (N, M)
    y : (N,)
    """
    N, M = X.shape
    # sum_j X_{i,j} w_j for each i
    s = X.dot(w)  # shape (N,)
    # sum of KL divergences
    return np.sum(kl_div(s, y))

def prox_kl_nnsx_mm(X, y, max_iter=100, tol=1e-6, verbose=True):
    """
    Solve min_{w >= 0} sum_{i=1}^N KL( sum_j X_{i,j} w_j || y_i )
    using the MM algorithm with closed-form updates at each iteration.

    Parameters
    ----------
    X : array, shape (N, M)
        Nonnegative.
    y : array, shape (N,)
        Nonnegative.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping based on change in w.
    verbose : bool
        If True, print w and L(w) at each iteration.

    Returns
    -------
    w : array, shape (M,)
        The nonnegative solution found by the MM procedure.
    """

    N, M = X.shape

    # 1) Initialize w
    w = np.ones(M)  # start with all ones (can also choose other positive init)

    # Precompute row sums of X to speed up denominators
    # But note that for the update we also need Z_{i,k}(w).
    # We'll just compute them on the fly each iteration if needed.

    for it in range(max_iter):
        w_old = w.copy()

        # Compute denominator for each i: sum_{ell} X_{i,ell} w_{ell}
        denom_i = X.dot(w_old)  # shape (N,)

        # Build Z_{i,k}(w_old)
        # Z_{i,k} = (X_{i,k} * w_old[k]) / ( sum_{ell} X_{i,ell} * w_old[ell] )
        # shape (N, M)
        Z = (X * w_old) / denom_i[:, None]  # broadcasting

        # Closed-form update for each k
        w_new = np.zeros_like(w_old)
        for k in range(M):
            # sum_{i=1}^N X_{i,k}
            denom_k = np.sum(X[:, k])

            # sum_{i=1}^N X_{i,k} * ln( (Z_{i,k} * y_i) / X_{i,k} )
            # Avoid division by zero with a small eps
            eps = 1e-15
            ratio = (Z[:, k] * y) / (X[:, k] + eps)  # shape (N,)
            numer_k = np.sum( X[:, k] * np.log(np.maximum(ratio, eps)) )

            w_new[k] = np.exp(numer_k / denom_k)

        w = w_new

        # Check progress
        obj_val = L(w, X, y)
        if verbose:
            print(f"MM Iteration {it+1}: w = {w}, L(w) = {obj_val:.6f}")

        # Stopping criterion
        if np.linalg.norm(w - w_old) < tol:
            break

    return w

def prox_kl_nnsx_direct(X, y, w_init=None, verbose=True):
    """
    Solve the same problem min_{w >= 0} sum_i KL( sum_j X_{i,j} w_j || y_i )
    using scipy.optimize.minimize with numerical differentiation,
    and enforcing w >= 0 via bounds.

    Parameters
    ----------
    X : array, shape (N, M)
    y : array, shape (N,)
    w_init : array, shape (M,) (optional)
        Initial guess for w. If None, defaults to ones vector.
    verbose : bool
        If True, print w and L(w) at each iteration (via callback).

    Returns
    -------
    res.x : array, shape (M,)
        The solution found by the optimizer.
    """

    N, M = X.shape
    if w_init is None:
        w_init = np.ones(M)

    # Objective function in the form expected by `minimize`
    def objective(w):
        return L(w, X, y)

    # We can define a callback to monitor progress.
    iteration_holder = {'count':0}
    def callback(w):
        iteration_holder['count'] += 1
        if verbose:
            print(f"Direct Iter {iteration_holder['count']}: w = {w}, L(w) = {objective(w):.6f}")

    # Bounds w >= 0
    bounds = [(0.0, None)] * M

    # Rely on numerical gradient => set 'jac=None'
    res = minimize(
        fun=objective,
        x0=w_init,
        method='SLSQP',
        bounds=bounds,
        callback=callback,
        options={'maxiter': 200, 'disp': False}  # Increase maxiter if needed
    )

    return res.x



def test_prox_kl_nnsx_mm():

    # -------------------
    # (1) Create a toy example
    # -------------------
    np.random.seed(0)

    # Suppose we have N=4, M=3
    N, M = 4, 3
    # Let X be nonnegative random
    X = np.random.rand(N, M) + 0.1  # shift so strictly positive
    # Let y be also positive
    y = np.array([0.5, 1.0, 2.0, 1.5])  # or random

    print("Toy Example Data:")
    print("X =\n", X)
    print("y =", y, "\n")

    # -------------------
    # (2) Solve using prox_kl_nnsx_mm
    # -------------------
    print("=== Solve with MM Algorithm (prox_kl_nnsx_mm) ===")
    w_mm = prox_kl_nnsx_mm(X, y, max_iter=10000, tol=1e-7, verbose=True)
    L_mm = L(w_mm, X, y)
    print("\nFinal MM result:")
    print("w_mm =", w_mm)
    print("L(w_mm) =", L_mm, "\n")

    # -------------------
    # (3) Solve using prox_kl_nnsx_direct
    # -------------------
    print("=== Solve with Direct Scipy (prox_kl_nnsx_direct) ===")
    w_direct = prox_kl_nnsx_direct(X, y, w_init=None, verbose=True)
    L_direct = L(w_direct, X, y)
    print("\nFinal Direct result:")
    print("w_direct =", w_direct)
    print("L(w_direct) =", L_direct, "\n")

    # -------------------
    # (4) Compare results
    # -------------------
    print("Comparison:")
    print(f"   w_mm      = {w_mm}")
    print(f"   L(w_mm)   = {L_mm:.6f}")
    print(f"   w_direct  = {w_direct}")
    print(f"   L(w_direct)= {L_direct:.6f}")

    print(f"   diff = {np.linalg.norm(w_mm - w_direct):.6f}")
    print(f"   diff_rel = {np.linalg.norm(w_mm - w_direct) / np.linalg.norm(w_mm):.6f}")
    assert np.allclose(w_mm, w_direct, atol=1e-3)
    print("====================> Test test_prox_kl_nnsx_mm passed!")


def test_otlm_fix():

    X, y, C = get_toy_example()

    print("\n=== Running test_otlm_fix ===")

    otlm = OTLM(datafit='fix', penalty='no', ε=0.1, λ=100, tol=1e-12, max_iter_mm=1000, max_iter=400, options={'disp':True})
    otlm.fit(C, X, y)

    correct_w =  np.array([0.87206043, 0.12790632])
    assert np.allclose(otlm.w, correct_w, rtol=1e-3),\
        f"w solutions differ more than expected: correct_w={correct_w} fitted_w={otlm.w}"

    print("test_otlm_fix passed: solutions match within tolerance!")


###############################################################################
##
## L1
##
###############################################################################

#------------------------------------------------------------------------------
# 1) Define utility functions
#------------------------------------------------------------------------------

def kl_div(a, b, eps=1e-12):
    """
    Kullback-Leibler divergence KL(a || b) = a log(a/b) - a + b.
    Both a,b must be >= 0.
    Add small eps inside log() to avoid log(0).
    """
    a_clip = np.maximum(a, eps)
    b_clip = np.maximum(b, eps)
    return a_clip * np.log(a_clip / b_clip) - a_clip + b_clip

def L_L1_objective(w, X, y, epsilon, alpha, eps=1e-12):
    """
    Compute L_{L1}(w) = epsilon * sum_i KL( sum_j X_{i,j} * w_j  ||  y_i ) + alpha * ||w||_1
    w must be nonnegative, but we do not enforce it here explicitly (the optimizer or algorithm does).
    """
    # Ensure w >= 0 numerically for the KL part
    w_clip = np.maximum(w, 0.0)
    
    # Predicted values s_i = sum_j X_{i,j} w_j
    s = X.dot(w_clip)
    
    # Sum of KL divergences
    kl_sum = 0.0
    for i in range(len(y)):
        kl_sum += kl_div(s[i], y[i], eps=eps)
    
    # L1 penalty
    l1_norm = np.sum(w_clip)
    
    return epsilon * kl_sum + alpha * l1_norm


#------------------------------------------------------------------------------
# 2) MM Algorithm for ProxNNSX L1
#------------------------------------------------------------------------------

def prox_kl_nnsx_l1_mm(X, y, epsilon, alpha, w_init, 
                       max_iter=100, tol=1e-6, verbose=True):
    """
    Solve:
        min_{w >= 0}  epsilon * sum_{i} KL( sum_{j} X_{i,j} w_j || y_i ) + alpha * ||w||_1
    using the Majorization-Minimization (MM) coordinate update derived in the notes.
    
    Args:
        X: 2D numpy array of shape (N, M)
        y: 1D numpy array of shape (N,)
        epsilon: float
        alpha: float
        w_init: 1D numpy array of shape (M,), initial guess (>= 0)
        max_iter: maximum number of MM iterations
        tol: tolerance for stopping (based on ||w^{(t+1)} - w^{(t)}||)
        verbose: if True, print iteration info
    
    Returns:
        w: the final solution (1D numpy array of shape (M,))
    """
    w = np.copy(w_init).astype(float)
    w = np.maximum(w, 0.0)  # ensure nonnegativity

    N, M = X.shape
    
    # Precompute row sums for "denominator" in the coordinate update
    # R_k = sum_{i=1}^N X_{i,k}.
    R = np.sum(X, axis=0)  # shape (M,)
    
    for t in range(max_iter):
        # Compute current objective
        L_curr = L_L1_objective(w, X, y, epsilon, alpha)
        
        if verbose:
            print(f"[MM] Iter {t}: w = {w}, L_L1 = {L_curr:.6f}")
        
        w_old = w.copy()
        
        # Compute Z_{i,k}(w): shape (N, M)
        # Z_{i,k} = (X_{i,k} * w_k) / sum_{l} X_{i,l} * w_l
        s = X.dot(w)  # shape (N,)
        # Avoid division by zero
        s_safe = np.maximum(s, 1e-12)
        
        Z = (X * w) / s_safe[:, None]  # broadcasting (N,1)
        
        # Coordinate-wise update
        for k in range(M):
            # sum_{i=1}^N X_{i,k} ln( X_{i,k} * w_k / [Z_{i,k}(w) * y_i] )
            # + alpha = 0  --> solve for w_k
            # We rewrite in the form from the notes:
            
            # S_k = sum_{i=1}^N X_{i,k} ln( Z_{i,k}(w) * y_i / X_{i,k} )
            # ln(w_k) = (1 / R_k) [ S_k - alpha / epsilon ]
            
            # compute S_k
            # watch out for log(0). Use a small clip if needed.
            numerator = Z[:, k] * y
            denom = X[:, k]
            # safe clip
            numerator_clip = np.maximum(numerator, 1e-12)
            denom_clip = np.maximum(denom, 1e-12)
            
            log_term = np.log(numerator_clip / denom_clip)
            
            S_k = np.sum(X[:, k] * log_term)
            
            # exponent
            exponent = (S_k - alpha / epsilon) / (R[k] + 1e-12)
            
            w_k_new = np.exp(exponent)
            
            # check for negativity or extremely small
            if w_k_new < 1e-30:
                w_k_new = 0.0
            
            w[k] = w_k_new
        
        # Check convergence
        step_size = np.linalg.norm(w - w_old)
        if step_size < tol:
            break
    
    # Final print
    L_final = L_L1_objective(w, X, y, epsilon, alpha)
    if verbose:
        print(f"[MM] Final: w = {w}, L_L1 = {L_final:.6f}")
    
    return w


#------------------------------------------------------------------------------
# 3) Direct Approach with scipy.optimize.minimize
#------------------------------------------------------------------------------

def prox_kl_nnsx_l1_direct(X, y, epsilon, alpha, w_init, 
                           max_iter=100, tol=1e-6, verbose=True):
    """
    Solve the same problem directly using scipy.optimize.minimize with
    numerical gradient (no explicit jac). We enforce w >= 0 using bounds.
    
    Args:
        X, y, epsilon, alpha: same meaning as above
        w_init: initial guess
        max_iter: maximum iteration
        tol: tolerance
        verbose: if True, print iteration info
    
    Returns:
        w: final solution
    """
    N, M = X.shape
    
    # Objective function
    def fun(w):
        return L_L1_objective(w, X, y, epsilon, alpha)

    # We'll store iteration info in a closure variable
    iteration_count = [0]  # list so we can modify inside callback
    
    def callback(wk):
        iteration_count[0] += 1
        val = fun(wk)
        if verbose:
            print(f"[Direct] Iter {iteration_count[0]}: w = {wk}, L_L1 = {val:.6f}")
    
    # Enforce w >= 0 via bounds
    bounds = [(0.0, None)] * M
    
    # Use 'SLSQP' or 'trust-constr' or similar. SLSQP often works well with bounds.
    res = minimize(fun, w_init, method='SLSQP', bounds=bounds, 
                   callback=callback, options={'maxiter': max_iter, 'ftol': tol})
    
    w_opt = res.x
    # Final print
    if verbose:
        print(f"[Direct] Final: w = {w_opt}, L_L1 = {fun(w_opt):.6f}, success={res.success}")
    
    return w_opt



def test_prox_kl_nnsx_l1_mm():

    np.random.seed(0)
    
    # (a) Create toy data
    N, M = 5, 3
    X = np.random.rand(N, M) * 2.0  # random data in [0,2)
    y = np.random.rand(N) * 2.0     # random positive targets
    epsilon = 1.0
    alpha = 0.1
    
    w_init = np.ones(M)  # initial guess
    
    print("Toy Data:")
    print("X =\n", X)
    print("y =\n", y)
    print("epsilon =", epsilon, ", alpha =", alpha)
    print("Initial w =", w_init)
    print("---------------------------------------------------")
    
    # (b) Solve via MM
    print("\nSolving ProxNNSX L1 via MM Algorithm...\n")
    w_mm = prox_kl_nnsx_l1_mm(X, y, epsilon, alpha, w_init, max_iter=20, tol=1e-7, verbose=True)
    L_mm = L_L1_objective(w_mm, X, y, epsilon, alpha)
    
    print("\nResult from MM approach:")
    print("w_mm =", w_mm)
    print("L_L1(w_mm) =", L_mm)
    print("---------------------------------------------------")
    
    # (c) Solve via scipy.optimize.minimize
    print("\nSolving ProxNNSX L1 via Direct Approach (scipy)...\n")
    w_direct = prox_kl_nnsx_l1_direct(X, y, epsilon, alpha, w_init, max_iter=20, tol=1e-7, verbose=True)
    L_direct = L_L1_objective(w_direct, X, y, epsilon, alpha)
    
    print("\nResult from Direct approach:")
    print("w_direct =", w_direct)
    print("L_L1(w_direct) =", L_direct)
    
    # (d) Compare
    print("\n---------------------------------------------------")
    print("Comparison of final results:")
    print(f"MM      => w = {w_mm},       L_L1 = {L_mm:.6f}")
    print(f"Direct  => w = {w_direct},   L_L1 = {L_direct:.6f}")


    print(f"   diff = {np.linalg.norm(w_mm - w_direct):.6f}")
    print(f"   diff_rel = {np.linalg.norm(w_mm - w_direct) / np.linalg.norm(w_mm):.6f}")
    assert np.allclose(w_mm, w_direct, atol=1e-3)
    print("====================> Test test_prox_kl_nnsx_l1_mm passed!")

def test_otlm_l1():

    X, y, C = get_toy_example()

    print("\n=== Running test_otlm_l1 ===")
    correct_w =  np.array([0.87231605, 0.12768395])
    
    otlm = OTLM(datafit='fix', penalty='l1', ε=0.1, λ=100, tol=1e-12, max_iter_mm=1, max_iter=400, options={'disp':True})
    otlm.fit(C, X, y)


    assert np.allclose(otlm.w, correct_w, rtol=1e-2),\
        f"w solutions differ more than expected: correct_w={correct_w} fitted_w={otlm.w}"

    otlm.fit(sparse.coo_array(C), sparse.coo_array(X), y)

    assert np.allclose(otlm.w, correct_w, rtol=1e-2),\
        f"w solutions differ more than expected: correct_w={correct_w} fitted_w={otlm.w}"


    print("test_otlm_l1 passed: solutions match within tolerance!")




###############################################################################
##
## L2
##
###############################################################################

from scipy.special import lambertw

def L_L2(w, X, y, eps, alpha):
    """
    Computes L_L2(w) = eps * sum_i KL(Xw[i] | y[i]) + alpha/2 * ||w||^2_2,
    where KL(a|b) = a ln(a/b) - a + b, assuming all entries are positive.
    """
    # Ensure w is an array
    w = np.asarray(w, dtype=float)
    # Compute Xw (predictions)
    Xw = X @ w  # shape (N,)

    # Avoid taking log of zero or negative values by a small shift if needed
    # (In real applications, ensure Xw > 0 and y > 0.)
    # We'll assume Xw and y are strictly positive for simplicity.

    # KL part: sum_i [ Xw[i]*ln(Xw[i]/y[i]) - Xw[i] + y[i] ]
    kl_part = np.sum(Xw * np.log(Xw / y) - Xw + y)

    # L2 regularization part: alpha/2 * ||w||^2
    l2_part = 0.5 * alpha * np.sum(w**2)

    return eps * kl_part + l2_part



def prox_kl_nnsx_l2_mm(X, y, eps, alpha, w0=None, max_iter=20, tol=1e-7):
    """
    Vectorized MM solver for the problem:
        min_{w >= 0}  L_L2(w) = eps * sum_i KL( (Xw)[i] || y[i] )  +  (alpha/2) * ||w||^2,
    where
        KL(a||b) = a ln(a/b) - a + b,
    and X, y, w >= 0.

    We use a Majorization–Minimization approach with closed-form updates via
    the Lambert W function, but do so in a vectorized fashion (no for-loops
    over rows or columns).

    Parameters
    ----------
    X : ndarray of shape (N, M), assumed nonnegative
    y : ndarray of shape (N,),   assumed positive
    eps : float > 0
    alpha : float > 0
    w0 : ndarray of shape (M,) or None
         If None, we initialize with w0 = ones(M).
    max_iter : int
        Maximum number of MM iterations.
    tol : float
        Stopping tolerance based on ||w^{(t+1)} - w^{(t)}||_2.

    Returns
    -------
    w : ndarray of shape (M,)
        The final solution w >= 0.
    """

    # --- 1. Initialization ---
    N, M = X.shape
    if w0 is None:
        w = np.ones(M, dtype=float)
    else:
        w = np.array(w0, dtype=float)

    # Precompute column sums S_k = sum_i X_{i,k}, assumed > 0 by problem statement.
    S = np.sum(X, axis=0)  # shape (M,)

    # Helper function: objective L_L2(w)
    def L_L2(w_vec):
        # Xw
        Xw = X @ w_vec  # shape (N,)

        # KL part: sum_i [ Xw[i] * ln(Xw[i]/y[i]) - Xw[i] + y[i] ]
        # Use small safeguarding if needed, but assuming positivity here:
        kl_part = np.sum(Xw * np.log(Xw / y) - Xw + y)

        # L2 part: alpha/2 * ||w||^2
        l2_part = 0.5 * alpha * np.sum(w_vec**2)
        return eps * kl_part + l2_part

    # --- 2. MM Iterations ---
    for t in range(max_iter):
        # Current objective
        L_curr = L_L2(w)

        # Print iteration, w, and objective
        print(f"MM Iter = {t}, w = {w}, L_L2(w) = {L_curr:.6f}")

        # Compute row-sums Xw and form Z_{i,j}(w)
        # Z_{i,j} = (X_{i,j} * w_j) / (Xw)_i
        Xw = X @ w  # shape (N,)
        # Avoid division by zero; assume positivity from problem statement.
        Z = (X * w[np.newaxis, :]) / Xw[:, np.newaxis]  # shape (N, M)

        # Vectorized D_k = sum_i X_{i,k} [ ln(X_{i,k}) - ln(Z_{i,k}) - ln(y_i) ]
        # Break it into terms:
        logX = np.log(X)                 # shape (N, M)
        logZ = np.log(Z + 1e-300)        # add tiny offset if needed for safety
        logy = np.log(y + 1e-300)        # shape (N,)
        # Each term is shape (N, M), then sum over rows => shape (M,)
        D = np.sum( X * (logX - logZ - logy[:, None]), axis=0 )  # shape (M,)

        # Solve for w_{k} via the Lambert W expression:
        #   w_k = (eps*S_k / alpha) * W( (alpha/(eps*S_k)) * exp(-D_k / S_k ) ).
        a = np.exp(-D / S)                   # shape (M,)
        b = alpha / (eps * S)                # shape (M,)
        ba = b * a                           # shape (M,)
        w_update = (eps * S / alpha) * lambertw(ba)  # shape (M,)

        # Take real part if lambertw returns a complex dtype with zero imaginary part
        w_new = np.real(w_update)

        # Enforce nonnegativity
        w_new[w_new < 0] = 0.0

        # Check convergence
        delta_norm = np.linalg.norm(w_new - w)
        w = w_new
        if delta_norm < tol:
            break

    # Final print
    L_final = L_L2(w)
    print(f"MM Iter = {t+1}, w = {w}, L_L2(w) = {L_final:.6f} (FINAL)")

    return w



def prox_kl_nnsx_l2_direct(X, y, eps, alpha, w0=None, max_iter=20):
    """
    Direct approach using scipy.optimize.minimize (with numerical gradients).
    Prints w and L_L2(w) at each step of the optimizer's outer iteration.
    
    Parameters:
    -----------
    X : array (N, M)
    y : array (N,)
    eps : float
    alpha : float
    w0 : initial guess (M,) or None
    max_iter : max iteration count for 'SLSQP'
    
    Returns:
    --------
    w : final solution
    """
    N, M = X.shape
    if w0 is None:
        w_init = np.ones(M, dtype=float)
    else:
        w_init = np.array(w0, dtype=float)
    
    # Bounds: w_j >= 0
    bnds = [(0, None) for _ in range(M)]
    
    # We define a callback to print iteration info
    def callback_func(w_candidate):
        val = L_L2(w_candidate, X, y, eps, alpha)
        print(f"Minimize step: w = {w_candidate}, L_L2(w) = {val:.6f}")
    
    # Run optimizer
    res = minimize(
        fun=lambda w_var: L_L2(w_var, X, y, eps, alpha),
        x0=w_init,
        method='SLSQP',
        bounds=bnds,
        callback=callback_func,   # to print progress
        options={'maxiter': max_iter, 'disp': False}
    )
    
    # Final
    w_final = res.x
    print(f"Direct solution finished. w = {w_final}, L_L2(w) = {L_L2(w_final, X, y, eps, alpha):.6f}")
    return w_final


def test_prox_kl_nnsx_l2_mm():

    # 1) Toy example
    np.random.seed(0)
    N, M = 5, 3
    X = np.random.rand(N, M) + 0.1  # ensure positivity
    y = np.random.rand(N) + 0.1     # ensure positivity
    
    eps = 1.0   # weighting on KL
    alpha = 0.5 # weighting on L2
    
    print("\n=== TOY EXAMPLE DATA ===")
    print("X =\n", X)
    print("y =", y)
    print("eps =", eps, "alpha =", alpha)
    
    # 2) Solve using MM
    print("\n=== SOLVING WITH MM ALGORITHM ===")
    w_mm = prox_kl_nnsx_l2_mm(X, y, eps, alpha, max_iter=100, tol=1e-6)
    
    # 3) Solve using direct approach
    print("\n=== SOLVING WITH DIRECT APPROACH (scipy.optimize.minimize) ===")
    w_direct = prox_kl_nnsx_l2_direct(X, y, eps, alpha, max_iter=100)
    
    # 4) Compare final results
    L_mm = L_L2(w_mm, X, y, eps, alpha)
    L_dir = L_L2(w_direct, X, y, eps, alpha)
    print("\n=== COMPARISON ===")
    print("MM solution:      w =", w_mm, "L_L2(w) =", L_mm)
    print("Direct solution:  w =", w_direct, "L_L2(w) =", L_dir)

    print(f"   diff = {np.linalg.norm(w_mm - w_direct):.6f}")
    print(f"   diff_rel = {np.linalg.norm(w_mm - w_direct) / np.linalg.norm(w_mm):.6f}")
    assert np.allclose(w_mm, w_direct, atol=1e-3)
    print("====================> Test test_prox_kl_nnsx_l2_mm passed!")

def test_otlm_l2():

    X, y, C = get_toy_example()

    print("\n=== Running test_otlm_l2 ===")

    otlm = OTLM(datafit='fix', penalty='l2', ε=0.1, λ=100, tol=1e-12, max_iter_mm=1000, max_iter=400, options={'disp':True})
    otlm.fit(C, X, y)

    correct_w =  np.array([0.87231605, 0.12768395])
    assert np.allclose(otlm.w, correct_w, rtol=1e-2),\
        f"w solutions differ more than expected: correct_w={correct_w} fitted_w={otlm.w}"

    print("test_otlm_l2 passed: solutions match within tolerance!")


if __name__ == "__main__":
        
    test_prox_kl_nnsx_mm() # test the original derived function
    test_otlm_fix() # test the implementation in the package
    
    test_prox_kl_nnsx_l1_mm() # test the original derived function
    test_otlm_l1() # test the implementation in the package

    test_prox_kl_nnsx_l2_mm() # test the original derived function
    test_otlm_l2() # test the implementation in the package
