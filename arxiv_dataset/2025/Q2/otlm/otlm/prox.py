import numpy as np
from scipy.optimize import minimize
from scipy.special import lambertw
from scipy import sparse


##################################################################################  
##
##
## Proximal operators
##
##
##################################################################################  

def proxdiv_fix(y, k, **options):

    return safediv(y, k)

def proxdiv_kl(y, k, ε, λ, **options):
    
    return (y/k)**(λ/(ε+λ))

def proxdiv_tv(y, k, ε, λ, **options):

    return np.minimum(safeexp(λ/ε), np.maximum(safeexp(-λ/ε), y/k) )

def proxdiv_box(y, k, lb, ub, **options):

    return np.minimum( ub * y/k, np.maximum(lb * y/k, 1) )

def proxdiv_quadratic(y, k, λ, ε, **options):

    nom = ε/λ * lambertw(λ/ε * k * safeexp(λ/ε * y), k=0).real  # Shape (N,)
    return safediv(nom, k)

def proxdiv_poisson(y, k, λ, ε, **options):

    nom = solve_poisson_marginal(p=y, x=k, lam=λ, eps=ε, small_threshold=1e-12)
    return safediv(nom, k) 

def proxdiv_kl_nnsx(X, k, ε, max_iter=100, tol=1e-8, w0=None, **options):

    # w = prox_kl_nnsx_direct(X, k, ε, max_iter=max_iter, w0=w0)
    w = prox_kl_nnsx_mm_jensen(X, k, max_iter=max_iter, tol=tol, w0=w0)

    return safediv(X.dot(w), k), w

def proxdiv_kl_nnsx_l1(X, k, ε, α, max_iter=100, tol=1e-8, w0=None, **options):

    # w = prox_kl_nnsx_l1_direct(X, k, ε, α, max_iter=max_iter)
    # w = prox_kl_nnsx_l1_mm_jensen(X, k, ε=ε, α=α, w_init=w0, max_iter=max_iter, tol=tol, verbose=options.get('verbose', False))
    w = prox_kl_nnsx_l1_mm_jensen_sp(X, k, ε=ε, α=α, w_init=w0, max_iter=max_iter, tol=tol, verbose=False)
    return safediv(X.dot(w), k), w

def proxdiv_kl_nnsx_l2(X, k, ε, α, max_iter=100, tol=1e-8, w0=None, **options):

    # w = prox_kl_nnsx_l2_direct(X, k, ε=ε, α=α, max_iter=max_iter)
    # w = prox_kl_nnsx_l2_mm_jensen(X, k, ε=ε, α=α, w_init=w0, max_iter=max_iter, tol=tol, verbose=options.get('verbose', False))
    w = prox_kl_nnsx_l2_mm_jensen_sp(X, k, ε=ε, α=α, w_init=w0, max_iter=max_iter, tol=tol, verbose=False)
    return safediv(X.dot(w), k), w


##################################################################################  
##
##
## Helper functions
##
##
##################################################################################  

def astr(x):
    return ' '.join([f'{x_:2.6e}' for x_ in x])

def print_arr(tag, a):

    if sparse.issparse(a):
        if a.getnnz() > 0:
            print('{:>20s} size={} nnz={} [{:.2f}%] min={:2.4e} max={:2.4e} mean={:2.4e}'.format(tag, a.shape, a.getnnz(), a.getnnz()/np.prod(a.shape)*100, a.min(), a.max(), a.mean()))
        else:
            print('{:>20s} all 0'.format(tag))
    else:
        if np.any(a):
            print('{:>20s} size={} nnz={} [{:.2f}%] min={:2.4e} max={:2.4e} mean={:2.4e}'.format(tag, a.shape, np.count_nonzero(a), np.count_nonzero(a)/np.prod(a.shape)*100, np.min(a[a!=0]), np.max(a[a!=0]), np.mean(a[a!=0])))
        else:
            print('{:>20s} all 0'.format(tag))


def safeexp(x):

    if hasattr(x, "__len__"):
        return np.exp(x, where=x<10000, out=np.full_like(x, np.inf))
    else:
        return np.exp(x) if x<10000 else np.inf

def safediv(nom, den, eps=1e-14):

    return np.divide(nom, den, out=np.zeros_like(nom), where=np.abs(den)>eps)


def solve_poisson_marginal(p, x, lam, eps, small_threshold=1e-12):
    """
    Solve the Poisson Marginal problem:
    
        min_{y >= 0} sum_i [ lambda * (y_i - p_i log(y_i)) 
                             + eps * (y_i log(y_i/x_i) - y_i + x_i) ]
    
    Parameters
    ----------
    p : 1D numpy array
        Observed Poisson counts (p_i >= 0).
    x : 1D numpy array
        Reference vector for the KL term (x_i > 0).
    lam : float
        Weight (lambda) for the Poisson negative log-likelihood term.
    eps : float
        Weight (epsilon) for the KL divergence term.
    small_threshold : float, optional
        A small threshold to treat variables as zero. Defaults to 1e-12.
    
    Returns
    -------
    y : 1D numpy array
        The coordinate-wise solution y_i.
    """

    # val = (lam * p / eps) / lambertw((lam * p / (eps * x)) * np.exp(lam / eps)).real
    # clamp_val = x * np.exp(-lam / eps)  # For p_i=0, y_i = x_i * exp(-lam/eps)
    # y = np.where(p < small_threshold, x * np.exp(-lam / eps), (lam * p / eps) / lambertw((lam * p / (eps * x)) * np.exp(lam / eps)).real)
    # y = np.where(p < small_threshold, 0, (lam * p / eps) / lambertw((lam * p / (eps * x)) * np.exp(lam / eps)).real)

    clamp_val = x * np.exp(-lam / eps)
    
    # val = np.where(eps*x < small_threshold, clamp_val,  (lam * p / eps) / lambertw((lam * p / (eps * x)) * np.exp(lam / eps)).real)
    # val = (lam * p / eps) / lambertw((lam * p / (eps * x)) * np.exp(lam / eps)).real

    # y = np.where(p < small_threshold, clamp_val, val)
    y = np.where(p < small_threshold, x * np.exp(-lam / eps), (lam * p / eps) / lambertw((lam * p / (eps * x)) * np.exp(lam / eps)).real)
    return y



##################################################################################  
##################################################################################  
##
##
# MM for Jensen's inequality
##
##
##################################################################################
##################################################################################  


##################################################################################  
#
# MM no penalty
#
##################################################################################  


def prox_kl_nnsx_mm_jensen(X, y, max_iter=100, tol=1e-6, verbose=True, w0=None):
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
    w = w0 if w0 is not None else np.ones(M)  # start with all ones (can also choose other positive init)

    # Precompute row sums of X to speed up denominators
    # But note that for the update we also need Z_{i,k}(w).
    # We'll just compute them on the fly each iteration if needed.

    denom_k = np.sum(X, axis=0)  # shape (M,)

    safelog = lambda x: np.log(x, where=x>1e-15, out=np.zeros_like(x))

    # print_arr('  y', y)

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
        ratio = (Z * y[:, None]) / (X + 1e-15)  # shape (N, M)
        # numer_k = np.sum(X * np.log(np.maximum(ratio, 1e-15)), axis=0)  # shape (M,)
        numer_k = np.sum(X * safelog(ratio), axis=0)  # shape (M,)
        w_new = np.exp(numer_k / denom_k)  # shape (M,)

        # print(f'  Prox KL NNSX Jensen MM iter {it:>4d}, w_new = {astr(w_new)} diff={np.linalg.norm(w_new - w_old):.6e}')
        # print_arr('    ratio', ratio)
        # print_arr('    numer_k', numer_k)
        # print_arr('    numer_k / denom_k', numer_k /  denom_k)
        w = w_new

        # Stopping criterion
        if np.linalg.norm(w - w_old) < tol:
            break

    return w

##################################################################################  
#
# MM with L1 penalty
#
##################################################################################  



def prox_kl_nnsx_l1_mm_jensen(X, y, ε, α, w_init=None, max_iter=100, tol=1e-6, verbose=True):
    """
    Solve:
        min_{w >= 0}  ε * sum_{i} KL( sum_{j} X_{i,j} w_j || y_i ) + α * ||w||_1
    using the Majorization-Minimization (MM) coordinate update derived in the notes.
    
    Args:
        X: 2D numpy array of shape (N, M)
        y: 1D numpy array of shape (N,)
        ε: float
        α: float
        w_init: 1D numpy array of shape (M,), initial guess (>= 0)
        max_iter: maximum number of MM iterations
        tol: tolerance for stopping (based on ||w^{(t+1)} - w^{(t)}||)
        verbose: if True, print iteration info
    
    Returns:
        w: the final solution (1D numpy array of shape (M,))
    """

    w_init = np.ones(X.shape[1]) if w_init is None else w_init
    w = np.copy(w_init).astype(float)
    w = np.maximum(w, 0.0)  # ensure nonnegativity

    N, M = X.shape
    
    # Precompute row sums for "denominator" in the coordinate update
    # R_k = sum_{i=1}^N X_{i,k}.
    R = np.sum(X, axis=0)  # shape (M,)
    
    for t in range(max_iter):
                
        w_old = w.copy()
        
        # Compute Z_{i,k}(w): shape (N, M)
        # Z_{i,k} = (X_{i,k} * w_k) / sum_{l} X_{i,l} * w_l
        s = X.dot(w)  # shape (N,)
        # Avoid division by zero
        s_safe = np.maximum(s, 1e-12)
        
        Z = (X * w) / s_safe[:, None]  # broadcasting (N,1)
        
        # Coordinate-wise update
        # Vectorized computation of S_k for all k simultaneously
        numerator = Z * y[:, np.newaxis]  # shape (N, M)
        denom = X  # shape (N, M)
        
        # Safe clipping
        numerator_clip = np.maximum(numerator, 1e-12)
        denom_clip = np.maximum(denom, 1e-12)
        
        # Compute log terms for all k
        log_term = np.log(numerator_clip / denom_clip)  # shape (N, M)
        
        # Compute S_k for all k: sum over rows (axis 0)
        S_k = np.sum(X * log_term, axis=0)  # shape (M,)
        
        # Compute exponents for all k
        exponent = (S_k - α / ε) / (R + 1e-12)  # shape (M,)
        
        # Compute new w values and apply threshold
        w_new = np.exp(exponent)  # shape (M,)
        w = np.where(w_new < 1e-30, 0.0, w_new)
        
        
        # Check convergence
        step_size = np.linalg.norm(w - w_old)
        if step_size < tol:
            break
   
    
    return w


##############################################################################
#                            Helper Functions                                #
##############################################################################

def safe_sum(X, axis=None):
    """
    Sums along 'axis' for both dense and sparse matrices,
    returning a 1D or scalar NumPy array (never a 2D sparse).
    """
    s = X.sum(axis=axis)
    if sparse.issparse(s):
        # For axis=0 or 1, s will be a sparse matrix of shape (1, M) or (N, 1).
        s = np.asarray(s).ravel()
    return s

def safe_dot(X, w):
    """
    Matrix-vector product X.dot(w), valid for both dense and sparse X.
    """
    return X.dot(w)

def row_scale(X, scale_vec):
    """
    Multiply each row of X by scale_vec[i].
    - For dense X shape (N, M): X * scale_vec[:, None].
    - For sparse X shape (N, M): sparse.diags(scale_vec).dot(X).
    """
    if sparse.issparse(X):
        D = sparse.diags(scale_vec)  # shape (N, N)
        return D.dot(X)
    else:
        return X * scale_vec[:, None]

def col_scale(X, scale_vec):
    """
    Multiply each column of X by scale_vec[j].
    - For dense: X * scale_vec[None, :].
    - For sparse: X.multiply(scale_vec).
    """
    if sparse.issparse(X):
        return X.multiply(scale_vec)  # each column k scaled by scale_vec[k]
    else:
        return X * scale_vec[np.newaxis, :]

def compute_Z(X, w, s):
    """
    Compute Z_{i,k} = (X_{i,k} * w_k) / s[i] in a vectorized way.
      - s = Xw, shape (N,).
      - For sparse X, do column-scale by w, then row-scale by 1/s.
    """
    # Step 1: column-scale by w
    Zw = col_scale(X, w)
    # Step 2: row-scale by 1/s
    inv_s = np.where(s > 0, 1.0 / s, 0.0)  # safeguard
    Z = row_scale(Zw, inv_s)
    return Z

def safe_clip(X, threshold=1e-12):
    """
    Clip all entries of X to be >= threshold, for both dense and sparse.
    """
    if sparse.issparse(X):
        # Work on .data
        np.maximum(X.data, threshold, out=X.data)
        return X
    else:
        return np.maximum(X, threshold)

def compute_log_term(numerator, denom, threshold=1e-12):
    """
    Compute log( numerator / denom ), clipping both to threshold,
    and returning a matrix of the same shape (dense or sparse).
    """
    
    # For simplicity, convert to dense if they are sparse with the same pattern.
    # Or do an elementwise approach. Here is a straightforward approach:
    if sparse.issparse(numerator) and sparse.issparse(denom):
        # num_dense = np.array(numerator.toarray())
        # den_dense = np.array(denom.toarray())
        # return np.log(num_dense, where=num_dense>0, out=np.zeros_like(num_dense)) - np.log(den_dense + 1e-15, where=den_dense>0, out=np.zeros_like(den_dense))  # returns a dense np.array
        
        numerator.data = np.log(numerator.data + 1e-15, out=np.zeros_like(numerator.data), where=numerator.data>0)
        return numerator - denom




    else:
        # Both dense
        return np.log(numerator + 1e-15) - denom

##############################################################################
#           Main Proximal KL + L1 Solver via MM (Jensen-based)               #
##############################################################################

def prox_kl_nnsx_l1_mm_jensen_sp(X, y, ε, α, w_init=None, max_iter=100, tol=1e-6, verbose=True):
    """
    Solve: 
      min_{w >= 0}  ε * sum_{i} KL( (Xw)[i] || y_i ) + α * ||w||_1
    using a Majorization-Minimization approach (no explicit loops over rows/cols).

    Args:
        X: 2D array (N, M), can be dense np.array or sparse scipy.sparse
        y: 1D np.array (N,), assumed positive
        ε: float > 0
        α: float >= 0
        w_init: None or shape (M,), >= 0
        max_iter: maximum number of MM iterations
        tol: stopping tolerance based on ||w^{(t+1)} - w^{(t)}||
        verbose: if True, prints iteration info

    Returns:
        w: 1D numpy array (M,), nonnegative solution
    """

    # --- 1. Initialization ---
    N, M = X.shape
    if w_init is None:
        w_init = np.ones(M, dtype=float)

    w = w_init.astype(float).copy()
    w = np.maximum(w, 0.0)  # ensure nonnegativity

    # Precompute column sums: R_k = sum_i X_{i,k}
    R = safe_sum(X, axis=0)  # shape (M,)

    logX = X.copy()
    if sparse.issparse(logX):
        logX.data = np.log(logX.data + 1e-15, where=logX.data>0, out=np.zeros_like(logX.data))
    else:
        logX = np.log(logX + 1e-15)

    # --- 2. MM Iterations ---
    for t in range(max_iter):
        w_old = w.copy()

        # 2a) Compute s = Xw
        s = safe_dot(X, w)   # shape (N,)
        # Safeguard
        s_safe = np.maximum(s, 1e-12)

        # 2b) Z_{i,k} = X_{i,k} * w_k / s[i]
        Z = compute_Z(X, w, s_safe)  # shape (N, M)

        # 2c) numerator = Z_{i,k} * y_i
        numerator = row_scale(Z, y)  # shape (N, M)

        # 2d) denom = X_{i,k} (just X)
        #     log_term = log( numerator / denom )
        log_term = compute_log_term(numerator, logX, threshold=1e-12)  
        # If compute_log_term returns a dense array, that is fine. 
        # We'll multiply by X again below in a shape-compatible way.

        # 2e) S_k = sum_i [X_{i,k} * log_term_{i,k}]
        #     => for sparse, do X.multiply(log_term).sum(axis=0).
        #        for dense, just (X * log_term).sum(axis=0).
        if sparse.issparse(X):
            # We have to multiply the (N,M) log_term with X (sparse) elementwise.
            # If log_term is dense, do X.multiply(log_term).
            # Then sum over rows => shape (M,).
            S_k = safe_sum(X.multiply(log_term), axis=0)
        else:
            # Dense
            S_k = np.sum(X * log_term, axis=0)  # shape (M,)

        # 2f) w_k update rule (exponent):
        #    exponent_k = [S_k - α/ε] / R_k
        #    w_k = exp(exponent_k), clipped at 0
        exponent = (S_k - (α / ε)) / np.maximum(R, 1e-12)
        w_new = np.exp(exponent)

        # Enforce nonnegativity & tiny threshold
        w = np.where(w_new < 1e-30, 0.0, w_new)

        # 2g) Convergence check
        step_size = np.linalg.norm(w - w_old)
        if verbose:
            print(f"Iter {t}: step={step_size:.3e}, ||w||={np.linalg.norm(w):.3e}")
        if step_size < tol:
            break

    return w


##################################################################################  
#
# MM with L2 penalty
#
##################################################################################  


def prox_kl_nnsx_l2_mm_jensen(X, y, ε, α, w_init=None, max_iter=20, tol=1e-7, verbose=False):
    """
    Vectorized MM solver for the problem:
        min_{w >= 0}  L_L2(w) = ε * sum_i KL( (Xw)[i] || y[i] )  +  (α/2) * ||w||^2,
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
    ε : float > 0
    α : float > 0
    w_init : ndarray of shape (M,) or None
         If None, we initialize with w_init = ones(M).
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
    if w_init is None:
        w = np.ones(M, dtype=float)
    else:
        w = np.array(w_init, dtype=float)

    # Precompute column sums S_k = sum_i X_{i,k}, assumed > 0 by problem statement.
    S = np.sum(X, axis=0)  # shape (M,)

    # Helper function: objective L_L2(w)
    def L_L2(w_vec):
        # Xw
        Xw = X @ w_vec  # shape (N,)

        # KL part: sum_i [ Xw[i] * ln(Xw[i]/y[i]) - Xw[i] + y[i] ]
        # Use small safeguarding if needed, but assuming positivity here:
        kl_part = np.sum(Xw * np.log(Xw / y) - Xw + y)

        # L2 part: α/2 * ||w||^2
        l2_part = 0.5 * α * np.sum(w_vec**2)
        return ε * kl_part + l2_part

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
        b = α / (ε * S)                # shape (M,)
        ba = b * a                           # shape (M,)
        w_update = (ε * S / α) * lambertw(ba)  # shape (M,)

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

##################################################################################  
#
# MM with L2 penalty, vectorized
#
##################################################################################  

##############################################################################
#               Helper Functions (Defined Outside the Main Solver)           #
##############################################################################

def safe_sum(X, axis=None):
    """Sum over the given axis for both dense and sparse X, 
    returning a 1D or scalar np.array for consistency."""
    s = X.sum(axis=axis)
    # For sparse, .sum(axis=0 or 1) returns a matrix; convert to np.array
    if sparse.issparse(s):
        return np.asarray(s).ravel()
    return s

def safe_log(X, eps=1e-300):
    """Elementwise log on both dense (ndarray) and sparse (csr, csc, coo) matrices,
    using a small positive offset eps to avoid log(0)."""
    if sparse.issparse(X):
        # Work on X.data if X is sparse
        X_coo = X.tocoo(copy=True)
        X_coo.data = np.log(X_coo.data + eps)
        return X_coo.tocsr()
    else:
        return np.log(X + eps)

def compute_Z(X, w, Xw):
    """
    Compute Z = (X * w_j) / (Xw_i) in a vectorized manner for both dense and sparse.
    - If X is shape (N, M) and w is shape (M,), then Xw is shape (N,).
    - For a dense array, we can broadcast directly.
    - For a sparse matrix, we do:
         1) column-scale by w,
         2) row-scale by 1/Xw.
    """
    if sparse.issparse(X):
        # Scale columns by w
        Z = X.multiply(w)  # each column k scaled by w[k]
        # Now scale rows by 1 / Xw[i].  Make a diagonal matrix of size N, 
        # then multiply from the left.
        invXw = 1.0 / Xw
        # sparse.diags(...) yields an N x N diagonal matrix. Then sparse.diags(invXw).dot(Z)
        # scales row i by invXw[i].
        Z = sparse.diags(invXw).dot(Z)
        return Z
    else:
        # Dense broadcast
        return (X * w[np.newaxis, :]) / Xw[:, np.newaxis]

def compute_D(X, logX, logZ, logy):
    """
    Compute 
      D[k] = sum_i X_{i,k} [logX_{i,k} - logZ_{i,k} - logy_i],
    in a vectorized manner for both dense and sparse X.
    We'll do it as:
      D_part1 = sum_i X_{i,k} [logX_{i,k} - logZ_{i,k}]
      D_part2 = sum_i X_{i,k} logy_i

    Then  D = D_part1 - D_part2
    """
    # -- Part 1 --
    # X.multiply(...) is elementwise for sparse. For dense, it's just (X * ...)
    # Summation over axis=0 yields shape(M,)
    if sparse.issparse(X):
        part1 = safe_sum(X.multiply(logX - logZ), axis=0)
    else:
        part1 = safe_sum(X * (logX - logZ), axis=0)
    
    # -- Part 2 --
    # sum_i X_{i,k} logy_i == the (k)-th entry of X[:,k]^T dot logy
    # => we can do X^T dot logy once and get shape(M,)
    part2 = X.T.dot(logy)  # works for dense or sparse
    # part2 is shape (M,) if logy is shape(N,)

    D = part1 - part2
    return D

def L_L2(w, X, y, eps, alpha):
    """
    Compute the objective L_L2(w) = 
        eps * sum_i [ Xw[i]*ln(Xw[i]/y[i]) - Xw[i] + y[i] ]  +  alpha/2 * ||w||^2
    for both dense and sparse X. 
    """
    Xw = X.dot(w)
    # KL part: sum_i [Xw_i ln(Xw_i / y_i) - Xw_i + y_i ]
    # Safeguard if needed, assuming Xw, y > 0
    ratio = Xw / y
    kl_part = np.sum(Xw * np.log(ratio + 1e-300) - Xw + y)
    # L2 part
    l2_part = 0.5 * alpha * np.sum(w**2)
    return eps * kl_part + l2_part

##############################################################################
#                           Main MM Solver                                   #
##############################################################################

def prox_kl_nnsx_l2_mm_jensen_sp(X, y, ε, α, w_init=None, max_iter=100, tol=1e-7, verbose=False):
    """
    Vectorized MM solver for the problem:

        min_{w >= 0}  L_L2(w) = eps * sum_i KL( (Xw)[i] || y[i] )  +  (alpha/2)*||w||^2

    where KL(a||b) = a ln(a/b) - a + b, and X, y, w >= 0.

    Uses Majorization–Minimization with closed-form Lambert W updates. Works for 
    both dense np.array X and scipy.sparse X, avoiding explicit for-loops over rows/cols.

    Parameters
    ----------
    X : ndarray or scipy.sparse of shape (N, M), nonnegative
    y : ndarray of shape (N,), positive
    ε : float > 0
    α : float > 0
    w_init : ndarray of shape (M,) or None
        If None, initialize with ones(M).
    max_iter : int
        Maximum number of MM iterations.
    tol : float
        Stopping tolerance based on L2-norm of w-updates.

    Returns
    -------
    w : ndarray of shape (M,)
        Nonnegative solution.
    """
    # --- 1. Initialization ---
    N, M = X.shape
    if w_init is None:
        w = np.ones(M, dtype=float)
    else:
        w = np.array(w_init, dtype=float)

    # Precompute column sums: S_k = sum_i X_{i,k} (assumed > 0)
    # For sparse X, .sum(axis=0) -> shape (1, M) so we flatten it
    S = safe_sum(X, axis=0)  # shape (M,)

    # Precompute log(y)
    logy = np.log(y + 1e-300)  # shape (N,)

    for t in range(max_iter):
        # Current objective
        if verbose:
            print(f"MM Iter = {t}, ||w||={np.linalg.norm(w):.4g}")

        # 2a) Compute Xw
        Xw = X.dot(w)  # shape (N,)

        # 2b) Form Z_{i,j} = [X_{i,j} * w_j] / Xw_i
        Z = compute_Z(X, w, Xw)

        # 2c) Precompute logX and logZ
        logX = safe_log(X)        # shape (N,M) or sparse
        logZ = safe_log(Z)        # shape (N,M) or sparse

        # 2d) Compute D_k = sum_i X_{i,k} [logX_{i,k} - logZ_{i,k} - logy_i ]
        D = compute_D(X, logX, logZ, logy)  # shape (M,)

        # 2e) Solve w_k via LambertW:
        #     w_k = (eps * S_k / alpha) * W( (alpha/(eps*S_k)) * exp(-D_k / S_k ) )
        # Safeguard S_k to avoid /0 if some column has sum=0
        ratio = -D / S
        a = np.exp(ratio)                 # shape (M,)
        b = α / (ε * S)            # shape (M,)
        ba = b * a                        # shape (M,)

        w_update = (ε * S / α) * lambertw(ba)
        w_new = np.real(w_update)  # just in case lambertw returns complex with 0 imag

        # Enforce nonnegativity
        w_new[w_new < 0] = 0.0

        # 2f) Check convergence
        delta_norm = np.linalg.norm(w_new - w)
        w = w_new
        if delta_norm < tol:
            break

    # Final objective
    if verbose:
        print(f"MM Iter = {t+1}, ||w||={np.linalg.norm(w):.4g}")
    return w




##################################################################################  
##################################################################################  
##
##
## Numerical solutions for the Prox KL NNSX problems
##
##
##################################################################################
##################################################################################  

def kl_nnsx_objective(w, X, t, ε):
    """
    Compute the objective:
        L(w) = ε * sum( [ (Xw)_i log((Xw)_i / t_i) - (Xw)_i + t_i ] )
    where w >= 0.

    Parameters
    ----------
    w : np.ndarray, shape (M,)
        Current solution vector (nonnegative).
    X : np.ndarray, shape (N, M)
        Nonnegative matrix.
    t : np.ndarray, shape (N,)
        Nonnegative vector (target).
    ε : float
        Positive scalar controlling the KL term.

    Returns
    -------
    float
        The value of the objective.
    """
    # Compute y = Xw
    y = X.dot(w)

    # To avoid log(0), we clip y to a small positive number
    #   but in practice ensure w, x, and X are nonnegative
    #   and Xw won't vanish if w is not strictly zero.
    y = np.clip(y, 1e-15, None)

    # Compute the KL sum
    # KL(y|t) = sum_i [ y_i log(y_i / t_i) - y_i + t_i ]
    # For numerical safety, also clip t
    t_clipped = np.clip(t, 1e-15, None)
    kl_vec = y * np.log(y / t_clipped) - y + t_clipped

    return ε * np.sum(kl_vec)

def prox_kl_nnsx_direct(X, t, ε, max_iter=1000, w_init=None):
    """
    Solve the Prox KL NNSX problem directly using scipy.optimize.minimize
    with numerical gradients. We'll enforce w >= 0 by specifying bounds,
    and add a callback to print each iteration's objective and w.

    Parameters
    ----------
    X : np.ndarray, shape (N, M)
        Nonnegative matrix.
    t : np.ndarray, shape (N,)
        Nonnegative vector.
    ε : float
        KL weight parameter.
    max_iter : int
        Maximum number of iterations for the minimize routine.
    w_init : np.ndarray, shape (M,)
        Initial solution vector (nonnegative).

    Returns
    -------
    w : np.ndarray, shape (M,)
        The final solution vector (nonnegative).
    """
    M = X.shape[1]

    def callback_fn(wk):
        # This callback prints the objective and w at each iteration
        obj_val = kl_nnsx_objective(wk, X, t, ε)
        print(f"[prox_kl_nnsx_direct] obj={obj_val:.6e}, w={astr(wk)}")

    # Initial guess
    w_init = w_init if w_init is not None else np.ones(M) / M

    # We create bounds to enforce w_i >= 0
    bounds = [(0, None)] * M
    
    # Minimize with numerical gradient (default 'BFGS' doesn't allow bounds easily).
    # We'll use 'SLSQP' or 'trust-constr'. 'SLSQP' is simpler for demonstration.
    res = minimize(lambda w: kl_nnsx_objective(w, X, t, ε),
                   w_init,
                   method='SLSQP',
                   tol=1e-8,
                   bounds=bounds,
                   callback=callback_fn,
                   options={'maxiter': max_iter, 'disp': False})

    return res.x


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
# 3) Direct Approach with scipy.optimize.minimize
#------------------------------------------------------------------------------

def prox_kl_nnsx_l1_direct(X, y, ε, α, w_init=None, max_iter=100, tol=1e-6, verbose=True):
    """
    Solve the same problem directly using scipy.optimize.minimize with
    numerical gradient (no explicit jac). We enforce w >= 0 using bounds.
    
    Args:
        X, y, , alpha: same meaning as above
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
        return L_L1_objective(w, X, y, ε, α)

    # We'll store iteration info in a closure variable
    iteration_count = [0]  # list so we can modify inside callback
    
    def callback(wk):
        iteration_count[0] += 1
        val = fun(wk)
        if verbose:
            print(f"[Direct] Iter {iteration_count[0]}: w = {wk}, L_L1 = {val:.6f}")
    
    # Enforce w >= 0 via bounds
    bounds = [(0.0, None)] * M

    w_init = w_init if w_init is not None else np.ones(M) / M
    
    # Use 'SLSQP' or 'trust-constr' or similar. SLSQP often works well with bounds.
    res = minimize(fun, w_init, method='SLSQP', bounds=bounds, 
                   callback=callback, options={'maxiter': max_iter, 'ftol': tol})
    
    w_opt = res.x
    # Final print
    if verbose:
        print(f"[Direct] Final: w = {w_opt}, L_L1 = {fun(w_opt):.6f}, success={res.success}")
    
    return w_opt


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



def prox_kl_nnsx_l2_direct(X, y, ε, α, w_init=None, max_iter=20):
    """
    Direct approach using scipy.optimize.minimize (with numerical gradients).
    Prints w and L_L2(w) at each step of the optimizer's outer iteration.
    
    Parameters:
    -----------
    X : array (N, M)
    y : array (N,)
    ε : float
    α : float
    w_init : initial guess (M,) or None
    max_iter : max iteration count for 'SLSQP'
    
    Returns:
    --------
    w : final solution
    """
    N, M = X.shape
    if w_init is None:
        w_init = np.ones(M, dtype=float)
    else:
        w_init = np.array(w0, dtype=float)
    
    # Bounds: w_j >= 0
    bnds = [(0, None) for _ in range(M)]
    
    # We define a callback to print iteration info
    def callback_func(w_candidate):
        val = L_L2(w_candidate, X, y, ε, α)
        print(f"Minimize step: w = {w_candidate}, L_L2(w) = {val:.6f}")
    
    # Run optimizer
    res = minimize(
        fun=lambda w_var: L_L2(w_var, X, y, ε, α),
        x0=w_init,
        method='SLSQP',
        bounds=bnds,
        callback=callback_func,   # to print progress
        options={'maxiter': max_iter, 'disp': False}
    )
    
    # Final
    w_final = res.x
    print(f"Direct solution finished. w = {w_final}, L_L2(w) = {L_L2(w_final, X, y, ε, α):.6f}")
    return w_final

def L_ElasticNet(w, X, y, ε, α):
    """
    The Elastic Net–penalized KL objective:
        L_ElasticNet(w) = ε * sum_i KL( sum_j X[i,j]*w[j],  y[i] )
                          + α * ||w||_1
                          + (1 - alpha) * 0.5 * ||w||_2^2
    
    Inputs
    ------
    w       : (M,) nonnegative parameter vector
    X       : (N, M) nonnegative data matrix
    y       : (N,) nonnegative target vector
    epsilon : weight on the KL term
    alpha   : trade-off parameter in [0, 1]
    
    Output
    ------
    scalar float objective value
    """
    # Compute the predicted sum_i = Xw
    s = X.dot(w)
    
    # KL part
    kl_sum = np.sum([kl_div(s[i], y[i]) for i in range(len(y))])
    
    # L1 part
    l1_part = np.sum(w)
    
    # L2 part
    l2_part = 0.5 * np.sum(w**2)
    
    return ε * kl_sum + α * l1_part + (1 - α) * l2_part
