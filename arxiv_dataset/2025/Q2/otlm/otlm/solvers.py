import time
import numpy as np
import scipy.sparse as sparse
import cvxpy as cp
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
from scipy.optimize import minimize

class OTLM_L2L2_QP():

    def __init__(self, alpha, lam, solver=cp.CLARABEL):

        self.alpha = alpha
        self.lam = lam
        self.solver = solver

    def fit(self, C, X, y):

        if sparse.issparse(C):
            self.fit_sparse(C, X, y)
        else:
            self.fit_dense(C, X, y)

    def fit_sparse(self, C, X, y):
        """
        Solve the quadratic program (in flattened form):
        
            minimize_{Q >= 0, w >= 0}   sum_{i,j} C[i,j]*Q[i,j]
                                    + (lam/2)*||Q^T * 1 - y||^2
                                    + (alpha/2)*||w||^2
            subject to:  Q * 1 = X * w,
                        Q >= 0,  w >= 0.
        
        using the qpsolvers.solve_qp interface with solver='clarabel'.

        Parameters
        ----------
        X : scipy.sparse.spmatrix, shape (N, M)
            Nonnegative design matrix (sparse).
        y : np.ndarray, shape (N,)
            Target vector (dense).
        C : scipy.sparse.spmatrix, shape (N, N)
            Cost matrix (sparse).
        alpha : float
            Regularization parameter (for ||w||^2).
        lam : float
            Regularization parameter (for ||Q^T 1 - y||^2).

        Returns
        -------
        Q_opt : scipy.sparse.csr_matrix, shape (N, N)
            Optimal Q (nonnegative).
        w_opt : np.ndarray, shape (M,)
            Optimal w (nonnegative).
        """

        C_csr = sparse.csr_array(C)
        C_coo = sparse.coo_array(C_csr)

        self.ns = X.shape[0]
        self.np = X.shape[1]
        self.nc = C.shape[0]

        # create LP arrays

        cols12 = np.sort(C_coo.reshape(-1,1).nonzero()[0])
        print('creating LP arrays, number of nonzeros:', len(cols12))

        # marginals for OT
        r, c, d = C_coo.row.astype(np.int64), C_coo.col.astype(np.int64), C_coo.data
        A_eq_11 = sparse.coo_array((np.ones(len(c)), (r, c+r*self.ns)), shape=(self.ns, self.ns**2))
        A_eq_21 = sparse.coo_array((np.ones(len(c)), (r+c*self.ns, c)), shape=(self.ns**2, self.ns)).reshape(self.ns, self.ns**2)
        c1, c2 = np.sort(A_eq_11.nonzero()[1]), np.sort(A_eq_21.nonzero()[1])
        assert np.all(c1==c2), 'error in creating A_eq array'
        assert np.all(cols12==c2), 'error in creating A_eq array'

        Hr = sparse.coo_array(sparse.csc_array(A_eq_11)[:,cols12])
        Hc = sparse.coo_array(sparse.csc_array(A_eq_21)[:,cols12])
        print('Hr.shape:', Hr.shape)
        print('Hc.shape:', Hc.shape)

        q = cp.Variable(len(cols12), nonneg=True)
        w = cp.Variable(self.np, nonneg=True)

        c_transport_cost = np.array(sparse.csr_array(C_coo.reshape(-1,1))[cols12].todense()).ravel()
        transport_cost_term = cp.sum(cp.multiply(c_transport_cost, q))            # sum_{i,j} C_ij * Q_ij
        datafit_term = (self.lam / 2) * cp.sum_squares(Hr @ q - y)  # (lambda/2)*||Q^T 1 - y||^2
        penalty_term = (self.alpha / 2) * cp.sum_squares(w)     # (alpha/2)*||w||^2
        constraints = [Hc @ q == X @ w]
        objective = cp.Minimize(transport_cost_term + datafit_term + penalty_term)
        prob = cp.Problem(objective, constraints)
    
        # run QP

        time_start = time.time()
        result = prob.solve(solver=self.solver)
        time_end = time.time()
        self.time = time_end - time_start

        # unpack
        Q = sparse.coo_array((q.value[:len(cols12)], (r, c)), shape=C.shape)
        w = w.value

        # coefs
        self.coef_ = w
        self.intercept_ = 0.0

        # transport plan
        self.transport_plan = Q


    def fit_dense(self, C, X, y):

        """
        Solve the quadratic program:
            minimize_{Q >= 0, w >= 0}   sum_{i,j} C[i,j]*Q[i,j]
                                        + (lambda/2)*||Q^T 1 - y||^2
                                        + (alpha/2)*||w||^2
            subject to: Q * 1 = X * w
                        Q >= 0, w >= 0

        Parameters
        ----------
        X : np.ndarray, shape (N, M)
            Nonnegative design matrix.
        y : np.ndarray, shape (N,)
            Target vector.
        C : np.ndarray, shape (N, N)
            Cost matrix, nonnegative.
        alpha : float
            Regularization parameter for the w-term.
        lam : float
            Regularization parameter for the (Q^T 1 - y) term.
        solver : str, optional
            CVXPY solver name, by default `cp.OSQP` (you can also use 'SCS', 'ECOS', 'GUROBI', etc.)

        Returns
        -------
        Q_opt : np.ndarray, shape (N, N)
            The optimized nonnegative matrix Q.
        w_opt : np.ndarray, shape (M,)
            The optimized nonnegative vector w.
        """

        # Dimensions
        N, M = X.shape

        # Define optimization variables in CVXPY
        # nonneg=True enforces elementwise non-negativity
        Q = cp.Variable((N, N), nonneg=True)
        w = cp.Variable(M, nonneg=True)

        # Define the objective
        ones_N = np.ones(N)  # Vector of ones of length N
        cost_term = cp.sum(cp.multiply(C, Q))            # sum_{i,j} C_ij * Q_ij
        reg_Q_term = (self.lam / 2) * cp.sum_squares(Q.T @ ones_N - y)  # (lambda/2)*||Q^T 1 - y||^2
        reg_w_term = (self.alpha / 2) * cp.sum_squares(w)     # (alpha/2)*||w||^2

        objective = cp.Minimize(cost_term + reg_Q_term + reg_w_term)

        # Constraints: Q * 1 = X * w, plus nonnegativity
        constraints = [Q @ ones_N == X @ w]
        

        # Formulate the problem
        prob = cp.Problem(objective, constraints)

        # Solve the QP
        time_start = time.time()
        prob.solve()
        time_end = time.time()
        self.time_elapsed = time_end - time_start

        # Extract the solutions
        Q_opt = Q.value
        w_opt = w.value

        self.coef_ = w_opt
        self.intercept_ = 0.0
        self.transport_plan = Q_opt


    def predict(self, X):

        return X.dot(self.coef_)



def solve_lp_with_cvxpy(c_vec, A, b_eq):
    """
    Solve the LP:
        minimize    c_vec^T x
        subject to  A x = b_eq
                    x >= 0
    using cvxpy.
    """

    import cvxpy as cp

    # Number of variables is the length of c_vec
    n = len(c_vec)

    # Define the variable x, constrained to be nonnegative
    x = cp.Variable(shape=(n,), nonneg=True)

    # Define the objective: Minimize c_vec^T x
    # Note: If 'c_vec' is a sparse array, ensure it's converted to a form cvxpy can handle
    objective = cp.Minimize(c_vec @ x)

    # Define the constraints: A x = b_eq
    # (cvxpy can handle sparse and dense, but you might need to ensure the shapes match)
    constraints = [A @ x == b_eq]

    # Form and solve the problem
    problem = cp.Problem(objective, constraints)
    result = problem.solve(verbose=True)  # By default, uses an appropriate solver like ECOS, OSQP, or GLPK

    return x.value






class OTLM_L1L1_LM():


    def __init__(self, alpha=1, lam=1):

        self.alpha = alpha
        self.lam = lam

    def fit(self, C, X, y):

        from scipy.optimize import linprog
        from scipy import sparse


        C_csr = sparse.csr_array(C)
        C_coo = sparse.coo_array(C_csr)

        self.ns = X.shape[0]
        self.np = X.shape[1]
        self.nc = C.shape[0]

        # create LP arrays

        cols12 = np.sort(C_coo.reshape(-1,1).nonzero()[0])

        # marginals for OT
        r, c, d = C_coo.row.astype(np.int64), C_coo.col.astype(np.int64), C_coo.data
        A_eq_11 = sparse.coo_array((np.ones(len(c)), (r, c+r*self.ns)), shape=(self.ns, self.ns**2))
        A_eq_21 = sparse.coo_array((np.ones(len(c)), (r+c*self.ns, c)), shape=(self.ns**2, self.ns)).reshape(self.ns, self.ns**2)
        c1, c2 = np.sort(A_eq_11.nonzero()[1]), np.sort(A_eq_21.nonzero()[1])
        assert np.all(c1==c2), 'error in creating LP A_eq array'
        assert np.all(cols12==c2), 'error in creating LP A_eq array'

        A_eq_11 = sparse.coo_array(sparse.csc_array(A_eq_11)[:,cols12])
        A_eq_21 = sparse.coo_array(sparse.csc_array(A_eq_21)[:,cols12])

        # slack variables for TV loss
        A_eq_12 = sparse.eye(self.ns, format='coo')
        A_eq_22 = sparse.coo_array(([],([],[])), shape=A_eq_12.shape)
        A_eq_13 = sparse.eye(self.ns, format='coo') * -1
        A_eq_23 = sparse.coo_array(([],([],[])), shape=A_eq_13.shape)

        # basis
        A_eq_24 = sparse.coo_array(X) * -1
        A_eq_14 = sparse.coo_array(([],([],[])), shape=X.shape)

        # combine tiles
        A_eq_1 = sparse.hstack((A_eq_11, A_eq_12, A_eq_13, A_eq_14))
        A_eq_2 = sparse.hstack((A_eq_21, A_eq_22, A_eq_23, A_eq_24))
        A_eq = sparse.vstack((A_eq_1, A_eq_2))
        b_eq = np.concatenate([y, np.zeros(X.shape[0])])

        # cost
        c_transport_cost = np.array(sparse.csr_array(C_coo.reshape(-1,1))[cols12].todense()).ravel()
        c_marginal_cost = np.full(X.shape[0], self.lam)
        c_feature_cost = np.full(X.shape[1], self.alpha)
        c_vec = np.concatenate([c_transport_cost, c_marginal_cost, c_marginal_cost, c_feature_cost])
    
        # run LP

        time_start = time.time()
        result = linprog(c=c_vec, A_eq=sparse.csc_array(A_eq), b_eq=b_eq, bounds=(0, None), method='highs', options={'disp':True})
        x = result.x
        # x = solve_lp_with_cvxpy(c_vec, sparse.csc_array(A_eq), b_eq)
        time_end = time.time()
        self.time = time_end - time_start

        # unpack
        Q = sparse.coo_array((x[:len(cols12)], (r, c)), shape=C.shape)
        w = x[-X.shape[1]:]

        # coefs
        self.coef_ = w
        self.intercept_ = 0.0

        # transport plan
        self.transport_plan = Q


    def predict(self, X):

        return X.dot(self.coef_)




class OTLM_L1_LM():

    def __init__(self, alpha=1):

        self.alpha = alpha

    def fit(self, C, X, y):
        """
        Solve the LP:
            min_{Q, w}    sum_{i,j} C[i,j] * Q[i,j] + alpha * sum_m w[m]
            subject to:
                Q >= 0,
                w >= 0,
                Q 1 = X w,     (row sums)
                Q^T 1 = y.     (column sums)

        Parameters
        ----------
        C : scipy.sparse.spmatrix, shape (N, N)
            The cost matrix. Must be nonnegative.
        X : scipy.sparse.spmatrix, shape (N, M)
            The feature/loading matrix. Must be nonnegative.
        y : np.ndarray, shape (N,)
            The right-hand side for the column-sum constraints. Must be nonnegative.
        alpha : float
            The regularization coefficient for the L1 norm of w.

        Returns
        -------
        Q_sparse : scipy.sparse.csr_matrix, shape (N, N)
            Optimal transport matrix Q in sparse format.
        w : np.ndarray, shape (M,)
            Optimal weight vector w in dense format.
        """

        # ---------------------------
        # 1. Gather dimensions
        # ---------------------------
        N = X.shape[0]
        M = X.shape[1]
        if C.shape != (N, N):
            raise ValueError(f"C must be of shape ({N}, {N})")
        if y.shape[0] != N:
            raise ValueError(f"y must be of shape ({N},), got {y.shape}")

        # ---------------------------
        # 2. Build the objective c
        # ---------------------------
        # We'll create a dense vector of length (N*N + M).
        # The first N*N entries correspond to Q_{ij}, the last M entries to w_m.
        c = np.zeros(N*N + M, dtype=float)

        # Fill in the cost of Q from the sparse matrix C
        # Indexing convention: Q_{i,j} maps to index idx = i*N + j
        C_coo = C.tocoo(copy=False)
        for (i, j, val) in zip(C_coo.row, C_coo.col, C_coo.data):
            c[i*N + j] = val

        # Fill in the cost for w (alpha * sum_m w_m)
        c[N*N : N*N + M] = self.alpha

        # ---------------------------
        # 3. Build the equality constraints A_eq x = b_eq
        #    We have 2N constraints:
        #      - N row-sum constraints: sum_j Q_{i,j} = sum_m X_{i,m} w_m
        #      - N col-sum constraints: sum_i Q_{i,j} = y_j
        # ---------------------------
        # The matrix A_eq has shape (2N) x (N*N + M).
        # We'll construct it in sparse COO form.

        row_idx = []
        col_idx = []
        data_vals = []

        b_eq = np.zeros(2*N, dtype=float)

        # ---- (a) Row-sum constraints: for i in 0..N-1
        # sum_j Q_{i,j} - sum_m X_{i,m} w_m = 0
        # That is: for each row i,
        #    Q_{i,0} + Q_{i,1} + ... + Q_{i,N-1} - X[i,0]*w_0 - ... - X[i,M-1]*w_{M-1} = 0
        #
        # We'll add +1 in Q block, -X[i,m] in w block.

        X_csr = X.tocsr(copy=False)
        for i in range(N):
            # For Q_{i,j}, j=0..N-1, place +1
            for j in range(N):
                row_idx.append(i)
                col_idx.append(i*N + j)
                data_vals.append(1.0)
            # For w_m, place -X[i,m] only if X[i,m] != 0
            row_start = X_csr.indptr[i]
            row_end = X_csr.indptr[i+1]
            for idx in range(row_start, row_end):
                m = X_csr.indices[idx]
                val = X_csr.data[idx]
                # Coefficient for w_m is -val
                row_idx.append(i)
                col_idx.append(N*N + m)
                data_vals.append(-val)
            # b_eq[i] = 0 (already set by default)

        # ---- (b) Column-sum constraints: for j in 0..N-1
        # sum_i Q_{i,j} = y_j
        #
        # We'll add +1 in Q block for each i, and 0 in w block.

        for j in range(N):
            row_number = N + j  # the row index in A_eq
            # sum_i Q_{i,j} = y_j
            for i in range(N):
                row_idx.append(row_number)
                col_idx.append(i*N + j)
                data_vals.append(1.0)
            b_eq[row_number] = y[j]

        # Build A_eq in COO format
        A_eq = coo_matrix(
            (data_vals, (row_idx, col_idx)),
            shape=(2*N, N*N + M)
        )

        # ---------------------------
        # 4. Nonnegativity bounds
        # ---------------------------
        # Q_{ij} >= 0, w_m >= 0
        # -> bounds = (0, None) for all variables
        bounds = [(0, None)] * (N*N + M)

        # ---------------------------
        # 5. Solve the LP
        # ---------------------------
        time_start = time.time()
        res = linprog(
            c,
            A_ub=None,
            b_ub=None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        time_end = time.time()
        self.time_elapsed = time_end - time_start

        if not res.success:
            raise RuntimeError(f"LinProg did not converge: {res.message}")

        # ---------------------------
        # 6. Extract Q and w from solution
        # ---------------------------
        solution = res.x
        Q_flat = solution[:N*N]  # first N*N entries
        w = solution[N*N:]       # last M entries

        # Convert Q_flat into an (N x N) sparse matrix (COO, then CSR)
        # Index i*N + j -> row i, col j
        row_indices = np.arange(N*N) // N
        col_indices = np.arange(N*N) % N
        Q_sparse = coo_matrix((Q_flat, (row_indices, col_indices)), shape=(N, N)).tocsr()

        self.coef_ = w
        self.intercept_ = 0.0
        self.transport_plan = Q_sparse


    def predict(self, X):

        return X.dot(self.coef_)

def solve_lp_with_cvxpy(c_vec, A, b_eq):
    """
    Solve the LP:
        minimize    c_vec^T x
        subject to  A x = b_eq
                    x >= 0
    using cvxpy.
    """

    import cvxpy as cp

    # Number of variables is the length of c_vec
    n = len(c_vec)

    # Define the variable x, constrained to be nonnegative
    x = cp.Variable(shape=(n,), nonneg=True)

    # Define the objective: Minimize c_vec^T x
    # Note: If 'c_vec' is a sparse array, ensure it's converted to a form cvxpy can handle
    objective = cp.Minimize(c_vec @ x)

    # Define the constraints: A x = b_eq
    # (cvxpy can handle sparse and dense, but you might need to ensure the shapes match)
    constraints = [A @ x == b_eq]

    # Form and solve the problem
    problem = cp.Problem(objective, constraints)
    result = problem.solve(verbose=True)  # By default, uses an appropriate solver like ECOS, OSQP, or GLPK
    result.x = x.value

    return x.value, result



def poisson_regression_no_log_link(X, y, alpha=1.0):
    """
    Fits a Poisson regression model of the form:
        mu_i = X_i^T w
    using L2 regularization on w and Poisson negative log-likelihood loss:

        L(w) = sum_{i=1 to n} [ mu_i - y_i * log(mu_i) ]
               + (alpha/2) * ||w||^2

    The optimization is done via scipy.optimize.minimize with 
    numerical gradient estimation (jac=None).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix.
    y : array-like of shape (n_samples,)
        Target values (must be nonnegative integers for Poisson).
    alpha : float, default=1.0
        Regularization strength for the L2 penalty.

    Returns
    -------
    w_opt : ndarray of shape (n_features,)
        The fitted weight vector.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    # Define the objective function: negative log-likelihood + L2 penalty
    def objective(w):
        # linear predictions
        mu = X.dot(w)
        # clip predictions to avoid log(0) or negative values
        mu_clipped = np.clip(mu, 1e-12, None)

        # Poisson negative log-likelihood (ignoring constants like log(y!))
        nll = np.sum(mu_clipped - y * np.log(mu_clipped))

        # L2 penalty
        penalty = 0.5 * alpha * np.sum(w**2)

        return nll + penalty

    # Initialize weights to zero (or small random values)
    w0 = np.zeros(n_features, dtype=float)

    # Use scipy.optimize.minimize with default numerical gradient
    result = minimize(objective, w0, method="BFGS", jac=None)

    # Fitted parameter vector
    w_opt = result.x

    return w_opt

class PoissonRegressorNoLogLink():

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):

        self.coef_ = poisson_regression_no_log_link(X, y, alpha=self.alpha)

    def predict(self, X):

        return X.dot(self.coef_)

    
class TVRegressor():

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):

        from sklearn.linear_model import QuantileRegressor

        qr = QuantileRegressor(quantile=0.5, alpha=self.alpha, fit_intercept=False)
        qr.fit(X, y)
        self.coef_ = qr.coef_
        self.intercept_ = 0

    def predict(self, X):

        return X.dot(self.coef_)


