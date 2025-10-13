from scipy import sparse
from .prox import *


class OTLM():

    def __init__(self, datafit='quadratic', penalty='l2', reg_type='kl', λ=1., ε=1., α=1e-6, β=1, max_iter=100, max_iter_mm=100, tol=1e-10, options={}):

        self.datafit = datafit.lower()
        self.penalty = penalty.lower()
        self.reg_type = reg_type.lower()
        self.α = α
        self.β = β
        self.ε = ε
        self.λ = λ
        self.max_iter = max_iter
        self.max_iter_mm = max_iter_mm
        self.tol = tol
        self.options = options
        self.options.setdefault('disp', False)
        

    def get_gibbs_kernel(self, C, y):

        if sparse.issparse(C):
            K = C.copy()
            np.exp( -K.data/self.ε, out=K.data )
           
            if self.reg_type == 'kl':
                r, c = K.row.astype(np.int64), K.col.astype(np.int64)
                yyt = sparse.coo_array((y[r] * y[c] , (r, c)), shape=K.shape)
                K = K.multiply(yyt)
            
            K = K.tocoo()

        else:
            if self.reg_type == 'entropy':
                K =  np.exp(-C/self.ε) 
            elif self.reg_type == 'kl':
                K =  np.exp(-C/self.ε) * np.outer(y, y)
       
        return K

    def fit(self, C, X, y, f=None, γ=None, D=None):

        y = y.astype(np.float64)
        n_samples, n_features = X.shape
        K = self.get_gibbs_kernel(C, y)
        
        u1 = np.ones(n_samples)
        u2 = np.ones(n_samples)

        u1_prev = u1.copy()
        u2_prev = u2.copy()
                
        # init w
        w = np.ones(n_features)

        if self.options['disp']:
            print_arr('w', w)
            print_arr('K', K)

        w_prev = w.copy()

        for i in range(self.max_iter):

            q1 = K.dot(u1)
            u2 = self.scaling_target(y, q1, **self.options)
            q2 = K.T.dot(u2)
            u1, w = self.scaling_source(X, q2, w0=w, D=D, f=f, γ=γ, **self.options)

            diff = max(np.linalg.norm(u1-u1_prev), np.linalg.norm(u2-u2_prev))
            diff_w = np.linalg.norm(w-w_prev)
            # if (diff < self.tol):
            if (diff_w < self.tol):
                break

            u1_prev = u1
            u2_prev = u2
            w_prev = w
            
            if self.options['disp']:
                if i%1 == 0:
                    print(f'OTLM iter {i:> 4d}, diff: {diff:.2e}, diff_w: {diff_w:.2e}, tol: {self.tol:.2e}')


        self.w = self.coef_ = w
        self.n_iter_ = i

        # calculate plan

        if sparse.issparse(K):
            r, c = K.row.astype(np.int64), K.col.astype(np.int64)
            U = sparse.coo_array((u1[r] * u2[c] , (r, c)), shape=K.shape)
            Q = K.multiply(U)
        else:
            Q = K * np.outer(u1, u2)

        self.Q = Q
        self.trans_cost_ = np.sum(C*Q)
        self.total_loss_ = self.total_loss(C, Q, y, X, w)

        if self.options['disp']:
            print(f'OTLM fit finished in {i} iterations, diff: {diff:.2e} tol: {self.tol:.2e}, diff_w: {diff_w:.2e}')

    def predict(self, X, target=False):

        if target:
            y = self.Q.sum(axis=0)
        else:
            y = X.dot(self.w)

        return y

    def total_loss(self, C, Q, y, X, w):

        # transport cost
        if sparse.issparse(C):
            total_loss = np.sum(C.multiply(Q))
        else:
            total_loss = np.sum(C*Q) 

        # entropic penalty
        if sparse.issparse(Q):
            total_loss += self.ε * np.sum( Q.multiply(safelog(Q)) ) 
        else:
            total_loss += self.ε * np.sum( Q * np.where(Q>0, safelog(Q), 0) ) 

        # feature penalty
        if self.penalty == 'l1':
            total_loss += self.α * np.sum(np.abs(w))

        elif self.penalty == 'l2':
            total_loss += self.α * np.sum(np.abs(w)**2)

        # datafit loss
        if self.datafit == 'kl':
            yp = X.dot(w)
            total_loss += self.λ * np.sum(yp * (safelog(yp) - safelog(y)))

        if self.datafit == 'tv':
            yp = X.dot(w)
            total_loss += self.λ * np.sum(np.abs(y-yp))

        if self.datafit == 'quadratic':
            yp = X.dot(w)
            total_loss += self.λ * np.sum((y-yp)**2)

        if self.datafit == 'poisson':
            yp = X.dot(w)
            total_loss += self.λ * np.sum((yp-y*np.log(yp)))
            

        return total_loss


    def scaling_target(self, y, k, **options):

        if self.datafit == 'fix':
            s = proxdiv_fix(y, k, **options)

        elif self.datafit == 'kl':
            s = proxdiv_kl(y, k, self.ε, self.λ, **options)
            
        elif self.datafit == 'tv':
            s = proxdiv_tv(y, k, self.ε, self.λ, **options)
            
        elif self.datafit == 'box':
            s = proxdiv_box(y, k, self.lb, self.ub, **options)
            
        elif self.datafit == 'quadratic':
            s = proxdiv_quadratic(y, k, self.λ, self.ε, **options)
            
        elif self.datafit == 'poisson':
            s = proxdiv_poisson(y, k, self.λ, self.ε, **options)

        return s

    def scaling_source(self, X, k, w0=None, D=None, f=None, γ=None, **options):

        if self.penalty == 'no':
            s, w = proxdiv_kl_nnsx(X, k, self.ε, max_iter=self.max_iter_mm, tol=self.tol, w0=w0, **options)

        elif self.penalty == 'l1':
            s, w = proxdiv_kl_nnsx_l1(X, k, self.ε, self.α, max_iter=self.max_iter_mm, tol=self.tol, w0=w0, **options)
     
        elif self.penalty == 'l2':
            s, w = proxdiv_kl_nnsx_l2(X, k, self.ε, self.α, max_iter=self.max_iter_mm, tol=self.tol, w0=w0, **options)
        
        return s, w


def safelog(x):

    if sparse.issparse(x):
        log_x = x.copy()
        log_x.data = np.log(log_x.data, where=log_x.data>0, out=np.zeros_like(log_x.data))
    else:
        log_x = np.log(x, where=x>0, out=np.zeros_like(x))
    
    return log_x