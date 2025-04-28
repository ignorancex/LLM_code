"""
@author: Ziad (zi.hatab@gmail.com)

Implementation of the multinetwork method to compute propagation constant 
from measurements of offsetted networks. Based on the paper:

Z. Hatab, A. Abdi, G. Steinbauer, M. E. Gadringer, and W. Bösch, 
"Propagation Constant Measurement Based on a Single Transmission Line Standard Using a Two-port VNA," 
2023, e-print: <https://arxiv.org/abs/2302.13859>.
"""

# python -m pip install numpy -U
import numpy as np 

# constants
c0 = 299792458
P = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]])

def s2t(S, pseudo=False):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T if pseudo else T/S[1,0]

def t2s(T, pseudo=False):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S if pseudo else S/T[1,1]

def compute_G_with_takagi(A):
    # implementation of Takagi decomposition to compute the matrix G used to determine the weighting matrix.
    # Singular value decomposition for the Takagi factorization of symmetric matrices
    # https://www.sciencedirect.com/science/article/pii/S0096300314002239
    u,s,vh = np.linalg.svd(A)
    u,s,vh = u[:,:2],s[:2],vh[:2,:]  # low-rank truncated (Eckart-Young-Mirsky theorem)
    phi = np.sqrt( s*np.diag(vh@u.conj()) )
    G = u@np.diag(phi)
    lambd = s[0]*s[1]  # this is the eigenvalue of the weighted eigenvalue problem (squared Frobenius norm of W)
    return G, lambd

def WLS(x,y,w=1):
    # Weighted least-squares for a single parameter estimation
    x = x*(1+0j) # force x to be complex type 
    return (x.conj().dot(w).dot(y))/(x.conj().dot(w).dot(x))

def Vgl(N):
    # inverse covariance matrix for propagation constant computation
    return np.eye(N-1, dtype=complex) - (1/N)*np.ones(shape=(N-1, N-1), dtype=complex)

def compute_gamma(X_inv, M, lengths, gamma_est, inx=0):
    # gamma = alpha + 1j*beta is determined through linear weighted least-squares    
    lengths = lengths - lengths[inx]
    EX = (X_inv@M)[[1,2],:]               # extract z and y columns
    EX = np.diag(1/EX[:,inx])@EX          # normalize to a reference line based on index `inx` (can be any)
    del_inx = np.arange(len(lengths)) != inx  # get rid of the reference line (i.e., thru)
    
    l = 2*lengths[del_inx]
    gamma_l = np.log((EX[0,:] + 1/EX[-1,:])/2)[del_inx]
    n = np.round( (gamma_l - gamma_est*l).imag/np.pi/2 )
    gamma_l = gamma_l - 1j*2*np.pi*n # unwrap
    
    return WLS(l, gamma_l, Vgl(len(l)+1))
    
def solve_quadratic(v1, v2, inx, x_est):
    # inx contain index of the unit value and product 
    v12,v13 = v1[inx]
    v22,v23 = v2[inx]
    mask = np.ones(v1.shape, bool)
    mask[inx] = False
    v11,v14 = v1[mask]
    v21,v24 = v2[mask]
    if abs(v12) > abs(v22):  # to avoid dividing by small numbers
        k2 = -v11*v22*v24/v12 + v11*v14*v22**2/v12**2 + v21*v24 - v14*v21*v22/v12
        k1 = v11*v24/v12 - 2*v11*v14*v22/v12**2 - v23 + v13*v22/v12 + v14*v21/v12
        k0 = v11*v14/v12**2 - v13/v12
        c2 = np.array([(-k1 - np.sqrt(-4*k0*k2 + k1**2))/(2*k2), (-k1 + np.sqrt(-4*k0*k2 + k1**2))/(2*k2)])
        c1 = (1 - c2*v22)/v12
    else:
        k2 = -v11*v12*v24/v22 + v11*v14 + v12**2*v21*v24/v22**2 - v12*v14*v21/v22
        k1 = v11*v24/v22 - 2*v12*v21*v24/v22**2 + v12*v23/v22 - v13 + v14*v21/v22
        k0 = v21*v24/v22**2 - v23/v22
        c1 = np.array([(-k1 - np.sqrt(-4*k0*k2 + k1**2))/(2*k2), (-k1 + np.sqrt(-4*k0*k2 + k1**2))/(2*k2)])
        c2 = (1 - c1*v12)/v22
    x = np.array( [v1*x + v2*y for x,y in zip(c1,c2)] )  # 2 solutions
    mininx = np.argmin( abs(x - x_est).sum(axis=1) )
    return x[mininx]

def mN_at_one_freq(Slines, lengths, ereff_est, kappa_est, f):
    # measurements
    Mi    = np.array([s2t(x) for x in Slines]) # convert to T-parameters
    Miinv = np.array([np.linalg.pinv(x) for x in Mi]) # get the inverse
    inx = np.tril_indices(len(lengths), k=-1)  # get indecies of unique pairs
    
    ## build pairs
    Mij    = Mi[inx[0]] - Mi[inx[1]]
    Mijinv = Miinv[inx[0]] - Miinv[inx[1]]
    
    M    = np.array([x.flatten('F') for x in Mij]).T
    Minv = np.array([x.flatten('F') for x in Mijinv]).T
    
    ## Compute W via Takagi decomposition (also the eigenvalue lambda is computed)
    G, lambd = compute_G_with_takagi(Minv.T@P@M)
    W = (G@np.array([[0,1j],[-1j,0]])@G.T).conj()

    gamma_est = 2*np.pi*f/c0*np.sqrt(-ereff_est+1j*np.finfo(float).eps)
    gamma_est = abs(gamma_est.real) + 1j*abs(gamma_est.imag)  # this to avoid sign inconsistencies 

    vij   = np.exp(-gamma_est*(lengths[inx[0]]-lengths[inx[1]])) - np.exp(gamma_est*(lengths[inx[0]]-lengths[inx[1]]))
    z_est = vij*np.exp(-gamma_est*(lengths[inx[0]]+lengths[inx[1]]))
    y_est = vij*np.exp(gamma_est*(lengths[inx[0]]+lengths[inx[1]]))
    W_est = -(kappa_est*(np.outer(y_est,z_est) - np.outer(z_est,y_est))).conj()
    W = -W if abs(W-W_est).sum() > abs(W+W_est).sum() else W # resolve the sign ambiguity
    
    ## weighted eigenvalue problem
    F = M@W@Minv.T@P     # measurements (weighted)   
    eigval, eigvec = np.linalg.eig(F-lambd*np.eye(4))
    inx = np.argsort(abs(eigval))
    v1 = eigvec[:,inx[1]]
    v2 = eigvec[:,inx[0]]
    v3 = eigvec[:,inx[3]]
    v4 = eigvec[:,inx[2]]
    x2__est = v2/v2[1]
    x2__est[2] = x2__est[0]*x2__est[-1]
    x3__est = v3/v3[2]
    x3__est[1] = x3__est[0]*x3__est[-1]
    x1__est = np.array([1, x3__est[3], x2__est[3], x3__est[3]*x2__est[3]])
    x4_est  = np.array([x3__est[0]*x2__est[0], x3__est[0], x2__est[0], 1])
    
    # solve quadratic equation for each column
    x1_ = solve_quadratic(v1, v4, [0,3], x1__est)
    x2_ = solve_quadratic(v2, v3, [1,2], x2__est)
    x3_ = solve_quadratic(v2, v3, [2,1], x3__est)
    x4  = solve_quadratic(v1, v4, [3,0], x4_est)
    
    # build the normalized cal coefficients (average the answers from range and null spaces)    
    a12 = (x2_[0] + x4[2])/2
    b21 = (x3_[0] + x4[1])/2
    a21_a11 = (x1_[1] + x3_[3])/2
    b12_b11 = (x1_[2] + x2_[3])/2
    X_  = np.kron([[1,b21],[b12_b11,1]], [[1,a12],[a21_a11,1]])
    
    X_inv = np.linalg.pinv(X_)
    
    ## Compute propagation constant
    MM = np.array([x.flatten('F') for x in Mi]).T
    gamma = compute_gamma(X_inv, MM, lengths, gamma_est)
    ereff = -(c0/2/np.pi/f*gamma)**2
    S = np.array([t2s(X_inv.dot(m).reshape((2,2),order='F')) for m in MM.T])
    kappa = (S[:,0,0]*S[:,1,1]/S[:,0,1]/S[:,1,0]).mean()
    
    return gamma, ereff, kappa, lambd

def correct_switch_term(S, GF, GR):
    '''
    correct switch terms of measured S-parameters at a single frequency point
    GF: forward (sourced by port-1)
    GR: reverse (sourced by port-2)
    '''
    S_new = S.copy()
    S_new[0,0] = (S[0,0]-S[0,1]*S[1,0]*GF)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[0,1] = (S[0,1]-S[0,0]*S[0,1]*GR)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[1,0] = (S[1,0]-S[1,1]*S[1,0]*GF)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[1,1] = (S[1,1]-S[0,1]*S[1,0]*GR)/(1-S[0,1]*S[1,0]*GF*GR)
    return S_new

class mN:
    """
    Multinetwork method to measure propagation constant.
    """
    def __init__(self, lines, line_lengths, ereff_est=1+0j, kappa_est=-1+0j, switch_term=None):
        """
        mN initializer.
        
        Parameters
        --------------
        lines : list of :class:`~skrf.network.Network`
             Measured offset networks. The first one is defined as zero offset.
                
        line_lengths : list of float
            Lengths of offset for each network.
            In the same order as the parameter 'lines'
                        
        ereff_est : complex
            Estimated effective permittivity.
            
        kappa_est : complex
                Estimated s-parameter ratio of offseted network.
        
        switch_term : list of :class:`~skrf.network.Network`
            list of 1-port networks. Holds 2 elements:
                1. network for forward switch term.
                2. network for reverse switch term.
        """
        self.f  = lines[0].frequency.f
        self.Slines = np.array([x.s for x in lines])
        self.lengths = np.array(line_lengths)
        self.ereff_est = ereff_est
        self.kappa_est = kappa_est
        if switch_term is not None:
            self.switch_term = np.array([x.s.squeeze() for x in switch_term])
        else:
            self.switch_term = np.array([self.f*0 for x in range(2)])
    
    def run_mN(self):
        # This runs the multiNetwork procedure
        gammas  = []
        lambds  = []        
        kappas  = []
        lengths   = self.lengths
        ereff_est = self.ereff_est
        kappa_est = self.kappa_est
        print('\nmN is running...')
        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            sw = self.switch_term[:,inx]
            
            # correct switch term
            Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sw) else Slines
            
            gamma, ereff_est, kappa_est, lambd = mN_at_one_freq(Slines, lengths=lengths, ereff_est=ereff_est, 
                                                                kappa_est=kappa_est, f=f)
            kappas.append(kappa_est)
            gammas.append(gamma)
            lambds.append(lambd)
            print(f'Frequency: {f*1e-9:.2f} GHz ... DONE!')
        self.kappa = np.array(kappas)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.lambd = np.array(lambds)

# EOF