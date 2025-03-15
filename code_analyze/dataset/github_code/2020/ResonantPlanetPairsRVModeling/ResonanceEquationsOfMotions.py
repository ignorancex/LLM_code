import rebound as rb
import reboundx
import numpy as np
import theano
import theano.tensor as T
from exoplanet.theano_ops.kepler import KeplerOp
from warnings import warn
from scipy.optimize import lsq_linear
DEBUG = False

def getOmegaMatrix(n):
    """
    Get the 2n x 2n skew-symmetric block matrix:
          [0 , I_n]
          [-I_n, 0 ]
    that appears in Hamilton's equations.

    Arguments
    ---------
    n : int
        Determines matrix dimension

    Returns
    -------
    numpy.array
    """
    return np.vstack(
        (
         np.concatenate([np.zeros((n,n)),np.eye(n)]).T,
         np.concatenate([-np.eye(n),np.zeros((n,n))]).T
        )
    )

def calc_DisturbingFunction_with_sinf_cosf(alpha,e1,e2,w1,w2,sinf1,cosf1,sinf2,cosf2):
    """
    Compute the value of the disturbing function
    .. math::
        \frac{a'}{|r-r'|} - a'\frac{r.r'}{|r'^3|}
    from a set of input orbital elements for coplanar planets.

    Arguments
    ---------
    alpha : float
        semi-major axis ratio
    e1 : float
        inner eccentricity
    e2 : float
        outer eccentricity
    w1 : float
        inner long. of peri
    w2 : float
        outer long. of peri
    sinf1 : float
        sine of inner planet true anomaly
    cosf1 : float
        cosine of inner planet true anomaly
    sinf2 : float
        sine of outer planet true anomaly
    cosf2 : float
        cosine of outer planet true anomaly

    Returns
    -------
    float :
        Disturbing function value
    """
    r1 = alpha * (1-e1*e1) /(1 + e1 * cosf1)
    _x1 = r1 * cosf1
    _y1 = r1 * sinf1
    Cw1 = T.cos(w1)
    Sw1 = T.sin(w1)
    x1 = Cw1 * _x1  - Sw1 * _y1
    y1 = Sw1 * _x1  + Cw1 * _y1

    r2 = (1-e2*e2) /(1 + e2 * cosf2)
    _x2 = r2 * cosf2
    _y2 = r2 * sinf2
    Cw2 = T.cos(w2)
    Sw2 = T.sin(w2)
    x2 = Cw2 * _x2  - Sw2 * _y2
    y2 = Sw2 * _x2  + Cw2 * _y2

    # direct term
    dx = (x2 - x1)
    dy = (y2 - y1)
    dr2 = dx*dx + dy*dy
    direct = 1 / T.sqrt(dr2)

    # indirect term
    r1dotr = (x2 * x1 + y2 * y1)
    r1sq = x2*x2 + y2*y2
    r1_3 = 1 / r1sq / T.sqrt(r1sq)
    indirect = -1 * r1dotr * r1_3

    return direct+indirect


def get_compiled_Hkep_Hpert_full():
        # resonance j and k
        j,k = T.lscalars('jk')
        s = (j-k) / k

        # Planet masses: m1,m2
        m1,m2 = T.dscalars(2)
        Mstar = 1
        eta0 = Mstar
        eta1 = eta0+m1
        eta2 =eta1+m2
        mtilde1 = m1 * (eta0/eta1)
        Mtilde1 = Mstar * (eta1/eta0)
        mtilde2 = m2 * (eta1/eta2)
        Mtilde2 = Mstar * (eta2/eta1)
        eps = m1 * m2 / (mtilde1 + mtilde2) / Mstar
        beta1 = mtilde1 / (mtilde1 + mtilde2)
        beta2 = mtilde2 / (mtilde1 + mtilde2)
        gamma = mtilde2/mtilde1
        

        # Dynamical variables:
        dyvars = T.vector()
        Q,sigma1, sigma2, I1, I2, amd = [dyvars[i] for i in range(6)]

        # Set lambda2=0
        l2 = T.constant(0.)
        l1 = -1 * k * Q 
        w1 = (1+s) * l2 - s * l1 - sigma1
        w2 = (1+s) * l2 - s * l1 - sigma2
        
        Gamma1 = I1
        Gamma2 = I2
        
        # Resonant semi-major axis ratio
        alpha_res = ((j-k)/j)**(2/3) * ((Mstar + m1) / (Mstar+m2))**(1/3)
        P0 = k * ( beta2 - beta1 * T.sqrt(alpha_res) ) / 2
        P = P0 - k * (s+1/2) * amd
        
        Ltot = beta1 * T.sqrt(alpha_res) + beta2 - amd
        L1 = Ltot/2 - P / k - s * (I1 + I2)
        L2 = Ltot/2 + P / k + (1 + s) * (I1 + I2)
        
        a1 = (L1 / beta1 )**2 * eta0 / eta1
        e1 = T.sqrt(1-(1-(Gamma1 / L1))**2)
        
        a2 = (L2 / beta2 )**2 * eta1 / eta2
        e2 = T.sqrt(1-(1-(Gamma2 / L2))**2)
        
        Hkep = - eta1 * beta1 / (2 * a1) / eta0  - eta2 * beta2 / (2 * a2) / eta1
        
        alpha = a1 / a2
        
        ko = KeplerOp()
        
        M1 = l1 - w1
        M2 = l2 - w2
        
        sinf1,cosf1 =  ko( M1, e1 + T.zeros_like(M1) )
        sinf2,cosf2 =  ko( M2, e2 + T.zeros_like(M2) )
        
        R = calc_DisturbingFunction_with_sinf_cosf(alpha,e1,e2,w1,w2,sinf1,cosf1,sinf2,cosf2)
        
        Hpert = -eps * R / a2
        
        omega_syn = T.sqrt(eta2/eta1)/a2**1.5 - T.sqrt(eta1/eta0) / a1**1.5
        gradHpert = T.grad(Hpert,wrt=dyvars)
        gradHkep = T.grad(Hkep,wrt=dyvars)
        grad_omega_syn = T.grad(omega_syn,wrt=dyvars)

        extra_ins = [m1,m2,j,k]
        ins = [dyvars] + extra_ins

        # Scalars
        omega_syn_fn = theano.function(
            inputs=ins,
            outputs=omega_syn,
            givens=None,
            on_unused_input='ignore'
        )
        Hpert_fn = theano.function(
            inputs=ins,
            outputs=Hpert,
            givens=None,
            on_unused_input='ignore'
        )
        Hkep_fn = theano.function(
            inputs=ins,
            outputs=Hkep,
            givens=None,
            on_unused_input='ignore'
        )
        # gradients
        grad_omega_syn_fn = theano.function(
            inputs=ins,
            outputs=grad_omega_syn,
            givens=None,
            on_unused_input='ignore'
        )
        gradHpert_fn = theano.function(
            inputs=ins,
            outputs=gradHpert,
            givens=None,
            on_unused_input='ignore'
        )
        gradHkep_fn = theano.function(
            inputs=ins,
            outputs=gradHkep,
            givens=None,
            on_unused_input='ignore'
        )
        return omega_syn_fn,Hkep_fn,Hpert_fn,grad_omega_syn_fn,gradHkep_fn,gradHpert_fn

def get_compiled_theano_functions(N_QUAD_PTS):
        # resonance j and k
        j,k = T.lscalars('jk')
        s = (j-k) / k

        # Planet masses: m1,m2
        m1,m2 = T.dscalars(2)
        Mstar = 1
        eta0 = Mstar
        eta1 = eta0+m1
        eta2 =eta1+m2
        mtilde1 = m1 * (eta0/eta1)
        Mtilde1 = Mstar * (eta1/eta0)
        mtilde2 = m2 * (eta1/eta2)
        Mtilde2 = Mstar * (eta2/eta1)
        eps = m1 * m2 / (mtilde1 + mtilde2) / Mstar
        beta1 = mtilde1 / (mtilde1 + mtilde2)
        beta2 = mtilde2 / (mtilde1 + mtilde2)
        gamma = mtilde2/mtilde1
        
        # Angle variable for averaging over
        Q = T.dvector('Q')

        # Dynamical variables:
        dyvars = T.vector()
        sigma1, sigma2, I1, I2, amd = [dyvars[i] for i in range(5)]


        # Quadrature weights
        quad_weights = T.dvector('w')
        
        # Set lambda2=0
        l2 = T.constant(0.)
        
        l1 = -1 * k * Q 
        w1 = (1+s) * l2 - s * l1 - sigma1
        w2 = (1+s) * l2 - s * l1 - sigma2
        
        Gamma1 = I1
        Gamma2 = I2
        
        # Resonant semi-major axis ratio
        alpha_res = ((j-k)/j)**(2/3) * ((Mstar + m1) / (Mstar+m2))**(1/3)
        P0 =  k * ( beta2 - beta1 * T.sqrt(alpha_res) ) / 2
        P = P0 - k * (s+1/2) * amd
        Ltot = beta1 * T.sqrt(alpha_res) + beta2 - amd
        L1 = Ltot/2 - P / k - s * (I1 + I2)
        L2 = Ltot/2 + P / k + (1 + s) * (I1 + I2)
        
        
        a1 = (L1 / beta1 )**2 * eta0 / eta1
        e1 = T.sqrt(1-(1-(Gamma1 / L1))**2)
        
        a2 = (L2 / beta2 )**2 * eta1 / eta2
        e2 = T.sqrt(1-(1-(Gamma2 / L2))**2)
        
        Hkep = - eta1 * beta1 / (2 * a1) / eta0  - eta2 * beta2 / (2 * a2) / eta1
        
        alpha = a1 / a2
        
        ko = KeplerOp()
        
        M1 = l1 - w1
        M2 = l2 - w2
        
        sinf1,cosf1 =  ko( M1, e1 + T.zeros_like(M1) )
        sinf2,cosf2 =  ko( M2, e2 + T.zeros_like(M2) )
        
        R = calc_DisturbingFunction_with_sinf_cosf(alpha,e1,e2,w1,w2,sinf1,cosf1,sinf2,cosf2)
        Rav = R.dot(quad_weights)
        
        Hpert = -eps * Rav / a2
        Htot = Hkep + Hpert

        ######################
        # Dissipative dynamics
        ######################
        tau_alpha_0, K1, K2, p = T.dscalars(4)
        sigma1dot_dis,sigma2dot_dis,I1dot_dis,I2dot_dis,amddot_dis = T.dscalars(5)
        sigma1dot_dis,sigma2dot_dis = T.as_tensor(0.),T.as_tensor(0.)
        
        # Define timescales
        tau_e1 = tau_alpha_0 / K1
        tau_e2 = tau_alpha_0 / K2
        tau_a1_0 = -1 * tau_alpha_0 * (1+alpha_res * gamma)/ (alpha_res * gamma)
        tau_a2_0 = -1 * alpha_res * gamma * tau_a1_0
        tau_a1 = 1 / (1/tau_a1_0 + 2 * p * e1*e1 / tau_e1 )
        tau_a2 = 1 / (1/tau_a2_0 + 2 * p * e2*e2 / tau_e2 )

        # Time derivative of orbital elements
        e1dot_dis = -1*e1 / tau_e1
        e2dot_dis = -1*e2 / tau_e2
        a1dot_dis = -1*a1 / tau_a1
        a2dot_dis = -1*a2 / tau_a2

        # Time derivatives of canonical variables
        I1dot_dis = L1 * e1 * e1dot_dis / ( T.sqrt(1-e1*e1) ) - I1 / tau_a1 / 2
        I2dot_dis = L2 * e2 * e2dot_dis / ( T.sqrt(1-e2*e2) ) - I2 / tau_a2 / 2
        Pdot_dis = -1*k * ( L2 / tau_a2 - L1 / tau_a1) / 4 - k * (s + 1/2) * (I1dot_dis + I2dot_dis)
        amddot_dis = Pdot_dis / T.grad(P,amd)

        #####################################################
        # Set parameters for compiling functions with Theano
        #####################################################
        
        # Get numerical quadrature nodes and weights
        nodes,weights = np.polynomial.legendre.leggauss(N_QUAD_PTS)
        
        # Rescale for integration interval from [-1,1] to [-pi,pi]
        nodes = nodes * np.pi
        weights = weights * 0.5
        
        # 'givens' will fix some parameters of Theano functions compiled below
        givens = [(Q,nodes),(quad_weights,weights)]

        # 'ins' will set the inputs of Theano functions compiled below
        #   Note: 'extra_ins' will be passed as values of object attributes
        #   of the 'ResonanceEquations' class 'defined below
        extra_ins = [m1,m2,j,k,tau_alpha_0,K1,K2,p]
        ins = [dyvars] + extra_ins
        

        # Define flows and jacobians.

        #  Conservative flow
        gradHtot = T.grad(Htot,wrt=dyvars)
        hessHtot = theano.gradient.hessian(Htot,wrt=dyvars)
        Jtens = T.as_tensor(np.pad(getOmegaMatrix(2),(0,1),'constant'))
        H_flow_vec = Jtens.dot(gradHtot)
        H_flow_jac = Jtens.dot(hessHtot)
        
        #  Dissipative flow
        dis_flow_vec = T.stack(sigma1dot_dis,sigma2dot_dis,I1dot_dis,I2dot_dis,amddot_dis)
        dis_flow_jac = theano.gradient.jacobian(dis_flow_vec,dyvars)
        
        
        # Extras
        dis_timescales = [tau_a1_0,tau_a2_0,tau_e1,tau_e2]
        orbels = [a1,e1,sigma1*k,a2,e2,sigma2*k]
        ##########################
        # Compile Theano functions
        ##########################
        
        if not DEBUG:
            # Note that compiling can take a while
            #  so I've put a debugging switch here 
            #  to skip evaluating these functions when
            #  desired.
            Rav_fn = theano.function(
                inputs=ins,
                outputs=Rav,
                givens=givens,
                on_unused_input='ignore'
            )
            Hpert_av_fn = theano.function(
                inputs=ins,
                outputs=Hpert,
                givens=givens,
                on_unused_input='ignore'
            )
            Htot_fn = theano.function(
                inputs=ins,
                outputs=Htot,
                givens=givens,
                on_unused_input='ignore'
            )
            
            H_flow_vec_fn = theano.function(
                inputs=ins,
                outputs=H_flow_vec,
                givens=givens,
                on_unused_input='ignore'
            )
            
            H_flow_jac_fn = theano.function(
                inputs=ins,
                outputs=H_flow_jac,
                givens=givens,
                on_unused_input='ignore'
            )
            
            dis_flow_vec_fn = theano.function(
                inputs=ins,
                outputs=dis_flow_vec,
                givens=givens,
                on_unused_input='ignore'
            )
            
            dis_flow_jac_fn = theano.function(
                inputs=ins,
                outputs=dis_flow_jac,
                givens=givens,
                on_unused_input='ignore'
            )

            dis_timescales_fn =theano.function(
                inputs=extra_ins,
                outputs=dis_timescales,
                givens=givens,
                on_unused_input='ignore'
            )

            orbels_fn = theano.function(
                inputs=ins,
                outputs=orbels,
                givens=givens,
                on_unused_input='ignore'
            )

        else:
            return  [lambda x: x for _ in range(8)]
        
        return Rav_fn,Hpert_av_fn,Htot_fn,H_flow_vec_fn,H_flow_jac_fn,dis_flow_vec_fn,dis_flow_jac_fn,dis_timescales_fn,orbels_fn
        

class ResonanceEquations():
    """
    A class for the model describing the dynamics of a pair of planar planets
    in/near a mean motion resonance.

    Includes the effects of dissipation.

    Attributes
    ----------
    j : int
        Together with k specifies j:j-k resonance
    
    k : int
        Order of resonance.
    
    alpha : float
        Semi-major axis ratio a_1/a_2

    eps : float
        Mass parameter m1*mu2 / (mu1+mu2)

    m1 : float
        Inner planet mass

    m2 : float
        Outer planet mass

    """
    def __init__(self,j,k, n_quad_pts = 40, m1 = 1e-5 , m2 = 1e-5,K1=100, K2=100, tau_alpha = 1e5, p = 1):
        self.j = j
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.K1 = K1
        self.K2 = K2
        self.tau_alpha = tau_alpha
        self.p = p 
        self.n_quad_pts = n_quad_pts
        funcs = get_compiled_theano_functions(n_quad_pts)

        self._Rav_fn,self._Hpert_av_fn,self._H,self._H_flow,self._H_jac,self._dis_flow,self._dis_jac,self._times_scales,self._orbels_fn = funcs

        self._omega_syn_fn, self._Hkep_fn, self._Hpert_fn,\
                self._omega_syn_grad_fn, self._Hkep_grad_fn, self._Hpert_grad_fn = get_compiled_Hkep_Hpert_full()
    
    @property
    def extra_args(self):
        return [self.m1,self.m2,self.j,self.k,self.tau_alpha,self.K1,self.K2,self.p]
    @property
    def Hpert_extra_args(self):
        return [self.m1,self.m2,self.j,self.k]

    @property
    def mu1(self):
        return self.m1 / (1 + self.m1)

    @property
    def mu2(self):
        return self.m2 / (1 + self.m2)

    @property
    def beta1(self):
        return self.mu1 / (self.mu1 + self.mu2)

    @property
    def beta1(self):
        return self.mu1 / (self.mu1 + self.mu2)

    @property
    def beta2(self):
        return self.mu2 / (self.mu1 + self.mu2)
    @property
    def eps(self):
        return self.m1 * self.mu2 / (self.mu1 + self.mu2)
    @property
    def alpha(self):
        return ((self.j - self.k) / self.j)**(2/3)
    @property
    def timescales(self):
        tscales_array = self._times_scales(*self.extra_args)
        return {name:scale for name,scale in zip(['tau_a1','tau_a2','tau_e1','tau_e2'],tscales_array)}

    def Rav(self,z):
        """
        Calculate the value of the Q-averaged disturbing function

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the Hamiltonian evaluated at z.
        """
        return self._Rav_fn(z,*self.extra_args)

    def Hpert_osc(self,Q,z):
        """
        Calculate the value of the un-averaged disturbing function

        Arguments
        ---------
        Q : float
            Synodic angle
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the (unaveraged) perturbation Hamiltonian evaluated at (Q,z).
        """
        arg = np.concatenate(([Q],z))
        Hpert_av = self._Hpert_av_fn(z,*self.extra_args)
        return self._Hpert_fn(arg,*self.Hpert_extra_args) - Hpert_av

    def gradHpert(self,Q,z):
        """
        Calculate the value of the averaged disturbing function

        Arguments
        ---------
        Q : float
            Synodic angle
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            The gradient of the (unaveraged) perturbation Hamiltonian evaluated at (Q,z).
        """
        arg = np.concatenate(([Q],z))
        return self._Hpert_grad_fn(arg,*self.Hpert_extra_args)

    def H(self,z):
        """
        Calculate the value of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the Hamiltonian evaluated at z.
        """
        return self._H(z,*self.extra_args)

    def H_kep(self,z):
        """
        Calculate the value of the Keplerian component of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the Keplerian part of the Hamiltonian evaluated at z.
        """
        Htot = self.H(z)
        Hpert = self.H_pert(z)
        return Htot - Hpert

    def H_pert(self,z):
        """
        Calculate the value of the perturbation component of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the perturbation part of the Hamiltonian evaluated at z.
        """
        Hpert = self._Hpert_av_fn(z,*self.extra_args)
        return Hpert

    def omega_syn(self,z):
        """
        Calculate the synodic frequency dH0/dP.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of dH0/dP evaluated at z.
        """

        zfull = np.concatenate(([0],z))
        return self._omega_syn_fn(zfull,*self.Hpert_extra_args)
    def grad_omega_syn(self,z):
        """
        Calculate the gradient of the synodic frequency, grad(dH0/dP).

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            The value of grad(dH0/dP) evaluated at z.
        """

        zfull = np.concatenate(([0],z))
        return self._omega_syn_grad_fn(zfull,*self.Hpert_extra_args)

    def H_flow(self,z):
        """
        Calculate flow induced by the Hamiltonian
        .. math:
            \dot{z} = \Omega \cdot \nablda_{z}H(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow vector
        """
        return self._H_flow(z,*self.extra_args)


    def H_flow_jac(self,z):
        """
        Calculate the Jacobian of the flow induced by the Hamiltonian
        .. math:
             \Omega \cdot \Delta H(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Jacobian matrix
        """
        return self._H_jac(z,*self.extra_args)

    def dis_flow(self,z):
        """
        Calculate flow induced by dissipative forces
        .. math:
            \dot{z} = f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow vector
        """
        return self._dis_flow(z,*self.extra_args)

    def dis_flow_jac(self,z):
        """
        Calculate the Jacobian of the flow induced by dissipative forces
        .. math:
            \dot{z} = \nabla f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow Jacobian
        """
        return self._dis_jac(z,*self.extra_args)

    def flow(self,z):
        """
        Calculate flow 
        .. math:
            \dot{z} = \Omega \cdot \nablda_{z}H(z) + f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow vector
        """
        return self._H_flow(z,*self.extra_args) + self._dis_flow(z,*self.extra_args)

    def flow_jac(self,z):
        """
        Calculate flow Jacobian
        .. math:
            \Omega \cdot \Delta H(z) + \\nabla f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Jacobian
        """
        return self._H_jac(z,*self.extra_args) + self._dis_jac(z,*self.extra_args)

    def dyvars_to_orbels(self,z):
        r"""
        Convert dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}
        to orbital elements
        .. math:
            (a_1,e_1,\\theta_1,a_2,e_2,\\theta_2)
        """

        return self._orbels_fn(z,*self.extra_args)

    def orbels_to_dyvars(self,orbels):
        r"""
        Convert orbital elements
        .. math:
            (a_1,e_1,\theta_1,a_2,e_2,\theta_2)
        to dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}
        """
        # Total angular momentum constrained by
        # Ltot = beta1 * sqrt(alpha_res) + beta2 - amd
        Mstar = 1
        m1 = self.m1
        m2 = self.m2
        k = self.k
        j = self.j
        alpha_res = ((j-k)/j)**(2/3) * ((Mstar + m1) / (Mstar+m2))**(1/3)
    
        s = (j - k) / k
        
    
        a1,e1,theta1,a2,e2,theta2 = orbels
        
        L1 = self.beta1 * np.sqrt(a1)
        L2 = self.beta2 * np.sqrt(a2)
        Gamma1 = L1 * (1 - np.sqrt(1-e1**2))
        Gamma2 = L2 * (1 - np.sqrt(1-e2**2))
        I1 = Gamma1
        I2 = Gamma2
        
        P = k * (L2-L1) / 2 - k * (s+1/2)*(I1+I2)
        Ltot = L1 + L2 - I1 - I2
        
        Lcirc = self.beta1 * np.sqrt(alpha_res) + self.beta2 
        P0 = k * (self.beta2 - self.beta1 * np.sqrt(alpha_res)) / 2
        # Now solve for re-scaling factor 'scale' and AMD value that satisfy
        #  scale * P = P0 - k*(s+1/2)*amd  
        #  scale * Ltot = Lcirc - amd
        # Solution is given by:
        #  scale = (2 k Lcirc s-k Lcirc+2 P0)/(k Ltot-2 P+2 k Ltot s)
        # (2 (Lcirc P-Ltot P0))/(k Ltot-2 P+2 k Ltot s)
    
        scale = (k * (1 + 2 * s) * Lcirc - 2 * P0) / (k * (1 + 2 * s) * Ltot - 2 * P)
        amd = Lcirc - scale * Ltot
        I1 *= scale
        I2 *= scale
        sigma1 = theta1 / k
        sigma2 = theta2 / k
        
        return np.array((sigma1,sigma2,I1,I2,amd))
    
        

    def gradHkep(self,z):
        zfull = np.concatenate(([0],z))
        return self._Hkep_grad_fn(zfull,*self.Hpert_extra_args)

    def mean_to_osculating_dyvars(self,Q,z,N = 256):
        r"""
        Apply perturbation theory to transfrom from the phase space coordiantes of the 
        averaged model to osculating phase space coordintates of the full phase space.

        Assumes that the transformation is being applied at the fixed point of the 
        averaged model.

        Arguemnts
        ---------
        Q : float or ndarry
            Value(s) of angle (lambda2-lambda1)/k at which to apply transformation.
            Equilibrium points of the averaged model correspond to orbits that are 
            periodic in Q in the full phase space.
        z : ndarray
            Dynamical variables of the averaged model:
                $\sigma_1,\sigma_2,I_1,I_2,AMD$
        N : int, optional 
            Number of Q points to evaluate functions at when performing Fourier 
            transformation. Default is 
                N=256
        Returns
        -------
        zosc : ndarray, (5,) or (M,5) 
            The osculating values of the dynamical varaibles for the
            input Q values. The dimension of the returned variables
            is set by the dimension of the input 'Q'. If Q is a 
            float, then z is an array of length 5. If Q is an 
            array of length M then zosc is an (M,5) array.
        """
        omega_syn = self.omega_syn(z)
        OmegaMtrx = getOmegaMatrix(2)
        Omega_del2H = self.H_flow_jac(z)[:-1,:-1]
        vals,S = np.linalg.eig(Omega_del2H)
        S = np.transpose([S.T[i] for i in (0,2,1,3)])
        s1 = (S.T @ OmegaMtrx @ S)[0,2]
        s2 = (S.T @ OmegaMtrx @ S)[1,3]
        S.T[0]*=1/np.sqrt(s1)
        S.T[2]*=1/np.sqrt(s1)
        S.T[1]*=1/np.sqrt(s2)
        S.T[3]*=1/np.sqrt(s2)
        Sinv = np.linalg.inv(S)
    
        Qarr = np.atleast_1d(Q) 
        dchi = np.zeros((4,len(Qarr)),dtype=complex)
    
        # Fill arrays for FFT
        Qs = np.linspace(0,2*np.pi,N)
        X = np.zeros((N,4),dtype=complex)
    
        gradHkep = self.gradHkep(z)
        domega_syn_dz = self.grad_omega_syn(z)[1:-1]
        domega_syn_dw = S.T @ domega_syn_dz
    
        for i,q in enumerate(Qs):
            gradHosc = self.gradHpert(q,z) + gradHkep
            X[i] = S.T @ (gradHosc)[1:-1]
            X[i] -= self.Hpert_osc(q,z) * domega_syn_dw/omega_syn
    
        omegas = +1*np.imag(np.diag(Sinv @ Omega_del2H @ S))[:2]
        for I in range(2):
            A = np.fft.fft(X[:,I])
            freqs = np.fft.fftshift(np.fft.fftfreq(N)*N)
            amps = np.fft.fftshift(A)/N
            for l in range(1,N//2 - 1):
                sig = -1j * self.k * amps[N//2+l] * np.exp(1j * freqs[N//2+l] * Qarr)  / (-l*omega_syn - self.k * omegas[I])
                sig +=-1j * self.k * amps[N//2-l] * np.exp(1j * freqs[N//2-l] * Qarr)  / (l*omega_syn - self.k * omegas[I])
                dchi[I] += sig

        dchi[2] = -1j * np.conjugate(dchi[0])
        dchi[3] = -1j * np.conjugate(dchi[1])
    
        # Get AMD correction
        s = (self.j - self.k) / self.k
        dAMD = np.array([self.Hpert_osc(q,z) for q in Qarr]) / omega_syn / (s+1/2)
    
        dsigmaI = np.transpose(-1 * (S @ OmegaMtrx @ dchi).T)
        result = np.transpose(z + np.vstack((dsigmaI,dAMD)).T)
        result = np.real(result) # trim small imaginary parts cause by numerical errors
        if result.shape[1] == 1:
            return result.reshape(-1)
        return result

    from scipy.optimize import lsq_linear
    def find_equilibrium(self,guess,dissipation=False,tolerance=1e-9,max_iter=10):
        """
        Use Newton's method to locate an equilibrium solution of 
        the equations of motion. 
    
        By default, an equilibrium of the dissipation-free equations
        is sought. In this case, the AMD value of the equilibrium 
        solution will be equal to the value of the initially supplied 
        guess of dynamical variables. If dissipative terms are included
        then the equilibrium will depend on the parameters K1 and K2
        as well as tau_alpha.
    
        Arguments
        ---------
        guess : ndarray
            Initial guess for the dynamical variables at the equilibrium.
        dissipation : bool, optional
            Whether dissipative terms are considered in the equations of
            motion. Default is False.
        tolerance : float, optional
            Tolerance for root-finding such that solution satisfies |f(z)| < tolerance
            Default value is 1E-9.
        max_iter : int, optional
            Maximum number of Newton's method iterations.
            Default is 10.
        include_dissipation : bool, optional
            Include dissipative terms in the equations of motion.
            Default is False
    
        Returns
        -------
        zeq : ndarray
            Equilibrium value of dynamical variables.
    
        Raises
        ------
        RuntimeError : Raises error if maximum number of Newton iterations is exceeded.
        """
        if dissipation:
            return self._find_dissipative_equilibrium(guess,tolerance,max_iter)
        else:
            return self._find_conservative_equilibrium(guess,tolerance,max_iter)

    def _find_dissipative_equilibrium(self,guess,tolerance=1e-9,max_iter=10):
        y = guess
        lb = np.array([-np.pi,-np.pi, -1* y[2], -1 * y[3],-np.inf])
        ub = np.array([np.pi,np.pi,np.inf,np.inf,np.inf])
        fun = self.flow
        jac = self.flow_jac
        f = fun(y)
        J = jac(y)
        it=0
            # Newton method iteration
        while np.linalg.norm(f)>tolerance and it < max_iter:
            # Note-- using constrained least-squares
            # to avoid setting actions to negative
            # values.
    
            # The lower bounds ensure that  I1 and I2 are positive quantities
            lb[2:-1] =  -1* y[2:-1]
            dy = lsq_linear(J,-f,bounds=(lb,ub)).x
            y = y + dy
            f = fun(y)
            J = jac(y)
            it+=1
            if it==max_iter:
                raise RuntimeError("Max iterations reached!")
        return y


    def _find_conservative_equilibrium(self,guess,tolerance=1e-9,max_iter=10):
        y = guess
        lb = np.array([-np.pi,-np.pi, -1* y[2], -1 * y[3]])
        ub = np.array([np.pi,np.pi,np.inf,np.inf])
        fun = self.H_flow
        jac = self.H_flow_jac
        f = fun(y)[:-1]
        J = jac(y)[:-1,:-1]
        it=0
            # Newton method iteration
        while np.linalg.norm(f)>tolerance and it < max_iter:
            # Note-- using constrained least-squares
            # to avoid setting actions to negative
            # values.
            
            # The lower bounds ensure that  I1 and I2 are positive quantities
            lb[2:] =  -1* y[2:-1]
            dy = lsq_linear(J,-f,bounds=(lb,ub)).x
            y[:-1] = y[:-1] + dy
            f = fun(y)[:-1]
            J = jac(y)[:-1,:-1]
            it+=1
            if it==max_iter:
                raise RuntimeError("Max iterations reached!")
        return y

    def dyvars_to_rebound_simulation(self,z,Q=0,pomega1=0,osculating_correction = True,include_dissipation = False,**kwargs):
        r"""
        Convert dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}
        to a Rebound simulation.

        Arguments
        ---------
        z : ndarray
            Dynamical variables
        pomega1 : float, optional
            Planet 1's longitude of periapse. 
            Default is 0
        Q : float, optional
            Angle variable Q = (lambda2-lambda1) / k. 
            Default is Q=0. Resonant periodic orbits
            are 2pi-periodic in Q.
        include_dissipation : bool, optional
            Include dissipative effects through 
            reboundx's external forces.
            Default is False

        Keyword Arguments
        -----------------
        Mstar : float
            Default=1.0
            Stellar mass.
        inc : float, default = 0.0
            Inclination of planets' orbits.
        period1 : float
            Default = 1.0
            Orbital period of inner planet
        units : tuple
            Default = ('AU','Msun','days')
            Units of rebound simulation.

        Returns
        -------
        tuple :
            Returns a tuple. The first item
            of the tuple is a rebound simulation. 
            The second item is a reboundx.Extras object
            if 'include_dissipation' is True, otherwise
            the second item is None.
        """
        mean_orbels = self.dyvars_to_orbels(z)
        a1_mean,_,_,a2_mean,_,_ = mean_orbels 
        if osculating_correction:
            zosc = self.mean_to_osculating_dyvars(Q,z)
            orbels = self.dyvars_to_orbels(zosc)
        else:
            orbels = mean_orbels
        j,k = self.j, self.k
        a1,e1,theta1,a2,e2,theta2 = orbels
        syn_angle = self.k * Q
        pomega2 = np.mod( pomega1 -  (theta2 - theta1) / k, 2*np.pi)
        M1 = np.mod( (theta1 - j*syn_angle) / k ,2*np.pi )
        l1 = np.mod(M1 + pomega1,2*np.pi)
        l2 = np.mod( syn_angle + l1,2*np.pi)
        
        Mstar = kwargs.pop('Mstar',1.0)
        inc = kwargs.pop('inc',0.0)
        period1 = kwargs.pop('period1',1.0)
        units  = kwargs.pop('units', ('AU','Msun','days'))

        sim = rb.Simulation()
        sim.units = units
        mpl1 = self.m1 * Mstar
        mpl2 = self.m2 * Mstar
        # Rescale distance scale so that the proper semi-major axis of planet 1 corresponds to orbital period 'period1'
        a1_physical = (a1 / a1_mean) * (sim.G * (Mstar + mpl1) * period1**2 / (4*np.pi*np.pi))**(1/3)
        a2_physical = a2 * a1_physical / a1
        sim.add(m=Mstar)
        sim.add(m = mpl1, a=a1_physical, e=e1, l=l1, pomega=pomega1, inc=inc, jacobi_masses=True)
        sim.add(m = mpl2, a=a2_physical, e=e2, l=l2, pomega=pomega2, inc=inc, jacobi_masses=True)
        sim.move_to_com()

        rebx = None
        if include_dissipation:
            ps = sim.particles
            n2 = ps[2].n
            rebx = reboundx.Extras(sim)
            mod = rebx.load_operator("modify_orbits_direct")
            rebx.add_operator(mod)
            mod.params["p"] = self.p
            timescales = self.timescales
            ps[1].params["tau_a"]=-1*timescales['tau_a1'] / n2
            ps[2].params["tau_a"]=-1*timescales['tau_a2'] / n2
            ps[1].params["tau_e"]=-1*timescales['tau_e1'] / n2
            ps[2].params["tau_e"]=-1*timescales['tau_e2'] / n2
        return sim,rebx
from scipy.integrate import odeint
def get_res_eq_integration_results(res_eqs,y0,times,dissipation=False):
    """
    Convenience function to run a numerical integration of resonance 
    equations and get results as a dictionary.

    Arguments
    ---------
    res_eqs : ResonanceEquations object
        Resonance equations object to use for integrations.
    y0 : ndarray
        Dynamical variables representing initial conditions.
    times : ndarray
        Times for which solution will be computed.
    dissipation : bool, optional
        Whether to include dissipative forces in the integration.
        Default is false.
    """
    if dissipation:
        ydot = lambda t,y: res_eqs.flow(y)
        ydot_jac = lambda t,y: res_eqs.flow_jac(y)
    else:
        ydot = lambda t,y: res_eqs.H_flow(y)
        ydot_jac = lambda t,y: res_eqs.H_flow_jac(y)
    
    ysoln=odeint(
        ydot,
        y0,
        times,
        Dfun=ydot_jac,
        tfirst=True
    )
    
    els = dict(
        zip(
            ['a1','e1','theta1','a2','e2','theta2'],
            np.transpose([res_eqs.dyvars_to_orbels(y) for y in ysoln])
        )
    )
    
    els['time'] = times
    els['Delta'] = ((res_eqs.j-res_eqs.k)/res_eqs.j)*(els['a2']/els['a1'])**(1.5)-1
    
    return {'elements':els,'solution':ysoln}
