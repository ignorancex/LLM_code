from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sys
if sys.version_info >= (3,0):
    xrange = range

from cosmoTransitions.generic_potential import generic_potential
from cosmoTransitions import helper_functions
from cosmoTransitions import transitionFinder
from cosmoTransitions.finiteT import Jb_spline as Jb
from cosmoTransitions.finiteT import Jf_spline as Jf


# parameters
v2 = 174.**2.
mZ2 = 91.2**2.
mW2 = 80.4**2.
mt2 = 172.9**2
mh1252 = 125.**2

# fix some plotting settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class model1(generic_potential):
    """ 
    The `__init__` function performs initialization specific for this abstract
    class. Subclasses should either override this initialization *but make sure
    to call the parent implementation*, or, more simply, override the
    :func:`init` method. In the base implementation, the former calls the latter
    and the latter does nothing. At a bare minimum, subclasses must set the
    `Ndim` parameter to the number of dynamic field dimensions in the model.

    One of the main jobs of this class is to provide an easy interface for
    calculating the phase structure and phase transitions. These are given by
    the methods :func:`getPhases`, :func:`calcTcTrans`, and
    :func:`findAllTransitions`.

    The following attributes can (and should!) be set during initialiation:

    Attributes
    ----------
    x_eps : float
        The epsilon to use in brute-force evalutations of the gradient and
        for the second derivatives. May be overridden by subclasses;
        defaults to 0.001.
    T_eps : float
        The epsilon to use in brute-force evalutations of the temperature
        derivative. May be overridden by subclasses; defaults to 0.001.
    num_boson_dof : int or None
        Total number of bosonic degrees of freedom, including radiation.
        This is used to add a field-independent but temperature-dependent
        contribution to the effective potential. It does not affect the relative
        pressure or energy density between phases, so it does not affect the
        critical or nucleation temperatures. If None, the total number of
        degrees of freedom will be taken directly from :meth:`boson_massSq`.
    num_fermion_dof : int or None
        Total number of fermionic degrees of freedom, including radiation.
        If None, the total number of degrees of freedom will be taken
        directly from :meth:`fermion_massSq`.
    """
    def init(self, NMSSMparams, **dargs):
        """
        Subclasses should override this method (not __init__) to do all model
        initialization. At a bare minimum, subclasses need to specify the number
        of dimensions in the potential with ``self.Ndim``.
        """
        self.Ndim = 3 #The number of dynamic field dimensions in the model. This *must* be overridden by subclasses during initialization.
        self.renormScaleSq = mt2 #The square of the renormalization scale to use in the MS-bar one-loop zero-temp potential. May be overridden by subclasses; defaults to 1000.0**2.
        self.Tmax = 1e3 # The maximum temperature to which minima should be followed. No transitions are calculated above this temperature. This is also used as the overall temperature scale in :func:`getPhases`. May be overridden by subclasses; defaults to 1000.0.
        self.deriv_order = 4 #Sets the order to which finite difference derivatives are calculated. Must be 2 or 4. May be overridden by subclasses; defaults to 4.
        """ some parameters for the functions checking paramter points """
        self.SameXThreshold = 5. # distance between two minima in field space before their locations are considered identical.
        self.mh125Threshold = 3. # tolerance for deviation of the mass of the SM-like mass eigenstate from Sqrt[mh1252]
        self.C_h125_HNSMThreshold = 0.2 # tolerance for the H^SM - H^NSM mixing angle
        self.C_h125_HSThreshold = 0.5 # tolerance for the H^SM - H^S mixing angle
        """ IR regular [GeV^2] added to squared Goldstone masses """
        self.muIR2 = 1. 
        """ create parameters """
        self.lam = NMSSMparams[0]
        self.kap = NMSSMparams[1]
        self.tb = NMSSMparams[2]
        self.mu = NMSSMparams[3]
        self.Alam = NMSSMparams[4]
        self.Akap = NMSSMparams[5]
        self.dl2 = NMSSMparams[6]
        self.M1 = NMSSMparams[7]
        self.M2 = NMSSMparams[8]
    ############################################################
    def V0(self, X):
        """ 
        this is the 'tree-level' potential (as a function of the three 
        CP-even neutral fields in the Higgs basis) in our effective theory:
        the NMSSM after integrating out all sfermions and the gluions,
        keeping only the H_u^4 operator from that matching procedure
        """
        X = np.asanyarray(X)
        HSM, HNSM, HS = X[...,0], X[...,1], X[...,2]
        r = (3.*HNSM**4.*self.lam**2.*self.mu*self.tb*(mZ2*(-1.+self.tb**2.)**2.+2.*v2*(2.*self.lam**2.*self.tb**2.+self.dl2))+24.*HNSM**3.*HSM*self.lam**2.*self.mu*self.tb**2.*(mZ2-mZ2*self.tb**2.+v2*(self.lam**2.*(-1.+self.tb**2.)+self.dl2))-24.*HNSM*HSM*self.lam*self.mu*self.tb*(v2*(-HS**2.*self.kap*self.lam**2.*(-1.+self.tb**4.)-self.Alam*self.lam*(np.sqrt(2.)*HS*self.lam-2.*self.mu)*(-1.+self.tb**4.)+2.*(-1.+self.tb**2.)*(self.lam*mZ2*self.tb+self.kap*self.mu**2.*(1.+self.tb**2.)-self.lam**3.*self.tb*v2)+2.*self.lam*self.tb**3.*v2*self.dl2)+HSM**2.*self.lam*self.tb*(mZ2-mZ2*self.tb**2.+v2*(self.lam**2.*(-1.+self.tb**2.)-self.tb**2.*self.dl2)))+self.tb*(4.*HS**2.*(1.+self.tb**2.)*v2*(self.kap*self.mu*(HS*(2.*np.sqrt(2.)*self.Akap+3.*HS*self.kap)*self.lam**2.-6.*self.Akap*self.lam*self.mu-12.*self.kap*self.mu**2.)*(1.+self.tb**2.)-6.*self.lam**3.*(-2.*self.kap*self.mu*self.tb+self.lam*(self.mu-self.Alam*self.tb+self.mu*self.tb**2.))*v2)+3.*HSM**4.*self.lam**2.*self.mu*(mZ2*(-1.+self.tb**2.)**2.+2.*self.tb**2.*v2*(2.*self.lam**2.+self.tb**2.*self.dl2))-12.*HSM**2.*self.lam*self.mu*v2*(2.*self.mu**2.*(1.+self.tb**2.)*(self.lam-2.*self.kap*self.tb+self.lam*self.tb**2.)-4.*self.Alam*self.lam*self.mu*(self.tb+self.tb**3.)+self.lam*(mZ2-2.*mZ2*self.tb**2.+mZ2*self.tb**4.+2.*HS*(np.sqrt(2.)*self.Alam+HS*self.kap)*self.lam*(self.tb+self.tb**3.)+self.lam**2.*(-HS**2.*(1.+self.tb**2.)**2.+4.*self.tb**2.*v2)+2.*self.tb**4.*v2*self.dl2)))+6.*HNSM**2.*self.lam*self.mu*(HSM**2.*self.lam*self.tb*(-mZ2*(1.-10.*self.tb**2.+self.tb**4.)+2.*v2*(self.lam**2.*(1.-4.*self.tb**2.+self.tb**4.)+3.*self.tb**2.*self.dl2))+v2*(4.*self.Alam*self.lam*self.mu*(1.+self.tb**2.)*(1.+self.tb**4.)+4.*self.mu**2.*(1.+self.tb**2.)*(-self.lam*self.tb*(1.+self.tb**2.)+self.kap*(1.+self.tb**4.))+2.*self.lam*self.tb*(mZ2+2.*HS*(np.sqrt(2.)*self.Alam+HS*self.kap)*self.lam*(self.tb+self.tb**3.)+self.lam**2.*(HS**2.*(1.+self.tb**2.)**2.-2.*(1.+self.tb**4.)*v2)+self.tb**2.*(mZ2*(-2.+self.tb**2.)-2.*v2*self.dl2)))))/(48.*self.lam**2.*self.mu*self.tb*(1.+self.tb**2.)**2.*v2)
        return r
    ############################################################
    def findMinimumV0(self, X=None):
        """
        Convenience function for finding the nearest minimum to `X` of the tree level potential
        """
        if X is None:
            X = self.approxZeroTMin()[0]
        minLoc = optimize.fmin(self.V0, X, disp=0)
        return minLoc
    ############################################################
    def boson_massSq(self, X, T):
        """
        array_like field-dependent squared tree-level masses of the bosons, including T-dependent 
        Daisy corrections. 
        Arranged in basis [h1, h2, h3, a1, a2, a3, c1, c2, WL, WT, ZL, ZT, gaL, gaT]
        where h_i are the neutral CP-even states
        the a_i are the neutral CP-odd states (including the neutral Goldstone)
        the c_i are the charged states (including the charged Goldstones)
        and WT/WL, ZT/ZL, and gaT/gaL are the transversal/longitudinal W-bosons, 
        Z-boson, and photon
        """
        X = np.asanyarray(X)
        HSM, HNSM, HS = X[...,0], X[...,1], X[...,2]
        T2 = T*T 
        # entries of the CP-even neutral squared mass matrix in Higgs basis (H^SM, H^NSM, H^S)
        m2s11 = (3.*HSM**2.*self.lam*mZ2*(-1.+self.tb**2.)**2.-4.*self.lam*self.tb**2.*v2**2.*(2.*self.lam**2.+self.tb**2.*self.dl2)+12.*HNSM*HSM*self.lam*(self.tb*(-1.+self.tb**2.)*(mZ2-self.lam**2.*v2)+self.tb**3.*v2*self.dl2)+HNSM**2.*self.lam*(-mZ2*(1.-10.*self.tb**2.+self.tb**4.)+2.*v2*(self.lam**2.*(1.-4.*self.tb**2.+self.tb**4.)+3.*self.tb**2.*self.dl2))+2.*v2*(-2.*self.mu**2.*(1.+self.tb**2.)*(self.lam-2.*self.kap*self.tb+self.lam*self.tb**2.)+4.*self.Alam*self.lam*self.mu*(self.tb+self.tb**3.)+self.lam*(-mZ2*(-1.+self.tb**2.)**2.-2.*np.sqrt(2.)*self.Alam*HS*self.lam*self.tb*(1.+self.tb**2.)+HS**2.*self.lam*(1.+self.tb**2.)*(self.lam-2.*self.kap*self.tb+self.lam*self.tb**2.)+3.*HSM**2.*self.tb**2.*(2.*self.lam**2.+self.tb**2.*self.dl2))))/(4.*self.lam*(1.+self.tb**2.)**2.*v2)
        m2s12 = (3.*HSM**2.*self.lam*self.tb*((-1.+self.tb**2.)*(mZ2-self.lam**2.*v2)+self.tb**2.*v2*self.dl2)+3.*HNSM**2.*self.lam*self.tb*(mZ2-mZ2*self.tb**2.+v2*(self.lam**2.*(-1.+self.tb**2.)+self.dl2))+HNSM*HSM*self.lam*(-mZ2*(1.-10.*self.tb**2.+self.tb**4.)+2.*v2*(self.lam**2.*(1.-4.*self.tb**2.+self.tb**4.)+3.*self.tb**2.*self.dl2))+v2*(HS**2.*self.kap*self.lam**2.*(-1.+self.tb**4.)+self.Alam*self.lam*(np.sqrt(2.)*HS*self.lam-2.*self.mu)*(-1.+self.tb**4.)-2.*(self.kap*self.mu**2.*(-1.+self.tb**4.)+self.lam*self.tb*(-1.+self.tb**2.)*(mZ2-self.lam**2.*v2)+self.lam*self.tb**3.*v2*self.dl2)))/(2.*self.lam*(1.+self.tb**2.)**2.*v2)
        m2s13 = HS*HSM*self.lam**2.+((np.sqrt(2.)*self.Alam+2.*HS*self.kap)*self.lam*(-2.*HSM*self.tb+HNSM*(-1.+self.tb**2.)))/(2.*(1.+self.tb**2.))
        m2s22 = (self.lam*mZ2*self.tb*(-12.*HNSM*HSM*self.tb*(-1.+self.tb**2.)+3.*HNSM**2.*(-1.+self.tb**2.)**2.-HSM**2.*(1.-10.*self.tb**2.+self.tb**4.))-4.*self.lam*self.tb*v2**2.*(self.lam**2.*(1.+self.tb**4.)+self.tb**2.*self.dl2)+2.*v2*((HS**2.+HSM**2.)*self.lam**3.*self.tb+2.*self.Alam*self.lam*(1.+self.tb**2.)*(self.mu+np.sqrt(2.)*HS*self.lam*self.tb**2.+self.mu*self.tb**4.)+2.*self.kap*(1.+self.tb**2.)*(HS**2.*self.lam**2.*self.tb**2.+self.mu**2.*(1.+self.tb**4.))+self.lam*self.tb*(mZ2*(-1.+self.tb**2.)**2.-2.*self.mu**2.*(1.+self.tb**2.)**2.+self.lam**2.*self.tb*(6.*HNSM**2.*self.tb+HSM**2.*self.tb*(-4.+self.tb**2.)+6.*HNSM*HSM*(-1.+self.tb**2.)+HS**2.*self.tb*(2.+self.tb**2.))+3.*(HNSM+HSM*self.tb)**2.*self.dl2)))/(4.*self.lam*self.tb*(1.+self.tb**2.)**2.*v2)
        m2s23 = HNSM*HS*self.lam**2.+((np.sqrt(2.)*self.Alam+2.*HS*self.kap)*self.lam*(2.*HNSM*self.tb+HSM*(-1.+self.tb**2.)))/(2.*(1.+self.tb**2.))
        m2s33 = np.sqrt(2.)*self.Akap*HS*self.kap+3.*HS**2.*self.kap**2.-(self.Akap*self.kap*self.mu)/self.lam-(2.*self.kap**2.*self.mu**2.)/self.lam**2.+(self.Alam*self.lam**2.*self.tb*v2)/(self.mu+self.mu*self.tb**2.)+(1./(2.*(1.+self.tb**2.)))*self.lam*(2.*HNSM*HSM*self.kap*(-1.+self.tb**2.)+HNSM**2.*(self.lam+2.*self.kap*self.tb+self.lam*self.tb**2.)+(self.lam-2.*self.kap*self.tb+self.lam*self.tb**2.)*(HSM**2.-2.*v2))
        # entries of the CP-odd neutral squared mass matrix in basis (A^NSM, A^S, G^0)
        m2a11 = (self.lam*mZ2*self.tb*(HNSM-HSM+(HNSM+HSM)*self.tb)*(-1.+self.tb**2.)*(HNSM*(-1.+self.tb)-HSM*(1.+self.tb))-4.*self.lam*self.tb*v2**2.*(self.lam**2.*(1.+self.tb**4.)+self.tb**2.*self.dl2)+2.*v2*(2.*self.Alam*self.lam*self.mu*(1.+self.tb**2.)*(1.+self.tb**4.)+2.*self.mu**2.*(1.+self.tb**2.)*(-self.lam*self.tb*(1.+self.tb**2.)+self.kap*(1.+self.tb**4.))+self.lam*self.tb*(mZ2-2.*mZ2*self.tb**2.+mZ2*self.tb**4.+2.*HS*(np.sqrt(2.)*self.Alam+HS*self.kap)*self.lam*(self.tb+self.tb**3.)+self.lam**2.*(2.*HNSM**2.*self.tb**2.+2.*HNSM*HSM*self.tb*(-1.+self.tb**2.)+HS**2.*(1.+self.tb**2.)**2.+HSM**2.*(1.+self.tb**4.))+HNSM**2.*self.dl2+2.*HNSM*HSM*self.tb*self.dl2+HSM**2.*self.tb**2.*self.dl2)))/(4.*self.lam*self.tb*(1.+self.tb**2.)**2.*v2)
        m2a12 = (self.Alam*HSM*self.lam)/np.sqrt(2.)-HS*HSM*self.kap*self.lam
        m2a13 = (HSM**2.*self.lam*self.tb*((-1.+self.tb**2.)*(mZ2-self.lam**2.*v2)+self.tb**2.*v2*self.dl2)+2.*HNSM*HSM*self.lam*self.tb**2.*(2.*mZ2+v2*(-2.*self.lam**2.+self.dl2))+HNSM**2.*self.lam*self.tb*(mZ2-mZ2*self.tb**2.+v2*(self.lam**2.*(-1.+self.tb**2.)+self.dl2))+v2*(HS**2.*self.kap*self.lam**2.*(-1.+self.tb**4.)+self.Alam*self.lam*(np.sqrt(2.)*HS*self.lam-2.*self.mu)*(-1.+self.tb**4.)-2.*(self.kap*self.mu**2.*(-1.+self.tb**4.)+self.lam*self.tb*(-1.+self.tb**2.)*(mZ2-self.lam**2.*v2)+self.lam*self.tb**3.*v2*self.dl2)))/(2.*self.lam*(1.+self.tb**2.)**2.*v2)
        m2a22 = 1./2.*(2.*HS**2.*self.kap**2.-2.*HNSM*HSM*self.kap*self.lam+HNSM**2.*self.lam**2.+HSM**2.*self.lam**2.-(4.*self.kap**2.*self.mu**2.)/self.lam**2.-(2.*self.Akap*self.kap*(np.sqrt(2.)*HS*self.lam+self.mu))/self.lam-2.*self.lam**2.*v2)+(1./(self.mu*(1.+self.tb**2.)))*(self.kap*self.lam*self.mu*(2.*HNSM*HSM-HNSM**2.*self.tb+HSM**2.*self.tb)+self.lam*(self.Alam*self.lam+2.*self.kap*self.mu)*self.tb*v2)
        m2a23 = -((self.Alam*HNSM*self.lam)/np.sqrt(2.))+HNSM*HS*self.kap*self.lam
        m2a33 = -((self.lam*mZ2*(HNSM-HSM+(HNSM+HSM)*self.tb)*(-1.+self.tb**2.)*(HNSM*(-1.+self.tb)-HSM*(1.+self.tb))+4.*self.lam*self.tb**2.*v2**2.*(2.*self.lam**2.+self.tb**2.*self.dl2)-2.*v2*(-2.*self.mu**2.*(1.+self.tb**2.)*(self.lam-2.*self.kap*self.tb+self.lam*self.tb**2.)+4.*self.Alam*self.lam*self.mu*(self.tb+self.tb**3.)+self.lam*(-mZ2-2.*HS*(np.sqrt(2.)*self.Alam+HS*self.kap)*self.lam*(self.tb+self.tb**3.)+self.lam**2.*(2.*HSM**2.*self.tb**2.-2.*HNSM*HSM*self.tb*(-1.+self.tb**2.)+HS**2.*(1.+self.tb**2.)**2.+HNSM**2.*(1.+self.tb**4.))+self.tb**2.*(-mZ2*(-2.+self.tb**2.)+(HNSM+HSM*self.tb)**2.*self.dl2))))/(4.*self.lam*(1.+self.tb**2.)**2.*v2))
        # entries of the charged squared mass matrix in basis (H^+, G-)
        m2c11 = (self.lam*self.tb*(-4.*HNSM*HSM*mZ2*self.tb*(-1.+self.tb**2.)+HNSM**2.*mZ2*(-1.+self.tb**2.)**2.+HSM**2.*(-mZ2*(-1.+self.tb**2.)**2.+2.*mW2*(1.+self.tb**2.)**2.))-4.*self.lam*self.tb*v2**2.*(self.lam**2.*(1.+self.tb**4.)+self.tb**2.*self.dl2)+2.*v2*(2.*self.Alam*self.lam*self.mu*(1.+self.tb**2.)*(1.+self.tb**4.)+2.*self.mu**2.*(1.+self.tb**2.)*(-self.lam*self.tb*(1.+self.tb**2.)+self.kap*(1.+self.tb**4.))+self.lam*self.tb*(mZ2*(-1.+self.tb**2.)**2.+2.*np.sqrt(2.)*self.Alam*HS*self.lam*self.tb*(1.+self.tb**2.)+HS**2.*self.lam*(1.+self.tb**2.)*(self.lam+2.*self.kap*self.tb+self.lam*self.tb**2.)+(HNSM+HSM*self.tb)*(2.*self.lam**2.*self.tb*(-HSM+HNSM*self.tb)+(HNSM+HSM*self.tb)*self.dl2))))/(4.*self.lam*self.tb*(1.+self.tb**2.)**2.*v2)
        m2c12 = (HSM**2.*self.lam*self.tb*((-1.+self.tb**2.)*(mZ2-self.lam**2.*v2)+self.tb**2.*v2*self.dl2)+HNSM**2.*self.lam*self.tb*(mZ2-mZ2*self.tb**2.+v2*(self.lam**2.*(-1.+self.tb**2.)+self.dl2))+HNSM*HSM*self.lam*(4.*mZ2*self.tb**2.-mW2*(1.+self.tb**2.)**2.+v2*(self.lam**2.*(-1.+self.tb**2.)**2.+2.*self.tb**2.*self.dl2))+v2*(HS**2.*self.kap*self.lam**2.*(-1.+self.tb**4.)+self.Alam*self.lam*(np.sqrt(2.)*HS*self.lam-2.*self.mu)*(-1.+self.tb**4.)-2.*(self.kap*self.mu**2.*(-1.+self.tb**4.)+self.lam*self.tb*(-1.+self.tb**2.)*(mZ2-self.lam**2.*v2)+self.lam*self.tb**3.*v2*self.dl2)))/(2.*self.lam*(1.+self.tb**2.)**2.*v2)
        m2c22 = (self.lam*(4.*HNSM*HSM*mZ2*self.tb*(-1.+self.tb**2.)+HSM**2.*mZ2*(-1.+self.tb**2.)**2.+HNSM**2.*(-mZ2*(-1.+self.tb**2.)**2.+2.*mW2*(1.+self.tb**2.)**2.))-4.*self.lam*self.tb**2.*v2**2.*(2.*self.lam**2.+self.tb**2.*self.dl2)+2.*v2*(-2.*self.mu**2.*(1.+self.tb**2.)*(self.lam-2.*self.kap*self.tb+self.lam*self.tb**2.)+4.*self.Alam*self.lam*self.mu*(self.tb+self.tb**3.)+self.lam*(-mZ2*(-1.+self.tb**2.)**2.-2.*np.sqrt(2.)*self.Alam*HS*self.lam*self.tb*(1.+self.tb**2.)+HS**2.*self.lam*(1.+self.tb**2.)*(self.lam-2.*self.kap*self.tb+self.lam*self.tb**2.)+self.tb*(HNSM+HSM*self.tb)*(2.*self.lam**2.*(HSM-HNSM*self.tb)+self.tb*(HNSM+HSM*self.tb)*self.dl2))))/(4.*self.lam*(1.+self.tb**2.)**2.*v2)
        # gauge bosons masses
        m2WL = ((HNSM**2.+HSM**2.)*mW2)/(2.*v2)
        m2WT = np.copy(m2WL)
        m2W3BL = -(1./2.)*(HNSM**2.+HSM**2.)*np.sqrt(mW2/v2)*np.sqrt((-mW2+mZ2)/v2)
        m2BBL = -(((HNSM**2.+HSM**2.)*(mW2-mZ2))/(2.*v2))
        m2W3BT = np.copy(m2W3BL)
        m2BBT = np.copy(m2BBL)
        # Daisy coefficients
        cdaisy_s11 = (1./(4.*(1.+self.tb**2.)*v2))*(mZ2+mZ2*self.tb**2.+mt2*(1.+self.tb**2.)+2.*mW2*(1.+self.tb**2.)+self.lam**2.*v2+self.lam**2.*self.tb**2.*v2+self.tb**2.*v2*self.dl2)
        cdaisy_s12 = mt2/(4.*self.tb*v2)+(self.tb*self.dl2)/(4.+4.*self.tb**2.)
        cdaisy_s22 = (2.*mW2+mZ2+mt2/self.tb**2.+self.lam**2.*v2+(v2*self.dl2)/(1.+self.tb**2.))/(4.*v2)
        cdaisy_s33 = 1./2.*(self.kap**2.+self.lam**2.)
        cdaisy_c11 = (4.*mW2+2.*mZ2+(3.*mt2)/self.tb**2.+2.*self.lam**2.*v2+(3.*v2*self.dl2)/(1.+self.tb**2.))/(12.*v2)
        cdaisy_c22 = (1./(12.*(1.+self.tb**2.)*v2))*(2.*mZ2+2.*mZ2*self.tb**2.+3.*mt2*(1.+self.tb**2.)+4.*mW2*(1.+self.tb**2.)+2.*self.lam**2.*v2+2.*self.lam**2.*self.tb**2.*v2+3.*self.tb**2.*v2*self.dl2)
        cdaisy_W3 =  m2WL = ((HNSM**2.+HSM**2.)*mW2)/(2.*v2)
        cdaisy_WL = 2.*(5./2.)*mW2/v2
        cdaisy_BL = -2.*(13./6.)*(mW2-mZ2)/v2
        # add Daisy contributions to masses
        m2s11 += cdaisy_s11*T2
        m2s12 += cdaisy_s12*T2
        m2s22 += cdaisy_s22*T2
        m2s33 += cdaisy_s33*T2
        m2a11 += cdaisy_s22*T2
        m2a13 += cdaisy_s12*T2
        m2a22 += cdaisy_s33*T2
        m2a33 += cdaisy_s11*T2
        m2c11 += cdaisy_c11*T2
        m2c12 += cdaisy_s12*T2
        m2c22 += cdaisy_c22*T2
        m2WL += cdaisy_WL*T2
        m2BBL += cdaisy_BL*T2
        # diagonalize mass matrices
        # neutral even Higgses
        m2s = np.array([[m2s11, m2s12, m2s13], [m2s12, m2s22, m2s23], [m2s13, m2s23, m2s33]])
        m2s = np.rollaxis(m2s, 0, len(m2s.shape))
        m2s = np.rollaxis(m2s, 0, len(m2s.shape))
        evalss = np.linalg.eigvalsh(m2s)
        for _ in range(len(evalss.shape)-1):
            evalss = np.rollaxis(evalss, 0, len(evalss.shape))
        m2h1, m2h2, m2h3 = evalss
        # neutral odd Higgses
        m2a = np.array([[m2a11, m2a12, m2a13], [m2a12, m2a22, m2a23], [m2a13, m2a23, m2a33]])
        m2a = np.rollaxis(m2a, 0, len(m2a.shape))
        m2a = np.rollaxis(m2a, 0, len(m2a.shape))
        evalsa = np.linalg.eigvalsh(m2a)
        for _ in range(len(evalsa.shape)-1):
            evalsa = np.rollaxis(evalsa, 0, len(evalsa.shape))
        # add the IR regulator to the entries of the Goldstones
        for ind, val in np.ndenumerate(evalsa):
            if np.abs(val) < self.muIR2:
                evalsa[ind] += self.muIR2
        # charged Higgses
        m2a1, m2a2, m2a3 = evalsa
        m2c = np.array([[m2c11,m2c12],[m2c12,m2c22]])
        m2c = np.rollaxis(m2c, 0, len(m2c.shape))
        m2c = np.rollaxis(m2c, 0, len(m2c.shape))
        evalsc = np.linalg.eigvalsh(m2c)
        for _ in range(len(evalsc.shape)-1):
            evalsc = np.rollaxis(evalsc, 0, len(evalsc.shape))
        # add the IR regulator to the entries of the Goldstones
        for ind, val in np.ndenumerate(evalsc):
            if np.abs(val) < self.muIR2:
                evalsc[ind] += self.muIR2
        m2c1, m2c2 = evalsc
        # photon-Z system
        m2ZgaT = np.array([[m2WT, m2W3BT],[m2W3BT, m2BBT]])
        m2ZgaT = np.rollaxis(m2ZgaT, 0, len(m2ZgaT.shape))
        m2ZgaT = np.rollaxis(m2ZgaT, 0, len(m2ZgaT.shape))
        evalsZgaT = np.linalg.eigvalsh(m2ZgaT)
        for _ in range(len(evalsZgaT.shape)-1):
            evalsZgaT = np.rollaxis(evalsZgaT, 0, len(evalsZgaT.shape))
        m2gaT, m2ZT = evalsZgaT
        m2ZgaL = np.array([[m2WL, m2W3BL],[m2W3BL, m2BBL]])
        m2ZgaL = np.rollaxis(m2ZgaL, 0, len(m2ZgaL.shape))
        m2ZgaL = np.rollaxis(m2ZgaL, 0, len(m2ZgaL.shape))
        evalsZgaL = np.linalg.eigvalsh(m2ZgaL)
        for _ in range(len(evalsZgaL.shape)-1):
            evalsZgaL = np.rollaxis(evalsZgaL, 0, len(evalsZgaL.shape))
        m2gaL, m2ZL = evalsZgaL
        # prepare output
        massSq = np.array([m2h1, m2h2, m2h3, m2a1, m2a2, m2a3, m2c1, m2c2, m2WL, m2WT, m2ZL, m2ZT, m2gaL, m2gaT])
        massSq = np.rollaxis(massSq, 0, len(massSq.shape))
        dof = np.array([1., 1., 1., 1., 1., 1., 2., 2., 2., 4., 1., 2., 1., 2.]) # The number of degrees of freedom for each particle
        c = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5]) # constant entering the CW potential potential
        return massSq, dof, c
    ############################################################
    def fermion_massSq(self, X):
        """
        array_like field-dependent squared tree-level masses of the fermions. 
        Arranged in basis [neu,cha,top]
        where neu are the neutralinos
        the cha are the charginos
        and top is the top quark
        """
        X = np.asanyarray(X)
        HSM, HNSM, HS = X[...,0], X[...,1], X[...,2]
        # entries of the neutralino mass matrix squared
        m2neu11 = self.M1**2.-((HNSM**2.+HSM**2.)*(mW2-mZ2))/(2.*v2)
        m2neu12 = -(1./2.)*(HNSM**2.+HSM**2.)*np.sqrt(mW2/v2)*np.sqrt((-mW2+mZ2)/v2)
        m2neu13 = -(((np.sqrt(2.)*self.M1*(HSM-HNSM*self.tb)+HS*self.lam*(HNSM+HSM*self.tb))*np.sqrt((-mW2+mZ2)/v2))/(2.*np.sqrt(1.+self.tb**2.)))
        m2neu14 = ((HS*self.lam*(HSM-HNSM*self.tb)+np.sqrt(2.)*self.M1*(HNSM+HSM*self.tb))*np.sqrt((-mW2+mZ2)/v2))/(2.*np.sqrt(1.+self.tb**2.))
        m2neu15 = np.zeros(HSM.shape)
        m2neu22 = self.M2**2.+((HNSM**2.+HSM**2.)*mW2)/(2.*v2)
        m2neu23 = ((np.sqrt(2.)*self.M2*(HSM-HNSM*self.tb)+HS*self.lam*(HNSM+HSM*self.tb))*np.sqrt(mW2/v2))/(2.*np.sqrt(1.+self.tb**2.))
        m2neu24 = -(((HS*self.lam*(HSM-HNSM*self.tb)+np.sqrt(2.)*self.M2*(HNSM+HSM*self.tb))*np.sqrt(mW2/v2))/(2.*np.sqrt(1.+self.tb**2.)))
        m2neu25 = np.zeros(HSM.shape)
        m2neu33 = (mZ2*(HSM-HNSM*self.tb)**2.+self.lam**2.*(HNSM**2.+HS**2.+2.*HNSM*HSM*self.tb+(HS**2.+HSM**2.)*self.tb**2.)*v2)/(2.*(1.+self.tb**2.)*v2)
        m2neu34 = ((HSM-HNSM*self.tb)*(HNSM+HSM*self.tb)*(-mZ2+self.lam**2.*v2))/(2.*(1.+self.tb**2.)*v2)
        m2neu35 = -((HS*self.lam*(2.*HNSM*self.kap-HSM*self.lam+2.*HSM*self.kap*self.tb+HNSM*self.lam*self.tb))/(2.*np.sqrt(1.+self.tb**2.)))
        m2neu44 = (mZ2*(HNSM+HSM*self.tb)**2.+self.lam**2.*((HSM-HNSM*self.tb)**2.+HS**2.*(1.+self.tb**2.))*v2)/(2.*(1.+self.tb**2.)*v2)
        m2neu45 = (HS*self.lam*(-2.*HSM*self.kap+HNSM*self.lam+2.*HNSM*self.kap*self.tb+HSM*self.lam*self.tb))/(2.*np.sqrt(1.+self.tb**2.))
        m2neu55 = 2.*HS**2.*self.kap**2.+1./2.*(HNSM**2.+HSM**2.)*self.lam**2.
        # entries of the chargino mass matrix squared
        m2cha11 = self.M2**2.+(mW2*(HSM-HNSM*self.tb)**2.)/((1.+self.tb**2.)*v2)
        m2cha12 = ((np.sqrt(2.)*HS*self.lam*(HSM-HNSM*self.tb)+2.*self.M2*(HNSM+HSM*self.tb))*np.sqrt(mW2/v2))/(2.*np.sqrt(1.+self.tb**2.))
        m2cha13 = np.zeros(HSM.shape)
        m2cha14 = np.zeros(HSM.shape)
        m2cha22 = (HS**2.*self.lam**2.)/2.+(mW2*(HNSM+HSM*self.tb)**2.)/((1.+self.tb**2.)*v2)
        m2cha23 = np.zeros(HSM.shape)
        m2cha24 = np.zeros(HSM.shape)
        m2cha33 = self.M2**2.+(mW2*(HNSM+HSM*self.tb)**2.)/((1.+self.tb**2.)*v2)
        m2cha34 = ((np.sqrt(2.)*HNSM*HS*self.lam+2.*HSM*self.M2+np.sqrt(2.)*HS*HSM*self.lam*self.tb-2.*HNSM*self.M2*self.tb)*np.sqrt(mW2/v2))/(2.*np.sqrt(1.+self.tb**2.))
        m2cha44 = (HS**2.*self.lam**2.)/2.+(mW2*(HSM-HNSM*self.tb)**2.)/((1.+self.tb**2.)*v2)
        # top mass squared
        m2top = (mt2*(HNSM+HSM*self.tb)**2.)/(2.*self.tb**2.*v2)
        # diagonalize mass matrices
        m2neu = np.array([[m2neu11, m2neu12, m2neu13, m2neu14, m2neu15], [m2neu12, m2neu22, m2neu23, m2neu24, m2neu25], [m2neu13, m2neu23, m2neu33, m2neu34, m2neu35], [m2neu14, m2neu24, m2neu34, m2neu44, m2neu45], [m2neu15, m2neu25, m2neu35, m2neu45, m2neu55]])
        m2neu = np.rollaxis(m2neu, 0, len(m2neu.shape))
        m2neu = np.rollaxis(m2neu, 0, len(m2neu.shape))
        evalsneu = np.linalg.eigvalsh(m2neu)
        for i in range(len(evalsneu.shape)-1):
            evalsneu = np.rollaxis(evalsneu, 0, len(evalsneu.shape))
        m2neu1, m2neu2, m2neu3, m2neu4, m2neu5 = evalsneu
        m2cha = np.array([[m2cha11, m2cha12, m2cha13, m2cha14], [m2cha12, m2cha22, m2cha23, m2cha24], [m2cha13, m2cha23, m2cha33, m2cha34], [m2cha14, m2cha24, m2cha34, m2cha44]])
        m2cha = np.rollaxis(m2cha, 0, len(m2cha.shape))
        m2cha = np.rollaxis(m2cha, 0, len(m2cha.shape))
        evalscha = np.linalg.eigvalsh(m2cha)
        for i in range(len(evalscha.shape)-1):
            evalscha = np.rollaxis(evalscha, 0, len(evalscha.shape))
        m2cha1, m2cha2, m2cha3, m2cha4 = evalscha
        # prepare output
        massSq = np.array([m2neu1, m2neu2, m2neu3, m2neu4, m2neu5, m2cha1, m2cha2, m2cha3, m2cha4, m2top])
        massSq = np.rollaxis(massSq, 0, len(massSq.shape))
        dof = np.array([2., 2., 2., 2., 2., 2., 2., 2., 2., 12.]) # note that since we entered the masses of the charginos in the basis of 4 Majorana states, they have 2 degrees of freedom each only
        return massSq, dof
    ############################################################
    def V1T0noCT(self, X, T, include_radiation=False):
        """
        The zero temperature potential without the counterterms 
        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : dummy, included so that the cosmoTransitions helper_functions can be used
        include_radiation : dummy, included so that the cosmoTransitions helper_functions can be used

        """
        T = 0.
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        return y
    ############################################################
    def set_CT_params(self):
        """
        find the coefficients of the counterterm potential to maintain 
        - the location of the physical minimum
        - the mass of the SM-like Higgs interaction eigenstate
        - singlet-doublet alignment
        """
        Xphys=np.array([np.sqrt(2.*v2), 0., np.sqrt(2.)*self.mu/self.lam])
        # create the gradient function
        fd = helper_functions.gradientFunction(self.V1T0noCT, self.x_eps, self.Ndim, self.deriv_order)
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = 0.
        T = np.asanyarray(T)[...,np.newaxis,np.newaxis]
        # get the first derivatives at the physical minimum
        DV1, DV2, DV3 = fd(Xphys, T, False)
        # create the hessian function
        dh = helper_functions.hessianFunction(self.V1T0noCT, self.x_eps, self.Ndim, self.deriv_order)
        # get the hessian
        hess = dh(Xphys, T, False)
        DV11 = hess[0,0]
        DV13 = hess[0,2]
        # solve for the coefficients of the counterterms
        self.dmHu2=(1./(4.*self.lam*self.tb**2.*np.sqrt(v2)))*(-2.*np.sqrt(2.)*DV2*self.lam*self.tb+2.*DV13*self.mu*(1.+self.tb**2.)-np.sqrt(2.)*DV1*self.lam*(1.+3.*self.tb**2.)+2.*self.lam*(DV11-mh1252)*(1.+self.tb**2.)*np.sqrt(v2))
        self.dmHd2=(-np.sqrt(2.)*DV1*self.lam+np.sqrt(2.)*DV2*self.lam*self.tb+DV13*self.mu*(1.+self.tb**2.))/(2.*self.lam*np.sqrt(v2))
        self.dms2=(-np.sqrt(2.)*DV3*self.lam+DV13*self.lam*np.sqrt(v2))/(2.*self.mu)
        self.ddl2=((1.+self.tb**2.)**2.*(np.sqrt(2.)*DV1+2.*(-DV11+mh1252)*np.sqrt(v2)))/(4.*self.tb**4.*v2**1.5)
        self.dlAlam=(DV13*(1.+self.tb**2.))/(2.*self.tb*np.sqrt(v2))
    #############################################################
    def V1CT(self, X):
        """ 
        this are the counterterms we add to the potential
        in order to maintain 
        - the location of the physical minimum
        - the mass of the SM-like Higgs interaction eigenstate
        - singlet-doublet alignment
        """
        X=np.asanyarray(X)
        HSM, HNSM, HS = X[...,0], X[...,1], X[...,2]
        # check if the coefficients of the coutnerterms are already determined
        try:
            self.dmHu2
            self.dmHd2
            self.dms2
            self.ddl2
            self.dlAlam
        except:
            self.set_CT_params()
        r = (1./(8.*(1.+self.tb**2.)**2.))*(self.ddl2*(HNSM+HSM*self.tb)**4.+4.*self.dmHu2*(HNSM+HSM*self.tb)**2.*(1.+self.tb**2.)+4.*(1.+self.tb**2.)*(self.dmHd2*(HSM-HNSM*self.tb)**2.+np.sqrt(2.)*self.dlAlam*HS*(-HSM+HNSM*self.tb)*(HNSM+HSM*self.tb)+self.dms2*HS**2.*(1.+self.tb**2.)))
        return r
    #############################################################
    def V1T(self, bosons, fermions, T, include_radiation=False):
        """
        The one-loop finite-temperature potential.
        
        This is the default function from cosmoTransitions.generic_potential
        except that include_radiation defaults to False

        This is generally not called directly, but is instead used by
        :func:`Vtot`.

        Note
        ----
        The `Jf` and `Jb` functions used here are
        aliases for :func:`finiteT.Jf_spline` and :func:`finiteT.Jb_spline`,
        each of which accept mass over temperature *squared* as inputs
        (this allows for negative mass-squared values, which I take to be the
        real part of the defining integrals.

        .. todo::
            Implement new versions of Jf and Jb that return zero when m=0, only
            adding in the field-independent piece later if
            ``include_radiation == True``. This should reduce floating point
            errors when taking derivatives at very high temperature, where
            the field-independent contribution is much larger than the
            field-dependent contribution.
        """
        # This does not need to be overridden.
        T2 = (T*T)[..., np.newaxis] + 1e-100
             # the 1e-100 is to avoid divide by zero errors
        T4 = T*T*T*T
        m2, nb, c = bosons
        y = np.sum(nb*Jb(m2/T2), axis=-1)
        m2, nf = fermions
        y += np.sum(nf*Jf(m2/T2), axis=-1)
        if include_radiation:
            if self.num_boson_dof is not None:
                nb = self.num_boson_dof - np.sum(nb)
                y -= nb * np.pi**4 / 45.
            if self.num_fermion_dof is not None:
                nf = self.num_fermion_dof - np.sum(nf)
                y -= nf * 7*np.pi**4 / 360.
        return y*T4/(2*np.pi*np.pi)
    #############################################################
    def V1T_from_X(self, X, T, include_radiation=False):
        """
        Calculates the mass matrix and resulting one-loop finite-T potential.

        This is the default function from cosmoTransitions.generic_potential
        except that include_radiation defaults to False
        
        Useful when calculate temperature derivatives, when the zero-temperature
        contributions don't matter.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V1T(bosons, fermions, T, include_radiation)
        return y
    #############################################################
    def Vtot(self, X, T, include_radiation=False):
        """
        The total finite temperature effective potential.

        This is the default function from cosmoTransitions.generic_potential
        except that include_radiation defaults to False
        and that the counterterm potential is included
        

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1CT(X)
        y += self.V1T(bosons, fermions, T, include_radiation)
        return y
    #############################################################
    def energyDensity(self,X,T,include_radiation=False):
        """
        This is the default function from cosmoTransitions.generic_potential
        except that include_radiation defaults to False
        """
        T_eps = self.T_eps
        if self.deriv_order == 2:
            dVdT = self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT *= 1./(2*T_eps)
        else:
            dVdT = self.V1T_from_X(X,T-2*T_eps, include_radiation)
            dVdT -= 8*self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT += 8*self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T+2*T_eps, include_radiation)
            dVdT *= 1./(12*T_eps)
        V = self.Vtot(X,T, include_radiation)
        return V - T*dVdT
    #############################################################
    def findTMinX(self,T):
        """
        Returns list of field locations minima at temperature T,
        starting from a grid of inital guesses startX
        """
        startX = np.meshgrid([-1000.,-100.,-10.,0.,10.,100.,1000],
            [-1000.,-100.,-10.,0.,10.,100.,1000],
            [-1000.,-100.,-10.,0.,10.,100.,1000])
        startX = np.asanyarray(startX)
        startX = np.rollaxis(startX,0,len(startX.shape))
        startX = startX.reshape(-1,startX.shape[-1])
        Xphys = np.array([np.sqrt(2.*v2), 0, np.sqrt(2)*self.mu/self.lam])
        locations = [self.findMinimum(Xphys,T=T)]
        for guessX in startX:
            locations.append(self.findMinimum(guessX,T=T))
        # use $Z_2$ symmetry of potential to choose <HSM> positive
        for i in range(len(locations)):
            if locations[i][0]<0:
                locations[i][0:2] = -locations[i][0:2]
        # find unique locations
        uniqueX=[]
        uniqueX.append(np.copy(locations.pop(0)))
        for X in locations:
            flag = True
            if np.linalg.norm(X) > 1e6:
                flag = False
            if flag:
                for Y in uniqueX:
                    if np.linalg.norm(X-Y) < self.SameXThreshold:
                        flag = False
                        break
            if flag:
                uniqueX.append(np.copy(X))
        return uniqueX
    #############################################################
    def approxZeroTMin(self):
        """
        Returns list field locations of the zero-temperature minima,
        using the findTMinX function
        The minima are also stored in self.ZeroTMinLocs
        """
        try:
            self.ZeroTMinLocs
        except:
            self.ZeroTMinLocs = self.findTMinX(0.)
        return self.ZeroTMinLocs
    #############################################################
    def checkZeroTVacuumStructure(self):
        """ 
        checks the zero temperature vacuum structure 

        Returns boolean flag:
            True if physical minimum is the global minimum
            False else
        """
        try:
            self.ZeroTMinLocs
        except:
            self.approxZeroTMin()
        minPotVal = self.Vtot(self.ZeroTMinLocs, 0.) # get potential values at minima
        Xglob = self.ZeroTMinLocs[np.argmin(minPotVal)] # get field location of global minimum
        Xphys = np.array([np.sqrt(2.*v2), 0, np.sqrt(2)*self.mu/self.lam]) # field location of the physical minimum
        flag = np.linalg.norm(Xglob-Xphys) < self.SameXThreshold
        return flag
    #############################################################
    def checkTmaxVacuumStructure(self):
        """ 
        checks the vacuum structure at Tmax

        Returns boolean flag:
            True if the global minimum at high T preserves EW symmetry
            False else
        """
        try:
            self.TmaxMinLocs
        except:
            self.TmaxMinLocs = self.findTMinX(self.Tmax)
        minPotVal = self.Vtot(self.TmaxMinLocs, self.Tmax)
        Xglob = self.TmaxMinLocs[np.argmin(minPotVal)]
        flag = np.linalg.norm(Xglob[0:2]-np.array([0.,0.])) < self.SameXThreshold
        return flag
    #############################################################
    def checkh125(self):
        """
        checks the CP-even neutral mass matrix at the physical minimum
        to ensure that
        - no mass eigenstate is tachyonic
        - there is a dominantly H^SM mass eigenstate with mass ~125 GeV
        - approximate alignment is satisfied

        Returns boolean flag:
            True if all checks passes,
            False else
        """
        Xphys = np.array([np.sqrt(2.*v2), 0, np.sqrt(2.)*self.mu/self.lam]) # field location of the physical minimum
        MS2 = self.d2V(Xphys, 0.)
        flag = True
        if np.linalg.det(MS2)<0:
            flag = False
        esys = np.linalg.eigh(MS2)
        masses = np.sqrt(esys[0])
        # find state with largest HSM component
        h125ind = np.argmax(np.abs(esys[1][0,:]))
        if np.abs(masses[h125ind]-np.sqrt(mh1252)) > self.mh125Threshold:
            flag = False
        if np.abs(esys[1][1,h125ind]) > self.C_h125_HNSMThreshold:
            flag = False
        if np.abs(esys[1][2,h125ind]) > self.C_h125_HSThreshold:
            flag = False
        return flag
    #############################################################
    def HiggsMassEigenstates(self):
        """
        returns tuple (masses, mixing angles) of Higgs mass eigenstates
        The mixing angles are given in the Higgs basis, and
        - the first entry corresponds to the SM-like mass eigentate
        - the second entry corresponds to the JSM-like mass eigentate
        - the third entry corresponds to the singlet-like mass eigentate
        """
        Xphys = np.array([np.sqrt(2.*v2), 0, np.sqrt(2.)*self.mu/self.lam]) # field location of the physical minimum
        MS2 = self.d2V(Xphys, 0.)
        esys = np.linalg.eigh(MS2)
        masses = np.sqrt(esys[0])
        h125ind = np.argmax(np.abs(esys[1][0,:]))
        Hind = np.argmax(np.abs(esys[1][1,:]))
        hSind = np.argmax(np.abs(esys[1][2,:]))
        return (masses[[h125ind,Hind,hSind]], np.array([esys[1][:,h125ind],esys[1][:,Hind],esys[1][:,hSind]]))
    #############################################################
    def getPhases(self,tracingArgs={}):
        """
        Find different phases as functions of temperature

        This is the default function from cosmoTransitions.generic_potential
        except that the list of minima is read from self.ZeroTminLocs
        if it has previously been computed
        And that the tolerance for minima being the same is altered
        
        Parameters
        ----------
        tracingArgs : dict
            Parameters to pass to :func:`transitionFinder.traceMultiMin`.
        
        Returns
        -------
        dict
            Each item in the returned dictionary is an instance of
            :class:`transitionFinder.Phase`, and each phase is
            identified by a unique key. This value is also stored in
            `self.phases`.
        """
        tstop = self.Tmax
        points = []
        try:
            self.ZeroTMinLocs
        except:
            self.approxZeroTMin()
        for x0 in self.ZeroTMinLocs:
            points.append([x0,0.0])
        tracingArgs_ = dict(forbidCrit=self.forbidPhaseCrit)
        tracingArgs_.update(tracingArgs)
        phases = transitionFinder.traceMultiMin(
            self.Vtot, self.dgradV_dT, self.d2V, points,
            tLow=0.0, tHigh=tstop, deltaX_target=100*self.x_eps,
            **tracingArgs_)
        self.phases = phases
        transitionFinder.removeRedundantPhases(
            self.Vtot, phases, self.x_eps*1e-2, self.SameXThreshold)
        return self.phases
    ############################################################
    def forbidPhaseCrit(self, X):
        """
        Returns True if a phase at point `X` should be discarded,
        False otherwise.
         Using the Z_2 symmetry of the 3d potential, this function
        forbids phases if H^SM is negative
        """
        X=np.asanyarray(X)
        HSM = X[...,0]
        return HSM < -1.*self.SameXThreshold
    ############################################################
    def dVtot_zeroT_HSonly(self, HS):
        T = 0.
        return self.gradV([0., 0., HS], T)[-1]
    ############################################################
    def get_vsprimeCW(self, atol=0.01):
        """
        returns the location of v_{S,CW}'
        """
        # get location of tree-level minima
        vs = self.mu/self.lam
        vsp = -(self.mu/self.lam - self.Akap/(2.*self.kap))
        tree_locs = np.sort([0., vs, vsp])
        maxHS = 100.*np.max(np.abs(tree_locs))
        # check that the potential is well-behaved:
        if self.dVtot_zeroT_HSonly(-maxHS) > 0:
            # the potential is not going up towards HS -> -infty
            return np.nan
        elif self.dVtot_zeroT_HSonly(maxHS) < 0:
            # the potential is not going up towards HS -> +infty
            return np.nan
        elif self.dVtot_zeroT_HSonly(-atol)*self.dVtot_zeroT_HSonly(atol) > 0:
            # missing the extremum around HS = 0
            return np.nan
        # try to find the non-trivial extrema:
        if self.dVtot_zeroT_HSonly(-maxHS)*self.dVtot_zeroT_HSonly(-atol) < 0:
            # derivative has structure --- | +++ 0 --- | +++, 
            # i.e., one of the extrema is at negative HS and one at positive HS
            HS = []
            try:
               HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, -maxHS, -atol))
               HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, atol, maxHS))
            except:
               # fail...
               return np.nan
        else:
            if np.abs(optimize.newton(self.dVtot_zeroT_HSonly, -maxHS, disp=False)) > atol:
                # derivative has structure --- | +++ | --- 0 +++, 
                # i.e., both extrema are at negative HS
                HS = []
                loc = optimize.newton(self.dVtot_zeroT_HSonly, -maxHS, disp=False)
                if self.dVtot_zeroT_HSonly(-maxHS)*self.dVtot_zeroT_HSonly(loc-atol) > 0:
                    try:
                        HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, -maxHS, loc+atol))
                        HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, loc+atol, -atol))
                    except:
                        # fail...
                        return np.nan
                else:
                    try:
                        HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, -maxHS, loc-atol))
                        HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, loc-atol, -atol))
                    except:
                        # fail...
                        return np.nan
            elif np.abs(optimize.newton(self.dVtot_zeroT_HSonly, maxHS, disp=False)) > atol:
                # derivative has structure --- 0 +++ | --- | +++, 
                # i.e., both extrema are at positive HS
                HS = []
                loc = optimize.newton(self.dVtot_zeroT_HSonly, maxHS, disp=False)
                if self.dVtot_zeroT_HSonly(loc+atol)*self.dVtot_zeroT_HSonly(maxHS) > 0:
                    try:
                        HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, atol, loc-atol))
                        HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, loc-atol, maxHS))
                    except:
                        # fail...
                        return np.nan
                else:
                    try:
                        HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, atol, loc+atol))
                        HS.append(optimize.toms748(self.dVtot_zeroT_HSonly, loc+atol, maxHS))
                    except:
                        # fail...
                        return np.nan
        if len(HS) != 2:
            # fail...
            return np.nan
        # return entry furthest away from v_S
        return HS[np.argmax(np.abs(HS - np.sqrt(2.)*self.mu/self.lam))]
    ############################################################
    def prettyPrintTcTrans(self):
        if self.TcTrans == None:
            self.calcTcTrans()
        if len(self.TcTrans) == 0:
            print("No transitions found.\n")
        for trans in self.TcTrans:
            trantype = trans['trantype']
            if trantype == 1:
                trantype = 'First'
            elif trantype == 2:
                trantype = 'Second'
            print("%s-order transition at Tc = %0.4g" % (trantype, trans['Tcrit']))
            print("High-T phase:\n  key = %s; vev = %s" % (trans['high_phase'], trans['high_vev']))
            print("Low-T phase:\n  key = %s; vev = %s" % (trans['low_phase'], trans['low_vev']))
            print("the low-T phases ends at \n T_0 = %0.4g; vev = %s" % (self.phases[trans['low_phase']].T[0], self.phases[trans['low_phase']].X[0]))
            print("")
    #############################################################
    def prettyPrintTnTrans(self):
        if self.TnTrans == None:
            self.findAllTransitions()
        if len(self.TnTrans) == 0:
            print("No transitions found.\n")
        for trans in self.TnTrans:
            trantype = trans['trantype']
            if trantype == 1:
                trantype = 'First'
            elif trantype == 2:
                trantype = 'Second'
            print("%s-order transition at Tnuc = %0.4g" % (trantype, trans['Tnuc']))
            print("High-T phase:\n  key = %s; vev = %s" % (trans['high_phase'], trans['high_vev']))
            print("Low-T phase:\n  key = %s; vev = %s" % (trans['low_phase'], trans['low_vev']))
            print("the low-T phases ends at \n T_0 = %0.4g; vev = %s" % (self.phases[trans['low_phase']].T[0], self.phases[trans['low_phase']].X[0]))
            print("Action = %0.4g" % trans['action'])
            print("Action / Tnuc = %0.6g" % (trans['action']/trans['Tnuc']))
            print("")
    #############################################################
    def plotPhases2D_HSM_HS(self, **plotArgs):
        if self.phases is None:
            self.getPhases()
        for key, p in self.phases.items():
            plt.plot(p.X[...,0], p.X[...,2], **plotArgs)
        plt.xlabel(r"$\langle H^{\rm SM} \rangle (T)$ [GeV]")
        plt.ylabel(r"$\langle H^{\rm S} \rangle (T)$ [GeV]")
        plt.tick_params(right=True,top=True)
        plt.tight_layout()
    #############################################################
    def plotPhases2D_HSM_HNSM(self, **plotArgs):
        if self.phases is None:
            self.getPhases()
        for key, p in self.phases.items():
            plt.plot(p.X[...,0], p.X[...,1], **plotArgs)
        plt.xlabel(r"$\langle H^{\rm SM} \rangle (T)$ [GeV]")
        plt.ylabel(r"$\langle H^{\rm NSM} \rangle (T)$ [GeV]")
        plt.tick_params(right=True,top=True)
        plt.tight_layout()
    #############################################################
    def plotPhases2D_HNSM_HS(self, **plotArgs):
        if self.phases is None:
            self.getPhases()
        for key, p in self.phases.items():
            plt.plot(p.X[...,1], p.X[...,2], **plotArgs)
        plt.xlabel(r"$\langle H^{\rm NSM} \rangle (T)$ [GeV]")
        plt.ylabel(r"$\langle H^{\rm S} \rangle (T)$ [GeV]")
        plt.tick_params(right=True,top=True)
        plt.tight_layout()
    #############################################################
    def plotPhasesT(self, **plotArgs):
        if self.phases is None:
            self.getPhases()
        fig, axs = plt.subplots(2,2)
        for key, p in self.phases.items():
            axs[0,0].plot(p.T, p.X[...,0], **plotArgs)
            axs[0,1].plot(p.T, p.X[...,1], **plotArgs)
            axs[1,0].plot(p.T, p.X[...,2], **plotArgs)
            axs[1,1].plot(p.T, np.sqrt(p.X[...,0]**2+p.X[...,1]**2), **plotArgs)
        axs[0,0].set(ylabel=r'$\langle H^{\rm SM} \rangle(T)$ [GeV]')
        axs[0,1].set(ylabel=r'$\langle H^{\rm NSM} \rangle(T)$ [GeV]')
        axs[1,0].set(xlabel=r'$T$ [GeV]', ylabel=r'$\langle H^{\rm S} \rangle(T)$ [GeV]')
        axs[1,1].set(xlabel=r'$T$ [GeV]', ylabel=r'$\sqrt{\langle H^{\rm SM} \rangle^2 + \langle H^{\rm NSM} \rangle^2}$ [GeV]')
        axs[0,0].tick_params(right=True,top=True)
        axs[0,1].tick_params(right=True,top=True)
        axs[1,0].tick_params(right=True,top=True)
        axs[1,1].tick_params(right=True,top=True)
        fig.tight_layout()
    #############################################################
    def plotPhasesTand2D(self, **plotArgs):
        if self.phases is None:
            self.getPhases()
        fig, axs = plt.subplots(3, 3, figsize=(10,9.5))
        for key, p in self.phases.items():
            axs[0,0].plot(p.T, p.X[...,0], **plotArgs)
            axs[1,0].plot(p.T, p.X[...,1], **plotArgs)
            axs[2,0].plot(p.T, p.X[...,2], **plotArgs)
            axs[0,1].plot(p.X[...,1], p.X[...,0], **plotArgs)
            axs[1,1].plot(p.X[...,2], p.X[...,1], **plotArgs)
            axs[0,2].plot(p.X[...,2], p.X[...,0], **plotArgs)
            axs[1,2].plot(p.T, np.sqrt(p.X[...,0]**2+p.X[...,1]**2), **plotArgs)
            potVal = [self.DVtot(p.X[i], p.T[i]) for i in range(len(p.T))]
            axs[2,1].plot(p.T, np.sign(potVal)*np.log10(np.abs(potVal)))
            potVal = self.Vtot(p.X, p.T)
            axs[2,2].plot(p.T, np.sign(potVal)*np.log10(np.abs(potVal)))
        axs[0,0].set(xlabel=r'$T$ [GeV]', ylabel=r'$\langle H^{\rm SM} \rangle(T)$ [GeV]')
        axs[1,0].set(xlabel=r'$T$ [GeV]', ylabel=r'$\langle H^{\rm NSM} \rangle(T)$ [GeV]')
        axs[2,0].set(xlabel=r'$T$ [GeV]', ylabel=r'$\langle H^{\rm S} \rangle(T)$ [GeV]')
        axs[1,2].set(xlabel=r'$T$ [GeV]', ylabel=r'$\sqrt{\langle H^{\rm SM} \rangle^2 + \langle H^{\rm NSM} \rangle^2}$ [GeV]')
        axs[0,1].set(xlabel=r'$\langle H^{\rm NSM} \rangle(T)$ [GeV]', ylabel=r'$\langle H^{\rm SM} \rangle(T)$ [GeV]')
        axs[1,1].set(xlabel=r'$\langle H^{\rm S} \rangle(T)$ [GeV]', ylabel=r'$\langle H^{\rm NSM} \rangle(T)$ [GeV]')
        axs[0,2].set(xlabel=r'$\langle H^{\rm S} \rangle(T)$ [GeV]', ylabel=r'$\langle H^{\rm SM} \rangle(T)$ [GeV]')
        axs[2,1].set(xlabel=r'$T$ [GeV]', ylabel=r'${\rm sgn}(DV_1(T)) {\rm log}_{10}(DV_1(T)/{\rm GeV}^4)$')
        axs[2,2].set(xlabel=r'$T$ [GeV]', ylabel=r'${\rm sgn}(V_1(T)) {\rm log}_{10}(V_1(T)/{\rm GeV}^4)$')
        axs[0,0].tick_params(right=True,top=True)
        axs[0,1].tick_params(right=True,top=True)
        axs[0,2].tick_params(right=True,top=True)
        axs[1,0].tick_params(right=True,top=True)
        axs[1,1].tick_params(right=True,top=True)
        axs[1,2].tick_params(right=True,top=True)
        axs[2,0].tick_params(right=True,top=True)
        axs[2,1].tick_params(right=True,top=True)
        axs[2,2].tick_params(right=True,top=True)
        fig.tight_layout()
        