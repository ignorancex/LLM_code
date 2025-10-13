import numpy as np
import yaml


from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from scipy.interpolate import interp1d
from ept_rsd_vel_moments_resum_fftw import RedshiftSpaceVelocityMomentsREPT

class PairwiseVelocityLikelihood(Likelihood):

    kdatfn: str
    datfn: str
    covfn: str
    diagonal_covariance: bool

    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool
    
    kmin: float
    p0max: float
    p2max: float
    p4max: float
    v1max: float
    v3max: float
    s0max: float
    s2max: float


    def initialize(self):
        """Sets up the class."""
        self.loadData()

        self.linear_param_dict = yaml.load(open(self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict) 
        #

    def get_requirements(self):
        # b1, b2, bs, b3, beta00, beta12, beta22, beta24, beta34, s00, s20, s12, s22

        req = {'ptmod': None,\
               'b1': None,\
               'b2': None,\
               'bs': None,\
               #'b3': None,\
               #'beta00': None,\
               #'beta12': None,\
               #'beta22': None,\
               #'beta24': None,\
               #'beta34': None,\
               #'s00': None,\
               #'s20': None,\
               #'s12': None,\
               #'s22': None,\
              }
        return(req)

    def get_can_provide_params(self):
        cps = ["nonmarg_chi2"]
        return cps

    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """Return a log-likelihood."""
        
        # Compute the theory prediction with lin. params. at prior mean
        thy_obs_0 = self.predict()
        self.Delta = self.dd - thy_obs_0
        
        # Now compute template
        self.templates = []
        for param in self.linear_param_dict.keys():
            thetas = self.linear_param_means.copy()
            thetas[param] += 1.0
            self.templates += [ self.predict(thetas=thetas) - thy_obs_0 ]
        
        self.templates = np.array(self.templates)
        
        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1./self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)
        #t4 = time.time()
        
        # Compute the modified chi2
        lnL  = -0.5 * np.dot(self.Delta,np.dot(self.cinv,self.Delta)) # this is the "bare" lnL
        lnL +=  0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va)) # improvement in chi2 due to changing linear params
        state['derived']['nonmarg_chi2'] = -2 * lnL
        if not self.optimize:
            lnL += - 0.5 * np.log( np.linalg.det(self.Lab) ) + 0.5 * self.Nlin * np.log(2*np.pi) # volume factor from the determinant
        
        state['logp'] = lnL
        
        return state['logp'], state['derived']['nonmarg_chi2']

        #

    def get_best_fit(self):
        try:
            self.pred_nl  = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.pred_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.pred_nl + self.pred_lin
        except:
            print("Make sure to first compute the posterior.")
    
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        # First load the data
        self.kbins = np.loadtxt(self.kdatfn)
        self.binning_matrix = None
        
        dat = np.loadtxt(self.datfn)
        self.kdat = dat[:,0]
        self.p0dat = dat[:,1]
        self.p2dat = dat[:,2]
        self.p4dat = dat[:,3]
        self.v1dat = dat[:,4]
        self.v3dat = dat[:,5]
        self.s0dat = dat[:,6]
        self.s2dat = dat[:,7]
        
        self.dd = np.concatenate( (self.p0dat, self.p2dat, self.p4dat,\
                                   self.v1dat, self.v3dat,\
                                   self.s0dat, self.s2dat) )
        
        # Now load the covariance matrix.
        if self.covfn is not None:
            cov = np.loadtxt(self.covfn)

            if self.diagonal_covariance:
                cov = np.diag(np.diag(cov))
                
        else:
            cov = 0.01**2 * np.diag( np.ones_like(self.dd) ) * np.max(self.dd)**2

        self.cov_raw = 1.0 * cov
        
        for n, kmax in enumerate([self.p0max, self.p2max, self.p4max, self.v1max, self.v3max, self.s0max, self.s2max]):
        # loop through each multipole
            kcut = (self.kdat > kmax) | (self.kdat < self.kmin)
            for i in np.nonzero(kcut)[0]:
                ii = i + n*self.kdat.size
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        
        #
    def predict(self, thetas=None):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        modPT= pp.get_result('ptmod')

        #
        b1   = pp.get_param('b1')
        b2   = pp.get_param('b2')
        bs   = pp.get_param('bs')

        if thetas is None:
            b3 = self.linear_param_means['b3']
            beta00 = self.linear_param_means['beta00']
            beta12 = self.linear_param_means['beta12']
            beta22 = self.linear_param_means['beta22']
            beta24 = self.linear_param_means['beta24']
            beta34 = self.linear_param_means['beta34']
            s00 = self.linear_param_means['s00']
            s20 = self.linear_param_means['s20']
            s12 = self.linear_param_means['s12']
            s22 = self.linear_param_means['s22']
            beta_fog = self.linear_param_means['beta_fog']
            s44 = self.linear_param_means['s44']
        else:
            b3 = thetas['b3']
            beta00 = thetas['beta00']
            beta12 = thetas['beta12']
            beta22 = thetas['beta22']
            beta24 = thetas['beta24']
            beta34 = thetas['beta34']
            s00 = thetas['s00']
            s20 = thetas['s20']
            s12 = thetas['s12']
            s22 = thetas['s22']
            beta_fog = thetas['beta_fog']
            s44 = thetas['s44']        
        
        #b3   = pp.get_param('b3')
        #beta00 = pp.get_param('beta00')
        #beta12 = pp.get_param('beta12')
        #beta22 = pp.get_param('beta22')
        #beta24 = pp.get_param('beta24')
        #beta34 = pp.get_param('beta34')
        #s00 = pp.get_param('s00')
        #s20 = pp.get_param('s20')
        #s12 = pp.get_param('s12')
        #s22 = pp.get_param('s22')
        
        bias = [b1, b2, bs, b3]
        cterm = [beta00, beta12, beta22, beta24, beta34]
        stoch = [s00, s20, s12, s22]
        fog = [beta_fog, s44]
        bvec = bias + cterm + stoch + fog

        self.kv = modPT.kv
        self.pells = modPT.combine_bias_terms(bvec, modPT.pselltable)
        self.vells = modPT.combine_bias_terms(bvec, modPT.vselltable)
        self.sells = modPT.combine_bias_terms(bvec, modPT.sselltable)

        if self.binning_matrix is None:

            self.binning_matrix = np.zeros((len(self.kdat), len(self.kv)))
            
            for ii, (kl, kr) in enumerate(zip(self.kbins[:-1], self.kbins[1:])):

                if kl > 0:
                    kint = np.logspace(np.log10(kl), np.log10(kr), 100)
                else:
                    kint = np.linspace(kl, kr, 100) # in practice we will probably always toss the first bin

                for jj, kk in enumerate(self.kv):
                    pvec = np.zeros_like(self.kv); pvec[jj] = 1.0
                    pfunc = interp1d(self.kv, pvec, kind='cubic', bounds_error=False, fill_value=0)

                    norm = np.trapz(kint**2, x=kint)
                    self.binning_matrix[ii,jj] = np.trapz(pfunc(kint) * kint**2, x=kint) / norm
                
            
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        tt = []
        for ell in [0,2,4]:
            tt += [ np.dot(self.binning_matrix, self.pells[ell]), ]
            #tt += [ interp1d(self.kv, self.pells[ell], kind='cubic')(self.kdat), ]
        for ell in [1,3]:
            tt += [ np.dot(self.binning_matrix, self.vells[ell]), ]
            #tt += [ interp1d(self.kv, self.vells[ell], kind='cubic')(self.kdat), ]
        for ell in [0,2]:
            tt += [ np.dot(self.binning_matrix, self.sells[ell]), ]
            #tt += [ interp1d(self.kv, self.sells[ell], kind='cubic')(self.kdat), ]

        return np.concatenate(tt)
        #


class PT_Theory(Theory):
    """A class to return a PT P_ell module."""
    # From yaml file.
    plinfn: str
    pnwfn: str
    Dz: float
    kmax: float
    kmin: float
    nk: float
    #
    def initialize(self):
        """Sets up the class."""
        self.klin, self.plin = np.loadtxt(self.plinfn, unpack=True); self.plin *= self.Dz**2
        self.knw, self.pnw = np.loadtxt(self.pnwfn, unpack=True); self.pnw *= self.Dz**2

        self.RVEPT = RedshiftSpaceVelocityMomentsREPT(self.klin, self.plin, pnw=self.pnw,\
                                                      nk=self.nk, kmin=self.kmin, kmax=self.kmax)

        
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        req = {\
               'fz': None,
              }
        return(req)
    def get_can_provide(self):
        """What do we provide: a PT class that can compute xi_ell."""
        return ['ptmod']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """Create and initialize the PT class."""
        # Make shorter names.
        pp   = self.provider
        
        # Get cosmological parameters
        fz = pp.get_param('fz')
        self.RVEPT.setup_rsd_spectra(fz)

        state['ptmod'] = self.RVEPT
        #