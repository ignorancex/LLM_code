'''
Base module for generating models of line intensity maps
'''

import numpy as np
import inspect
import astropy.units as u
import astropy.constants as cu

from scipy.interpolate import interp1d,RegularGridInterpolator
from scipy.special import sici,erf,legendre,j1,roots_legendre,dawsn
from scipy.stats import poisson
from scipy.integrate import quad,simps

import finufft

dawsn_over_x_non_divergent = lambda x:np.where(x!=0,dawsn(x)/x,1)

try:
    import camb
    NoCamb = False
except:
    NoCamb = True
try:
    from classy import Class
    NoClass = False
except:
    NoClass = True
if NoCamb and NoClass:
    raise ValueError('You need to have either camb or class installed to run lim.')


from source.tools._utils import * 
from source.tools._vid_tools import binedge_to_binctr
import source.luminosity_functions as lf
import source.mass_luminosity as ml
import source.bias_fitting_functions as bm
import source.halo_mass_functions as HMF

class LineModel(object):
    '''
    An object containing all of the relevant astrophysical quantities of a
    LIM model.
    
    The purpose of this class is to calculate many quantities associated with
    a line intensity map, for now mostly with the goal of predicting a power
    spectrum from a different model.
    
    Models are defined by a number of input parameters defining a cosmology,
    and a prescription for assigning line luminosities.  These luminosities
    can either be drawn directly from a luminosity function, or assigned
    following a mass-luminosity relation.  In the latter case, abuundances are
    assigned following a mass function computed with pylians.
    
    Most methods in this class are given as @cached_properties, which means
    they are computed once when the method is called, then the outputs are
    saved for future calls.  Input parameters can be changed with the included
    update() method, which when called will reset the cached properties so
    they can be recomputed with the new values.
    
    WARNING: Parameter values should ONLY be changed with the update() method.
             Changing values any other way will NOT refresh the cached values
    
    Note that the lim package uses astropy units througout.  Input parameters
    must be assigned with the proper dimensions, or an error will be raised.
    
    New models can be easily created. In the case of 'LF' models, add a new
    function with the desired form to luminosity_functions.py.  For 'ML'
    models, do the same for mass_luminosity.py
    
    Defaults to the model from Breysse et al. (2017)
    
    INPUT PARAMETERS:
    
    <COSMO PARAMETERS>
    -------------------
    cosmo_code:          Whether to use class or camb (default: 'camb')
    
    cosmo_input_camb:    Dictionary to read and feed to camb
    
    cosmo_input_class:   Dictionary to read and feed to class
    
    <ASTROPHYSICAL PARAMETERS>
    ---------------------------

    model_type:     Either 'LF' for a luminosity function model or 'ML' for a
                    mass-luminosity model.  Any other value will raise an
                    error.  Note that some outputs are only available for one
                    model_type. (Default = 'LF')
    
    model_name:     Name of line emission model.  Must be the name of a
                    function defined in luminosity_functions.py (for
                    model_name='LF') or mass_luminosity.py (for model_name=
                    'ML'). (Default = 'SchCut')
                    
    model_par:      Dictionary containing the parameters of the chosen model
                    (Default = Parameters of Breysse et al. 2017 CO model)
                    
    hmf_model:      Fitting function for the halo m to choose from those
                    present in halo_mass_functions.py (default:'ST')
                    
    bias_model:     Fitting function for the bias model to choose from those
                    present in bias_fitting_functions.py (default:'ST99')
                    
    bias_par:       A dictionary to pass non-standard values for the parameters
                    of each bias model
                    
    nu:             Rest frame emission frequency of target line
                    (Default = 115 GHz, i.e. CO(1-0))
                    
    nuObs:          Observing frequency, defines target redshift
                    (Default = 30 GHz, i.e. z=2.8 for CO)
                    
    Mmin:           Minimum mass of line-emitting halo. (Default = 10^9 Msun)
    
    Mmax:           Maximum mass of line emitting halo.  Rarely a physical
                    parameter, but necessary to define high-mass cutoffs for
                    mass function integrals (Default = 10^15 Msun)
                    
    nM:             Number of halo mass points (Default = 500)
    
    Lmin:           Minimum luminosity for luminosity function calculations
                    (Default = 100 Lsun)
                    
    Lmax:           Maximum luminosity for luminosity function calculations
                    (Default = 10^8 Lsun)
                    
    nL:             Number of luminosity points (Default = 5000)
    
    v_of_M:         Function returning the unitful FWHM of the line profile of
                    emission given halo mass.
                    Line widths are not applied if v_of_M is None.
                    (default = None)
                    (example: lambda M:50*u.km/u.s*(M/1e10/u.Msun)**(1/3.) )
                    
    line_incli:     Bool, if accounting for randomly inclined line profiles.
                    (default = True; does not matter if v_of_M = None)
    
    <Pk PARAMETERS>
    ----------------
    
    kmin:           Minimum wavenumber for power spectrum computations
                    (Default = 10^-2 Mpc^-1)
                    
    kmax:           Maximum wavenumber for power sepctrum computations
                    (Default = 10 Mpc^-1)
    
    nk:             Number of wavenumber points (Default = 100)
    
    k_kind:         Whether you want k vector to be binned in linear or
                    log space (options: 'linear','log'; Default:'log')
    
    sigma_scatter:  Width of log-scatter in mass-luminosity relation, defined
                    as the width of a Gaussian distribution in log10(L) which
                    preserves the overall mean luminosity.  See Li et al.
                    (2015) for more information. (Default = 0.0)
                    
    fduty:          Duty cycle for line emission, as defined in Pullen et al.
                    2012 (Default = 1.0)
                    
    do_onehalo:     Bool, if True power spectra are computed with one-halo
                    term included (Default = False)
                    
    do_Jysr:        Bool, if True quantities are output in Jy/sr units rather
                    than brightness temperature (Default = False)
                    
    do_RSD:         Bool, if True power spectrum includes RSD (Default:False)
    
    sigma_NL:       Scale of Nonlinearities (Default: 7 Mpc)
    
    nmu:            number of mu bins
    
    FoG_damp:       damping term for Fingers of God (Default:'Lorentzian'
    
    smooth:         smoothed power spectrum, convoluted with beam/channel
                    (Default: False)
                    
    do_conv_Wkmin:  Convolve the power spectrum with Wkmin instead of using a exponential suppression. 
                    Only relevant if smooth==True. (Default = False)
                    Assumes a cylindrical volume
                    
    nonlinear:      Using the non linear matter power spectrum in PKint (from halofit)
                    (Boolean, default = False)
                    
    <VID PARAMETERS>
    -----------------
    
    smooth_VID:     Bool. Whether to model or not extended observed profiles
                    for the VID (default: True)
                    
    Tmin_VID:       Minimum temperature to compute the temperature PDF (default: 1e-2 uK)
    
    Tmax_VID:       Maximum temperature to compute the temperature PDF (default: 1e3 uK)
    
    nT:             Number of points in temperature to compute the PDF (default: 1e5)
                        
    Nbin_hist:      Number of bins to compute the VID histogram from the PDF (default=101)
    
    subtract_VID_mean:  Remove the mean from the VID measurements (default=False)
    
    linear_VID_bin: Using a linear sampling for the VID bins in the histogram 
                    (Boolean, default=False, which results in log binning)

    Lsmooth_tol:    Number of decimals for the tolerance of the ratio between
                    the luminosity in a voxel and the maximum luminsity when 
                    smooth_VID == True (Default: 7)
                    
    T0_Nlogsigma:   Scale (log) for the T0 array for the lognormal temperature distribution
                    to model the intrinsic scatter of luminosities (default: 4)
                    
    fT0_min:        Minimum value for the Fourier conjugate of the temperature
                    for the Fourier transform of the lognormal temperature distribution
                    to model the intrinsic scatter of luminosities
                    (default: 1e-5*u.uK**-1)
                    
    fT0_max:        Maximum value for the Fourier conjugate of the temperature
                    for the Fourier transform of the lognormal temperature distribution
                    to model the intrinsic scatter of luminosities
                    (default: 1e5*u.uK**-1)
                    
    nfT0:           Number of points in the array of the Fourier conjugate of the temperature
                    for the Fourier transform of the lognormal temperature distribution
                    to model the intrinsic scatter of luminosities
                    (default: 1000)
                    
    nT:             Number of points in the Temperature array for the PT
                    (default: 2**18)
    
    n_leggauss_nodes_FT:    Number of nodes in the Legendre-Gauss quadrature
                            for the NUFFTs. Can be an integer or a file with
                            them already computed. (Default: ../nodes1e5.txt)
                            
    n_leggauss_nodes_IFT:   Number of nodes in the Legendre-Gauss quadrature
                            for the backwards NUFFTs. Can be an integer or a file with
                            them already computed. (Default: ../nodes1e4.txt)
    
    sigma_PT_stable:        Standard deviation of a dummy Gaussian to ensure 
                            stability in the PT computation (especially for the
                            clustering part) when there is no noise. 
                            (default: 0.05*u.uK)

    DOCTESTS:
    >>> m = LineModel()
    >>> m.hubble
    0.6774
    >>> m.z
    <Quantity 2.833...>
    >>> m.dndL[0:2]
    <Quantity [  7.08...e-05,  7.15...e-05] 1 / (Mpc3 solLum)>
    >>> m.bavg
    <Quantity 1.983...>
    >>> m.nbar
    <Quantity 0.281... 1 / Mpc3>
    >>> m.Tmean
    <Quantity 1.769... uK>
    >>> m.Pk[0:2]
    <Quantity [ 108958..., 109250...] Mpc3 uK2>
    '''
    
    def __init__(self,
                 ##############
                 # COSMO params
                 ##############
                 cosmo_code = 'camb',
                 cosmo_input_camb=dict(f_NL=0,H0=67.36,cosmomc_theta=None,ombh2=0.02237, omch2=0.12, 
                               omk=0.0, neutrino_hierarchy='degenerate', 
                               num_massive_neutrinos=3, mnu=0.06, nnu=3.046, 
                               YHe=None, meffsterile=0.0, standard_neutrino_neff=3.046, 
                               TCMB=2.7255, tau=None, deltazrei=None, bbn_predictor=None, 
                               theta_H0_range=[10, 100],w=-1.0, wa=0., cs2=1.0, 
                               dark_energy_model='ppf',As=2.1e-09, ns=0.9649, nrun=0, 
                               nrunrun=0.0, r=0.0, nt=None, ntrun=0.0, 
                               pivot_scalar=0.05, pivot_tensor=0.05,
                               parameterization=2,halofit_version='mead2020'),
                 cosmo_input_class=dict(f_NL=0,H0=67.36,omega_b=0.02237, omega_cdm=0.12, 
                               A_s=2.1e-9,n_s=0.9649,
                               N_ncdm=3, m_ncdm='0.02,0.02,0.02', N_ur = 0.00641,
                               output='mPk,mTk'),
                 ###############
                 # ASTRO params
                 ###############
                 model_type='LF',
                 model_name='SchCut', 
                 model_par={'phistar':9.6e-11*u.Lsun**-1*u.Mpc**-3,
                 'Lstar':2.1e6*u.Lsun,'alpha':-1.87,'Lmin':5000*u.Lsun},
                 hmf_model='ST',
                 bias_model='ST99',
                 bias_par={}, #Otherwise, write a dict with the corresponding values
                 nu=115*u.GHz,
                 nuObs=30*u.GHz,
                 Mmin=1e9*u.Msun,
                 Mmax=1e15*u.Msun,
                 nM=500,
                 Lmin=10*u.Lsun,
                 Lmax=1e8*u.Lsun,
                 nL=5000,
                 v_of_M=None,
                 line_incli=True,
                 ###########
                 # Pk params
                 ###########
                 kmin = 1e-2*u.Mpc**-1,
                 kmax = 10.*u.Mpc**-1,
                 nk = 100,
                 k_kind = 'log',
                 sigma_scatter=0.,
                 fduty=1.,
                 do_onehalo=False,
                 do_Jysr=False,
                 do_RSD=True,
                 sigma_NL=7*u.Mpc,
                 nmu=1000,
                 FoG_damp='Lorentzian',
                 smooth=False,
                 do_conv_Wkmin = False,
                 nonlinear=False,
                 ############
                 #VID params
                 ############
                 smooth_VID = True,
                 Tmin_VID=1.0e-2*u.uK,
                 Tmax_VID=100.*u.uK,
                 Nbin_hist=100,
                 linear_VID_bin=False,
                 subtract_VID_mean=False,
                 #VID precision parameters
                 Lsmooth_tol=7,
                 T0_Nlogsigma=4,
                 fT0_min=1e-5*u.uK**-1,
                 fT0_max=1e4*u.uK**-1,
                 fT_min=1e-5*u.uK**-1,
                 fT_max=1e5*u.uK**-1,
                 nfT0=1000,
                 nT=2**18,
                 n_leggauss_nodes_FT='../nodes1e5.txt',
                 n_leggauss_nodes_IFT='../nodes1e4.txt',
                 sigma_PT_stable=0.05*u.uK,
                 ############
                 # BSM edit - add ncdm parameters
                 ############
                 do_ncdm=False,
                 kcut=0.5/u.Mpc,
                 slope=0.1
                 ):
        

        # Get list of input values to check type and units
        self._lim_params = locals()
        self._lim_params.pop('self')
        
        # Get list of input names and default values
        self._default_lim_params = get_default_params(LineModel.__init__)
        # Check that input values have the correct type and units
        check_params(self._lim_params,self._default_lim_params)
        
        # Set all given parameters
        for key in self._lim_params:
            setattr(self,key,self._lim_params[key])

            
        # Create overall lists of parameters (Only used if using one of 
        # lim's subclasses
        self._input_params = {} # Don't want .update to change _lim_params
        self._default_params = {}
        self._input_params.update(self._lim_params)
        self._default_params.update(self._default_lim_params)
        
        # Create list of cached properties
        self._update_list = []
        self._update_cosmo_list = []
        self._update_obs_list = []
        self._update_vid_list = []
        
        # Check if model_name is valid
        check_model(self.model_type,self.model_name)
        check_bias_model(self.bias_model)
        check_halo_mass_function_model(self.hmf_model)

        #Set cosmology and call camb or class
        if self.cosmo_code == 'camb':
            self.cosmo_input_camb = self._default_params['cosmo_input_camb']
            for key in cosmo_input_camb:
                self.cosmo_input_camb[key] = cosmo_input_camb[key]

            self.camb_pars = camb.set_params(H0=self.cosmo_input_camb['H0'], cosmomc_theta=self.cosmo_input_camb['cosmomc_theta'],
                 ombh2=self.cosmo_input_camb['ombh2'], omch2=self.cosmo_input_camb['omch2'], omk=self.cosmo_input_camb['omk'],
                 neutrino_hierarchy=self.cosmo_input_camb['neutrino_hierarchy'], 
                 num_massive_neutrinos=self.cosmo_input_camb['num_massive_neutrinos'],
                 mnu=self.cosmo_input_camb['mnu'], nnu=self.cosmo_input_camb['nnu'], YHe=self.cosmo_input_camb['YHe'], 
                 meffsterile=self.cosmo_input_camb['meffsterile'], 
                 standard_neutrino_neff=self.cosmo_input_camb['standard_neutrino_neff'], 
                 TCMB=self.cosmo_input_camb['TCMB'], tau=self.cosmo_input_camb['tau'], 
                 deltazrei=self.cosmo_input_camb['deltazrei'], 
                 bbn_predictor=self.cosmo_input_camb['bbn_predictor'], 
                 theta_H0_range=self.cosmo_input_camb['theta_H0_range'],
                 w=self.cosmo_input_camb['w'], cs2=self.cosmo_input_camb['cs2'], 
                 dark_energy_model=self.cosmo_input_camb['dark_energy_model'],
                 As=self.cosmo_input_camb['As'], ns=self.cosmo_input_camb['ns'], 
                 nrun=self.cosmo_input_camb['nrun'], nrunrun=self.cosmo_input_camb['nrunrun'], 
                 r=self.cosmo_input_camb['r'], nt=self.cosmo_input_camb['nt'], ntrun=self.cosmo_input_camb['ntrun'], 
                 pivot_scalar=self.cosmo_input_camb['pivot_scalar'], 
                 pivot_tensor=self.cosmo_input_camb['pivot_tensor'],
                 parameterization=self.cosmo_input_camb['parameterization'],
                 halofit_version=self.cosmo_input_camb['halofit_version'])
                 
            self.camb_pars.WantTransfer=True    
            self.camb_pars.Transfer.accurate_massive_neutrinos = True
                
        elif self.cosmo_code == 'class':
            if not 'f_NL' in self.cosmo_input_class:
                self.cosmo_input_class['f_NL'] = 0.
            pk_pars = {}
            if not ('P_k_max_1/Mpc' in self.cosmo_input_class or 'P_k_max_h/Mpc' in self.cosmo_input_class):
                pk_pars['P_k_max_1/Mpc'] = 100
            if not 'z_max_pk' in self.cosmo_input_class:
                pk_pars['z_max_pk'] = 15.
            if not 'format' in self.cosmo_input_class:
                pk_pars['format'] = 'camb'
            #accelerate class computation: check precision with defaul! (default:10)
            if not 'k_per_decade_for_pk' in self.cosmo_input_class:
                pk_pars['k_per_decade_for_pk'] = 6
            if self.nonlinear:
                pk_pars['non linear'] = 'HMCODE'

            self.class_pars = merge_dicts([self.cosmo_input_class,pk_pars])
            del self.class_pars['f_NL']
            if 'output' not in self.class_pars:
                self.class_pars['output'] = 'mPk,mTk'
            else:
                if 'mPk' not in self.class_pars['output']:
                    self.class_pars['output'] = 'mPk,' + self.class_pars['output']
                if 'mTk' not in self.class_pars['output'] or 'dTk' not in self.class_pars['output']:
                    self.class_pars['output'] = 'mTk,' + self.class_pars['output']
            
            # increase z_max_pk if needed
        else:
            raise ValueError("Only 'class' or 'camb' can be used as cosmological Boltzmann code. Please, choose between them")
        
        if self.nT % 2:
            print('nT must be even: increasing it by 1 to have it even')
            self.nT = self.nT+1
        
    #################
    # Get cosmology #
    #################
    
    @cached_cosmo_property
    def zcosmo(self):
        '''
        Get the z array to call camb and interpolate cosmological quantities
        '''
        zmax = 15.  #Increase if interested in higher redshifts.
                    #If cosmo_code == 'class', change it also in the initialization of line_model()
                    # or in the input parameters
        Nz = 150
        if self.z > zmax:
            raise ValueError('Required z_obs outside interpolation region. Increase zmax or change nuObs')
        return np.linspace(0.,zmax,Nz)
    
    
    @cached_cosmo_property
    def cosmo(self):
        '''
        Compute the cosmological evolution, using camb or class
        '''
        if self.cosmo_code == 'camb':
            self.camb_pars.set_matter_power(redshifts=list(self.zcosmo))#, 
            return camb.get_results(self.camb_pars)
        else:
            cos = Class()
            cos.set(self.class_pars)
            cos.compute()
            return cos
   
   
    @cached_property
    def transfer_m(self):
        '''
        return matter transfer for the z of interest
        Argument k in 1/Mpc
        '''
        if self.cosmo_code == 'camb':
            #Find two closest (above and below) indices values for z in zcosmo
            zz = self.zcosmo[::-1] #camb sortes earlier first
            iz_down = np.where(zz - self.z < 0)[0][0]
            iz_up = iz_down - 1
            dz = zz[iz_up] - zz[iz_down]
            
            #Get the transfer
            T = self.cosmo.get_matter_transfer_data()
            kvec = (T.transfer_z('k/h',-1)*self.Mpch**-1).to(u.Mpc**-1)
            Tk_up = T.transfer_z('delta_tot',iz_up)
            Tk_down = T.transfer_z('delta_tot',iz_down)
            #interpolate in z (linear)
            Tz = Tk_down*(1.-(self.z-zz[iz_down])/dz) + Tk_up*(self.z-zz[iz_down])/dz
        else:
            T = self.cosmo.get_transfer(self.z,'camb')
            kvec = (T['k (h/Mpc)']*self.Mpch**-1).to(u.Mpc**-1)
            Tz = T['-T_tot/k2']
        #interpolate in k (linear)
        return log_interp1d(kvec,Tz)
            
   
    @cached_property
    def transfer_cb(self):
        '''
        return cdm+b transfer for the z of interest. 
        Argument k in 1/Mpc
        '''
        if self.cosmo_code == 'camb':
            #Find two closest (above and below) indices values for z in zcosmo
            zz = self.zcosmo[::-1] #camb sortes earlier first
            iz_down = np.where(zz - self.z < 0)[0][0]
            iz_up = iz_down - 1
            dz = zz[iz_up] - zz[iz_down]
            
            #Get the transfer
            T = self.cosmo.get_matter_transfer_data()
            kvec = (T.transfer_z('k/h',-1)*self.Mpch**-1).to(u.Mpc**-1)
            Tk_up = T.transfer_z('delta_nonu',iz_up)
            Tk_down = T.transfer_z('delta_nonu',iz_down)
            #interpolate in z (linear)
            Tz = Tk_down*(1.-(self.z-zz[iz_down])/dz) + Tk_up*(self.z-zz[iz_down])/dz
        else:
            T = self.cosmo.get_transfer(self.z,'camb')
            kvec = (T['k (h/Mpc)']*self.Mpch**-1).to(u.Mpc**-1)
            Tz = (self.cosmo.Omega0_cdm()*T['-T_cdm/k2'] + 
                  (self.cosmo.Omega0_m()-self.cosmo.Omega0_cdm())*T['-T_b/k2'])/self.cosmo.Omega0_m()
        #interpolate in k (linear)
        return log_interp1d(kvec,Tz)

       
    @cached_cosmo_property
    def f_NL(self):
        if self.cosmo_code == 'camb':
            return self.cosmo_input_camb['f_NL']
        else:
            return self.cosmo_input_class['f_NL']
        
        
    @cached_cosmo_property
    def Alcock_Packynski_params(self):
        '''
        Returns the quantities needed for the rescaling for Alcock-Paczyinski
           Da/rs, H*rs, DV/rs
        '''
        if self.cosmo_code == 'camb':
            BAO_pars = self.cosmo.get_BAO(self.zcosmo[1:],self.camb_pars)
            #This is rs/DV, H, DA, F_AP
            rs = self.cosmo.get_derived_params()['rdrag']
            DA = BAO_pars[:,2]
            DV = rs/BAO_pars[:,0]
            Hz = BAO_pars[:,1]
            
        elif self.cosmo_code == 'class':
            rs = self.cosmo.rs_drag()
            Nz = len(self.zcosmo[1:])
            DA, Hz = np.zeros(Nz),np.zeros(Nz)
            for i in range(Nz):
                DA[i] = self.cosmo.angular_distance(self.zcosmo[i+1])
                Hz[i] = self.cosmo.Hubble(self.zcosmo[i+1])*cu.c.to(u.km/u.s).value 
            prefact = cu.c.to(u.km/u.s).value*self.zcosmo[1:]*(1.+self.zcosmo[1:])**2
            DV = (prefact*DA**2/Hz)**(1./3.)
            
        DA_over_rs_int = interp1d(self.zcosmo[1:],DA/rs,kind='cubic',
                                  bounds_error=False,fill_value='extrapolate')
        DV_over_rs_int = interp1d(self.zcosmo[1:],DV/rs,kind='cubic',
                                  bounds_error=False,fill_value='extrapolate')
        H_times_rs_int = interp1d(self.zcosmo[1:],Hz*rs,kind='cubic',
                                  bounds_error=False,fill_value='extrapolate')
    
        return DA_over_rs_int, H_times_rs_int,DV_over_rs_int
        
        
    @cached_cosmo_property
    def PKint(self):
        '''
        Get the interpolator for the matter power spectrum as function of z and k 
        if mnu > 0 -> P_cb (without neutrinos)
        k input in 1/Mpc units
        P(k) output in Mpc^3 units
        '''
        if self.cosmo_code == 'camb':
            zmax = self.zcosmo[-1]
            nz_step=64
            if self.camb_pars.num_nu_massive != 0:
                var = 8
            else:
                var = 7
            PK = camb.get_matter_power_interpolator(self.camb_pars, zmin=0, 
                                                    zmax=zmax, nz_step=nz_step, 
                                                    zs=None, kmax=100, nonlinear=self.nonlinear,
                                                    var1=var, var2=var, hubble_units=False, 
                                                    k_hunit=False, return_z_k=False,
                                                    k_per_logint=None, log_interp=False, 
                                                    extrap_kmax=True)
            return PK.P
        else:
            if self.cosmo.Omega_nu != 0:
                return self.cosmo.get_pk_cb_array
            else:
                return self.cosmo.get_pk_array
        
        
    @cached_property
    def f_eff(self):
        '''
        Get the interpolator for the effective f as function of k for the 
        redshift of interest (includes the tiling to multiply by mu)
        
        if mnu = 0: f_eff = f_m; if mnu > 0: f_eff = f_cb
        '''
        dz = 1e-4
        if self.cosmo_code == 'camb':
            fs8lin = self.cosmo.get_fsigma8()
            s8lin = self.cosmo.get_sigma8()
            fz = interp1d(self.zcosmo[::-1],fs8lin/s8lin,kind='cubic')(self.z)
            #Apply correction if massive nu
            if self.camb_pars.num_nu_massive != 0:
                factor = self.transfer_m(self.k.value)/self.transfer_cb(self.k.value)
            else:
                factor = self.transfer_m(self.k.value)/self.transfer_m(self.k.value)
        else:
            fz = self.cosmo.scale_independent_growth_factor_f(self.z)
            #Apply correction if massive nu
            if self.cosmo.Omega_nu != 0:
                factor = self.transfer_m(self.k.value)/self.transfer_cb(self.k.value)
            else:
                factor = self.transfer_m(self.k.value)/self.transfer_m(self.k.value)
        return np.tile(fz*factor,(self.nmu,1))

                   
    @cached_cosmo_property
    def Dgrowth(self):
        '''
        Get the growth factor (for matter) as function of z
        (Dgrowth(z=0) = 1.)
        '''
        if self.cosmo_code == 'camb':
            s8lin = self.cosmo.get_sigma8()
            return interp1d(self.zcosmo[::-1],s8lin/s8lin[-1],kind='cubic',
                            bounds_error=False,fill_value='extrapolate')
        else:
            Nz = len(self.zcosmo)
            D = np.zeros(Nz)
            for iz in range(Nz):
                D[i] = self.cosmo.scale_independent_growth_factor(self.zcosmo[iz])
            return interp1d(self.zcosmo,D,kind='cubic',
                            bounds_error=False,fill_value='extrapolate')
        
    
    ####################
    # Define 1/h units #
    ####################
    @cached_cosmo_property
    def hubble(self):
        '''
        Normalized hubble parameter (H0.value/100). Used for converting to
        1/h units.
        '''
        if self.cosmo_code == 'camb':
            return self.camb_pars.H0/100.
        else:
            return self.cosmo.h()
    
    
    @cached_cosmo_property
    def Mpch(self):
        '''
        Mpc/h unit, required for interacting with hmf outputs
        '''
        return u.Mpc / self.hubble
        
        
    @cached_cosmo_property
    def Msunh(self):
        '''
        Msun/h unit, required for interacting with hmf outputs
        '''
        return u.Msun / self.hubble
    
    
    #################################
    # Properties of target redshift #
    #################################  
    @cached_property
    def z(self):
        '''
        Emission redshift of target line
        '''
        return (self.nu/self.nuObs-1.).value
    
    
    @cached_property
    def H(self):
        '''
        Hubble parameter at target redshift
        '''
        if self.cosmo_code == 'camb':
            return self.cosmo.hubble_parameter(self.z)*(u.km/u.Mpc/u.s)
        else:
            return self.cosmo.Hubble(self.z)*(u.Mpc**-1)*cu.c.to(u.km/u.s)
        
        
    @cached_property
    def CLT(self):
        '''
        Coefficient relating luminosity density to brightness temperature
        '''
        if self.do_Jysr:
            x = cu.c/(4.*np.pi*self.nu*self.H*(1.*u.sr))
            return x.to(u.Jy*u.Mpc**3/(u.Lsun*u.sr))
        else:
            x = cu.c**3*(1+self.z)**2/(8*np.pi*cu.k_B*self.nu**3*self.H)
            return x.to(u.uK*u.Mpc**3/u.Lsun)
    
    
    #########################################
    # Masses, luminosities, and wavenumbers #
    #########################################
    @cached_property
    def M(self):
        '''
        List of masses for computing mass functions and related quantities
        '''
        return ulogspace(self.Mmin,self.Mmax,self.nM)
    
    
    @cached_property
    def L(self):
        '''
        List of luminosities for computing luminosity functions and related
        quantities.
        '''
        return ulogspace(self.Lmin,self.Lmax,self.nL)
        
        
    @cached_property
    def k_edge(self):
        '''
        Wavenumber bin edges
        '''
        if self.k_kind == 'log':
            return ulogspace(self.kmin,self.kmax,self.nk+1)
        elif self.k_kind == 'linear':
            return ulinspace(self.kmin,self.kmax,self.nk+1)
        else:
            raise ValueError('Invalid value of k_kind. Choose between\
             linear or log')
    
    
    @cached_property
    def k(self):
        '''
        List of wave numbers for power spectrum and related quantities
        '''
        Nedge = self.k_edge.size
        return (self.k_edge[0:Nedge-1]+self.k_edge[1:Nedge])/2.
    
    
    @cached_property
    def dk(self):
        '''
        Width of wavenumber bins
        '''
        return np.diff(self.k_edge)
        
        
    @cached_property
    def mu_edge(self):
        '''
        cos theta bin edges
        '''
        return np.linspace(-1,1,self.nmu+1)
        
        
    @cached_property
    def mu(self):
        '''
        List of mu (cos theta) values for anisotropic, or integrals
        '''
        Nedge = self.mu_edge.size
        return (self.mu_edge[0:Nedge-1]+self.mu_edge[1:Nedge])/2.
        
        
    @cached_property
    def dmu(self):
        '''
        Width of cos theta bins
        '''
        return np.diff(self.mu_edge)
        
        
    @cached_property
    def ki_grid(self):
        '''
        Grid of k for anisotropic
        '''
        return np.meshgrid(self.k,self.mu)[0]
        
        
    @cached_property
    def mui_grid(self):
        '''
        Grid of mu for anisotropic
        '''
        return np.meshgrid(self.k,self.mu)[1]
        
        
    @cached_property
    def k_par(self):
        '''
        Grid of k_parallel
        '''
        return self.ki_grid*self.mui_grid
        
        
    @cached_property
    def k_perp(self):
        '''
        Grid of k_perpendicular
        '''
        return self.ki_grid*np.sqrt(1.-self.mui_grid**2.)
    
    
    #####################
    # Line luminosities #
    #####################
    @cached_property
    def dndL(self):
        '''
        Line luminosity function. 
        '''
        if self.model_type=='LF':
            return getattr(lf,self.model_name)(self.L,self.model_par)
        else:
            #compute LF from the conditional LF
            if self.Lmin > self.LofM[self.LofM.value>0][0]:
                print('Warning! reduce Lmin to cover all luminosities of the model')
            if self.Lmax < np.max(self.LofM):
                print('Warning! increase Lmax to cover all luminosities of the model')
            #assume a lognormal PDF for the CLF with minimum logscatter of 0.05
            CLF_of_M = np.zeros((self.nM,self.nL))*self.dndM.unit*self.L.unit**-1
            sigma = max(self.sigma_scatter,0.05)
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.model_name=='TonyLi':
                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                #assume sigma and sig_SFR are totally uncorrelated
                sigma = (sigma**2 + sig_SFR**2/alpha**2)**0.5
                sigma_base_e = sigma*2.302585
            else:
                sigma_base_e = sigma*2.302585
            for iM in range(self.nM):
                CLF_of_M[iM,:] = lognormal(self.L,np.log(self.LofM[iM].value)-0.5*sigma_base_e**2.,sigma_base_e)*self.dndM[iM]
            LF = np.zeros(self.nL)*self.L.unit**-1*self.dndM.unit*self.M.unit
            for iL in range(self.nL):
                LF[iL] = np.trapz(CLF_of_M[:,iL],self.M)
            #Add a cut off at low luminosities to ease computations. Default 0*u.Lsun
            #LF *= np.exp(-self.dndL_Lcut/self.L)
            return LF
        
        
    @cached_property
    def LofM(self):
        '''
        Line luminosity as a function of halo mass.
        
        'LF' models need this to compute average bias, and always assume that
        luminosity is linear in M.  This is what is output when this function
        is called on an LF model.  NOTE that in this case, this should NOT be
        taken to be an accurate physical model as it will be off by an overall
        constant.
        '''
        if self.model_type=='LF':
            LF_par = {'A':1.,'b':1.,'Mcut_min':self.Mmin,'Mcut_max':self.Mmax}
            L = getattr(ml,'MassPow')(self,self.M,LF_par,self.z)
        else:
            L = getattr(ml,self.model_name)(self,self.M,self.model_par,self.z)
        return L

    # BSM edit - define functions to calculate HMF correction for non-zero fnl, and define non-CDM transfer function
    @cached_property
    def S3_dS3(self):
        '''
        The skewness and derivative with respect to mass of the skewness. 
        Used to calculate the correction to the HMF due to non-zero fnl, 
        as presented in 2009.01245.

        Their parameter k_cut is equivalent to our klim, not to be confused
        with the ncdm parameter. k_lim represents the cutoff in the skewness 
        integral, we opt for no cutoff and thus set it to a very small value.
        This can be changed if necessary.
        '''
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3
        if self.cosmo_code == 'camb':
            rhoM = rho_crit*(self.camb_pars.omegam-self.camb_pars.omeganu)
            As = self.cosmo_input_camb['As']
            ns = self.cosmo_input_camb['ns']
            kpiv = self.cosmo_input_camb['pivot_scalar']*u.Mpc**-1
        else: 
            rhoM = rho_crit*(self.cosmo.Omega0_m()-self.cosmo.Omega_nu)
            As = self.cosmo_input_class['A_s']
            ns = self.cosmo_input_class['n_s']
            kpiv = self.cosmo_input_class['k_pivot']*u.Mpc**-1

        klim = 1.e-10 # has units 1/u.Mpc
        k1 = np.logspace(np.log10(klim),2.698,128)*u.Mpc**-1
        k2 = np.logspace(np.log10(klim),2.698,128)*u.Mpc**-1
        phi = np.linspace(-0.995,0.995,128) #not -1,1 to avoid nans bc k1+k2=0 

        k1_grid = np.meshgrid(k1,k2)[0]
        k2_grid = np.meshgrid(k1,k2)[1]

        S3 = np.zeros(self.nM)
        dS3 = np.zeros(self.nM)

        for iM in range(self.nM):
            dummy_S3 = np.zeros(len(phi))
            dummy_dS3 = np.zeros(len(phi))
            for iphi in range(len(phi)):
                k12_grid = np.sqrt(k1_grid**2+k2_grid**2+2*k1_grid*k2_grid*phi[iphi])
                inds = np.where(k12_grid.value<=klim)
                
                R = (3.0*self.M[iM]/(4.0*np.pi*rhoM))**(1.0/3.0)
                x = ((k1_grid*R).decompose()).value
                W1 = 3.0*(np.sin(x) - x*np.cos(x))/(x)**3 
                dW1 = (-9*np.sin(x) + 9*x*np.cos(x) + 3*(x**2)*np.sin(x))/((3*self.M[iM].value)*(x**3))
                x = ((k2_grid*R).decompose()).value
                W2 = 3.0*(np.sin(x) - x*np.cos(x))/(x)**3
                dW2 = (-9*np.sin(x) + 9*x*np.cos(x) + 3*(x**2)*np.sin(x))/((3*self.M[iM].value)*(x**3))
                x = ((k12_grid*R).decompose()).value
                W12 = 3.0*(np.sin(x) - x*np.cos(x))/(x)**3
                dW12 = (-9*np.sin(x) + 9*x*np.cos(x) + 3*(x**2)*np.sin(x))/((3*self.M[iM].value)*(x**3))
                
                dWs = (dW1*W2*W12 + W1*dW2*W12 + W1*W2*dW12)

                T1 = (-5*self.transfer_m(k1_grid.value)*k1_grid.value**2)/3
                T2 = (-5*self.transfer_m(k2_grid.value)*k2_grid.value**2)/3
                T12 = (-5*self.transfer_m(k12_grid.value)*k12_grid.value**2)/3
                P1 = (2*np.pi**2/k1_grid**3)*(9/25)*As*(k1_grid/kpiv)**(ns-1)
                P2 = (2*np.pi**2/k2_grid**3)*(9/25)*As*(k2_grid/kpiv)**(ns-1)

                integ_S3 = k1_grid**2*k2_grid**2*T1*T2*T12*W1*W2*W12*P1*P2
                integ_S3[inds] = 0
                dummy_temp_S3 = np.trapz(integ_S3,k1_grid,axis=1)
                dummy_S3[iphi] = np.trapz(dummy_temp_S3,k2)

                integ_dS3 = k1_grid**2*k2_grid**2*T1*T2*T12*dWs*P1*P2                    
                integ_dS3[inds] = 0
                dummy_temp_dS3 = np.trapz(integ_dS3,k1_grid,axis=1)
                dummy_dS3[iphi] = np.trapz(dummy_temp_dS3,k2)
                
            S3[iM] = np.trapz(dummy_S3,phi)
            dS3[iM] = np.trapz(dummy_dS3,phi)
    
        S3 *= self.f_NL*6/8/np.pi**4
        dS3 *= self.f_NL*6/8/np.pi**4
        return -S3, -dS3/u.Msun

    @cached_property
    def kappa3_dkappa3dM(self):
        '''
        Calculates kappa_3 its derivative with respect to halo mass M from 2009.01245
        '''

        S3, dS3_dM = self.S3_dS3

        kappa3 = S3/(self.sigmaM**3)
        dkappa3dM = (dS3_dM - 3*S3*self.dsigmaM_dM/self.sigmaM)/(self.sigmaM**3)

        return kappa3, dkappa3dM
    

    @cached_property
    def Delta_HMF(self):
        '''
        The correction to the HMF due to non-zero f_NL, as presented in 2009.01245.
        '''
        nuc = 1.42/self.sigmaM
        dnuc_dM = -1.42*self.dsigmaM_dM/(self.sigmaM)**2
        kappa3, dkappa3_dM = self.kappa3_dkappa3dM

        H2nuc = nuc**2-1
        H3nuc = nuc**3-3*nuc

        F1pF0p = (kappa3*H3nuc - H2nuc*dkappa3_dM/dnuc_dM )/6

        return F1pF0p
    
    # @cached_property
    def transfer_ncdm(self,ncdmk):
        '''
        Transfer function to suppress small-scale power due to non-CDM models as presented in 2404.11609.
        '''

        # make sure k's are in the proper units

        k_cut = self.kcut.to(1./u.Mpc).value
        k = ncdmk.to(1./u.Mpc).value

        if self.do_ncdm:
            if self.f_NL != 0:
                raise ValueError('Cannot have non-zero f_NL and non-CDM.')
            else:
                # Initialize Tk with ones of the same shape as ncdmk
                Tk = np.ones_like(k)
                # Apply the transfer function conditionally
                mask = k > k_cut
                Tk[mask] = (k[mask] / k_cut) ** (-self.slope)
        else:
            Tk = np.ones_like(k)
        return Tk
    #        
        
    @cached_property
    def dndM(self):
        '''
        Halo mass function, using functions in halo_mass_functions.py
        '''
        Mvec = self.M.to(self.Msunh)
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(self.Msunh*self.Mpch**-3) #h^2 Msun/Mpc^3
        #Use Omega_m or Omega_cdm+Omega_b wheter mnu = 0 or > 0
        if self.cosmo_code == 'camb':
            rhoM = rho_crit*(self.camb_pars.omegam-self.camb_pars.omeganu)
        else:
            rhoM = rho_crit*(self.cosmo.Omega0_m()-self.cosmo.Omega_nu)
        
        mf = getattr(HMF,self.hmf_model)(self,Mvec,rhoM)
        
        # BSM edit - add HMF correction for non-zero fnl
        #return mf.to(u.Mpc**-3*u.Msun**-1)
        if self.f_NL == 0:
            return mf.to(u.Mpc**-3*u.Msun**-1)
        else:
            if self.do_ncdm:
                raise ValueError('Cannot have non-zero f_NL and non-CDM.')
            else:
                return mf.to(u.Mpc**-3*u.Msun**-1)*(1+self.Delta_HMF)        
        
        
    @cached_property
    def sigmaM(self):
        '''
        Mass (or cdm+b) variance at target redshift
        '''
        #Get R(M) and P(k)
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3
        # BSM edit - change k range to match HMF correction
        #k = np.logspace(-2,2,128)*u.Mpc**-1
        if self.f_NL != 0: k = np.logspace(-3,3,128)*u.Mpc**-1
        else: k = np.logspace(-2,2,128)*u.Mpc**-1
        #
        #Use rho_m or rho_cb depending on mnu
        if self.cosmo_code == 'camb':
            # BSM edit - add ncdm transfer function
            Pk = (self.PKint(self.z,k.value)*u.Mpc**3 )*self.transfer_ncdm(k)
            #
            rhoM = rho_crit*(self.camb_pars.omegam-self.camb_pars.omeganu)
        else:
            # BSM edit - add ncdm transfer function
            Pk = (self.PKint(k.value,np.array([self.z]),len(k),1,0)*u.Mpc**3)*self.transfer_ncdm(k)
            #
            rhoM = rho_crit*(self.cosmo.Omega0_m()-self.cosmo.Omega_nu)

        R = (3.0*self.M/(4.0*np.pi*rhoM))**(1.0/3.0)

        #Get the window of a configuration space tophat
        kvec = (np.tile(k,[R.size,1]).T)
        Pk = np.tile(Pk,[R.size,1]).T
        R = np.tile(R,[k.size,1])
        x = ((kvec*R).decompose()).value
        W = 3.0*(np.sin(x) - x*np.cos(x))/(x)**3 
        
        #Compute sigma(M)
        integrnd = Pk*W**2*kvec**2/(2.*np.pi**2)
        sigma = np.sqrt(np.trapz(integrnd,kvec[:,0],axis=0))
        
        return sigma
        

    @cached_cosmo_property
    def sigmaMz0(self):
        '''
        Mass (or cdm+b) variance at redshift 0
        '''
        #Get R(M) and P(k)
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3
        k = np.logspace(-2,2,128)*u.Mpc**-1
        #Use rho_m or rho_cb depending on mnu
        if self.cosmo_code == 'camb':
            # BSM edit - add ncdm transfer function
            Pk = (self.PKint(0.,k.value)*u.Mpc**3)*self.transfer_ncdm(k)
            #
            rhoM = rho_crit*(self.camb_pars.omegam-self.camb_pars.omeganu)
        else:
            # BSM edit - add ncdm transfer function
            Pk = (self.PKint(k.value,np.array([0.]),len(k),1,0)*u.Mpc**3)*self.transfer_ncdm(k)
            #
            rhoM = rho_crit*(self.cosmo.Omega0_m()-self.cosmo.Omega_nu)

        R = (3.0*self.M/(4.0*np.pi*rhoM))**(1.0/3.0)

        #Get the window of a configuration space tophat
        kvec = (np.tile(k,[R.size,1]).T)
        Pk = np.tile(Pk,[R.size,1]).T
        R = np.tile(R,[k.size,1])
        x = ((kvec*R).decompose()).value
        W = 3.0*(np.sin(x) - x*np.cos(x))/(x)**3 
        
        #Compute sigma(M)
        integrnd = Pk*W**2*kvec**2/(2.*np.pi**2)
        sigma = np.sqrt(np.trapz(integrnd,kvec[:,0],axis=0))
        
        return sigma
        
        
    @cached_property
    def dsigmaM_dM(self):
        '''
        Computes the derivative of sigma(M) with respect to M at target redshift
        '''
        sigmaint = log_interp1d(self.M,self.sigmaM,fill_value='extrapolate')
        Mminus = self.M/1.0001
        Mplus =  self.M*1.0001
        sigma_minus = sigmaint(Mminus.value)
        sigma_plus = sigmaint(Mplus.value)
        return (sigma_plus-sigma_minus)/(Mplus-Mminus)
        
        
    @cached_cosmo_property
    def dsigmaM_dM_z0(self):
        '''
        Computes the derivative of sigma(M) with respect to M at z=0
        '''
        sigmaint = log_interp1d(self.M,self.sigmaMz0,fill_value='extrapolate')
        Mminus = self.M/1.0001
        Mplus =  self.M*1.0001
        sigma_minus = sigmaint(Mminus.value)
        sigma_plus = sigmaint(Mplus.value)
        return (sigma_plus-sigma_minus)/(Mplus-Mminus)
    
    
    @cached_property
    def bofM(self):
        '''
        Halo bias as a function of mass (and scale, if fNL != 0).  
        '''
        # nonlinear overdensity
        dc = 1.686
        nu = dc/self.sigmaM
        
        bias = np.tile(getattr(bm,self.bias_model)(self,dc,nu),(self.k.size,1)).T
        Delta_b = 0.
        if self.f_NL != 0:
            #get the transfer function, depending on whether mnu = 0 or mnu > 0
            if self.cosmo_code == 'camb':
                if self.camb_pars.num_nu_massive != 0:
                    Tk = self.transfer_cb(self.k.value)
                else:
                    Tk = self.transfer_m(self.k.value)
                Om0 = self.camb_pars.omegam
            else:
                if self.cosmo.Omega_nu != 0:
                    Tk = self.transfer_cb(self.k.value)
                else:
                    Tk = self.transfer_m(self.k.value)
                Om0 = self.cosmo.Omega0_m()
            #Compute non-Gaussian correction Delta_b
            factor = self.f_NL*dc*                                      \
                      3.*Om0*(100.*self.hubble*(u.km/u.s/u.Mpc))**2./   \
                     (cu.c.to(u.km/u.s)**2.*self.k**2*(Tk/np.max(Tk))*self.Dgrowth(self.z))
            Delta_b = (bias-1.)*np.tile(factor,(self.nM,1))
            
        return bias + Delta_b
        
        
    @cached_property
    def c_NFW(self):
        '''
        concentration-mass relation for the NFW profile.
        Following Diemer & Joyce (2019)
        c = R_delta / r_s (the scale radius, not the sound horizon)
        '''
        #smaller sampling of M
        Mvec = ulogspace(self.Mmin,self.Mmax,256).value
        #fit parameters
        kappa = 0.42
        a0 = 2.37
        a1 = 1.74
        b0 = 3.39
        b1 = 1.82
        ca = 0.2
        #Compute the effective slope of the growth factor
        dz = self.z*0.001
        alpha_eff = -(np.log(self.Dgrowth(self.z+dz))-np.log(self.Dgrowth(self.z-dz)))/ \
                    (np.log(1.+self.z+dz)-np.log(1.+self.z-dz))
        #Compute the effective slope to the power spectrum (as function of M)
        fun_int = -2.*3.*self.M/self.sigmaM*self.dsigmaM_dM-3.
        neff = interp1d(np.log10(self.M.value),fun_int,fill_value='extrapolate',kind='linear')(np.log10(kappa*Mvec))
        #Quantities for c
        A = a0*(1.+a1*(neff+3))
        B = b0*(1.+b1*(neff+3))
        C = 1.-ca*(1.-alpha_eff)
        nu = 1.686/log_interp1d(self.M.value,self.sigmaM)(Mvec)
        arg = A/nu*(1.+nu**2/B)
        #Compute G(x), with x = r/r_s, and evaluate c
        x = np.logspace(-3,3,256)
        g = np.log(1+x)-x/(1.+x)

        c = np.zeros(len(Mvec))
        for iM in range(len(Mvec)):
            G = x/g**((5.+neff[iM])/6.)
            invG = log_interp1d(G,x,fill_value='extrapolate',kind='linear')
            c[iM] = C*invG(arg[iM])
            
        return log_interp1d(Mvec,c,fill_value='extrapolate',kind='cubic')(self.M.value)
        
        
    @cached_property
    def ft_NFW(self):
        '''
        Fourier transform of NFW profile, for computing one-halo term
        '''
        #Radii of the SO collapsed (assuming 200*rho_crit)
        Delta = 200.
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3
        R_NFW = (3.*self.M/(4.*np.pi*Delta*rho_crit))**(1./3.)
        #get characteristic radius
        r_s = np.tile(R_NFW/self.c_NFW,(self.nk,1)).T
        #concentration to multiply with ki
        c = np.tile(self.c_NFW,(self.nk,1)).T
        gc = np.log(1+c)-c/(1.+c)
        #argument: k*rs
        ki = np.tile(self.k,(self.nM,1))
        x = ((ki*r_s).decompose()).value        
        si_x, ci_x = sici(x)
        si_cx, ci_cx = sici((1.+c)*x)
        u_km = (np.cos(x)*(ci_cx - ci_x) +
                  np.sin(x)*(si_cx - si_x) - np.sin(c*x)/((1.+c)*x))
        return u_km/gc
        
        
    @cached_property
    def bavg(self):
        '''
        Average luminosity-weighted bias for the given cosmology and line
        model.  ASSUMED TO BE WEIGHTED LINERALY BY MASS FOR 'LF' MODELS
        
        Includes the effect of f_NL (inherited from bofM)
        '''
        #Apply dNL correction if model_type = TOY
        if self.model_type == 'TOY':
            dc = 1.686
            Delta_b = 0.
            b_line = self.model_par['bmean']*np.ones(self.nk)
            if self.f_NL != 0:
                #get the transfer function, depending on whether mnu = 0 or mnu > 0
                if self.cosmo_code == 'camb':
                    if self.camb_pars.num_nu_massive != 0:
                        Tk = self.transfer_cb(self.k.value)
                    else:
                        Tk = self.transfer_m(self.k.value)
                    Om0 = self.camb_pars.omegam
                else:
                    if self.cosmo.Omega_nu != 0:
                        Tk = self.transfer_cb(self.k.value)
                    else:
                        Tk = self.transfer_m(self.k.value)
                    Om0 = self.cosmo.Omega0_m()
                #Compute non-Gaussian correction Delta_b
                factor = self.f_NL*dc*                                      \
                          3.*Om0*(100.*self.hubble*(u.km/u.s/u.Mpc))**2./   \
                         (cu.c.to(u.km/u.s)**2.*self.k**2*(Tk/np.max(Tk))*self.Dgrowth(self.z))
                Delta_b = (bias-1.)*np.tile(factor,(self.nM,1))
                b_line += Delta_b
        else:
            # Integrands for mass-averaging
            factor = np.tile(self.LofM*self.dndM,(self.nk,1)).T
            itgrnd1 = self.bofM*factor
            itgrnd2 = factor
            
            b_line = np.trapz(itgrnd1,self.M,axis=0) / np.trapz(itgrnd2,self.M,axis=0)
        
        return b_line 
    
    
    @cached_property
    def nbar(self):
        '''
        Mean number density of galaxies, computed from the luminosity function
        in 'LF' models and from the mass function in 'ML' models
        '''
        if self.model_type=='LF':
            nbar = np.trapz(self.dndL,self.L)
        else:
            nbar = np.trapz(self.dndM,self.M)
        return nbar        
        
    #############################
    # Power spectrum quantities #
    #############################
    @cached_property
    def RSD(self):
        '''
        Kaiser factor and FoG for RSD
        '''
        if self.do_RSD == True:
            kaiser = (1.+self.f_eff/self.bavg*self.mui_grid**2.)**2. #already squared
            
            if self.FoG_damp == 'Lorentzian':
                FoG = (1.+0.5*(self.k_par*self.sigma_NL).decompose()**2.)**-2.
            elif self.FoG_damp == 'Gaussian':
                FoG = np.exp(-((self.k_par*self.sigma_NL)**2.)
                        .decompose()) 
            else:
                raise ValueError('Only Lorentzian or Gaussian damping terms for FoG')
                
            return FoG*kaiser
        else:
            return np.ones(self.Pm.shape)

    @cached_property
    def Wline(self):
        '''
        Fourier-space factor for Gaussian line profile, in k-mu-M grid.
        Applicable for shot noise power spectrum.
        
        By Dongwoo T. Chung
        '''
        if self.v_of_M is not None:
            vvec = self.v_of_M(self.M).to(u.km/u.s)
            sigma_v_of_M = ((1+self.z)/self.H*vvec/2.35482).to(u.Mpc)
            if self.line_incli:
                return dawsn_over_x_non_divergent(2/3**0.5*self.k_par[...,None]*sigma_v_of_M[None,None,:])
            else:
                return np.exp(-(self.k_par[...,None]*sigma_v_of_M[None,None,:])**2)
        else:
            return np.ones(self.Pm.shape+self.M.shape)
            
    @cached_property
    def Wline_clust(self):
        '''
        Fourier-space factor for Gaussian line profile, in k-mu-M grid.
        Just sqrt(Wline) for line_incli==False, but subtly different otherwise.
        
        By Dongwoo T. Chung
        '''
        if self.v_of_M is not None:
            if self.line_incli:
                vvec = self.v_of_M(self.M).to(u.km/u.s)
                sigma_v_of_M = ((1+self.z)/self.H*vvec/2.35482).to(u.Mpc)
                return dawsn_over_x_non_divergent((2/3)**0.5*self.k_par[...,None]*sigma_v_of_M[None,None,:])
            else:
                return self.Wline**0.5
        else:
            return np.ones(self.Pm.shape+self.M.shape)

    @cached_property
    def Pm(self):
        '''
        Matter power spectrum from the interpolator computed by camb. 
        '''
        if self.cosmo_code == 'camb':
            # BSM edit - add ncdm transfer function
            T2k = np.tile(self.transfer_ncdm(self.k),(self.nmu,1))
            return (self.PKint(self.z,self.ki_grid.value)*u.Mpc**3)*T2k
            #
        else:
            # BSM edit - add ncdm transfer function
            Pkvec = (self.PKint(self.k.value,np.array([self.z]),self.nk,1,0)*u.Mpc**3)*self.transfer_ncdm(self.k)
            #
            return np.tile(Pkvec,(self.nmu,1))         
    
    @cached_property
    def Lmean(self):
        '''
        Sky-averaged luminosity density at nuObs from target line.  Has
        two cases for 'LF' and 'ML' models
        '''
        if self.model_type=='LF':
            itgrnd = self.L*self.dndL
            Lbar = np.trapz(itgrnd,self.L)
        elif self.model_type == 'ML':
            itgrnd = self.LofM*self.dndM
            Lbar = np.trapz(itgrnd,self.M)*self.fduty
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.model_name=='TonyLi':
                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                Lbar = Lbar*np.exp((alpha**-2-alpha**-1)
                                    *sig_SFR**2*np.log(10)**2/2.)
        return Lbar
        
        
    @cached_property
    def L2mean(self):
        '''
        Sky-averaged squared luminosity density at nuObs from target line.  Has
        two cases for 'LF' and 'ML' models
        '''
        if self.model_type=='LF':
            itgrnd = self.L**2*self.dndL
            L2bar = np.trapz(itgrnd,self.L)
        elif self.model_type=='ML':
            itgrnd = self.LofM**2*self.dndM
            L2bar = np.trapz(itgrnd,self.M)*self.fduty
            # Add L vs. M scatter
            L2bar = L2bar*np.exp(self.sigma_scatter**2*np.log(10)**2)
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.model_name=='TonyLi':
                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                L2bar = L2bar*np.exp((2.*alpha**-2-alpha**-1)
                                    *sig_SFR**2*np.log(10)**2)
        return L2bar
        
        
    @cached_property
    def Tmean(self):
        '''
        Sky-averaged brightness temperature at nuObs from target line.  
        You can directly input Tmean using TOY model
        '''
        if self.model_type == 'TOY':
            return self.model_par['Tmean']
        else:
            return self.CLT*self.Lmean
        
    @cached_property
    def Pshot(self):
        '''
        Shot noise amplitude for target line at frequency nuObs.  Has two
        cases for 'LF' and 'ML' models. 
        You can directly input T2mean using TOY model
        '''
        
        if self.model_type == 'TOY':
            return self.model_par['Pshot']
        #Consider line broadening (code by Dongwoo T. Chung)
        elif self.v_of_M is not None:
            if self.model_type == 'ML':
                itgrnd = (self.LofM**2*self.dndM)[None,None,:]*self.Wline
                L2bar = np.trapz(itgrnd,self.M)*self.fduty
                # Add L vs. M scatter
                L2bar = L2bar*np.exp(self.sigma_scatter**2*np.log(10)**2)
                # Special case for Tony Li model- scatter does not preserve LCO
                if self.model_name=='TonyLi':
                    alpha = self.model_par['alpha']
                    sig_SFR = self.model_par['sig_SFR']
                    L2bar = L2bar*np.exp((2.*alpha**-2-alpha**-1)
                                        *sig_SFR**2*np.log(10)**2)
            else:
                print("Line width modelling only available for ML models")
                L2bar = 1.*self.L2mean
            return self.CLT**2*L2bar
        else:
            return self.CLT**2*self.L2mean
        
    @cached_property
    def Pk_twohalo(self):
        '''
        Two-halo term in power spectrum, equal to Tmean^2*bavg^2*Pm if
        do_onehalo=False
        '''
        if self.do_onehalo:
            if self.model_type=='LF':
                print("One halo term only available for ML models")
                wt = self.Tmean*self.bavg
            else:
                if self.v_of_M is not None:
                    Mass_Dep = (self.LofM*self.dndM)[None,None,:]*self.Wline_clust
                    itgrnd = (self.ft_NFW*self.bofM).T[None,:,:]*Mass_Dep
                    wt = self.CLT*np.trapz(itgrnd,self.M,axis=2)*self.fduty
                    # Special case for SFR(M) scatter in Tony Li model
                    if self.model_name=='TonyLi':
                        alpha = self.model_par['alpha']
                        sig_SFR = self.model_par['sig_SFR']
                        wt = wt*np.exp((alpha**-2-alpha**-1)
                                        *sig_SFR**2*np.log(10)**2/2.)
                else:
                    Mass_Dep = self.LofM*self.dndM
                    itgrnd = np.tile(Mass_Dep,(self.k.size,1)).T*self.ft_NFW*self.bofM
                    wt = self.CLT*np.trapz(itgrnd,self.M,axis=0)*self.fduty
                    # Special case for SFR(M) scatter in Tony Li model
                    if self.model_name=='TonyLi':
                        alpha = self.model_par['alpha']
                        sig_SFR = self.model_par['sig_SFR']
                        wt = wt*np.exp((alpha**-2-alpha**-1)
                                        *sig_SFR**2*np.log(10)**2/2.)
        else:
            if self.v_of_M is not None:
                if self.model_type == 'ML':
                    itgrnd = (self.LofM*self.dndM)[None,None,:]*self.Wline_clust
                    itgrnd*= self.bofM.T[None,:,:]
                    wt = self.CLT*np.trapz(itgrnd,self.M)*self.fduty
                    # Special case for Tony Li model- scatter does not preserve LCO
                    if self.model_name=='TonyLi':
                        alpha = self.model_par['alpha']
                        sig_SFR = self.model_par['sig_SFR']
                        wt*= np.exp((alpha**-2-alpha**-1)
                                            *sig_SFR**2*np.log(10)**2/2.)
                else:
                    print("Line width modelling only available for ML models")
                    wt = self.Tmean*self.bavg
            else:
                wt = self.Tmean*self.bavg
            
        return wt**2*self.Pm
        
        
    @cached_property
    def Pk_onehalo(self):
        '''
        One-halo term in power spectrum
        '''
        if self.do_onehalo:
            if self.model_type=='LF':
                print("One halo term only available for ML models")
                return np.zeros(self.Pm.shape)*self.Pshot.unit
            else:
                if self.v_of_M is not None:
                    Mass_Dep = (self.LofM**2*self.dndM)[None,None,:]*self.Wline
                    itgrnd = (self.ft_NFW**2.).T[None,:,:]*Mass_Dep
                    #add effect for the scatter in LCO
                    itgrnd = itgrnd*np.exp(self.sigma_scatter**2*np.log(10)**2)
                    # Special case for Tony Li model- scatter does not preserve LCO
                    if self.model_name=='TonyLi':
                        alpha = self.model_par['alpha']
                        sig_SFR = self.model_par['sig_SFR']
                        itgrnd = itgrnd*np.exp((2.*alpha**-2-alpha**-1)
                                            *sig_SFR**2*np.log(10)**2)
                    wt = np.trapz(itgrnd,self.M,axis=2)*self.fduty
                    return self.CLT**2.*wt
                else:
                    Mass_Dep = self.LofM**2.*self.dndM
                    itgrnd = np.tile(Mass_Dep,(self.nk,1)).T*self.ft_NFW**2.
                    #add effect for the scatter in LCO
                    itgrnd = itgrnd*np.exp(self.sigma_scatter**2*np.log(10)**2)
                                
                    # Special case for Tony Li model- scatter does not preserve LCO
                    if self.model_name=='TonyLi':
                        alpha = self.model_par['alpha']
                        sig_SFR = self.model_par['sig_SFR']
                        itgrnd = itgrnd*np.exp((2.*alpha**-2-alpha**-1)
                                            *sig_SFR**2*np.log(10)**2)
                    wt = np.trapz(itgrnd,self.M,axis=0)*self.fduty
                    return np.tile(self.CLT**2.*wt,(self.nmu,1))
        else:
            return np.zeros(self.Pm.shape)*self.Pshot.unit
    
    
    @cached_property    
    def Pk_clust(self):
        '''
        Clustering power spectrum of target line, i.e. power spectrum without
        shot noise.
        '''
        return (self.Pk_twohalo+self.Pk_onehalo)*self.RSD
        
    
    @cached_property    
    def Pk_shot(self):
        '''
        Shot-noise power spectrum of target line, i.e. power spectrum without
        clustering
        '''
        return self.Pshot*np.ones(self.Pm.shape)
        
    
    @cached_property    
    def Pk(self):
        '''
        Full line power spectrum including both clustering and shot noise 
        as function of k and mu.
        
        If do_conv_Wkmin, convolve with the survey mask window assuming a 
        cylindrical volume
        '''
        if self.smooth:
            if self.do_conv_Wkmin:
                Pkres = self.Wk*(self.Pk_clust+self.Pk_shot)
                
                #Get the vector to integrate over
                qe = np.logspace(-4,2,self.nk+1)
                q = 0.5*(qe[:-1]+qe[1:])*u.Mpc**-1
                muq = self.mu
                
                qi_grid,muqi_grid = np.meshgrid(q,muq)
                q_par = qi_grid*muqi_grid
                q_perp = qi_grid*np.sqrt(1-muqi_grid**2)
                
                #get the window to convolve with
                L_perp=np.sqrt(self.Sfield/np.pi)
                Wpar = 2*np.sin((q_par*self.Lfield/2).value)/q_par
                Wperp = 2*np.pi*L_perp*j1(q_perp*L_perp)/q_perp
                Wconv = Wpar*Wperp
                
                #Do the convolution
                Pconv = np.zeros(Pkres.shape)*Pkres.unit*self.Vfield.unit
                Pkres_interp = RegularGridInterpolator((self.k.value,self.mu),Pkres.T.value, bounds_error=False, fill_value=0)
                for ik in range(self.nk):
                    for imu in range(self.nmu):
                        #Get the unconvolved power spectrum in the sum of vectors
                        qsum_grid,musum_grid = add_vector(self.k[ik],self.mu[imu],qi_grid,-muqi_grid)
                        Pconv[imu,ik] = np.trapz(np.trapz(qi_grid**2*Pkres_interp((qsum_grid.value,musum_grid.value))*Pkres.unit*np.abs(Wconv**2)/(2*np.pi)**2,muq,axis=0),q)

                return Pconv/self.Vfield
            else:
                return self.Wk*(self.Pk_clust+self.Pk_shot)
        else:
            return self.Pk_clust+self.Pk_shot
            
        
    @cached_property
    def Pk_0(self):
        '''
        Monopole of the power spectrum as function of k
        '''
        return 0.5*np.trapz(self.Pk,self.mu,axis=0)
        
        
    @cached_property
    def Pk_2(self):
        '''
        Quadrupole of the power spectrum as function of k
        '''
        L2 = legendre(2)
        return 2.5*np.trapz(self.Pk*L2(self.mui_grid),self.mu,axis=0)
        
        
    @cached_property
    def Pk_4(self):
        '''
        Hexadecapole of the power spectrum as function of k
        '''
        L4 = legendre(4)
        return 4.5*np.trapz(self.Pk*L4(self.mui_grid),self.mu,axis=0)
        
        
    def Pk_l(self,l):
        '''
        Multipole l of the power spectrum
        '''
        if l == 0:
            return self.Pk_0
        elif l == 2:
            return self.Pk_2
        elif l == 4:
            return self.Pk_4
        else:
            Ll = legendre(l)
            return (2.*l+1.)/2.*np.trapz(self.Pk*Ll(self.mui_grid),
                                        self.mu,axis=0)
                 
                 
    #############################################
    #############################################
    ### Voxel Intensity Distribution Functions ##
    #############################################
    #############################################
        
    @cached_vid_property
    def T(self):
        '''
        Centers of intensity bins (defined from the nufft1d1 routine)
        '''
        dT = 2*self.Tmax_VID/self.nT
        Tvec = dT*np.arange(-self.nT/2,self.nT/2)
        return Tvec
        
    ##########################################
    # Single-source characteristic functions #
    ##########################################
    
    @cached_vid_property
    def smooth_vox_pop(self):
        '''
        Number of voxels with contributions from a single source after
        smoothing, as function of mass if line broadening is included. 
        (assumes a voxes is determined by the FWHM of the beam and the std 
        of the channel)
        
        Returns unique values and the fraction of number counts for each
        
        Assumes Gaussian smoothing!
        '''
        par_side = self.sigma_par.value/0.4247
        perp_side = self.sigma_perp.value/0.4247
        if self.smooth_VID:
            Nsigma_prof = 20
            supersample = 2
            if self.v_of_M is not None:
                if self.line_incli:
                    vvec = self.v_of_M(self.M).to(u.km/u.s)
                    sigma_v_of_M = ((1+self.z)/self.H*vvec/2.35482).to(u.Mpc).value
                    sigma_par_res = self.sigma_par.value
                    #Get the smoothing scales and the number of voxels that occupies
                    sperp = self.sigma_perp.value
                    Nvox_perp = int(2*Nsigma_prof*sperp/perp_side+1)
                    Nvox_perp += Nvox_perp % 2 == 0
                    lenvec_perp = Nvox_perp*supersample + Nvox_perp*supersample%2
                    x_perp = np.linspace(-Nvox_perp*perp_side/2,Nvox_perp*perp_side/2,lenvec_perp)
                    #line broadening convolved with the resolution: numerically using convolution theorem
                    Ngauss = 800 #number of evaluations for the integral
                    nodes,weights = roots_legendre(Ngauss) #obtain the nodes and weights for the GL integral
                    #populate the voxels
                    unique_L, counts = [[] for iM in range(self.nM)],[[] for iM in range(self.nM)]      
                    for iM in range(self.nM):
                        spar = np.sqrt(sigma_par_res**2. + sigma_v_of_M[iM]**2.)
                        kmax,kmin = 3*spar,0
                        kvec = (kmax-kmin)/2*nodes + (kmax+kmin)/2 
                        fPDF_res = np.exp(-kvec**2*sigma_par_res**2/2)
                        #Get the smoothing scales and the number of voxels that occupies
                        Nvox_par = int(2*Nsigma_prof*spar/par_side+1)
                        Nvox_par += Nvox_par % 2 == 0
                        lenvec_par = Nvox_par*supersample + Nvox_par*supersample%2
                        x_par = np.linspace(-Nvox_par*par_side/2,Nvox_par*par_side/2,lenvec_par)
                        #convolution along the LOS first
                        fPDF_broad = dawsn_over_x_non_divergent((2/3)**0.5*kvec*sigma_v_of_M[iM])
                        fPDF_tot = fPDF_broad*fPDF_res
                        #go back to configuration space
                        xvec = np.linspace(-Nvox_par*par_side/2,Nvox_par*par_side/2,100)
                        fPDF_tot_int = (kmax-kmin)/2*weights*fPDF_tot #normalization, weights and new variable for the change of interval
                        Ptot = (finufft.nufft1d3(kvec,fPDF_tot_int.real+1j*fPDF_tot_int.imag,xvec,eps=1e-6,isign=1)/np.pi/2 + \
                             finufft.nufft1d3(-kvec,fPDF_tot_int.real-1j*fPDF_tot_int.imag,xvec,eps=1e-6,isign=1)/np.pi/2).real 
                        Ptot_i = interp1d(xvec,Ptot,kind='cubic')
                        #Get the integrated luminosity that would be in each voxel after smoothing
                        L_dif = np.zeros((lenvec_par-1,lenvec_perp-1,lenvec_perp-1))
                        for i in range(lenvec_par-1):
                            x_interval = (x_par[i+1]-x_par[i])/2*nodes + (x_par[i+1]+x_par[i])/2
                            los_integral = np.sum((x_par[i+1]-x_par[i])/2*Ptot_i(x_interval)*weights)
                            for j in range(lenvec_perp-1):
                                perp_integral = (erf(x_perp[j+1]/2**0.5/sperp)-erf(x_perp[j]/2**0.5/sperp))/2
                                for k in range(lenvec_perp-1):
                                    L_dif[i,j,k] = los_integral*perp_integral*(erf(x_perp[k+1]/2**0.5/sperp)-erf(x_perp[k]/2**0.5/sperp))/2
                        #downsample to get values for voxel size
                        L_dif_new = np.zeros((int(L_dif.shape[0]/supersample)+1,int(L_dif.shape[1]/supersample)+1,int(L_dif.shape[2]/supersample)+1))
                        for i in range(L_dif_new.shape[0]):
                            for j in range(L_dif_new.shape[1]):
                                for k in range(L_dif_new.shape[2]):
                                    L_dif_new[i,j,k] = np.sum(L_dif[supersample*i:supersample*(i+1),supersample*j:supersample*(j+1),supersample*k:supersample*(k+1)])
                        Lmax = np.max(L_dif_new)
                        #get unique values with a given tolerance, removing values below that tolerance
                        unique_L_iM, counts_iM = np.unique(np.around(L_dif_new/Lmax,decimals=self.Lsmooth_tol),return_counts = True)
                        unique_L_iM *= Lmax
                        #renormalize unique_L
                        unique_L_iM /= np.sum(unique_L_iM*counts_iM)
                        unique_L[iM],counts[iM] = unique_L_iM[1:], counts_iM[1:]
                    return unique_L, counts
                else:
                    vvec = self.v_of_M(self.M).to(u.km/u.s)
                    sigma_v_of_M = ((1+self.z)/self.H*vvec/2.35482).to(u.Mpc).value
                    #Get the smoothing scales and the number of voxels that occupies
                    sperp = self.sigma_perp.value
                    Nvox_perp = int(2*Nsigma_prof*sperp/perp_side+1)
                    Nvox_perp += Nvox_perp % 2 == 0
                    lenvec_perp = Nvox_perp*supersample + Nvox_perp*supersample%2
                    x_perp = np.linspace(-Nvox_perp*perp_side/2,Nvox_perp*perp_side/2,lenvec_perp)
                    #line broadening convolved with the resolution
                    unique_L, counts = [[] for iM in range(self.nM)],[[] for iM in range(self.nM)] 
                    for iM in range(self.nM):
                        #Get the smoothing scales and the number of voxels that occupies
                        spar = np.sqrt(self.sigma_par.value**2. + sigma_v_of_M[iM]**2.)
                        Nvox_par = int(2*Nsigma_prof*spar/par_side+1)
                        Nvox_par += Nvox_par % 2 == 0
                        lenvec_par = Nvox_par*supersample + Nvox_par*supersample%2
                        x_par = np.linspace(-Nvox_par*par_side/2,Nvox_par*par_side/2,lenvec_par)
                        #Get the integrated luminosity that would be in each voxel after smoothing
                        L_dif = np.zeros((lenvec_par-1,lenvec_perp-1,lenvec_perp-1))
                        for i in range(lenvec_par-1):
                            los_integral = (erf(x_par[i+1]/2**0.5/spar)-erf(x_par[i]/2**0.5/spar))/2
                            for j in range(lenvec_perp-1):
                                perp_integral = (erf(x_perp[j+1]/2**0.5/sperp)-erf(x_perp[j]/2**0.5/sperp))/2
                                for k in range(lenvec_perp-1):
                                    L_dif[i,j,k] = los_integral*perp_integral*(erf(x_perp[k+1]/2**0.5/sperp)-erf(x_perp[k]/2**0.5/sperp))/2
                        #downsample to get values for voxel size
                        L_dif_new = np.zeros((int(L_dif.shape[0]/supersample)+1,int(L_dif.shape[1]/supersample)+1,int(L_dif.shape[2]/supersample)+1))
                        for i in range(L_dif_new.shape[0]):
                            for j in range(L_dif_new.shape[1]):
                                for k in range(L_dif_new.shape[2]):
                                    L_dif_new[i,j,k] = np.sum(L_dif[supersample*i:supersample*(i+1),supersample*j:supersample*(j+1),supersample*k:supersample*(k+1)])
                        Lmax = np.max(L_dif_new)
                        #get unique values with a given tolerance, removing values below that tolerance
                        unique_L_iM, counts_iM = np.unique(np.around(L_dif_new/Lmax,decimals=self.Lsmooth_tol),return_counts = True)
                        unique_L_iM *= Lmax
                        #renormalize unique_L
                        unique_L_iM /= np.sum(unique_L_iM*counts_iM)
                        unique_L[iM],counts[iM] = unique_L_iM[1:], counts_iM[1:]
                    return unique_L, counts
            else:
                #Get the smoothing scales and the number of voxels that occupies
                spar,sperp = self.sigma_par.value,self.sigma_perp.value
                Nvox_par = int(2*Nsigma_prof*spar/par_side+1)
                Nvox_par += Nvox_par % 2 == 0
                Nvox_perp = int(2*Nsigma_prof*sperp/perp_side+1)
                Nvox_perp += Nvox_perp % 2 == 0
                lenvec_par = Nvox_par*supersample + Nvox_par*supersample%2
                lenvec_perp = Nvox_perp*supersample + Nvox_perp*supersample%2
                x_par = np.linspace(-Nvox_par*par_side/2,Nvox_par*par_side/2,lenvec_par)
                x_perp = np.linspace(-Nvox_perp*perp_side/2,Nvox_perp*perp_side/2,lenvec_perp)
                #Get the integrated luminosity that would be in each voxel after smoothing
                L_dif = np.zeros((lenvec_par-1,lenvec_perp-1,lenvec_perp-1))
                for i in range(lenvec_par-1):
                    los_integral = (erf(x_par[i+1]/2**0.5/spar)-erf(x_par[i]/2**0.5/spar))/2
                    for j in range(lenvec_perp-1):
                        perp_integral = (erf(x_perp[j+1]/2**0.5/sperp)-erf(x_perp[j]/2**0.5/sperp))/2
                        for k in range(lenvec_perp-1):
                            L_dif[i,j,k] = los_integral*perp_integral*(erf(x_perp[k+1]/2**0.5/sperp)-erf(x_perp[k]/2**0.5/sperp))/2
                L_dif_new = np.zeros((int(L_dif.shape[0]/supersample)+1,int(L_dif.shape[1]/supersample)+1,int(L_dif.shape[2]/supersample)+1))
                for i in range(L_dif_new.shape[0]):
                    for j in range(L_dif_new.shape[1]):
                        for k in range(L_dif_new.shape[2]):
                            L_dif_new[i,j,k] = np.sum(L_dif[supersample*i:supersample*(i+1),supersample*j:supersample*(j+1),supersample*k:supersample*(k+1)])
                Lmax = np.max(L_dif_new)
                #get unique values with a given tolerance, removing values below that tolerance
                unique_L, counts = np.unique(np.around(L_dif_new/Lmax,decimals=self.Lsmooth_tol),return_counts = True)
                unique_L *= Lmax
                #renormalize unique_L
                unique_L /= np.sum(unique_L*counts)
                return unique_L[1:], counts[1:]
        else:
            #if no smoothing, just pass 1/sypersample^3 to have no effect in fPT
            return np.array([1]),np.array([1])
    
    @cached_vid_property
    def XLT(self):
        '''
        Constant relating total luminosity in a voxel to its observed
        intensity.  Equal to CLT/Vvox
        '''
        return self.CLT/self.Vvox
        
    @cached_vid_property
    def leggaus_prep_FT(self):
        '''
        Get the position of the nodes for the Gauss-Legendre quadrature, and
        the corresponding weights. Can be precomputed and read a table, or 
        computed using scipy. 
        This will be used for the Fourier transform
        '''
        if type(self.n_leggauss_nodes_FT) == str:
            mat = np.loadtxt(self.n_leggauss_nodes_FT)
            return mat[:,0],mat[:,1]
        else:
            return roots_legendre(self.n_leggauss_nodes_FT)
    
    @cached_vid_property
    def leggaus_prep_IFT(self):
        '''
        Get the position of the nodes for the Gauss-Legendre quadrature, and
        the corresponding weights. Can be precomputed and read a table, or 
        computed using scipy. 
        This will be used for the Inverse Fourier transform
        '''
        if type(self.n_leggauss_nodes_IFT) == str:
            mat = np.loadtxt(self.n_leggauss_nodes_IFT)
            return mat[:,0],mat[:,1]
        else:
            return roots_legendre(self.n_leggauss_nodes_IFT)
            
    @cached_vid_property
    def fP1_0_fun(self):
        '''
        Generates interpolated function for the single-source characteristic
        function at an arbitrary halo mass.  Assumes a lognormal scatter with
        width given by self.sigma_scatter.
        '''
        #change between unit temperature and luminosity
        LofM0 = 1*self.Tmean.unit/self.XLT
        #dummy temperature (and corresponding luminosity) vector
        exp = self.T0_Nlogsigma*self.sigma_scatter
        Tmax_log = 10**(exp)*self.Tmean.unit
        Tmin_log = 10**-(2*exp)*self.Tmean.unit
        Nbin_log = 2**15+1
        
        Tedge_log = ulogspace(Tmin_log,Tmax_log,Nbin_log)
        Tlog = binedge_to_binctr(Tedge_log)
        L = Tlog/self.XLT
        
        #Mean-preserving lognormal distribution to multiply
        mu0 = 0.5*self.sigma_scatter**2*np.log(10)-np.log10(LofM0/u.Lsun)
        P1_0 = (np.exp(-(np.log10(L/u.Lsun)+mu0)**2/(2*self.sigma_scatter**2))/
                (self.XLT*L*self.sigma_scatter*np.log(10)*np.sqrt(2*np.pi)))
        #normalize just in case
        P1_0 = P1_0/np.trapz(P1_0,Tlog)
        #FT and interpolate. First get the node positions in the interval of interest
        nodes,weights = self.leggaus_prep_FT
        T2 = (np.max(Tlog)-np.min(Tlog))/2*nodes + (np.max(Tlog)+np.min(Tlog))/2
        #interpolate and evaluate in interva (dimensionless, already normalized)
        P1_0_i = interp1d(Tlog,P1_0,bounds_error=False,fill_value=0)(T2)*self.Tmean.unit**-1
        P1_toFT = (np.max(Tlog)-np.min(Tlog))/2*weights*P1_0_i
        #positions in Fourier space
        fT0 = ulogspace(self.fT0_min,self.fT0_max,self.nfT0)
        #compute the FT
        fP1_0 = finufft.nufft1d3(T2,P1_toFT+0j,fT0,eps=1e-6,isign=-1)
        #interpolate FT
        return interp1d(fT0,fP1_0,fill_value=(1.+0j,0.+0j),bounds_error=False)
        
    def calc_fP1_pos(self,fT,Li):
        '''
        Function to calculate the single-source characteristic function for
        sources with a given mean luminosity Li.  Only works for positive fT
        values, full characteristic function is supplied by self.calc_fP1
        '''
        if np.any(fT<0):
            raise ValueError('All fT values given to fP1_pos must be positive or zero')

        LofM0 = 1*self.Tmean.unit/self.XLT
        Lratio = Li/LofM0
        return self.fP1_0_fun(fT*Lratio)
        
    def calc_fP1(self,Li):
        '''
        Calculates the full single-source characteristic function for sources
        with a given mean luminosity Li.  Uses calc_fP1_pos and symmetry requirements
        given that the real-space P1 is real.
        Not used right now (fT always positive + symmetry considerations), 
        but may be useful for the future
        '''
        if np.any(self.fT<0):
            fT_a = -self.fT[self.fT<0]
            fP1_a = np.conjugate(self.calc_fP1_pos(fT_a,Li))
        else:
            fP1_a = np.array([])
        
        fT_b = self.fT[self.fT>=0]
        fP1_b = self.calc_fP1_pos(fT_b,Li)
        return np.append(fP1_a,fP1_b)
        
    @cached_vid_property
    def fT_and_edges(self):
        '''
        Fourier conjugate bin centers 
        (and indices for each limit interval)
        '''
        nodes,weights = self.leggaus_prep_IFT
        #Have some fT edges log-spaced, the rest lin-space, keeping 3pi distance for nufft
        Npi = 2
        self.Npi_fT = Npi
        #number of log bins
        dT = 2*self.Tmax_VID/self.nT
        fTlog_max = Npi*np.pi/dT
        dex_fTlog = int(np.ceil(np.log10(fTlog_max/self.fT_min)+10))
        fT_bins_log = np.logspace(np.log10(self.fT_min.value),np.log10(fTlog_max.value),dex_fTlog)*self.fT_min.unit
        #number of lin bins
        fT_bins_lin = np.arange(fTlog_max*dT,self.fT_max*dT,2*Npi*np.pi)[1:]/dT
        #put them together
        fT_bins = np.concatenate((fT_bins_log,fT_bins_lin))
        
        fT = np.zeros(len(nodes)*(len(fT_bins)-1))*self.fT_min.unit
        fT_Nind = np.arange(len(fT_bins))*len(nodes)
        for ifT in range(len(fT_bins)-1):
            fT[fT_Nind[ifT]:fT_Nind[ifT+1]] = (fT_bins[ifT+1]-fT_bins[ifT])/2*nodes + (fT_bins[ifT+1]+fT_bins[ifT])/2
        #return the fT and the bins for the interval edges and the number of log bins
        return fT, fT_Nind, dex_fTlog
    
    ###################
    # VID calculation #
    ###################
    
    @cached_vid_property
    def fPT_N(self):
        '''
        Characteristic function of instrumental noise, assumed to be Gaussian
        '''
        fT = self.fT_and_edges[0]
        if self.do_Jysr:
            sigmaN = ( self.sigma_N/np.sqrt(self.tpix*self.Nfeeds) ).to(u.Jy/u.sr)
        else: 
            sigmaN = self.sigma_N
        return np.exp(-fT**2*sigmaN**2/2.)

    @cached_vid_property
    def PT_N(self):
        '''
        Noise probability distribution
        '''
        if self.do_Jysr:
            sigmaN = ( self.sigma_N/np.sqrt(self.tpix*self.Nfeeds) ).to(u.Jy/u.sr)
        else: 
            sigmaN = self.sigma_N
        return np.exp(-self.T**2/(2*sigmaN**2))/np.sqrt(2*np.pi*sigmaN**2)

    @cached_vid_property
    def Pvar(self):
        '''
        Variance of the matter power spectrum in scales of a cubic voxel
        (side length perp to line of sight corresponding to sigma_FWHM)
        '''
        kx = np.logspace(-3,2,200)
        kx = np.append(-kx[::-1],kx)*self.k.unit
        ky = np.logspace(-3,2,200)
        ky = np.append(-ky[::-1],ky)*self.k.unit
        kz = np.logspace(-3,2,200)
        kz = np.append(-kz[::-1],kz)*self.k.unit
        k = (kx[:,None,None]**2+ky[None,:,None]**2+kz[None,None,:]**2)**0.5
        
        bh = np.trapz(self.bofM*np.tile(self.dndM,(self.nk,1)).T,self.M,axis=0)/self.nbar
        if self.do_RSD:
            kaiser = (1.+np.mean(self.f_eff)/bh[0]*(kz[None,None,:]/k)**2.)**2. #already squared
            if self.FoG_damp == 'Lorentzian':
                FoG = (1.+0.5*(kz[None,None,:]*self.sigma_NL).decompose()**2.)**-2.
            elif self.FoG_damp == 'Gaussian':
                FoG = np.exp(-((kz[None,None,:]*self.sigma_NL)**2.).decompose()) 
            else:
                raise ValueError('Only Lorentzian or Gaussian damping terms for FoG')
        else:
            kaiser, FoG = 1,1
            
        if self.cosmo_code == 'camb':
            #use the nonlinear Pk
            if self.camb_pars.num_nu_massive != 0:
                var = 8
            else:
                var = 7
            
            PK = camb.get_matter_power_interpolator(self.camb_pars, zmin=self.z-0.01, 
                                                    zmax=self.z+0.01, nz_step=3, 
                                                    zs=None, kmax=100, nonlinear=True,
                                                    var1=var, var2=var, hubble_units=False, 
                                                    k_hunit=False, return_z_k=False,
                                                    k_per_logint=None, log_interp=False, 
                                                    extrap_kmax=True)
            # BSM edit - add ncdm transfer function
            #Pm = PK.P(self.z,k.value)*u.Mpc**3
            T2k = self.transfer_ncdm(k)
            Pm = (PK.P(self.z,k.value)*u.Mpc**3)*T2k
            #
        else:
            #with class, we would have to recompute everything
            # BSM edit - add ncdm transfer function
            #Pm = self.PKint(self.z,k.value)*u.Mpc**3
            T2k = self.transfer_ncdm(k)
            Pm = (self.PKint(self.z,k.value)*u.Mpc**3)*T2k
            #

        rad_perp = self.sigma_perp/0.4247
        rad_par = self.sigma_par/0.4247
        kxr = (kx*rad_perp).decompose().value/2.
        kyr = (ky*rad_perp).decompose().value/2.
        kzr = (kz*rad_perp).decompose().value/2.
        Wkx = np.sin(kxr)/kxr
        Wky = np.sin(kyr)/kyr
        Wkz = np.sin(kzr)/kzr

        itgrnd = (Wkx[:,None,None]*Wky[None,:,None]*Wkz[None,None,:])**2*Pm*FoG*kaiser/(8*np.pi**3)
        return simps(simps(simps(itgrnd,kz,axis=2),ky,axis=1),kx)
        
    @cached_vid_property
    def exp_fPT_signal(self):
        '''
        Exponential of the signal-only characteristic funtion for line intensity
        '''
        fT = self.fT_and_edges[0]
        Lsmooth,counts = self.smooth_vox_pop
        dndM = self.dndM
        bM = self.bofM[:,0]
        LofM = self.LofM
        pp = np.zeros((self.nM,len(fT)),dtype='complex128')*u.Msun**-1
        if self.v_of_M is not None and self.smooth_VID:
            for iM in range(self.nM):
                Vprof = self.Vvox*np.sum(counts[iM])
                Lfraction = counts[iM]/np.sum(counts[iM])
                NLsmooth = len(Lsmooth[iM])
                #add the effect of the smoothing
                f_dummy = np.zeros(len(fT),dtype='complex128')
                for jj in range(NLsmooth):
                    f_dummy += Lfraction[jj]*self.calc_fP1_pos(fT,LofM[iM]*Lsmooth[iM][jj])
                pp[iM,:] = Vprof*dndM[iM]*(f_dummy-1)
        else:
            Vprof = self.Vvox*np.sum(counts)
            Lfraction = counts/np.sum(counts)
            NLsmooth = len(Lsmooth)
            for iM in range(self.nM):
                #add the effect of the smoothing
                f_dummy = np.zeros(len(fT),dtype='complex128')
                for jj in range(NLsmooth):
                    f_dummy += Lfraction[jj]*self.calc_fP1_pos(fT,LofM[iM]*Lsmooth[jj])
                pp[iM,:] = Vprof*dndM[iM]*(f_dummy-1)            
        exp_un = np.trapz(pp,self.M,axis=0)
        exp_cl = np.trapz(pp*bM[:,None],self.M,axis=0)**2*self.Pvar/2.
        return exp_un+exp_cl
        
    @cached_vid_property
    def fPT_S(self):
        '''
        Signal-only characteristic function for line intensity.
        
        Unstable at high-k due to clustering term. Includes a control "noise"
        contribution to avoid problems.
        
        NOTE: Often the inverse Fourier transform of this will be unreliable
        for linearly-spaced T bins
        '''
        # ~ #Get nodes, weights, positions in Fourier space and imtervals for intensities
        fT = self.fT_and_edges[0]
        exp_control = -fT**2*self.sigma_PT_stable**2/2.
        if self.subtract_VID_mean:
            exp_shift = 1j*fT*self.Tmean
        else:
            exp_shift = 0.
        #add everything
        return np.exp(self.exp_fPT_signal+exp_control+exp_shift)
        
    @cached_vid_property
    def fPT(self):
        '''
        Full characteristic function of line intensity
        
        In this case we don't use the control "noise", we have the actual noise
        '''
        # ~ #Get nodes, weights, positions in Fourier space and imtervals for intensities
        fT = self.fT_and_edges[0]
        if self.do_Jysr:
            sigmaN = ( self.sigma_N/np.sqrt(self.tpix*self.Nfeeds) ).to(u.Jy/u.sr)
        else: 
            sigmaN = self.sigma_N
        exp_noise = -fT**2*sigmaN**2/2.
        
        if self.subtract_VID_mean:
            exp_shift = 1j*fT*self.Tmean
        else:
            exp_shift = 0.
        #add everything
        return np.exp(self.exp_fPT_signal+exp_noise+exp_shift)
        
    @cached_vid_property
    def PT_S(self):
        '''
        Probability distribution of measuring signal intensity between T and T+dT
        in any given voxel. (includes the control "noise")
        '''
        #Get nodes, weights, positions in Fourier space and intervals for intensities
        nodes,weights = self.leggaus_prep_IFT
        fT, fT_Nind,NlogfT = self.fT_and_edges
        nfT_interval = len(fT_Nind)-1
        T = self.T
        PT = np.zeros(self.nT,dtype='complex128')
        dT = 2*self.Tmax_VID/self.nT
        Npi = int(fT[fT_Nind[NlogfT-1]]*dT/np.pi)
        #create the nufft plan (type1)
        plan = finufft.Plan(1,(self.nT,),isign=1,eps=1e-6)
        #inverse fourier transform computed piecewise-
        for ifT in range(nfT_interval):
            #prepare for the IFT
            print(ifT)
            if ifT == nfT_interval-1:
                fTmax = self.fT_max
            else:
                fTmax = fT[fT_Nind[ifT+1]]
            fTvec = dT*fT[fT_Nind[ifT]:fT_Nind[ifT+1]]
            if ifT >= NlogfT-1:
                fTvec -= 2*Npi*np.pi*(ifT-NlogfT+2)
            fPT_toFT = ((fTmax-fT[fT_Nind[ifT]])/2*weights*self.fPT_S[fT_Nind[ifT]:fT_Nind[ifT+1]])#.astype('complex64')
            #positive fft
            plan.setpts(fTvec)
            PT += plan.execute(fPT_toFT)
            #negative fft
            plan.setpts(-fTvec)
            PT += plan.execute(fPT_toFT.conj())

        #normalize with the FT convention and give units
        PT = PT.real/np.pi/2*self.Tmean.unit**-1
        #check normalization
        nrm = np.trapz(PT,self.T)
        print(PT,self.T)
        print('norm of PT is = ', nrm)
        if abs(nrm-1)>5e-2:
            print('PT not properly normalized.')
        return PT
        
    @cached_vid_property
    def PT(self):
        '''
        Probability distribution of measuring total intensity between T and T+dT
        in any given voxel. (includes the instrumental noise)
        '''
        #Get nodes, weights, positions in Fourier space and intervals for intensities
        nodes,weights = self.leggaus_prep_IFT
        fT, fT_Nind,NlogfT = self.fT_and_edges
        nfT_interval = len(fT_Nind)-1
        T = self.T
        PT = np.zeros(self.nT,dtype='complex128')
        dT = 2*self.Tmax_VID/self.nT
        Npi = self.Npi_fT
        #create the nufft plan (type1)
        plan = finufft.Plan(1,(self.nT,),isign=1,eps=1e-6)
        #inverse fourier transform computed piecewise-
        for ifT in range(nfT_interval):
            #prepare for the IFT
            # print(ifT)
            if ifT == nfT_interval-1:
                fTmax = self.fT_max
            else:
                fTmax = fT[fT_Nind[ifT+1]]
            fTvec = dT*fT[fT_Nind[ifT]:fT_Nind[ifT+1]]
            if ifT >= NlogfT-1:
                fTvec -= 2*Npi*np.pi*(ifT-NlogfT+2)
            fPT_toFT = ((fTmax-fT[fT_Nind[ifT]])/2*weights*self.fPT[fT_Nind[ifT]:fT_Nind[ifT+1]])#.astype('complex64')
            #positive fft
            plan.setpts(fTvec)
            PT += plan.execute(fPT_toFT)
            #negative fft
            plan.setpts(-fTvec)
            PT += plan.execute(fPT_toFT.conj())

        #normalize with the FT convention and give units
        PT = PT.real/np.pi/2*self.Tmean.unit**-1
        #check normalization
        nrm = np.trapz(PT,self.T)
        #print(PT,self.T)
        print('norm of PT is = ', nrm)
        if abs(nrm-1)>5e-2:
            print('PT not properly normalized.')
        return PT
        
    ########################
    # Predicted histograms #
    ########################
                                            
    @cached_vid_property
    def Tedge_i(self):
        '''
        Edges of histogram bins
        '''
        if self.Tmin_VID.value <= 0:
            print('Note that, for negative intensities, the binning has to be linear')
            Te = ulinspace(self.Tmin_VID,self.Tmax_VID,self.Nbin_hist+1)
        else:
            if self.linear_VID_bin:
                Te = ulinspace(self.Tmin_VID,self.Tmax_VID,self.Nbin_hist+1)
            else:
                Te = ulogspace(self.Tmin_VID,self.Tmax_VID,self.Nbin_hist+1)
        
        return Te
        
    @cached_vid_property
    def Ti(self):
        '''
        Centers of histogram bins
        '''
        return binedge_to_binctr(self.Tedge_i)
        
    @cached_vid_property
    def Bi(self):
        '''
        Predicted VID of voxels with a given binned temperature. 
        Normalized by the number of voxels (i.e., sum(Bi)=1)
        '''
        Pi = interp1d(self.T.value,self.PT.value,fill_value=0.,bounds_error=False)
        B = np.zeros(self.Nbin_hist)
        for ii in range(self.Nbin_hist):
            B[ii] = quad(Pi,self.Tedge_i[ii].value,self.Tedge_i[ii+1].value)[0]
        return B
        
    @cached_vid_property
    def Bi_S(self):
        '''
        Predicted VID of voxels with a given binned temperature. 
        Normalized by the number of voxels (i.e., sum(Bi)=1)
        '''
        Pi = interp1d(self.T.value,self.PT_S.value,fill_value=0.,bounds_error=False)
        B = np.zeros(self.Nbin_hist)
        for ii in range(self.Nbin_hist):
            B[ii] = quad(Pi,self.Tedge_i[ii].value,self.Tedge_i[ii+1].value)[0]
        return B
        
    @cached_vid_property
    def Bi_N(self):
        '''
        Predicted VID of voxels with a given binned temperature. 
        Normalized by the number of voxels (i.e., sum(Bi)=1)
        '''
        Pi = interp1d(self.T.value,self.PT_N.value,fill_value=0.,bounds_error=False)
        B = np.zeros(self.Nbin_hist)
        for ii in range(self.Nbin_hist):
            B[ii] = quad(Pi,self.Tedge_i[ii].value,self.Tedge_i[ii+1].value)[0]
        return B
                                       
                                        
    ######################################
    # Draw galaxies to test convolutions #    
    ######################################
    
    def DrawTest(self,Ndraw):
        '''
        Function which draws sample galaxy populations from input number count
        and luminosity distributions.  Outputs Ndraw histograms which can be
        compared to self.Bi
        '''
        h = np.zeros([Ndraw,self.Ti.size])
        
        # PofN must be exactly normalized
        PofN = self.PofN/self.PofN.sum()
        
        for ii in range(0,Ndraw):
            # Draw number of galaxies in each voxel
            N = np.random.choice(self.Ngal,p=PofN,size=self.Nvox.astype(int))
            
            # Draw galaxy luminosities
            Ledge = np.logspace(0,10,10**4+1)*u.Lsun
            Lgal = binedge_to_binctr(Ledge)
            dL = np.diff(Ledge)
            PL = log_interp1d(self.L.value,self.dndL.value)(Lgal.value)*dL
            PL = PL/PL.sum() # Must be exactly normalized
            
            T = np.zeros(self.Nvox.astype(int))*u.uK
            
            for jj in range(0,self.Nvox):
                if N[jj]==0:
                    L = 0.*u.Lsun
                else:
                    L = np.random.choice(Lgal,p=PL,size=N[jj])*u.Lsun
                    
                T[jj] = self.XLT*L.sum()
            
            h[ii,:] = np.histogram(T,bins=self.Tedge_i)[0]
        
        if Ndraw==1:
            # For simplicity of later use, returns a 1-D array if Ndraw=1
            return h[0,:]
        else:
            return h

        
    ########################################################################
    # Method for updating input parameters and resetting cached properties #
    ########################################################################
    def update(self, **new_params):
        # Check if params dict contains valid parameters
        check_invalid_params(new_params,self._default_params)
        
        # If model_type or model_name is updated, check if model_name is valid
        if ('model_type' in new_params) and ('model_name' in new_params):
            check_model(new_params['model_type'],new_params['model_name'])
        elif 'model_type' in new_params:
            check_model(new_params['model_type'],self.model_name)
        elif 'model_name' in new_params:
            check_model(self.model_type,new_params['model_name'])
            
        #if bias_model is updated, check if bias_model is valid
        if 'bias_model' in new_params:
            check_bias_model(new_params['bias_model'])
        
        #if hmf_model is updated, check if hmf_model is valid
        if 'hmf_model' in new_params:
            check_halo_mass_function_model(new_params['hmf_model'])
    
        if 'nT' in new_params:
            if new_params['nT'] % 2:
                print('nT must be even: increasing it by 1 to have it even')
                new_params['nT'] = new_params['nT']+1
    
        #List of observable parameters:
        obs_params = ['Tsys_NEFD','Nfeeds','beam_FWHM','Delta_nu','dnu',
                      'tobs','Omega_field','Nfield','N_FG_par','N_FG_perp',
                      'do_FG_wedge','a_FG','b_FG']
                      
        vid_params = ['Tmin_VID','Tmax_VID','nT','fT_min','fT_max',
                      'Nsigma_prof','Lsmooth_tol',
                      'Nbin_hist','subtract_VID_mean','linear_VID_bin',
                      'T0_Nlogsigma','fT0_min','fT0_max','nfT0',
                      'n_leggauss_nodes_FT','n_leggauss_nodes_IFT',
                      'nfT_interval','nT_interval','sigma_PT_stable']
        
        # Clear cached properties so they can be updated. If only obs changes,
        #   only update cached obs and vid properties. If only vid changes,
        #   update cached vid properties. Otherwise, cached normal properties
        #   always will be updated, and cosmo, only if needed
        if all(item in obs_params for item in new_params.keys()):
            for attribute in self._update_obs_list:
                delattr(self,attribute)
            self._update_obs_list = []
            for attribute in self._update_vid_list:
                delattr(self,attribute)
            self._update_vid_list = []
            #if smooth Pk, needs to be recomputed, since Pk changes
            if self.smooth:
                Pklist = ['Pk','Pk_0','Pk_2','Pk_4']
                for attribute in Pklist:
                    if attribute in self._update_list:
                        delattr(self,attribute)
                        self._update_list.remove(attribute)
        elif all(item in vid_params for item in new_params.keys()):
            for attribute in self._update_vid_list:
                delattr(self,attribute)
            self._update_vid_list = []
        else:
            for attribute in self._update_obs_list:
                delattr(self,attribute)
            self._update_obs_list = []
            for attribute in self._update_vid_list:
                delattr(self,attribute)
            self._update_vid_list = []
            for attribute in self._update_list:
                delattr(self,attribute)
            self._update_list = []

        # Clear cached cosmo properties only if needed, and update camb_pars
        if 'cosmo_input_camb' in new_params and self.cosmo_code == 'camb':
            if len(list(new_params['cosmo_input_camb'].keys())) == 1 and \
               list(new_params['cosmo_input_camb'].keys())[0] == 'f_NL':
                if 'f_NL' in self._update_cosmo_list:
                    delattr(self,'f_NL')
                    self._update_cosmo_list.remove('f_NL')
                    self.cosmo_input_camb['f_NL'] = new_params['cosmo_input_camb']['f_NL']
            else:
                for attribute in self._update_cosmo_list:
                    delattr(self,attribute)
                self._update_cosmo_list = []
                
                for key in new_params['cosmo_input_camb']:
                    self.cosmo_input_camb[key] = new_params['cosmo_input_camb'][key]
                
                self.camb_pars = camb.set_params(H0=self.cosmo_input_camb['H0'], cosmomc_theta=self.cosmo_input_camb['cosmomc_theta'],
                     ombh2=self.cosmo_input_camb['ombh2'], omch2=self.cosmo_input_camb['omch2'], omk=self.cosmo_input_camb['omk'],
                     neutrino_hierarchy=self.cosmo_input_camb['neutrino_hierarchy'], 
                     num_massive_neutrinos=self.cosmo_input_camb['num_massive_neutrinos'],
                     mnu=self.cosmo_input_camb['mnu'], nnu=self.cosmo_input_camb['nnu'], YHe=self.cosmo_input_camb['YHe'], 
                     meffsterile=self.cosmo_input_camb['meffsterile'], 
                     standard_neutrino_neff=self.cosmo_input_camb['standard_neutrino_neff'], 
                     TCMB=self.cosmo_input_camb['TCMB'], tau=self.cosmo_input_camb['tau'], 
                     deltazrei=self.cosmo_input_camb['deltazrei'], 
                     bbn_predictor=self.cosmo_input_camb['bbn_predictor'], 
                     theta_H0_range=self.cosmo_input_camb['theta_H0_range'],
                     w=self.cosmo_input_camb['w'],wa=self.cosmo_input_camb['wa'],cs2=self.cosmo_input_camb['cs2'], 
                     dark_energy_model=self.cosmo_input_camb['dark_energy_model'],
                     As=self.cosmo_input_camb['As'], ns=self.cosmo_input_camb['ns'], 
                     nrun=self.cosmo_input_camb['nrun'], nrunrun=self.cosmo_input_camb['nrunrun'], 
                     r=self.cosmo_input_camb['r'], nt=self.cosmo_input_camb['nt'], ntrun=self.cosmo_input_camb['ntrun'], 
                     pivot_scalar=self.cosmo_input_camb['pivot_scalar'], 
                     pivot_tensor=self.cosmo_input_camb['pivot_tensor'], 
                     parameterization=self.cosmo_input_camb['parameterization'],
                     halofit_version=self.cosmo_input_camb['halofit_version'])
            del new_params['cosmo_input_camb']
                     
        elif 'cosmo_input_class' in new_params and self.cosmo_code == 'class':
            if new_params['cosmo_input_class'].keys() == ['f_NL']:
                if len(list(new_params['cosmo_input_class'].keys())) == 1 and \
                   list(new_params['cosmo_input_class'].keys())[0] == 'f_NL':
                    delattr(self,'f_NL')
                    self._update_cosmo_list.remove('f_NL')
                    self.cosmo_input_class['f_NL'] = new_params['cosmo_input_class']['f_NL']
            else:
                for attribute in self._update_cosmo_list:
                    delattr(self,attribute)
                self._update_cosmo_list = []
                
                self.class_pars = merge_dicts([self.class_pars,new_params['cosmo_input_class']])
                
            del new_params['cosmo_input_class']
                 
        #If the mass range changes, but cosmo_input_camb/class doesn't, sigmaM related functions
        #   at z=0 (otherwise, need to change always when z changes, no cosmo_propery)
        #   need to be updated 
        if ('Mmin' in new_params) or ('Mmax' in new_params):
            if 'sigmaMz0' in self._update_cosmo_list:
                delattr(self,'sigmaMz0')
                self._update_cosmo_list.remove('sigmaMz0')
            if 'dsigmaM_dM_z0' in self._update_cosmo_list:
                delattr(self,'dsigmaM_dM_z0')
                self._update_cosmo_list.remove('dsigmaM_dM_z0')
                 
        #Avoid cosmo_code as an update
        if 'cosmo_code' in new_params:
            print("Please, use a new lim() run if you want to use other Boltzmann code")
            del new_params['cosmo_code']
            
        #update parameters
        for key in new_params:
            setattr(self, key, new_params[key])
                 
            
    #####################################################
    # Method for resetting to original input parameters #
    #####################################################
    def reset(self):
        self.update(**self._input_params)
    
            
############
# Doctests #
############

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)
        
