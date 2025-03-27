import numpy as np
from . import power
from . import decorators
from . import peak_statistics

# Planck 18 cosmology
cosm_planck_18 = {'omega_cdm': 0.26067, 'omega_baryon': 0.04897, 'hubble': 0.6766, 'ns': 0.9665, 'tau': 0.0561, 'A_s': 2.105e-09, 'neutrino_mass': 0.0}

# A few examples of how to define different WIMP models.
# However, I have not really tested other cases than Delos & White (2022)
# The main challenge is to correctly estimate the free streaming scale and the phase space density constraint.
# I am using some approximations, by rescaling results, but in principle these may have some complicated dependence
# on the thermal history of the universe.
wimp_100gev_delos_white_2022 = {"type":"wimp_bertschinger", "kfs":1.06e6, "m_gev":100., "tkd_mev":30., "ad":5.332e-12}
wimp_rescaled_1tev = {"type":"wimp_bertschinger_rescaled", "m_gev":1000., "tkd_mev":30., "ad":5.332e-12}
wimp_rescaled_10tev = {"type":"wimp_bertschinger_rescaled", "m_gev":10000., "tkd_mev":30., "ad":5.332e-12}

# The free streaming length scales in Green, Hofmann & Schwartz (2005) are slightly different to the Bertschinger ones
# I use the Bertschinger one above, to be consistent with Delos & White (2022). However, I have no cluse which one
# is actually correct.
wimp_green_hofmann_schwarz = {"type":"wimp_ghs", "m_gev":100., "tkd_mev":30., "ad":5.332e-12}

# In principle, we could also treat WDM here, but so far I have only implemented this partially.
#wdm_bode = {"type":"wdm"}

def wimp_transfer_bertschinger(k, kfs):
    return np.exp(-0.5*(k/kfs)**2)

def transfer_green_hofmann_schwarz(k, m=100., Tkd=30.): # m in GeV Tkd in MeV
    kd  = 3.76e7 * (m/100)**0.5 * (Tkd/30.)**0.5
    kfs = 1.7e6 * (m/100.)**0.5 * (Tkd/30.)** 0.5 / (1. + np.log( (Tkd/30.) )/19.2)
    
    print("kfs: %.3e" % kfs)
    
    return (1. - 2./3. * (k/kfs)**2) * np.exp(-(k/kfs)**2. - (k/kd)**2.)

class CuspDistribution():
    def __init__(self, cachedir=None, zref=30.6, kmatch=5e3, cosmology=cosm_planck_18, dm_model=wimp_100gev_delos_white_2022):
        """This class allows to create a Cusp Distribution.
        
        This by sampling peaks of the primordial density field as described in Bardeen et al (1986),
        (also known as 'BBKS86' (https://ui.adsabs.harvard.edu/abs/1986ApJ...304...15B))
        and assuming that they collapse into prompt cusps, as explained in arXiv:2209.11237 (Delos & White, 2022)       
        
        cachedir : A directory where some results may be cached for speeding up future cllas
        zref : reference redshift for the power spectrum. Usually you don't need to modify this
        kmatch : scale where power spectra are matched between CLASS and small scale model. Usually you don't need to modify this
        cosmology : A dictionary with cosmological parameters. See cusp_distribution.cosm_planck_18 for an example
        dm_model : A dictionary that describes a dark matter model. This is used to infer the free streaming length, 
                   the cut-off of the power spectrum and the primordial phase space density constraint. Look at
                   cusp_distribution.wimp_100gev_delos_white_2022 for an example. Disclaimer: Other cases have not 
                   been thoroughly tested. However, they should be safe if you directly provide kfs and v0 (see code below).
                   
        the main function you will want to use is CuspDistribution.sample_cusps(...)
        """
        
        self.aref = 1./(1.+zref)
        
        self.cachedir = cachedir
        self.k = np.logspace(-6, 10, 1000)
        self.cosmology = cosmology
        self.dm_model = dm_model
                
        def get_base_spectrum(zref, kmatch, cosmology):
            self.myclass = power.get_class(z=zref, kmax=2.*kmatch, cosm=cosmology)
            return power.dimless_cdmpower_all_scales(self.k, self.myclass, z=zref, kmatch=kmatch, A_s=cosmology["A_s"]) / self.aref**2
        
        if self.cachedir is not None:
            get_base_spectrum = decorators.h5cache(get_base_spectrum, file="%s/power_spectrum.hdf5" % self.cachedir)
            
        self.Dk_cdm_base = get_base_spectrum(zref, kmatch, self.cosmology)
        self.Dk_cdm = self.Dk_cdm_base * self.transfer_function(self.k)**2
        
        self.ps = peak_statistics.PeakStatistics(self.k, self.Dk_cdm)
        
    def transfer_function(self, k):
        if "transfer_function" in self.dm_model:
            return self.cutoff_modell["transfer_function"](k)
        
        if self.dm_model["type"] == "wimp_bertschinger":
            self.kfs = self.dm_model["kfs"]
            return wimp_transfer_bertschinger(k, self.kfs)
        elif self.dm_model["type"] == "wimp_bertschinger_rescaled":
            # In principle the Bertschinger model requires me to integrate the thermal history of the universe
            # For simplicity I just rescale the known free streaming scale from Delos & White (2022)
            # since the free streaming scale should be proportional to 1/kfs ~ v0
            # However, I think this neglects a small effect that the time free streaming begins (a_d)
            # may change depending on the decoupling time and temperature.
            # These corrections are small and I'll just neglect it for the sake of simplicity
            kfs = 1.06e6 * (2.76867e-08 / self.v0())
            self.kfs = kfs
            return wimp_transfer_bertschinger(k, kfs)
        elif self.dm_model["type"] == "wimp_ghs":
            return transfer_green_hofmann_schwarz(k, m=self.dm_model["m_gev"], Tkd=self.dm_model["tkd_mev"])
        else:
            raise ValueError("Don't know how to get transfer function for this dm. model %s " % self.dm_model["type"])
    
    def v0(self):
        """Characteristic velocity scale today"""
        c = 299792458.0
        if "v0" in self.dm_model.keys():
            return self.dm_model["v0"]
        if ("ad" in self.dm_model.keys()) & ("tkd_mev" in self.dm_model.keys()) & ("m_gev" in self.dm_model.keys()):
            t, m = self.dm_model["tkd_mev"]*1e6, self.dm_model["m_gev"]*1e9
            return  np.sqrt(t/m)*self.dm_model["ad"]  * c / 1e3
        else:
            raise ValueError("Not implemented v0 for this dm model")
    
    def fmax(self):
        if "fmax" in self.dm_model.keys():
            return self.dm_model["fmax"]
        
        if "wdm" in self.dm_model["type"]:
            from scipy.special import zeta
            C = 1./(zeta(3.)*12.*np.pi) # 0.0221
        elif "wimp" in self.dm_model["type"]:
            C = (2.*np.pi)**(-3./2.) # 0.0635
        else:
            raise ValueError("fmax not implemented v0 for this dm model")
            
        G = 43.0071057317063 * 1e-10
        v0 =  self.v0()

        rho_dm = 3. * (self.cosmology["hubble"] * 100.)**2 / (8.*np.pi*G) * self.cosmology["omega_cdm"]

        fmax = C * v0**-3 * rho_dm

        print("Inferred v0=%.3e, fmax=%.3e using mx=%gGeV, Tkd=%gMeV, ad=%.2e" % (v0, fmax, self.dm_model["m_gev"], self.dm_model["tkd_mev"], self.dm_model["ad"]))
        
        return fmax
        
            

    def sample_peaks(self, N = 100000, seed=None, numin=0.):
        def cached_sample_peaks(N, k, Dk, numin=0., seed=None):
            #peakstatistics = peak_statistics.PeakStatistics(k, Dk)
            
            if seed is not None:
                np.random.seed(seed)

            return self.ps.sample_peaks(N, numin=numin)
            
        if self.cachedir is not None:
            cached_sample_peaks = decorators.h5cache(cached_sample_peaks, file="%s/peaks.hdf5" % self.cachedir)

        return cached_sample_peaks(N, k=self.k, Dk=self.Dk_cdm, numin=numin, seed=seed)
    
    def sample_cusps(self, N = 100000, seed=None, numin=0., units_pc=True, onlyvalid=True):
        """Samples a distribution of cusps
        
        N : number of cusps
        seed : random number generator seed
        numin : only consider initial peaks with  delta > numin*sigma. Probably you don't want to change this from 0
        units_pc : convert all units to parsec. If False will return Mpc units.
        onlyvalid : If True, will only return valid peaks that have collapsed to cusps by a=1. These may be less than the
                    initial sampled peaks. If False it will also include invalid peaks and the number will be exactly N
                    
        returns a dictionary with several entries. I state units for "units_pc = True"
        A : density normalization in Msol/pc**-3/2
        R : Comoving Lagrangian peak radius in pc
        rcusp : physical outer boundary radius of the cusp in pc
        Mcusp : mass of the cusp in Msol
        acoll : collapse scale factor of the cusp
        valid : whether the cusp is valid (if onlyvalid is True, this should always be True)
        J : J-factor = integral(rho**2). This assumes a constant density core as in Delos & White (2022)
               units are Msol**2/pc**3
        rcore : core radius where the phase space density constraint gets violated (pc)
        Bcusp : resiliance of the outer boundary to shocks in km/s/pc
        Bcore : resiliance of the core to shocks in km/s/pc
        """
        nu,x,e,p = self.sample_peaks(N, seed=seed, numin=numin)
        
        G = 43.0071057317063e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
        rho0 = 3.*(100.*self.cosmology["hubble"])**2 / (8.*np.pi*G) * self.cosmology["omega_cdm"] # Msol/Mpc

        fec = np.zeros_like(e)
        valid = (e**2 - p*np.abs(p))  < 0.26
        fec[valid] = ellipsoidal_collapse_correction(e[valid], p[valid], maxiter=40)

        deltaref = nu*self.ps.sigma0 * self.aref  
        # Note that we have normalized everything so that delta(aref) = delta * aref
        # where aref is during matter domination
        g = 0.901
        acoll = (fec * 1.686 / deltaref)**(1./g) * self.aref 
        acoll[acoll == 0.] = np.nan

        delta = nu * self.ps.sigma0
        L = -x * self.ps.sigma2

        R = np.sqrt(np.abs(delta/L))  #/ h

        A = 24. * rho0 * acoll**(-1.5) * R**1.5
        rcusp = 0.11 * acoll * R
        Mcusp = (8.*np.pi/3.) * A * rcusp**1.5

        def J_of_cusp(A, rcusp, rcore):
            return (4.*np.pi / 3.) * A**2 * (1. + 3.*np.log(rcusp/rcore))

        def rcore_of_cusp(A, fmax):
            """Uses units of Msol and Mpc"""
            return (2.89896590246533e-5 * G**-3 * fmax**-2 / A)**(2./9.)
        fmax = self.fmax()
        rcore = rcore_of_cusp(A, fmax)

        J = J_of_cusp(A, rcusp, rcore)
        Bcusp, Bcore = np.sqrt((8. * np.pi * A * G) / (rcusp**(1.5)*3.)), np.sqrt((8. * np.pi * A * G) / (rcore**(1.5)*3.))

        if units_pc:
            res = dict(A=A/1e9, R=R*1e6, rcusp=rcusp*1e6, Mcusp=Mcusp, acoll=acoll, valid=valid, J=J/1e18, rcore=rcore*1e6, Bcusp=Bcusp/1e6, Bcore=Bcore/1e6)
        else:
            res = dict(A=A, R=R, rcusp=rcusp, Mcusp=Mcusp, acoll=acoll, valid=valid, J=J, rcore=rcore, Bcusp=Bcusp, Bcore=Bcore)
            
        H0 = 100.*self.cosmology["hubble"]
        Gmpc = 43.0071057317063 * 1e-10
        rho0 = 3.*H0**2 / (8.*np.pi*Gmpc) * (self.cosmology["omega_cdm"])
        
        if numin == 0.:
            mdm = rho0 * (N / self.ps.ntot_positive)
        else:
            print("Warning, didn't think about numin != 0 for global statistics")
            mdm = np.nan

        if onlyvalid:
            for key in res:
                res[key] = res[key][valid]

        res["mdm_tot"] = mdm # total Lag. dark matter volume = mass represented by the cusps
        res["mdm_per_cusp"] = res["mdm_tot"] / len(res["Mcusp"])
        res["fmax"] = fmax

        return res

def omega_rad(a, Tcmb = 2.7255, h=0.678, geff_ur=3.044):
    c = 299792458.0
    sig = 5.670374419e-8 # Stefan boltzmann constnat

    rho_r = 4./c * sig * Tcmb**4

    H0 = 100. * h
    rho_crit = 3.*(H0*1e3 / (3.085677581491367e16*1e6))**2 / (8.*np.pi*6.6743e-11)

    omega_photon = (rho_r)/(rho_crit*c**2)
    
    Tnu = Tcmb * (4./11.)**(1./3.)
    
    omega_nu = 7./22. * geff_ur * (4./11.)**(1./3.) * omega_photon
    
    return (omega_nu + omega_photon) * a**-4
    
    #return (geff_ur/2.) * omega_photon * a**-4 

def hubble(a, h=0.678, omega_m=0.3, omega_l=0.7, Tcmb=2.7255, geff_ur=3.044):
    return 100.*h * np.sqrt(omega_m*a**-3 + omega_l + omega_rad(a, h=h, Tcmb=Tcmb, geff_ur=geff_ur))


def free_streaming_length(mx_gev=100., td_mev=30., ad=5.332e-12, amax=1., h=0.678, omega_m=0.3, omega_l=0.7, cumulative=False, Tcmb=2.7255, geff_ur=3.044):
    """Infer the free-streaming length by integration as in https://arxiv.org/pdf/astro-ph/0607319.pdf eqn.48
    
    However, right now this function is not used. Instead we rescale the free streaming length infered by Delos & White(2022)
    """
    def f(a):
        Hl = hubble(a, omega_m=omega_m, omega_l=omega_l, Tcmb=Tcmb, geff_ur=geff_ur, h=h) / 299792458.0e-3 # Hubble in 1/Mpc

        return 1./(Hl *a**3)
    
    mx_ev, Td_ev = mx_gev*1e9, td_mev*1e6
    
    # At early times the conformal time is propotional to a
    # therefure a/a_d = tau/tau_d
    astart = ad*1.05  

    a = np.logspace(np.log10(astart),np.log10(amax), 100000)
    
    if cumulative:
        integral = peak_statistics.cum_simpson(f, a)
    
        return a, ad * np.sqrt(6.*Td_ev/5./mx_ev) * integral
    else:
        from scipy.integrate import simps
        integral = simps(f(a), a)
    
        return ad * np.sqrt(6.*Td_ev/5./mx_ev) * integral

def ellipsoidal_collapse_correction(e, p, maxiter=20):
    def myfunc(f, e, p):
        return 1+0.47*(5*(e**2-p*np.abs(p))*f**2)**0.615
    
    fnew = np.ones_like(e) * 1.1
    for i in range(0,20):
        fnew = myfunc(fnew, e, p)
        
    return fnew


def get_v0(T, m, ad):
    c = 299792458.0
    return  np.sqrt(T * m)*ad / m * c / 1e3

def fmax_wimp(h=0.68, omega_dm=0.26,  mx=100, Td=30., ad=5.332e-12):
    """Following https://arxiv.org/pdf/2207.05082.pdf

    mx : WIMP mass in GeV
    Td : Decoupling Temperature in MeV
    omgega_dm: dark matter (not full matter) density paramater, mx in kev
    ad : scale factor of decoupling (where the temperature of the universe is Td)
         This can be put to None to use an approximation by the neutrino temperature
         which may have errors of order 10% if the wimp decoupled a bit before the
         neutrinos

    the phase space density of WIMP's
    """
    mev, Tdev = mx*1e9, Td*1e6
    
    c = 299792458.0
    G = 43.0071057317063 * 1e-10

    if ad is None:
        print("Approximating ad by assuming evaluating T(a_d)=Td while using the temperature T(a) of the Neutrino background.\n"
              "This may give inaccurate results by 10-20%. For full accuracy use a full thermal history and determine ad")
        Tcmb = 2.725 #K
        kb = 8.617333262e-5 # eV/kelvin
        Tnu = Tcmb*(4./11.)**(1./3.) * kb   # in eV
        ad = (Tnu/Tdev)

    v0 =  np.sqrt(Tdev * mev)*ad / mev * c / 1e3  # velocity today in km/s

    rho_dm = 3. * (h * 100.)**2 / (8.*np.pi*G) * omega_dm

    return (2.*np.pi)**(-3./2.) * v0**-3 * rho_dm