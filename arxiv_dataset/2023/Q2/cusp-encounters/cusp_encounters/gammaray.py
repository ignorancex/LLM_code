import numpy as np
import sys
import os
from scipy.integrate import simps
import scipy.constants as const

# This is a helper file to handle the I/O of observational data sets
# and to integrate spectral data.
# Mostly this depends upon the Isotropic Gamma Ray Background (IGRB)
# as measured in Ackerman et al. (2015) arXiv:1410.3696, the spectral
# approximation of the Galactic Centre Excess (GCE) of Di Mauro (2021)
# arXiv:2101.04694 and on some formula in Delos & White (2022) 
# arXiv:2209.11237
# We have to download the data table of Ackermann et al. (2015)
# to make everything work out

if not os.path.exists("../data/apj504089t3_mrt.txt"):
    print("""If you want to use all aspects of gammaray module, you need to download
             the isotropic gamma ray background table (Ackermann et al, 2015) from
             https://iopscience.iop.org/article/10.1088/0004-637X/799/1/86#apj504089t3
             (scrolling a little bit down and "Download table as: Data") and place it 
             under the name "data/apj504089t3_mrt.txt"
          """)

data_ackermann = None

def load_ackermann_2015():
    """loads the IGRB table from Ackermann et al (2015)"""
    global data_ackermann, ackermann_2015_loaded, ackermann_2015_error
    data_ackermann = np.loadtxt("../data/apj504089t3_mrt.txt", skiprows=61, dtype="str")
    
def igrb_dI_dloge(E_mev, mode="A"):
    """returns interpolated (and transformed) rows of the IGRB table"""
    if data_ackermann is None:
        load_ackermann_2015()
    
    sel = data_ackermann[:,0] == mode
    data = np.float64(data_ackermann[sel, 1:])
    
    Erange, y = data[:,0:2], data[:, 2:]
    Ecent = np.sqrt(Erange[:,0]*Erange[:,1])
    
    dn_dlogE = y / (np.log(Erange[:,1]) - np.log(Erange[:,0]))[:,np.newaxis]
    dphi_dlogE = dn_dlogE * Ecent[:,np.newaxis]
    
    return np.stack([np.interp(np.log(E_mev), np.log(Ecent), dphi_dlogE[:,col]) for col in range(0,y.shape[1])], axis=-1)

def igrb_fit_dI_dloge(E, I100=0.95e-7, gamma=2.32, Ecut=279e3):
    """the spectral fit to the IGRB data presented in Ackermann et al. (2015)"""
    dnde = I100 * (E / 100.)**-gamma * np.exp(-E/Ecut)    
    return dnde*E**2

def igrb_integral(Emin_mev=1e3, Emax_mev=1e4):
    """Integrates the (energy weighted) IGRB table in some energy range
    
    output units are in MeV / cm**2 s sr"""
    try:
        logE = np.linspace(np.log(Emin_mev), np.log(Emax_mev), 10000)
    
        dIdlogE = igrb_dI_dloge(np.exp(logE))
        return simps(y=dIdlogE, x=logE, axis=0)
    except:
        assert (Emin_mev == 1e3) & (Emax_mev == 1e4)
        print("Warning: something went wrong loading the IGRB table, using hard coded results")
        return np.array([6.94262388e-04, 9.01401512e-05, 9.01667939e-05, 3.08368725e-04, 2.81933621e-04, 1.20284736e-03, 1.05113588e-04, 1.05147831e-04, 3.14153601e-04, 3.59118605e-04, 5.04664807e-04, 5.32264490e-05, 5.32264490e-05])
    
    
def extragalactic_dI_dloge(E_GeV, cusp_dis, cusps):
    """Predicts the spectrum of the extragalactic IGRB due to prompt cusps 
    see arxiv:2209.11237 eqn. (16)
    
    output units are in MeV / cm**2 s sr
    """
    
    def Ngce(E_GeV):
        return 8.6e14 * (E_GeV)**(0.27 - 0.27*np.log(E_GeV)) # out units Mev/s
    
    c = cusp_dis.cosmology
    
    H0 = 100.*c["hubble"]
    omega_m = c["omega_cdm"]+c["omega_baryon"]
    def H(z, omega_m=omega_m, omega_l=1.-omega_m):
        a = 1./(1.+z)
        return H0 * np.sqrt(omega_m*a**-3 + omega_l)
    def integrand(E,z):
        Ep = E*(1.+z)
        return Ngce(Ep)/Ep**2 / H(z)
    
    z = np.linspace(0, 30., 500)
    
    G = 43.0071057317063e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
    rho0 = 3.*H0**2 / (8.*np.pi*G) * c["omega_cdm"]
    Jcusps_ov_M = (np.sum(cusps["J"])/cusps["mdm_tot"]) * 1e18 # Msol Mpc**-3

    Integral = simps(y=integrand(E_GeV[:,np.newaxis], z), x=z, axis=1)

    return E_GeV**2 * (const.c/1e3) / (4.*np.pi) * rho0 * Jcusps_ov_M * Integral / ((100.*1e6*const.parsec)**2)

def extragalactic_integral(cusp_dis, cusps, Emin_mev=1e3, Emax_mev=1e4):
    """Predicts the integrated amplitude of the extragalactic IGRB due to 
    prompt cusps
    
    output units are in MeV / cm**2 s sr
    """
    logE = np.linspace(np.log(Emin_mev), np.log(Emax_mev), 10000)
    dIdlogE = extragalactic_dI_dloge(np.exp(logE)/1000., cusp_dis, cusps)
    return simps(y=dIdlogE, x=logE, axis=0)

def intragalactic_dI_dloge(E_GeV, Jalphatot):
    """Given a J-factor column density, predicts the surface brightness 
    spectrum under the assumption that the surface brightness of the GCE 
    is explained by the dark matter annihilation signal from the smooth halo
    
    E_GeV : Energy the spectrum should be evaluated in GeV
    Jalphatot : J-column density = line of sight integral of rho**2.
                Units have to be in Msol**2/pc**5
    
    output units are in MeV / cm**2 s sr
    """
    def Ngce(E_GeV):
        return 8.6e14 * (E_GeV)**(0.27 - 0.27*np.log(E_GeV)) # out units Mev/s
    
    return 1./(4.*np.pi) * Ngce(E_GeV) * (Jalphatot* 1e18)  / (100.*const.parsec)**2

def intragalactic_integral(Jalphatot=1., Emin_mev=1e3, Emax_mev=1e4):
    """Given a J-factor column density, predicts the surface brightness 
    under the assumption that the surface brightness of the GCE 
    is explained by the dark matter annihilation signal from the smooth halo
    
    Jalphatot : J-column density = line of sight integral of rho**2.
                Units have to be in Msol**2/pc**5
    
    output units are in MeV / cm**2 s sr
    """
    
    logE = np.linspace(np.log(Emin_mev), np.log(Emax_mev), 10000)
    normfac = simps(y=intragalactic_dI_dloge(np.exp(logE)/1000., 1.), x=logE, axis=0)
        
    return normfac*Jalphatot
