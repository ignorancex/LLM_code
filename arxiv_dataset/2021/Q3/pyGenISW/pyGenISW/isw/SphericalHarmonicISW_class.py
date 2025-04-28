import numpy as np
import healpy as hp
from scipy import integrate
import TheoryCL


class SphericalHarmonicISW(TheoryCL.CosmoLinearGrowth):

    """Class for computing the ISW approximately using the spherical harmonics of
    a map of the density contrast.
    """

    def __init__(self):
        """Initialises the class.

        Parameters
        ----------
        CosmoLinearGrowth : class
            Parent class for calculating Cosmological linear growth functions.
        """
        TheoryCL.CosmoLinearGrowth.__init__(self)
        self.Tcmb = 2.7255
        self.C = 3e8


    def Spherical_Harmonic_Scaling(self, alm_den, cl_td, cl_dd):
        """Returns the ISW alms from the alms of the density contrast of a single slice using
        Wiener filtering.

        Parameters
        ----------
        alm_den : array
            Alms of the density contrast.
        cl_td : array
            Cross correlation cls for the ISW and density contrast. Given from l=2.
        cl_dd : array
            Auto correlation cls for the density contrast. Given from l=2.

        Returns
        -------
        alm_t : array
            Alms for the ISW.

        Notes
        -----

        For the time being you will need to enter precalculated Cls. Later on this will
        be expanded so that this can calculated by the class itself.
        """
        fl = cl_td/cl_dd
        fl = np.concatenate([np.zeros(2), fl])
        alm_t = hp.almxfl(alm_den, fl)
        return alm_t


    def Francis_Peacock(self, alm_den, zmin, zmax, mode=False):
        """Returns the ISW alms from the alms of the density contrast of a single slice using
        the Francis & Peacock (2009) approximation (see https://arxiv.org/abs/0909.2494).

        Parameters
        ----------
        alm_den : array
            Alms of the density contrast.
        zmin : array
            Minimum redshift for the slice.
        zmax : array
            Maximum redshift for the slice.
        mode : int
            1 -- uses the approximation as given in Francis & Peacock (2009),
            2 -- uses (r_max^3 - r_min^3)/3 instead of delta_r*r_eff^2.
            3 -- integrates H*(1-f)*r^2

        Returns
        -------
        alm_t : array
            Alms for the ISW.
        """
        rmin = self.get_rz(zmin)
        rmax = self.get_rz(zmax)
        if mode == 1:
            reff = (3./4.)*(rmax**4. - rmin**4.)/(rmax**3. - rmin**3.)
            zeff = self.get_zr(reff)
            delta_r = rmax - rmin
            Heff = self.get_Hz(zeff)
            feff = self.get_fz(zeff)
            fl = 3.*((1e2*self.h0)**2.)*self.omega_m*Heff*(1.-feff)
            fl *= delta_r * reff**2.
        elif mode == 2:
            reff = (3./4.)*(rmax**4. - rmin**4.)/(rmax**3. - rmin**3.)
            zeff = self.get_zr(reff)
            Heff = self.get_Hz(zeff)
            feff = self.get_fz(zeff)
            fl = 3.*((1e2*self.h0)**2.)*self.omega_m*Heff*(1.-feff)
            fl *= (rmax**3. - rmin**3.)/3.
        elif mode == 3:
            r = np.linspace(rmin, rmax, 1000)
            fl = 3.*((1e2*self.h0)**2.)*self.omega_m*integrate.simps(self.get_Hr(r)*(1.-self.get_fr(r))*r**2., r)
        lmax = hp.Alm.getlmax(len(alm_den))
        l = np.arange(lmax+1).astype('float')[2:]
        fl /= l*(l+1)*self.C**3.
        # units
        fl *= 1e9/(self.h0**3.)
        fl = np.concatenate([np.zeros(2), fl])
        alm_t = hp.almxfl(alm_den, fl)
        return alm_t
