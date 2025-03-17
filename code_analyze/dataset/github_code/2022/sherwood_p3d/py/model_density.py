import numpy as np
import os
import sherwood_simulation as she_sim

def f_of_z(z):
    """Get the dimensionless linear growth rate for a snapshot redshift."""
    growth_rates = {2.0: 0.95676691, 2.4: 0.96949307, 2.8: 0.97763031,
        3.2: 0.98303738}
        
    return growth_rates[z]


class PowerInterpolator(object):
    """Stores power at one redshift, and interpolates."""

    def __init__(self,fname):
        """Read file containing power at a given redshift."""

        # read data from file (in units of Mpc/h)
        data=np.loadtxt(fname)
        k=data[:,0]
        P=data[:,1]

        # we will do the interpolation in log(k), log(P)
        self.lnk=np.log(k)
        self.lnP=np.log(P)

    def P_hMpc(self,k_hMpc):
        """Interpolate power to input wavenumber k_hMpc"""

        # interpolator works in ln(k)
        lnk = np.log(k_hMpc)
        lnP = np.interp(lnk,self.lnk,self.lnP)
        return np.exp(lnP)


class LinearDensityModel(object):
    """Object describing the linear density power spectrum at all redshifts."""

    def __init__(self):
        """Set up model to describe the linear power spectrum."""

        # make sure the environmental variable is set
        assert ('SHERWOOD' in os.environ),'Define SHERWOOD'
        basedir=os.environ['SHERWOOD']

        # base directory with linear power files
        basedir+='/data/linear_pk/'

        # dictionary containing linear power for all snapshots
        self.linP={}

        for snap in range(8,12):
            fname=basedir+'/lin_{}.dat'.format(snap)
            self.linP[snap]=PowerInterpolator(fname)


    def linP_hMpc(self,z,k_hMpc):
        """ Compute linear power at input redshift and wavenumber (in h/Mpc).
            - z: input redshift (must correspond to one of the files read)
            - k_hMpc: input wavenumber or (array) in h/Mpc. """

        # make sure input redshift is in the dictionary
        snap=she_sim.snapshot_from_redshift(z)
        assert snap in self.linP,'input snapshot not in list '+str(snap)

        # interpolate linear power to input wavenumber (in Mpc/h)
        return self.linP[snap].P_hMpc(k_hMpc)

