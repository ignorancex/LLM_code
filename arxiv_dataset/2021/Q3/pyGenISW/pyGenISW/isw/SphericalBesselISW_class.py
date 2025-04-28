import numpy as np
from scipy import integrate, interpolate
import healpy as hp
import subprocess
import TheoryCL

from .. import utils
from .. import bessel


class SphericalBesselISW(TheoryCL.CosmoLinearGrowth):

    """Class for computing the ISW using spherical Bessel Transforms from maps
    of the density contrast given in redshift slices.
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
        self.temp_path = None
        self.sbt_zmin = None
        self.sbt_zmax = None
        self.sbt_zedge_min = None
        self.sbt_zedge_max = None
        self.slice_in_range = None
        self.sbt_rmin = None
        self.sbt_rmax = None
        self.sbt_kmin = None
        self.sbt_kmax = None
        self.sbt_lmax = None
        self.sbt_nmax = None
        self.sbt_redge_min = None
        self.sbt_redge_max = None
        self.uselightcone = None
        self.temp_path = None
        self.boundary_conditions = None
        self.sim_dens = None

    def setup(self, zmin, zmax, zedge_min, zedge_max, kmin=None, kmax=0.1,
              lmax=None, nmax=None, uselightcone=True, temp_path='temp/',
              boundary_conditions='derivative'):
        """Finds the slices that are required to compute the SBT coefficients from.

        Parameters
        ----------
        zmin : float
            Minimum redshift for spherical Bessel transform.
        zmax : float
            Maximum redshift for spherical Bessel transform.
        zedge_min : array
            Minimum redshift edge for each slice.
        zedge_max : array
            Maximum redshift edge for each slice.
        kmin : float
            Minium Fourier mode to consider.
        kmax : float
            Maximum Fourier mode to consider.
        lmax : int
            Maximum l mode to compute to, if None will be computed based on kmax.
        nmax : int
            Maximum n mode to comput to, if None will be computed based on kmax.
        uselightcone : bool
            True if density contrast maps are given as a lightcone and not all at
            redshift 0.
        boundary_conditions : str
            - normal : boundaries where spherical bessel function is zero.
            - derivative : boundaries where the derivative of the spherical Bessel
              function is zero.
        """
        if zedge_min.min() > zmin:
            print('zmin given,', zmin, 'is smaller than the zmin of the redshift slices. Converting zmin to zmin_edges.zmin().')
            self.sbt_zmin = zedge_min.min()
        else:
            self.sbt_zmin = zmin
        if zedge_max.max() < zmax:
            print('zmax given,', zmax, 'is larger than the zmax of the redshift slices. Converting zmax to zmax_edges.zmax().')
            self.sbt_zmax = zedge_max.max()
        else:
            self.sbt_zmax = zmax
        self.sbt_zedge_min = zedge_min
        self.sbt_zedge_max = zedge_max
        self.slice_in_range = np.where((self.sbt_zedge_min <= self.sbt_zmax))[0]
        self.sbt_rmin = TheoryCL.growth.get_r(self.sbt_zmin, self.omega_m, self.omega_l)
        self.sbt_rmax = TheoryCL.growth.get_r(self.sbt_zmax, self.omega_m, self.omega_l)
        self.sbt_kmin = kmin
        self.sbt_kmax = kmax
        if lmax is None:
            self.sbt_lmax = int(self.sbt_rmax*self.sbt_kmax) + 1
        else:
            self.sbt_lmax = lmax
        if nmax is None:
            self.sbt_nmax = int(self.sbt_rmax*self.sbt_kmax/np.pi) + 1
        else:
            self.sbt_nmax = nmax
        self.sbt_redge_min = TheoryCL.growth.get_r(self.sbt_zedge_min, self.omega_m, self.omega_l)
        self.sbt_redge_max = TheoryCL.growth.get_r(self.sbt_zedge_max, self.omega_m, self.omega_l)
        self.uselightcone = uselightcone
        self.temp_path = temp_path
        utils.create_folder(self.temp_path)
        if boundary_conditions == 'normal' or boundary_conditions == 'derivative':
            self.boundary_conditions = boundary_conditions
        else:
            print("boundary_conditions can only be 'normal' or 'derivative', not", boundary_conditions)


    def slice2alm(self, map_slice, index):
        """Given a density contrast map and its corresponding index (for its
        zedges minimum and maximum) slice2alm will convert the map to its
        spherical harmonics and save the files.

        Parameters
        ----------
        map_slice : array
            Healpix density contrast map.
        index : int
            Index of the slice for its zedges.
        """
        if index in self.slice_in_range:
            map_ = map_slice
            wl = hp.sphtfunc.pixwin(hp.get_nside(map_), lmax=self.sbt_lmax)
            alm = hp.map2alm(map_, lmax=self.sbt_lmax, verbose=False)
            alm = hp.almxfl(alm, 1./wl)
            condition = np.where(self.slice_in_range == index)[0]
            np.savetxt(self.temp_path+'map_alm_'+str(condition[0])+'.txt', np.dstack((alm.real, alm.imag))[0])
        else:
            print('Slice not in zmin and zmax range.')

    def alm2sbt(self):
        """Converts spherical harmonic coefficients in redshift slices to spherical
        Bessel coefficients. Stored as delta_lmn in units of (Mpc/h)^(1.5).
        """
        l = np.arange(self.sbt_lmax+1)[2:]
        n = np.arange(self.sbt_nmax+1)[1:]
        l_grid, n_grid = np.meshgrid(l, n, indexing='ij')
        self.l_grid = l_grid
        self.n_grid = n_grid
        qln_grid = np.zeros(np.shape(self.l_grid))
        print('Finding zeros for Bessel function up to n = '+str(self.sbt_nmax))
        for i in range(0, len(self.l_grid)):
            l_val = self.l_grid[i][0]
            if i < 10:
                if self.boundary_conditions == 'normal':
                    qln_grid[i] = bessel.get_qln(l_val, self.sbt_nmax, nstop=100)
                elif self.boundary_conditions == 'derivative':
                    qln_grid[i] = bessel.get_der_qln(l_val, self.sbt_nmax, nstop=100)
            else:
                if self.boundary_conditions == 'normal':
                    qln_grid[i] = bessel.get_qln(l_val, self.sbt_nmax, nstop=100,
                                                 zerolminus1=qln_grid[i-1],
                                                 zerolminus2=qln_grid[i-2])
                elif self.boundary_conditions == 'derivative':
                    qln_grid[i] = bessel.get_der_qln(l_val, self.sbt_nmax, nstop=100,
                                                     zerolminus1=qln_grid[i-1],
                                                     zerolminus2=qln_grid[i-2])
            TheoryCL.utils.progress_bar(i, len(self.l_grid))
        self.kln_grid = qln_grid/self.sbt_rmax
        print('Constructing l and n value grid')
        if self.boundary_conditions == 'normal':
            self.Nln_grid = ((self.sbt_rmax**3.)/2.)*bessel.get_jl(self.kln_grid*self.sbt_rmax, self.l_grid+1)**2.
        elif self.boundary_conditions == 'derivative':
            self.Nln_grid = ((self.sbt_rmax**3.)/2.)*(1. - self.l_grid*(self.l_grid+1.)/((self.kln_grid*self.sbt_rmax)**2.))
            self.Nln_grid *= bessel.get_jl(self.kln_grid*self.sbt_rmax, self.l_grid)**2.
        if self.sbt_kmin is None and self.sbt_kmax is None:
            l_grid_masked = self.l_grid
            n_grid_masked = self.n_grid
            kln_grid_masked = self.kln_grid
            Nln_grid_masked = self.Nln_grid
        else:
            l_grid_masked = []
            n_grid_masked = []
            kln_grid_masked = []
            Nln_grid_masked = []
            for i in range(0, len(self.l_grid)):
                if self.sbt_kmin is None and self.sbt_kmax is None:
                    condition = np.arange(len(self.kln_grid[i]))
                elif self.sbt_kmin is None:
                    condition = np.where(self.kln_grid[i] <= self.sbt_kmax)[0]
                elif self.sbt_kmax is None:
                    condition = np.where(self.kln_grid[i] >= self.sbt_kmin)[0]
                else:
                    condition = np.where((self.kln_grid[i] >= self.sbt_kmin) & (self.kln_grid[i] <= self.sbt_kmax))[0]
                if len(condition) != 0:
                    l_grid_masked.append(self.l_grid[i, condition])
                    n_grid_masked.append(self.n_grid[i, condition])
                    kln_grid_masked.append(self.kln_grid[i, condition])
                    Nln_grid_masked.append(self.Nln_grid[i, condition])
            l_grid_masked = np.array(l_grid_masked, dtype=object)
            n_grid_masked = np.array(n_grid_masked, dtype=object)
            kln_grid_masked = np.array(kln_grid_masked, dtype=object)
            Nln_grid_masked = np.array(Nln_grid_masked, dtype=object)
        self.l_grid_masked = l_grid_masked
        self.n_grid_masked = n_grid_masked
        self.kln_grid_masked = kln_grid_masked
        self.Nln_grid_masked = Nln_grid_masked
        # New part
        print('Pre-compute spherical Bessel integrals')
        _interpolate_jl_int = []
        for i in range(0, len(self.l_grid_masked)):
            _xmin = 0.
            _xmax = (self.kln_grid_masked[i]*self.sbt_rmax).max() + 1.
            _x = np.linspace(_xmin, _xmax, 10000)
            _jl_int = np.zeros(len(_x))
            _jl_int[1:] = integrate.cumtrapz((_x**2.)*bessel.get_jl(_x, l_grid[i][0]), _x)
            _interpolate_jl_int.append(interpolate.interp1d(_x, _jl_int, kind='cubic', bounds_error=False, fill_value=0.))
            TheoryCL.utils.progress_bar(i, len(self.l_grid_masked))
        print('Computing spherical Bessel Transform from spherical harmonics')
        for which_slice in range(0, len(self.slice_in_range)):
            index = self.slice_in_range[which_slice]
            r_eff = (3./4.)*(self.sbt_redge_max[index]**4. - self.sbt_redge_min[index]**4.)/(self.sbt_redge_max[index]**3. - self.sbt_redge_min[index]**3.)
            Dz_eff = self.get_Dr(r_eff)
            Sln = np.zeros(np.shape(self.kln_grid))
            for i in range(0, len(l_grid)):
                if self.sbt_kmin is None and self.sbt_kmax is None:
                    condition = np.arange(len(self.kln_grid[i]))
                elif self.sbt_kmin is None:
                    condition = np.where(self.kln_grid[i] <= self.sbt_kmax)[0]
                elif self.sbt_kmax is None:
                    condition = np.where(self.kln_grid[i] >= self.sbt_kmin)[0]
                else:
                    condition = np.where((self.kln_grid[i] >= self.sbt_kmin) & (self.kln_grid[i] <= self.sbt_kmax))[0]
                if len(condition) != 0:
                    Sln[i, condition] += np.array([(1./(np.sqrt(self.Nln_grid_masked[i][j])*self.kln_grid_masked[i][j]**3.))*(_interpolate_jl_int[i](self.kln_grid_masked[i][j]*self.sbt_redge_max[index]) - _interpolate_jl_int[i](self.kln_grid_masked[i][j]*self.sbt_redge_min[index])) for j in range(0, len(self.l_grid_masked[i]))])
            data = np.loadtxt(self.temp_path + 'map_alm_'+str(which_slice)+'.txt', unpack=True)
            delta_lm_real = data[0]
            delta_lm_imag = data[1]
            delta_lm = delta_lm_real + 1j*delta_lm_imag
            if self.uselightcone == True:
                delta_lm /= Dz_eff
            if which_slice == 0:
                l_map, m_map = hp.Alm.getlm(hp.Alm.getlmax(len(delta_lm)))
                delta_lmn = np.zeros((self.sbt_nmax, len(delta_lm)), dtype='complex')
                conditions1 = []
                conditions2 = []
                for i in range(0, len(Sln[0])):
                    if self.sbt_kmin is None and self.sbt_kmax is None:
                        condition = np.arange(len(self.kln_grid[:, i]))
                    elif self.sbt_kmin is None:
                        condition = np.where(self.kln_grid[:, i] <= self.sbt_kmax)[0]
                    elif self.sbt_kmax is None:
                        condition = np.where(self.kln_grid[:, i] >= self.sbt_kmin)[0]
                    else:
                        condition = np.where((self.kln_grid[:, i] >= self.sbt_kmin) & (self.kln_grid[:, i] <= self.sbt_kmax))[0]
                    if len(condition) == 0:
                        lmax = 0
                    else:
                        lmax = self.l_grid[condition, i].max()
                    condition1 = np.where(self.l_grid[:, i] <= lmax)[0]
                    condition2 = np.where(l_map <= lmax)[0]
                    conditions1.append(condition1)
                    conditions2.append(condition2)
                conditions1 = np.array(conditions1, dtype=object)
                conditions2 = np.array(conditions2, dtype=object)
            for i in range(0, len(Sln[0])):
                _delta_lmn = np.zeros(len(delta_lm), dtype='complex')
                _delta_lmn[conditions2[i].astype('int')] = hp.almxfl(delta_lm[conditions2[i].astype('int')], np.concatenate([np.zeros(2), Sln[conditions1[i].astype('int'), i]]))
                delta_lmn[i] += _delta_lmn
            TheoryCL.utils.progress_bar(which_slice, len(self.slice_in_range), indexing=True, num_refresh=len(self.slice_in_range))
        self.delta_lmn = delta_lmn


    def save_sbt(self, prefix=None):
        """Saves spherical Bessel transform coefficients.

        Parameters
        ----------
        prefix : str
            Prefix for file containing spherical Bessel transform.
        """
        if prefix is None:
            fname = 'sbt_zmin_'+str(self.sbt_zmin)+'_zmax_'+str(self.sbt_zmax)+'_lmax_'+str(self.sbt_lmax)+'_nmax_'+str(self.sbt_nmax)
        else:
            fname = prefix + '_sbt_zmin_'+str(self.sbt_zmin)+'_zmax_'+str(self.sbt_zmax)+'_lmax_'+str(self.sbt_lmax)+'_nmax_'+str(self.sbt_nmax)
        if self.boundary_conditions == 'normal':
            fname += '_normal.npz'
        elif self.boundary_conditions == 'derivative':
            fname += '_derivative.npz'
        np.savez(fname, kln_grid=self.kln_grid, kln_grid_masked=self.kln_grid_masked, l_grid_masked=self.l_grid_masked,
                 Nln_grid_masked=self.Nln_grid_masked, delta_lmn=self.delta_lmn)

    def sbt2isw_alm(self, zmin=None, zmax=None):
        """Returns the ISW spherical harmonics between zmin and zmax from the computed
        spherical Bessel Transform.

        Parameters
        ----------
        zmin : float
            Minimum redshift for ISW computation.
        zmax : float
            Maximum redshift for ISW computation.
        """
        if zmin is None:
            zmin = self.sbt_zmin
        if zmax is None:
            zmax = self.sbt_zmax
        r = np.linspace(self.get_rz(zmin), self.get_rz(zmax), 1000)
        Dz = self.get_Dr(r)
        Hz = self.get_Hr(r)
        fz = self.get_fr(r)
        DHF = Dz*Hz*(1.-fz)
        Iln = np.zeros(np.shape(self.kln_grid))
        for i in range(0, len(self.kln_grid)):
            if self.sbt_kmin is None and self.sbt_kmax is None:
                condition = np.arange(len(self.kln_grid[i]))
            elif self.sbt_kmin is None:
                condition = np.where(self.kln_grid[i] <= self.sbt_kmax)[0]
            elif self.sbt_kmax is None:
                condition = np.where(self.kln_grid[i] >= self.sbt_kmin)[0]
            else:
                condition = np.where((self.kln_grid[i] >= self.sbt_kmin) & (self.kln_grid[i] <= self.sbt_kmax))[0]
            if len(condition) != 0:
                Iln[i, condition] += np.array([(1./np.sqrt(self.Nln_grid_masked[i][j]))*integrate.simps(DHF*bessel.get_jl(self.kln_grid_masked[i][j]*r, self.l_grid_masked[i][j]), r) for j in range(0, len(self.l_grid_masked[i]))])
            TheoryCL.utils.progress_bar(i, len(self.kln_grid))
        alm_isw = np.zeros(len(self.delta_lmn[0]), dtype='complex')
        for i in range(0, len(self.delta_lmn)):
            alm_isw += hp.almxfl(self.delta_lmn[i], np.concatenate([np.zeros(2), Iln[:, i]/(self.kln_grid[:, i]**2.)]))
        alm_isw *= 3.*self.omega_m*((100.*self.h0)**2.)/(self.C**3.)
        alm_isw *= 1e9/(self.h0**3.)
        return alm_isw

    def sbt2isw_map(self, zmin, zmax, nside=256):
        """Returns a healpix map of the ISW between zmin and zmax computed from
        the spherical Bessel Transform.

        Parameters
        ----------
        zmin : float
            Minimum redshift for ISW computation.
        zmax : float
            Maximum redshift for ISW computation.
        nside : int
            Nside for healpix map.
        """
        alm_isw = self.sbt2isw_alm(zmin, zmax)
        map_isw = hp.alm2map(alm_isw, nside)*self.Tcmb
        return map_isw

    def clean_temp(self):
        """Removes temporary spherical harmonic files."""
        if self.slice_in_range is not None:
            for i in range(0, len(self.slice_in_range)):
                subprocess.call('rm -r ' + self.temp_path, shell=True)
