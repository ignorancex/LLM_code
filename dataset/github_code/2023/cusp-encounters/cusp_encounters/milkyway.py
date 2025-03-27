import numpy as np
import sys
import os

sys.path.append("../../adiabatic-tides")

import adiabatic_tides as at

def RvirOfMvir(mvir, delta=200., h=0.678, omega_m=0.308, omega_l=0.692, a=1.):
    G = 43.0071057317063 * 1e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2 * (omega_m*a**-3 + omega_l)
    
    return  np.cbrt(mvir / (rhocrit * 4.*np.pi/3. * delta)) * 1e6  # in pc

def potential_miyamoto_nagai(x, M=2e10, b=300e-6, a=3000e-6, G=43.0071057317063e-4):
    """Miyamoto Nagai (1975) potential (https://articles.adsabs.harvard.edu/pdf/1975PASJ...27..533M)
    see also http://astro.utoronto.ca/~bovy/AST1420/notes/notebooks/07.-Flattened-Mass-Distributions.html
    """
    # assuming the disk lies in the x-y plane
    R = np.sqrt(np.square(x[...,0]) + np.square(x[...,1]))
    z = x[...,2]
    
    return - G*M / np.sqrt(np.clip(R**2 + (np.sqrt(z**2 + b**2) + a)**2, 1e-20, None))

def miyamoto_nagai_parameters_of_disk(M, Rd, b):
    # See arxiv:1502.00627
    coeffs = [{}, {}, {}]
    coeffs[0]["M_over_Md"] = -0.0090, 0.0640, -0.1653, 0.1164, 1.9487
    coeffs[1]["M_over_Md"] =  0.0173,-0.0903,  0.0877, 0.2029,-1.3077
    coeffs[2]["M_over_Md"] = -0.0051, 0.0287, -0.0361,-0.0544, 0.2242
    coeffs[0]["a_over_Rd"] = -0.0358, 0.2610, -0.6987,-0.1193, 2.0074
    coeffs[1]["a_over_Rd"] = -0.0830, 0.4992, -0.7967,-1.2966, 4.4441
    coeffs[2]["a_over_Rd"] = -0.0247, 0.1718, -0.4124,-0.5944, 0.7333
    
    pars = [{},{},{}]
    for i in range(0,3):
        x = b / np.clip(Rd, 1e-20, None)
        xn = np.array(x)[..., np.newaxis]**np.array([4,3,2,1,0])
        pars[i]["M"] = np.sum(xn*coeffs[i]["M_over_Md"], axis=-1) * M
        pars[i]["a"] = np.sum(xn*coeffs[i]["a_over_Rd"], axis=-1) * Rd

    return pars

def disk_potential(x, M, Rd, b, G=43.0071057317063e-4):
    # Approximation of Disk through three miyamato nagai potentials (arxiv:1502.00627)
    pars = miyamoto_nagai_parameters_of_disk(M, Rd, b)
    
    pot = 0.
    for i in range(0,3):
        pot += potential_miyamoto_nagai(x=x, M=pars[i]["M"], b=b, a=pars[i]["a"], G=G)
    return pot

def finite_differences_n(x, f, deriv=1, h=1e-5, **kwargs):
    """Calculate the nth order derivatives of a function
    
    x : locations where to evaluate the derivatives shape (..., ndim)
    f : function f(x, **kwargs)
    deriv : derivative, if > 1 this function will be called recursively
    h : step-size that will be used for finite differences
    **kwargs: will be passed through to the function
    
    returns : n-th derivative tensor, e.g. for deriv=2 shape = (..., ndim, ndim) 
    """
    if deriv==0:
        return f(x, **kwargs)
    
    grad = np.zeros(np.shape(f(x, **kwargs)) + (3,)*deriv)
    
    dim = x.shape[-1]
    eij = np.diag(np.ones(dim))
    
    for ax in range(0, dim):
        xp = x + h * eij[ax]
        xm = x - h * eij[ax]
        
        fp = finite_differences_n(xp, f, deriv=deriv-1, h=h, **kwargs)
        fm = finite_differences_n(xm, f, deriv=deriv-1, h=h, **kwargs)
        
        grad[...,ax] = (fp - fm) / (2*h)

    return grad

def gather_array_dict(datain, comm, root=0, axis=0, niter=1):
    """This is an mpi helper function which gathers a dictionary of numpy arrays to a root process"""
    data = {}
    for key in datain:
        data[key] = []
        for i in range(0, niter):
            datasend = datain[key][i::niter]
            if comm.Get_rank() == root:
                data[key] = data[key] + list(comm.gather(datasend, root=root))
            else:
                comm.gather(datasend, root=root)
        if comm.Get_rank() == root:
            data[key] = np.stack(data[key], axis=axis)
    if comm.Get_rank() == root:
        return data
    else:
        return None

class MilkyWay():
    def __init__(self, h=0.679, adiabatic_contraction=True, cachedir=None, mass_halo=1e12, conc_initial=8.71, mode="cautun_2020"):
        """
        A mass and potential model of the Milky Way. It uses several baryonic components plus a contracted dark matter halo
        
        cachedir : pass a directory where some cachefiles may be created to speed up future calls
        
        The main function you will want to use is MilkyWay.create_dm_orbits(...)
        
        the variable "mode" changes the way the adiabatic contraction is calculated
        It can be "cautun_2020" (arXiv:1911.04557) or "sellwood_mcgaugh_2005" (arxiv:0507589).
        It turned out that cautun works better for getting a halo that is consistent with
        the DM self-annihilation signal.
        """


        
        self.h = h
        self.G = G = 43.0071057317063 * 1e-4  #  Grav. constant in pc (km/s)^2 / Msol
        self.cachedir = cachedir
        self.mode = mode
        
        self.par = {}
        self.par["halo"] = dict(mass = mass_halo, conc=conc_initial)
        self.par["stardisk"] = dict(mass = 4.100e+10, scalelength = 2.500e+03, thickness = 3.500e+02)
        self.par["gasdisk"]  = dict(mass = 1.9e10, scalelength=7e3, thickness = 8e1)
        self.par["bulge"]  = dict(mass = 9e9, scalelength=5e2)
        
        self.profile_nfw = at.profiles.NFWProfile(conc=self.par["halo"]["conc"], m200c=self.par["halo"]["mass"])
        
        self.adiabatic_contraction = adiabatic_contraction
        if adiabatic_contraction:
            # Create a spherically averaged mean baryon profile
            # Using the MonteCarloProfile is a hack to use the
            # already implemented potential calculation
            rbins = np.logspace(-6, 1, 1000) * 200e3 
            rcent = np.sqrt(rbins[1:]*rbins[:-1])
            mcent = self.enclosed_mass(rbins[1:], components="sbg") - self.enclosed_mass(rbins[:-1], components="sbg")
            self.profile_baryon_mean = at.profiles.MonteCarloProfile(rcent/1e6, mcent, rbins=rbins/1e6)
            
            if mode == "cautun_2020":
                #arXiv:1911.04557 eqn (11)
                fbar = 0.1878 # Planck18
                etabar = self.profile_baryon_mean.m_of_r(rbins/1e6) / (fbar*self.profile_nfw.m_of_r(rbins/1e6))
                #print(etabar)
                M_dm = self.profile_nfw.m_of_r(rbins/1e6) * (0.45 + 0.38*(etabar + 1.16)**0.53)
                rho_dm = (M_dm[1:] - M_dm[:-1]) / (4.*np.pi/3.*(rbins[1:]**3. - rbins[:-1]**3))
                #print(rho_dm)
                
                profile_contracted_nfw = at.profiles.NumericalProfile(0.5*(rbins[1:] + rbins[:-1])/1e6, rho_dm*1e18)
                
                self.potential_profile = at.profiles.CompositeProfile(profile_contracted_nfw, self.profile_baryon_mean)
                
                self.profile_contracted_nfw = at.profiles.NumericalProfile(0.5*(rbins[1:] + rbins[:-1])/1e6, rho_dm*1e18, potential_profile=self.potential_profile)
                
            elif mode == "sellwood_mcgaugh_2005":
                
                if self.cachedir is None:
                    h5cache = None
                else:
                    h5cache = "%s/contracted_nfw.hdf5" % self.cachedir

                self.profile_contracted_nfw = at.profiles.AdiabaticProfile(prof_initial=self.profile_nfw, 
                                         prof_pert=self.profile_baryon_mean, 
                                         niter=20,
                                         verbose=True,
                                         h5cache = h5cache)
            else:
                raise ValueError("Unknown mode for the adiabatic contraction: %s" % mode)
            self.profile_halo = self.profile_contracted_nfw
        else:
            self.profile_halo = self.profile_nfw
        
    def potential_halo(self, x):
        return self.profile_halo.self_potential(np.linalg.norm(x, axis=-1)/1e6)
    
    def potential_bulge(self, x):
        M, a = self.par["bulge"]["mass"], self.par["bulge"]["scalelength"]

        r = np.linalg.norm(x, axis=-1)
        
        return - self.G * M / (r + a)
    
    def potential_disk(self, x,  mode="stardisk"):
        assert mode in ("stardisk", "gasdisk")
        
        M, Rd, b = self.par[mode]["mass"], self.par[mode]["scalelength"], self.par[mode]["thickness"]

        return disk_potential(x, M=M, Rd=Rd, b=b, G=self.G)
    
    def disk_surface_density(self, r, mode="stardisk"):
        assert mode in ("stardisk", "gasdisk")
        
        M, Rd, b = self.par[mode]["mass"], self.par[mode]["scalelength"], self.par[mode]["thickness"]
        
        Sig = M / (2.*np.pi*Rd**2)
        return Sig * np.exp(-r/Rd)
    
    def enclosed_mass_disk(self, r,  mode="stardisk"):
        assert mode in ("stardisk", "gasdisk")
        
        M, Rd, b = self.par[mode]["mass"], self.par[mode]["scalelength"], self.par[mode]["thickness"]

        return M * (1. - (r + Rd) / Rd * np.exp(- r / Rd) )
    
    def enclosed_mass_bulge(self, r):
        M, a = self.par["bulge"]["mass"], self.par["bulge"]["scalelength"]

        return M * r**2 / (r + a)**2
    
    def enclosed_mass_halo(self, r):
        return self.profile_halo.self_m_of_r(r/1e6)
    
    def enclosed_mass(self, r, components="hbsg"):
        mass = 0.
        if "h" in components:
            mass += self.enclosed_mass_halo(r)
        if "b" in components:
            mass += self.enclosed_mass_bulge(r)
        if "s" in components:
            mass += self.enclosed_mass_disk(r, mode="stardisk")
        if "g" in components:
            mass += self.enclosed_mass_disk(r, mode="gasdisk")
        return mass
    
    def enclosed_mass_derivative(self, r, components="hbsg", deriv=1, eps=1e-3):
        def M(r):
            return self.enclosed_mass(r, components=components)
        
        dx = r*eps
        return (M(r+0.5*dx) - M(r-0.5*dx)) / dx
    
    def radial_mean_tide(self, r, components="hbsg"):
        M, dMdr = self.enclosed_mass(r, components=components), self.enclosed_mass_derivative(r, components=components)
        
        return self.G * (2.*M/r**3 - dMdr/r**2)

    def potential(self, x, components="hbsg"):
        potential = 0.
        if "h" in components:
            potential += self.potential_halo(x)
        if "b" in components:
            potential += self.potential_bulge(x)
        if "s" in components:
            potential += self.potential_disk(x, mode="stardisk")
        if "g" in components:
            potential += self.potential_disk(x, mode="gasdisk")
        return potential
    
    def acceleration(self, x, components="hbsg", dx=0.1):
        def pot(x):
            return self.potential(x, components=components)
        
        return -finite_differences_n(x, pot, deriv=1, h=dx)
    
    def tidal_tensor(self, x, components="hbsg", dx=0.1):
        def pot(x):
            return self.potential(x, components=components)
        
        return -finite_differences_n(x, pot, deriv=2, h=dx)
    
    def density(self, x, components="hbsg", dx=0.1):
        T = self.tidal_tensor(x, components=components, dx=dx)
        
        return - np.trace(T, axis1=-1, axis2=-2) / (4.*np.pi*self.G)
    
    def integrate_orbit(self, x0, v0, ti, components="hbsg"):
        vfac = 978461.94   # year / (parsec / km/s)
        
        lastacc = 0.
        
        xi = np.zeros(ti.shape + x0.shape)
        vi = np.zeros(ti.shape + v0.shape)
        
        xi[0], vi[0] = x0, v0

        for i in range(1, len(ti)):
            dt = (ti[i] - ti[i-1])
            
            # kick
            if i == 1:
                acc = self.acceleration(xi[i-1], components=components)
            else:
                acc = lastacc
            vip = vi[i-1] + 0.5 * dt * acc / vfac
            
            # drift
            xi[i] = xi[i-1] + vip * dt / vfac
            
            # kick
            acc = self.acceleration(xi[i], components=components)

            vi[i] = vip + 0.5 * dt * acc / vfac
            
            lastacc = acc
        
        return xi, vi
    
    def integrate_orbit_with_info(self, x0, v0, ti, components="hbsg", subsamp=10, with_info=False, verbose=False):
        vfac = 978461.94   # year / (parsec / km/s)
        
        res = {}
        for key in "pos", "vel", "acc":
            res[key] = []
        
        x, v = np.copy(x0), np.copy(v0)
        lastacc = np.zeros_like(x)
        
        if with_info:
            last_dens_star, last_dens_dm, column_dens_star  = np.zeros(x0.shape[:-1]),np.zeros(x0.shape[:-1]),np.zeros(x0.shape[:-1])
            chi_dm, chi_star, nperi  = np.zeros(x0.shape[:-1]),np.zeros(x0.shape[:-1]),np.zeros(x0.shape[:-1])
            r2_m2, r2_m1  = np.sum(x**2,axis=-1), np.sum(x**2,axis=-1)
            
            for key in "dens_star", "dens_dm", "scolumndens", "chi_star", "chi_dm", "nperi":
                res[key] = []
        
        def snap(i=0):
            if verbose:
                print("snap %d, step %d / %d" % (i/subsamp, i, len(ti)))
            
            res["pos"].append(np.copy(x))
            res["vel"].append(np.copy(v))
            res["acc"].append(np.copy(lastacc))
            
            if with_info:
                res["dens_star"].append(np.copy(last_dens_star))
                res["dens_dm"].append(np.copy(last_dens_dm))
                res["scolumndens"].append(np.copy(column_dens_star))
                res["chi_star"].append(np.copy(chi_star))
                res["chi_dm"].append(np.copy(chi_dm))
                res["nperi"].append(np.copy(nperi))
            
        snap()

        for i in range(1, len(ti)):
            dt = (ti[i] - ti[i-1])
            
            # kick
            if i == 1:
                acc = self.acceleration(x, components=components)
            else:
                acc = lastacc
            v = v + 0.5 * dt * acc / vfac
            
            # drift
            x = x + v * dt / vfac
            
            # kick
            acc = self.acceleration(x, components=components)

            v = v + 0.5 * dt * acc / vfac
            
            if with_info:
                
                dens_star = self.density(x, components="bs")
                dens_dm = self.density(x, components="h")
        
                chi_star += 0.5*(dens_star + last_dens_star)*dt * 1.022012156719175e-06
                chi_dm += 0.5*(dens_dm + last_dens_dm)*dt * 1.022012156719175e-06
                
                # The column density we just integrate less accurate, because we don't really need it
                column_dens_star += dens_star * np.linalg.norm(v, axis=-1) * 1.0220e-06 * dt # pc/year / (km/s)   

                r2 = np.sum(x**2, axis=-1)
                nperi += (r2_m1 < r2) & (r2_m1 < r2_m2)
                
                last_dens_star, last_dens_dm = dens_star, dens_dm
                r2_m2, r2_m1 = r2_m1, r2

            lastacc = acc
            
            if i % subsamp == 0:
                snap(i)
                
        for key in res:
            res[key] = np.array(res[key])

        return res
    
    def create_dm_orbits(self, ntot=10000, nsteps=10000, tmax=1e10, rmax=500e3, components="hbsg", seed=42, addinfo=False, subsamp=1, adaptive=False, verbose=False, mpicomm=None):
        """Creates a set of orbits inside the Milky Way potential.
        
        ntot : number of orbits
        nsteps : number of integration steps. The smaller radii you want to resolve the more steps you need.
        tmax : end time of the simulation in years
        rmax : maximal radius to sample towards in parsec
        components : keep this at "hbsg", unless you only want to consider the gravitational potential of individual components
        seed : random seed
        addinfo : If yes, will infer several additional quantities, beyond positions and velocities -- see below. If True, takes quite a bit longer
        subsamp : return only every subsamp-th step in the result (Using too low subsamp, may produce memory problems!)
        adaptive : use unequal mass-sampling. This is recommend, but you will have to make sure that you properly mass weight your results!
        verbose : verbosity
        mpicomm : if you pass an mpi communicator e.g. mpi4py.MPI.COMM_WORLD, you can use several mpi processes to speed up the calculation.
        
        returns a dictionary with the following entries (all units Msol, pc and km/s):
              mass : masses, shape (ntot,)
              pos : position, shape (ntot, nout, 3)   where nout = nsteps/subsamp
              vel : velocity, shape (ntot, nout, 3)
              acc : acceleration, shape (ntot, nout, 3)
        if addinfo, additionally we have: (all shapes = (ntot, nout))
              dens_star : the stellar mass density at each time (Msol/pc**3)
              dens_dm : the dark matter mass density at each time (Msol/pc**3)
              chi_star : time-integral over the stellar mass. Unit is Msol/pc**2 / (km/s)
              chi_dmr : time-integral over the dark matter mass. Unit is Msol/pc**2 / (km/s)
              scolumndens : the total encountered stellar column density. Unit is Msol/pc**2
        """
        if rmax is None:
            rmax = self.profile_nfw.r200c*1e6
        
        if self.cachedir is None:
            h5cache = None
        else:
            import h5py
            h5cache = "%s/milkyway_orbits.hdf5" % self.cachedir
            #h5path = "/orbit_rmax%.4e_ntot%d_seed%d_components%s_nsteps%d_tmax%.4e_xinfo%d" % (rmax/1e6, ntot, seed, components, nsteps, tmax, addinfo)
            #if subsamp is not None:
            h5path = "/orbit_rmax%.4e_ntot%d_seed%d_components%s_nsteps%d_tmax%.4e_xinfo%d_sub%d" % (rmax/1e6, ntot, seed, components, nsteps, tmax, addinfo, subsamp)
            if adaptive:
                h5path += "_adaptive"
            if self.mode == "cautun_2020":
                h5path += "_cautun"
        
            if os.path.exists(h5cache):
                with h5py.File(h5cache, "r") as h5file:
                    #print(h5path)
                    if h5path in h5file:
                        print("reading")
                        res = at.h5methods.h5py_read_dict(h5cache, path=h5path)

                        return res
        print("Integrating orbits...")
        
        # calculation
        np.random.seed(seed)
        ti = np.linspace(0, tmax, nsteps)
        
        if adaptive:
            def res_of_r(r):
                return 1./r
        else:
            res_of_r = None
            
        if mpicomm is not None:
            ntotall = ntot
            ntot = ntot // mpicomm.Get_size()
            seed = seed + mpicomm.Get_rank()
            
            print("ntot per task: %d" % ntot)
            
        if self.mode == "sellwood_mcgaugh_2005":
            pos, vel, mass = self.profile_halo.sample_particles_uniform(ntot, rmax=rmax/1e6, nsteps_chain=1000, res_of_r=res_of_r, seed=seed)
        else:
            pos, vel, mass = self.profile_halo.sample_particles(ntot, rmax=rmax/1e6, res_of_r=res_of_r, seed=seed)
        
        res = self.integrate_orbit_with_info(pos*1e6, vel, ti, components=components, with_info=addinfo, subsamp=subsamp, verbose=verbose)
        
        if mpicomm is not None:
            # make sure we never communicate to much at once
            # (because mpi is a master at creating integer overflows!)
            niter = int(np.ceil((ntot*nsteps/subsamp) / 1000000))
            print("Doing %d communications" % niter)
            
            
            for key in res:
                res[key] = np.rollaxis(res[key], 1, 0) # put particle dimensions in front, since we can split along it
            res["mass"] = mass
            resnew = gather_array_dict(res, mpicomm, root=0, niter=niter)
            if mpicomm.Get_rank() != 0:
                return  None
            
            for key in res.keys():
                if key != "mass":
                    shape = resnew[key].shape
                    print("shape", resnew[key].shape)
                    res[key] = np.moveaxis(resnew[key].reshape((-1,) + shape[2:]), 0, 1) # put the particle axis back to 1
            res["mass"] = resnew["mass"].flatten() * (ntot * 1. / ntotall) #
        else:
            res["mass"] = mass

        if h5cache is not None:
            import h5py
            with h5py.File(h5cache, "a") as h5file:
                print("writing")
                at.h5methods.h5py_write_dict(h5file, res, path=h5path, overwrite=True, verbose=1)
                
        return res