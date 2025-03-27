import numpy as np
import sys
sys.path.append("../adiabatic-tides")
import adiabatic_tides as at

class PhasespaceTruncatedCusp(at.profiles.MonteCarloProfile):
    def __init__(self, A=1., rcore=None, ecore=None, fmax=None, rmin=1e-9, rmax=1e15, nbins=10000):
        """Set up a r**-3/2 cusp with a phase space core.
        
        The profile is constructed by assuming that the phase space distribution
        is given by
        
        f(E) = f0 * (E + Ec)**(-9/2)
        
        where Ec is the energy scale of the core. This profile has a maximum
        phase space density fmax = f0*Ec**(-9/2) and a core radius defined through
        phi_powerlaw(rc) = Ec
        
        Note that for Ec=0 a r**-3/2 powerlaw is exactly recovered
        
        The corresponding density profile is not analytic, but we infer it numerically
        Note that a reasonable approximation (20% accuracy) is given by
        rho(r) = A*(r**4 + rc**4)**(-3./8.)
        
        A : normalization of the powerlaw profile
        
        provide exactly one of the following parameters:
        
        rcore : core radius
        ecore : core energy scale
        fmax : maximum phase space density
        
        numerical parameters (usually don't need changing):
        
        rmin : minimum radius
        rmax : maximum radius
        nbins : number of steps used for the integration
        """       
        self.base_powerlaw = at.profiles.PowerlawProfile(slope=-1.5, rhoc=A)
        
        self.A = A
        self.f0 = self.base_powerlaw.f0
        
        if rcore is not None:
            self.rcore = rcore
            self.ecore = self.base_powerlaw.potential(rcore)
            self.fmax = self.f0 * self.ecore**(-9/2.)
        elif ecore is not None:
            #phi = cprof.base_powerlaw.phic * ri**0.5
            self.ecore = ecore
            self.rcore = (self.ecore / self.base_powerlaw.phic)**2.
            if ecore > 0.:
                self.fmax = self.f0 * self.ecore**(-9/2.)
            else:
                self.fmax = np.infty
        elif fmax is not None:
            self.fmax = fmax
            self.ecore = (self.f0 / self.fmax) ** (2./9.)
            self.rcore = (self.ecore / self.base_powerlaw.phic)**2.
        else:
            raise ValueError("Please give either rcore, ecore or fmax")
            
        rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins)*self.r0()
        
        super().__init__(rbins=rbins, ancorphi="rmin")
        
        self._integrate_profile()
        
        
    def _integrate_profile(self): # Ec=1., 
        def F_rho_of_phi(phi, Ec=1., f0=1.):
             # Note that f0 = fmax*E**(9/2)
            return f0*64*np.pi*np.sqrt(2)/(105.*(Ec + phi)**3)

        ri = self.rbins

        phi = np.zeros_like(ri)
        mofr = np.zeros_like(ri)
        rho = np.zeros_like(ri)

        for i in range(1, len(ri)):
            dr = ri[i] - ri[i-1]

            rho[i-1] = F_rho_of_phi(phi[i-1], f0=self.f0, Ec=self.ecore)
            mofr[i] = mofr[i-1] + 4.*np.pi*ri[i-1]**2 * rho[i-1] * dr
            phi[i]  = phi[i-1] + (self.G * mofr[i-1] / ri[i-1]**2)*dr
            
        self._set_mass_profile(np.sqrt(rho[1:])*np.sqrt(rho[:-1]), mofr)
        
    def density_analytic_approx(self, r):
        return self.A*(r**4 + self.rcore**4)**(-3./8.)
        
    def f_of_e(self, e):
        return self.f0 * (e + self.ecore)**(-9./2.)
        
    def r0(self):
        if self.rcore > 0:
            return self.rcore
        else:
            return 1.
        
    def frac_rperi_in_range(self, r, rperi_min=0., rperi_max=1.):
        phii = self.potential(r)
        
        def F_rho_of_phi(phi, Ec=self.ecore, f0=self.f0):
            return (64./105.)*np.sqrt(2)*np.pi*f0/(Ec + phi)**3
        
        phi_rpmin = self.potential(rperi_min)
        phi_rpmax = self.potential(rperi_max)
        
        phistar_min = (phii - phi_rpmin * (rperi_min / r)**2) / (1. - (rperi_min/r)**2)
        phistar_max = (phii - phi_rpmax * (rperi_max / r)**2) / (1. - (rperi_max/r)**2)
        
        fnorm = F_rho_of_phi(phi=phii)
        
        res = np.zeros_like(r)
        res[r >= rperi_min]  = F_rho_of_phi(phi=phistar_min[r >= rperi_min])*np.sqrt(1. - (rperi_min/r[r >= rperi_min])**2)
        res[r >= rperi_max]  -= F_rho_of_phi(phi=phistar_max[r >= rperi_max])*np.sqrt(1. - (rperi_max/r[r >= rperi_max])**2)

        return  res / fnorm
    
    def sample_el_givenr_and_rperi_range(self, r, rperi_min=0., rperi_max=1., niter=50):
        phii = self.potential(r)
        
        # This function indicates the density at radius r, but selected only on
        # particles with E < Emax
        def fcum(E, phi, Ec=self.ecore, f0=self.f0):
            return ((8./105.)*np.sqrt(2)*np.pi*f0*(E - phi)**(3./2.)
                    *(8.*E**2 + 28.*E*Ec + 12*E*phi + 35.*Ec**2 + 42*Ec*phi + 15*phi**2)
                    /((E + Ec)**(7/2)*(Ec + phi)**3))
        
        def F_rho_of_phi(phi, Ec=self.ecore, f0=self.f0):
            return (64./105.)*np.sqrt(2)*np.pi*f0/(Ec + phi)**3

        phi_rpmin = self.potential(rperi_min)
        phi_rpmax = self.potential(rperi_max)
        
        phistar_min = (phii - phi_rpmin * (rperi_min / r)**2) / (1. - (rperi_min/r)**2)
        phistar_max = (phii - phi_rpmax * (rperi_max / r)**2) / (1. - (rperi_max/r)**2)
        
        phistar = phistar_max
        
        fnorm = F_rho_of_phi(phii) * self.frac_rperi_in_range(r, rperi_min=rperi_min, rperi_max=rperi_max)
        
        
        sel1 = r >= rperi_min
        sel2 = r >= rperi_max
        
        # This function indicates the fraction of particles at radius r with rperi < rperi_max
        # that have an energy level smaller than E
        def fcumE_sel(E):
            res = np.zeros_like(E)
            s1, s2 = sel1 & (E >= phistar_min), sel2 & (E >= phistar_max)
            #res = fcum(E, phi=phii)
            res[s1] = fcum(E[s1], phi=phistar_min[s1])*np.sqrt(1. - (rperi_min/r[s1])**2)
            res[s2] -= fcum(E[s2], phi=phistar_max[s2])*np.sqrt(1. - (rperi_max/r[s2])**2)
            
            return  res / fnorm
        
        Fi = np.random.uniform(0., 1., np.shape(r))
        def myfunc(E, ftarget=Fi):
            return fcumE_sel(E) - ftarget
        
        Ei = at.mathtools.vectorized_binary_search(myfunc, xlow=phii, xhigh=self.potential(np.max(self.rbins))*np.ones_like(r), ftarget=Fi, niter=niter)
        
        def fcum_L_unnormed(L, E=Ei, phi=phii, r=r):
            return -np.sqrt(2.*np.clip(E - phi - 0.5*L**2/r**2, 0, None))# - np.sqrt(2.*(E - phi))
        
        #Lmin = np.zeros_like(r)
        Lmin = rperi_min * np.sqrt(np.clip(2.*(Ei - phi_rpmin), 0., None))
        Lmax = r * np.sqrt(2*(Ei - phii))
        Lmax[sel2] = rperi_max * np.sqrt(2.*(Ei[sel2] - phi_rpmax))
        
        assert(np.all(~np.isnan(Lmax)))
        assert(np.all(~np.isnan(fcum_L_unnormed(Lmax))))

        def fcum_L(L, E=Ei, phi=phii, r=r):
            res = (fcum_L_unnormed(L) - fcum_L_unnormed(Lmin)) / (fcum_L_unnormed(Lmax) - fcum_L_unnormed(Lmin))
            
            res[L <= Lmin] = 0.
            res[L >= Lmax] = 1.
            #assert np.all(~np.isnan(res))
            
            return res
        
        Fi = np.random.uniform(0., 1., np.shape(r))
        def myfunc(L, ftarget=Fi):
            return fcum_L(L) - ftarget
        Li = at.mathtools.vectorized_binary_search(myfunc, xlow=Lmin, xhigh=Lmax, ftarget=Fi, niter=niter, mode="mean")

        return Ei, Li
    
    def sample_particles_rperi_range(self, ntot=10000, rperi_min=0., rperi_max=None, rmax=None, seed=None, niter_e=50):
        """Samples particles consistent with the phasespace distribution
        
        ntot : number of particles to sample
        rperi_min : minimum pericenter radius
        rperi_max : maximum pericenter radius
        rmax : Maximum radius to sample
        seed : random seed
        
        returns : pos, vel, mass  -- the positions, velocities and masses
               masses are normalized so that Sum(mass) = M(<rmax)
        """
        #float_err_handling = np.geterr()
        #np.seterr(divide="ignore", invalid="ignore") 
        
        if seed is not None:
            np.random.seed(seed)
        if rmax is None:
            rmax = self.r0()
        if rperi_max is None:
            rperi_max = rmax
            
        def fweight(r):
            return self.frac_rperi_in_range(r, rperi_min=rperi_min, rperi_max=rperi_max)
            
        ri, mass = self.sample_r(ntot, rmax=rmax, fweight=fweight, get_mass=True)
        ei, li = self.sample_el_givenr_and_rperi_range(ri, rperi_min=rperi_min, rperi_max=rperi_max, niter=niter_e)
        
        # A random radial vector
        u_r = at.mathtools.random_direction(ri.shape, 3)
        # A random orthogonal vector
        u_t = at.mathtools.random_direction(ri.shape, 3)
        u_t = u_t - np.sum(u_t*u_r, axis=-1)[...,np.newaxis] * u_r
        u_t = u_t / np.linalg.norm(u_t, axis=-1)[...,np.newaxis]
        
        vr2 = 2.*(ei - 0.5*(li**2/ri**2) - self.potential(ri))
        vr = np.sign(np.random.uniform(-0.5, 0.5, size=ri.shape)) * np.sqrt(vr2)
        vt = np.sqrt(li**2/ri**2)
        
        pos = u_r * ri[...,np.newaxis]
        vel = u_r * vr[...,np.newaxis] + u_t * vt[...,np.newaxis]
        
        return pos,vel,mass#,ri,ei,li
    
    def sample_particles_rperisplits(self, npersplit=10000, rsplits=(0., 1., 10.), seed=None, niter_e=50, flatten=True):
        if seed is not None:
            np.random.seed(seed)
        rmax = rsplits[-1]
        nsplits = len(rsplits) - 1
        pos,vel,mass = np.zeros((nsplits, npersplit, 3)),np.zeros((nsplits, npersplit, 3)),np.zeros((nsplits, npersplit))
        for i in range(0, nsplits):
            pos[i],vel[i],mass[i] = self.sample_particles_rperi_range(ntot=npersplit, rperi_min=rsplits[i], rperi_max=rsplits[i+1], rmax=rmax)
        
        if flatten:
            return pos.reshape(-1,3), vel.reshape(-1,3), mass.reshape(-1)
        else:
            return pos, vel, mass
        
    
    def sample_e_givenr(self, r, niter=50):
        Fi = np.random.uniform(0., 1., np.shape(r))
        phii = self.potential(r)
        
        def fcum_E(E, phi=phii, E_c=self.ecore):
            return ((1./840.)*(E - phi)**(3/2)*(105.*E_c**3 + 315.*E_c**2*phi + 315.*E_c*phi**2 + 105.*phi**3)
                    *(8*E**2 + 28*E*E_c + 12*E*phi + 35*E_c**2 + 42*E_c*phi + 15*phi**2)
                    /((E + E_c)**(7/2)*(E_c + phi)**3) )
        
        def myfunc(E, ftarget=Fi):
            return fcum_E(E) - ftarget
        
        Ei = at.mathtools.vectorized_binary_search(myfunc, xlow=phii, xhigh=self.potential(np.max(self.rbins)), ftarget=Fi, niter=niter)
        
        return Ei
    
    def sample_eeff_givenr(self, r, niter=50):
        Fi = np.random.uniform(0., 1., np.shape(r))
        phii = self.potential(r)
        
        def FcumEeff_rel(E_eff, phi=phii, E_c=self.ecore):
            return (1./8.)*np.sqrt(E_eff-phi)*((15.*E_c**2 + 20.*E_c*E_eff + 10*E_c*phi 
                                                + 8*E_eff**2 + 4*E_eff*phi + 3*phi**2)
                                               /(E_c + E_eff)**(5./2.))
        
        def myfunc(E, ftarget=Fi):
            return FcumEeff_rel(E) - ftarget
        
        Ei = at.mathtools.vectorized_binary_search(myfunc, xlow=phii, xhigh=self.potential(np.max(self.rbins)), ftarget=Fi, niter=niter)
        
        return Ei
    
    def sample_r(self, ntot=100, rmax=None, fweight=None, get_mass=False):
        """Sample radii from the density profile"""
        if rmax is None:
            rmax = self.r0()

        fsamp = np.random.uniform(0., 1., ntot)
        
        if fweight is None:
            mmax = self.m_of_r(rmax)
            rsamp = np.interp(fsamp, self.q["mofr"]/mmax, self.rbins)
        else:
            mi = (self.q["mofr"][1:] - self.q["mofr"][:-1]) #* fweight(n)
            mcum = np.concatenate([[0.], np.cumsum(mi * fweight(np.sqrt(self.rbins[1:]*self.rbins[:-1])))])
            mmax = np.interp(rmax, self.rbins, mcum)
            
            rsamp = np.interp(fsamp, mcum/mmax, self.rbins)

        if get_mass:
            return rsamp, np.ones(ntot)*(mmax/ntot)
        else:
            return rsamp
    
    def sample_particles(self, ntot=10000, rmax=None, seed=None, niter_e=50):
        """Samples particles consistent with the phasespace distribution
        
        ntot : number of particles to sample
        rmax : Maximum radius to sample
        seed : random seed
        
        returns : pos, vel, mass  -- the positions, velocities and masses
               masses are normalized so that Sum(mass) = M(<rmax)
        """
        #float_err_handling = np.geterr()
        #np.seterr(divide="ignore", invalid="ignore") 
        
        if seed is not None:
            np.random.seed(seed)
        if rmax is None:
            rmax = self.r0()
            
        ri = self.sample_r(ntot, rmax=rmax)
        ei = self.sample_e_givenr(ri, niter=niter_e)
        
        pos = at.mathtools.random_direction(ri.shape, 3) * ri[...,np.newaxis]
        
        phi = self.potential(ri)
        vi = np.sqrt(2.*(ei-phi))
        vel = at.mathtools.random_direction(ri.shape, 3) * vi[...,np.newaxis]
        
        mass = np.ones_like(ri) * self.m_of_r(rmax) / ntot
        
        return pos,vel,mass
    
    def to_string(self):
        mystr = "A%.5e_rcore%.5e_rmin%.5e_rmax%.5e_nbins%d" % (self.A, self.rcore, np.min(self.rbins), np.max(self.rbins), len(self.rbins))
        
        return mystr
    
    def _initialize_numerical_scales(self):
        """Sets some default values for numerical scales"""
        super()._initialize_numerical_scales()
        
        self._sc["fintegration_nstepsE"] = 501
        self._sc["fintegration_nstepsL"] = 201
        #self._sc["fintegration_nstepsE"] = 201
        #self._sc["fintegration_nstepsL"] = 101
        self._sc["ip_e_of_jl_nbinsE"] = 2000
        self._sc["ip_e_of_jl_nbinsE"] = 1000
        #self._sc["ip_e_of_jl_nbinsL"] = 200
        #self._sc["ip_e_of_jl_nbinsL"] = 100

        self._sc["log_emin"] = -18
        self._sc["rmin"] = self.r0() * 1e-3
        self._sc["rmax"] = self.r0() * 1e8
        #self._sc["rperimin"] = self.r0() * 1e-12
        #self._sc["niter_apoperi"] = 35
        
        self._sc["rel_interpolation_kind"] = "linear"
    
    
def write_gputree_ic(file, autoname=False, overwrite=False, Np2=16, A=1., cored=True, rmax=100., niter_e=50, convert_units=True, seed=42):
    if autoname:
        dirname = file
        if cored:
            file = "ic_cored_N2p%d_rmax%g_seed%d.dat" % (Np2, rmax, seed)
        else:
            file = "ic_powerlaw_N2p%d_rmax%g_seed%d.dat" % (Np2, rmax, seed)
        print(file)
        file = "%s/%s" % (dirname, file)
    
    import os
    if os.path.exists(file):
        if not overwrite:
            return
    
    N = 2**Np2
    if cored:
        cprof = PhasespaceTruncatedCusp(rcore=1e-6, A=A*1e9, rmin=1e-6*1e-9, rmax=1e-6*1e18)
    else:
        cprof = PhasespaceTruncatedCusp(rcore=1e-9, A=A*1e9, rmin=1e-6*1e-9, rmax=1e-6*1e18)
    icdtype = np.dtype([('pos', (np.float32,3)), ('vel', (np.float32,3)), ('mass', np.float32), ('phi', np.float32), ('id', np.int32)])
    
    pos, vel, mass = cprof.sample_particles(N, rmax=rmax*1e-6, niter_e=niter_e, seed=seed)
    
    data_ic = np.zeros(N, dtype=icdtype)
    data_ic["pos"], data_ic["vel"], data_ic["mass"] = pos, vel, mass
    data_ic["id"] = np.arange(N)
    
    if convert_units:
        data_ic["pos"] *= 1e6
        data_ic["vel"] /= np.sqrt(43.0071e-4) # Grav. constant in Mpc (km/s)^2 / Msol times (Mpc/pc)
    
    with open(file, "wb") as fd:
        data_ic.tofile(fd)
        
    return data_ic

def write_gputree_ic_splitted(file, autoname=False, overwrite=False, npersplit=2**16, A=1., cored=True, rsplits=(0.,1.,10.,100.,1000.,10000.), niter_e=50, convert_units=True, seed=42):
    nsplits = len(rsplits)-1
    N = npersplit*nsplits
    rmax = rsplits[-1]
    if autoname:
        dirname = file
        if cored:
            file = "ic_%dsplit_cored_Ndm%d_ns%d_rmax%g_seed%d.dat" % (nsplits, N, npersplit, rmax, seed)
        else:
            file = "ic_%dsplit_powerlaw_Ndm%d_ns%d_rmax%g_seed%d.dat" % (nsplits, N, npersplit, rmax, seed)
        print(file)
        file = "%s/%s" % (dirname, file)
    
    import os
    if os.path.exists(file):
        if not overwrite:
            print("File exists %s" % file)
            return
    
    if cored:
        cprof = PhasespaceTruncatedCusp(rcore=1e-6, A=A*1e9, rmin=1e-6*1e-11, rmax=1e-6*1e22)
    else:
        cprof = PhasespaceTruncatedCusp(rcore=1e-9, A=A*1e9, rmin=1e-6*1e-11, rmax=1e-6*1e22)
    
    
    icdtype = np.dtype([('pos', (np.float32,3)), ('vel', (np.float32,3)), ('mass', np.float32), ('phi', np.float32), ('id', np.int32)])
    
    pos, vel, mass = cprof.sample_particles_rperisplits(npersplit, rsplits=np.array(rsplits)*1e-6, niter_e=niter_e, seed=seed)
    assert len(pos) == N
    
    data_ic = np.zeros(N, dtype=icdtype)
    data_ic["pos"], data_ic["vel"], data_ic["mass"] = pos, vel, mass
    data_ic["id"] = np.arange(N)
    
    if convert_units:
        data_ic["pos"] *= 1e6
        data_ic["vel"] /= np.sqrt(43.0071e-4) # Grav. constant in Mpc (km/s)^2 / Msol times (Mpc/pc)
    
    with open(file, "wb") as fd:
        data_ic.tofile(fd)
        
    return data_ic