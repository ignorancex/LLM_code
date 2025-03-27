import numpy as np
from scipy.integrate import simps, quad
from scipy.special import erf
from scipy.stats import multivariate_normal

def cum_simpson(f, xi, fi=None, **kwargs):
    """Returns the cumulative integral of a function using the simpson rule
    
    f : a function
    xi : evaluation points (the midpoint of each interval will as well be evaluated)
    fi : function values at xi, providing these is just an optimization
    ** kwargs : key word arguments that will be passed through to the function
    
    returns : cumulative integral from xi[0] to xi[:]. Shape is same as xi
    """
    if fi is None:
        fi = f(xi, **kwargs)
    
    xh = 0.5*(xi[1:] + xi[:-1])
    
    Is = (xi[1:] - xi[:-1])/6. * (fi[1:] + 4.*f(xh, **kwargs) + fi[:-1])
    
    return np.insert(np.cumsum(Is, axis=0), 0, 0.)

def f(x):
    """Eqn A15"""
    def fana(x):
        return ((x**3 - 3.*x) * (erf(x*np.sqrt(2.5)) + erf(x*np.sqrt(2.5)/2.))/2.
                + np.sqrt(2./(5.*np.pi)) * ((x**2*31./4. + 8./5.)*np.exp(-5.*x**2/8.)
                                            + (x**2/2. - 8./5.)*np.exp(-5.*x**2/2.))
               )
    def flim0(x):
        return 3.**5 * 5.**1.5 / (7. * 2.**11 * np.sqrt(2.*np.pi)) * x**8 * (1. - x**2.*5./8.)

    def fliminf(x):
        return x**3 - 3.*x

    return np.where(x > 0.1, fana(x), flim0(x))

def sample_metropolis_hastings(f, x0, stepsize=[1.,0.5], nsteps=[100,100], keepall=False):
    """Does an mcmc sampling of a probability distribution function
    only returns the last step of each chain.
    
    f : a pdf to sample
    x0 : start locations
    stepsize : standard deviation(s) for step sizes
    nsteps : total number of steps to perform, if array stepsize is adapted after nsteps
    """
    
    x = np.copy(x0)
    f0 = f(x)
    
    if keepall:
        xs = []
        
    def one_chain(nsteps, stepsize):
        for i in range(0, nsteps):
            dx = np.random.normal(loc=0., scale=stepsize, size=x.shape)

            f1 = f(x + dx)

            alpha = f1 / f0
            u = np.random.uniform(0., 1., size=x.shape[0])
            accept = u <= alpha

            x[accept] = (x+dx)[accept]
            f0[accept] = f1[accept]

            if keepall:
                xs.append(np.copy(x))
                
    nsteps, stepsize = np.atleast_1d(nsteps), np.atleast_1d(stepsize)
    
    for nstepsi, stepsizei in zip(nsteps, stepsize):
        one_chain(nstepsi, stepsizei)
        
    if keepall:
        return np.array(xs)
    else:
        return x

class PeakStatistics():
    def __init__(self, k, Dk, numax=20.):
        """This class is an implementation of the peak statistics 
        from BBKS 1986.
        
        The main function you will want to use is sample_peaks()
        
        This code has been inspired by and tested against
        https://github.com/delos/microhalo-models (Delos et al, 2019)
        
        k : wavenumbers
        Dk : dimensionless power spectrum at those wavenumbers
        numax : maximal peak height to consider. (20 is more than 
                wide enough)
        """
        
        self.k = k
        self.Dk = Dk
        
        self.sigma0 = self.compute_sigmaj(0)
        self.sigma1 = self.compute_sigmaj(1)
        self.sigma2 = self.compute_sigmaj(2)
        
        self.Rstar = np.sqrt(3.) * self.sigma1 / self.sigma2
        self.gamma = self.sigma1**2 / (self.sigma2 * self.sigma0)
        # Total number density of peaks (eq. 4.11b) -- analytic:
        self.ntot_ana = (29. - 6.*np.sqrt(6.))/(5.**1.5 * 2. * (2.*np.pi)**2 * self.Rstar**3)
        
        self.setup_numerics(numax=numax)
        
        self.ntot_positive = self.ntot - self.nsmaller_than_nu(0.)
        
    def setup_numerics(self, numax=20., nbins=1000):
        self.numax = numax
        
        # Setup G-interpolator
        self.xsip = np.linspace(-numax*self.gamma, numax*self.gamma, nbins)
        self.Gip = self.G_num(self.xsip)
        
        
        # Get cumulative nu distribution
        self.nus = np.linspace(-numax, numax, nbins)
        def f(nu):
            return self.dN_dnu(nu, mode="interp")
        self.ncum_of_nu = cum_simpson(f, self.nus)
        self.ntot = self.ntot_num = self.ncum_of_nu[-1]
        
        # Get cumulative p(x|nu) distribution
        
        
    def nsmaller_than_nu(self, nu):
        return np.interp(nu, self.nus, self.ncum_of_nu)
        
    def dimless_power(self, k):
        return np.interp(k, self.k, self.Dk)
    
    def compute_sigmaj(self, j=0):
        """BBKS eq. 4.cb, but using log-integration"""
        return np.sqrt(simps(self.Dk*self.k**(2*j), np.log(self.k)))
    
    def G_fit(self, w):
        y = self.gamma
        """BBKS eq. 4.4"""
        def A(y):
            return 5./2. / (9. - 5.*y**2)
        def B(y):
            return 432. / (np.sqrt(10*np.pi) * (9. - 5*y**2)**(5./2.))
        def C1(y):
            return 1.84 + 1.13*(1.-y**2)**5.72
        def C2(y):
            return 8.91 + 1.27*np.exp(6.51*y**2)
        def C3(y):
            return 2.58*np.exp(1.05*y**2)
        return ((w**3 - 3*y**2*w + (B(y)*w**2 + C1(y)) * np.exp(-A(y)*w**2))
                / (1. + C2(y)*np.exp(-C3(y)*w)))
    
    def G_num(self, xs):
        """Eqn A19, numerical evaluation"""
        y = self.gamma
        def G_single(xs):
            """Eqn. A19"""
            def g_integrand(x):
                return f(x) * np.exp(-(x - xs)**2. / (2.*(1. - y**2)))

            val, err = quad(g_integrand, 0., np.inf) / np.sqrt(2.*np.pi*(1. - y**2))

            return val

        return np.vectorize(G_single)(xs)
    
    def G_interp(self, xs):
        """Using interpolation table of numerical evaluation of Eqn A19"""
        return np.exp(np.interp(xs, self.xsip, np.log(self.Gip)))
    
    def G(self, nu, mode="interp"):
        if mode == "numerical":
            return self.G_num(nu)
        elif mode == "fit":
            return self.G_fit(nu)
        elif mode == "interp":
            return self.G_interp(nu)
        else:
            raise ValueError("Unknown mode: %s" % mode)
    
    def dN_dnu(self, nu, mode="interp"):
        """BBKS eq. 4.3"""
        y = self.gamma
        
        return 1./((2.*np.pi)**2 * self.Rstar**3)  * np.exp(-nu**2/2.) * self.G(y*nu, mode=mode)
    
    def sample_nu(self, size, numin=None):
        """sample values of nu through inverse distribution function sampling"""
        fi = np.random.uniform(0., 1., size=size)
        if numin is None:
            return np.interp(fi, self.ncum_of_nu / self.ntot_num, self.nus)
        else:
            n0 = self.nsmaller_than_nu(numin)
            return np.interp(fi, (self.ncum_of_nu - n0) / (self.ntot_num - n0), self.nus)
        
    def px_given_nu(self, x,nu, mode="interp"):
        """eqn. 7.5"""
        y = self.gamma
        xstar = y*nu
        return np.exp(-(x-xstar)**2 / (2.*(1-y**2))) / np.sqrt(2.*np.pi * (1. - y**2)) * f(x) / self.G(y*nu, mode=mode)
    
    def sample_x_given_nu(self, nu, pmax = 1.):
        xmax = self.numax
        
        nsamp = len(nu)
        x = np.zeros_like(nu)
        filled = np.zeros(nu.shape, dtype=np.bool)
        i = 0
        while np.sum(filled) < nsamp:
            nleft = np.sum(~filled)
            xsamp = np.random.uniform(0., xmax, nleft)
            frej = self.px_given_nu(xsamp, nu[~filled]) / pmax

            assert np.max(frej) < 1., "If this fails, increase pmax %g" % np.max(frej)
            fsamp = np.random.uniform(0., 1., nleft)
            
            x[~filled] = xsamp * (fsamp < frej)
            filled[~filled] = fsamp < frej
            i += 1
            assert i < 10000
        return x
    
    def p_density_ep_given_x(self, e, p, x):
        """eqn. 7.6
        Note: these are the ellipticity and prolateness of the density field
              In ellipsoidal collapse we usually use those of the potential
              (= ratios of eigenvalues of the tidal tensor)
        """
        def chi(e,p):
            """eqn C3"""
            return ( ((0. <= e) & (e <= 0.25) & (-e <= p) & (p <= e)) 
                    |((0.25 <= e) & (e <= 0.5) & (-(1 - 3*e) <= p) & (p <=e)) )
        
        def W(e, p):
            """eqn C4"""
            return e*(e**2 - p**2)*(1.-2*p)*((1. + p)**2 - 9*e**2)*chi(e,p)
        
        return (3.**2 * 5.**2.5)/np.sqrt(2.*np.pi) * x**8 / f(x) * np.exp(-2.5*x**2*(3.*e**2 + p**2)) * W(e,p)
    
    def p_potential_ep_given_nu(self, e, p, nu):
        """eqn. A3  of Sheth & Tormen (1999) arxiv:9907024
        Ellipticity and prolateness of the tidal tensor
        """
        constraints = (e > 0.) & (np.abs(p) <= e)
        
        return 1125./np.sqrt(10.*np.pi) * e*(e**2-p**2) * nu**5 * np.exp(-2.5*nu**2*(3*e**2 + p**2)) * constraints
        
    def sample_potential_ep_metropolis(self, nu, nsamp=None, nsteps=[50,50,50,50], stepsize=[0.4,0.2,0.1,0.05], keepall=False):
        """Samples ellipticity e and prolateness p of the potential/tidal tensor
        
        This is done through an MCMC chain / Metropolis hastings algorithm. 
        The stepsize gets changed after every nsteps
        interval. The default steps lead to extremely good convergence, fewer
        steps could be used
        """
        
        # we sample the variable ep * nu, since it seems reasonably
        # well distributed < 1
        def myfunc(ep_times_nu):
            return self.p_potential_ep_given_nu(ep_times_nu[...,0]/nu, ep_times_nu[...,1]/nu, nu)
        
        if nsamp is None:
            nsamp = np.shape(nu)
        
        # Initial conditions in the valid domain of e,p
        e_x_nu0 = np.random.uniform(0., 1., nsamp)
        p_x_nu0 = np.random.uniform(-e_x_nu0, e_x_nu0, nsamp)
        ep_x_nu0 = np.stack([e_x_nu0, p_x_nu0], axis=-1)
        
        epfin_x_nu = sample_metropolis_hastings(myfunc, ep_x_nu0, nsteps=nsteps, stepsize=stepsize, keepall=keepall)
        return epfin_x_nu[...,0]/nu, epfin_x_nu[...,1]/nu
        
        #return (np.exp(-15./2. * e**2 *nu**2) * (1. - 15.*e**2*nu**2) * erf(np.sqrt(5./2.)*e*nu)
        #        - 3.*np.sqrt(10./np.pi)*e*nu * np.exp(-10.*e**2*nu**2) + erf(np.sqrt(10.)*e*nu))

    def sample_density_ep_given_x_metropolis(self, x, nsamp=None, nsteps=[70,70,70], stepsize=[0.2,0.1,0.05], keepall=False):
        """Samples ellipticity e and prolateness p given a steepness x
        
        Note: these are the ellipticity and prolateness of the density field
              In ellipsoidal collapse we usually use those of the potential
              (= ratios of eigenvalues of the tidal tensor)
        
        This is done through an MCMC chain / Metropolis hastings algorithm. 
        The stepsize gets changed after every nsteps
        interval. I found this for any x<20 to be very well converged. (Actually have as
        many steps would also do.) Therefore, this should work well in all cases
        """
        def myfunc(ep):
            return self.p_density_ep_given_x(ep[...,0], ep[...,1], x)
        
        if nsamp is None:
            nsamp = np.shape(x)
        
        # Initial conditions in the valid domain of e,p
        e0 = np.random.uniform(0., 0.5, nsamp)
        pmin = np.where(e0 < 0.25, -e0, -(1-3*e0))
        p0 = np.random.uniform(pmin, e0, nsamp)
        ep0 = np.stack([e0, p0], axis=-1)
        
        epfin = sample_metropolis_hastings(myfunc, ep0, nsteps=nsteps, stepsize=stepsize, keepall=keepall)
        return epfin[...,0], epfin[...,1]
    
    def sample_peaks(self, nsamp, numin=None):
        """Sample peaks parameterized through four parameters:
        
        returns he peak height nu = delta/sigma_delta, the steepness 
        x = - L / sigma_L, the ellipcity e = (l1 - l3) / (2 delta) and
        the prolateness p = (l1 - 2l2 + l3)/(2 delta)
        where L is the laplacian of the density field and li are the 
        eigenvalues of the deformation tensor, which are all positive
        due to the peak constraint.
        
        nsamp : number of peaks to sample
        numin : minimal peak height. E.g. when using spherical collapse
                arguments you might want to set this to 0 or even larger.
        """
        
        nu = self.sample_nu(nsamp, numin=numin)
        x = self.sample_x_given_nu(nu)
        e, p = self.sample_potential_ep_metropolis(nu)
        
        return nu,x,e,p