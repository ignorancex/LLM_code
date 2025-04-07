#Third party imports
from scipy.integrate import quad
from scipy.optimize import root
import scipy.special as sc
import numpy as np
import pandas as pd
import astropy.constants as const
import astropy.units as u
import numpy as np
from sympy import symbols
from IPython.display import display
from astropy.cosmology import WMAP9 as cosmo
# Project 
from density import density

class plummer(density):
    '''
        #TODO: add gradient
    '''
    def __init__(self,lims=(0,np.inf),**theta):
        '''
        Params:
        a     : plummer radius
        Mass  : mass  
        rho_s : scale density
        #! Either mass or scale density must be provided -- both is ok as long as they're consistent
        #TODO: Print warning if both mass and density are provided-- but they are not consistent
        '''
        self._a     = theta['a']
        self._rhalf = (.5**(-2/3) - 1)**(-.5)*self._a # 3D half mass radius
        self._lims = lims
        try:
            if theta['mass'].value:
                self._mass = theta['mass']
                self._rho_s = 3*self._mass/(4*np.pi*self._a**3)
            elif theta['rho'].value:
                self._rho_s = theta['rho_s']
                self._mass  = (4/3) *np.pi*self._a**3 * self._rho_s
        except: # If non are provided assume that the mass = 1
            self._mass  = 1.0
            self._rho_s = 3*self._mass/(4*np.pi*self._a**3)
    
    def rhalf(self):
        '''
        3D half Mass radius
        '''
        return self._rhalf
        
    def profile2d(self,R):
        ''' 
        2D surface density
        '''
        R = R if isinstance(R,u.Quantity) else R*u.kpc
        Rp = self._a
        m3D = self._mass
        coeff = m3D*Rp**-2/np.pi
        x = R/Rp
        return coeff * (1 + x**2)**(-2)

    def profile3d(self,r):
        r = r if isinstance(r,u.Quantity) else r*u.kpc
        a   = self._a
        rho = self._rho_s
        x  = r/a
        out =rho*(1+x**2)**(-5/2)
        return out

    def mass(self,r):
        '''M(r)'''
        r = r if isinstance(r,u.kpc) else r*u.kpc
        if r.value == np.inf:
            super(plummer,self).mass(r)

        a   = self._a
        Mtot = self._mass
        term1 = Mtot *r**3
        term2 = (r**2 + a**2)**(3/2)
        return term1/term2

    def sample_r(self,N):
        '''
        sample from plummer sphere -- inverse transform-sampling
        '''
        y = np.random.uniform(size=N) # Draw random numbers from 0 to 1
        r = self._a/ np.sqrt(y**(-2 / 3) - 1)
        return r

    def sample_xyz(self,N):
        '''
        draw x,y,z by sampling uniformly from a sphere
        '''
        r = self.sample_r(N)    
        xyz = np.zeros((3,N))
        phi = np.random.uniform(0, 2 * np.pi, size=N)
        temp = np.random.uniform(size=N)
        theta = np.arccos( 1 - 2 *temp)

        xyz[0] = r * np.cos(phi) * np.sin(theta)
        xyz[1] = r * np.sin(theta)*np.cos(theta)
        xyz[2] = r * np.cos(theta) # Check the angle conventions again! 

        return xyz
    def sample_R(self,N):
        r = self.sample_r(N)    
        xyz = np.zeros((2,N))
        phi = np.random.uniform(0, 2 * np.pi, size=N)
        temp = np.random.uniform(size=N)
        theta = np.arccos( 1 - 2 *temp)

        xyz[0] = r * np.cos(phi) * np.sin(theta)
        xyz[1] = r * np.sin(theta)*np.cos(theta)

        return np.sqrt(xyz[0]**2+xyz[1]**2)

    def __str__(self):
        expr= symbols(r'\rho(r)=\frac{3M_{0}}{4\pi{}a^3}\left(1+\frac{r^2}{a^2}\right)^{-\frac{5}{2}}')
        display(expr)
        return "M = %.2e %s\na = %.2e %s"%(self._mass.value,self._mass.unit,self._a.value,self._a.unit)

    def __repr__(self):
        return self.__str__()

class HerquistZhao(density):

    def __init__(self,lims=(0,np.inf),log=False,**theta):
        '''
        rho_d    : scale density -- units: [M L^-3] if mass density,  [L^-3] for number density
        r_d      : scale radius -- units: [L]
        a        : inner slope -- for  r/r_s < 1  rho_star ~ r^{-a} 
        b        : characterizes width of transition between inner slope and outer slope
        c        : outer slope -- for r/r_s >> 1 rho_star ~ r^{-c}
        Notes:
        #!for c > 3 mass diverges r -> infinity
        References:
            Baes 2021      : http://dx.doi.org/10.1093/mnras/stab634
            An & Zhao 2012 : http://dx.doi.org/10.1093/mnras/sts175
            Zhao 1996      : http://dx.doi.org/10.1093/mnras/278.2.488
        '''
        self._a = theta['a']
        self._b = theta['b']
        self._c = theta['c']

        if log:
            self._rho_s = np.exp(theta['rho_s'])
            self._r_s   = np.exp(theta['r_s'])
        else:
            self._rho_s = theta['rho_s']
            self._r_s   = theta['r_s']

    def profile3d(self,r):
        '''
        '''
        x = r/self._r_s
        nu = (self._a-self._c)/self._b
        return self._rho_s*x**(-self._a)*(1+x**self._b)**nu

    def mass(self,r):
        '''
        Mass profile using Gauss's Hypergeometric function
        Notes:
            See mass_error- too see comparision with calculating the mass by integrating the desity profile -- no errors until r/rs > 1e8
            #? Coeff calculation throws a warning at some point -- but doesnt when I try individual values?
            #! treatment of units is very clunky
        '''        
        x      = (r/self._r_s).value
        coeff  = 4*np.pi *self._rho_s*(r**3)*(x**(-self._a))/(3-self._a) 
        gauss  = sc.hyp2f1((3-self._a)/self._b,(self._c-self._a)/self._b,(-self._a+self._b+3)/self._b,-x**self._b)
        return coeff*gauss

    def potential(self,r):
        '''
        Potential for Hernquist density:
        phi_in = -GM/r  term -- see mass()
        phi_out = integral_{r}^{\infty} \rho(r) r dr -- calculated using a slightly different hypergeometric function -- evaluated at r and a very large number 
        '''
        func = lambda r: -r**2 *(r/self._r_s)**(-self._a) *\
            (sc.hyp2f1((2 - self._a)/self._b, -((self._a - self._c)/self._b), 1 + (2 - self._a)/self._b, -(r/self._r_s).value**self._b))/(-2 + self._a)
        
        phi_in  = (-const.G*self.mass(r)/r).to(u.km**2/u.s**2)
        phi_out =  -(4*np.pi*const.G*self._rho_s*(func(1e20*u.kpc)-func(r))).to(u.km**2/u.s**2)
        
        return phi_in+phi_out

    def __str__(self):
        expr= symbols(r'\rho(r)=\frac{\rho_s}{\left(\frac{r}{r_s}\right)^{a}\left(1+\left(\frac{r}{r_s}\right)^{b}\right)^{\frac{c-a}{b}}}')
        display(expr)
        out = 'rho_s = %.2e %s\nr_s = %.2e %s \na = %d\nb = %d \nc = %d\n'%(self._rho_s.value,self._rho_s.unit,self._r_s.value,self._r_s.unit,self._a,self._b,self._c)
        return out

    def __repr__(self):
        return self.__str__()
        
    def r200(self,r):
        '''
        Radius at which the average density of the halo is equal to 200 times the critical density
        Notes:
        '''
        rho_crit = cosmo.critical_density(0).to(u.solMass/u.kpc**3)
        Delta    = 200
        func     = lambda x: self.mass(r)/(4*np.pi*(r**3)/3) - Delta*rho_crit        
        
        self.r200 = root(func,x0=25)['x'][0] *u.kpc
        self.M200 = self.mass(self.r200)
        return self.r200

class System:
    '''
    #TODO Dealing with units is very clunky -- need to fix  
    '''
    
    def __init__(self,**kwargs):
        '''
        Params:
        tracer      : stellar class
        dark_matter : density class
        beta        : (double) between (-np.inf,1)
        pm          : (bool): include proper motions
        '''
        self._tracer = kwargs['tracer']
        self._dm     = kwargs['dark_matter']
        self._beta   = kwargs['beta']
        self._pm     = kwargs['pm']
        # self._G      = const.G.to(u.kpc**3 *u.solMass**(-1) * u.s**(-2)).value
        self._G   = 4.5171031e-39

    def disp_integrand(self,x,R,kernel):
        
        kern  = kernel(x,R,self._beta)
        nu_star = self._tracer.profile3d(x).value
        mass_dm = self._dm.mass(x).value
        return kern * nu_star*mass_dm/x**(2-2*self._beta)

    def dispersion_pmt(self,R):
        '''
        Project Velocity Dispersion see
        '''
        coeff = 2*self._G/self._tracer.profile2d(R).value     
        R,units = R.value,R.unit
        output = np.array([quad(self.disp_integrand,i,np.inf,args=(i,self.Fpmt))[0] for i in R])
        return ((coeff * output)*u.kpc**2/u.s**2).to(u.km**2/u.s**2).value
    def dispersion_pmr(self,R):
        '''
        Project Velocity Dispersion see
        '''
        coeff = 2*self._G/self._tracer.profile2d(R).value     
        R,units = R.value,R.unit
        output = np.array([quad(self.disp_integrand,i,np.inf,args=(i,self.Fpmr))[0] for i in R])
        return ((coeff * output)*u.kpc**2/u.s**2).to(u.km**2/u.s**2).value
    
    def dispersion(self,R):
        '''
        Project Velocity Dispersion see
        '''
        coeff = 2*self._G/self._tracer.profile2d(R).value     
        R,units = R.value,R.unit
        output = np.array([quad(self.disp_integrand,i,np.inf,args=(i,self.F))[0] for i in R])
        return ((coeff * output)*u.kpc**2/u.s**2).to(u.km**2/u.s**2).value

    def F(self,r,R,beta):
        '''
        Integration Kernel for constants stellar anisotropy
        #TODO Add references?
        '''

        t1 = .5* R**(1-2*beta)
        w  = (R/r)**2
        if beta >1/2:
            t2  = beta * sc.betainc(beta+0.5,0.5,w) - sc.betainc(beta-.5,.5,w)
            t3  = np.sqrt(np.pi)*(3/2 -beta)*sc.gamma(beta-1/2)/sc.gamma(beta)
            out = t1*(t2+t3)
        elif (beta==.5) or (beta==-.5):
            u = R/r 
            out = u**(2*beta -1)*np.arccosh(u) - beta*np.sqrt(1 - u**(-2)) 
        else: 
            a        = beta+0.5
            b        = 0.5
            betainc  = ((w**(a))/a) * sc.hyp2f1(a,1-b,a+1,w)
            a2       = beta-0.5
            betainc2 = ((w**(a2))/a2) * sc.hyp2f1(a2,1-b,a2+1,w)
            t2       = beta * betainc - betainc2
            t3       = np.sqrt(np.pi)*(3/2 -beta)*sc.gamma(beta-1/2)/sc.gamma(beta)
            out = t1*(t2+t3)
        return out
    
    def Fpmt(self,r,R,beta):
        t1 = .5*(beta-1)* R**(1-2*beta)
        w  = (R/r)**2
        if beta > 1/2:
            t2  = sc.betainc(beta-0.5,0.5,w)
            t3  = np.sqrt(np.pi)*sc.gamma(beta-1/2)/sc.gamma(beta)
            
        else: 
            a = beta-0.5
            b = 0.5
            betainc = ((w**(a))/a) * sc.hyp2f1(a,1-b,a+1,w)
            
            t2  = betainc

            t3  = np.sqrt(np.pi)*sc.gamma(beta-1/2)/sc.gamma(beta)

        out = t1*(t2-t3)

        return out
    
    def Fpmr(self,r,R,beta):
        t1 = .5* R**(1-2*beta)
        w  = (R/r)**2
        if beta > 1/2:
            t2  = (beta-1) * sc.betainc(beta-0.5,0.5,w) - sc.betainc(beta+.5,.5,w)
            t3  = np.sqrt(np.pi)*sc.gamma(beta-1/2)/(2*sc.gamma(beta)) 
        else: 
            a   = beta - 0.5
            b   = 0.5
            betainc = (beta-1)* ((w**(a))/a) * sc.hyp2f1(a,1-b,a+1,w)
            a2  = beta + 0.5
            betainc2 = beta *((w**(a2))/a2) * sc.hyp2f1(a2,1-b,a2+1,w)
            
            t2  = betainc - betainc2
            t3  = np.sqrt(np.pi)*sc.gamma(beta-1/2)/(2*sc.gamma(beta))

        out = t1*(t2+t3)

        return out 

    def derivatives(self,R,delta=.01):
        '''
        NOTES:
            #! MUST SWITCH TO  AUTOGRAD -- hypergeometric function is problematic

        '''
        df_params  = {'_beta':self._beta}
        params     = {**df_params,**self._dm.__dict__}
        self.df    = pd.DataFrame(columns = params)
        temp       = self._dm.__dict__.copy()
        cols       = ['_a','_rho_s','_r_s' ,'_beta','_b','_c']
        self.df    = self.df[cols]
        self.sigma = self.dispersion(R)
        
        for key_value in self._dm.__dict__.items():
            if key_value[0]== '_rho_s' or key_value[0]=='_r_s':
                units = key_value[1].unit
                self._dm.__setattr__(key_value[0], np.exp(np.log(temp[key_value[0]].value)-delta)*units)
                t1 = self.dispersion(R)
                self._dm.__setattr__(key_value[0], np.exp(np.log(temp[key_value[0]].value)+delta)*units)
                t2 = self.dispersion(R)
                self._dm.__setattr__(key_value[0],temp[key_value[0]])
            else:
                self._dm.__setattr__(key_value[0], temp[key_value[0]]-delta)
                t1 = self.dispersion(R)
                self._dm.__setattr__(key_value[0], temp[key_value[0]]+delta)
                t2 = self.dispersion(R)
                self._dm.__setattr__(key_value[0],temp[key_value[0]])
            
            self.df[key_value[0]] = (t2-t1)/(2*delta)

        self.__setattr__('_beta',df_params['_beta']-delta)
        t1 = self.dispersion(R)
        self.__setattr__('_beta',df_params['_beta']+delta)
        t2 = self.dispersion(R)
        self.df['_beta'] = (t2-t1)/(2*delta)
        self.__setattr__('_beta',df_params['_beta'])

        return self.df

    def Fisher(self,R,dv):
        '''
        Notes:
            #todo: Get rid of double for loop
        '''
        # elements = self.derivatives(R)
        # self.MFisher = np.zeros((6,6))
        # for i in range(6):
        #     for j in range(6):
        #         self.MFisher[i,j] = .5*np.sum(elements[elements.columns[i]]*elements[elements.columns[j]]/((self.sigma+dev.to(u.km/u.s).value**2)**2))
        elements = self.derivatives(R).to_numpy()
        self.MFisher = .5*np.dot(elements.T/(self.sigma+dv.to(u.km/u.s).value**2)**2,elements)
        return self.MFisher

    def Covariance(self,R,dv,priors):
        F  = self.Fisher(R,dv)
        Pr =  np.diag(1/np.array(list(priors.values()))**2)
        self.Cov = np.linalg.solve(F + Pr,np.identity(6))
        self.diags = np.sqrt(np.diagonal(self.Cov))
        self.out = pd.DataFrame([self.diags],columns=[r'$\sigma_a$',r'$\sigma_{\ln\rho_s}$',r'$\sigma_{\ln{r_s}}$' ,r'$\sigma_{\beta}$',r'$\sigma_{b}$',r'$\sigma_{c}$'])
        return self.out,self.Cov