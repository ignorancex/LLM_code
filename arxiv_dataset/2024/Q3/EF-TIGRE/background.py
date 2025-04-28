#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (c)  Michele Mancarella <michele.mancarella@unimib.it>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.



## Background quantities
import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate


from EFTfunctions import regulator


def Efun(x, Om, w, Or ):
    # This is H/H_0, or h, i a wcdm flat model
    return np.sqrt( Om*np.exp(-3*x) +(1-Om-Or)*np.exp(-3*x*(1+w))+Or*np.exp(-4*x) )



class BackgroundBetaGamma(object):
    
    def __init__(self, Om0, Ob0, w, EFTfuncts, zin=10, tol=1e-15, bounds_error=False):    
        pass
    

class BackgroundGammac(object):
    
    def __init__(self, Om0, Ob0, w,  EFTfuncts, zin=10, tol=1e-15, bounds_error=False, Or0=0., xin_cs=None):
        
        self.Ob0 = Ob0
        self.Om0 = Om0
        self.wDE = w
        self.Oc0 = self.Om0 - self.Ob0 
        self.tol=tol
        self.Or0=Or0
        
        # Assume flatness and no radiation
        self.ODE0 = 1-Om0
        
        self.EFTfuncts = EFTfuncts
        
        self.bounds_error = bounds_error
        
        self.zin=zin
        self.xin = -np.log(1+zin)
        
        if xin_cs==None:
            self.xin_cs=self.xin
        else:
            self.xin_cs=xin_cs

    

    def h_LCDM(self,x, ):
        '''
        this h = H/H0 in LCDM 
        '''
        return Efun(x, self.Om0, -1., self.Or0)
    

    def zeta_LCDM(self, x, OmL=None, wDE=-1.):
        
        '''
        this is zeta = h'/h = 1/H^2 * dH/dt
        '''
        
        if OmL is None:
            OmL = self.Om_LCDM(x)
                    
        Og = self.Or_LCDM(x)
        
        val = -3/2* ( OmL+4*Og/3 ) # OK

        return val 

    
    def eta(self, x, zeta=None):
    
        '''
        this is eta = 1+2/3 zeta
        '''
        if zeta is None:
            zeta=self.zeta(x)
    
        return 1+2*zeta/3
    
    
    def Om_LCDM(self, x, ):
    
        '''
        Evolution of Omega_m in a LCDM bg
        '''

        
        return self.Om0*np.exp(-3*x)/self.h_LCDM(x)**2
    
    
    def Or_LCDM(self, x, ):
        return self.Or0*np.exp(-4*x)/self.h_LCDM(x)**2
        
        
    
    def _Omfun(self, x, g ):
        
        '''
        Evolution of Omega_m in arbitrary background.
        g is defined as the ratio between the solution and the evolution in LCDM
        '''
        
        return g*self.Om_LCDM(x)
    
    
    def _Ocfun(self, x, g, Ob):
        '''
        CDM density fraction
        '''
        
        return  self._Omfun(x, g )-Ob 
    
    
    def _find_init_conditions(self, verbose=False):
        
        '''
        Find initial conditions to preserve the CMB constraint; needed if w!=-1
        
        Should return Om0, ODE0,  Omin, ODEin, Ocin, Obin
        
        '''
        

        
        hsqin = self.h_LCDM(self.xin)**2
        
        self.Omin = self.Om0*np.exp(-3*self.xin)/(hsqin)
        self.Orin = self.Or0*np.exp(-4*self.xin)/(hsqin)
        
        self.ODEin = 1-self.Omin-self.Orin #self.ODE0/(self.h(self.xin)**2)
        
        self.Ocin = self.Omin*self.Oc0/(self.Ob0+self.Oc0)
        self.Obin = self.Omin*self.Ob0/(self.Ob0+self.Oc0)
        
      
    
    
    def _dY_dx(self, x, Y):
        
        '''
        Differential equation for the background.
        Y = (Ob, g)
        Returns a vector (Ob', g') for the integrator
        
        '''

        # values with modifications
        Ob, g, Or = Y[0], Y[1], Y[2]
        Omval = self._Omfun(x, g)
        Oc = Omval - Ob
        Ode = 1 - Omval- Or

        # values in LCDM
        OLcdm = self.Om_LCDM(x)
        OrLcdm = self.Or_LCDM(x)
        
        # EFT functions
        gammac = self.EFTfuncts.gammac(x, OLcdm, OrLcdm, self.Om0, self.Or0)
        alM  = self.EFTfuncts.alphaM(x, OLcdm, OrLcdm, self.Om0, self.Or0)

        zeta = - (3/2) * ( 1+ self.wDE*Ode + Or/3  ) 
        # Need to write zeta in terms of Ob and g to solve the differential equations. 
        
        Obprime = -Ob*( 3 + alM + 2*zeta )
        gprime = 3*gammac * ( Oc /OLcdm ) - alM*g
        
        Orprime = -Or*(4+2*zeta+alM)
        
        return Obprime, gprime, Orprime
    
    
 
    
    def solve(self,res=None, method='RK45', rtol=1e-3, atol=1e-6, H_int='backward', verbose=False):
        
        '''
        Solves differential equation for the background.
        
        From scipy:
            
            rtol, atolfloat or array_like, optional
            Relative and absolute tolerances. The solver keeps the local error estimates less than 
            atol + rtol * abs(y). Here rtol controls a relative accuracy (number of correct digits),
            while atol controls absolute accuracy (number of correct decimal places). 
            To achieve the desired rtol, set atol to be lower than the lowest value that can be 
            expected from rtol * abs(y) so that rtol dominates the allowable error. 
            If atol is larger than rtol * abs(y) the number of correct digits is not guaranteed. 
            Conversely, to achieve the desired atol set rtol such that rtol * abs(y) is always 
            lower than atol. If components of y have different scales, it might be beneficial to 
            set different atol values for different components by passing array_like with shape (n,)
            for atol. Default values are 1e-3 for rtol and 1e-6 for atol.
        
        '''
        
    
        
        self._find_init_conditions( verbose=verbose)
        if verbose:
            print('Initial conditions: Ob=%s, g=%s' %(self.Obin, 1))
        xspan = ( self.xin, 0) # interval
        if res is not None:
            # grid to evaluate solutions. Pass as t_eval if needed
            #xpoints = np.linspace(self.xin, 0, res ) 
            #zgrid = np.geomspace(1e-6, self.zin, res) 
            #xpoints = -np.log(1+zgrid)[::-1]
            self.xpoints = np.sort( np.unique( np.concatenate( [np.linspace(self.xin, -1, int(res/2) ), np.geomspace(-1, -1e-10, int(res/2) ), np.array([0]) ])))


        else:
            self.xpoints=None
        Y0 = [self.Obin, 1, self.Orin] # initial condition
        
        
        self.sol =  solve_ivp( self._dY_dx, xspan, Y0, method=method, t_eval=self.xpoints, dense_output=False, events=None, vectorized=False, args=None, rtol=rtol, atol=atol)
   
        
        self.s1 = interpolate.interp1d( self.sol.t, self.sol.y[0], kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)
        self.s2 = interpolate.interp1d( self.sol.t, self.sol.y[1], kind='cubic', bounds_error=self.bounds_error, fill_value=( np.NaN ,np.NaN), assume_sorted=False)
        self.s3 = interpolate.interp1d( self.sol.t, self.sol.y[2], kind='cubic', bounds_error=self.bounds_error, fill_value=( np.NaN ,np.NaN), assume_sorted=False)
        
        self.Hubbleint(H_int=H_int)
        self.angdist()
    
    def is_stable(self, res=200):
        
        xx = np.linspace( self.xin_cs, 0, res )
        
        cs2al = regulator( self.cs2al(xx), self.tol)
        #print(cs2al)
        
        return np.all( cs2al>=0 )#  & (cs2al<=1) )

    
    def is_X_physical(self, X, res=200):

        xx = np.linspace( self.xin, 0, res )

        ob_ov_oc = self.s1(xx)/(1-self.s1(xx))

        return np.all( X<=1+ob_ov_oc )
  
    
    # Background solutions. To be called only after bg equations have been solved !
    def Ob(self, x):
        return self.s1(x)
        
    def Or(self, x):
        return self.s3(x)
    
    def g(self, x):
        return self.s2(x)
            
    def Om(self, x):
        return self._Omfun(x, self.s2(x) )
    
    def Oc(self, x):
        return self._Omfun(x, self.s2(x))-self.s1(x)  
    
    def ob(self, x):
        return self.s1(x)/self._Omfun(x, self.s2(x) )
    
    def oc(self, x):
        return 1-self.s1(x)/self._Omfun(x, self.s2(x) )
    
    def ODE(self, x):
        return 1-self._Omfun(x, self.s2(x) )-self.s3(x)

    def zeta(self, x, OmL=None, Om=None, Or=None, OrL=None):
        
        '''
        this is zeta = h'/h = 1/H^2 * dH/dt
        '''
        
        #return num/den
        if Om is None:
            Om = self._Omfun( x, self.s2(x) )
        if OmL is None:
            OmL=self.Om_LCDM(x)
        if Or is None:
            Or = self.Or(x)
        if OrL is None:
            OrL = self.Or_LCDM(x)
            
        gammac = self.EFTfuncts.gammac(x, OmL, OrL, self.Om0, self.Or0)
            
        OD = (1-Om-Or) 

        val = - (3/2) * ( 1+OD*self.wDE+Or/3  )       

        return val
    
    def cs_sqrtal(self, x):
        return np.sqrt(self.cs2al(x))
    
    def cs2al(self, x):
        
        OmL = self.Om_LCDM(x)
        OmR = self.Or_LCDM(x)
        
        aB = self.EFTfuncts.alphaB(x, OmL, OmR, self.Om0, self.Or0)
        aM = self.EFTfuncts.alphaM(x, OmL, OmR, self.Om0, self.Or0)
        
        Om = self._Omfun(x, self.s2(x) )
        Or = self.Or(x)
            
        zeta = self.zeta(x, OmL=OmL, Om=Om, Or=Or, OrL=OmR)

        zetaL = self.zeta_LCDM(x, OmL=OmL)

        alB_prime = self.EFTfuncts.alphaB0/(1-self.Om0-self.Or0)*( 2*zetaL*(OmL+OmR)+3*OmL+4*OmR )

        res = -2*( (1+aB)*(zeta-aM+aB) + alB_prime + 3*Om/2+2*Or )

            
        return res
    
    def xi(self, x):
        OmL = self.Om_LCDM(x)
        OmR = self.Or_LCDM(x)
        return -self.EFTfuncts.alphaM(x, OmL, OmR, self.Om0, self.Or0)+self.EFTfuncts.alphaB(x, OmL, OmR, self.Om0, self.Or0)
    
    def gammac(self, x, OmL=None, OmR=None):
        if OmL==None:
            OmL = self.Om_LCDM(x)
        if OmR==None:
            OmR = self.Or_LCDM(x)          
        return self.EFTfuncts.gammac(x, OmL, OmR, self.Om0, self.Or0)
    
    def Theta(self, x):
        return np.real(3*self.gammac(x))
    

    
    def h(self, x) :
        return self.hinterp(x) 
    
    def da(self, z) :  
        return self.dainterp(z)
   
    def dc(self, z) :  
        return self.dcinterp(z) 
    
    def dc_LCDM(self, z) :  
        return self.dcinterp_LCDM(z) 
    
    
    
    def angdist(self, res=500):
        '''
        Dimensionless angular diameter distance to redshift z (i.e. without the factor c/H0 in front )
        
        d_a = d_c/(1+z)
        
        To be evaluated only after solving for the bg !
        '''
        
        self.zgrid_da = np.exp(-self.xpoints)-1 
        dagrid = []
        
        for z in self.zgrid_da:
            zz = np.linspace(0, z, res)
            Evals = self.h(-np.log(1+zz))  
            dagrid.append(np.trapz(1/Evals, zz)/(1+z))
        self.dagrid = np.array(dagrid)
        
        self.dainterp = interpolate.interp1d( self.zgrid_da, self.dagrid, kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)

    
    def comdist(self, res=500):
        '''
        Dimensionless comoving distance to redshift z (i.e. without the factor c/H0 in front )
                
        To be evaluated only after solving for the bg !
        '''
        
        self.zgrid_dc = np.exp(-self.xpoints)-1 
        dcgrid = []
        dcgrid_LCDM = []
        
        for z in self.zgrid_dc:
            zz = np.linspace(0, z, res)
            Evals = self.h(-np.log(1+zz))  
            Evals_LCDM = self.h_LCDM(-np.log(1+zz))
            dcgrid.append(np.trapz(1/Evals, zz))
            dcgrid_LCDM.append(np.trapz(1/Evals_LCDM, zz))
            
        self.dcgrid = np.array(dcgrid)
        self.dcgrid_LCDM = np.array(dcgrid_LCDM)
        
        self.dcinterp = interpolate.interp1d( self.zgrid_dc, self.dcgrid, kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)
        self.dcinterp_LCDM = interpolate.interp1d( self.zgrid_dc, self.dcgrid_LCDM, kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)

    def Hubbleint(self, H_int = 'backward', res=500):
        '''
        Hubble parameter obtained integrating zeta.
        To be evaluated only after solving for the bg !
        '''
        self.xgrid_Hubble = self.xpoints 
        hgrid = []
        
        for x in self.xgrid_Hubble:
        
        
            if H_int == 'backward':
    
                xx = np.linspace(0, x, res)
                ivals = self.zeta(xx)
    
                integral = np.trapz(ivals, xx)
                h = np.exp(integral)
                #
            elif H_int == 'forward': 
                hin = Efun(self.xin, self.Om0, -1., self.Or0) # Initial value of H, obtained by evolving H0 backwards with LambdaCDM. Here we use the dimensionless quantity, without the H0 in front
    
                xx = np.linspace(self.xin, x, res)
                ivals = self.zeta(xx)
    
                integral = np.trapz(ivals, xx)
                h = hin * np.exp(integral)
                
            else:
                raise ValueError('H_int can be forward or backward')
            
            hgrid.append(h)
        
        # Interpolate the integral for fast evaluation
        self.hgrid = np.array(hgrid)
        
        self.hinterp = interpolate.interp1d( self.xgrid_Hubble, self.hgrid, kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)
        
        
    
    
