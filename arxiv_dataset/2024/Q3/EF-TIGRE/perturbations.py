#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#    Copyright (c)  Michele Mancarella <michele.mancarella@unimib.it>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


## Logic for perturbations
import numpy as np
import os
from scipy.integrate import solve_ivp, quad
from scipy import interpolate
from mul_nu1 import interpolate_Pin

class Perturbations(object):
    
    def __init__(self, bg, zin=10, tol=1e-15, X=1.0, bounds_error=False):
        
        self.bg = bg
        self.tol = tol
        self.zin=zin
        self.xin=-np.log(1+self.zin) #Initial x=log(a)
        self.X=X
        self.bounds_error = bounds_error

        if type(bg).__name__ == 'BackgroundBetaGamma':
            self.is_beta_gamma = True
        elif type(bg).__name__ == 'BackgroundGammac':
            self.is_beta_gamma = False
        else:
            raise ValueError('Wrong background class name passed.')


    def _find_init_conditions(self, verbose=False):
        
        if self.is_beta_gamma:

            bg2=self.bg.EFTfuncts.betagamma**2 #betagamma2 parameter. In this parametrization, betagamma2 is constant
        
            Obin=self.bg.Ob(self.xin) # Capital Omega baryons
            Ocin=self.bg.Oc(self.xin) # Capital Omega cdm
            
            bias=1/(2*Ocin)*(-Obin+Ocin+bg2*Ocin+np.sqrt(4*Obin*Ocin+(Obin-(1+bg2)*Ocin)**2))
        
            # Compute T_b(xin), T_b_prime(xin)
            # Solution to Eq.(4.11) in 1509.02191, normalized at zin

            self.T_bin = 1.
            self.T_b_primein = (-1+np.sqrt(25+12*bg2*(-Obin+Ocin+bg2*Ocin+np.sqrt(4*Obin*Ocin+(Obin-(1+bg2)*Ocin)**2))))/4           

            #Then T_c(xin), T_c_prime(xin) = bias(xin)* (T_b(xin), T_b_prime(xin))
            
            self.T_cin=bias*self.T_bin
            self.T_c_primein=bias*self.T_b_primein
            
        else:

            #  In this case all modifications go to zero at initial time, so the I.C. are the same al LCDM
            
            self.T_bin = 1.
            self.T_b_primein = 1.
            

            self.T_cin=1.
            self.T_c_primein=1.


     
    def _dY_dx(self, x, Y):
        
        '''
        Differential equation for the perturbations.
        Y = (T_b, T_c, T_b', T_c')
        Returns a vector Y' = (Z_b, Z_c, Z_b', Z_c') for the integrator

        
        
        ''' 

        if type(self.bg).__name__ == 'BackgroundBetaGamma':
            raise NotImplementedError()
        else:

            # NOTE that the implementation here is specific for the time parametrization
            # assumed in the paper, for efficiency. Needs to be changed if using a different one!

            cs2al = self.bg.cs2al(x)
    
            OmL = self.bg.Om_LCDM(x)
            OmR = self.bg.Or_LCDM(x)
    
            OL_LCDM = 1-OmL-OmR
            OL0 = (1-self.bg.Om0-self.bg.Or0) 
    
            OL_ov_Ol0 = OL_LCDM/OL0
            OL_ov_Ol0_sq = (OL_ov_Ol0)**2
            
    
            c0 = OL_ov_Ol0*(self.bg.EFTfuncts.alphaB0-self.bg.EFTfuncts.alphaM0+3*self.bg.EFTfuncts.gammac0)
    
            bxi_plus_bg_sq = np.where( np.abs(c0)<self.tol, 0, (c0**2)*2/cs2al )
            
            c1 = OL_ov_Ol0*(self.bg.EFTfuncts.alphaB0-self.bg.EFTfuncts.alphaM0)
    
            bxisq =  np.where( np.abs(c1)<self.tol, 0, 2/cs2al*(c1**2) )
    
    
            bxi_sq_plus_bxi_bg = np.where( (np.abs(c0)<self.tol) | (np.abs(c1)<self.tol) , 0, 2/cs2al*c1*c0 )
            
    
            zeta = self.bg.zeta(x)
            gc = self.bg.gammac(x)
            
            ob = self.bg.ob(x)
            oc = 1-ob 
            Om = self.bg.Om(x)
    
            betabb = 1 + bxisq        
            betabc = 1 + bxi_sq_plus_bxi_bg 
            betacc = 1 + bxi_plus_bg_sq 
            
            T_b, T_c, Z_b, Z_c = Y[0], Y[1], Y[2], Y[3]
            Z_bprime = - (2+zeta) * Z_b + 3./2. * Om * (betabb * ob * T_b + betabc * oc * T_c)
            Z_cprime = - (2+zeta + 3 * gc ) * Z_c + 3./2. * Om * (betabc * ob * T_b + betacc * oc * T_c)
    
            return [Z_b, Z_c, Z_bprime, Z_cprime]
    
    
    def solve(self, res=None, method='RK45', rtol=1e-3, atol=1e-6, verbose=False):
        '''
        Solves differential equation for the perturbations. Note that the solutions are normalized at zin
        '''

         #Tb(xin), Tc(xin), Tb'(xin), Tc'(xin) 
        self._find_init_conditions( verbose=verbose)
        if verbose:
            print('Initial conditions: Tb(xin)=%s, Tc(xin)=%s, Tbprime(xin)=%s, Tcprime(xin)=%s' %(self.T_bin, self.T_cin, self.T_b_primein, self.T_c_primein))
        xspan = (self.xin, 0) # interval
        if res is not None:
            #xpoints = np.linspace(self.xin, 0, res ) # grid to evaluate solutions. Pass as t_eval if needed
            xpoints = np.sort( np.unique( np.concatenate( [np.linspace(self.xin, -1, int(res/2) ), np.geomspace(-1, -1e-10, int(res/2) ), np.array([0]) ])))
        else:
            xpoints=None
        Y0 = [self.T_bin, self.T_cin, self.T_b_primein, self.T_c_primein] # initial conditions

        #Solver
        self.sol =  solve_ivp(self._dY_dx, xspan, Y0, method=method, t_eval=xpoints, dense_output=False, events=None, vectorized=False, args=None, rtol=rtol, atol=atol)
        
        self.s1 = interpolate.interp1d( self.sol.t,self.sol.y[0], kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)
        self.s2 = interpolate.interp1d( self.sol.t,self.sol.y[1], kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)
        self.s3 = interpolate.interp1d( self.sol.t,self.sol.y[2], kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)
        self.s4 = interpolate.interp1d( self.sol.t,self.sol.y[3], kind='cubic', bounds_error=self.bounds_error, fill_value=(np.NaN, np.NaN), assume_sorted=False)
    
    
    # Define functions needed to compute the signal, after running self.solve()
    
    
    def T_b(self, x):
        return self.s1(x)
    
    def T_c(self, x): 
        return self.s2(x)
    
    
    def T_b_prime(self, x):
        return self.s3(x)
    
    def T_c_prime(self, x): 
        return self.s4(x)
    
    def T_m(self, x): 
        ''' Matter transfer function'''

        ob =  self.bg.ob(x)
        oc=1-ob
        return (self.s2(x)*oc+self.s1(x)*ob)  
    
    def bc(self, x): 
        ''' Dark matter bias '''
        ob =  self.bg.ob(x)
        oc=1-ob
        return self.s2(x)/(self.s2(x)*oc+self.s1(x)*ob) 
    
    def bb(self, x): 
        ''' Baryons  bias '''
        ob =  self.bg.ob(x)
        oc=1-ob
        return self.s1(x)/(self.s2(x)*oc+self.s1(x)*ob)
    
    def f_eff(self, x): #First order derivative of T_m w.r.t x=log(a); Argument needs to be between xin (z=zin) and 0 (z=0) 

            
        ob =  self.bg.ob(x)
        oc=1-ob
    
        return (self.s3(x) * (self.X *ob + 1 - self.X) + self.X * self.s4(x) * oc)/(self.s2(x)*oc+self.s1(x)*ob) 
   
    # Calculating Gamma(x) and mu_Psi(x) 

    def beta_gamma(self, x):
        
        cs2al = self.bg.cs_sqrtal(x)
        
        if self.is_beta_gamma:
            res= np.full(np.array(x).shape,self.bg.EFTfuncts.betagamma)
        else:
            res=np.where( np.abs(self.bg.gammac(x))<self.tol, 0, np.where( np.abs(cs2al)>self.tol, 3*np.sqrt(2)*self.bg.gammac(x)/cs2al, np.inf))
        return res
    
    
    def beta_xi(self, x):
        
        cs2al = self.bg.cs_sqrtal(x)
        xi = self.bg.xi(x)
        
        res= np.where( np.abs(xi)<self.tol, 0, np.where(np.abs(cs2al)>self.tol, np.sqrt(2)*xi/cs2al, np.inf))
        return res
        
        
    def Gamma(self, x):
        return np.where( np.abs(self.mu_Psi(x))>self.tol, np.real(self.beta_gamma(x)*(self.beta_xi(x)+self.beta_gamma(x)*self.bc(x)*self.bg.oc(x))/self.mu_Psi(x)), 0)
        
    def mu_Psi(self, x):
        return np.abs(1+self.beta_xi(x)*(self.beta_xi(x)+self.beta_gamma(x)*self.bc(x)*self.bg.Oc0/self.bg.Om0))
    
    #Note: Theta already contained in background

    def beta_B(self, x): #relevant for lensing
    
        alphaB = self.bg.EFTfuncts.alphaB( x,self.bg.Om_LCDM(x), self.bg.Or_LCDM(x), self.bg.Om0, self.bg.Or0)
        
        return np.where(np.abs(self.bg.cs_sqrtal(x))>self.tol, np.sqrt(2)*alphaB/self.bg.cs_sqrtal(x), 0)
    
    def beta_Lens(self, x): #relevant for lensing
        return 1+(self.beta_B(x)+self.beta_xi(x))*(self.beta_xi(x)+self.beta_gamma(x)*self.bc(x)*self.bg.Oc0/self.bg.Om0)
    
    
    