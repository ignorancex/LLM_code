#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (c)  Michele Mancarella <michele.mancarella@unimib.it>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.



## Logic for 2 point correlation function
import numpy as np
import itertools as it
import scipy.special as sc


class Signal(object):
    
    #Note: pop1=['b','f'] defines the default value for pop1, if not specified otherwise. One can also define an instance of the class with e.g. just ['b'] or just ['f']
    
    def __init__(self, bg, pert, mul_nu1, X=1, H0=1/2997.9, b1 = [0.554, 0.554], b2 = [0.783, 0.783], sB_fit=[0.953,0.691,0.180], sF_fit=[-0.105,0.010,-0.029], which_multipoles=['monopole','dipole','quadrupole','hexadecapole'], pop = ['b', 'f'], tol=1e-15, zin=10, dipole_boost=1): #For efficiency
        
        self.bg = bg
        self.pert = pert
        self.tol = tol
        
        self.b1B = b1[0]
        self.b1F = b1[1]
        self.b2B = b2[0]
        self.b2F = b2[1]
        
        self.sB_a=sB_fit[0]
        self.sB_b=sB_fit[1]
        self.sB_c=sB_fit[2]
        self.sF_a=sF_fit[0]
        self.sF_b=sF_fit[1]
        self.sF_c=sF_fit[2]
        
        self.which_multipoles = which_multipoles
        self.pop = pop
        self.X=X      #Deviation of amount of DM in a galaxy w-r-to cosmic average. NOW INCLUDED IN PERTURBATIONS, POSSIBLY REMOVE IT HERE
        
        self.H0=H0    
        # This is H_0 in units of h*c/Mpc. The numerical value is simply equal to 100*10^3/c (the factor 100 from the definition of small h, the factor 10^3 to convert km to m, and c to convert m/s to units of c). The value of h does not enter
        
        self.mu0=mul_nu1[0]
        self.mu2=mul_nu1[1] 
        self.mu4=mul_nu1[2]
        self.nu1=mul_nu1[3]

        self.zin=zin

        self.dipole_boost = dipole_boost
        
        self._multipoles= {'monopole': self._monopole,
                           'dipole': self._dipole, 
                           'quadrupole': self._quadrupole,
                           'hexadecapole': self._hexadecapole} # Specifies a list of functions (defined below)
        
     # calculate_signal(...) appends all multipoles (with all possible combinations of populations) into one list
    
    def calculate_signal(self, d, z): # Note that we cannot pass a list in both d and z because in the definition of the multipoles we multiply mu(d) or nu(d) x the redshift-dependent part.
        
        for i, m in enumerate(self.which_multipoles): 
            if i==0: 
                signal = self._multipoles[m](d,z) # _multipoles[m](d,z) is a list, containing the m-th multipole for all pairs of populations at separation d and redshift z 
                
            else:
                mul_ = self._multipoles[m](d,z)
                signal += mul_ # appends the next multipole to the list
        
        return signal # Returns a list of 8 numpy arrays corresponding to the multipoles computed for the given separations. 
        

        
    def _monopole(self, d, z, which_comb='default'): #Needs to be multiplied by mu_0
                  #default value of which_comb is [['b','b'],['b','f'],['f','f']]; You can specify it otherwise if you e.g. just want [['b','f']]
        if which_comb=='default':

            which_comb=list(it.combinations_with_replacement(self.pop,2)) #NG: I'm doing it this way because directly defining something with "self" as a default value does not work
        
        zin=self.zin #Redshift zin at which the functions P_in, mu_0, mu_2, mu_4, nu_1 are defined
        T=self.pert.T_m(-np.log(1+z))/self.pert.T_m(-np.log(1+zin)) #Make sure that this is normalized at zin!!!
        
        mu0=self.mu0(d)
        
        #print('monopole mu0: %s' %mu0)
        
        monopole=[]
        for i, comb in enumerate(which_comb):
            bpop1= self.galaxybias(comb[0], z)*T #bias of which_comb[i][0], multiplied by transfer function
            bpop2= self.galaxybias(comb[1], z)*T #bias of which_comb[i][1], multiplied by transfer function
            f=self.pert.f_eff(-np.log(1+z))*T #f_eff, multiplied by transfer function
            
            #print('monopole f_eff for comb %s: %s' %(comb, f))
            
            monopole += [mu0*(bpop1*bpop2+1/3*(bpop1+bpop2)*f+1/5*f**2)]  
            
        #print('monopole : %s' %(monopole))

        return monopole
    
    
    def _dipole(self, d, z, which_comb='default'):
        dipole_nowide=self._dipole_nowide(d, z, which_comb=which_comb)
        dipole_onlywide=self._dipole_onlywide(d, z, which_comb=which_comb)
        return  [ (x + y)*self.dipole_boost for x, y in zip(dipole_nowide, dipole_onlywide)]
    
    def _dipole_nowide(self, d, z, which_comb='default'): 
        if which_comb=='default':
            which_comb=list(it.combinations(self.pop,2)) #Only allowing combinations of *distinct* populations for the dipole
        a=1/(1+z)
        x=np.log(a)
        Om0=self.bg.Om0
        r = self.bg.da(z)/a #comoving distance to z,  without H_0 . 
        #print('dipole r: %s' %r)
        
        hdot=1+self.bg.zeta(x) # This is the *conformal* time derivative of H/H^2, with H being the *conformal* Hubble parameter 
        
        Gamma=self.pert.Gamma(x)
        mu=self.pert.mu_Psi(x)
        Theta=self.bg.Theta(x)
        h=self.bg.h(x)      #Defined h here as the other parameters to simplify
        #print('dipole h: %s' %h)
        zin=self.zin #Redshift zin at which the functions P_in, mu_0, mu_2, mu_4, nu_1 are defined
        T=self.pert.T_m(-np.log(1+z))/self.pert.T_m(-np.log(1+zin)) #Make sure that this is normalized at zin!!!
        f=self.pert.f_eff(x)*T #f_eff multiplied by transfer function
        
        nu1=d*self.H0*self.nu1(d) #Factor d*H0 necessary because the file does not contain it

        xc=self.X*self.bg.oc(0) #Fraction of DM in a galaxy

        dipole_nowide=[] 
        for i, comb in enumerate(which_comb):
            bpop1= self.galaxybias(comb[0], z)*T #galaxy bias of which_comb[i][0], multiplied by transfer function
            bpop2= self.galaxybias(comb[1], z)*T #galaxy bias of which_comb[i][1], multiplied by transfer function
            
            spop1= self.magbias(comb[0], z) #magnification bias of which_comb[i][0] 
            spop2= self.magbias(comb[1], z) #magnification bias of which_comb[i][1] 
            
            Deltab=bpop1-bpop2
            Deltas=spop1-spop2
            
            dipole_nowideA = a*h*f*(5*(bpop1*spop2-bpop2*spop1)*(1-1/(r*h*a))+Deltab*(2/(r*h*a)+hdot)) #tested
            dipole_nowideB = a*h*3*f**2*(-Deltas)*(1-1/(r*h*a)) #tested
            dipole_nowideC = np.real(a*h*Deltab*xc*(Theta*f-3/2*Om0/(a**3*h**2)*Gamma*mu*T)) #NEEDS TO BE TESTED! #Gamma, mu0, Theta might develop small imaginary part depending on the input values for alpha_M, alpha_B, beta_gamma
            dipole_nowide += [(dipole_nowideA+dipole_nowideB+dipole_nowideC)*nu1]
        return dipole_nowide
    
    
    
    def _dipole_onlywide(self, d, z, which_comb='default'): #Needs to be multiplied by d*mu_2
        if which_comb=='default':

            which_comb=list(it.combinations(self.pop,2))
            
        a=1/(1+z)
        x=np.log(a)
        
        zin=self.zin #Redshift zin at which the functions P_in, mu_0, mu_2, mu_4, nu_1 are defined
        T=self.pert.T_m(-np.log(1+z))/self.pert.T_m(-np.log(1+zin)) #Make sure that this is normalized at zin!!!
        f=self.pert.f_eff(x)*T #f_eff multiplied by transfer function
        #Om0=self.bg.Om0
        r=self.bg.da(z)/a #comoving distance to z, multiplied by H_0 

        mu2=self.mu2(d) 
        
        dipole_onlywide=[] 
        for i, comb in enumerate(which_comb):
            bpop1= self.galaxybias(comb[0], z)*T #galaxy bias of which_comb[i][0], multiplied by transfer function
            bpop2= self.galaxybias(comb[1], z)*T #galaxy bias of which_comb[i][1], multiplied by transfer function  
            Deltab=bpop1-bpop2
            
            dipole_onlywide += [-2/5*Deltab*f*self.H0/r*d*mu2] 
        return dipole_onlywide
      
        

    def _quadrupole(self, d, z, which_comb='default'): 
        
        a=1/(1+z)
        x=np.log(a)

        if which_comb=='default':
               which_comb=list(it.combinations_with_replacement(self.pop,2))
        
        zin=self.zin #Redshift zin at which the functions P_in, mu_0, mu_2, mu_4, nu_1 are defined
        T=self.pert.T_m(-np.log(1+z))/self.pert.T_m(-np.log(1+zin)) #Make sure that this is normalized at zin!!!
        f=self.pert.f_eff(x)*T #f_eff multiplied by transfer function 

        mu2=self.mu2(d)

        quadrupole=[]
        for i, comb in enumerate(which_comb):
            bpop1= self.galaxybias(comb[0], z)*T #bias of which_comb[i][0], multiplied by transfer function
            bpop2= self.galaxybias(comb[1], z)*T #bias of which_comb[i][1], multiplied by transfer function
            quadrupole += [-(2/3*(bpop1+bpop2)*f+4/7*f**2)*mu2] #NOTE: f_eff not written yet.  
        return quadrupole
    
    
    
    def _hexadecapole(self, d, z): #Needs to be multiplied by mu_4

        a=1/(1+z)               #DSB: added this here. It was complaining that x was not defined.
        x=np.log(a)
        
        zin=self.zin #Redshift zin at which the functions P_in, mu_0, mu_2, mu_4, nu_1 are defined
        T=self.pert.T_m(-np.log(1+z))/self.pert.T_m(-np.log(1+zin)) #Make sure that this is normalized at zin!!!
        f=self.pert.f_eff(x)*T #f_eff multiplied by transfer function 
        
        mu4=self.mu4(d)
        
        hexadecapole = [8/35*f**2*mu4]
        return hexadecapole
    
    def galaxybias(self, pop, z, Deltab=1): 
        if pop=='b':
                  return self.b1B*np.exp(self.b2B*z)+Deltab/2
        else:
                  return self.b1F*np.exp(self.b2F*z)-Deltab/2
            
    def magbias(self, pop, z): #3-parameter fit with a polylogarithmic function
        if pop=='b':
            magbias=self.sB_a+self.sB_b*np.log(z)+self.sB_c*np.log(z)**2 
            return magbias 
        else:
            magbias=self.sF_a+self.sF_b*np.log(z)+self.sF_c*np.log(z)**2  
            return magbias 
    
    
    
    
    
    
    
    
