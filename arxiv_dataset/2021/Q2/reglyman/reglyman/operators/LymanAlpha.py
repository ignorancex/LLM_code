from regpy.operators import Operator
from regpy import util

from nbodykit.lab import *

import numpy as np

'''
Computes the Lyman-alpha forest from the neutral hydrogen density
Contains the Forward Operator, derivative and adjoint of Frechet-derivative

In:
->domain, codomain: domain and codomain of forward operator, a regpy.discrs object
->parameters: dictionary containing keywords describing the box and the physical state of the IGM
->redshift, redshift_spect: Redshift of pixels in domain and in codomain (need not to be the same)
->Hubble_vel: Hubble velocities of the pixels in domain and codomain (need not to be the same)
->vel_pec: peculiar velocities (if known)
->compute_vel: estimate peculiar velocities along single lines of sight (not recommended)
->flux_data: The output of the computation is the flux (if no it is the optical depth)
->gamma: The argument is the density n or (if gamma=True) the parameter gamma=log(n)
'''

    
class LymanAlphaHydrogen(Operator):
    def __init__(self, domain, parameters, redshift, redshift_spect, Hubble_vel, Hubble_vel_spect, vel_pec=None, compute_vel=False, flux_data=True, gamma=False, codomain=None):
        codomain=codomain or domain
        
        self.redshift_background=parameters["redshift_back"]
        self.N_pix=parameters["sampling"]
        self.N_space=self.N_pix[0]
        self.N_spect=self.N_pix[1]

        cosmo=cosmology.Planck15
        
        #H in km/(s*Mpc)
        H=100*cosmo.efunc(self.redshift_background)        

#       speed of light in km/s
        speed_of_light=cosmo.C
       
        #integral range in h**(-1) Mpc
        #This unit matches with output of nbodykit simulation (where distance has units h**(-1)Mpc)
        integral_range=parameters["width"]
       
        #integral range in km/s
        #The computation is only exact for very small ranges, otherwise the variation of the Hubble constant needs to be taken into account
        #Note that Hubble constant is also corrected by h**(-1), so we get rid of the dimensionless Hubble constant
        integral_range*=H/(1+self.redshift_background)
        Delta_omega=integral_range/self.N_space
        
        #Parsec in units of meters
        Parsec=3.0856775714409184*10**16
        MegaParsec=10**6*Parsec

        #I_alpha in m**2 (Effective Ly-alpha cross section for resonant scattering)
        I_alpha=4.45*10**(-22)
        
        #Compute prefactor, common to do here to avoid large float number computing later
        #Prefactor had units of km/s
        self.Prefactor=I_alpha*speed_of_light/(H*cosmo.h)*MegaParsec*Delta_omega
        self.Prefactor_adjoint=self.Prefactor*self.N_space/self.N_spect

        #EOS of the IGM
        self.beta=parameters["beta"]
        self.alpha=2-1.4*self.beta
        self.T_med=parameters["t_med"]     
        
        #tildeC in units of 1/m**3
        self.tildeC=parameters["tilde_c"]
        
        
        self.redshift=redshift
        self.redshift_spect=redshift
        self.Hubble_vel=Hubble_vel
        self.Hubble_vel_spect=Hubble_vel_spect
        
        if compute_vel:
            self.vel_pec=np.empty(self.redshift.shape)
        else:
            self.vel_pec=vel_pec
        
        #Perform sanity checks
        if compute_vel:
            assert 2*np.array(redshift.shape)==np.array(domain.shape)
        else:
            assert redshift.shape==domain.shape
        assert redshift_spect.shape==codomain.shape
        assert np.size(Hubble_vel)==self.N_space
        assert np.size(Hubble_vel_spect)==self.N_spect
        assert self.vel_pec.shape==redshift.shape

        self.compute_vel=compute_vel
        self.flux_data=flux_data
        self.gamma=gamma
        
        self.density=np.empty(self.redshift.shape)
        self.density_baryon=np.empty(self.redshift.shape)
        self.flux=np.empty(self.N_spect)
                
        super().__init__(
                domain=domain,
                codomain=codomain)

#Evaluate the forward operator
    def _eval(self, argument, differentiate, **kwargs):
        if self.compute_vel:
            x=argument[0:self.N_space]
            vel_pec=argument[self.N_space:2*self.N_space]
        else:
            vel_pec=self.vel_pec
            x=argument

        if self.gamma:
            density=np.exp(x)
        else:
            density=x
        density_baryon=self._find_baryonic_density(density)

#Computes the evaluation of the intergral operator by applying an integral kernel
        self._compute_kernel(density_baryon, vel_pec)
        tau=self._apply_kernel(density)
        if differentiate:
            self.density=density
            self.density_baryon=density_baryon
            self._compute_kernel_2(density_baryon, vel_pec)
            self._compute_kernel_3(density_baryon, vel_pec)
        if self.flux_data:
            flux=np.exp(-tau)
            if differentiate:
                self.flux=flux
            toret=flux 
        else:
            toret=tau
        return toret

#Computes the Frechet-derivative of the integral operator        
    def _derivative(self, h_data, **kwargs):
        if self.compute_vel:
            h=h_data[0:self.N_space]
            h_vel=h_data[self.N_space:2*self.N_space]
        else:
            h=h_data

        density=self.density
        density_baryon=self.density_baryon
            
        if self.gamma:
            h=density*h
                
#First term in derivative matches the evaluation applied to the direction of descent   
        d_tau_1=self._apply_kernel(h)

#Second term of derivative (inner derivative of the integration kernel)
        bar_deriv=self._hydrogen_to_baryonic_deriv(density, density_baryon, h)
        d_tau_2=self._apply_kernel_2(density_baryon, density*bar_deriv)
        if self.flux_data:
            toret=-self.flux*(d_tau_1+d_tau_2)
        else:                
            toret=d_tau_1+d_tau_2

#If the velocity is part of the vector of unknowns we also need to differentiate in the direction of the velocity                
        if self.compute_vel:
            d_tau_3=self._apply_kernel_3(density_baryon, vel_pec, density*h_vel)

            if self.flux_data:
                toret+=-self.flux*d_tau_3
            else:
                toret+=d_tau_3
                        
        return toret

#Computes the adjoint of derivative
    def _adjoint(self, g, **kwargs):
        if self.flux_data:
            y=-self.flux*g
        else:
            y=g
            
        density=self.density
        density_baryon=self.density_baryon
        
        d_rho_1=self._apply_kernel_adjoint(y)
        d_rho_2=self._apply_kernel_2_adjoint(density_baryon, y)*density
        d_rho_2=self._hydrogen_to_baryonic_adjoint(d_rho_2)
                
        if self.gamma:
            toret=density*(d_rho_1+d_rho_2)
        else:
            toret=d_rho_1+d_rho_2
                
        if self.compute_vel:
            d_rho_3=self._apply_kernel_3_adjoint(y)*density
            toret=np.concatenate(toret, d_rho_3)
        return toret
    
    def _apply_kernel(self, vector):
        toadd = self.toadd * self.Prefactor 
        toret = np.trapz(toadd*vector)
        return toret 
    
    def _compute_kernel(self, density, vel_pec):
        broadening=12.849*(self.T_med)**0.5*(density)**self.beta
        argument=np.zeros((self.N_spect, self.N_space))
        for i in range(self.N_spect):
            argument[i, :]=self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.toadd=1/(np.sqrt(np.pi)*broadening)*np.exp(-(argument/broadening)**2)          
    
    
    def _apply_kernel_adjoint(self, vector):
        toadd = self.toadd * self.Prefactor_adjoint
        toret=np.trapz(toadd.transpose()*vector)
        return toret      

    
    def _apply_kernel_2(self, density, vector):
        d_broadening=12.849*(self.T_med)**0.5*(density)**(self.beta-1)*self.beta
        toadd = self.deriv_broadening * self.Prefactor * d_broadening
        toret = np.trapz(toadd*vector)
        return toret   
    
    def _compute_kernel_2(self, density, vel_pec):
        broadening=12.849*(self.T_med)**0.5*(density)**self.beta
        argument=np.zeros((self.N_spect, self.N_space))
        for i in range(self.N_spect):
            argument[i, :]=self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.deriv_broadening=2/(np.sqrt(np.pi)*broadening**4)*np.exp(-(argument/broadening)**2)*(argument)**2-1/(np.sqrt(np.pi)*broadening**2)*np.exp(-(argument/broadening)**2)  
    
    def _apply_kernel_2_adjoint(self, density, vector):
        d_broadening=12.849*(self.T_med)**0.5*(density)**(self.beta-1)*self.beta
        toadd = self.deriv_broadening * self.Prefactor_adjoint * d_broadening
        toret=np.trapz(toadd.transpose()*vector)
        return toret     

    
    def _apply_kernel_3(self, vector):
        toadd = self.deriv_3 * self.Prefactor
        toret=np.trapz(toadd*vector)
        return toret
    
    def _compute_kernel_3(self, density, vel_pec):
        broadening=12.849*(self.T_med)**0.5*(density)**self.beta
        argument=np.zeros((self.N_spect, self.N_space))
        for i in range(self.N_spect):
            argument[i, :]=self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.deriv_3=2/(np.sqrt(np.pi)*broadening)*np.exp(-(argument/broadening)**2)*(argument/broadening)  
    
    def _apply_kernel_3_adjoint(self, vector):
        toadd = self.deriv_3 * self.Prefactor_adjoint
        toret=np.trapz(toadd.transpose()*vector)
        return toret
    
#Computes the projection of the neutral hydrogen density to the overdensity of baryonic matter and its derivative and adjoint of derivative
    def _find_baryonic_density(self, density):
        density_baryon=(density/(self.tildeC*(1+self.redshift)**6))**(1/(self.alpha))
        return density_baryon
    
    def _hydrogen_to_baryonic_deriv(self, density_hydrogen, density_baryon, h):
        return 1/self.alpha*density_baryon/density_hydrogen*h
    
    def _hydrogen_to_baryonic_adjoint(self, density_hydrogen, density_baryon, y):
        return 1/self.alpha*density_baryon/density_hydrogen*y

#Same set of functions as above, but the Ly-alpha forest is computed from the baryonic overdensity instead    
class LymanAlphaBar(Operator):
    def __init__(self, domain, parameters, redshift, redshift_spect, Hubble_vel, Hubble_vel_spect, vel_pec=None, compute_vel=False, flux_data=True, gamma=False, codomain=None):
        codomain=codomain or domain
        
        self.redshift_background=parameters["redshift_back"]
        self.N_pix=parameters["sampling"]
        self.N_space=self.N_pix[0]
        self.N_spect=self.N_pix[1]

        cosmo=cosmology.Planck15
        
        #H in km/(s*Mpc)
        H=100*cosmo.efunc(self.redshift_background)        
       
        speed_of_light=cosmo.C

        integral_range=parameters["width"]
        integral_range*=H/(1+self.redshift_background)
        Delta_omega=integral_range/self.N_space
        
        Parsec=3.0856775714409184*10**16
        MegaParsec=10**6*Parsec
        I_alpha=4.45*10**(-22)
                
        self.Prefactor=I_alpha*speed_of_light/(H*cosmo.h)*MegaParsec*Delta_omega
        self.Prefactor_adjoint=self.Prefactor*self.N_space/self.N_spect
        
        self.beta=parameters["beta"]
        self.alpha=2-1.4*self.beta
        
        self.T_med=parameters["t_med"]     
        
        #tildeC in units of 1/m**3
        self.tildeC=parameters["tilde_c"]
        
        self.redshift=redshift
        self.redshift_spect=redshift_spect
        self.Hubble_vel=Hubble_vel
        self.Hubble_vel_spect=Hubble_vel_spect
        if compute_vel:
            self.vel_pec=np.empty(self.redshift.shape)
        else:
            self.vel_pec=vel_pec
            
        #Perform sanity checks
        assert redshift.shape==domain.shape
        assert redshift_spect.shape==codomain.shape
        assert np.size(Hubble_vel)==self.N_space
        assert np.size(Hubble_vel_spect)==self.N_spect
        assert vel_pec.shape==domain.shape
        
        self.compute_vel=compute_vel
        self.flux_data=flux_data
        self.gamma=gamma
        
        self.density=np.empty(self.redshift.shape)
        self.density_baryon=np.empty(self.redshift.shape)
        self.flux=np.empty(self.redshift.shape)
        
        
        super().__init__(
               domain=domain,
               codomain=codomain)
        
#The debsity vector now holds the baryonic overdensity field        
#Now the neutral hydrogen fraction has to be found from the baryonic overdensity field
#done in the same way as in the data creation
    def _eval(self, argument, differentiate, **kwargs):
        if self.compute_vel:
            x=argument[0:self.N_space]
            vel_pec=argument[self.N_space:2*self.N_space]
        else:
            vel_pec=self.vel_pec
            x=argument

        if self.gamma:
            density_baryon=np.exp(x)
        else:
            density_baryon=x
        density_hydrogen=self._find_neutral_hydrogen_fraction(density_baryon)

        self._compute_kernel(density_baryon, vel_pec)
        tau=self._apply_kernel(density_hydrogen)
        if differentiate:
            self.density_hydrogen=density_hydrogen
            self.density_baryon=density_baryon
            self._compute_kernel_2(density_baryon, vel_pec)
            self._compute_kernel_3(density_baryon, vel_pec)
        if self.flux_data:
            flux=np.exp(-tau)
            if differentiate:
                self.flux=flux
            toret=flux 
        else:
            toret=tau
        return toret
        
    
    def _derivative(self, h_data,**kwargs):
        if self.compute_vel:
            h=h_data[0:self.N_space]
            h_vel=h_data[self.N_space:2*self.N_space]
        else:
            h=h_data

        density_hydrogen=self.density_hydrogen
        density_baryon=self.density_baryon
            
        if self.gamma:
            h=density_baryon*h
                
        hyd_deriv=self._find_neutral_hydrogen_fraction_deriv(density_baryon, h)            
        
        d_tau_1=self._apply_kernel(hyd_deriv)

        d_tau_2=self._apply_kernel_2(density_baryon, density_hydrogen*h)
        if self.flux_data:
            toret=-self.flux*(d_tau_1+d_tau_2)
        else:                
            toret=d_tau_1+d_tau_2
                
        if self.compute_vel:
            d_tau_3=self._apply_kernel_3(density_hydrogen*h_vel)

            if self.flux_data:
                toret+=-self.flux*d_tau_3
            else:
                toret+=d_tau_3
                        
        return toret

    def _adjoint(self, g,  **kwargs):
        if self.flux_data:
            y=-self.flux*g
        else:
            y=g
            
        density_hydrogen=self.density_hydrogen
        density_baryon=self.density_baryon

        d_rho_1=self._apply_kernel_adjoint(y)
        d_rho_1=self._find_neutral_hydrogen_fraction_adj(density_baryon, d_rho_1)

        d_rho_2=self._apply_kernel_2_adjoint(density_baryon, y)*density_hydrogen
                
        if self.gamma:
            toret=density_baryon*(d_rho_1+d_rho_2)
        else:
            toret=d_rho_1+d_rho_2
                
        if self.compute_vel:
            d_rho_3=self._apply_kernel_3_adjoint(y)*density_hydrogen
            toret=np.concatenate(toret, d_rho_3)
        return toret
    
    
    def _apply_kernel(self, vector):
        toadd = self.toadd * self.Prefactor 
        toret = np.trapz(toadd*vector)
        return toret 
    
    def _compute_kernel(self, density, vel_pec):
        broadening=12.849*(self.T_med)**0.5*(density)**self.beta
        argument=np.zeros((self.N_spect, self.N_space))
        for i in range(self.N_spect):
            argument[i, :]=self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.toadd=1/(np.sqrt(np.pi)*broadening)*np.exp(-(argument/broadening)**2)          
    
    
    def _apply_kernel_adjoint(self, vector):
        toadd = self.toadd * self.Prefactor_adjoint
        toret=np.trapz(toadd.transpose()*vector)
        return toret      

    
    def _apply_kernel_2(self, density, vector):
        d_broadening=12.849*(self.T_med)**0.5*(density)**(self.beta-1)*self.beta
        toadd = self.deriv_broadening * self.Prefactor * d_broadening
        toret = np.trapz(toadd*vector)
        return toret   
    
    def _compute_kernel_2(self, density, vel_pec):
        broadening=12.849*(self.T_med)**0.5*(density)**self.beta
        argument=np.zeros((self.N_spect, self.N_space))
        for i in range(self.N_spect):
            argument[i, :]=self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.deriv_broadening=2/(np.sqrt(np.pi)*broadening**4)*np.exp(-(argument/broadening)**2)*(argument)**2-1/(np.sqrt(np.pi)*broadening**2)*np.exp(-(argument/broadening)**2)  
    
    def _apply_kernel_2_adjoint(self, density, vector):
        d_broadening=12.849*(self.T_med)**0.5*(density)**(self.beta-1)*self.beta
        toadd = self.deriv_broadening * self.Prefactor_adjoint * d_broadening
        toret=np.trapz(toadd.transpose()*vector)
        return toret     

    
    def _apply_kernel_3(self, vector):
        toadd = self.deriv_3 * self.Prefactor
        toret=np.trapz(toadd*vector)
        return toret
    
    def _compute_kernel_3(self, density, vel_pec):
        broadening=12.849*(self.T_med)**0.5*(density)**self.beta
        argument=np.zeros((self.N_spect, self.N_space))
        for i in range(self.N_spect):
            argument[i, :]=self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.deriv_3=2/(np.sqrt(np.pi)*broadening)*np.exp(-(argument/broadening)**2)*(argument/broadening)  
    
    def _apply_kernel_3_adjoint(self, vector):
        toadd = self.deriv_3 * self.Prefactor_adjoint
        toret=np.trapz(toadd.transpose()*vector)
        return toret


    def _find_neutral_hydrogen_fraction(self, density_baryon):
        return self.tildeC*(1+self.redshift)**6*density_baryon**(self.alpha)
    
    def _find_neutral_hydrogen_fraction_deriv(self, density_baryon, h):
        return self.tildeC*(1+self.redshift)**6*self.alpha*density_baryon**(self.alpha-1)*h
    
    def _find_neutral_hydrogen_fraction_adj(self, density_baryon, y):
        return self.tildeC*(1+self.redshift)**6*self.alpha*density_baryon**(self.alpha-1)*y
    

        











    
    
    
