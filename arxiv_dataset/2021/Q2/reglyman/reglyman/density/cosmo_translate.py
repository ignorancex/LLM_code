from regpy.util import classlogger

import astropy.units as u
import astropy.cosmology as ac

import numpy as np
from scipy import interpolate

import logging

#This module computes cosmological translations
#The comoving distance is first translated into redshift and then into Hubble-velocity
#The computation is done in the astropy-package
#WARNING: As long the data are created using nbodykit, a change of units between astropy and nbodykit needs to be taken into account
#nbodykit: Mpc/h
#astropy: Mpc

class Cosmo_Translate():
    log=classlogger
    def __init__(self, comoving, chosen_cosmo, cosmo=None):
        if chosen_cosmo=='Planck15':
            self.cosmo=ac.Planck15
        elif chosen_cosmo=='UserDefined':
            self.cosmo=cosmo.to_astropy()
        else:
            print(chosen_cosmo, 'not implemented right now')
            print('Continue with Planck15 cosmology instead')
            self.cosmo = ac.Planck15  

        #Translate comoving distances into redshifts
        #x=int_0^z c dz/H(z)
        self.redshift=np.zeros(np.size(comoving))
        for i in range(np.size(comoving)):
            self.redshift[i]=ac.z_at_value(self.cosmo.comoving_distance, comoving[i]*u.Mpc/self.cosmo._h)
        self.comoving=comoving
        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Cosmology Calculator initialized')
    
    #Translates redshifts into Hubble-flow in km/s
    #v_H=H(z) proper_distance=H(z) a(z) x    
    def Compute_Hubble_Velocity(self):
        Hubble_vel=self.cosmo.H(self.redshift)*self.cosmo.scale_factor(self.redshift)*self.comoving/self.cosmo._h*u.Mpc*u.s/u.km
        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Hubble velocities computed')        
        return Hubble_vel
    
    def Compute_Redshift(self):
        return self.redshift

    def Compute_Pec_Vel(self, hubble, vel):
        toret=np.zeros(vel.shape)
        for i in range(vel.shape[0]):
            flow=hubble+vel[i, :]
            comoving=flow/(self.cosmo.H(self.redshift)*self.cosmo.scale_factor(self.redshift)*u.Mpc*u.s/u.km)
            flow_redshift=np.zeros(np.size(comoving))
            for j in range(np.size(comoving)):
                flow_redshift[j]=ac.z_at_value(self.cosmo.comoving_distance, comoving[j]*u.Mpc)-self.redshift[j]
            toret[i, :]=3*10**5*flow_redshift/self.redshift
        return toret

    #If the data should be given in equal size bins in Hubble-velocities, then an interpolation is needed
    def Rebin_all(self, density, vel, Hubble_vel, N_new):
        inter_dens=interpolate.interp1d(Hubble_vel, density)
        inter_com=interpolate.interp1d(Hubble_vel, self.comoving)
        inter_red=interpolate.interp1d(Hubble_vel, self.redshift)
        inter_vel=interpolate.interp1d(Hubble_vel, vel)
        vel_max=np.max(Hubble_vel)
        vel_min=np.min(Hubble_vel)
        Delta_omega=(vel_max-vel_min)/N_new
        Hubble_vel=Delta_omega*np.linspace(0, N_new, N_new)+vel_min
        density=inter_dens(Hubble_vel)
        comoving_new=inter_com(Hubble_vel)
        redshift_new=inter_red(Hubble_vel)
        vel_new=inter_vel(Hubble_vel)
        return [density, vel_new, Hubble_vel, comoving_new, redshift_new]
   
    #Interpolate only delta and vel  
    def Rebin_delta_vel(self, density, vel, hubble, hubble_rebin, N_new):
        inter_dens=interpolate.interp1d(hubble, density)
        inter_vel=interpolate.interp1d(hubble, vel)
        density=inter_dens(hubble_rebin)
        vel_new=inter_vel(hubble_rebin)
        return [density, vel_new]  

    def Rebin_delta(self, density, hubble, hubble_rebin, N_new):
        inter_dens=interpolate.interp1d(hubble, density)
        density=inter_dens(hubble_rebin)
        return density
      
    #Convert astropy Hubble-velocities to numpy, i.e. reduce unit
    def Convert_To_Numpy(self, Hubble_vel):
        hubble=np.zeros(Hubble_vel.shape)
        for i in range(Hubble_vel.shape[0]):
            hubble[i]=float(Hubble_vel[i])
        return hubble
    
    

    

    
