from scipy.integrate import quad
import scipy.special as sc
import astropy.constants as c
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

class density:
    '''
    Class for spherical stellar density profiles 
    '''
    def __init__(self,params):
        self.params = params

    def profile3d(self,R):
        pass

    def profile2d(self,R):
        f = lambda x,y: self.profile3d(x)*x/np.sqrt(x**y-y**2)
        R = np.atleast_1d(R)
        return 2* np.array([quad(f,i,np.inf,args=(i,) )[0] for i in R])

    def plot2d(self,R,plot_params={'lw': 3},figsize=(8,8)):
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(R,self.profile2d(R),**plot_params) 
        ax.set(xlabel = 'R [kpc]',
               ylabel =r'$\Sigma(R)\;\left[\mathrm{N\,kpc^{-2}}\right]$',
               xscale = 'log',
               yscale = 'log')
        return fig,ax
    
    def plot3d(self,r,plot_params={'lw': 3},figsize=(8,8)):
        '''
        Plot 3D spherical density profile
        
        Params:
        r : radius [kpc] 
        
        '''
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(r,self.profile3d(r),**plot_params)
        ax.set(xlabel = 'r [kpc]', 
               ylabel =r'$\rho(r)\;\left[\mathrm{N\,kpc^{-3}}\right]$',
               xscale = 'log',
               yscale = 'log')  
        return fig,ax
    
    def mass(self,r):
        r, runits = r.value, r.unit
        dunit = self.profile3d(0*runits).unit
        print(dunit)

        f    = lambda x:4*np.pi*(x**2)*self.profile3d(x*runits).value
        mass = quad(f,0,r)[0]*runits**3 *dunit
        return mass 

    def mhalf(self):
        '''
        Mass at half-light radius
        '''
        pass