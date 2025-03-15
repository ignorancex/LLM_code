
#imports
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

from functions.grid_functions import reduce_res
from functions.utils import der
from functions.data_reading import read_fitgridhdf5

def potential_der(i,filepath,plot=False):
    """
    reads fitgrid and fitpot files and calculates potential related quantities- i is snapshot. 
    Gives out fit_grid, fit_pot and RPhi- fit of potential wrt R
    """    
    print('Reading fitgrid and fitpot files')
  
    fit_grid,fit_pot = read_fitgridhdf5(i,filepath)
  
    ##derivative with respect to R calculation:
    ##smoothening the values, fitting spline and bspline
    #taking the R values- only points within 1 pc of the galactic plane because the derivatives need to be calculated at z=0.
######(20000,2000) is grid_R[1:] which I removed from the fitgrid.py file and directly inputting the shape here
  
    fit_grid_new, fit_pot_new,R_res,z_res= reduce_res(fit_grid,fit_pot,0,50,20000,2000)
    print(f'Doing RPhi calc for i={i} Myr')
    x=np.unique(fit_grid_new[abs(fit_grid_new[:,1])<2][:,0])
  
    #averaging over the potential for same R values- different z values.
  
    y=fit_pot_new[abs(fit_grid_new[:,1])<2].reshape(int(fit_grid_new[abs(fit_grid_new[:,1])<2].shape[0]/len(x)), len(x)).mean(axis=0)
  
    #Gaussian filtering
    y_smooth = gaussian_filter1d(y,30,mode='nearest')
  
    #bspline fitting
    t,c,kb = interpolate.splrep(x,y_smooth,s=200,k=3)
    RPhi = interpolate.BSpline(t,c,kb,extrapolate=True)
  
    #defining finer grid
    xx = np.linspace(0.1,20000,100000)
    yy = RPhi(xx)
    if plot: 
      ###plots Potential wrt R and circular velocity wrt R
        plt.close('all')
        plt.figure(figsize=(10,10))
        plt.plot(x,y,'k-',lw=2,alpha=0.1,label='raw_grid')
        plt.plot(x,y_smooth,'y-',lw=2,label='gaussian_smooth')
        plt.plot(xx,yy,'b--',label='bspline')
        plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0)) 
        plt.xlabel('R(pc)')
        plt.ylabel('Potential (km/s)$^{2}$')
        plt.legend()
        plt.savefig('/g/data/jh2/ax8338/action/results/R/RPhi'+str(i)+'.png')
        plt.show()
        plt.close('all')
        plt.figure(figsize=(10,10))
        plt.plot(xx[2:-2], np.sqrt(der(yy,xx)[0]*xx[2:-2]),'--',label='bspline')
        plt.xlabel('R(pc)')
        plt.ylabel('circular velocity (km/s)')
        plt.legend()
        plt.savefig('/g/data/jh2/ax8338/action/results/R/circular_vel'+str(i)+'.png')
        plt.show()
      
    print(f'Leaving pot_der function and returning fit_grid, fit_pot and RPhi for i={i} i.e {i} Myr')
    return fit_grid,fit_pot,RPhi 
