import numpy as np
import matplotlib.pyplot as plt

class Analysis_Tools():
    def __init__(self):
        return 
    
    def Histogram(self, delta):
        # histogram of 1+delta in log-spaced bins
        bins = np.logspace(-7, np.log10(30.), 100)
        _ = plt.hist(delta.ravel(), bins=bins)
        
        # format the axes
        plt.xscale('log')
        plt.xlabel(r"$1+\delta$")
        plt.ylabel(r"$N_\mathrm{cells}$")
        plt.xlim(1e-4, 500)
        plt.show()
        
    def Plot_Density(self, delta, vmax):
        plt.pcolor(delta[:,:,0], vmin=0, vmax=vmax)
        plt.colorbar()
        plt.show()
        
    def Find_Gaussian(self, delta):
        y=np.log(delta)
        mean=np.mean(y)
        sigma=np.std(y)
        return [mean, sigma]
