import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

#Generating noise and fake data
def noise_generator(times, std):
    return np.random.randn(times.size)*std #Generating Gaussian noise with a sigma of choice
        
def fake_data_generator(times, noise):
    data = np.ones_like(times) 
    data += noise
    return data

def mc_sine(x, period):
    unperfectness = [0.0017432*np.random.normal() for i in range(len(x))] #Add some variation in the sine
    return 0.034677*np.sin((2*np.pi/period)*x) + np.array(unperfectness) #Return a sine with the typical rotation period

def Gauss(x, a, b, c): 
    return a*np.exp(-0.5*((x-b)/c)**2)/(np.sqrt(2*np.pi)*c) #a*np.exp((-(x-b)**2)/(2*c**2))
    
def main(times, flux, eflux, period, tries =150, n_bins = 20):
    found_periods = []
    
    for i in range(tries):
        fake_flux = fake_data_generator(times, noise_generator(times, np.std(flux)))
        fake_flux += mc_sine(times, period) #Add the sine to the fake data
        
        periods= np.linspace(3.1,3.3,500) #Define the LombScargle grid to check. This currently fully determines how well the Gaussian is fit.
        power = LombScargle(times, fake_flux-np.mean(fake_flux), eflux).power(frequency = 1./periods)
        #--------------------------------------------------------------------------------------------------
        if periods[np.argmax(power)] == 3.1 or periods[np.argmax(power)] == 3.3:
            pass
        else:
            found_periods.append(periods[np.argmax(power)])
        #---------------------------------------------------------------------------------------------------
    n, bins, patches = plt.hist(found_periods, n_bins, range = (min(found_periods)-0.01, max(found_periods)+0.01), facecolor='blue', alpha=0.5)
    bin_centers = [(bins[i-1]+bins[i])/2. for i in range(1, len(bins)) ]
    try:
        popt, pcov = curve_fit(Gauss, bin_centers, n)
    except RuntimeError:
        print("Failed to fit")
        return 'Fail'
    return abs(popt[2])   
    
