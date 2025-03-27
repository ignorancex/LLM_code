 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sukrit, modifying sukrit and azra's past codes
"""
import numpy as np #This furnishes array operations.
import scipy as sp
from scipy import interpolate as interp
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.integrate
import pdb
import pandas as pd
#import cookbook.py as cookbook

cm2nm=1.0E+7 #1 cm in nm
hc=1.98645e-9 #value of h*c in erg*nm, useful to convert from ergs/cm2/s/nm to photons/cm2/s/nm

###Plotting switches.
plotBr=False
plotI=False
plotCl=False

plotFeBF42=False
plotferrocyanide=False
plot_NO3=False

plot_CO3_HCO3=False
plot_HSm=False
plotHSO3m=True
plotSO3m=True
plotSNP_FeCl2_FeSO4=False

plot_halide_ferrous_ocean=False
plot_halide_freshwater_lake=False
plot_halide_carbonate_lake=False
plot_ferrous_lake=False


##############
###Kelly Colors, via https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors/13781114#13781114
##############
kelly_colors = dict(vivid_yellow=(255.0/256.0, 179.0/256.0, 0.0/256.0),
                    strong_purple=(128.0/256.0, 62.0/256.0, 117.0/256.0),
                    vivid_orange=(255.0/256.0, 104.0/256.0, 0.0/256.0),
                    very_light_blue=(166.0/256.0, 189.0/256.0, 215.0/256.0),
                    vivid_red=(193.0/256.0, 0.0/256.0, 32.0/256.0),
                    grayish_yellow=(206.0/256.0, 162.0/256.0, 98.0/256.0),
                    medium_gray=(129.0/256.0, 112.0/256.0, 102.0/256.0),
                    # these aren't good for people with defective color vision:
                    vivid_green=(0.0/256.0, 125.0/256.0, 52.0/256.0),
                    strong_purplish_pink=(246.0/256.0, 118.0/256.0, 142.0/256.0),
                    strong_blue=(0.0/256.0, 83.0/256.0, 138.0/256.0),
                    strong_yellowish_pink=(255.0/256.0, 122.0/256.0, 92.0/256.0),
                    strong_violet=(83.0/256.0, 55.0/256.0, 122.0/256.0),
                    vivid_orange_yellow=(255.0/256.0, 142.0/256.0, 0.0/256.0),
                    strong_purplish_red=(179.0/256.0, 40.0/256.0, 81.0/256.0),
                    vivid_greenish_yellow=(244.0/256.0, 200.0/256.0, 0.0/256.0),
                    strong_reddish_brown=(127.0/256.0, 24.0/256.0, 13.0/256.0),
                    vivid_yellowish_green=(147.0/256.0, 170.0/256.0, 0.0/256.0),
                    deep_yellowish_brown=(89.0/256.0, 51.0/256.0, 21.0/256.0),
                    vivid_reddish_orange=(241.0/256.0, 58.0/256.0, 19.0/256.0),
                    dark_olive_green=(35.0/256.0, 44.0/256.0, 22.0/256.0))

##############
###Define interpolator functions for errors.
##############

def linear_interp(xs, x_ref, y_ref, y_ref_errs):
    """
    WARNING HUGE ASSUMPTION: ASSUMES x_ref (and xs?) INCREASE MONOTONICALLY
    MUST FORMAT INPUTS TO COMPLY.
    

    Parameters
    ----------
    xs : abscissa to project to
    x_ref : abscissa of input data
    y_ref : y-values of input data
    y_ref_errs : errors on y-values of input data
    Returns
    -------
    ys: interpolated values. Should be identical to numpy interp.
    y_errs: interpolated errors, calculated assuming uncorrelated Gaussian errors.
    
    https://stackoverflow.com/questions/24616079/how-to-interpolate-using-nearest-neighbours-for-high-dimension-numpy-python-arra
    """
    ###Assume 0 unless otherwise known.
    ys=np.zeros(np.shape(xs))
    y_errs=np.zeros(np.shape(xs))
    
    ###Get indices
    rights = np.searchsorted(x_ref, xs, 'left')
    
    for ind in range(0, len(xs)):
        right=rights[ind]
        left=right-1
        x=xs[ind]
        if right==0 and x<x_ref[0]: #If x is less than the x_ref range entirely, zero the values
            ys[ind]=0
            y_errs[ind]=0
        elif right==0 and x==x_ref[0]: #if x is exactly the beginning of the x_ref range, then it should just be that value. 
            ys[ind]=y_ref[0]
            y_errs[ind]=y_ref_errs[0]
        elif right==len(x_ref): #If x is greater than the x_ref range entirely, the it should be 0. 
            ys[ind]=0
            y_errs[ind]=0
        else:

            x1=x_ref[left]
            y1=y_ref[left]
            y1_err=y_ref_errs[left]
            
            x2=x_ref[right]
            y2=y_ref[right]
            y2_err=y_ref_errs[right]
            
            a=(x2-x)/(x2-x1)
            b=1.0-a
            ys[ind]=a*y1 + b*y2
            y_errs[ind]=np.sqrt((a*y1_err)**2.0 + (b*y2_err)**2.0)
  
    ###Compare to what np.interp returns, just to be safe.
    y_np_interp=np.interp(xs, x_ref, y_ref, left=0, right=0)
    residuals=np.abs(ys-y_np_interp)/(0.5*(ys+y_np_interp)) #residuals. Should be 0.
    
    if np.sum(residuals>10.0*np.finfo(float).eps)>0: #if the residuals exceed 10x the precision of a float ANYWHERE, kill everything dramatically so we see it. 
        ys=np.nan
        y_errs=np.nan
        print('Error in custom linear interpolation function') 
    return ys, y_errs

# ###Test
# test_abscissa=np.arange(1.0, 5.0, step=1.0)
# test_data=np.random.random_sample(np.shape(test_abscissa))*100.0 #random data
# test_errs=np.sqrt(test_data)#np.zeros(np.shape(test_abscissa))

# interp_abscissa=np.arange(0, 6.0, step=0.1)

# interp_data, interp_data_errs=linear_interp(interp_abscissa, test_abscissa, test_data, test_errs)

# np_interp_data=np.interp(interp_abscissa, test_abscissa, test_data, left=0, right=0)

# fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

# ax.errorbar(test_abscissa, test_data , yerr=test_errs, linewidth=2, linestyle='-', color='black', label=r'Input Data', capsize=5)
# ax.errorbar(interp_abscissa, np_interp_data, yerr=0.0, linewidth=2, linestyle='--', color='green', label=r'Input Data', capsize=0)
# ax.errorbar(interp_abscissa, interp_data , yerr=interp_data_errs, linewidth=2, linestyle=':', color='red', label=r'Interp', capsize=0)

# plt.show()
##############
###Define timescale functions
##############

def get_timescale(process, total_abs_abs, total_abs_wav, depth):
    """
    Takes:
        -name of process, options are nucleobase-photolysis, cytidine-deamination, 2AO-photolysis, 2AI-photolysis, 2AT-photolysis
        -total linear decadic absorption coefficient for the solution, in cm**-1, as a function of wavelength
        -Wavelengths corresponding to those absorption coefficients, in nm
        -Depth of the reservoir to be simulated, in cm.
    """
    if process=='nucleobase-photolysis':
        #Direct photodissociation of nucleobases, following the simplifying assumption of Cleaves et al. 1998, i.e. that nucleobase photolysis rate is proportional to the transmittance at 260 nm
        abs_260nm=np.interp(260., total_abs_wav, total_abs_abs) #Get decadic absorption coefficient at 260 nm.
        def transmittance(x, a):
            return 10.0**(-a*x/0.5) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        timescale_enhancement=depth/scipy.integrate.quad(transmittance, 0, depth, args=(abs_260nm), epsabs=0, epsrel=1e-3,limit=1000)[0]
        
    if process=='cytidine-deamination':
        df=pd.read_excel('./todd_data_for_sukrit.xlsx', sheet_name='C-to-U') 
        todd_wav_centers=np.array(df['Wavelength (nm)']) #nm
        todd_rates=np.array(df['C to U']) #rel rate (min^-1 cm^-2 -->???)
        
        ###From paper: bin widths are 10 nm, 5 nm either side.
        todd_wav_left=todd_wav_centers-5.0
        todd_wav_right=todd_wav_centers+5.0
        
        ###Densify the rate data as a function of wavelength, to enable
        todd_wav_dense=np.linspace(todd_wav_left[0], todd_wav_right[-1], num=1000, endpoint=True)#new, denser wavelength grid
        todd_rates_dense=np.zeros(np.shape(todd_wav_dense))
        for ind in range(0, len(todd_rates)):
            wav_inds=np.where((todd_wav_dense>=todd_wav_left[ind]) & (todd_wav_dense<=todd_wav_right[ind]))
            todd_rates_dense[wav_inds]=np.ones(np.shape(wav_inds))*todd_rates[ind]
        def integrand(x, wav): #main integral, actual solution
            a=np.interp(wav, total_abs_wav, total_abs_abs)
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return ((10.0**(-a*x/0.5))*phi) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        def integrand_1d(wav): #surface only
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return phi
        
        timescale_enhancement=(depth*scipy.integrate.quad(integrand_1d, todd_wav_left[0], todd_wav_right[-1], epsabs=0, epsrel=1e-3, limit=1000)[0])/(scipy.integrate.dblquad(integrand, todd_wav_left[0], todd_wav_right[-1], lambda wav: 0, lambda wav: depth, epsabs=0, epsrel=1e-3)[0])
    if process=='2AO-photolysis':
        df=pd.read_excel('./todd_data_for_sukrit.xlsx', sheet_name='Aminoazoles') 
        todd_wav_centers=np.array(df['Wavelength (nm)']) #nm
        todd_rates=np.array(df['AO']) #rel rate (min^-1 cm^-2 -->???)
        
        ###From paper: bin widths are 10 nm, 5 nm either side.
        todd_wav_left=todd_wav_centers-5.0
        todd_wav_right=todd_wav_centers+5.0
        
        ###Densify the rate data as a function of wavelength, to enable
        todd_wav_dense=np.linspace(todd_wav_left[0], todd_wav_right[-1], num=1000, endpoint=True)#new, denser wavelength grid
        todd_rates_dense=np.zeros(np.shape(todd_wav_dense))
        for ind in range(0, len(todd_rates)):
            wav_inds=np.where((todd_wav_dense>=todd_wav_left[ind]) & (todd_wav_dense<=todd_wav_right[ind]))
            todd_rates_dense[wav_inds]=np.ones(np.shape(wav_inds))*todd_rates[ind]
        def integrand(x, wav): #main integral, actual solution
            a=np.interp(wav, total_abs_wav, total_abs_abs)
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return ((10.0**(-a*x/0.5))*phi) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        def integrand_1d(wav): #surface only
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return phi
        
        timescale_enhancement=(depth*scipy.integrate.quad(integrand_1d, todd_wav_left[0], todd_wav_right[-1], epsabs=0, epsrel=1e-3, limit=1000)[0])/(scipy.integrate.dblquad(integrand, todd_wav_left[0], todd_wav_right[-1], lambda wav: 0, lambda wav: depth, epsabs=0, epsrel=1e-3)[0])
    
    if process=='2AI-photolysis':
        df=pd.read_excel('./todd_data_for_sukrit.xlsx', sheet_name='Aminoazoles') 
        todd_wav_centers=np.array(df['Wavelength (nm)']) #nm
        todd_rates=np.array(df['AI']) #rel rate (min^-1 cm^-2 -->???)
        
        ###From paper: bin widths are 10 nm, 5 nm either side.
        todd_wav_left=todd_wav_centers-5.0
        todd_wav_right=todd_wav_centers+5.0
        
        ###Densify the rate data as a function of wavelength, to enable
        todd_wav_dense=np.linspace(todd_wav_left[0], todd_wav_right[-1], num=1000, endpoint=True)#new, denser wavelength grid
        todd_rates_dense=np.zeros(np.shape(todd_wav_dense))
        for ind in range(0, len(todd_rates)):
            wav_inds=np.where((todd_wav_dense>=todd_wav_left[ind]) & (todd_wav_dense<=todd_wav_right[ind]))
            todd_rates_dense[wav_inds]=np.ones(np.shape(wav_inds))*todd_rates[ind]
        def integrand(x, wav): #main integral, actual solution
            a=np.interp(wav, total_abs_wav, total_abs_abs)
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return ((10.0**(-a*x/0.5))*phi) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        def integrand_1d(wav): #surface only
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return phi
        
        timescale_enhancement=(depth*scipy.integrate.quad(integrand_1d, todd_wav_left[0], todd_wav_right[-1], epsabs=0, epsrel=1e-3, limit=1000)[0])/(scipy.integrate.dblquad(integrand, todd_wav_left[0], todd_wav_right[-1], lambda wav: 0, lambda wav: depth, epsabs=0, epsrel=1e-3)[0])
  
    if process=='2AT-photolysis':
        df=pd.read_excel('./todd_data_for_sukrit.xlsx', sheet_name='Aminoazoles') 
        todd_wav_centers=np.array(df['Wavelength (nm)']) #nm
        todd_rates=np.array(df['AT']) #rel rate (min^-1 cm^-2 -->???)
        
        ###From paper: bin widths are 10 nm, 5 nm either side.
        todd_wav_left=todd_wav_centers-5.0
        todd_wav_right=todd_wav_centers+5.0
        
        ###Densify the rate data as a function of wavelength, to enable
        todd_wav_dense=np.linspace(todd_wav_left[0], todd_wav_right[-1], num=1000, endpoint=True)#new, denser wavelength grid
        todd_rates_dense=np.zeros(np.shape(todd_wav_dense))
        for ind in range(0, len(todd_rates)):
            wav_inds=np.where((todd_wav_dense>=todd_wav_left[ind]) & (todd_wav_dense<=todd_wav_right[ind]))
            todd_rates_dense[wav_inds]=np.ones(np.shape(wav_inds))*todd_rates[ind]
        def integrand(x, wav): #main integral, actual solution
            a=np.interp(wav, total_abs_wav, total_abs_abs)
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return ((10.0**(-a*x/0.5))*phi) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        def integrand_1d(wav): #surface only
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return phi
        
        timescale_enhancement=(depth*scipy.integrate.quad(integrand_1d, todd_wav_left[0], todd_wav_right[-1], epsabs=0, epsrel=1e-3, limit=1000)[0])/(scipy.integrate.dblquad(integrand, todd_wav_left[0], todd_wav_right[-1], lambda wav: 0, lambda wav: depth, epsabs=0, epsrel=1e-3)[0])
     
    return timescale_enhancement

##############
###Load TOA irradiance, surface intensity from Ranjan & Sasselov 2017, specifically when we matched Rugheimer et al. 2015 exactly. 
##############
uv_wav, uv_toa_intensity, uv_surface_flux, uv_surface_intensity=np.genfromtxt('./rugheimer_earth_epoch0.dat', skip_header=1, skip_footer=0, usecols=(2,3,4,6), unpack=True) #From Ranjan & Sasselov 2017, the GitHub. 

###Convert from ergs to photons
uv_toa_intensity *= uv_wav/hc
uv_surface_flux *= uv_wav/hc
uv_surface_intensity *= uv_wav/hc

###Restrict data to 190 to 310 nm
inds=np.where((uv_wav>=190.0) & (uv_wav <=310.0))

uv_wav=uv_wav[inds]
uv_toa_intensity=uv_toa_intensity[inds]
uv_surface_flux=uv_surface_flux[inds]
uv_surface_intensity=uv_surface_intensity[inds]

##############
###Load data
##############
data_wav={} #dict to hold wavelength scale for measured molar absorptivities (nm)
data_abs={} #dict to hold molar absorptivities for measured molar absorptivities (M**-1 cm**-1.)


###
#Lit Data
###

# ###Pure water
# h2o_quickenden_wav1, h2o_quickenden_abs1 = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/quickenden.dat', skip_header=2, unpack=True, usecols=(0,1))#nm, cm**-1. Includes both scattering, absorption. 

h2o_quickenden_wav, h2o_quickenden_abs, h2o_quickenden_abs_err= np.genfromtxt('./Azra_Project/quickenden_with_uncerts.dat', skip_header=2, unpack=True, usecols=(0,1,2))#nm, cm**-1. Includes both scattering, absorption. 

# fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)
# ax.errorbar(h2o_quickenden_wav, h2o_quickenden_abs , yerr=h2o_quickenden_abs_err, linewidth=2, linestyle=':', color='red', label=r'quick with errs', capsize=5)
# ax.errorbar(h2o_quickenden_wav1, h2o_quickenden_abs1, yerr=0.0, linewidth=2, marker='o', linestyle='--', color='black', label=r'quickenden orig', capsize=0)
# ax.set_yscale('log')
# plt.show()

###Modern oceans: (Smith & Baker) Warning, guessing <300 nm.
#pureoceans_smith1981_wav, pureoceans_smith1981_abs=np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/smithbaker_purest.dat', skip_header=2, skip_footer=0, unpack=True, usecols=(0,1)) #purest modern natural water; nm, cm**-1.

###Modern oceans: (Morel+2007)
pureoceans_morel_wav=np.array([300., 305., 310., 315., 320., 325., 330., 335., 340., 345., 350., 355., 360., 365., 370., 375., 380., 385., 390., 395.])
pureoceans_morel_abs=(np.array([0.01150, 0.0110, 0.01050, 0.01013, 0.00975, 0.00926, 0.00877, 0.00836, 0.00794, 0.00753, 0.00712, 0.00684, 0.00656, 0.00629, 0.00602, 0.00561, 0.00520, 0.00499, 0.00478, 0.00469]) + 0.5*np.array([0.0226, 0.0211, 0.0197, 0.0185, 0.0173, 0.0162, 0.0152, 0.0144, 0.0135, 0.0127, 0.0121, 0.0113, 0.0107, 0.0099, 0.0095, 0.0089, 0.0085, 0.0081, 0.0077, 0.0072])) * 0.01 #K_dsw2, converted from m**-1 to cm**-1.

###From Azra's work.
data_wav['Br_azra'], data_abs['Br_azra'] = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/johnson_br-truncated.dat', skip_header=2, unpack=True, usecols=(0,1))# Br-. Truncated to remove data >220 nm, which looks unreliable to me. 
data_wav['I_azra'], data_abs['I_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/guenther_i.dat', skip_header=2, unpack=True, usecols=(0,1)) #I-
data_wav['NO2_azra'], data_abs['NO2_azra'] = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/mack_no2.dat', skip_header=2, unpack=True, usecols=(0,1))#NO2-
data_wav['NO3_azra'], data_abs['NO3_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/mack_no3.dat', skip_header=2, unpack=True, usecols=(0,1)) #NO3-
data_wav['guenther_hsm'], absthunk = np.genfromtxt('./guenther_2001_HSm.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  absorbance
data_abs['guenther_hsm']=absthunk/(50.0E-6*1.0) #50 uM concentration of HS-, 1 cm pathlength

data_wav['Fe(BF4)2_azra'], data_abs['Fe(BF4)2_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/fontana_fe2bf4.dat', skip_header=2, unpack=True, usecols=(0,1)) #Fe(BF4)2
data_wav['FeCl2_azra'], data_abs['FeCl2_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/fontana_fecl2.dat', skip_header=2, unpack=True, usecols=(0,1)) #FeCl2
data_wav['FeSO4_azra'], data_abs['FeSO4_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/fontana_feso4.dat', skip_header=2, unpack=True, usecols=(0,1)) #FeSO4
data_wav['gelbstoff_azra'], data_abs['gelbstoff_azra'] = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/cleaves_gelbstoff.dat', skip_header=2, unpack=True, usecols=(0,1))#organic gunk
data_wav['KCl_azra'], data_abs['KCl_azra'] = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/perkampus_kcl.dat', skip_header=2, unpack=True, usecols=(0,1))



###From Birkmann+2018
#Assume 5% error based on their past studies.
df=pd.read_excel('./birkmann_2018_data.xlsx', sheet_name='Br-I-Cl', skiprows=0)
#pdb.set_trace()
data_wav['birk_BrICl']=df['Wavelength (nm)']
data_abs['birk_Br']=df['Br- (M**-1 cm**-1)']
data_abs['birk_err_Br']=0.05*df['Br- (M**-1 cm**-1)']
data_abs['birk_I']=df['I- (M**-1 cm**-1)']
data_abs['birk_err_I']=0.05*df['I- (M**-1 cm**-1)']
data_abs['birk_Cl']=df['Cl- (M**-1 cm**-1)']
data_abs['birk_err_Cl']=0.05*df['Cl- (M**-1 cm**-1)']
df=pd.read_excel('./birkmann_2018_data.xlsx', sheet_name='NO3-OH', skiprows=0)
data_wav['birk_NO3OH']=df['Wavelength (nm)']
data_abs['birk_NO3']=df['NO3- (M**-1 cm**-1)']
data_abs['birk_err_NO3']=0.05*df['NO3- (M**-1 cm**-1)']
#data_abs['birk_OH']=df['OH- (M**-1 cm**-1)']
df=pd.read_excel('./birkmann_2018_data.xlsx', sheet_name='SO4-HCO3-CO3', skiprows=0)
data_wav['birk_SO4HCO3CO3']=df['Wavelength (nm)']
# data_abs['birk_SO4']=df['SO4(2-) (M**-1 cm**-1)']
data_abs['birk_HCO3']=df['HCO3- (M**-1 cm**-1)']
data_abs['birk_err_HCO3']=0.05*df['HCO3- (M**-1 cm**-1)']
data_abs['birk_CO3']=df['CO3(2-) (M**-1 cm**-1)']
data_abs['birk_err_CO3']=0.05*df['CO3(2-) (M**-1 cm**-1)']


###From Strizhakov+2014, for sodium nitroprusside. Need to concatenate shortwave (log scale) and longwave (linear scale) data here. 

strizhakov_snp_shortwav_wav, strizhakov_snp_shortwav_abs=np.genfromtxt('./strizhakov_fig4_snp_shortwave_Dataset.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  M**-1 cm**-1

strizhakov_snp_longwav_wav, strizhakov_snp_longwav_abs=np.genfromtxt('./strizhakov_fig4_snp_longwave_Dataset.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  M**-1 cm**-1

data_wav['strizhakov_SNP']=np.concatenate((strizhakov_snp_shortwav_wav, strizhakov_snp_longwav_wav))
data_abs['strizhakov_SNP']=np.concatenate((strizhakov_snp_shortwav_abs, strizhakov_snp_longwav_abs))

###Let's get ferrocyanide and ferricyanide in on the party.
hc_eVnm=1240. #h*c in eV*nm

##Ferrocyanide from Ross+2018
wavthunk, absthunk = np.genfromtxt('./ross_fig3_ferrocyanide.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#eV,  M**-1 cm**-1
data_wav['ross_ferrocyanide']=np.flip(hc_eVnm/wavthunk) #convert eV to nm
data_abs['ross_ferrocyanide']=np.flip(absthunk)


##Ferricyanide from Ross+2018
wavthunk, absthunk = np.genfromtxt('./ross_fig3_ferricyanide.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#eV,  M**-1 cm**-1
data_wav['ross_ferricyanide']=np.flip(hc_eVnm/wavthunk) #convert eV to nm
data_abs['ross_ferricyanide']=np.flip(absthunk)


###
#Bisulfite/Sulfite
###

#Beyad
df=pd.read_excel('./beyad_conc_spectra_sulfurIV.xlsx', sheet_name='Spectra', skiprows=0)
data_wav['beyad']=np.flip(np.nan_to_num(df['Wavelength'])) #nm
data_abs['beyad_hso3']=np.flip(np.nan_to_num(df['HSO3']))
data_abs['beyad_so3']=np.flip(np.nan_to_num(df['SO3']))


# #From Golding
# data_wav['golding_hso3'], data_abs['golding_hso3'] = np.genfromtxt('./golding_fig1_HSO3m.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  M**-1 cm**-1
# data_wav['golding_so3'], data_abs['golding_so3'] = np.genfromtxt('./golding_fig1_SO3mm.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  M**-1 cm**-1

#From Hayon
data_wav['hayon_so3'], data_abs['hayon_so3'] = np.genfromtxt('./hayon_fig2_bisulfite_sulfite_SO3mm.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  M**-1 cm**-1
data_wav['hayon_hso3'], data_abs['hayon_hso3'] = np.genfromtxt('./hayon_fig2_bisulfite_sulfite_HSO3m.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  M**-1 cm**-1

##From Fischer+1996
#Import, splice, sort...
fischer_sulfite_short_wav, fischer_sulfite_short_abs, fischer_sulfite_long_wav, fischer_sulfite_long_abs, fischer_bisulfite_short_wav, fischer_bisulfite_short_abs, fischer_bisulfite_long_wav, fischer_bisulfite_long_abs = np.genfromtxt('./fischer_1996_siv_data.csv', skip_header=1, unpack=True,  delimiter=',')#nm,  M**-1 cm**-1


###Get rid of NaNs due to ragged arrays...
inds1=np.logical_not(np.isnan(fischer_sulfite_short_wav))
fischer_sulfite_short_wav=fischer_sulfite_short_wav[inds1]
fischer_sulfite_short_abs=fischer_sulfite_short_abs[inds1]

inds2=np.logical_not(np.isnan(fischer_sulfite_long_wav))
fischer_sulfite_long_wav=fischer_sulfite_long_wav[inds2]
fischer_sulfite_long_abs=fischer_sulfite_long_abs[inds2]

inds3=np.logical_not(np.isnan(fischer_bisulfite_short_wav))
fischer_bisulfite_short_wav=fischer_bisulfite_short_wav[inds3]
fischer_bisulfite_short_abs=fischer_bisulfite_short_abs[inds3]

inds4=np.logical_not(np.isnan(fischer_bisulfite_long_wav))
fischer_bisulfite_long_wav=fischer_bisulfite_long_wav[inds4]
fischer_bisulfite_long_abs=fischer_bisulfite_long_abs[inds4]


fischer_sulfite_wav=np.flip(np.concatenate((fischer_sulfite_long_wav, fischer_sulfite_short_wav)))
fischer_sulfite_abs=np.flip(np.concatenate((fischer_sulfite_long_abs, fischer_sulfite_short_abs)))

fischer_bisulfite_wav=np.flip(np.concatenate((fischer_bisulfite_long_wav, fischer_bisulfite_short_wav)))
fischer_bisulfite_abs=np.flip(np.concatenate((fischer_bisulfite_long_abs, fischer_bisulfite_short_abs)))

#Is in units of m^2/mol = 10*M^-1 cm^-1. Convert and load
data_wav['fischer_so3']=fischer_sulfite_wav
data_abs['fischer_so3']=fischer_sulfite_abs*10.0
data_wav['fischer_hso3']=fischer_bisulfite_wav
data_abs['fischer_hso3']=fischer_bisulfite_abs*10.0

###
#Our data
###

###From Gabi/Corinna's work
df=pd.read_excel('./2020-08-23_Extinktion-coefficients-Sukrit-V7_mod.xlsx', sheet_name='Tabelle1') 
data_wav['LK']=np.nan_to_num(df['Wavelength']) #nm LK=Lozano-Kufner
data_abs['NaBr_LK']=np.nan_to_num(df['NaBr'])
data_abs['NaBr_err_LK']=np.nan_to_num(df['NaBr-error'])

data_abs['KBr_LK']=np.nan_to_num(df['KBr'])
data_abs['KBr_err_LK']=np.nan_to_num(df['KBr-error'])

data_abs['NaHCO3_LK']=np.nan_to_num(df['NaHCO3'])
data_abs['NaHCO3_err_LK']=np.nan_to_num(df['NaHCO3-error'])

data_abs['NaCl_LK']=np.nan_to_num(df['NaCl'])
data_abs['NaCl_err_LK']=np.nan_to_num(df['NaCl-error'])

#data_abs['KCl_LK']=np.nan_to_num(df['KCl'])
#data_abs['KCl_err_LK']=np.nan_to_num(df['KCl-error'])

data_abs['Fe(BF4)2_LK']=np.nan_to_num(df['Fe(BF4)2'])
data_abs['Fe(BF4)2_err_LK']=np.nan_to_num(df['Fe(BF4)2-error'])

data_abs['K_ferrocyanide_LK']=np.nan_to_num(df['K4[Fe(CN)6]'])
data_abs['K_ferrocyanide_err_LK']=np.nan_to_num(df['K4[Fe(CN)6]-error'])

data_abs['NaI_LK']=np.nan_to_num(df['NaI'])
data_abs['NaI_err_LK']=np.nan_to_num(df['NaI-error'])

data_abs['KI_LK']=np.nan_to_num(df['KI'])
data_abs['KI_err_LK']=np.nan_to_num(df['KI-error'])

data_abs['NaNO3_LK']=np.nan_to_num(df['NaNO3'])
data_abs['NaNO3_err_LK']=np.nan_to_num(df['NaNO3-error'])


########################
###Synthesize composite data
########################
###Splice the I data
data_abs['I_LK']=np.copy(data_abs['NaI_LK'])
data_abs['I_LK'][np.where(data_wav['LK']>260.0)]=np.copy(data_abs['KI_LK'][np.where(data_wav['LK']>260.0)])

data_abs['I_err_LK']=np.copy(data_abs['NaI_err_LK'])
data_abs['I_err_LK'][np.where(data_wav['LK']>260.0)]=np.copy(data_abs['KI_err_LK'][np.where(data_wav['LK']>260.0)])


###Splice the sulfite data
#Use the Fischer data where the Beyad data terminate
min_beyad_so3=np.min(data_wav['beyad'])
so3_fischer_inds=np.where(data_wav['fischer_so3']<min_beyad_so3)
data_wav['so3_composite']=np.concatenate((data_wav['fischer_so3'][so3_fischer_inds], data_wav['beyad']))
data_abs['so3_composite']=np.concatenate((data_abs['fischer_so3'][so3_fischer_inds], data_abs['beyad_so3']))


###Splice the bisulfite data
#Use the Fischer data where the Beyad data terminate. Log-linear interpolate in between.
interp_wav=np.linspace(data_wav['fischer_hso3'][-1], data_wav['beyad'][0], num=10, endpoint=True)
interp_wav=interp_wav[1:-1] #don't overwrite actual data
interp_abs=10.0**(np.interp(interp_wav, np.array([data_wav['fischer_hso3'][-1], data_wav['beyad'][0]]), np.log10(np.array([data_abs['fischer_hso3'][-1], data_abs['beyad_hso3'][0]])))) 
data_wav['hso3_composite']=np.concatenate((data_wav['fischer_hso3'], interp_wav, data_wav['beyad']))
data_abs['hso3_composite']=np.concatenate((data_abs['fischer_hso3'], interp_abs, data_abs['beyad_hso3']))

########################
###Quality of input data?
########################
if plotBr:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'Br$^-$')    
    ax.plot(data_wav['Br_azra'], data_abs['Br_azra'] , linewidth=2, linestyle='--', color='blue', label=r'Br$^-$ (Johnson+2002)')
    # ax.plot(data_wav['birk_BrICl'], data_abs['birk_Br'] , linewidth=2, linestyle='--', marker='d', color='red', label=r'Br$^-$ (Birkmann+2018)')
    ax.errorbar(data_wav['birk_BrICl'], data_abs['birk_Br'] , yerr=data_abs['birk_err_Br'], linewidth=2, linestyle='-', marker='o', color='red', label=r'Br$^-$ (Birkmann+2018)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['KBr_LK'] , yerr=data_abs['KBr_err_LK'], linewidth=2, linestyle='-', marker='o', color='green', label=r'KBr (This Work)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['NaBr_LK'], yerr=data_abs['NaBr_err_LK'],linewidth=2, linestyle='-', marker='o', color='black', label=r'NaBr (This Work)', capsize=5)
    
    # ###Plot the upper limit on absorption
    # nabr_wav_inds=np.squeeze(np.where(data_abs['NaBr_LK']==0.0))
    # kbr_wav_inds=np.squeeze(np.where(data_abs['KBr_LK']==0.0))
    
    # nabr_upperlim=data_abs['NaBr_LK'][nabr_wav_inds[0]-1] + 1.0*data_abs['NaBr_err_LK'][nabr_wav_inds[0]-1]
    # kbr_upperlim=data_abs['KBr_LK'][kbr_wav_inds[0]-1] + 1.0*data_abs['KBr_err_LK'][kbr_wav_inds[0]-1]

    # ax.plot(data_wav['LK'][nabr_wav_inds], np.ones(len(nabr_wav_inds))*nabr_upperlim, linestyle=':', color='black')
    # ax.plot(data_wav['LK'][kbr_wav_inds], np.ones(len(kbr_wav_inds))*kbr_upperlim, linestyle=':', color='green')

    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.set_ylim([1.0E-1, 1.0E4])
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/Brm.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/Brm.jpg', orientation='portrait', format='jpg')
  
if plotI:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'I$^-$')    
    ax.plot(data_wav['I_azra'], data_abs['I_azra'] , linewidth=2, linestyle='--', color='blue', label=r'I$^-$ (Guenther+2001)')
    # ax.plot(data_wav['birk_BrICl'], data_abs['birk_I'] , linewidth=2, linestyle='--', marker='d', color='red', label=r'I$^-$ (Birkmann+2018)')
    ax.errorbar(data_wav['birk_BrICl'], data_abs['birk_I'], data_abs['birk_err_I'] , linewidth=2, linestyle='--', marker='d', color='red', label=r'I$^-$ (Birkmann+2018)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['KI_LK'], yerr=data_abs['KI_err_LK'], linewidth=2, linestyle='-', marker='o', color='green', label=r'KI (This Work)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['NaI_LK'], yerr=data_abs['NaI_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'NaI (This Work)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['I_LK'], yerr=0, linewidth=1.5, linestyle='-', color='purple', label='I$^-$ (this work)')
    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/Im.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/Im.jpg', orientation='portrait', format='jpg')
    
if plotCl:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'Cl$^-$')    
    ax.plot(data_wav['KCl_azra'], data_abs['KCl_azra'] , linewidth=2, linestyle='--', color='blue', label=r'KCl (Perkampus+1992)')
    # ax.plot(data_wav['birk_BrICl'], data_abs['birk_Cl'] , linewidth=2, linestyle='--', marker='d', color='red', label=r'Cl$^-$ (Birkmann+2018)')
    ax.errorbar(data_wav['birk_BrICl'], data_abs['birk_Cl'], yerr=data_abs['birk_err_Cl'], linewidth=2, linestyle='-', marker='o', color='red', label=r'Cl$^-$ (Birkmann+2018)', capsize=5)

#    ax.errorbar(data_wav['LK'], data_abs['KCl_LK'], yerr=data_abs['KCl_err_LK'], linewidth=2, linestyle='-', marker='o', color='green', label=r'KCl (This Work)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['NaCl_LK'], yerr=data_abs['NaCl_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'NaCl (This Work)', capsize=5)
    
    
    # ###Plot the upper limit on absorption
    # nacl_wav_inds=np.squeeze(np.where(data_abs['NaCl_LK']==0.0))
    
    # nacl_upperlim=data_abs['NaCl_LK'][nacl_wav_inds[0]-1] + 1.0*data_abs['NaCl_err_LK'][nacl_wav_inds[0]-1]

    # ax.plot(data_wav['LK'][nacl_wav_inds], np.ones(len(nacl_wav_inds))*nacl_upperlim, linestyle=':', color='black')

    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/Clm.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/Clm.jpg', orientation='portrait', format='jpg')

    
if plotFeBF42:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'Fe(BF$_4$)$_2$')    
    ax.errorbar(data_wav['LK'], data_abs['Fe(BF4)2_LK'], data_abs['Fe(BF4)2_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'Fe(BF$_4$)$_2$ (This Work)', capsize=5)

    ax.errorbar(data_wav['Fe(BF4)2_azra'], data_abs['Fe(BF4)2_azra'] , yerr=0, linewidth=2, linestyle='--', color='blue', label=r'Fe(BF$_4$)$_2$ (Fontana+2007)')

    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/FeBF42.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/FeBF42.jpg', orientation='portrait', format='jpg')            

if plotSNP_FeCl2_FeSO4: 
    #We need to do.
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'Fe(II) Compounds From Literature')    
    ax.plot(data_wav['strizhakov_SNP'], data_abs['strizhakov_SNP'] , linewidth=2, linestyle='-', color='green', label=r'Na$_2$Fe(CN)$_5$NO (Strizhakov+2014)')
    ax.plot(data_wav['FeCl2_azra'], data_abs['FeCl2_azra'] , linewidth=2, linestyle='-',  color='red', label=r'FeCl$_2$ (Fontana+2007)')
    ax.plot(data_wav['FeSO4_azra'], data_abs['FeSO4_azra'] , linewidth=2, linestyle='-',  color='blue', label=r'FeSO$_4$ (Fontana+2007)')
    ax.plot(data_wav['ross_ferricyanide'], data_abs['ross_ferricyanide'] , linewidth=2, linestyle='-',  color='hotpink', label=r'K$_3$Fe(CN)$_6$ (Ross+2018)')
    # ax.plot(data_wav['tabata_feoh'], data_abs['tabata_feoh'] , linewidth=2, linestyle='-',  color='orange', label=r'FeOH$^+$ (Tabata+2021)')

    ax.errorbar(data_wav['LK'], data_abs['Fe(BF4)2_LK'], data_abs['Fe(BF4)2_err_LK'], linewidth=1, linestyle='-', marker='o', color='black', label=r'Fe(BF$_4$)$_2$ (This Work)', capsize=5)

    ax.errorbar(data_wav['LK'], data_abs['K_ferrocyanide_LK'],yerr=data_abs['K_ferrocyanide_err_LK'], linewidth=1, linestyle='--', color='purple', label=r'K$_4$Fe(CN)$_6$ (This Work)',capsize=2)

    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
#    ax.set_ylim([1E-1, 1.0E0])
    
    plt.savefig('./Plots/molecules/SNP_etc.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/SNP.jpg', orientation='portrait', format='jpg')
    
    
if plotferrocyanide: 
    #We need to do.
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'Ferrocyanide')    
    ax.errorbar(data_wav['LK'], data_abs['K_ferrocyanide_LK'],yerr=data_abs['K_ferrocyanide_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'K$_4$Fe(CN)$_6$ (This Work)',capsize=5)

    ax.errorbar(data_wav['ross_ferrocyanide'], data_abs['ross_ferrocyanide'], yerr=0, linewidth=2, linestyle='-', color='red', label=r'K$_4$Fe(CN)$_6$ (Ross+2018)')


    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
#    ax.set_ylim([1E-1, 1.0E0])
    
    plt.savefig('./Plots/molecules/ferrocyanide.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/ferrocyanide.jpg', orientation='portrait', format='jpg')


if plot_NO3:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'NaNO$_3$')    
    ax.errorbar(data_wav['LK'], data_abs['NaNO3_LK'], data_abs['NaNO3_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'NaNO$_3$ (This Work)', capsize=5)
    ax.errorbar(data_wav['birk_NO3OH'], data_abs['birk_NO3'], data_abs['birk_err_NO3'], linewidth=2, linestyle='-', marker='o', color='red', label=r'Birkmann+2018', capsize=5)


    ax.errorbar(data_wav['NO3_azra'], data_abs['NO3_azra'] , yerr=0, linewidth=2, linestyle='--', color='blue', label=r'NO$_3^-$ (Mack & Bolton 1999)')

    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/NaNO3.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/NaNO3.jpg', orientation='portrait', format='jpg') 

if plot_CO3_HCO3:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'NaNO$_3$')    

    ax.errorbar(data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'] , yerr=data_abs['birk_err_CO3'], linewidth=2, linestyle='-', marker='d', color='black', label=r'CO$_3^{2-}$ (Birkmann+2018)', capsize=5)
    ax.errorbar(data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'] , yerr=data_abs['birk_err_HCO3'],linewidth=2, linestyle='-', marker='d', color='red', label=r'HCO$_3^-$ (Birkmann+2018)', capsize=5)

    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/CO3_HCO3.pdf', orientation='portrait', format='pdf')

if plot_HSm:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)


    ax.plot(data_wav['guenther_hsm'], data_abs['guenther_hsm'] , linewidth=2, linestyle='-',  color='black', label=r'HS$^{-}$ (Guenther+2001)')
    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/HSm.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/NaNO3.jpg', orientation='portrait', format='jpg')            

if plotHSO3m:
    #Buck+1954 for related Na2SO3
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'NaHSO$_3$')    
    # ax.plot(data_wav['golding_hso3'], data_abs['golding_hso3'] , linewidth=2, linestyle='-', color='red', label=r'HSO$_3^-$ (Golding 1960)')
    # ax.plot(data_wav['hayon_hso3'], data_abs['hayon_hso3'] , linewidth=2, linestyle='-', color='orange', label=r'HSO$_3^-$ (Hayon+1972)')
    ax.plot(data_wav['hso3_composite'], data_abs['hso3_composite'] , linewidth=4, linestyle='-', color='red', label=r'HSO$_3^-$ (Composite)')
    ax.plot(data_wav['beyad'], data_abs['beyad_hso3'] , linewidth=2, linestyle='-', color='black', marker='d', label=r'HSO$_3^-$ (Beyad+2014)')
    ax.plot(data_wav['fischer_hso3'], data_abs['fischer_hso3'], linewidth=2, linestyle='-', color='purple', label=r'HSO$_3^-$ (Fischer+1996)')
    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    ####
    #Save spectrum for use by other codes. 
    ####
    towrite=np.zeros((len(data_wav['hso3_composite']), 2))
    towrite[:,0]=data_wav['hso3_composite']
    towrite[:,1]=data_abs['hso3_composite']
   
    np.savetxt('./GeneratedSpectra/bisulfite_spectrum.dat', towrite, delimiter=' ', fmt='%3.2f %1.6e', newline='\n', header='Wavelength     HSO3- Molar Decadic Absorption Coefficient \n (nm)   (M$^{-1}$cm$^{-1}$)')

    
    plt.savefig('./Plots/molecules/NaHSO3.pdf', orientation='portrait', format='pdf')
    # plt.savefig('./Plots/molecules/NaHSO3.jpg', orientation='portrait',papertype='letter', format='jpg')
#    
    
if plotSO3m:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    # ax.set_title(r'SO$_3^{2-}$')    
    # ax.plot(data_wav['buck_Na2SO3'], data_abs['buck_Na2SO3'] , linewidth=2, linestyle='-', color='blue', label=r'Na$_2$SO$_3$ (Buck+1954)')
    # ax.plot(data_wav['golding_so3'], data_abs['golding_so3'] , linewidth=2, linestyle='-', color='yellow', label=r'SO$_3^-$ (Golding 1960)')
    # ax.plot(data_wav['hayon_so3'], data_abs['hayon_so3'] , linewidth=2, linestyle='-', color='green', label=r'SO$_3^-$ (Hayon+1972)')    
    ax.plot(data_wav['so3_composite'], data_abs['so3_composite'] , linewidth=4, linestyle='-', color='red', label=r'SO$_3^{2-}$ (Composite)')    
    ax.plot(data_wav['beyad'], data_abs['beyad_so3'] , linewidth=2, linestyle='-',marker='d', color='black', label=r'SO$_3^{2-}$ (Beyad+2014)')    
    ax.plot(data_wav['fischer_so3'], data_abs['fischer_so3'] , linewidth=2, linestyle='-', color='purple', label=r'SO$_3^{2-}$ (Fischer+1996)')    


    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    ####
    #Save spectrum for use by other codes. 
    ####
    towrite=np.zeros((len(data_wav['so3_composite']), 2))
    towrite[:,0]=data_wav['so3_composite']
    towrite[:,1]=data_abs['so3_composite']
   
    np.savetxt('./GeneratedSpectra/sulfite_spectrum.dat', towrite, delimiter=' ', fmt='%3.2f %1.6e', newline='\n', header='Wavelength SO3[2-] Molar Decadic Absorption Coefficient \n (nm)   (M$^{-1}$cm$^{-1}$)')
    
    plt.savefig('./Plots/molecules/SO3m.pdf', orientation='portrait', format='pdf')
    # plt.savefig('./Plots/molecules/SO3m.jpg', orientation='portrait',papertype='letter', format='jpg')

if plot_halide_ferrous_ocean:    
    conc_NaCl_mod=0.6 #M
    conc_NaBr_mod=0.9*1.0E-3 #M
    conc_NaI_mod=0.5*1.0E-6 #M
    
    halide_low=0.5# lower estimate of salinity
    halide_high=2.0#upper estimate of salinity
 
    conc_fe_low=1.0E-9 #M. Halevy+2017, green rust paper. Big difference due to photooxidation 
    conc_fe_high=100.0E-6 #M. Via Konhauser+2017, Zheng+2018 (not published)

    conc_hco3_low=2E-3#M
    conc_hco3_high=0.2#M
    
    conc_co3_low=0.2E-6 #M
    conc_co3_high=1.0E-3 #M

    conc_NO3_low=1.0E-11
    conc_NO3_high=1.0E-6
    
    #get error interpolated from lit data
    thunk, h2o_err_interp=linear_interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, h2o_quickenden_abs_err)
    thunk, co3_err_interp=linear_interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], data_abs['birk_err_CO3'])
    thunk, hco3_err_interp=linear_interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], data_abs['birk_err_HCO3'])
    
    ocean_low=halide_low*(conc_NaCl_mod*data_abs['NaCl_LK']+conc_NaBr_mod*data_abs['NaBr_LK']+conc_NaI_mod*data_abs['I_LK']) + conc_fe_low*data_abs['Fe(BF4)2_LK'] + conc_NO3_low*data_abs['NaNO3_LK'] + conc_hco3_low*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0) + conc_co3_low*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0) + np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0)
    
    ocean_low_err=np.sqrt((conc_fe_low*data_abs['Fe(BF4)2_err_LK'])**2.0 + (conc_NO3_low*data_abs['NaNO3_err_LK'])**2.0 + (h2o_err_interp)**2.0 + (conc_co3_low*co3_err_interp)**2.0 + (conc_hco3_low*hco3_err_interp)**2.0 + (halide_low**2.0)*((conc_NaCl_mod*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr_mod*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI_mod*data_abs['I_err_LK'])**2.0))
    
    ocean_high=halide_high*(conc_NaCl_mod*data_abs['NaCl_LK']+conc_NaBr_mod*data_abs['NaBr_LK']+conc_NaI_mod*data_abs['I_LK']) + conc_fe_high*data_abs['Fe(BF4)2_LK'] + conc_NO3_high*data_abs['NaNO3_LK'] + conc_hco3_high*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0) + conc_co3_high*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0) + np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0)
    
    ocean_high_err=np.sqrt((conc_fe_high*data_abs['Fe(BF4)2_err_LK'])**2.0 + (conc_NO3_high*data_abs['NaNO3_err_LK'])**2.0 +(h2o_err_interp)**2.0 + (conc_co3_high*co3_err_interp)**2.0 + (conc_hco3_high*hco3_err_interp)**2.0 + (halide_high**2.0)*((conc_NaCl_mod*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr_mod*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI_mod*data_abs['I_err_LK'])**2.0))
    
    fig, ax=plt.subplots(2, figsize=(8., 9.), sharex=True, sharey=True)   
    ax[0].set_title('Ocean, Low-Absorption Endmember')
    ax[0].errorbar(data_wav['LK'], ocean_low, yerr=ocean_low_err, linewidth=2, linestyle='-', marker='o', color='black', label='Total')
    ax[0].errorbar(data_wav['LK'], halide_low*conc_NaCl_mod*data_abs['NaCl_LK'], yerr=halide_low*conc_NaCl_mod*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_yellow'], label='Cl$^-$')
    ax[0].errorbar(data_wav['LK'], halide_low*conc_NaBr_mod*data_abs['NaBr_LK'], yerr=halide_low*conc_NaBr_mod*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['strong_purple'], label='Br$^-$')
    ax[0].errorbar(data_wav['LK'], halide_low*conc_NaI_mod*data_abs['I_LK'], yerr=halide_low*conc_NaI_mod*data_abs['I_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_orange'], label='I$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NO3_low*data_abs['NaNO3_LK'] , yerr=conc_NO3_low*data_abs['NaNO3_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['very_light_blue'], label='NO$_3^{-}$')    
    ax[0].errorbar(data_wav['LK'], conc_fe_low*data_abs['Fe(BF4)2_LK'], yerr=conc_fe_low*data_abs['Fe(BF4)2_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_red'], label='Fe$^{2+}$ ')    
    ax[0].errorbar(data_wav['LK'], conc_hco3_low*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0), yerr=conc_hco3_low*hco3_err_interp, linestyle='--', color=kelly_colors['grayish_yellow'], label='HCO$_3^-$')
    ax[0].errorbar(data_wav['LK'], conc_co3_low*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0), yerr=conc_co3_low*co3_err_interp,  linestyle='--', color=kelly_colors['medium_gray'], label='CO$_3^{2-}$')
    ax[0].errorbar(data_wav['LK'],np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0),yerr=h2o_err_interp, linestyle='--', color=kelly_colors['strong_violet'], label='H$_2$O')

    ax[1].set_title('Ocean, High-Absorption Endmember')
    ax[1].errorbar(data_wav['LK'], ocean_high, yerr=ocean_high_err, linewidth=2, linestyle='-', marker='o', color='black', label='Total')
    ax[1].errorbar(data_wav['LK'], halide_high*conc_NaCl_mod*data_abs['NaCl_LK'], yerr=halide_high*conc_NaCl_mod*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_yellow'], label='Cl$^-$')
    ax[1].errorbar(data_wav['LK'], halide_high*conc_NaBr_mod*data_abs['NaBr_LK'], yerr=halide_high*conc_NaBr_mod*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['strong_purple'], label='Br$^-$')
    ax[1].errorbar(data_wav['LK'], halide_high*conc_NaI_mod*data_abs['I_LK'], yerr=halide_high*conc_NaI_mod*data_abs['I_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_orange'], label='I$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NO3_high*data_abs['NaNO3_LK'] , yerr=conc_NO3_high*data_abs['NaNO3_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['very_light_blue'], label='NO$_3^{-}$')    
    ax[1].errorbar(data_wav['LK'], conc_fe_high*data_abs['Fe(BF4)2_LK'], yerr=conc_fe_high*data_abs['Fe(BF4)2_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_red'], label='Fe$^{2+}$')    
    ax[1].errorbar(data_wav['LK'], conc_hco3_high*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0),  yerr=conc_hco3_high*hco3_err_interp, linestyle='--', color=kelly_colors['grayish_yellow'], label='HCO$_3^-$')
    ax[1].errorbar(data_wav['LK'], conc_co3_high*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0),  yerr=conc_co3_high*co3_err_interp, linestyle='--', color=kelly_colors['medium_gray'], label='CO$_3^{2-}$')
    ax[1].errorbar(data_wav['LK'],np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0),yerr=h2o_err_interp,linestyle='--', color=kelly_colors['strong_violet'], label='H$_2$O')


    
    ax[0].legend(ncol=1, loc='best')
    # ax[1].legend(ncol=1, loc='best')

    ax[0].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')    
    ax[1].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')
    
    ax[1].set_xscale('linear')
    ax[1].set_xlabel('Wavelength (nm)')    
    ax[1].set_yscale('log')
    ax[1].set_ylim([1E-3, 1.0E1]) 
    ax[1].set_xlim([200., 300.])
    
    ax[0].axhline(1.0E0, color='black', linestyle=':')
    ax[0].axhline(1.0E-1, color='black', linestyle=':')
    ax[0].axhline(1.0E-2, color='black', linestyle=':')
    ax[0].axhline(1.0E-3, color='black', linestyle=':')

    ax[1].axhline(1.0E0, color='black', linestyle=':')
    ax[1].axhline(1.0E-1, color='black', linestyle=':')
    ax[1].axhline(1.0E-2, color='black', linestyle=':')
    ax[1].axhline(1.0E-3, color='black', linestyle=':')
    
    plt.savefig('./Plots/molecules/prebiotic-ocean.pdf', orientation='portrait', papertype='letter', format='pdf')
    
    
    ###Quantities for paper text
    #low-absorption, 220 nm
    ind=np.where(data_wav['LK']==220.0)
    print(r'Ocean low, 220 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/ocean_low[ind], np.log10(np.exp(1.0))*ocean_low_err[ind]/ocean_low[ind]**2.0))
    
    #high-absorption, 260 nm
    ind=np.where(data_wav['LK']==260.0)
    print(r'Ocean high, 260 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/ocean_high[ind], np.log10(np.exp(1.0))*ocean_high_err[ind]/ocean_high[ind]**2.0))

#Plot prebiotic lake (freshwater)
if plot_halide_freshwater_lake:
    #These are all mean values
    conc_NaCl=0.2*1.0E-3 #M
    conc_NaBr=0.15*1.0E-6 #M
    conc_NaI=40.*1.0E-9 #M
    conc_NO3=0.05E-9 #M
    conc_hco3=1.0E-3 #M
    conc_co3=100.0E-9 #M
    conc_fe_low=0.1E-6#M
    conc_hso3_low=0.0#*1.0E-6 #M
    conc_so3_low=0.0#*0.8E-6 #M

    conc_NaI_high=0.6*1E-6 #M
    conc_NO3_high=10E-6 #M
    conc_fe_high=0.1E-3 #M
    conc_hs_high=8.0E-11 #M
    conc_hso3_high=200.0E-6 #M
    conc_so3_high=100.0E-6 #M
    
    #get error interpolated from lit data
    thunk, h2o_err_interp=linear_interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, h2o_quickenden_abs_err)
    thunk, co3_err_interp=linear_interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], data_abs['birk_err_CO3'])
    thunk, hco3_err_interp=linear_interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], data_abs['birk_err_HCO3'])    
    
    freshwater_lake=conc_NaCl*data_abs['NaCl_LK'] + conc_NaBr*data_abs['NaBr_LK'] + conc_NaI*data_abs['I_LK'] + conc_NO3*data_abs['NaNO3_LK'] + conc_fe_low*data_abs['Fe(BF4)2_LK'] + conc_hco3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0) + conc_co3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0) + conc_hso3_low*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0) + conc_so3_low*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0) + np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0)
    
    freshwater_lake_err=np.sqrt( (conc_NaCl*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI*data_abs['I_err_LK'])**2.0 + (conc_NO3*data_abs['NaNO3_err_LK'])**2.0 + (conc_fe_low*data_abs['Fe(BF4)2_err_LK'])**2.0 + (h2o_err_interp)**2.0 + (conc_co3*co3_err_interp)**2.0 + (conc_hco3*hco3_err_interp)**2.0 )
    
    freshwater_lake_high=conc_NaCl*data_abs['NaCl_LK'] + conc_NaBr*data_abs['NaBr_LK'] + conc_NaI_high*data_abs['I_LK'] + conc_NO3_high*data_abs['NaNO3_LK'] + conc_fe_high*data_abs['Fe(BF4)2_LK'] + conc_hco3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0) + conc_co3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0) + conc_hs_high*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0) + conc_hso3_high*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0) + conc_so3_high*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0) + np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0)
    
    freshwater_lake_high_err=np.sqrt( (conc_NaCl*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI_high*data_abs['I_err_LK'])**2.0 + (conc_NO3_high*data_abs['NaNO3_err_LK'])**2.0 + (conc_fe_high*data_abs['Fe(BF4)2_err_LK'])**2.0 + (h2o_err_interp)**2.0 + (conc_co3*co3_err_interp)**2.0 + (conc_hco3*hco3_err_interp)**2.0)
    
    fig, ax=plt.subplots(2, figsize=(8., 9.), sharex=True, sharey=True)
   
    ax[0].set_title('Freshwater Lake, Low-Absorption Endmember')
    ax[0].errorbar(data_wav['LK'], freshwater_lake, yerr=freshwater_lake_err, linewidth=2, linestyle='-', marker='o', color='black', label='Total')
    ax[0].errorbar(data_wav['LK'], conc_NaCl*data_abs['NaCl_LK'], yerr=conc_NaCl*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_yellow'], label='Cl$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NaBr*data_abs['NaBr_LK'], yerr=conc_NaBr*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['strong_purple'], label='Br$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NaI*data_abs['I_LK'], yerr=conc_NaI*data_abs['I_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_orange'], label='I$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NO3*data_abs['NaNO3_LK'] , yerr=conc_NO3*data_abs['NaNO3_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['very_light_blue'], label='NO$_3^{-}$')    
    ax[0].errorbar(data_wav['LK'], conc_fe_low*data_abs['Fe(BF4)2_LK'], yerr=conc_fe_low*data_abs['Fe(BF4)2_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_red'], label='Fe$^{2+}$')    
    ax[0].errorbar(data_wav['LK'], conc_hco3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0), yerr=conc_hco3*hco3_err_interp, linestyle='--', color=kelly_colors['grayish_yellow'], label='HCO$_3^-$')
    ax[0].errorbar(data_wav['LK'], conc_co3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0), yerr=conc_co3*co3_err_interp, linestyle='--', color=kelly_colors['medium_gray'], label='CO$_3^{2-}$')
    ax[0].plot(data_wav['LK'],conc_hso3_low*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['vivid_green'], label='HSO$_3^-$')
    ax[0].plot(data_wav['LK'],conc_so3_low*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['strong_purplish_pink'], label='SO$_3^{2-}$')
    ax[0].plot(data_wav['LK'],0.0*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0),linestyle='--', color=kelly_colors['strong_blue'], label='HS$^{-}$')
    ax[0].errorbar(data_wav['LK'],np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0),yerr=h2o_err_interp, linestyle='--', color=kelly_colors['strong_violet'], label='H$_2$O')

    ax[1].set_title('Freshwater Lake, High-Absorption Endmember')
    ax[1].errorbar(data_wav['LK'], freshwater_lake_high, yerr=freshwater_lake_high_err, linewidth=2, linestyle='-', marker='o', color='black', label='Total')
    ax[1].errorbar(data_wav['LK'], conc_NaCl*data_abs['NaCl_LK'], yerr=conc_NaCl*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_yellow'], label='Cl$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NaBr*data_abs['NaBr_LK'], yerr=conc_NaBr*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['strong_purple'], label='Br$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NaI_high*data_abs['I_LK'], yerr=conc_NaI_high*data_abs['I_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_orange'], label='I$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NO3_high*data_abs['NaNO3_LK'] , yerr=conc_NO3_high*data_abs['NaNO3_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['very_light_blue'], label='NO3')    
    ax[1].errorbar(data_wav['LK'], conc_fe_high*data_abs['Fe(BF4)2_LK'], yerr=conc_fe_high*data_abs['Fe(BF4)2_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_red'], label='Fe$^{2+}$')    
    ax[1].errorbar(data_wav['LK'], conc_hco3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0), yerr=conc_hco3*hco3_err_interp, linestyle='--', color=kelly_colors['grayish_yellow'], label='HCO$_3^-$')
    ax[1].errorbar(data_wav['LK'], conc_co3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0), yerr=conc_co3*co3_err_interp, linestyle='--', color=kelly_colors['medium_gray'], label='CO$_3^{2-}$')
    ax[1].plot(data_wav['LK'],conc_hso3_high*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['vivid_green'], label='HSO$_3^-$')
    ax[1].plot(data_wav['LK'],conc_so3_high*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['strong_purplish_pink'], label='SO$_3^{2-}$')
    ax[1].plot(data_wav['LK'],conc_hs_high*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0),linestyle='--', color=kelly_colors['strong_blue'], label='HS$^{-}$')
    ax[1].errorbar(data_wav['LK'],np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0),yerr=h2o_err_interp, linestyle='--', color=kelly_colors['strong_violet'], label='H$_2$O')


    
    ax[0].legend(ncol=1, loc='best')
    # ax[1].legend(ncol=1, loc='best')

    ax[0].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')    
    ax[1].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')
    
    ax[1].set_xscale('linear')
    ax[1].set_xlabel('Wavelength (nm)')    
    ax[1].set_yscale('log')
    ax[1].set_ylim([1E-3, 1.0E-0]) 
    ax[1].set_xlim([200., 300.])
    
    ax[0].axhline(1.0E0, color='black', linestyle=':')
    ax[0].axhline(1.0E-1, color='black', linestyle=':')
    ax[0].axhline(1.0E-2, color='black', linestyle=':')
    ax[0].axhline(1.0E-3, color='black', linestyle=':')

    ax[1].axhline(1.0E0, color='black', linestyle=':')
    ax[1].axhline(1.0E-1, color='black', linestyle=':')
    ax[1].axhline(1.0E-2, color='black', linestyle=':')
    ax[1].axhline(1.0E-3, color='black', linestyle=':')
    
    plt.savefig('./Plots/molecules/freshwater-lake.pdf', orientation='portrait', format='pdf')
    
    # print('Freshwater lake, nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', freshwater_lake, data_wav['LK'], 100.0)))
    # print('Freshwater lake, cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', freshwater_lake, data_wav['LK'], 100.0)))
    # print('Freshwater lake, 2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', freshwater_lake, data_wav['LK'], 100.0)))
    # print('Freshwater lake, 2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', freshwater_lake, data_wav['LK'], 100.0)))
    # print('Freshwater lake, 2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', freshwater_lake, data_wav['LK'], 100.0)))
    # print()
    # print('High-I Freshwater lake, nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', freshwater_lake_high, data_wav['LK'], 100.0)))
    # print('High-I Freshwater lake, cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', freshwater_lake_high, data_wav['LK'], 100.0)))
    # print('High-I Freshwater lake, 2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', freshwater_lake_high, data_wav['LK'], 100.0)))
    # print('High-I Freshwater lake, 2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', freshwater_lake_high, data_wav['LK'], 100.0)))
    # print('High-I Freshwater lake, 2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', freshwater_lake_high, data_wav['LK'], 100.0)))
    # print()
    
    ###Quantities for paper text
    #low-absorption, 220 nm
    ind=np.where(data_wav['LK']==235.0)
    print(r'Freshwater high, 235 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/freshwater_lake_high[ind], np.log10(np.exp(1.0))*freshwater_lake_high_err[ind]/freshwater_lake_high[ind]**2.0))
    
    #high-absorption, 260 nm
    ind=np.where(data_wav['LK']==260.0)
    print(r'Freshwater high, 260 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/freshwater_lake_high[ind], np.log10(np.exp(1.0))*freshwater_lake_high_err[ind]/freshwater_lake_high[ind]**2.0))


    
#Plot prebiotic lake (closed-basin carbonate/phosphate rich)
if plot_halide_carbonate_lake:
    #Low
    conc_NaBr_low=1.0E-3 #M
    conc_NaCl_low=1.0E-1 #M
    conc_NaI_low=40.*1.0E-9#M
    conc_no3_low=5.0E-9#M
    conc_hs_low=0.0# 3.0E-11 #M
    conc_hso3_low=0.0#1.0E-6 #M
    conc_so3_low=0.0#0.8E-6 #M
    conc_hco3_low=50.0E-3 #M
    conc_co3_low=7.0E-6 #M
    
    #High
    conc_NaBr_high=10.0*1E-3 #M
    conc_NaCl_high=6.0#M
    conc_NaI_high=600.0*1E-9 #M
    conc_no3_high=1.0E-3 #M
    conc_hs_high=8.0E-9#M
    conc_hso3_high=200.0E-6 #M
    conc_so3_high=400.0E-6#M
    conc_hco3_high=0.1 #M
    conc_co3_high=7.0E-3 #M
 
    #get error interpolated from lit data
    thunk, h2o_err_interp=linear_interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, h2o_quickenden_abs_err)
    thunk, co3_err_interp=linear_interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], data_abs['birk_err_CO3'])
    thunk, hco3_err_interp=linear_interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], data_abs['birk_err_HCO3'])
    
    carbonate_lake=conc_NaCl_low*data_abs['NaCl_LK'] + conc_NaBr_low*data_abs['NaBr_LK'] + conc_NaI_low*data_abs['I_LK'] + conc_no3_low*data_abs['NaNO3_LK'] + conc_hs_low*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0) + conc_hco3_low*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0) + conc_co3_low*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0) + conc_hso3_low*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0) + conc_so3_low*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0) + np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0)
    
    carbonate_lake_err=np.sqrt( (conc_NaCl_low*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr_low*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI_low*data_abs['I_err_LK'])**2.0 + (conc_no3_low*data_abs['NaNO3_err_LK'])**2.0 + (h2o_err_interp)**2.0 + (conc_co3_low*co3_err_interp)**2.0 + (conc_hco3_low*hco3_err_interp)**2.0)
    
    carbonate_lake_high=conc_NaCl_high*data_abs['NaCl_LK'] + conc_NaBr_high*data_abs['NaBr_LK'] + conc_NaI_high*data_abs['I_LK'] + conc_no3_high*data_abs['NaNO3_LK'] + conc_hs_high*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0) + conc_hco3_high*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0) + conc_co3_high*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0)  + conc_hso3_high*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0) + conc_so3_high*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0) + np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0)
    
    carbonate_lake_high_err=np.sqrt( (conc_NaCl_high*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr_high*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI_high*data_abs['I_err_LK'])**2.0 + (conc_no3_high*data_abs['NaNO3_err_LK'])**2.0 + (h2o_err_interp)**2.0 + (conc_co3_high*co3_err_interp)**2.0 + (conc_hco3_high*hco3_err_interp)**2.0)
    
    fig, ax=plt.subplots(2, figsize=(8., 9.), sharex=True, sharey=True)
   
    ax[0].set_title('Carbonate Lake, Low-Absorption Endmember')
    ax[0].errorbar(data_wav['LK'], carbonate_lake, yerr=carbonate_lake_err, linewidth=2, linestyle='-', marker='o', color='black', label='Total')
    ax[0].errorbar(data_wav['LK'], conc_NaCl_low*data_abs['NaCl_LK'], yerr=conc_NaCl_low*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_yellow'], label='Cl$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NaBr_low*data_abs['NaBr_LK'], yerr=conc_NaBr_low*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['strong_purple'], label='Br$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NaI_low*data_abs['I_LK'], yerr=conc_NaI_low*data_abs['I_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_orange'], label='I$^-$')
    ax[0].errorbar(data_wav['LK'], conc_no3_low*data_abs['NaNO3_LK'] , yerr=conc_no3_low*data_abs['NaNO3_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['very_light_blue'], label='NO$_3^{-}$')    
    ax[0].errorbar(data_wav['LK'], conc_hco3_low*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0), yerr=conc_hco3_low*hco3_err_interp, linestyle='--', color=kelly_colors['grayish_yellow'], label='HCO$_3^-$')
    ax[0].errorbar(data_wav['LK'], conc_co3_low*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0), yerr=conc_co3_low*co3_err_interp, linestyle='--', color=kelly_colors['medium_gray'], label='CO$_3^{2-}$')
    ax[0].plot(data_wav['LK'],conc_hso3_low*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['vivid_green'], label='HSO$_3^-$')
    ax[0].plot(data_wav['LK'],conc_so3_low*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['strong_purplish_pink'], label='SO$_3^{2-}$')
    ax[0].plot(data_wav['LK'],conc_hs_low*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0),linestyle='--', color=kelly_colors['strong_blue'], label='HS$^{-}$')
    ax[0].errorbar(data_wav['LK'],np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0), yerr=h2o_err_interp, linestyle='--', color=kelly_colors['strong_violet'], label='H$_2$O')


    ax[1].set_title('Carbonate Lake, High-Absorption Endmember')
    ax[1].errorbar(data_wav['LK'], carbonate_lake_high, yerr=carbonate_lake_high_err, linewidth=2, linestyle='-', marker='o', color='black', label='Total')
    ax[1].errorbar(data_wav['LK'], conc_NaCl_high*data_abs['NaCl_LK'], yerr=conc_NaCl_high*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_yellow'], label='Cl$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NaBr_high*data_abs['NaBr_LK'], yerr=conc_NaBr_high*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['strong_purple'], label='Br$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NaI_high*data_abs['I_LK'], yerr=conc_NaI_high*data_abs['I_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_orange'], label='I$^-$')
    ax[1].errorbar(data_wav['LK'], conc_no3_high*data_abs['NaNO3_LK'] , yerr=conc_no3_high*data_abs['NaNO3_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['very_light_blue'], label='NO$_3^-$')    
    ax[1].errorbar(data_wav['LK'], conc_hco3_high*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0), yerr=conc_hco3_high*hco3_err_interp, linestyle='--', color=kelly_colors['grayish_yellow'], label='HCO$_3^-$')
    ax[1].errorbar(data_wav['LK'], conc_co3_high*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0), yerr=conc_co3_high*co3_err_interp, linestyle='--', color=kelly_colors['medium_gray'], label='CO$_3^{2-}$')
    ax[1].plot(data_wav['LK'],conc_hso3_high*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['vivid_green'], label='HSO$_3^-$')
    ax[1].plot(data_wav['LK'],conc_so3_high*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['strong_purplish_pink'], label='SO$_3^{2-}$')
    ax[1].plot(data_wav['LK'],conc_hs_high*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0),linestyle='--', color=kelly_colors['strong_blue'], label='HS$^{-}$')
    ax[1].errorbar(data_wav['LK'],np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0),yerr=h2o_err_interp, linestyle='--', color=kelly_colors['strong_violet'], label='H$_2$O')


    
    ax[0].legend(ncol=1, loc='best')
    # ax[1].legend(ncol=1, loc='best')

    ax[0].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')    
    ax[1].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')
    
    ax[1].set_xscale('linear')
    ax[1].set_xlabel('Wavelength (nm)')    
    ax[1].set_yscale('log')
    ax[1].set_ylim([1E-3, 1.0E2]) 
    ax[1].set_xlim([200., 300.])

    ax[0].axhline(1.0E1, color='black', linestyle=':')    
    ax[0].axhline(1.0E0, color='black', linestyle=':')
    ax[0].axhline(1.0E-1, color='black', linestyle=':')
    ax[0].axhline(1.0E-2, color='black', linestyle=':')
    ax[0].axhline(1.0E-3, color='black', linestyle=':')

    ax[1].axhline(1.0E1, color='black', linestyle=':')
    ax[1].axhline(1.0E0, color='black', linestyle=':')
    ax[1].axhline(1.0E-1, color='black', linestyle=':')
    ax[1].axhline(1.0E-2, color='black', linestyle=':')
    ax[1].axhline(1.0E-3, color='black', linestyle=':')
    
    plt.savefig('./Plots/molecules/carbonate-lake.pdf', orientation='portrait', format='pdf')

    ######
    #Plot UV in simulated reservoir
    ######
    ##Project pure water absorption to relevant wavelength scale, superpose
    carbonate_lake_transmission=np.interp(uv_wav, data_wav['LK'], carbonate_lake, left=0, right=0)  #set absorptions to 0 where therem is no data; completely unphysical, but we shouldn't (and don't need to) trust those data anyway. Better to know clearly when things are full of shit. 
    thunk, carbonate_lake_transmission_err=linear_interp(uv_wav, data_wav['LK'], carbonate_lake,carbonate_lake_err)
    
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True, sharey=True)
    ax.set_title('Upper Bounds on Carbonate Lake UV')
    ax.plot(uv_wav, uv_toa_intensity, linewidth=2, linestyle='-', marker='o', color='black', label='Stellar Irradiation')
    ax.plot(uv_wav, uv_surface_intensity, linewidth=2, linestyle='-', marker='o', color='purple', label='Surface')
    # ax.plot(uv_wav, uv_surface_intensity*10.0**(-carbonate_lake_transmission*1.0/0.5), linewidth=2, linestyle='-', marker='o', color='blue', label='1 cm')
    T_1=uv_surface_intensity*10.0**(-carbonate_lake_transmission*1.0/0.5)
    T_1_err=T_1*((1.0/0.5)*np.log(10.0))*carbonate_lake_transmission_err
    ax.errorbar(uv_wav, T_1, yerr=T_1_err, linewidth=2, linestyle='-', marker='o', color='blue', label='1 cm')
    # ax.plot(uv_wav, uv_surface_intensity*10.0**(-carbonate_lake_transmission*10.0/0.5), linewidth=2, linestyle='-', marker='o', color='cyan', label='10 cm')
    T_10=uv_surface_intensity*10.0**(-carbonate_lake_transmission*10.0/0.5)
    T_10_err=T_10*((10.0/0.5)*np.log(10.0))*carbonate_lake_transmission_err
    ax.errorbar(uv_wav,T_10,yerr=T_10_err, linewidth=2, linestyle='-', marker='o', color='cyan', label='10 cm')
    # ax.plot(uv_wav, uv_surface_intensity*10.0**(-carbonate_lake_transmission*100.0/0.5), linewidth=2, linestyle='-', marker='o', color='green', label='100 cm')
    T_100=uv_surface_intensity*10.0**(-carbonate_lake_transmission*100.0/0.5)
    T_100_err=T_100*((100.0/0.5)*np.log(10.0))*carbonate_lake_transmission_err
    ax.errorbar(uv_wav, T_100, yerr=T_100_err, linewidth=2, linestyle='-', marker='o', color='green', label='100 cm')

    
    ax.legend(ncol=1, loc='best')
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')    
    ax.set_yscale('log')
    ax.set_ylabel(r'Intensity (photons cm$^{-2}$ s$^{-1}$ nm$^{-1}$)')
    ax.set_ylim([1E8, 1.0E14]) 
    ax.set_xlim([200., 300.])
    plt.savefig('./Plots/molecules/uv_carbonate-lake.eps', orientation='portrait', format='eps')
    
    # ####
    # ##
    # ####
    # print('Carbonate lake, nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', carbonate_lake, data_wav['LK'], 100.0)))
    # print('Carbonate lake, cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', carbonate_lake, data_wav['LK'], 100.0)))
    # print('Carbonate lake, 2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', carbonate_lake, data_wav['LK'], 100.0)))
    # print('Carbonate lake, 2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', carbonate_lake, data_wav['LK'], 100.0)))
    # print('Carbonate lake, 2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', carbonate_lake, data_wav['LK'], 100.0)))
    # print()
    # print('Carbonate lake (High), nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', carbonate_lake_high, data_wav['LK'], 10.0)))
    # print('Carbonate lake (High), cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', carbonate_lake_high, data_wav['LK'], 10.0)))
    # print('Carbonate lake (High), 2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', carbonate_lake_high, data_wav['LK'], 10.0)))
    # print('Carbonate lake (High), 2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', carbonate_lake_high, data_wav['LK'], 10.0)))
    # print('Carbonate lake (High), 2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', carbonate_lake_high, data_wav['LK'], 10.0)))
    # print()
    
    ###Quantities for paper text
    #low-absorption, 220 nm
    ind=np.where(data_wav['LK']==220.0)
    print(r'Carbonate low, 220 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/carbonate_lake[ind], np.log10(np.exp(1.0))*carbonate_lake_err[ind]/carbonate_lake[ind]**2.0))
    
    #high-absorption, 260 nm
    ind=np.where(data_wav['LK']==260.0)
    print(r'Carbonate high, 260 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/carbonate_lake_high[ind], np.log10(np.exp(1.0))*carbonate_lake_high_err[ind]/carbonate_lake_high[ind]**2.0))
    
    ind=np.where(data_wav['LK']==230.0)
    print(r'Carbonate high, 230 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/carbonate_lake_high[ind], np.log10(np.exp(1.0))*carbonate_lake_high_err[ind]/carbonate_lake_high[ind]**2.0))
    
#########
#########
#########
    
if plot_ferrous_lake:
    conc_NaCl=0.2*1.0E-3 #M
    conc_NaBr=0.15*1.0E-6 #M
    conc_NaI=40.*1.0E-9 #M
    conc_NO3=0.05E-9 #M
    conc_hco3=1.0E-3 #M
    conc_co3=100.0E-9 #M
    conc_ferrocyanide_low=0.1E-6#M
    conc_hso3_low=0.0#1.0E-6 #M
    conc_so3_low=0.0#0.8E-6 #M

    conc_NaI_high=0.6*1E-6 #M
    conc_NO3_high=10E-6 #M
    conc_ferrocyanide_high=0.1E-3 #M
    conc_hs_high=8.0E-11 #M
    conc_hso3_high=200.0E-6 #M
    conc_so3_high=100.0E-6 #M

    #get error interpolated from lit data
    thunk, h2o_err_interp=linear_interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, h2o_quickenden_abs_err)
    thunk, co3_err_interp=linear_interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], data_abs['birk_err_CO3'])
    thunk, hco3_err_interp=linear_interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], data_abs['birk_err_HCO3'])
    
    ferrocyanide_lake=conc_NaCl*data_abs['NaCl_LK'] + conc_NaBr*data_abs['NaBr_LK'] + conc_NaI*data_abs['I_LK'] + conc_NO3*data_abs['NaNO3_LK'] + conc_ferrocyanide_low*data_abs['K_ferrocyanide_LK'] + conc_hco3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0) + conc_co3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0) + conc_hso3_low*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0) + conc_so3_low*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0) + np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0)
    
    ferrocyanide_lake_err=np.sqrt( (conc_NaCl*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI*data_abs['I_err_LK'])**2.0 + (conc_NO3*data_abs['NaNO3_err_LK'])**2.0 + (conc_ferrocyanide_low*data_abs['K_ferrocyanide_err_LK'])**2.0 + (h2o_err_interp)**2.0 + (conc_co3*co3_err_interp)**2.0 + (conc_hco3*hco3_err_interp)**2.0 )
    
    ferrocyanide_lake_high=conc_NaCl*data_abs['NaCl_LK'] + conc_NaBr*data_abs['NaBr_LK'] + conc_NaI_high*data_abs['I_LK'] + conc_NO3_high*data_abs['NaNO3_LK'] + conc_ferrocyanide_high*data_abs['K_ferrocyanide_LK'] + conc_hco3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0) + conc_co3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0) + conc_hs_high*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0) + conc_hso3_high*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0) + conc_so3_high*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0) + np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0)
    
    ferrocyanide_lake_high_err=np.sqrt( (conc_NaCl*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI_high*data_abs['I_err_LK'])**2.0 + (conc_NO3_high*data_abs['NaNO3_err_LK'])**2.0 + (conc_ferrocyanide_high*data_abs['K_ferrocyanide_err_LK'])**2.0 + (h2o_err_interp)**2.0 + (conc_co3*co3_err_interp)**2.0 + (conc_hco3*hco3_err_interp)**2.0 )
    
    fig, ax=plt.subplots(2, figsize=(8., 9.), sharex=True, sharey=True)
   
    ax[0].set_title('Ferrocyanide Lake, Low-Absorption Endmember')
    ax[0].errorbar(data_wav['LK'], ferrocyanide_lake, yerr=ferrocyanide_lake_err, linewidth=2, linestyle='-', marker='o', color='black', label='Total')
    ax[0].errorbar(data_wav['LK'], conc_NaCl*data_abs['NaCl_LK'], yerr=conc_NaCl*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_yellow'], label='Cl$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NaBr*data_abs['NaBr_LK'], yerr=conc_NaBr*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['strong_purple'], label='Br$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NaI*data_abs['I_LK'], yerr=conc_NaI*data_abs['I_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_orange'], label='I$^-$')
    ax[0].errorbar(data_wav['LK'], conc_NO3*data_abs['NaNO3_LK'] , yerr=conc_NO3*data_abs['NaNO3_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['very_light_blue'], label='NO$_3^{-}$')    
    ax[0].errorbar(data_wav['LK'], conc_hco3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0), yerr=conc_hco3*hco3_err_interp, linestyle='--', color=kelly_colors['grayish_yellow'], label='HCO$_3^-$')
    ax[0].errorbar(data_wav['LK'], conc_co3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0), yerr=conc_co3*co3_err_interp, linestyle='--', color=kelly_colors['medium_gray'], label='CO$_3^{2-}$')
    ax[0].plot(data_wav['LK'],conc_hso3_low*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['vivid_green'], label='HSO$_3^-$')
    ax[0].plot(data_wav['LK'],conc_so3_low*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['strong_purplish_pink'], label='SO$_3^{2-}$')
    ax[0].plot(data_wav['LK'],0.0*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0),linestyle='--', color=kelly_colors['strong_blue'], label='HS$^{-}$')
    ax[0].errorbar(data_wav['LK'], conc_ferrocyanide_low*data_abs['K_ferrocyanide_LK'], yerr=conc_ferrocyanide_low*data_abs['K_ferrocyanide_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['deep_yellowish_brown'], label='Fe(CN)$_6^{4-}$')    
    ax[0].errorbar(data_wav['LK'],np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0),yerr=h2o_err_interp, linestyle='--', color=kelly_colors['strong_violet'], label='H$_2$O')

    ax[1].set_title('Ferrocyanide Lake, High-Absorption Endmember')
    ax[1].errorbar(data_wav['LK'], ferrocyanide_lake_high, yerr=ferrocyanide_lake_high_err, linewidth=2, linestyle='-', marker='o', color='black', label='Total')
    ax[1].errorbar(data_wav['LK'], conc_NaCl*data_abs['NaCl_LK'], yerr=conc_NaCl*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_yellow'], label='Cl$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NaBr*data_abs['NaBr_LK'], yerr=conc_NaBr*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['strong_purple'], label='Br$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NaI_high*data_abs['I_LK'], yerr=conc_NaI_high*data_abs['I_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['vivid_orange'], label='I$^-$')
    ax[1].errorbar(data_wav['LK'], conc_NO3_high*data_abs['NaNO3_LK'] , yerr=conc_NO3_high*data_abs['NaNO3_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['very_light_blue'], label='NO3')    
    ax[1].errorbar(data_wav['LK'], conc_hco3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_HCO3'], left=0, right=0), yerr=conc_hco3*hco3_err_interp, linestyle='--', color=kelly_colors['grayish_yellow'], label='HCO$_3^-$')
    ax[1].errorbar(data_wav['LK'], conc_co3*np.interp(data_wav['LK'], data_wav['birk_SO4HCO3CO3'], data_abs['birk_CO3'], left=0, right=0), yerr=conc_co3*co3_err_interp, linestyle='--', color=kelly_colors['medium_gray'], label='CO$_3^{2-}$')
    ax[1].plot(data_wav['LK'],conc_hso3_high*np.interp(data_wav['LK'], data_wav['hso3_composite'], data_abs['hso3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['vivid_green'], label='HSO$_3^-$')
    ax[1].plot(data_wav['LK'],conc_so3_high*np.interp(data_wav['LK'], data_wav['so3_composite'], data_abs['so3_composite'], left=0, right=0), linestyle='--', color=kelly_colors['strong_purplish_pink'], label='SO$_3^{2-}$')
    ax[1].plot(data_wav['LK'],conc_hs_high*np.interp(data_wav['LK'], data_wav['guenther_hsm'], data_abs['guenther_hsm'], left=0, right=0),linestyle='--', color=kelly_colors['strong_blue'], label='HS$^{-}$')
    ax[1].errorbar(data_wav['LK'], conc_ferrocyanide_high*data_abs['K_ferrocyanide_LK'], yerr=conc_ferrocyanide_high*data_abs['K_ferrocyanide_err_LK'], linewidth=1, linestyle='--', color=kelly_colors['deep_yellowish_brown'], label='Fe(CN)$_6^{4-}$')    
    ax[1].errorbar(data_wav['LK'],np.interp(data_wav['LK'], h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0),yerr=h2o_err_interp, linestyle='--', color=kelly_colors['strong_violet'], label='H$_2$O')


    
    ax[0].legend(ncol=1, loc='best')
    # ax[1].legend(ncol=1, loc='best')

    ax[0].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')    
    ax[1].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')
    
    ax[1].set_xscale('linear')
    ax[1].set_xlabel('Wavelength (nm)')    
    ax[1].set_yscale('log')
    ax[1].set_ylim([1E-3, 1.0E1]) 
    ax[1].set_xlim([200., 300.])
    
    ax[0].axhline(1.0E0, color='black', linestyle=':')
    ax[0].axhline(1.0E-1, color='black', linestyle=':')
    ax[0].axhline(1.0E-2, color='black', linestyle=':')
    ax[0].axhline(1.0E-3, color='black', linestyle=':')

    ax[1].axhline(1.0E0, color='black', linestyle=':')
    ax[1].axhline(1.0E-1, color='black', linestyle=':')
    ax[1].axhline(1.0E-2, color='black', linestyle=':')
    ax[1].axhline(1.0E-3, color='black', linestyle=':')
    
    plt.savefig('./Plots/molecules/ferrocyanide-lake.pdf', orientation='portrait', format='pdf')
    
    ######
    #Plot UV in simulated reservoir
    ######
    ##Project pure water absorption to relevant wavelength scale, superpose
    ferrocyanide_lake_transmission=np.interp(uv_wav, data_wav['LK'], ferrocyanide_lake_high, left=0, right=0)  #set absorptions to 0 where therem is no data; completely unphysical, but we shouldn't (and don't need to) trust those data anyway. 
    thunk, ferrocyanide_lake_transmission_err=linear_interp(uv_wav, data_wav['LK'], ferrocyanide_lake_high,ferrocyanide_lake_high_err)

    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True, sharey=True)
    ax.set_title('Ferrocyanide Lake UV')
    ax.plot(uv_wav, uv_toa_intensity, linewidth=2, linestyle='-', marker='o', color='black', label='Stellar Irradiation')

    ax.plot(uv_wav, uv_surface_intensity, linewidth=2, linestyle='-', marker='o', color='purple', label='Surface')
    # ax.plot(uv_wav, uv_surface_intensity*10.0**(-ferrocyanide_lake_transmission*1.0/0.5), linewidth=2, linestyle='-', marker='o', color='blue', label='1 cm')
    T_1=uv_surface_intensity*10.0**(-ferrocyanide_lake_transmission*1.0/0.5)
    T_1_err=T_1*((1.0/0.5)*np.log(10.0))*ferrocyanide_lake_transmission_err
    ax.errorbar(uv_wav, T_1, yerr=T_1_err, linewidth=2, linestyle='-', marker='o', color='blue', label='1 cm')

    # ax.plot(uv_wav, uv_surface_intensity*10.0**(-ferrocyanide_lake_transmission*10.0/0.5), linewidth=2, linestyle='-', marker='o', color='cyan', label='10 cm')
    T_10=uv_surface_intensity*10.0**(-ferrocyanide_lake_transmission*10.0/0.5)
    T_10_err=T_10*((10.0/0.5)*np.log(10.0))*ferrocyanide_lake_transmission_err
    ax.errorbar(uv_wav, T_10, yerr=T_10_err, linewidth=2, linestyle='-', marker='o', color='cyan', label='10 cm')

    # ax.plot(uv_wav, uv_surface_intensity*10.0**(-ferrocyanide_lake_transmission*100.0/0.5), linewidth=2, linestyle='-', marker='o', color='green', label='100 cm')
    T_100=uv_surface_intensity*10.0**(-ferrocyanide_lake_transmission*100.0/0.5)
    T_100_err=T_100*((100.0/0.5)*np.log(10.0))*ferrocyanide_lake_transmission_err
    ax.errorbar(uv_wav, T_100, yerr=T_100_err, linewidth=2, linestyle='-', marker='o', color='green', label='100 cm')

    
    

    ax.legend(ncol=1, loc='best')
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')    
    ax.set_yscale('log')
    ax.set_ylabel(r'Intensity (photons cm$^{-2}$ s$^{-1}$ nm$^{-1}$)')
    ax.set_ylim([1E8, 1.0E14]) 
    ax.set_xlim([200., 300.])
    plt.savefig('./Plots/molecules/uv_ferrocyanide-lake.eps', orientation='portrait', format='eps')
    
    # print('Ferrocyanide Lake, High-Absorption Endmember')
    # print('Nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', ferrocyanide_lake_high, data_wav['LK'], 100.0)))
    # print('Cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', ferrocyanide_lake_high, data_wav['LK'], 100.0)))
    # print('2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', ferrocyanide_lake_high, data_wav['LK'], 100.0)))
    # print('2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', ferrocyanide_lake_high, data_wav['LK'], 100.0)))
    # print('2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', ferrocyanide_lake_high, data_wav['LK'], 100.0)))
    # print()
    
    ###Quantities for paper text
    #high-absorption, 260 nm
    ind=np.where(data_wav['LK']==300.0)
    print(r'Ferrocyanide high, 300 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/ferrocyanide_lake_high[ind], np.log10(np.exp(1.0))*ferrocyanide_lake_high_err[ind]/ferrocyanide_lake_high[ind]**2.0))
    
    #high-absorption, 260 nm
    ind=np.where(data_wav['LK']==260.0)
    print(r'Ferrocyanide high, 260 nm, OD=1 at: {0}$\pm{1} cm'.format(np.log10(np.exp(1.0))/ferrocyanide_lake_high[ind], np.log10(np.exp(1.0))*ferrocyanide_lake_high_err[ind]/ferrocyanide_lake_high[ind]**2.0))