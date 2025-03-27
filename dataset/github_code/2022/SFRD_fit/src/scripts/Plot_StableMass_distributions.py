"""
# Plotting the stable BH mass distribution for several SFRD Z-distribution variations
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
# configure backend here
matplotlib.use('Agg')

import seaborn as sns
from scipy import stats

import h5py as h5 
from astropy.table import Table
import astropy.units as u
from astropy import constants as const

# Chosen cosmology 
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value

import json
import argparse
import gc

# Custum scripts
import MassDistHelperFunctions as mfunc
import paths
import init_values as In
import Plot_Mass_distributions as pltmass


######################################
## locations
save_loc    =  str(paths.figures) + '/'
data_dir    =  str(paths.data) + '/'


######################################
## PLOT setttings
plt.rc('font', family='serif')
from matplotlib import rc
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
fsize, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 30,25,25,30
for obj in ['axes','xtick','ytick']:
    plt.rc(obj, labelsize=MEDIUM_SIZE)          # controls default text sizes
for obj in ['figure','axes']:
    plt.rc(obj, titlesize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize



#################################################################################################
#                                                                                               #
#                                          Call plots                                           #
#                                                                                               #
#################################################################################################
if __name__ == "__main__": 

    # Initialize values
    In.init()
    
    only_stable = True
    only_CE     = False
    channel_string = 'stable'

    print('making Figure ', save_loc + '/Mass_distributions_'+ channel_string+'_SFRD_variations.pdf')

    fig = plt.figure( figsize = (24, 28))

    ####################################################
    # width of SFRD at z=0
    ####################################################
    #add first subplot in layout that has 3 rows and 2 columns
    subplot1 = fig.add_subplot(321)

    ax1 = pltmass.plot_mass_distribution(sim_dir = data_dir, rate_file='/RateData/'+str(In.rate_file_name), simulation_data = '/'+str(In.COMPASfilename),
                           x_key = 'M_moreMassive',  rate_keys  = ['Rates_mu00.025_muz-0.049_alpha-1.79_sigma0%s_sigmaz0.048_a0.017_b1.487_c4.442_d5.886_zBinned'%(x) for x in [0.7, 1.129, 2.0]], channel_string = channel_string,
                           show_hist = False, show_KDE = True,  plot_LIGO = True, Color =  'navy',
                           only_CE = only_CE, only_stable = only_stable, 
                           bootstrap = False, bootstraps = 50, save_name = 'SFRD_width_variations.pdf', titletext = "Width of metallicity dist."+"\n"+r"$\omega_0$, (scale $z=0$)",
                           labels = [r'$\mathrm{Narrow: \ }  (\omega_0 = 0.70) \  \mathcal{R}_{0.2} = \ $',
                                     r'$\mathrm{Fiducial: \ } (\omega_0 = 1.13) \ \mathcal{R}_{0.2}= \ $', 
                                     r'$\mathrm{Wide: \ } (\omega_0 = 2.00) \  \mathcal{R}_{0.2} = \ $'],
                          multipanel = True, subplot = subplot1)

    ####################################################
    # Redshift evolution of the width
    ####################################################
    #add Second subplot in layout that has 3 rows and 2 columns
    subplot2 = fig.add_subplot(322)

    ax2 = pltmass.plot_mass_distribution(sim_dir = data_dir,rate_file='/RateData/'+str(In.rate_file_name) , simulation_data = '/'+str(In.COMPASfilename),
                           x_key = 'M_moreMassive',  rate_keys = ['Rates_mu00.025_muz-0.049_alpha-1.79_sigma01.129_sigmaz%s_a0.017_b1.487_c4.442_d5.886_zBinned'%(x) for x in [0.0, 0.048, 0.1]],channel_string = channel_string,
                           show_hist = False, show_KDE = True,  plot_LIGO = True, Color = '#00a6a0', 
                           only_CE = only_CE, only_stable = only_stable,
                           bootstrap = False, bootstraps = 50, save_name = 'SFRD_zevol_width_variations.pdf',  titletext = "Redshift evol. width of metallicity dist." +"\n"+ r"$\omega_z$, (scale z evol.)",
                           labels = [r'$\mathrm{Flat \ width: \ } (\omega_z = 0.00) \ \mathcal{R}_{0.2} = \ $',
                                     r'$\mathrm{Fiducial: \ } (\omega_z = 0.05) \ \mathcal{R}_{0.2}= \ $', 
                                     r'$\mathrm{Steep \ width: \ } (\omega_z = 0.10) \ \mathcal{R}_{0.2} = \ $'],
                            multipanel = True, subplot = subplot2)


    ####################################################
    # Mean metallicity at z=0
    ####################################################
    #add third subplot in layout that has 3 rows and 2 columns
    subplot3 = fig.add_subplot(323)

    ax3 = pltmass.plot_mass_distribution(sim_dir = data_dir,rate_file='/RateData/'+str(In.rate_file_name) , simulation_data = '/'+str(In.COMPASfilename),
                           x_key = 'M_moreMassive',  rate_keys = ['Rates_mu0%s_muz-0.049_alpha-1.79_sigma01.129_sigmaz0.048_a0.017_b1.487_c4.442_d5.886_zBinned'%(x) for x in [0.007, 0.025, 0.035]],channel_string = channel_string,
                           show_hist = False, show_KDE = True,  plot_LIGO = True, Color = '#e1131d', 
                           only_CE = only_CE, only_stable = only_stable,
                           bootstrap = False, bootstraps = 50, save_name = 'SFRD_meanZ_variations.pdf',  titletext = 'Mean metallicity'+"\n"+r"$\mu_0$",
                           labels = [r'$\mathrm{low \ <Z_0> : \ } (\mu_0 = 0.007) \ \mathcal{R}_{0.2} = \ $',
                                     r'$\mathrm{Fiducial : \ }  (\mu_0 = 0.025) \ \mathcal{R}_{0.2} = \ $',
                                     r'$\mathrm{high \ <Z_0> : \ } (\mu_0 = 0.035) \ \mathcal{R}_{0.2} = \ $'],
                            multipanel = True, subplot = subplot3)

    ####################################################
    # Redshift evolution of mean metallicity
    ####################################################
    #add 4th subplot in layout that has 3 rows and 2 columns
    subplot4 = fig.add_subplot(324)

    ax4 = pltmass.plot_mass_distribution(sim_dir = data_dir,rate_file='/RateData/'+str(In.rate_file_name) , simulation_data = '/'+str(In.COMPASfilename),
                           x_key = 'M_moreMassive',  rate_keys = ['Rates_mu00.025_muz%s_alpha-1.79_sigma01.129_sigmaz0.048_a0.017_b1.487_c4.442_d5.886_zBinned'%(x) for x in [0.0, -0.049, -0.5]],channel_string = channel_string,
                           show_hist = False, show_KDE = True,  plot_LIGO = True, Color = '#ff717b', 
                           only_CE = only_CE, only_stable = only_stable,
                           bootstrap = False, bootstraps = 50, save_name = 'SFRD_zevol_mean_variations.pdf', titletext = "Redshift evol. of mean metallicity" +"\n"+ r"$\mu_z$", 
                           labels = [r'$\mathrm{Flat: \ }  (\mu_z = 0.0) \ \mathcal{R}_{0.2} = \ $',
                                     r'$\mathrm{Fiducial: \ } (\mu_z = -0.05) \ \mathcal{R}_{0.2}= \ $', 
                                     r'$\mathrm{Steep: \ } (\mu_z = -0.5) \ \mathcal{R}_{0.2} = \ $'],
                            multipanel = True, subplot = subplot4)


    ####################################################
    # Skewness
    ####################################################
    #add 5th subplot in layout that has 3 rows and 2 columns
    subplot5 = fig.add_subplot(325)

    ax5 = pltmass.plot_mass_distribution(sim_dir = data_dir,rate_file='/RateData/'+str(In.rate_file_name) , simulation_data = '/'+str(In.COMPASfilename),
                           x_key = 'M_moreMassive',  rate_keys = ['Rates_mu00.025_muz-0.049_alpha%s_sigma01.129_sigmaz0.048_a0.017_b1.487_c4.442_d5.886_zBinned'%(x) for x in [0.0, -1.79, -6.0]],channel_string = channel_string,
                           show_hist = False, show_KDE = True, plot_LIGO = True, Color = '#acbf00', 
                           only_CE = only_CE, only_stable = only_stable,
                           bootstrap = False, bootstraps = 50, save_name = 'SFRD_skewness_variations.pdf', titletext = "Skewness of metallicity dist." +"\n"+ r"$\alpha$, (shape)", 
                           labels = [r'$\mathrm{Symmetric: \ } (\alpha = 0.0)   \ \mathcal{R}_{0.2} = \ $',
                                     r'$\mathrm{Fiducial: \  } (\alpha = -1.77)  \ \mathcal{R}_{0.2}= \ $', 
                                     r'$\mathrm{Skewed: \    } (\alpha = -6)  \ \mathcal{R}_{0.2} = \ $'],
                            multipanel = True, subplot = subplot5)


    ####################################################
    # Star formation norm
    ####################################################
    #add 6th subplot in layout that has 3 rows and 2 columns
    subplot6 = fig.add_subplot(326)


    ax6 = pltmass.plot_mass_distribution(sim_dir = data_dir, rate_file='/RateData/'+str(In.rate_file_name), simulation_data = '/'+str(In.COMPASfilename),
                           x_key = 'M_moreMassive',  
                           rate_keys = ['Rates_mu00.025_muz-0.049_alpha-1.79_sigma01.129_sigmaz0.048_a0.01_b2.6_c3.2_d6.2_zBinned',
                                       'Rates_mu00.025_muz-0.049_alpha-1.79_sigma01.129_sigmaz0.048_a0.017_b1.487_c4.442_d5.886_zBinned', 
                                       'Rates_mu00.025_muz-0.049_alpha-1.79_sigma01.129_sigmaz0.048_a0.03_b2.6_c3.3_d5.9_zBinned'],
                                 channel_string = channel_string,
                           show_hist = False, show_KDE = True,  plot_LIGO = True, Color = '#ecb05b', 
                           only_CE = only_CE, only_stable = only_stable,
                           bootstrap = False, bootstraps = 50, save_name = 'SFRD_skewness_variations.pdf', titletext = "Overall SFR history"+"\n"+ r'$ \mathrm{SFRD(}z\rm{)} \ [a,b,c,d]$', 
                           labels = [r'$\mathrm{Madau \ and \ Fragos \ 2017: }  \ \mathcal{R}_{0.2}= \ $', 
                                     r'$\mathrm{Fiducial: \ } \ \mathcal{R}_{0.2}= \ $', 
                                     r'$\mathrm{Approx. \ to \ upper \ limit:}  \ \mathcal{R}_{0.2} = \ $'],
                            multipanel = True, subplot = subplot6)



    ####################################################
    # Final plot properties
    fig.savefig(save_loc + '/Mass_distributions_'+ channel_string+'_SFRD_variations.pdf' , bbox_inches='tight')
    # plt.show()




