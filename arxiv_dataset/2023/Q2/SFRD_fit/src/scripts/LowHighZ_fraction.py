# Notebook to make plot comparing fractional contribution low and high metallicities
# Code from Ruediger Pakmor, original plot is by Martyna Chruslinska

import os
import h5py
import numpy as np 
import matplotlib.pyplot as plt
import paths
from pylab import *
import seaborn as sns

from scipy import interpolate
from astropy.table import Table

import astropy.units as u
# from astropy.cosmology import WMAP9, z_at_value
from astropy.cosmology import Planck18  as cosmo# Planck 2018
from astropy.cosmology import z_at_value

############################
# Custom scripts
import get_ZdepSFRD as Z_SFRD
import importlib
import init_values as In

############################
##PLOT setttings
from matplotlib import rc
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', family='serif')
rc('text', usetex=True)
fsize, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 30,20,20,30
for obj in ['axes','xtick','ytick']:
    plt.rc(obj, labelsize=MEDIUM_SIZE)          # controls default text sizes
for obj in ['figure','axes']:
    plt.rc(obj, titlesize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


#######################
# Definitions
SOLAR_METALLICITY = 0.0127
fig_data_dir = str(paths.data) +'/Figure5/' 
TNG               = 50
lvl               = 1


##############################
def getStellarMassMetallicityTNG():
    fout = fig_data_dir + "StellarMassMetallicityTNG%d-%d.hdf5" % (TNG,lvl)
    if os.path.exists( fout ):
        return
  
    print( "Computing data TNG%d" % TNG )

    snap = 99
    s    = gadget_readsnap( snap, snappath=TNGpath, snapbase='snap_', loadonlytype=[5], loadonly=['mass'], chunk=0 )
    Mass = np.zeros( (nBinsRedshift,nBinsMetallicity) )

    fname = "%s/snapdir_%03d/snap_%03d.%s.hdf5" % (TNGpath, snap, snap, "%d")
    for ifile in range( s.num_files ):
        with h5py.File(fname % ifile, "r") as f:
            print( "Reading file %d/%d." % (ifile,s.num_files) )

            pStars = f["PartType4"]

            Ages   = pStars["GFM_StellarFormationTime"][:]
            Masses = pStars["GFM_InitialMass"][:].astype('f8')
            Metals = pStars["GFM_Metallicity"][:] / SOLAR_METALLICITY # convert to solar metallicity

        for ir in range(nBinsRedshift):
            i, = np.where( Ages >= 1./(1.+redshifts[ir]) )

            Mass[ir,0] += Masses[i].sum()

            k, = np.where( Metals[i] < 0.1 )
            Mass[ir,1] += Masses[i[k]].sum()

            k, = np.where( Metals[i] > 1.0 )
            Mass[ir,2] += Masses[i[k]].sum()      

    with h5py.File(fout, "w") as f:
        f.create_dataset('BinsRedshift', data=redshifts )
        f.create_dataset('Mass', data=Mass )

        
##############################
def getStellarMassMetallicityIllustris():
    fout = fig_data_dir + "StellarMassMetallicityIllustris.hdf5" 
    if os.path.exists( fout ):
        return

    print( "Computing data Illustris" )

    snap = 135
    s    = gadget_readsnap( snap, snappath=IllustrisPath + "Illustris-1/output/", snapbase='snap_', loadonlytype=[5], loadonly=['mass'], chunk=0 )
    Mass = np.zeros( (nBinsRedshift,nBinsMetallicity) )

    fname = "%s/Illustris-1/output/snapdir_%03d/snap_%03d.%s.hdf5" % (IllustrisPath, snap, snap, "%d")
    for ifile in range( s.num_files ):
        with h5py.File(fname % ifile, "r") as f:
            print( "Reading file %d/%d." % (ifile,s.num_files) )

            pStars = f["PartType4"]

            Ages   = pStars["GFM_StellarFormationTime"][:]
            Masses = pStars["GFM_InitialMass"][:].astype('f8')
            Metals = pStars["GFM_Metallicity"][:] / SOLAR_METALLICITY # convert to solar metallicity

            for ir in range(nBinsRedshift):
                i, = np.where( Ages >= 1./(1.+redshifts[ir]) )

                Mass[ir,0] += Masses[i].sum()

                k, = np.where( Metals[i] < 0.1 )
                Mass[ir,1] += Masses[i[k]].sum()

                k, = np.where( Metals[i] > 1.0 )
                Mass[ir,2] += Masses[i[k]].sum() 

  
    with h5py.File(fout, "w") as f:
        f.create_dataset('BinsRedshift', data=redshifts )
        f.create_dataset('Mass', data=Mass )

##############################
def getStellarMassMetallicitySimba():
    fout = fig_data_dir + "StellarMassMetallicitySimba.hdf5" 
    if os.path.exists( fout ):
        return

    print( "Computing data Simba" )

    Mass = np.zeros( (nBinsRedshift,nBinsMetallicity) )

    snap  = 151
    fname = "%s/Simba-L100n1000FP/output/snapdir_%03d/snap_%03d.0.hdf5" % (IllustrisPath, snap, snap)
    with h5py.File(fname, "r") as f:
        pStars = f["PartType4"]

    Ages   = pStars["StellarFormationTime"][:]
    Masses = pStars["Masses"][:].astype('f8')
    Metals = pStars["Metallicity"][:][:,2:].sum(axis=1) / SOLAR_METALLICITY # convert to solar metallicity

    for ir in range(nBinsRedshift):
        i, = np.where( Ages >= 1./(1.+redshifts[ir]) )

        Mass[ir,0] += Masses[i].sum()

        k, = np.where( Metals[i] < 0.1 )
        Mass[ir,1] += Masses[i[k]].sum()

        k, = np.where( Metals[i] > 1.0 )
        Mass[ir,2] += Masses[i[k]].sum()      
  
    with h5py.File(fout, "w") as f:
        f.create_dataset('BinsRedshift', data=redshifts )
        f.create_dataset('Mass', data=Mass )

        
##############################       
def getStellarMassMetallicityEagle():
    fout = fig_data_dir + "StellarMassMetallicityEagle.hdf5" 
    if os.path.exists( fout ):
        return

    print( "Computing data Eagle" )

    snap = 28
    s    = gadget_readsnap( snap, snappath=IllustrisPath + "Eagle-L68n1504FP/output/", snapbase='snap_', loadonlytype=[5], loadonly=['mass'], chunk=0 )
    Mass = np.zeros( (nBinsRedshift,nBinsMetallicity) )

    fname = "%s/Eagle-L68n1504FP/output/snapdir_%03d/snap_%03d.%s.hdf5" % (IllustrisPath, snap, snap, "%d")
    for ifile in range( s.num_files ):
        with h5py.File(fname % ifile, "r") as f:
            print( "Reading file %d/%d." % (ifile,s.num_files) )

            pStars = f["PartType4"]

            Ages   = pStars["GFM_StellarFormationTime"][:]
            Masses = pStars["GFM_InitialMass"][:].astype('f8')
            Metals = pStars["GFM_Metallicity"][:] / SOLAR_METALLICITY # convert to solar metallicity

            for ir in range(nBinsRedshift):
                i, = np.where( Ages >= 1./(1.+redshifts[ir]) )

                Mass[ir,0] += Masses[i].sum()

                k, = np.where( Metals[i] < 0.1 )
                Mass[ir,1] += Masses[i[k]].sum()

                k, = np.where( Metals[i] > 1.0 )
                Mass[ir,2] += Masses[i[k]].sum()      

    with h5py.File(fout, "w") as f:
        f.create_dataset('BinsRedshift', data=redshifts )
        f.create_dataset('Mass', data=Mass )

            
    
##############################
# Read Martyna's data
##############################
def read_Chruslinskadata():
    """
    dataChruslinskaZ01 = [
      [0.17,0.33,0.24,0.43,0.19,0.36],
      [0.09,0.18,0.13,0.27,0.10,0.21,0.12,0.21,0.17,0.31,0.13,0.24],
      [0.03,0.07,0.06,0.15,0.04,0.09,0.02,0.06,0.05,0.14,0.03,0.08]
    ]

    dataChruslinskaZ10 = [
      [0.01,0.19,0.01,0.12,0.01,0.17],
      [0.02,0.27,0.01,0.17,0.02,0.24,0.02,0.26,0.01,0.17,0.02,0.23],
      [0.08,0.55,0.04,0.37,0.07,0.49,0.08,0.55,0.04,0.37,0.06,0.49]
    ]
    

    """
    # Name of the low and high Z extreme
    low_Z_file =  '214'+ 'f14SB'+'BiC'+ '_FMR270'  #+ '_FOH_z_dM.dat'
    Low_Z_extreme  = [] # will be filled with [Z10, Z01] for the 3 redshifts
    high_Z_file = '302'+ 'f14SB'+'Boco'+ '_FMR270' #+ '_FOH_z_dM.dat'
    High_Z_extreme = [] # will be filled with [Z10, Z01] for the 3 redshifts

    dataChruslinska19_Z01 = [[],[],[]]
    dataChruslinska19_Z10 = [[],[],[]]

    dataChruslinska21_Z01 = [[],[],[]]
    dataChruslinska21_Z10 = [[],[],[]]

    for iz in range(3):
        z = ["10.0", "3", "0.5"][iz]
        print('\n z=',z)
        with open( fig_data_dir + "/stellar_mass_fractions_Zsun_Asplund09_zmax_%s.dat" % z, "r" ) as f:
            lines = f.readlines()
            
        for line in lines[1:]:
            Z01 = float(line.split()[2])
            Z10 = float(line.split()[3])

            # check if model_id == low or high Z file
            if str(line.split()[1]) == low_Z_file:
                print('found low Z extreme: ', line.split()[1] )
                print(Z10, Z01)
                Low_Z_extreme.append([Z10, Z01])
                
            if str(line.split()[1]) == high_Z_file:
                print('found high Z extreme: ', line.split()[1] )
                print(Z10, Z01)
                High_Z_extreme.append([Z10, Z01])
            
            if line.startswith( "ChN19" ):
                dataChruslinska19_Z01[iz] += [Z01]
                dataChruslinska19_Z10[iz] += [Z10]
            else:
                dataChruslinska21_Z01[iz] += [Z01]
                dataChruslinska21_Z10[iz] += [Z10]

    return Low_Z_extreme, High_Z_extreme, dataChruslinska19_Z01, dataChruslinska19_Z10, dataChruslinska21_Z01, dataChruslinska21_Z10


##############################
def get_SFRDzZ(redshifts, metals = [], min_logZ_COMPAS = np.log(1e-4), max_logZ_COMPAS = np.log(0.03),
               metal_params = [], SFR_Params = [], min_logZ=-12.0, max_logZ=0.0, step_logZ =0.01): 
    """
    """
    mu_0, mu_z, omega_0, omega_z, alpha = metal_params
    a, b, c, d                          = SFR_Params
    # metallicity distribution
    dPdlogZ, metallicities, step_logZ, p_draw_metallicity = \
                    Z_SFRD.skew_metallicity_distribution(redshifts, mu_0=mu_0, mu_z=mu_z,alpha = alpha, 
                                                  omega_0=omega_0, omega_z =omega_z, min_logZ  =-12.0, max_logZ  =0.0, step_logZ = 0.01,
                                                  metals = metals ) #np.logspace(-5,1, num=1000)
    #print('step_logZ', step_logZ, 'min(np.log10(metals))', min(np.log10(metals)), 'max(np.log10(metals))', max(np.log10(metals)))
    # SFR
    sfr        = Z_SFRD.Madau_Dickinson2014(redshifts, a=a, b=b, c=c,  d=d) # Msun year-1 Mpc-3 
    # Combine it into a SFRD
    model_SFRD = (sfr*(dPdlogZ*step_logZ  ).T ).value #step_logZ
    
    return model_SFRD, metallicities, step_logZ




############################################################
#### MAIN ########
############################################################
if __name__ == "__main__": 

    # Initialize values
    In.init()

    ##############################  
    # load Cosmolocigal simulations
    getStellarMassMetallicityTNG()
    getStellarMassMetallicityIllustris()
    getStellarMassMetallicitySimba()
    getStellarMassMetallicityEagle()
    ##############################

    ##############################
    # Open Simulation data
    ##############################
    with h5py.File(fig_data_dir + "StellarMassMetallicityTNG%d-%d.hdf5" % (TNG,lvl), "r") as f:
        TNGBinsRedshift = f["BinsRedshift"][:]
        TNGMass         = f["Mass"][:]

    with h5py.File(fig_data_dir + "StellarMassMetallicityIllustris.hdf5", "r") as f:
        IllustrisBinsRedshift = f["BinsRedshift"][:]
        IllustrisMass         = f["Mass"][:]

    with h5py.File(fig_data_dir + "StellarMassMetallicitySimba.hdf5", "r") as f:
        SimbaBinsRedshift = f["BinsRedshift"][:]
        SimbaMass         = f["Mass"][:]

    with h5py.File(fig_data_dir + "StellarMassMetallicityEagle.hdf5", "r") as f:
        EagleBinsRedshift = f["BinsRedshift"][:]
        EagleMass         = f["Mass"][:]



    ##############################
    # Observational constraints from Chruslinska 2021
    Low_Z_extreme, High_Z_extreme, dataChruslinska19_Z01, dataChruslinska19_Z10, dataChruslinska21_Z01, dataChruslinska21_Z10 = read_Chruslinskadata()

    print('Low_Z_extreme', Low_Z_extreme)
    print('High_Z_extreme', High_Z_extreme)
    print('dataChruslinska21_Z01', dataChruslinska21_Z01)


    print( "TNG:", TNGMass[:,1] / TNGMass[:,0], TNGMass[:,2] / TNGMass[:,0] )
    print( "Illustris:", IllustrisMass[:,1] / IllustrisMass[:,0], IllustrisMass[:,2] / IllustrisMass[:,0] )
    print( "Simba:", SimbaMass[:,1] / SimbaMass[:,0], SimbaMass[:,2] / SimbaMass[:,0] )
    print( "Eagle:", EagleMass[:,1] / EagleMass[:,0], EagleMass[:,2] / EagleMass[:,0] )




    ##############################################################################
    ## Load the TNG 100 data
    # This starformation rate comes from the gas particles in TNG: StarFormationRate = "Instantaneous star formation rate of this gas cell."
    # See also: 
    # https://www.tng-project.org/data/docs/specifications/#parttype0
    # This is why we have to convert the rate to a density by dividing by the co-moving box size. 
    ##############################################################################
    interpolate_data = True

    # Load TNG100 data
    with h5py.File(paths.data / "SFRMetallicityFromGasTNG100.hdf5", "r") as file:
        MetalBins            = file["MetalBins"][:] #60
        TNG100_center_Zbin   = (MetalBins[:-1] + MetalBins[1:])/2.
        TNG100_Lookbacktimes = file["Lookbacktimes"][:] #100
        BoxSfr               = file["Sfr"][:]
    # Convert SFR from sfr/box to sfr cMpc-3
    littleh    = 0.6774
    Rbox       = 75/littleh
    TNG100_SFR = BoxSfr / Rbox**3 *u.Mpc**-3

    ## The model comes in SFRD/DeltaZ, make sure your data does as well!! 
    step_fit_logZ       = np.diff(np.log(MetalBins))[0]    
    TNG100_cosmic_SFR   = TNG100_SFR#/step_fit_logZ

    ## Convert lookback times to redshifts
    # the last value of Lookbacktimes = 0, which is problematic for z calculation
    TNG100_redshifts = [z_at_value(cosmo.lookback_time,t*u.Gyr,method='Bounded') for t in TNG100_Lookbacktimes[:-1]] 
    TNG100_redshifts.append(0) # put redshift zero back at the end
    TNG100_redshifts = np.array(TNG100_redshifts)

    #########################################
    if interpolate_data:
        #########################################
        # Interpolate the simulation data
        f_interp = interpolate.interp2d(TNG100_Lookbacktimes, TNG100_center_Zbin, TNG100_cosmic_SFR.T, kind='cubic')

        redshift_new         = np.arange(0, 10.1, 0.05)         # Retrieve values at higher res regular intervals
        Lookbacktimes_new    = [cosmo.lookback_time(z).value for z in redshift_new]

        log_TNG100_center_Zbin = np.log10(TNG100_center_Zbin)
        metals_new             = np.logspace(min(log_TNG100_center_Zbin), max(log_TNG100_center_Zbin), int(1e3))

        SFRDnew = f_interp(Lookbacktimes_new,metals_new)

        SFRDnew = SFRDnew.T
        # Original TNG data was in decreasing t_lookback, so reshape your new interpolated thing the same way
        SFRDnew           = SFRDnew[::-1]
        redshift_new      = redshift_new[::-1]
        Lookbacktimes_new = np.array(Lookbacktimes_new[::-1])
        #########################################
        print(50*'*', '\nYou are using the interpolated version')
        # # switch to new interpolated data
        TNG100_cosmic_SFR    = SFRDnew
        TNG100_Lookbacktimes = Lookbacktimes_new
        MetalBins            = metals_new
        TNG100_center_Zbin   = metals_new#(metals_new[:-1] + metals_new[1:])/2.
        TNG100_redshifts     = redshift_new

    else:
        print('working with non-interpolated data')
        print('np.shape(TNG100_cosmic_SFR)', np.shape(TNG100_cosmic_SFR))
        
    # Checking our interpolation result
    # z_i = np.argmin(TNG100_redshifts-0.5)
    #     plt.step(np.log10(TNG100_center_Zbin), TNG100_cosmic_SFR[z_i,:], where = 'mid', label = lab)
    # We are going to convert SFR to total stellar mass formed

    TNG100_centered_SFR       = (TNG100_cosmic_SFR[:-1,:] + TNG100_cosmic_SFR[1:,:])/2.          # Take the SFRD at the center of each bin in lookback time    
    TNG100_Lookbacktimes_yr   = TNG100_Lookbacktimes *u.Gyr.to(u.yr)                             # Multiply this by the diff in lookback time to get the total stellar mass formed in each lookback time bin
    TNG100_stellarM_formed_dZ = (TNG100_centered_SFR * abs(np.diff(TNG100_Lookbacktimes_yr))[:,np.newaxis])  # This gives us the stellar mass formed at t, per Mpc^-3 per d logZ (it's still 2D)
    TNG100_center_redshifts   = (TNG100_redshifts[:-1] + TNG100_redshifts[1:])/2.                 # Take the center of the redshift bins to match the shape of TNG100_stellarM_formed_dZ


    ##############################################################################
    """
    # Now start plotting the 3-panels
    We want to include 
     - Illustris/Simba/TNG50 data
     - Chruslinska 2019, 2021 data
     - TNG 100
     - model fit for TNG 100 
     - Variations of our model fit
    """
    # Now we can make a redshift bool
    redshift_points  = [0.5, 3., 10]
    redshift_bools   = [TNG100_center_redshifts < redshift_points[2], TNG100_center_redshifts < redshift_points[1], TNG100_center_redshifts < redshift_points[0]]

    # And the metallicity boundaries for 'low' and 'high' metallicity
    Zsun         = 0.0127 #for TNG
    low_metals   = MetalBins < Zsun/10.
    high_metals  = MetalBins > Zsun

    ###########
    # Model values
    redshifts         = np.arange(0,10.1, 0.01)
    center_resh       = (redshifts[:-1] + redshifts[1:])/2.
    model_reds_bools  = [center_resh < redshift_points[2], center_resh < redshift_points[1], center_resh < redshift_points[0]]
    # I think it's important that you pick a high num --> small dlogZ # else the metal bools will quickly become inaccurate
    metals        = np.logspace(min(np.log10(MetalBins)), max(np.log10(MetalBins)), num=int(1000) ) #TNG100_center_Zbin# len(MetalBins)
    # center_metals = (metals[:-1] + metals[1:])/2.
    low_Z         = metals < Zsun/10.
    high_Z        = metals > Zsun

    ###########
    # Start drawing Figure
    f, ax = plt.subplots(figsize = (12,18), sharey=True, sharex=True)

    # Remove overall axes to avoid overlap
    plt.setp(ax, xticks=[], yticks=[])
    ax.axis('off')


    colors = sns.husl_palette(3)

    ######################
    # !! We are working from right to left because that's the way the cosmological simulations are set up
    # So we'll start plotting the right most plot (z<10) and work to the left
    for ir in range(3):
        ###########
        print('\n Working on redshift z <', redshift_points[2-ir])
        
        ###########
        #add first subplot in layout that has 3 rows and 1 columns == fig.add_subplot(311)
        axstr   = '31'+str(3-ir) #Reverse order such that lowest z is plotted on left
        subplot = f.add_subplot(int(axstr))
        ax = subplot
        
        ######################
        # Cosmological Simulations
        l, = ax.plot( 100 * TNGMass[ir,2] / TNGMass[ir,0], 100 * TNGMass[ir,1] / TNGMass[ir,0], 'o', mec="None", label=None, color=colors[ir], markersize = 15)
        l, = ax.plot( 100 * IllustrisMass[ir,2] / IllustrisMass[ir,0], 100 * IllustrisMass[ir,1] / IllustrisMass[ir,0], 's', mec="None", color=colors[ir] , markersize = 15)
        l, = ax.plot( 100 * SimbaMass[ir,2] / SimbaMass[ir,0], 100 * SimbaMass[ir,1] / SimbaMass[ir,0], 'D', mec="None", color=colors[ir] , markersize = 15)
        l, = ax.plot( 100 * EagleMass[ir,2] / EagleMass[ir,0], 100 * EagleMass[ir,1] / EagleMass[ir,0], 'v', mec="None", color=colors[ir] , markersize = 15)
        
    #     # TNG100
    #     print('max TNG redshifts', max(TNG100_center_redshifts[redshift_bools[ir]] ) )
    #     sfrd_at_redshift = TNG100_stellarM_formed_dZ[redshift_bools[ir],:]
    #     TNG100_flowZ     = 100 * np.sum(sfrd_at_redshift[:,low_metals]) /np.sum(sfrd_at_redshift[:,:]) 
    #     TNG100_fhighZ    = 100 * np.sum(sfrd_at_redshift[:,high_metals]) /np.sum(sfrd_at_redshift) 
    #     print('total sfr',sum(sfrd_at_redshift))
    # #     print('frac highZ sfr',TNG100_fhighZ, 'frac lowZ sfr',     TNG100_flowZ, '\n\n')
    #     l, = ax.plot(TNG100_fhighZ, TNG100_flowZ, '*',  markerfacecolor = 'none', markeredgecolor = 'k', markersize = 30, zorder = 10)
        
        ######################
        # Model fits
        for fit_param_file in ['test_best_fit_parameters.txt']:#, 'MartynaLOWZ_best_fit_parameters.txt', 'MartynaHIGHZ_best_fit_parameters.txt']:
            print(fit_param_file)
            SzZParams =  Table.read(paths.data / fit_param_file , format = 'csv') # Read in best fit parameters
            # compute the model
            SFRDzZ, metallicities, step_logZ = get_SFRDzZ(redshifts, metals = metals,
                                               metal_params = SzZParams['# mu0','muz','omega0','omegaz','alpha0'][0],
                                               SFR_Params = SzZParams['sf_a','sf_b','sf_c','sf_d'][0])
            #'integrate' your SFR to get Msun formed. Shape SFRDzZ = metal x redshift
            model_centered_SFR       = (SFRDzZ[:,:-1] + SFRDzZ[:,1:])/2.                                   # Take the SFR at the center redshifts
            model_dt                 = np.diff(cosmo.lookback_time(redshifts)*u.Gyr.to(u.yr)  )            # convert z to lookback time and take st steps
            model_stellarM_formed_dZ = model_centered_SFR * model_dt[np.newaxis,:]                         # Multiply by dt for tot stellar mass formed in each lookback time bin
            model_SFRD_uptoz         = np.sum(model_stellarM_formed_dZ[:,model_reds_bools[ir]], axis = 1 ) # restrict to all redshifts up to x and sum over all these redshifts
            
            lowZfraction             = 100 * np.sum(model_SFRD_uptoz[low_Z])/np.sum(model_SFRD_uptoz)  
            highZfraction            = 100 * np.sum(model_SFRD_uptoz[high_Z])/np.sum(model_SFRD_uptoz) 
            ax.plot(highZfraction, lowZfraction,'*', markerfacecolor = 'none', markeredgecolor = 'k', markersize = 30)
        
        
        ######################
        # Observations (Chruslinska)
        l, = ax.plot( np.array(dataChruslinska19_Z10[ir]), np.array(dataChruslinska19_Z01[ir]), 'X', mec="k", mew=0.5, color=colors[ir], alpha=0.2 , markersize = 20)
        l, = ax.plot( np.array(dataChruslinska21_Z10[ir]), np.array(dataChruslinska21_Z01[ir]), 'P', mec="None", color=colors[ir], alpha=0.2 , markersize = 20)

        # Explicitely show the low and high z extreme models
        ax.plot( Low_Z_extreme[ir][0], Low_Z_extreme[ir][1], 'P', mec="None", color=colors[ir], alpha=1. , markersize = 20)
        ax.plot( High_Z_extreme[ir][0], High_Z_extreme[ir][1], 'P', mec="None", color=colors[ir], alpha=1. , markersize = 20)

        ####################################
        # Variations on the SFRD
        ##################################### 
        Fid_SzZParams    =  Table.read(paths.data / In.fit_param_filename, format = 'csv')
        metal_param_keys = ['# mu0','muz','omega0','omegaz','alpha0']
        symbols          = [r'$\mu_0$',r'$\mu_z$',r'$\omega_0$',r'$\omega_z$',r'$\alpha_0$']
        #                    0.025,   -0.05, 1.125,    0.05,   -1.7 
        variations = [[0.007, 0.035],[0.0, -0.5],[0.7, 2],[0.0, 0.1],[0, -6] ]
        var_colors = ['#e1131d' ,'#ff717b','navy',  '#00a6a0',  '#acbf00', '#ecb05b']

        #####################################
        for v, param_key in enumerate(metal_param_keys):
            # plot all the variations
            variation = Fid_SzZParams.copy()
            var = variations[v]
            for i in range(2): #Repeat for low and high value
                variation[param_key] = var[i]
                model_SFRD, metallicities, step_logZ = get_SFRDzZ(redshifts, metals = metals, 
                                               metal_params = variation['# mu0','muz','omega0','omegaz','alpha0'][0],
                                               SFR_Params = variation['sf_a','sf_b','sf_c','sf_d'][0])
                #'integrate' your SFR to get Msun formed. Shape SFRDzZ = metal x redshift
                model_centered_SFR       = (model_SFRD[:,:-1] + model_SFRD[:,1:])/2.                                   # Take the SFR at the center redshifts
                model_dt                 = np.diff(cosmo.lookback_time(redshifts)*u.Gyr.to(u.yr)  )            # convert z to lookback time and take st steps
                model_stellarM_formed_dZ = model_centered_SFR * model_dt[np.newaxis,:]                         # Multiply by dt for tot stellar mass formed in each lookback time bin
                model_SFRD_uptoz         = np.sum(model_stellarM_formed_dZ[:,model_reds_bools[ir]], axis = 1 ) # restrict to all redshifts up to x and sum over all these redshifts
                
                lowZfraction             = 100 * np.sum(model_SFRD_uptoz[low_Z])/np.sum(model_SFRD_uptoz)  
                highZfraction            = 100 * np.sum(model_SFRD_uptoz[high_Z])/np.sum(model_SFRD_uptoz)            
                ax.plot(highZfraction, lowZfraction,'x' , markersize = 25, color = 'k', zorder = 10)#var_colors[v]
                ax.annotate(r'%s:%s'%(str(symbols[v]), var[i]),
                            xy=(highZfraction, lowZfraction), xycoords='data',
                            xytext=(-15,50-25*(-1)**i), textcoords='offset points',zorder = 10, color =  'k', rotation = 30,
                           arrowprops=dict(arrowstyle = '-',  facecolor='grey'))
    #             ax.text(highZfraction+1, lowZfraction+2 + 2*(-1)**v, s = r'%s:%s'%(str(symbols[v]), var[i]), 
    #                     zorder = 10, color =  'k', rotation = 30)
        
        
        ###########
        # plotvalues
        ax.set_xlabel( "$\% M_* (Z > Z_\odot) / M_\mathrm{*,tot}$" , size =35)
        ax.set_ylabel( "$\% M_* (Z < 0.1 Z_\odot) / M_\mathrm{*,tot}$" , size =35)
        ax.set_ylim( 0, 45. )
        ax.set_xlim( 0, 85. )
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.text(0.7, 0.9, '$z<%3.1f$' % TNGBinsRedshift[ir] , transform = ax.transAxes, size =25)

        
    ###########
    # Plot legend in last plot
    l, = ax.plot( -1, -1, 'X', alpha=0.3, mfc="None", mec="k" )
    l2 = [l]
    l, = ax.plot( -1, -1, 'P', alpha=0.3, mfc="None", mec="k" )
    l2 += [l]
    for m in ['o','s','D','v','*']:
        l, = ax.plot( -1, -1, m, mfc="None", mec="k" )
        l2 += [l]
    legend2 = matplotlib.pyplot.legend( l2, ["$\mathrm{Chruslinska19}$", "$\mathrm{Chruslinska21}$","$\mathrm{TNG50}$", "$\mathrm{Illustris}$", "$\mathrm{SIMBA}$", "$\mathrm{EAGLE}$", "$\mathrm{TNG100, \ fit}$", ], 
                                       frameon=False, numpoints=1, fontsize=20,loc=(0.025,0.32), markerscale = 2.)#loc=(0.55,0.50) )
    # ax.add_artist(legend2)
    f.tight_layout()
    f.savefig(str(paths.figures)+ "/High_Low_metalFractionSFMass.pdf"  )
    plt.show()

