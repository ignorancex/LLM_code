######################################
## Imports
import numpy as np
import h5py as h5
import os

from astropy.table import vstack, Table, Column
import astropy.units as u
from astropy import constants as const

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import ticker, cm

from scipy import stats

# Chosen cosmology 
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value



#########################################
# Bin rate density over crude z-bin
#########################################
def get_crude_rate_density(intrinsic_rate_density, fine_redshifts, crude_redshifts):
    """
        A function to take the 'volume averaged' intrinsic rate density for large (crude) redshift bins. 
        This takes into account the change in volume for different redshift shells

        !! This function assumes an integrer number of fine redshifts fit in a crude redshiftbin !!
        !! We also assume the fine redshift bins and crude redshift bins are spaced equally in redshift !!
        
        Args:
            intrinsic_rate_density    --> [2D float array] Intrinsic merger rate density for each binary at each redshift in 1/yr/Gpc^3
            fine_redshifts            --> [list of floats] Edges of redshift bins at which the rates where evaluated
            crude_redshifts           --> [list of floats] Merger rate for each binary at each redshift in 1/yr/Gpc^3

        Returns:
            crude_rate_density       --> [2D float array] Intrinsic merger rate density for each binary at new crude redshiftbins in 1/yr/Gpc^3

    """
    # Calculate the volume of the fine redshift bins
    fine_volumes       = cosmo.comoving_volume(fine_redshifts).to(u.Gpc**3).value
    fine_shell_volumes = np.diff(fine_volumes) #same len in z dimension as weight

    
    # Multiply intrinsic rate density by volume of the redshift shells, to get the number of merging BBHs in each z-bin
    N_BBH_in_z_bin         = (intrinsic_rate_density[:,:] * fine_shell_volumes[:])
    
    # !! the following asusmes your redshift bins are equally spaced in both cases!!
    # get the binsize of 
    fine_binsize, crude_binsize    = np.diff(fine_redshifts), np.diff(crude_redshifts) 
    if np.logical_and(np.all(np.round(fine_binsize,8) == fine_binsize[0]),  np.all(np.round(crude_binsize,8) == crude_binsize[0]) ):
        fine_binsize    = fine_binsize[0]
        crude_binsize   = crude_binsize[0] 
    else:
        print('Your fine redshifts or crude redshifts are not equally spaced!,',
              'fine_binsize:', fine_binsize, 'crude_binsize', crude_binsize)
        return -1

    # !! also check that your crude redshift bin is made up of an integer number of fine z-bins !!
    i_per_crude_bin = crude_binsize/fine_binsize 
    print('i_per_crude_bin', i_per_crude_bin)
    if (i_per_crude_bin).is_integer():
        i_per_crude_bin = int(i_per_crude_bin)
    else: 
        print('your crude redshift bin is NOT made up of an integer number of fine z-bins!: i_per_crude_bin,', i_per_crude_bin)
        return -1
    
    
    # add every i_per_crude_bin-th element together, to get the number of merging BBHs in each crude redshift bin
    N_BBH_in_crudez_bin    = np.add.reduceat(N_BBH_in_z_bin, np.arange(0, len(N_BBH_in_z_bin[0,:]), int(i_per_crude_bin) ), axis = 1)
    
    
    # Convert crude redshift bins to volumnes and ensure all volumes are in Gpc^3
    crude_volumes       = cosmo.comoving_volume(crude_redshifts).to(u.Gpc**3).value
    crude_shell_volumes = np.diff(crude_volumes)# split volumes into shells 
    
    
    # Finally tunr rate back into an average (crude) rate density, by dividing by the new z-volumes
    # In case your crude redshifts don't go all the way to z_first_SF, just use N_BBH_in_crudez_bin up to len(crude_shell_volumes)
    crude_rate_density     = N_BBH_in_crudez_bin[:, :len(crude_shell_volumes)]/crude_shell_volumes
    
    return crude_rate_density


#########################################
# Chirp mass
#########################################
def Mchirp(m1, m2):
    chirp_mass = np.divide(np.power(np.multiply(m1, m2), 3./5.), np.power(np.add(m1, m2), 1./5.))
    return chirp_mass    
   

################################################
def hdf5_to_astropy(hdf5_file, group = 'BSE_System_Parameters' ):
    """convert your hdf5 table to astropy.table for easy indexing etc
    hdf5_file  =  Hdf5 file you would like to convert
    group      =  Data group that you want to acces
    """
    Data         = hdf5_file[group]#
    table = Table()
    for key in list(Data.keys()):
        table[key] =  Data[key]
    return table
#########################################
# Read data
#########################################
def read_data(loc = '/output/COMPAS_Output_wWeights.h5', rate_key = 'Rates_mu00.025_muz-0.05_alpha-1.77_sigma01.125_sigmaz0.05_zBinned', read_SFRD = True, DCO_type = "BBH", dcomask_key = 'DCOmask', verbose=False):
    """
        Read DCO, SYS and merger rate data, necesarry to make the plots in this 
        
        Args:
            loc                  --> [string] Location of data
            rate_key             --> [string] group key name of COMPAS HDF5 data that contains your merger rate
            DCO_type             --> [string] Which favour of dco do you want [BBH", "NSNS", "BHNS", "BBH_BHNS"]
            read_SFRD            --> [bool] If you want to also read in sfr data
            verbose              --> [bool] If you want to print statements while reading in 

        Returns:
            DCO                        --> [astropy table] contains all your double compact object
            DCO_mask                   --> [array of bool] reduces your DCO table to your systems of interest (DCO_type)
            rate_mask                  --> [array of bool] reduces intrinsic_rate_density to systems (flavour) of interest
            redshifts                  --> [array of floats] list of redshifts where you calculated the merger rate
            Average_SF_mass_needed     --> [float]    Msun SF needed to produce the binaries in this simulation
            intrinsic_rate_density     --> [2D array] merger rate in N/Gpc^3/yr
            intrinsic_rate_density_z0  --> [2D array] merger rate in N/Gpc^3/yr at finest/lowest redshift bin calculated

    """
    if verbose: print('Reading ',loc)
    ################################################
    ## Open hdf5 file
    File        = h5.File(loc ,'r')
    if verbose: print(File.keys(), File[rate_key].keys())
         
    DCO = Table()
    
    dcokey,  syskey, CEcount = 'BSE_Double_Compact_Objects', 'BSE_System_Parameters', 'CE_Event_Counter'
    try: 
        DCO['SEED']                  = File[dcokey]['SEED'][()] 
    except:
        dcokey, syskey, CEcount = 'DoubleCompactObjects', 'SystemParameters', 'CE_Event_Count'
        DCO['SEED']                  = File[dcokey]['SEED'][()] 

    DCO[CEcount]                 = File[dcokey][CEcount][()] 
    DCO['Mass(1)']               = File[dcokey]['Mass(1)'][()]
    DCO['Mass(2)']               = File[dcokey]['Mass(2)'][()]
    DCO['M_tot']                 = DCO['Mass(1)'] + DCO['Mass(2)']
    DCO['M_moreMassive']         = np.maximum(File[dcokey]['Mass(1)'][()], File[dcokey]['Mass(2)'][()])
    DCO['M_lessMassive']         = np.minimum(File[dcokey]['Mass(1)'][()], File[dcokey]['Mass(2)'][()])
    DCO['Mchirp']                = Mchirp(DCO['M_moreMassive'], DCO['M_lessMassive'])
    DCO['q_final']               = DCO['M_lessMassive']/DCO['M_moreMassive']
    
    DCO['mixture_weight']        = File[dcokey]['mixture_weight'][()]
    DCO['MT_Donor_Hist(1)']      = File[dcokey]['MT_Donor_Hist(1)'][()]
    DCO['MT_Donor_Hist(2)']      = File[dcokey]['MT_Donor_Hist(2)'][()]
    DCO['Stellar_Type(1)']       = File[dcokey]['Stellar_Type(1)'][()]
    DCO['Stellar_Type(2)']       = File[dcokey]['Stellar_Type(2)'][()]
    
    DCO['Merges_Hubble_Time']    = File[dcokey]['Merges_Hubble_Time'][()]
    DCO['Optimistic_CE']         = File[dcokey]['Optimistic_CE'][()]
    DCO['Immediate_RLOF>CE']     = File[dcokey]['Immediate_RLOF>CE'][()]
#     DCO['CH_on_MS(1)']           = File[dcokey]['CH_on_MS(1)'][()]
#     DCO['CH_on_MS(2)']           = File[dcokey]['CH_on_MS(2)'][()]

    DCO['Metallicity@ZAMS(1)']   = File[dcokey]['Metallicity@ZAMS(1)'][()] 
    DCO['Stellar_Type@ZAMS(1)']  = File[dcokey]['Stellar_Type@ZAMS(1)'][()]
    DCO['Stellar_Type@ZAMS(2)']  = File[dcokey]['Stellar_Type@ZAMS(2)'][()]
    DCO['Mass@ZAMS(1)']          = File[dcokey]['Mass@ZAMS(1)'][()]
    DCO['Mass@ZAMS(2)']          = File[dcokey]['Mass@ZAMS(2)'][()]

#     SYS_DCO_seeds_bool           = np.in1d(File[syskey]['SEED'][()], DCO['SEED']) #Bool to point SYS to DCO
#     try: 
#         DCO['Metallicity@ZAMS(1)']   = File[dcokey]['Metallicity@ZAMS(1)'][()] 
#     except:
#         DCO['Metallicity@ZAMS(1)']  = File[syskey]['Stellar_Type@ZAMS(1)'][SYS_DCO_seeds_bool]

#     DCO['Stellar_Type@ZAMS(1)']  = File[syskey]['Stellar_Type@ZAMS(1)'][SYS_DCO_seeds_bool]
#     DCO['Stellar_Type@ZAMS(2)']  = File[syskey]['Stellar_Type@ZAMS(2)'][SYS_DCO_seeds_bool]
#     DCO['Mass@ZAMS(1)']          = File[syskey]['Mass@ZAMS(1)'][SYS_DCO_seeds_bool]
#     DCO['Mass@ZAMS(2)']          = File[syskey]['Mass@ZAMS(2)'][SYS_DCO_seeds_bool]
    
    
    ############################
    # Add a bool for HG-HG donors
    firstMT_HG  =  np.array([DCO['MT_Donor_Hist(1)'][i][0] == '2' for i in range(len(DCO['MT_Donor_Hist(1)'])) ])
    secondMT_HG =  np.array([DCO['MT_Donor_Hist(2)'][i][0] == '2' for i in range(len(DCO['MT_Donor_Hist(2)'])) ])

    DCO['experiencedHGHG_MT'] = np.full(len(DCO), False)
    DCO['experiencedHGHG_MT'][firstMT_HG* secondMT_HG] = True

    ############################
    # Add a bool for case A donors
    firstMT_MS  =  np.array([DCO['MT_Donor_Hist(1)'][i][0] == '1' for i in range(len(DCO['MT_Donor_Hist(1)'])) ])
    DCO['first_MT_MS'] = np.full(len(DCO), False)
    DCO['first_MT_MS'][firstMT_MS] = True

    secondMT_MS =  np.array([DCO['MT_Donor_Hist(2)'][i][0] == '1' for i in range(len(DCO['MT_Donor_Hist(2)'])) ])
    DCO['second_MT_MS'] = np.full(len(DCO), False)
    DCO['second_MT_MS'][secondMT_MS] = True
    
    rateDCO_mask = np.full(len(DCO), False)
    
    if read_SFRD:
        ################################################
        ## Read merger rate related data
        rateDCO_mask              = File[rate_key][dcomask_key][()] # Mask from DCO to merging systems 
        print('sum(rateDCO_mask)', sum(rateDCO_mask))
        redshifts                 = File[rate_key]['redshifts'][()]
        intrinsic_rate_density    = File[rate_key]['merger_rate'][()]
        intrinsic_rate_density_z0 = File[rate_key]['merger_rate_z0'][()] #Rate density at z=0 for the smallest z bin
        try:
            Average_SF_mass_needed    = File[rate_key]['Average_SF_mass_needed'][()]
        except:
            Average_SF_mass_needed = 0.

        print('len(DCO[rateDCO_mask])', len(DCO[rateDCO_mask]),'np.shape(intrinsic_rate_density)',np.shape(intrinsic_rate_density) )
     
    
    ############################
    def get_bools(table):
        # Make a custom DCO mask
        pessimistic_CE = table['Optimistic_CE'] == False
        immediateRLOF  = table['Immediate_RLOF>CE'] == False
    #     notCHE         = np.logical_and(DCO['CH_on_MS(1)'] ==False, DCO['CH_on_MS(2)'] ==False) #Remove CHE systems
        notCHE         = np.logical_and(table['Stellar_Type@ZAMS(1)'] != 16, table['Stellar_Type@ZAMS(2)'] != 16) #Remove CHE systems

        BBH_bool  = np.logical_and(table['Stellar_Type(1)'] == 14, table['Stellar_Type(2)'] == 14)
        NSNS_bool = np.logical_and(table['Stellar_Type(1)'] == 13, table['Stellar_Type(2)'] == 13)
        BHNS_bool = np.logical_or(np.logical_and(table['Stellar_Type(1)'] == 13, table['Stellar_Type(2)'] == 14),
                                  np.logical_and(table['Stellar_Type(1)'] == 14, table['Stellar_Type(2)'] == 13))
        return pessimistic_CE, immediateRLOF, notCHE, BBH_bool, NSNS_bool, BHNS_bool
    
    pessimistic_CE, immediateRLOF, notCHE, BBH_bool, NSNS_bool, BHNS_bool = get_bools(DCO)
    R_pessimistic_CE, R_immediateRLOF, R_notCHE, R_BBH_bool, R_NSNS_bool, R_BHNS_bool = get_bools(DCO[rateDCO_mask])
    
    
    if DCO_type  == "BBH":
        DCO_mask  = BBH_bool * pessimistic_CE * immediateRLOF * notCHE * rateDCO_mask
        rate_mask = R_BBH_bool * R_pessimistic_CE * R_immediateRLOF * R_notCHE

    if DCO_type  == "NSNS":
        DCO_mask  = NSNS_bool * pessimistic_CE * immediateRLOF * notCHE * rateDCO_mask
        rate_mask = R_NSNS_bool * R_pessimistic_CE * R_immediateRLOF * R_notCHE
        
    if DCO_type  == "BHNS":
        DCO_mask  = BHNS_bool * pessimistic_CE * immediateRLOF * notCHE * rateDCO_mask
        rate_mask = R_BHNS_bool * R_pessimistic_CE * R_immediateRLOF * R_notCHE
    
    # You want both BBH and BHNS
    elif DCO_type == "BBH_BHNS":
        DCO_mask  = np.logical_or(BBH_bool, BHNS_bool) * pessimistic_CE * immediateRLOF * rateDCO_mask
        rate_mask = np.logical_or(R_BBH_bool, R_BHNS_bool) * R_pessimistic_CE * R_immediateRLOF * R_notCHE
        print('sum(np.logical_or(BBH_bool, BHNS_bool))', sum(np.logical_or(BBH_bool, BHNS_bool)),'sum(R_pessimistic_CE)', sum(R_pessimistic_CE),'sum(R_immediateRLOF)', sum(R_immediateRLOF),'sum(R_notCHE)', sum(R_notCHE) )
 
    
    if read_SFRD:
        File.close()
        return DCO, rateDCO_mask, DCO_mask, rate_mask, redshifts, Average_SF_mass_needed, intrinsic_rate_density, intrinsic_rate_density_z0 
    else:
        File.close()
        return DCO, DCO_mask
        

#########################################
# Read data
#########################################
def read_SFRDdata(loc = '/output/COMPAS_Output_wWeights.h5', rate_key = 'Rates_mu00.025_muz-0.05_alpha-1.77_sigma01.125_sigmaz0.05_zBinned', verbose=False,dcomask_key = 'DCOmask'):
    """
        Read DCO, SYS and merger rate data, necesarry to make the plots in this 
        
        Args:
            loc                  --> [string] Location of data
            rate_key             --> [string] group key name of COMPAS HDF5 data that contains your merger rate
            verbose              --> [bool] If you want to print statements while reading in 

        Returns:
            rate data   --> [2D float array] Intrinsic merger rate density for each binary at new crude redshiftbins in 1/yr/Gpc^3

    """
    if verbose: print('Reading ',loc)
    ################################################
    ## Open hdf5 file
    File        = h5.File(loc ,'r')
    if verbose: print(File.keys(), File[rate_key].keys())
         
    ################################################
    ## Open hdf5 file and read relevant columns
    File        = h5.File(loc ,'r')

    ################################################
    ## Read merger rate related data
    DCO_mask                  = File[rate_key][dcomask_key][()] # Mask from DCO to merging BBH 
    redshifts                 = File[rate_key]['redshifts'][()]
    intrinsic_rate_density    = File[rate_key]['merger_rate'][()]
    intrinsic_rate_density_z0 = File[rate_key]['merger_rate_z0'][()] #Rate density at z=0 for the smallest z bin in your simulation
    try:
        Average_SF_mass_needed    = File[rate_key]['Average_SF_mass_needed'][()]
    except:
        Average_SF_mass_needed = 0.

    File.close()
    
    return DCO_mask, redshifts, Average_SF_mass_needed, intrinsic_rate_density, intrinsic_rate_density_z0     

