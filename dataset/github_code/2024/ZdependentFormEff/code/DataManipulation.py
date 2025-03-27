############################################################################
# # Data manipulation

# Read the full COMPAS data, and extract only the information that you need (specifically only the systems that are every a DCO)

# Then we want to add relevant information about the system in the following cases:

#  * The first mass transfer that the binary engaged in
#  * The first mass transfer that star 2 engaged in
#  * The mass transfer that lead to a stellar merger (if any)
#  * The supernova information
############################################################################
import numpy as np
import os 
import pandas as pd
import h5py as h5
import gc


# Turn off natural name warning for panda tables (this is due to '@' and '>' in the COMPAS column names)
import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)



############################################################################
# ## Get the potential DCO progenitors

# Read the full COMPAS data, and extract systems that become a DCO at any of the simulated metallicities
############################################################################

def create_first_potential_DCO_progenitors_table(datar_root, sim_name, channel_key):

    save_name_table = f'potential_DCO_progenitors{channel_key}.h5'
    print(save_name_table)

    # Initialize a list to store all SEEDS that ever become a DCO
    All_DCO_seeds = []

    # check if your table exists
    if os.path.isfile(datar_root+ f'/{sim_name}/{save_name_table}'):
        print('first DCO table already exists, loading it')
        potential_DCO_progenitors = pd.read_hdf(datar_root + f'/{sim_name}/{save_name_table}', key='All_DCO')

        return potential_DCO_progenitors

    else:
        # Loop over all directories starting wiht "logZ"
        print(f'reading from {datar_root}/{sim_name}/')
        for i, dir in enumerate(os.listdir(datar_root+ f'/{sim_name}/')):

            if dir.startswith('logZ'):
                print(f"Reading DCO seeds from {dir}")

                # Open the HDF5 file for all systems at a given metallicity
                data = h5.File(datar_root+ f'/{sim_name}/{dir}/COMPAS_Output.h5', 'r')

                # Get the seeds that ever become a DCO
                DCO_seeds = pd.Series(data['BSE_Double_Compact_Objects']['SEED'][()])

                # Simple case: you want all DCOs
                if channel_key == '':
                    channel_bool = np.ones_like(DCO_seeds, dtype=bool)
                
                # Add extra constraints based on the channel you are interested in
                else:
                    # I didn't save the CHE bool in most variations :(
                    try: 
                        CHE_in_DCO = data['BSE_Double_Compact_Objects']['CH_on_MS(1)'][()]      
                    except:          
                        # Create a mask to map between the DCO seeds and the seeds in the system parameters
                        SYS_SEED            = data['BSE_System_Parameters']['SEED'][()]
                        SYS_DCO_mask        = np.in1d(SYS_SEED, DCO_seeds)
                        #print('SAFETY CHECK FOR THE IN1D SEED COUPLING: ', np.nonzero(np.array(SYS_SEED[SYS_DCO_mask]).flatten() - np.array(DCO_seeds).flatten()) )

                        CHE_bool            = data['BSE_System_Parameters']['CH_on_MS(1)'][()]
                        CHE_in_DCO          = CHE_bool[SYS_DCO_mask]

                    if channel_key == '_stable':
                        CE_Event_Counter    = pd.Series(data['BSE_Double_Compact_Objects']['CE_Event_Counter'][()])
                        channel_bool        = np.logical_and(CE_Event_Counter == 0, CHE_in_DCO == 0 ) # stable and non-CHE

                    elif channel_key == '_CE':
                        CE_Event_Counter    = pd.Series(data['BSE_Double_Compact_Objects']['CE_Event_Counter'][()])
                        channel_bool        = np.logical_and(CE_Event_Counter > 0, CHE_in_DCO == 0 ) # Common envelope and non-CHE

                    elif channel_key == '_CHE':
                        channel_bool        = CHE_in_DCO == 1

                    else:
                        raise Exception(f'Unknown channel key {channel_key}')

                # Add them to the list of all DCO seeds
                All_DCO_seeds.extend(DCO_seeds[channel_bool])
            else:
                continue

        # take the unique seeds (some SEEDS might make a DCO at multiple metallicities)
        All_DCO_seeds  = np.unique(All_DCO_seeds)

        # Save the seeds to a file
        np.savetxt(datar_root+ f'/{sim_name}/All_DCO_seeds{channel_key}.txt', All_DCO_seeds)

        # Open the HDF5 file for all systems at all metallicities (This is heavy on the memory)
        All_data = h5.File(datar_root+ f'/{sim_name}/COMPAS_Output_combinedZ.h5', 'r')

        # Create a mask to select only the systems that could potentially become a DCO
        SYS_mask = np.in1d(All_data['BSE_System_Parameters']['SEED'][()], All_DCO_seeds)

        # Read the HDF5 datasets as pandas dataframes
        SYS = pd.DataFrame()
        # chosen to allow for rerunning of systems and other interesting parameters
        SYS_keys_of_interest = ['SEED', 'Metallicity@ZAMS(1)', 'Stellar_Type(1)', 'Stellar_Type(2)','CE_Event_Counter', 'Mass@ZAMS(1)', 'Mass@ZAMS(2)','SemiMajorAxis@ZAMS',
                                'Merger','Merger_At_Birth','Unbound', 'Immediate_RLOF>CE','Optimistic_CE', 'Applied_Kick_Magnitude(1)', 'Applied_Kick_Magnitude(2)', 'CH_on_MS(1)',
                                'SN_Kick_Magnitude_Random_Number(1)','SN_Kick_Phi(1)','SN_Kick_Theta(1)','SN_Kick_Mean_Anomaly(1)',
                                'SN_Kick_Magnitude_Random_Number(2)','SN_Kick_Phi(2)','SN_Kick_Theta(2)','SN_Kick_Mean_Anomaly(2)' ]
        for key in SYS_keys_of_interest:
            # You cant directly apply the mask to the HDF5 dataset, so you have to read it first
            read_data = All_data['BSE_System_Parameters'][key][()]
            SYS[key] = read_data[SYS_mask]

        # Same mask for the DCO 
        DCO_mask = np.in1d(All_data['BSE_Double_Compact_Objects']['SEED'][()], All_DCO_seeds)

        DCO = pd.DataFrame()
        DCO_keys_of_interest = ['SEED', 'Metallicity@ZAMS(1)', 'Merges_Hubble_Time', 'SemiMajorAxis@DCO','Coalescence_Time', 'Eccentricity@DCO', 'MT_Donor_Hist(1)', 'MT_Donor_Hist(2)', 'Mass(1)', 'Mass(2)']
        for key in DCO_keys_of_interest:
            read_data = All_data['BSE_Double_Compact_Objects'][key][()]
            DCO[key] = read_data[DCO_mask]

        # Merge the SYS and DCO dataframes to make potential_DCO_progenitors
        potential_DCO_progenitors = SYS.merge(DCO, on=['SEED', 'Metallicity@ZAMS(1)'], how='left')
        # Create a unique SEED that is a combination of SEED and metallicity
        potential_DCO_progenitors['unique_Z_SEED'] = [f"{seed}_{Z:.5f}" for seed, Z in zip(potential_DCO_progenitors['SEED'], potential_DCO_progenitors['Metallicity@ZAMS(1)'])]

        # test that this worked (every seed should occur len(metallicity) times, once for each Z)
        potential_DCO_seeds, counts = np.unique(potential_DCO_progenitors['SEED'], return_counts=True)
        print('potential_DCO_seeds', potential_DCO_seeds, 'counts', counts)

        # Save the total dataframe
        potential_DCO_progenitors.to_hdf(datar_root+ f'/{sim_name}/{save_name_table}', key='All_DCO', mode='w')

        return potential_DCO_progenitors

############################################################################
# ### Now we add the RLOF information to the potential DCO progenitor table

# Use the above created table to start from and add
# * mass transfer 1
# * First MT from star 2
# * the mass transfer that lead to a merger

############################################################################

def add_RLOF_info_to_potential_DCO_progenitors(datar_root, sim_name, channel_key):

    save_name_table = f'potential_DCO_progenitors_RLOFinfo{channel_key}.h5'

    # check if your table exists
    if os.path.isfile(datar_root+ f'/{sim_name}/{save_name_table}'):
        print('RLOF DCO table already exists, loading it')
        potential_DCO_progenitors = pd.read_hdf(datar_root + f'/{sim_name}/{save_name_table}', key='All_DCO')

        return potential_DCO_progenitors

    else:
        # Read the beginning of the potential DCO progenitors table that you made above
        potential_DCO_progenitors = pd.read_hdf(f'{datar_root}/{sim_name}/potential_DCO_progenitors{channel_key}.h5', key='All_DCO')

        # take the unique seeds (some SEEDS might make a DCO at multiple metallicities)
        unique_potentialDCO_seeds = np.unique(potential_DCO_progenitors['SEED'])

        # Open the HDF5 file for all systems at a given metallicity
        with h5.File(datar_root+ f'/{sim_name}/COMPAS_Output_combinedZ.h5', 'r') as All_data:
            ####################################
            # Read RLOF data
            RLOF = pd.DataFrame()
            # Select only the RLOF events for systems that could potentially become a DCO
            RLOF_mask = np.in1d(All_data['BSE_RLOF']['SEED'][()], unique_potentialDCO_seeds)

            ########################################################################################
            # Create the new RLOF columns in the potential DCO progenitors table
            print('start reading RLOF data')
            RLOF_keys = ['SEED', 'Metallicity@ZAMS(1)','SemiMajorAxis<MT', 'SemiMajorAxis>MT', 'Radius(1)<MT', 'Radius(2)<MT', 'Radius(1)>MT', 
                        'Radius(2)>MT', 'Mass(1)<MT', 'Mass(2)<MT', 'Mass(1)>MT', 'Mass(2)>MT','Stellar_Type(1)<MT', 'Stellar_Type(2)<MT', 
                        'Stellar_Type(1)>MT', 'Stellar_Type(2)>MT', 'MT_Event_Counter', 'CEE>MT', 'RLOF(1)>MT', 'RLOF(2)>MT', 'Merger']
            for key in RLOF_keys:
                read_data = All_data['BSE_RLOF'][key][()]
                RLOF[key] = read_data[RLOF_mask]
            print('add a few extra cols to RLOF')
            RLOF['unique_Z_SEED'] = [f"{seed}_{Z:.5f}" for seed, Z in zip(RLOF['SEED'], RLOF['Metallicity@ZAMS(1)'])]
            RLOF['M1_M2<MT'] = RLOF['Mass(1)<MT']/RLOF['Mass(2)<MT'].astype(float)
            RLOF.rename(columns={'Merger': 'RLOF_Merger'}, inplace=True) # rename so it wont conflict with the SYS['Merger']


        ########################################################################################
        # Now make subselections of this RLOF table that we can later add to the potential DCO progenitors table
        # first_RLOF_table    = first_RLOF_table.add_prefix('firstMT_')                                               # Add prefix to all the keys
        # first_RLOF_table.rename(columns={'firstMT_unique_Z_SEED': 'unique_Z_SEED'}, inplace=True)                   # Except the unique_Z_SEED
        ################################
        # First MT event
        print('Adding the MT information for the first MT ')
        first_MT_event_bool = RLOF['MT_Event_Counter'] == 1
        first_RLOF_table    = RLOF[first_MT_event_bool]

        # Add prefix to all the keys, Except 'SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'
        first_RLOF_table.rename(columns={col: 'firstMT_' + col for col in first_RLOF_table.columns if col not in ['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED']}, inplace=True)

        # a 'first MT' should only happen once per unique_Z_SEED
        s, counts = np.unique(first_RLOF_table['unique_Z_SEED'], return_counts = True)
        print('this should be 1', np.unique(counts))

        ################################
        # mass transfer that lead to a merger
        print('Adding the MT information for the mass transfer that lead to a merger')
        RLOF_Merger_bool            = RLOF['RLOF_Merger'] == 1
        MT_leading_to_merger_table  = RLOF[RLOF_Merger_bool]

        # Add prefix to all the keys, Except 'SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'
        MT_leading_to_merger_table.rename(columns={col: 'MT_lead_to_merger_' + col for col in MT_leading_to_merger_table.columns if col not in ['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED']}, inplace=True)

        # a 'mass transfer leading to stellar merger' should only happen once per unique_Z_SEED
        s, counts = np.unique(MT_leading_to_merger_table['unique_Z_SEED'], return_counts = True)
        print('this should be 1', np.unique(counts))

        ################################################################
        # First mass transfer from the second star
        print('Adding the MT information for the first MT from star 2')
        star_2_is_RLOF              = RLOF['RLOF(2)>MT'] == 1 # Star 2 is RLOF
        star_2_is_RLOF_table        = RLOF[star_2_is_RLOF]

        # Find the minimum 'MT_Event_Counter' for each 'unique_Z_SEED' where Star 2 is RLOF
        Minimun_MT_event_count_bool = np.where(star_2_is_RLOF_table['MT_Event_Counter'] == star_2_is_RLOF_table.groupby('unique_Z_SEED')['MT_Event_Counter'].transform('min'), True, False)
        first_MT_from_star2_table   = star_2_is_RLOF_table[Minimun_MT_event_count_bool].copy()

        # Add prefix to all the keys, Except 'SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'
        first_MT_from_star2_table.rename(columns={col: 'star2_firstMT_' + col for col in first_MT_from_star2_table.columns if col not in ['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED']}, inplace=True)

        # lastly, the first MT from star 2 should also only happen once per unique_Z_SEED
        s, counts = np.unique(first_MT_from_star2_table['unique_Z_SEED'], return_counts = True)
        print('this should be 1', np.unique(counts))

        ########################
        # Empty RLOF to save mem
        del RLOF
        gc.collect()  # Force garbage collector to release unreferenced memory

        print('start merging tables')
        # Merge this info with the potential_DCO_progenitors
        potential_DCO_progenitors = potential_DCO_progenitors.merge(first_RLOF_table, on=['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'], how='left')
        print('done with first_RLOF_table')
        del first_RLOF_table # to save memory
        # Merge info with the potential_DCO_progenitors
        potential_DCO_progenitors = potential_DCO_progenitors.merge(MT_leading_to_merger_table, on=['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'], how='left')
        print('done with MT_leading_to_merger_table')
        # Merge this info with the potential_DCO_progenitors
        potential_DCO_progenitors = potential_DCO_progenitors.merge(first_MT_from_star2_table, on=['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'], how='left')
        print('done with first_MT_from_star2_table')

        #################################################################################
        # Save the dataframe
        print('Done!, Saving the potential DCO progenitors with MT info')
        potential_DCO_progenitors.to_hdf(datar_root+ f'/{sim_name}/{save_name_table}', key='All_DCO', mode='w')

        return potential_DCO_progenitors

############################################################################
# ### Finally also supernova information 

# #####  'Supernova_State'
# *  No supernova = 0 
# *  Star 1 is the supernova 	 = 1 
# *  Star 2 is the supernova 	 = 2 
# *  Both stars are supernovae 	 = 3

# ##### 'SN_Type(SN)'
# *  NONE 	 = 0 
# *  CCSN 	 = 1 
# *  ECSN 	 = 2 
# *  PISN 	 = 4 
# *  PPISN 	 = 8 
# *  USSN 	 = 16 
# *  AIC 	     = 32 
# *  SNIA 	 = 64 
# *  HeSD 	 = 128

############################################################################
def Add_SN_info_to_potential_DCO_progenitors(datar_root, sim_name, channel_key):

    save_name_table = f'potential_DCO_progenitors_Allinfo{channel_key}.h5'

    # check if your table exists
    if os.path.isfile(datar_root+ f'/{sim_name}/{save_name_table}'):
        print('Table already exists, loading it')
        potential_DCO_progenitors = pd.read_hdf(datar_root + f'/{sim_name}/{save_name_table}', key='All_DCO')

    else:
        # Read the beginning of the potential DCO progenitors table that you made above
        potential_DCO_progenitors = pd.read_hdf(f'{datar_root}/{sim_name}/potential_DCO_progenitors_RLOFinfo{channel_key}.h5', key='All_DCO')

        # take the unique seeds (some SEEDS might make a DCO at multiple metallicities)
        unique_potentialDCO_seeds = np.unique(potential_DCO_progenitors['SEED'])

        # Open the HDF5 file for all systems at a given metallicity
        All_data = h5.File(datar_root+ f'/{sim_name}/COMPAS_Output_combinedZ.h5', 'r')
        #################################################################################
        # Finally, add supernove information
        print('Adding the SN information')
        with h5.File(datar_root+f'/{sim_name}/COMPAS_Output_combinedZ.h5', 'r') as All_data:        
            # Read SN info as pandas dataframes
            SNe = pd.DataFrame()
            
            # Select only the SN events for systems that could potentially become a DCO
            SN_mask = np.in1d(All_data['BSE_Supernovae']['SEED'][()], unique_potentialDCO_seeds)

            SN_keys_of_interest = ['SEED', 'Metallicity@ZAMS(1)', 'SN_Type(SN)', 'Supernova_State',\
                                'Unbound', 'Applied_Kick_Magnitude(SN)', 'Fallback_Fraction(SN)', 'Mass_CO_Core@CO(SN)', \
                                'Mass_Core@CO(SN)', 'Mass_He_Core@CO(SN)', 'Mass_Total@CO(SN)', 'Orb_Velocity<SN']
            for key in SN_keys_of_interest:
                read_data   = All_data['BSE_Supernovae'][key][()]
                SNe[key]    = read_data[SN_mask]

            #Add unique seed key
            SNe['unique_Z_SEED'] = [f"{seed}_{Z:.5f}" for seed, Z in zip(SNe['SEED'], SNe['Metallicity@ZAMS(1)'])]


        ########################################################################################
        # Now make subselections of this SN table that we can add to the potential DCO progenitors table
        # # Star 1 is going SN
        # star1_SN = SNe[SNe['Supernova_State'] == 1]
        # # Star 2 is going SN
        # star2_SN = SNe[SNe['Supernova_State'] == 2]
        ################################
        # First star is going SN
        print('Adding the SN information for star 1 going SN')
        star1_going_SN_bool = SNe['Supernova_State'] == 1
        SN1_table    = SNe[star1_going_SN_bool].copy()

        # Add prefix to all the keys, Except 'SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'
        SN1_table.rename(columns={col: 'SN_star1_' + col for col in SN1_table.columns if col not in ['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED']}, inplace=True)

        # Star 1 should only go SN once per unique_Z_SEED
        s, counts = np.unique(SN1_table['unique_Z_SEED'], return_counts = True)
        print('this should be 1', np.unique(counts))

        # Merge this info with the potential_DCO_progenitors
        potential_DCO_progenitors = potential_DCO_progenitors.merge(SN1_table, on=['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'], how='left')
        print('done with SN1')
        del SN1_table # to save memory

        ################################
        # Second star is going SN
        print('Adding the SN information for star 2 going SN')
        star2_going_SN_bool = SNe['Supernova_State'] == 2
        SN2_table    = SNe[star2_going_SN_bool].copy()

        # Add prefix to all the keys, Except 'SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'
        SN2_table.rename(columns={col: 'SN_star2_' + col for col in SN2_table.columns if col not in ['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED']}, inplace=True)

        # Star 1 should only go SN once per unique_Z_SEED
        s, counts = np.unique(SN2_table['unique_Z_SEED'], return_counts = True)
        print('this should be 1', np.unique(counts))

        # Merge this info with the potential_DCO_progenitors
        potential_DCO_progenitors = potential_DCO_progenitors.merge(SN2_table, on=['SEED', 'Metallicity@ZAMS(1)', 'unique_Z_SEED'], how='left')
        print('done with SN1')
        del SN2_table # to save memory

        # Add the SN info to the potential DCO progenitors
        # potential_DCO_progenitors['SN_Type(1)'] = potential_DCO_progenitors['unique_Z_SEED'].map(SNe[SNe['Supernova_State'] == 1].set_index('unique_Z_SEED')['SN_Type(SN)']).fillna(-1)
        # potential_DCO_progenitors['SN_Type(2)'] = potential_DCO_progenitors['unique_Z_SEED'].map(SNe[SNe['Supernova_State'] == 2].set_index('unique_Z_SEED')['SN_Type(SN)']).fillna(-1)

        # del SNe # empty SNe to save memory
        gc.collect()  # Force garbage collector to release unreferenced memory

        #################################################################################
        # Save the dataframe
        print('Done!, Saving the potential DCO progenitors with SN info')
        potential_DCO_progenitors.to_hdf(datar_root+ f'/{sim_name}/{save_name_table}', key='All_DCO', mode='w')

        return potential_DCO_progenitors



############################################################################
############################################################################

def main(sim_name = 'NewWinds_RemFryer2012', channel_key = '', compas_v = "v03.01.02"):
    """
    Args:
        sim_name (str): what simulation to use, e.g. 'OldWinds_RemFryer2012_noBHkick'
        channel_key (str): which channel to run the analysis on options: default = '' for all  ('_stable', '_CE', '_CHE'  )
        compas_v (str):  "v03.01.02", or  "v02.46.01" v02.35.02"
    """
    ##############
    home_dir = os.path.expanduser("~") 
    datar_root =  f"{home_dir}/ceph/CompasOutput/{compas_v}/"
    ##############

    # Create initial set of potential DCO progenitors
    potential_DCO_progenitors = create_first_potential_DCO_progenitors_table(datar_root, sim_name, channel_key)

    # Add RLOF information to the potential DCO progenitors table
    potential_DCO_progenitors = add_RLOF_info_to_potential_DCO_progenitors(datar_root, sim_name, channel_key)

    # Add SN information to the potential DCO progenitors table
    potential_DCO_progenitors = Add_SN_info_to_potential_DCO_progenitors(datar_root, sim_name, channel_key)    

    # Split between BBH, BHNS and NSNS progenitors (test)
    with h5.File(datar_root+f'{sim_name}/COMPAS_Output_combinedZ.h5', 'r') as All_data:
        DCO = All_data['BSE_Double_Compact_Objects']
        st1 = DCO['Stellar_Type(1)'][()]
        st2 = DCO['Stellar_Type(2)'][()]
        dco_merger = DCO['Merges_Hubble_Time'][()]  
        DCO_seed = DCO['SEED'][()]
        # Now I want to add a bool that tells me if this system is ever a BBH, BHNS or BNS progenitor
        BBH_bool = np.logical_and(st1 == 14,st2 == 14)
        BHNS_bool = np.logical_or(np.logical_and(st1 == 13,st2 == 14),
                                np.logical_and(st1 == 14,st2 == 13) )
        NSNS_bool = np.logical_and(st1 == 13,st2 == 13)
        merger_bool = dco_merger == 1

        # Split our potential DCO progenitors into BBH, BHNS and NSNS progenitors
        potential_DCO_progenitors['pot_BHBH_bool'] = np.in1d(potential_DCO_progenitors['SEED'], np.unique(DCO_seed[BBH_bool*merger_bool]) )
        potential_DCO_progenitors['pot_BHNS_bool'] = np.in1d(potential_DCO_progenitors['SEED'], np.unique(DCO_seed[BHNS_bool*merger_bool]) )
        potential_DCO_progenitors['pot_NSNS_bool'] = np.in1d(potential_DCO_progenitors['SEED'], np.unique(DCO_seed[BHNS_bool*merger_bool]) )

        # potential_BBH_progenitors  = potential_DCO_progenitors[np.in1d(potential_DCO_progenitors['SEED'], np.unique(DCO_seed[BBH_bool*merger_bool]) )]
        # potential_BHNS_progenitors = potential_DCO_progenitors[np.in1d(potential_DCO_progenitors['SEED'], np.unique(DCO_seed[BHNS_bool*merger_bool]) )]
        potential_NSNS_progenitors = potential_DCO_progenitors[np.in1d(potential_DCO_progenitors['SEED'], np.unique(DCO_seed[NSNS_bool*merger_bool]) )]
        print(potential_NSNS_progenitors.keys(), 'potential_NSNS_progenitors.keys()')

    print(f'Finished with {sim_name},  {channel_key} ')

############################################################################
############################################################################

