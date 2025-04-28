import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import shutil
import pandas as pd
import numpy as np
from pathlib import Path, PosixPath, WindowsPath
from octolyzer import key_descriptions, utils


KEY_LAYER_DICT = {"ILM": "Inner Limiting Membrane",
                  "RNFL": "Retinal Nerve Fiber Layer",
                  "GCL": "Ganglion Cell Layer",
                  "IPL": "Inner Plexiform Layer",
                  "INL": "Inner Nuclear Layer",
                  "OPL": "Outer Plexiform Layer",
                  "ELM": "External Limiting Membrane", # Outer nuclear layer
                  "PR1": "Photoreceptor Layer 1",
                  "PR2": "Photoreceptor Layer 2",
                  "RPE": "Retinal Pigment Epithelium",
                  "BM": "Bruch's Membrane Complex", 
                  "CHORupper": "Choroid - Sclera boundary",
                  "CHORlower": "Bruch's Membrane - Choroid boundary"}

# Load in previously analysed results (if saved out)
def load_files(fname_path, logging_list=[], analyse_square=0, only_slo=0, only_oct = 0, verbose=1):
    '''
    For any files detected as previously analysed SLO, try load them for
    collating files
    '''
    msg = "Loading in measurements..."
    if verbose:
        print(msg)
    logging_list.append(msg)
    if isinstance(Path(fname_path), PosixPath):
        fname = str(fname_path).split('/')[-1]
    elif isinstance(Path(fname_path), WindowsPath):
        fname = str(fname_path).split("\\")[-1]
    output_fname = os.path.join(fname_path, f"{fname}_output.xlsx")

    # Load metadata
    meta_df = pd.read_excel(output_fname, sheet_name="metadata")
    outputs = [meta_df]
    oct_dfs = []
    slo_dfs = []

    # Load in SLO measurements
    if not only_oct:
        if meta_df.bscan_type.iloc[0] == "Peripapillary":
            for r in ["B", "C", "whole"]:
                try:
                    df = pd.read_excel(output_fname, sheet_name=f"slo_measurements_{r}")
                except:
                    df = pd.DataFrame()
                slo_dfs.append(df)
        else:
            try:
                df = pd.read_excel(output_fname, sheet_name=f"slo_measurements_whole")
            except:
                df = pd.DataFrame()
            slo_dfs.append(df)
        outputs.append(slo_dfs)

    # Load in OCT measurements
    if not only_slo:
        if meta_df.bscan_type.iloc[0] == "Ppole":
            if analyse_square:
                keys = ['etdrs_measurements', 'etdrs_volume_measurements', 'square_measurements', 'square_volume_measurements']
            else:
                keys = ['etdrs_measurements', 'etdrs_volume_measurements']
        else:
            keys = ['oct_measurements']
        for key in keys:
            try:
                df = pd.read_excel(output_fname, sheet_name=key)
                oct_dfs.append(df)
            except:
                if verbose:
                    print(f'{key} does not exist. Skipping.')
        outputs.append(oct_dfs)
    
    msg = "Successfully loaded in all measurements!\n"
    if verbose:
        print(msg)
    logging_list.append(msg)
    outputs.append(logging_list)

    return outputs



def collate_results(result_dict, save_directory, analyse_choroid=1, analyse_slo=0):
    '''
    Wrapper function to collate all SLO results
    '''
    print(f"Collecting all results together into one output file.")
    all_logging_list = []
    all_slo_df = pd.DataFrame()
    all_ind_df = pd.DataFrame()
    all_oct_macula_df = pd.DataFrame()
    all_oct_ppole_df = pd.DataFrame()
    all_oct_sq_ppole_df = pd.DataFrame()
    all_oct_peri_df = pd.DataFrame()
    for fname_type, output in result_dict.items():
        
        fname = fname_type.split(".")[0]
        all_logging_list.append(f"\n\nANALYSIS LOG OF {fname_type}")

        # Create default dataframe for failed individuals and save out error
        if isinstance(output['oct'], str):
            log = output['log']
            metadata = output['metadata']
            slo_dfs = pd.DataFrame()
            oct_dfs = pd.DataFrame()
            fname_path = os.path.join(save_directory, fname)
            all_logging_list.extend(log)    
            with open(os.path.join(fname_path, f"{fname}_log.txt"), "w") as f:
                for line in log:
                    f.write(line+"\n")
            ind_df = pd.DataFrame(metadata, index=[0])
            # ind_df = pd.DataFrame({"Filename":fname_type}, index=[0])
        
        # Otherwise, collate measurements and save out segmentations if specified
        else:    
            
            ind_df = output['metadata']
            bscan_type = ind_df.bscan_type.iloc[0]
            slo_dfs = output['slo']
            oct_dfs = output['oct']
            N_oct = len(oct_dfs)
            logging_list = output['log']

            # If cannot find measurements/etadata, create default dataframe and bypass
            # the segmentation visualisation
            all_logging_list.extend(logging_list)

            # process SLO measurement DFs and save out global dataframe
            flat_slo_df = pd.DataFrame()
            if len(slo_dfs[0]) > 0:
                rtypes = []
                for df in slo_dfs:
                    if len(df) == 0:
                        continue
                    # flatten
                    dfarr = df.values.flatten()
    
                    # Collect all columns in flattened DF
                    dfcols = list(df.columns)
                    rtype = df.zone.iloc[0]
                    rtypes.append(rtype)
                    vtypes = df.vessel_map.values
                    dfcols = [col+f"_{vtype}_{rtype}" for vtype in vtypes for col in dfcols]
                    df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])
    
                    # Remove indexing columns and concatenate with different zones
                    cols_to_drop = df_flatten.columns[df_flatten.columns.str.contains("vessel_map|zone")]
                    df_flatten = df_flatten.drop(cols_to_drop, axis=1, inplace=False)
                    flat_slo_df = pd.concat([flat_slo_df, df_flatten], axis=1)    

                # Order feature columns by importance in literature
                order_str = ["fractal_dimension", "vessel_density", "tortuosity_density", 'average_global_calibre', 
                             'average_local_calibre', 'CRAE_Knudtson', 'CRVE_Knudtson', 'AVR']
                order_str_rv = [col+f"_{vtype}_{rtype}" 
                                    for rtype in rtypes[::-1] 
                                        for vtype in vtypes for col in order_str]
                flat_slo_df = flat_slo_df[order_str_rv]
                flat_slo_df = flat_slo_df.loc[:, ~(flat_slo_df == -1).any()]
                flat_slo_df = flat_slo_df.rename({f"AVR_artery-vein_{rtype}":f"AVR_{rtype}" 
                                                  for rtype in rtypes}, inplace=False, axis=1)

                # Concatenate measurements with metadata filename
                ind_slo_df = pd.concat([ind_df['Filename'], flat_slo_df], axis=1)

            # process OCT measurement DFs and save out global dataframe
            flat_oct_df = pd.DataFrame()
            flat_sq_oct_df = pd.DataFrame()
            if bscan_type != 'Ppole':
                for idx, df in enumerate(oct_dfs):
                    
                    # Remove choroid measurements if not analysing choroid
                    if not analyse_choroid:
                        df = df[df['layer'] != 'CHORupper_CHORlower']
                    
                    # flatten
                    dfarr = df.values[:,1:].flatten()
                
                    # Collect all columns in flattened DF
                    dfcols = list(df.columns[1:])
                    ltypes = df.layer.values
                    dfcols = [f"{layer}_{col}" for layer in ltypes for col in dfcols]
                    df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])

                    # Concatenate to flatten dF
                    flat_oct_df = pd.concat([flat_oct_df, df_flatten], axis=1).dropna(axis=1)
            else:
                for idx, df in enumerate(oct_dfs[:2]):

                    # Remove choroid measurements if not analysing choroid
                    if not analyse_choroid:
                        df = df[~df['map_name'].isin(['choroid_vessel', 'choroid_CVI', 'choroid'])]
                    
                    # flatten
                    dfarr = df.values[:,2:].flatten()
                
                    # Collect all columns in flattened DF. 
                    dfcols = list(df.columns[2:])
                    mtypes = df.map_name.values
                    retinal_maps = [mtype for mtype in mtypes if "choroid" not in mtype]
                    for key in ['retina', 'inner_retina', 'outer_retina']:
                        if key in retinal_maps:
                            retinal_maps.remove(key)
                    # mtypes = [mtype.split('_')[0] if mtype not in ['choroid_CVI', 'choroid_vessel']+retinal_maps else mtype for mtype in mtypes]
                    # mtypes = [mtype if mtype not in ['choroid_CVI', 'choroid_vessel'] else ''.join(mtype.split('_')) for mtype in mtypes]
                    
                    # For thickness/area, then for volume
                    if idx == 0:
                        dfcols = [col+f"_{mtype}_[um]" 
                                          if mtype not in ['choroid_CVI', 'choroid_vessel'] 
                                          else [col+f"_{mtype}", col+f"_{mtype}_[um2]"][mtype=='vessel'] 
                                    for mtype in mtypes for col in dfcols]
                    else:
                        dfcols = [col+f"_{mtype}_[mm3]" for mtype in mtypes for col in dfcols]

                    df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])

                    # Concatenate to flatten dataframe
                    flat_oct_df = pd.concat([flat_oct_df, df_flatten], axis=1).dropna(axis=1)

                if N_oct > 2:
                    for idx, df in enumerate(oct_dfs[2:]):
                        # flatten
                        dfarr = df.values[:,2:].flatten()
                    
                        # Collect all columns in flattened DF. 
                        dfcols = list(df.columns[2:])
                        mtypes = df.map_name.values
                        retinal_maps = [mtype for mtype in mtypes if "choroid" not in mtype]
                        for key in ['retina', 'inner_retina', 'outer_retina']:
                            if key in retinal_maps:
                                retinal_maps.remove(key)
                        # mtypes = [mtype.split('_')[0] if mtype not in ['choroid_CVI', 'choroid_vessel']+retinal_maps else mtype for mtype in mtypes]
                        # mtypes = [mtype if mtype not in ['choroid_CVI', 'choroid_vessel'] else ''.join(mtype.split('_')) for mtype in mtypes]
                        
                        # For thickness/area, then for volume
                        if idx == 0:
                            dfcols = [col+f"_{mtype}_[um]" 
                                              if mtype not in ['choroid_CVI', 'choroid_vessel'] 
                                              else [col+f"_{mtype}", col+f"_{mtype}_[um2]"][mtype=='vessel'] 
                                        for mtype in mtypes for col in dfcols]
                        else:
                            dfcols = [col+f"_{mtype}_[mm3]" for mtype in mtypes for col in dfcols]
    
                        df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])
    
                        # Concatenate to flatten dataframe
                        flat_sq_oct_df = pd.concat([flat_sq_oct_df, df_flatten], axis=1).dropna(axis=1)
                        
            # Concatenate measurements with metadata
            ind_oct_df = pd.concat([ind_df['Filename'], flat_oct_df], axis=1)
            if N_oct > 2:
                ind_sq_oct_df = pd.concat([ind_df['Filename'], flat_sq_oct_df], axis=1)
            # print(all_oct_sq_ppole_df)
            # print(N_oct)
                
            # Concatenate to create global dataframe of SLO results
            if len(slo_dfs[0]) > 0:
                all_slo_df = pd.concat([all_slo_df, ind_slo_df], axis=0)
    
            # Append row to global dataframe of OCT results dependent on bscan type
            if bscan_type == 'Ppole':
                all_oct_ppole_df = pd.concat([all_oct_ppole_df, ind_oct_df], axis=0)
                if N_oct > 2:
                    all_oct_sq_ppole_df = pd.concat([all_oct_sq_ppole_df, ind_sq_oct_df], axis=0)
            elif bscan_type == 'Peripapillary':
                all_oct_peri_df = pd.concat([all_oct_peri_df, ind_oct_df], axis=0)
            else:
                all_oct_macula_df = pd.concat([all_oct_macula_df, ind_oct_df], axis=0)

        # Concenate metadata to global dataframe of metadata, robust to suppress NaN warnings
        # from different Bscan types having different columns
        if (len(all_ind_df)) > 0 & (len(ind_df) > 0):
            all_ind_df = pd.concat([all_ind_df.fillna(""), ind_df.fillna("")], axis=0)
        else:
            all_ind_df = ind_df.copy()

    # Reset index
    all_ind_df = all_ind_df.reset_index(drop=True)
    all_slo_df = all_slo_df.reset_index(drop=True)
    all_oct_ppole_df = all_oct_ppole_df.reset_index(drop=True)
    all_oct_sq_ppole_df = all_oct_sq_ppole_df.reset_index(drop=True)
    all_oct_peri_df = all_oct_peri_df.reset_index(drop=True)
    all_oct_macula_df = all_oct_macula_df.reset_index(drop=True)

    # Define columns which indicate any processing failures
    fail_cols = []
    if all_oct_macula_df.shape[0] > 0:
        fail_cols.extend(['bscan_missing_fovea', 
                          'slo_missing_fovea', 
                          'missing_retinal_oct_linescan_measurements',
                          'missing_choroid_oct_linescan_measurements',])
    if all_oct_peri_df.shape[0] > 0:
        fail_cols.extend(['optic_disc_overlap_warning'])
    if 'FAILED' in all_ind_df.columns:
        fail_cols.append('FAILED')
    fail_df = all_ind_df[all_ind_df[fail_cols].any(axis=1)][['Filename'] + fail_cols]
    fail_df.replace(1, True, inplace=True)
    fail_df.replace(False, '', inplace=True)
    fail_df.replace(0, '', inplace=True)   

    # Remove any rows in all_oct_macula_df which are just -1s, i.e. fovea was not detected
    all_oct_macula_df = all_oct_macula_df[~(all_oct_macula_df.iloc[:, 1:]==-1).all(axis=1)]

    # Layer keys, ordered anaomtically
    key_df = pd.DataFrame({"key":KEY_LAYER_DICT.keys(),
                        "layer":KEY_LAYER_DICT.values()})
    if not analyse_choroid:
        key_df = key_df[~key_df.key.str.contains("CHOR")]
  
    # save out global metadata and measurements
    with pd.ExcelWriter(os.path.join(save_directory, f'analysis_output.xlsx')) as writer:
        # write metadata
        all_ind_df.to_excel(writer, sheet_name='metadata', index=False)

        # SLO
        if all_slo_df.shape[0] > 0:
            all_slo_df.to_excel(writer, sheet_name='SLO_measurements', index=False)
        else:
            if analyse_slo:
                print('WARNING: analyse_slo flag is 1, but there are no SLO measurements loaded!')

        # OCT measurements, save out sheets if populated
        if all_oct_macula_df.shape[0] > 0:
            all_oct_macula_df.to_excel(writer, sheet_name='OCT_Linescan_measurements', index=False)
            
        if all_oct_ppole_df.shape[0] > 0: 
            all_oct_ppole_df.to_excel(writer, sheet_name='OCT_Ppole_ETDRS_measurements', index=False)
            img_path = os.path.join(MODULE_PATH, os.path.join('figures','etdrs_posterior_pole_grid.png'))
            fname = os.path.split(img_path)[1]
            shutil.copy(img_path, os.path.join(save_directory, fname))
            
        if all_oct_sq_ppole_df.shape[0] > 0:
            all_oct_sq_ppole_df.to_excel(writer, sheet_name='OCT_Ppole_Square_measurements', index=False)
            img_path = os.path.join(MODULE_PATH, os.path.join('figures','square_posterior_pole_grid.png'))
            fname = os.path.split(img_path)[1]
            shutil.copy(img_path, os.path.join(save_directory, fname))
            
        if all_oct_peri_df.shape[0] > 0:
            all_oct_peri_df.to_excel(writer, sheet_name='OCT_Peripapillary_measurements', index=False)

        # Save out datasheet specifying any files which failed for some reason
        if fail_df.shape[0] > 0:
            fail_df.to_excel(writer, sheet_name='Failed', index=False)

        # Save out metadata key and descriptions
        key_descriptions.metakey_df.to_excel(writer, sheet_name='metadata_keys', index=False)

        # write out layer keys
        key_df.to_excel(writer, sheet_name="layer_keys", index=False)

        # Save out column descriptions if necessary
        if all_oct_macula_df.shape[0] > 0:
            key_descriptions.linescanradial_df.to_excel(writer, sheet_name='OCT_Linescan_keys', index=False)
        if all_oct_ppole_df.shape[0] > 0:
            key_descriptions.ppole_df.to_excel(writer, sheet_name='OCT_Ppole_ETDRS_keys', index=False)
        if all_oct_peri_df.shape[0] > 0:
            key_descriptions.peripapillary_df.to_excel(writer, sheet_name='SLOCT_Peripapillary_keysO_keys', index=False)
        if all_slo_df.shape[0] > 0:
            key_descriptions.slo_df.to_excel(writer, sheet_name='SLO_keys', index=False)
   
    # save out log
    with open(os.path.join(save_directory, f"analysis_log.txt"), "w") as f:
        for line in all_logging_list:
            f.write(line+"\n")