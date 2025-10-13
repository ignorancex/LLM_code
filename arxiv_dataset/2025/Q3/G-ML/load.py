"""
Created on Mon Feb 03 2025, 11:16:32

@author: Amoy Ashesh
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from parameters import file_paths
import os

def mark_data(save_data:bool):
    """
    This function marks the targets in the dataset.
    
    Parameters:
    -----------
    save_data : bool
        Boolean variable to save the data.

    Returns:
    --------
    dt : pandas.core.frame.DataFrame
        The marked dataset.
    """
    file,catalogue = file_paths.raw_data_file, file_paths.catalogue_file
    try:
        if file is not None:
            # Load the raw data
            hdul = fits.open(file)
            data = hdul[1].data
            raw_df = pd.DataFrame(data)
            # Create a column for marking the target
            dt = raw_df.copy()
            dt['target'] = 0 
            # Load the catalogue             
            hdul_cat = fits.open(catalogue)
            data_cat = hdul_cat[1].data
            df_cat = pd.DataFrame(data_cat)
            df_cat.drop(columns=['solution_id1','solution_id2'], inplace=True)
            # Mark the targets
            source_ids_set = set(df_cat['source_id1']).union(set(df_cat['source_id2']))
            dt['target'] = dt['source_id'].isin(source_ids_set).astype(int)
            # Filtering 
            dt = dt.dropna(axis=1, how='any')
            print("Data loaded successfully. Here's a summary:")
            # Save the marked dataset in the folder marked
            if save_data:
                if not os.path.exists('marked'):
                    os.makedirs('marked')
                dt.to_csv('marked/marked_data.csv', index=False)
            return dt
    except FileNotFoundError:
        print(f"Error: File not found at {file}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file} is empty")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        return None

# def load_models:



def inspect():
    """
    This function can be used to inspect the initial dataset.
    
    Parameters:
    -----------
    None.

    Returns:
    --------
    raw_df : pandas.core.frame.DataFrame
        The raw dataset.
    """
    file,catalogue = file_paths.raw_data_file, file_paths.catalogue_file
    try:
        if file is not None:
            # Perform operations on the loaded data
            hdul = fits.open(file)
            data = hdul[1].data
            raw_df = pd.DataFrame(data)
            print(f"Number of rows: {len(raw_df)}")
            print(f"Number of columns: {len(raw_df.columns)}")
            print("\nFirst few rows of the data:")
            print(raw_df.head())
            return raw_df
    except FileNotFoundError:
        print(f"Error: File not found at {file}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file} is empty")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        return None