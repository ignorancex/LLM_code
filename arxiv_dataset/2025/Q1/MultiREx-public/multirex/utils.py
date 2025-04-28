#########################################
#  __  __      _ _   _ ___ ___          #
# |  \/  |_  _| | |_(_) _ \ __|_ __     #
# | |\/| | || | |  _| |   / _|\ \ /     #
# |_|  |_|\_,_|_|\__|_|_|_\___/_\_\     #
# Planetary spectra generator           #
#########################################

"""
MultiREx Utilities Module

This module provides utility functions for the MultiREx library, primarily focused on
downloading and managing external data files needed for spectrum generation.

The main functions in this module are:
    - get_stellar_phoenix: Downloads Phoenix stellar spectra models
    - get_gases: Downloads opacity database for atmospheric gases
    - list_gases: Lists available gases in the opacity database
"""

import numpy as np
import gdown
import os
import zipfile
from taurex.cache import OpacityCache

def get_stellar_phoenix(path=""):
    """Download the Phoenix stellar spectra from the Google Drive link and
    extract the content to the specified path.
    
    This function automates the download and extraction of Phoenix stellar model files,
    which are used for more accurate stellar spectrum modeling compared to blackbody models.
    If the Phoenix directory already exists at the specified path, no download occurs.
    
    Args:
        path (str, optional): 
            Directory path where the Phoenix folder will be created
            and model files will be downloaded. If empty string, uses current directory.
            Defaults to "".
    
    Returns:
        str: Path to the Phoenix directory containing the stellar model files.
    
    Note:
        This function requires an internet connection for the initial download.
        The Phoenix models are approximately 2GB in size.
    """


    phoenix_path = os.path.join(path, 'Phoenix')
    # ZIP file URL
    url = 'https://drive.google.com/uc?id=1fgKjDu9H26y5WMwRZaMCuSpHhx8zc0pR'
    # Local ZIP file name
    zip_path = os.path.join(path, 'Phoenix.zip')

    # Check if the Phoenix directory already exists
    if not os.path.exists(phoenix_path):
        
        if path == "":
            print("The path where the Phoenix stellar spectra will be downloaded is : ",
              "current directory")
        else:
            print("The path where the Phoenix stellar spectra will be downloaded is: ",
              path)
        
        # Download the ZIP file
        gdown.download(url, zip_path, quiet=False)

        # Unzip the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)

        # Delete the ZIP file after extraction
        os.remove(zip_path)
    else:
        print("The directory to Phoenix already exists in the specified path: ",
              path if path != "" else "current directory")
    return phoenix_path

def get_gases(path=""):   
    """Download the opacity database from the Google Drive link and 
    extract the content to the specified path.
    
    This function automates the download and extraction of opacity data files for
    atmospheric gases, which are required for spectrum generation. The opacity data
    is used by TauREx to calculate the absorption of light by different gases in
    the atmosphere. If the opacity database already exists at the specified path,
    no download occurs.
    
    Args:
        path (str, optional): Directory path where the opacity database will be
        downloaded and extracted. If empty string, uses current directory.
        Defaults to "".
    
    Note:
        This function requires an internet connection for the initial download.
        After downloading, the opacity path is automatically set in the TauREx
        OpacityCache for immediate use.
        
        The opacity database is approximately 3GB in size.
    """
    
    
    # Define the directory path where the content will be extracted
    molecule_path = os.path.join(path,'opacidades-todas')
    if os.path.exists(molecule_path):
        print("The directory to the opacity database already exists in the specified path: ",
              path if path != "" else "current directory")
    else:
        # URL of the ZIP file to download
        url = 'https://drive.google.com/uc?id=1z7R0hD1IBuYo-nnl7dpE_Ls2337a0uv6'
        # Local ZIP file name
        zip_path = path+"/opacidades-todas.zip"

        # Check if the directory already exists
        if not os.path.exists(molecule_path):
            
            if path == "":
                print("The path where the opacity database will be downloaded is : ",
                "current directory")
            else:
                print("The path where the opacity database will be downloaded is: ",
                path)
            
            # Download the ZIP file
            gdown.download(url, zip_path, quiet=False)

            # Unzip the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path)

            # Delete the ZIP file after extraction
            os.remove(zip_path)
        else:
            print("The directory to the opacity database already exists in the specified path: ",
                path if path != "" else "current directory")
            
    OpacityCache().clear_cache()
    xsec_path=molecule_path
    OpacityCache().set_opacity_path(xsec_path)
    
def list_gases():
    """List all available gases in the opacity database.
    
    This function prints the names of all atmospheric gases available in the
    current opacity database. These gases can be used in the composition of
    an Atmosphere object.
    
    Returns:
        None: The list of available gases is printed to the console.
        
    Note:
        You must first download the opacity database using get_gases() before
        this function will show the complete list of available gases.
        
    Example:
        >>> import multirex.utils as Util
        >>> Util.get_gases()  # Download the opacity database
        >>> Util.list_gases()  # List available gases
    """
    print("Available gases in the database:")
    print(list(OpacityCache().find_list_of_molecules()))
