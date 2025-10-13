import numpy as np
import os, sys, glob, platform, configparser, ast, tempfile
from abc import ABC
from pathlib import Path
import matplotlib.pyplot as plt

import os.path
from importlib.resources import files
import chebgreen

# Greenlearning imports
import torch, scipy

# Initialize a parser for the settings.ini file
parser = configparser.ConfigParser(inline_comment_prefixes="#")

# Define the path to the chebGreen package
chebgreen_path = Path(str(files(chebgreen)))

# Read the settings.ini file
if os.path.isfile('settings.ini'):
    print(f'Loading settings from {os.getcwd()}/settings.ini.')
    parser.read('settings.ini')
else:
    print('Loading settings from the package.')
    print('To use a custom settings.ini file, please place it in the current working directory.')
    path = str(files(chebgreen) / "settings.ini")
    parser.read(path)

# Define a configuration class for handling precision settings
# tf.float64 doesn't work for Apple tf2
class Config(ABC):
    def __init__(self, precision):
        # Initialize the Config object with a precision setting
        self.precision = precision
    
    def __call__(self, package):
        # Define a dictionary mapping precision settings to numpy types
        config = {
            32: {np: np.float32, torch: torch.float},
            64: {np: np.float64, torch: torch.double},
        }[self.precision]
        # Return the numpy type corresponding to the precision setting
        return config[package]
 
# Create a Config object with the precision setting from the settings.ini file
config = Config(parser['GENERAL'].getint('precision'))

# Set the appropriate matlab path according to system
operatingSystem = platform.uname().system
if operatingSystem == "Darwin":
    # Search for the MATLAB application on MacOS
    pathList = glob.glob("/Applications/MATLAB*.app/bin/matlab")
    if pathList:
        # If found, set the MATLABPath to the first found path
        MATLABPath = pathList[0]
    else:
        # If not found, raise an error
        raise('Matlab not found')
elif operatingSystem == "Linux":
    # Search for the MATLAB application on Linux
    pathList = glob.glob("/usr/local/MATLAB/*/bin/matlab")
    if pathList:
        # If found, set the MATLABPath to the first found path
        MATLABPath = pathList[0]
    else:
        # If not found, raise an error
        raise('Matlab not found')
else:
    # If the operating system is not supported, raise an error
    raise('Operating system not supported. Please set MATLABPath manually.')

# Set the appropriate compute acceleration, default to CPU if nothing is available
if parser.has_option('GENERAL', 'device'):
    dev = parser['GENERAL']['device']
    assert(dev in ['cpu', 'mps'] or dev.startswith('cuda'), 'Invalid device for torch.')
    device = torch.device(dev)
else:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')


def print_settings(file = None):
    """
    Print the settings from the settings.ini file
    """
    if not file:
       print("ChebGreen settings:", file = file)
    for section in parser.sections():
        print(f"[{section}]", file = file)
        for item in parser.items(section):
            setting_name, value = item
            print(f"{setting_name} = {value}", file = file)
        print('', file = file)

    if not file:
        print(f"MATLAB Path = {MATLABPath}", file = file)
        print(f"PyTorch device = {device}", file = file)
    