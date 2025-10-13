import numpy as np
import numpy.ma as ma
from astropy.io import ascii
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
from functions.seda_utils import convolve_spectrum

# this code is available on GitHub: github.com/BDNYC/Doublet_Quantifier
from doublet_quantifier.curvefit import *




def files_to_spec(files,
                  path_to_files,
                  wavelength_region = (1.1, 1.8)):
    """
    Reads spectral data files and organizes them into a dictionary.

    Parameters:
    files (list of str): List of filenames of the spectral data files to be read.
    path_to_files (str, optional): The directory path where the spectral data files are located.
                                   Default is '/Users/allybaldelli/Desktop/AMNH-stuff/quantifying_clouds_ally/diamondback-data/t1500'.
    wavelength_region (tuple of float, optional): The wavelength range (min, max) to extract from the spectra.
                                                  Default is (1.1, 1.8).

    Returns:
    dict: A dictionary where the keys are modified filenames and the values are numpy arrays with shape (3, length).
          The first row of each array contains the wavelength data, the second row contains the flux data, and the
          third row is initialized to ones.

    This function performs the following tasks:
    1. Initializes an empty dictionary to store the spectral data.
    2. Iterates over each file in the `files` list.
    3. Constructs a key for the dictionary by modifying the filename.
    4. Reads the spectral data from each file using the specified path and filename.
    5. Extracts the wavelength and flux data, masking the data outside the specified `wavelength_region`.
    6. Creates a numpy array to store the masked wavelength and flux data.
    7. Stores the array in the dictionary with the constructed key.
    8. Returns the dictionary containing all the processed spectral data.

    """
        
    data_dictionary = {} # empty dictionary for spectra
    print('creating dictionary of spectra')
    for file in tqdm(files):
        name = file[6:-22]+' ' + file[-22:-20]

        path = path_to_files + file
        data = ascii.read(path, guess=False, names=[
                        'wavelength', 'flux'], data_start=3, delimiter='	')
        region = ma.masked_inside(data['wavelength'].data, wavelength_region[0], wavelength_region[1]).mask
        length = region.sum()
        data_array = np.ones((3, length))
        data_array[0, :] = data[region]['wavelength'].data
        data_array[1, :] = data[region]['flux'].data

        data_dictionary[name] = data_array.copy()

    del data
    return data_dictionary

def spec_to_parameter(data_dictionary, resolution, show_plots = False,
                      continuum_region = [1.15, 1.19], absorption_region = [1.165, 1.183]):
    """
    Convolves spectral data and fits pseudo-Voigt profiles to the spectra, returning a dictionary of convolved data
    and a DataFrame of fitted parameters.

    Parameters:
    data_dictionary (dict): Dictionary containing spectral data arrays. The keys are source names, and the values are
                            numpy arrays with shape (3, length), where the first row is wavelength data and the second
                            row is flux data.
    resolution (float): resolution calculated at the center of the absorption_region.
    show_plots (bool, optional): If True, plots the fitting results for each spectrum. Default is False.
    continuum_region (list of float, optional): The wavelength range [min, max] to be used for continuum fitting.
                                                Default is [1.15, 1.19].
    absorption_region (list of float, optional): The wavelength range [min, max] to be used for absorption fitting.
                                                 Default is [1.165, 1.183].
   

    Returns:
    tuple: A tuple containing:
           - df (pandas.DataFrame): DataFrame of fitted parameters with columns ['nu1', 'nu2', 'A1', 'A2', 'FWHM1', 'FWHM2', 'μ1', 'μ2', 'gravity', 'logg', 'clouds'].
           - convolve_data_dict (dict): Dictionary of convolved spectral data, with the same keys as `data_dictionary`.

    This function performs the following tasks:
    1. Initializes an empty dictionary to store the convolved spectral data.
    2. Creates a Gaussian kernel for convolution.
    3. Iterates over each spectrum in `data_dictionary`:
        a. Convolves the flux data with the Gaussian kernel.
        b. Stores the convolved data in `convolve_data_dict`.
        c. Fits the convolved spectrum using a pseudo-Voigt profile and extracts parameters.
        d. Optionally plots the fitting results.
    4. Creates a DataFrame to store the fitted parameters.
    5. Extracts gravity and cloud data from the keys of `data_dictionary` and adds them to the DataFrame.
    6. Returns the DataFrame of parameters and the dictionary of convolved data.

    Example usage:
    ```
    data_dictionary = {
        'source1': np.array([...]),
        'source2': np.array([...])
    }
    df, convolve_data_dict = spec_to_parameter(data_dictionary, resolution, show_plots=True,
                                               continuum_region=[1.14, 1.18], absorption_region=[1.165, 1.183],
                                               Gaussian_stddev=50)
    ```

    Note:
    - The function uses `Gaussian1DKernel` from `astropy.convolution` for convolution.
    - The function uses `fit_two_curves` to fit the pseudo-Voigt profiles. This function is assumed to be defined elsewhere.
    - The function uses `numpy` and `pandas` for data manipulation.
    - The function uses `tqdm` for displaying a progress bar while processing the spectra.
    """
 
    param_list = np.zeros((len(data_dictionary), 8))
    
    # empty dictionary for the data to be convolved 
    convolve_data_dict = {}
    
    
    for i, source in enumerate(data_dictionary):

        lam_R = sum(absorption_region)/2 # wavelength reference to estimate the spectral resolution of the input spectrum
        convolve_dict = convolve_spectrum(data_dictionary[source][0, :], data_dictionary[source][1, :], lam_R, resolution)

        length = len(convolve_dict['wl_conv'])
        convolve_data_array = np.ones((3, length))
        convolve_data_array[0, :] = convolve_dict['wl_conv']
        convolve_data_array[1, :] = convolve_dict['flux_conv']

        convolve_data_dict[source] = convolve_data_array

        cont_parameters, params_p, params_sd_p = fit_two_curves(convolve_data_array, continuum_region, absorption_region, 
                                                                function='pseudo-voigt',
                                                                show=show_plots, bin_size=200)
        param_list[i, :] = params_p

        if show_plots:
            plt.title("Fitting results for g" + source)
            plt.show()


    # creating DataFrame of parameters
    df = pd.DataFrame(param_list, columns=[
                'nu1', 'nu2', 'A1', 'A2', 'FWHM1', 'FWHM2', 'μ1', 'μ2'])
    df['name'] = list(data_dictionary.keys())

    # adding gravity of the data to df
    gravity = [int(i.split()[0]) for i in data_dictionary.keys()]
    df['gravity'] = gravity
    df['logg'] = np.log10(df.gravity * 100)
    
    # adding f_sed (clouds) to df
    clouds = [i.split()[1] for i in data_dictionary.keys()]
    for i, cloud in enumerate(clouds):
        try:
            clouds[i] = int(cloud[-1])
        except:
            clouds[i] = 10

    df['clouds'] = clouds

    return df, convolve_data_dict