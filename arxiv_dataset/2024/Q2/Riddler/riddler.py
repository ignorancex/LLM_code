## Import all required modules
import sys, os, shutil
#os.environ["OMP_NUM_THREADS"] = "1"
# Disable TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import glob

import pandas as pd
import numpy as np
import spectres
from extinction import ccm89, remove
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy.signal import savgol_filter as sf
from scipy import interpolate


from pickle import dump
from pickle import load

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras import backend as K


from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy import constants as csts
from astropy import units as u

import cProfile
from cProfile import Profile
from pstats import SortKey, Stats

from multiprocessing import Pool
from multiprocessing import set_start_method

import warnings

import typing

import ultranest
import ultranest.stepsampler




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
root_dir = "./"





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Arguments
## Name of the SN to be fit
sn_name = sys.argv[1]
## Which model will be used during fitting, currently either W7 or N100
model_type = sys.argv[2]

## Flag for whether to restart a previous run or not, 0 or 1
RESTART_FLAG = sys.argv[3]

## Index for which NN to use. Properties of NNs are given below
NN_index = int(sys.argv[4])




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Define parameters setting up neural networks
## Currently all neural networks trained using the following parameters
start = 3010
stop = 9000
bins = 1000
new_wl = np.logspace(np.log10(start), np.log10(stop), bins )

## Hyperparameters of the neural networks
## Only the top 6 are used and included here
if model_type == 'W7':
    NN_list = [
        [740414, 'W7', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 1e-3, 400, 2, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse', 45556, 2],
        [740415, 'W7', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 1e-3, 400, 3, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse',  5686, 3],
        [740416, 'W7', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 1e-3, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse',  3612, 4],
        [740434, 'W7', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 5e-4, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse',  5501, 16],
        [740436, 'W7', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 1e-4, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse', 33900, 17],
        [740437, 'W7', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 5e-5, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse', 49901, 18],
]
elif model_type == 'N100':
    NN_list = [
        [740448, 'N100', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 1e-3, 200, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse',  7983, 8],
        [740456, 'N100', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big',  5000, 1e-3, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse',  1399, 13],
        [740457, 'N100', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big',  2500, 1e-3, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse',   863, 14],
        [740462, 'N100', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 5e-4, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse',  6313, 16],
        [740463, 'N100', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 1e-4, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse', 31254, 17],
        [740464, 'N100', 3010, 9000, 1000, 9000, 'log',    'single_smoothed', 2, 25, 'StandardScaler', 'big', 10000, 5e-5, 400, 4, 'none',  0.00,  'leaky_relu', 'None', 'None', 0.0, 'Nadam', 'mse', 49342, 18],
]


NN = NN_list[NN_index]

## Parameters related to training data
## Many of these are holdovers from older testing runs and are not really relevant
model_type = NN[1]
start = NN[2]
stop = NN[3]
bins = NN[4]
max_wl = NN[5]
spacing_type = NN[6]
data_processing = NN[7]
order = NN[8]
window = NN[9]
scaler_type = NN[10]
dataset = NN[11]


## Parameters related to the neural network
## Hyperparameters of the specific neural network that will be used during fitting
BATCH_SIZE = NN[12]
LEARNING_RATE = NN[13]
NEURONS = NN[14]
LAYERS = NN[15]
BATCH_NORMALISATION = NN[16]
DROPOUT = NN[17]
ACTIVATION = NN[18]
WEIGHT_INITIALIZER = NN[19]
WEIGHT_REGULARIZER = NN[20]
WEIGHT_REGULARIZER_FLOAT = NN[21]
OPTIMIZER_NAME = NN[22]
LOSS_TYPE = NN[23]

BEST_EPOCH = NN[24]

MODEL_INDEX = NN[25]

label = "start" + str(start) + "_stop" + str(stop) + "_bins" + str(bins) + "_maxwl" + str(max_wl) + "_" + spacing_type + "/" + data_processing + "_o" + str(order) + "_wl" + str(window) + "/NN_bs" + str(BATCH_SIZE) + "_lr" + "{:.1e}".format(LEARNING_RATE) + "_neuron" + str(NEURONS) + "_layer" + str(LAYERS) + "_bn" + BATCH_NORMALISATION + "_drop" + "{:.2f}".format(DROPOUT) + "_act" + ACTIVATION + "_init" + WEIGHT_INITIALIZER + "_regul" + WEIGHT_REGULARIZER + "_" + "{:.1e}".format(WEIGHT_REGULARIZER_FLOAT) + "_opt" + OPTIMIZER_NAME + "_" + LOSS_TYPE + "/" + scaler_type + "/"



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Function to predict spectra to be used in fitting -- vectorised for ultranest
def predict_spectra_vectorised(theta):
    ## Theta is a big list of variables, the number of variables depends on the number of spectra that are being fit
    ## Go through the theta list and format all the parameters so the neural networks can handle them. Neural networks expect 4 parameters for each spectrum
    ## Each set of input parameters covers N spectra, and there are M sets of input parameters. This reshapes the input parameters to be in the correct format for the neural network to handle all at once
    all_parameters = np.zeros((theta.shape[0],len(observed_spectra_jd),4))
    for i in np.arange(0, len(observed_spectra_jd)):
        all_parameters[:,i,0] = observed_spectra_jd[i] - theta[:,1]
        all_parameters[:,i,1] = theta[:,i*2+3]
        all_parameters[:,i,2] = theta[:,i*2+4]
        all_parameters[:,i,3] = theta[:,2] * 1e51

    ## Reshape to handle all spectra simultaneously
    all_parameters = all_parameters.reshape((all_parameters.shape[0] * all_parameters.shape[1], all_parameters.shape[-1]))

    ## Log and transform all the parameters for generating spectra
    current_params_transformed = label_fit.transform( np.log10(all_parameters) )
    
    ## Predict spectra on the transformed parameters and convert back to flux space
    NN_predictions = trained_NN.predict_on_batch(current_params_transformed)
    NN_spectra = data_fit.inverse_transform(NN_predictions)
    NN_spectra = np.power(10, np.float64(NN_spectra))

    ## The NN accuracy depends on the time and velocity of the spectra
    ## Use interpolators to get the accuracy for each set of parameters
    NN_accuracy = interpolate_values(NN_accuracy_interpolator, all_parameters[:,0], all_parameters[:,2])
    NN_spectra_err = NN_spectra * NN_accuracy.T
        
    ## Reshape the spectra and errors to be in the same format as the input, i.e. grouping the N spectra together for a given set of input parameters
    NN_spectra = NN_spectra.reshape( (theta.shape[0],len(observed_spectra_jd),bins) )
    NN_spectra_err = NN_spectra_err.reshape( (theta.shape[0],len(observed_spectra_jd),bins) )
    
    return NN_spectra, NN_spectra_err
    
    



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Function to predict spectrum for a single set of input parameters -- not used during fitting and therefore not vectorised
def predict_spectra(theta):
    ## Theta is a set of inputs parameters covering multiple spectra. There are 4 required parameters for each spectrum, this is to ensure the correct format for the neural network
    ## Go through the theta list and format all the parameters so the neural networks can handle them
    all_parameters = []
    for i in np.arange(0, len(observed_spectra_jd)):
        time_since_explosion = observed_spectra_jd[i] - theta[1]
        ke = theta[2] * 1e51
        luminosity = theta[i*2+3]
        velocity = theta[i*2+4]
        spectrum_parameters = [time_since_explosion, luminosity, velocity, ke]
        all_parameters.append(spectrum_parameters)
        
    all_parameters = np.array(all_parameters)

    ## Log and transform all the parameters for generating spectra
    current_params_transformed = label_fit.transform( np.log10(all_parameters) )

    ## Predict spectra on the transformed parameters and convert back to flux space
    NN_predictions = trained_NN.predict_on_batch(current_params_transformed)
    NN_spectra = data_fit.inverse_transform(NN_predictions)
    NN_spectra = np.power(10, np.float64(NN_spectra))
    
    ## The NN accuracy depends on the time and velocity of the spectra
    ## Use interpolators to get the accuracy for each set of parameters
    NN_accuracy = interpolate_values(NN_accuracy_interpolator, all_parameters[:,0], all_parameters[:,2])
    NN_spectra_err = NN_spectra * NN_accuracy.T

    return NN_spectra, NN_spectra_err
    
    
    

    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Function to calculate the log likelihood to be used in fitting -- vectorised for ultranest
def log_likelihood_vectorised(theta):

    ## The first parameter is the log_f parameter, which is the scaling factor for the uncertainty
    ## This is a single parameter that applies to all spectra so need to reshape
    log_f = theta[:,0]
    log_f = log_f[...,np.newaxis]
    log_f = log_f[...,np.newaxis]

    ## Get the spectra and the error
    model_spectra, model_spectra_err = predict_spectra_vectorised(theta)

    ## Create an array to hold the likelihood. By default set to a very large negative number
    result = np.ones( (theta.shape[0]) ) * -1e100

    ## If any flux is outside this range, then reject. 
    ## Have some offset that depends on the flux to avoid numerical issues and plateaus in the likelihood
    mask = (model_spectra < -1e41) | (model_spectra > 1e41) | (model_spectra_err > 1e41)
    points_mask = np.sum(np.sum(mask, axis=1), axis=1) > 0
    result[points_mask]= -1e100 * np.abs(np.mean(np.mean(model_spectra, axis=1),axis=1))[points_mask]

    ## Get data
    yweight = data_yweight
    yerr = data_yerr
    y = data_y

    ## Get the total uncertainty
    ## Includes systematic uncertainty (log_f), model uncertainty (NN accuracy), and data uncertainty    
    sigma2 = ( np.power( model_spectra, 2.0) * np.exp(2 * log_f) ) + np.power( model_spectra_err, 2.0) + np.power( yerr, 2.0)

    ## For accepted points, calculate the likelihood
    ## Also include weighting for each wavelength
    result[~points_mask] = -0.5 * np.sum(np.sum( ( np.power(y - model_spectra, 2.0) / sigma2 + np.log(sigma2) + np.log(2.*np.pi) ) * yweight, axis=1 ), axis=1)[~points_mask]
    return result





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Function to transform the parameters to be used in fitting -- vectorised for ultranest
def prior_transform_vectorised(cube):
    params = cube.copy()

    ## Transform log_f
    params[:,0] = cube[:,0] * (np.log(1.) - np.log(1e-10)) + np.log(1e-10)

    ## Transform time since explosion
    ## First spectrum must be more than 5d after explosion, last spectrum must be less than 25d before explosion
    time_of_explosion_earliest = observed_spectra_jd[-1] - 25.
    time_of_explosion_latest = observed_spectra_jd[0] - 5.
    params[:,1] = cube[:,1] * (time_of_explosion_latest - time_of_explosion_earliest) + time_of_explosion_earliest

    ## Transform KE
    low = 1.0
    high = 1.8
    params[:,2] = cube[:,2] * (high - low) + low
    
    ## Transform luminosities and velocities
    for i in np.arange(len(observed_spectra_jd)):
        ## Get time since explosion for each spectrum, taking explosion epoch into account
        time_since_explosion = observed_spectra_jd[i] - params[:,1]

        ## Get the boundaries for luminosity and velocity, which depend on time since explosion
        luminosity_upper_bound = np.interp(time_since_explosion, luminosity_boundaries_xs, luminosity_boundaries_ys_max)
        luminosity_lower_bound = np.interp(time_since_explosion, luminosity_boundaries_xs, luminosity_boundaries_ys_min)
        velocity_upper_bound = np.interp(time_since_explosion, velocity_boundaries_xs, velocity_boundaries_ys_max)
        velocity_lower_bound = np.interp(time_since_explosion, velocity_boundaries_xs, velocity_boundaries_ys_min)
        
        ## Transform the luminosity, which is uniform between the boundaries
        params[:,i*2 + 3] = cube[:,i*2 + 3] * (luminosity_upper_bound - luminosity_lower_bound) + luminosity_lower_bound

        ## Transform the velocity, which is uniform between the boundaries for the first spectrum. Later spectra are uniform between the previous spectrum's velocity and the boundary
        if i == 0:
            params[:,i*2 + 4] = cube[:,i*2 + 4] * (velocity_upper_bound - velocity_lower_bound) + velocity_lower_bound
        else:
            params[:,i*2 + 4] = cube[:,i*2 + 4] * (params[:,(i-1)*2 + 4] - velocity_lower_bound) + velocity_lower_bound

    return params
    
    


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Function for ultranest fitting
def run_ultranest_fitting():

    # Set path where results will be saved
    output_path = root_dir + "Outputs/" + sn_name + "/" + model_type + "/" + label
    try:
        os.makedirs(output_path)
    except:
        ""
    ## Set up the ultranest sampler
    sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood_vectorised, prior_transform_vectorised, vectorized=True, log_dir=output_path, resume=RESTART_FLAG)
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps, generate_direction=ultranest.stepsampler.generate_mixture_random_direction,)
    
    ## Run sampler, print results, and plot
    result = sampler.run(show_status=True, min_num_live_points=1000)
    sampler.print_results()
    
    sampler.plot()
    
    return sampler




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Make the theta more easily readable
def format_theta(theta):

    all_parameters = []
    
    ## For each spectrum, go through and get the 4 parameters
    for i in np.arange(0, len(observed_spectra_jd)):
        time_since_explosion = observed_spectra_jd[i] - theta[1]
        ke = theta[2] * 1e51
        luminosity = theta[i*2+3]
        velocity = theta[i*2+4]
        spectrum_parameters = [time_since_explosion, luminosity, velocity, ke]
        all_parameters.append(spectrum_parameters)

    all_parameters = np.array(all_parameters)
    
    return all_parameters





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Create interpolators to calculate the NN accuracy as a function of time and velocity
def create_interpolator(x_values, y_values, z_values):
    interpolators = []
    for i in range(z_values.shape[2]):
        # Create an interpolator for each set of z values
        interpolator = interpolate.RectBivariateSpline(x_values, y_values, z_values[:,:,i], kx=3, ky=3)
        interpolators.append(interpolator)
    return interpolators

def interpolate_values(interpolators, x, y):
    # Use the interpolators to get the interpolated values
    return np.array([interpolator(x, y, grid=False) for interpolator in interpolators])




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Get a list of the observed spectra. Should all be in the input folder
observed_spectra = sorted(glob.glob(root_dir + "Inputs/" + sn_name + "/*.dat"))
# Load up a csv file containing the dates of each spectrum and the properties of the SN
observed_spectra_properties = pd.read_csv(root_dir + "Inputs/" + sn_name + "/properties.csv")

## Get the redshift, distance modulus, and extinction of the SN
observed_spectra_redshift = observed_spectra_properties['Redshift'].values[0]
observed_spectra_distance_modulus = observed_spectra_properties['Distance modulus'].values[0]
observed_spectra_Av = observed_spectra_properties['Av'].values[0]

## Try to convert distance modulus to distance. Could fail if distance modulus is not given or is NaN, in which case use 1.
try:
    distance = (np.power(10, (observed_spectra_distance_modulus + 5.) / 5.) * u.pc).to('cm')
    distance_corr = 4. * np.pi * np.power(distance, 2)
except:
    distance_corr = 1.


obs_spectra_list = []
obs_spectra_err_list = []
obs_spectra_weight_list = []
i = 0
# For each spectrum, put into restframe, correct for distance and reddening, bin to same resolution as models, and smooth
for observed_spectrum in observed_spectra:

    ## Load up the observed spectrum
    ## Spectrum should be in the format of wavelength, flux, flux error, and weight
    spectrum = pd.read_csv(observed_spectrum, delim_whitespace=True, header=None, names=("Wave", "Flux", "Flux_err", "Weight"))
    ## Convert to restframe
    spectrum['RestWave'] = spectrum['Wave'] / (1.+observed_spectra_redshift)
    
    ## Correct for distance and reddening
    spectrum['Lum'] = spectrum['Flux'] * distance_corr
    spectrum['Lum_err'] = spectrum['Flux_err'] * distance_corr
    spectrum['Lum_extcorr'] = remove(ccm89(spectrum['Wave'].values, observed_spectra_Av, 3.1), spectrum['Lum'].values)
    spectrum['Lum_err_extcorr'] = remove(ccm89(spectrum['Wave'].values, observed_spectra_Av, 3.1), spectrum['Lum_err'].values)
    
    ## Bin to same resolution as models
    binned_flux = spectres.spectres(new_wl, spectrum['RestWave'].values, spectrum['Lum_extcorr'].values, fill = 0)
    binned_flux_err = spectres.spectres(new_wl, spectrum['RestWave'].values, spectrum['Lum_err_extcorr'].values, fill = 0)

    ## Smooth the spectrum
    smoothed_flux = sf(binned_flux, window_length=window, polyorder=order)
    smoothed_flux_err = sf(binned_flux_err, window_length=window, polyorder=order)

    ## Get the weights at the binned resolution of the spectrum
    interpolated_weights = np.interp(new_wl, spectrum['RestWave'].values, spectrum['Weight'].values)
    
    ## Add spectra, errors, and weights to lists
    obs_spectra_list.append( smoothed_flux )
    obs_spectra_err_list.append( smoothed_flux_err )
    obs_spectra_weight_list.append( interpolated_weights )
        
    i = i + 1

## Convert to numpy arrays
obs_spectra_list = np.array(obs_spectra_list)
obs_spectra_err_list = np.array(obs_spectra_err_list)
obs_spectra_weight_list = np.array(obs_spectra_weight_list)

## Get dates of spectra
observed_spectra_jd = observed_spectra_properties['Time'].values

## Get data
data_x = new_wl
data_y = np.float64(obs_spectra_list)
data_yerr = np.float64(obs_spectra_err_list)
data_yweight = np.float64(obs_spectra_weight_list)

## Set parameter names, which depends on the number of spectra
param_names = ['log_f', 'time_expolosion_jd', 'ke']
for i in np.arange(len(observed_spectra_jd)):
    param_names.append('luminosity_spec' + str(i+1))
    param_names.append('velocity_spec' + str(i+1))

nsteps = 2 * len(param_names)





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Load up boundaries for the input parameters

luminosity_boundaries = pd.read_csv(root_dir + "Inputs/Data/luminosity_boundaries.csv", index_col=0)
velocity_boundaries = pd.read_csv(root_dir + "Inputs/Data/velocity_boundaries.csv", index_col=0)


luminosity_boundaries_xs = luminosity_boundaries['Times'].values
luminosity_boundaries_ys_max = luminosity_boundaries['max_luminosity'].values
luminosity_boundaries_ys_min = luminosity_boundaries['min_luminosity'].values

velocity_boundaries_xs = velocity_boundaries['Times'].values
velocity_boundaries_ys_max = velocity_boundaries['max_velocity'].values
velocity_boundaries_ys_min = velocity_boundaries['min_velocity'].values




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Load up stuff for the NN

## Get the correct transformation files
## This is a holdover from older testing runs and is not really relevant. All of the NNs use the same transformation, which is a StandardScaler
if scaler_type == "QuantileNormalScaler":
    scaler_filename = "_quantile_normal.pkl"
    scaler_prefix = "_quantile"
elif scaler_type == "StandardScaler":
    scaler_filename = "_standard.pkl"
    scaler_prefix = "_standard"
elif scaler_type == "MinMaxScaler":
    scaler_filename = "_minmax.pkl"
    scaler_prefix = "_minmax"

## Set the path to the training data
training_data_path = root_dir + "Inputs/Data/TrainingData/" + model_type + "/" + "start" + str(start) + "_stop" + str(stop) + "_bins" + str(bins) + "_" + spacing_type + "/" + data_processing + "_o" + str(order) + "_wl" + str(window) + "/" + scaler_type

## Load up the transformation files for labels (input parameters) and data (spectra)
with open(training_data_path + "/label_fit_logged" + scaler_filename, 'rb') as label_fit_file:
    label_fit = load(label_fit_file)
with open(training_data_path + "/data_fit_logged" + scaler_filename, 'rb') as data_fit_file:
    data_fit = load(data_fit_file)
    
## Set the path to the specific neural network used during fitting
path = root_dir + "NeuralNetworks/" + model_type + "/" + label + "/"
## Load up the neural network
trained_NN = tf.keras.models.load_model(path + "/cp-" + str(BEST_EPOCH).zfill(5) + ".ckpt", compile=False)

## Load up the accuracy of the neural network and create an interpolator
## The accuracy is defined for a series of time and velocity bins
accuracy_npzfile = np.load(path + "binned_fe.npz")

accuracy_time_midpoint     = accuracy_npzfile['x']
accuracy_velocity_midpoint = accuracy_npzfile['y']
accuracy_binned_mean_fe    = accuracy_npzfile['mean_fe']
accuracy_binned_max_fe     = accuracy_npzfile['max_fe']

NN_accuracy_interpolator = create_interpolator(accuracy_time_midpoint, accuracy_velocity_midpoint, accuracy_binned_mean_fe)






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Function to plot the best fitting results
def plot_results(theta):

    # Set path where results are stored
    output_path = root_dir + "Outputs/" + sn_name + "/" + model_type + "/" + label

    ## Get the spectra and errrors
    spectra, spectra_err = predict_spectra(theta)
    
    # Generate plot of predicted spectra vs. observed spectra
    plt.figure(num=None, figsize=(6, 10), dpi=300, facecolor='w', edgecolor='k')

    grid = plt.GridSpec(len(observed_spectra_jd), 1, wspace=0.2, hspace=0)
    ax = []

    for i in np.arange(len(obs_spectra_list)):

        ax.append(plt.subplot(grid[i]))

        ax[i].minorticks_on()
        ax[i].tick_params(axis="both", which="major", direction="in", top='On', right='On', length=8)
        ax[i].tick_params(axis="both", which="minor", direction="in", top='On', right='On', length=4)

        ax[i].set_xlim(3005, 9000)


        ax[i].plot(new_wl, obs_spectra_list[i], color='k', label='Observed')
        ax[i].plot(new_wl, spectra[i], color='r', label='NN')

    ax[-1].legend()

    plt.savefig(output_path + '/predictions.pdf', bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

    return






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Main function
def main():

    print("Run fitting")
    sampler = run_ultranest_fitting()
    print("Finished fitting")

    ## Get the best fit parameters
    theta = np.array(sampler.results['posterior']['mean'])

    ## Plot the best fit results
    plot_results(theta)    


    
    

    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Run main function
if __name__ == "__main__":
    ## Old versions of the code used to use the multiprocessing module, which required setting the start method to spawn
    #set_start_method("spawn")
    main()
