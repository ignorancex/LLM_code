import numpy as np
import pandas as pd
import src.observational_data.firas_data as firas_data
import src.utilities.constants as const
import src.utilities.settings as settings
import src.nii_intensities.spiral_arm_model as sam
import src.utilities.utilities as ut
import matplotlib.pyplot as plt
import src.nii_intensities.gum_cygnus as gum_cygnus
from scipy.interpolate import griddata
import datetime
import os
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

# Define the ranges to remove
RANGES_TO_REMOVE = [
    (280 + 2.5, 280 - 7.5),
    (5 + 2.5, 5 - 2.5),
    (330 + 7.5, 330 - 2.5),
    (47.5, 42.5),
    (350 + 2.5, 350 - 2.5)
]

RANGES_TO_REMOVE_BIN_CENTERS = [
    280, 275, 5, 330, 335, 45, 350
]


def slice_numpy_array(array, start, end):
    """ Function to remove a specified slice from a numpy array based on start and end indices.

    Args:
        array (np.array): The array from which to remove the slice
        start (int): The start index of the slice to remove
        end (int): The end index of the slice to remove
    Returns:
        np.array: The array with the specified slice removed
    """
    if start > end:
        raise ValueError("The start index must be smaller than the end index")
    if start < 0 or end >= len(array):  # Prevent out-of-bounds access
        raise ValueError("The start and end indices must be within the length of the array")
    # Use np.delete to remove elements from start to end (inclusive of end)
    return np.delete(array, slice(start, end + 1))


def filter_chi_squared(observational_data, observational_data_variance, modelled_data):
    """ Function to filter the observational and modelled data to remove ranges which makes the algorithm overfit the data. The ranges to remove are specified in the function.
    
    Args:
        observational_data (np.array): The observational data
        observational_data_variance (np.array): The variance of the observational data
        modelled_data (np.array): The modelled data
    Returns:
        observational_data (np.array): The filtered observational data
        observational_data_variance (np.array): The filtered variance of the observational data
        modelled_data (np.array): The filtered modelled data    
    """

    dl = 0.2    # increments in dl (degrees):
    # np.array with values for galactic longitude l in degrees.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.concatenate((l1, l2)) # same way longitudes is created in spiral_arm_model.py
    # Start with a mask of True for all elements, indicating they are to be kept
    mask = np.ones(len(longitudes), dtype=bool)

    # Update the mask to False for the ranges to be removed
    for long_left, long_right in RANGES_TO_REMOVE:
        left_index = np.abs(longitudes - long_left).argmin()
        right_index = np.abs(longitudes - long_right).argmin()
        if left_index > right_index:
            left_index, right_index = right_index, left_index  # Ensure left is less than right
        mask[left_index:right_index + 1] = False

    # Apply the mask to all data arrays
    observational_data = observational_data[mask]
    observational_data_variance = observational_data_variance[mask]
    modelled_data = modelled_data[mask]

    return observational_data, observational_data_variance, modelled_data


def chi_squared(observational_data, observational_data_variance, modelled_data):
    """ Function to calculate the chi-squared value of the modelled data compared to the observational data

    Args:
        observational_data (np.array): The observational data
        observational_data_variance (np.array): The variance of the observational data
        modelled_data (np.array): The modelled data
    Returns:
        chi_squared (float): The chi-squared value of the modelled data compared to the observational data"""
    if len(observational_data) != len(modelled_data) != len(observational_data_variance):
        raise ValueError("The length of the observational data, observational data variance and modelled data must be the same. Lengths are, respectively: ", len(observational_data), len(observational_data_variance), len(modelled_data))
    # Filter the data
    observational_data, observational_data_variance, modelled_data = filter_chi_squared(observational_data, observational_data_variance, modelled_data)
    chi_squared = np.sum(((observational_data - modelled_data) ** 2) / observational_data_variance)
    return chi_squared


def load_firas_data():
    """ Function to load the FIRAS data and variance, and expand the data to match the modelled data to facilitate the chi-squared calculation 

    Returns:
        expanded_firas_intensity (np.array): The FIRAS data expanded to match the modelled data
        expanded_firas_variance (np.array): The FIRAS variance expanded to match the modelled data
    """
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    firas_variance = (line_flux_error / 2) ** 2
    intensities_modelled = load_modelled_data()
    # the length of the firas data and modelled data differ. We need to expand the firas data to match the modelled data
    longitude_values = np.linspace(0, 360, len(intensities_modelled)) # create a list of longitudes for which we have modelled data
    binned_longitudes, bin_edges = np.histogram(longitude_values, bins=bin_edges_line_flux) # bin the longitudes in the same way as the firas data
    expanded_firas_intensity = np.repeat(line_flux, repeats=binned_longitudes) # expand the firas data to match the modelled data by repeating the values a number of times determined by the binning of longitude values
    expanded_firas_variance = np.repeat(firas_variance, repeats=binned_longitudes) # expand the firas variance to match the modelled data by repeating the values a number of times determined by the binning of longitude values
    return expanded_firas_intensity, expanded_firas_variance


def load_modelled_data(filename_arm_intensities='intensities_per_arm_b_max_5.npy'):
    """ Function to load all modelled components of the total NII intensity

    Returns:
        intensities_modelled (1D np.array): The modelled NII intensity data
    """
    intensities_per_arm = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/{filename_arm_intensities}') 
    intensities_modelled = np.sum(intensities_per_arm[:4], axis=0) # sum the intensities of the four spiral arms
    if settings.add_local_arm == True: # add the local arm contribution
        try: 
            intensities_modelled += intensities_per_arm[4]
        except: 
            logging.warning("The local arm has not been added to the modelled data. You can generate it in spiral_arm_model.py")
    if settings.add_gum_cygnus == True: # add the contribution from the nearby OBA
        try:
            gum = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_gum.npy')
            cygnus = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_cygnus.npy')
            gum_cygnus = gum + cygnus
            intensities_modelled += gum_cygnus
        except:
            logging.warning("The Gum Nebula and Cygnus Loop contributions have not been added to the modelled data. You can generate them in gum_cygnus.py")
    return intensities_modelled


def interpolate_density_one_arm(h_spiral_arm, arm_angle, pitch_angle, transverse_distances, transverse_densities_initial, arm_index, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, rho_min_spiral_arm=None, rho_max_spiral_arm=None):
    """ Integrates the densities of a single spiral arm over the entire galactic plane. The returned density is in units of kpc^-2. 

    Args:
        h_spiral_arm (float): The scale height of the spiral arm
        arm_angle (float): The angle of the spiral arm in radians
        pitch_angle (float): The pitch angle of the spiral arm in
        transverse_distances (np.array): The transverse distances of the spiral arm
        transverse_densities_initial (np.array): The transverse densities of the spiral arm
        arm_index (int): The index of the spiral arm
        rho_min_sagittarius (float): The minimum distance from the galactic center to the beginning of the spiral arm
        rho_max_sagittarius (float): The maximum distance from the galactic center to the beginning of the spiral arm
        sigma_devoid (float): The dispersion of the devoid region of the Sagittarius-Carina spiral arm

    Returns:
        None. The interpolated densities are saved to a file
    """
    x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/x_grid.npy')
    y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/y_grid.npy')
    num_grid_subdivisions = settings.num_grid_subdivisions
    if num_grid_subdivisions < 1:
        raise ValueError("num_grid_subdivisions must be larger than 0")
    
    # generate the spiral arm medians
    if rho_min_spiral_arm is None or rho_max_spiral_arm is None:
        rho_min_spiral_arm = const.rho_min_spiral_arm[arm_index]
        rho_max_spiral_arm = const.rho_max_spiral_arm[arm_index]
    theta, rho = sam.spiral_arm_medians(arm_angle, pitch_angle, rho_min_spiral_arm, rho_max_spiral_arm)
    # generate the spiral arm points
    x, y = sam.generate_spiral_arm_coordinates(rho, transverse_distances, theta, pitch_angle)
    # generate the spiral arm densities
    density_spiral_arm = sam.generate_spiral_arm_densities(rho, transverse_densities_initial, h_spiral_arm, arm_index=arm_index, transverse_distances=transverse_distances, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid) #####
    for sub_grid in range(num_grid_subdivisions):
        if sub_grid == num_grid_subdivisions - 1:
            x_grid_sub = x_grid[sub_grid * int(len(x_grid) / num_grid_subdivisions):]
            y_grid_sub = y_grid[sub_grid * int(len(y_grid) / num_grid_subdivisions):]
        else:
            x_grid_sub = x_grid[sub_grid * int(len(x_grid) / num_grid_subdivisions): (sub_grid + 1) * int(len(x_grid) / num_grid_subdivisions)]
            y_grid_sub = y_grid[sub_grid * int(len(y_grid) / num_grid_subdivisions): (sub_grid + 1) * int(len(y_grid) / num_grid_subdivisions)]
        # calculate interpolated density for the spiral arm
        interpolated_arm = griddata((x, y), density_spiral_arm, (x_grid_sub, y_grid_sub), method='cubic', fill_value=0)
        interpolated_arm[interpolated_arm < 0] = 0 # set all negative values to 0
        np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{arm_index}_{sub_grid}.npy', interpolated_arm)
    return      


def calc_effective_area_one_arm(h=const.h_spiral_arm, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, arm_index=None, rho_min_spiral_arm=None, rho_max_spiral_arm=None):
    """ Function to calculate the effective area for one single spiral arm. The density of the spiral arm is integrated over the entire galactic plane.
    The returned effective area is in units of kpc^2 and updates the correspodning entry in the effective_area_per_spiral_arm.npy file

    Args:
        h (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        arm_index (int): The index of the spiral arm. If not provided, the function will raise a ValueError

    Returns:
        float: Effective area for the spiral arm.
    """
    if arm_index is None:
        raise ValueError("The arm index must be provided. Exiting...")
    
    filepath = f'{const.FOLDER_GALAXY_DATA}/effective_area_per_spiral_arm.npy'
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    d_x = 70 / 3000 # distance between each interpolated point in the x direction. 70 kpc is the diameter of the Milky Way, 3000 is the number of points
    d_y = 70 / 3000 # distance between each interpolated point in the y direction. 70 kpc is the diameter of the Milky Way, 3000 is the number of points
    grid_x, grid_y = np.mgrid[-35:35:3000j, -35:35:3000j]
    # generate the spiral arm medians
    if rho_min_spiral_arm is None or rho_max_spiral_arm is None:
        rho_min_spiral_arm = const.rho_min_spiral_arm[arm_index]
        rho_max_spiral_arm = const.rho_max_spiral_arm[arm_index]
    theta, rho = sam.spiral_arm_medians(arm_angles[arm_index], pitch_angles[arm_index], rho_min_spiral_arm, rho_max_spiral_arm)
    # generate the spiral arm points
    x, y = sam.generate_spiral_arm_coordinates(rho, transverse_distances, theta, pitch_angles[arm_index])
    # generate the spiral arm densities
    density_spiral_arm = sam.generate_spiral_arm_densities(rho, transverse_densities_initial, h, arm_index=arm_index, transverse_distances=transverse_distances, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
    # calculate interpolated density for the spiral arm
    interpolated_density = griddata((x, y), density_spiral_arm, (grid_x, grid_y), method='cubic', fill_value=0)
    interpolated_density[interpolated_density < 0] = 0 # set all negative values to 0
    # add the interpolated density to the total galactic density 
    try:
        effective_area = np.load(filepath)
    except:
        logging.warning("The effective area file does not exist. Creating a new one...")
        effective_area = sam.calc_effective_area_per_spiral_arm(h=const.h_spiral_arm, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, readfile_effective_area=True)
    effective_area[arm_index] = np.sum(interpolated_density) * d_x * d_y
    np.save(filepath, effective_area)
    return effective_area


def optimize_spiral_arm_start_angle(delta, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, fractional_contribution=const.fractional_contribution, h_spiral_arm=const.h_spiral_arm, sigma_arm=const.sigma_arm, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, theta_max_local = const.theta_max_local):
    """ Function to optimize the start angle of the spiral arms. The function will loop over all possible combinations of start angles for the spiral arms, as determined by parameter delta, and calculate the chi-squared value for each combination. 
    The start angles that give the lowest chi-squared value will be returned.
    
    Args:
        delta: float, the step size for the arm angles to be tested. Can either be an integer, or a float between 0 and 1
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        fractional_contribution (list, optional): Fractional contribution of each spiral arm. Defaults to fractional_contribution.
        h_spiral_arm (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        rho_min_sagittarius (float, optional): The minimum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_min_sagittarius.
        rho_max_sagittarius (float, optional): The maximum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_max_sagittarius.
        sigma_devoid (float, optional): The dispersion of the devoid region of the Sagittarius-Carina spiral arm. Defaults to sigma_devoid.
        theta_max_local (float, optional): The maximum angle of the local arm. Defaults to theta_max_local.

    Returns:
        np.array: The optimized start angles for the spiral arms
    """
    logging.info("Optimizing spiral arm start angle")
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"start_angle_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data() # load the Firas data
    initial_arm_angles = np.degrees(arm_angles.copy()) # keep track of the initial arm angles
    initial_arm_angles[:4] -= delta # keep track of the initial arm angles, subtract delta only for the four main spiral arms and not the local arm
    arm_angles = np.degrees(arm_angles.copy()) # angles used for the optimization
    arm_angles[:4] -= delta # subtract delta only for the four main spiral arms and not the local arm
    best_angles = np.degrees(arm_angles.copy()) # initialize the best angles to the initial angles
    # check if delta is an integer or a float
    if delta == 0:
        logging.warning("delta is equal to zero. No optimization will be performed. Exiting...")
        return
    elif delta >= 1:
        scale = 1
    else: 
        scale = 10
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    num_angles_to_sample = int(delta * scale * 2 + 1) # multiply by 2 and add 1 to sample angles in range existing_angles +- delta.
    chi_squared_min = np.inf
    # update the local arm with the new value for theta_max
    rho_max_local = const.rho_min_local * np.exp(np.tan(const.pitch_local) * (theta_max_local - const.theta_start_local))
    interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=np.radians(arm_angles[4]), pitch_angle=pitch_angles[4], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
    calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=np.radians(arm_angles), pitch_angles=pitch_angles, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
    with open(filename_log, 'w') as file:
        # i = Norma-Cygnus, j = Perseus, k = Sagittarius-Carina, l = Scutum-Crux
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, rho_min_sagittarius, rho_max_sagittarius, sigma_devoid, theta_max_local, Chi-squared\n')
        for i in range(num_angles_to_sample): 
            arm_angles[0] = initial_arm_angles[0] + i / scale
            interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=np.radians(arm_angles[0]), pitch_angle=pitch_angles[0], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=0)
            calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=np.radians(arm_angles), pitch_angles=pitch_angles, arm_index=0)
            for j in range(num_angles_to_sample): 
                arm_angles[1] = initial_arm_angles[1] + j / scale
                interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=np.radians(arm_angles[1]), pitch_angle=pitch_angles[1], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=1)
                calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=np.radians(arm_angles), pitch_angles=pitch_angles, arm_index=1)
                for k in range(num_angles_to_sample):
                    # Sagittarius-Carina arm. Remember to add the updated parameters for the devoid region
                    arm_angles[2] = initial_arm_angles[2] + k / scale
                    interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=np.radians(arm_angles[2]), pitch_angle=pitch_angles[2], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
                    calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=np.radians(arm_angles), pitch_angles=pitch_angles, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
                    for l in range(num_angles_to_sample): 
                        arm_angles[3] = initial_arm_angles[3] + l / scale
                        interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=np.radians(arm_angles[3]), pitch_angle=pitch_angles[3], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=3)
                        calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=np.radians(arm_angles), pitch_angles=pitch_angles, arm_index=3)
                        sam.calc_modelled_intensity(readfile_effective_area=True, interpolate_all_arms=False, calc_gum_cyg=False, arm_angles=np.radians(arm_angles), pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm)
                        intensities_modelled = load_modelled_data()
                        chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
                        logging.info(f'Checking arm_angles: {arm_angles[:4]}. Pitch angles: {np.degrees(pitch_angles)}, fractional contribution: {fractional_contribution}, h: {h_spiral_arm}, sigma: {sigma_arm}, chi_squared: {chi_squared_val}')
                        file.write(f'{arm_angles[0]}, {arm_angles[1]}, {arm_angles[2]}, {arm_angles[3]}, {np.degrees(pitch_angles[0])}, {np.degrees(pitch_angles[1])}, {np.degrees(pitch_angles[2])}, {np.degrees(pitch_angles[3])}, {fractional_contribution[0]}, {fractional_contribution[1]}, {fractional_contribution[2]}, {fractional_contribution[3]}, {h_spiral_arm}, {sigma_arm}, {rho_min_sagittarius}, {rho_max_sagittarius}, {sigma_devoid}, {np.degrees(theta_max_local)}, {chi_squared_val}\n')
                        if chi_squared_val < chi_squared_min:
                            chi_squared_min = chi_squared_val
                            best_angles = arm_angles.copy()
    print('Best arm start angles:', best_angles)
    print('Best chi-squared:', chi_squared_min)
    return np.radians(best_angles)


def optimize_spiral_arm_pitch_angle(delta, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, fractional_contribution=const.fractional_contribution, h_spiral_arm=const.h_spiral_arm, sigma_arm=const.sigma_arm, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, theta_max_local = const.theta_max_local):
    """ Function to optimize the pitch angle of the spiral arms. The function will loop over all possible combinations of pitch angles for the spiral arms, as determined by parameter delta, and calculate the chi-squared value for each combination.
    The pitch angles that give the lowest chi-squared value will be returned.
    
    Args:
        delta: float, the step size for the arm angles to be tested. Can either be an integer, or a float between 0 and 1
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        fractional_contribution (list, optional): Fractional contribution of each spiral arm. Defaults to fractional_contribution.
        h_spiral_arm (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        rho_min_sagittarius (float, optional): The minimum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_min_sagittarius.
        rho_max_sagittarius (float, optional): The maximum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_max_sagittarius.
        sigma_devoid (float, optional): The dispersion of the devoid region of the Sagittarius-Carina spiral arm. Defaults to sigma_devoid.
        theta_max_local (float, optional): The maximum angle of the local arm. Defaults to theta_max_local.
    Returns:
        np.array: The optimized pitch angles for the spiral arms
    """
    logging.info("Optimizing pitch angles")
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"pitch_angle_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data() # load the Firas data
    initial_pitch_angles = np.degrees(pitch_angles.copy()) # keep track of the initial pitch angles
    initial_pitch_angles[:4] -= delta # subtract delta only for the four main spiral arms and not the local arm
    pitch_angles = np.degrees(pitch_angles.copy()) # angles used for the optimization
    pitch_angles[:4] -= delta # subtract delta only for the four main spiral arms and not the local arm
    best_pitch_angles = np.degrees(pitch_angles.copy()) # initialize the best angles to the initial angles
    # check if delta is an integer or a float. 
    if delta == 0:
        logging.warning("delta is equal to zero. No optimization will be performed. Exiting...")
        return
    elif delta >= 1:
        scale = 1
    else: 
        scale = 10 
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    num_angles_to_sample = int(delta * scale * 2 + 1) # multiply by 2 and add 1 to sample angles in range existing_angles +- delta.
    chi_squared_min = np.inf
    # update the local arm with the new value for theta_max
    rho_max_local = const.rho_min_local * np.exp(np.tan(const.pitch_local) * (theta_max_local - const.theta_start_local))
    interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[4], pitch_angle=np.radians(pitch_angles[4]), transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
    calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=np.radians(pitch_angles), arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
    with open(filename_log, 'w') as file:
        # i = Norma-Cygnus, j = Perseus, k = Sagittarius-Carina, l = Scutum-Crux
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, rho_min_sagittarius, rho_max_sagittarius, sigma_devoid, theta_max_local, Chi-squared \n')
        for i in range(num_angles_to_sample): 
            pitch_angles[0] = initial_pitch_angles[0] + i / scale # unit of degrees
            interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[0], pitch_angle=np.radians(pitch_angles[0]), transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=0)
            calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=np.radians(pitch_angles), arm_index=0)
            for j in range(num_angles_to_sample): 
                pitch_angles[1] = initial_pitch_angles[1] + j / scale # unit of degrees
                interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[1], pitch_angle=np.radians(pitch_angles[1]), transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=1)
                calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=np.radians(pitch_angles), arm_index=1)
                for k in range(num_angles_to_sample):
                    # Sagittarius-Carina arm. Remember to add the updated parameters for the devoid region
                    pitch_angles[2] = initial_pitch_angles[2] + k / scale # unit of degrees
                    interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[2], pitch_angle=np.radians(pitch_angles[2]), transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
                    calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=np.radians(pitch_angles), arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
                    for l in range(num_angles_to_sample): 
                        pitch_angles[3] = initial_pitch_angles[3] + l / scale # unit of degrees
                        interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[3], pitch_angle=np.radians(pitch_angles[3]), transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=3)
                        calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=np.radians(pitch_angles), arm_index=3)
                        sam.calc_modelled_intensity(readfile_effective_area=True, interpolate_all_arms=False, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=np.radians(pitch_angles), fractional_contribution=fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm)
                        intensities_modelled = load_modelled_data()
                        chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
                        logging.info(f'Checking pitch_angles: {pitch_angles[:4]}. Arm angles: {np.degrees(arm_angles)}, fractional contribution: {fractional_contribution}, h: {h_spiral_arm}, sigma: {sigma_arm}, chi_squared: {chi_squared_val}')
                        file.write(f'{np.degrees(arm_angles[0])}, {np.degrees(arm_angles[1])}, {np.degrees(arm_angles[2])}, {np.degrees(arm_angles[3])}, {pitch_angles[0]}, {pitch_angles[1]}, {pitch_angles[2]}, {pitch_angles[3]}, {fractional_contribution[0]}, {fractional_contribution[1]}, {fractional_contribution[2]}, {fractional_contribution[3]}, {h_spiral_arm}, {sigma_arm}, {rho_min_sagittarius}, {rho_max_sagittarius}, {sigma_devoid}, {np.degrees(theta_max_local)}, {chi_squared_val}\n')
                        if chi_squared_val < chi_squared_min:
                            chi_squared_min = chi_squared_val
                            best_pitch_angles = pitch_angles.copy()
    print('Best pitch angles:', best_pitch_angles)
    print('Best chi-squared:', chi_squared_min)
    return np.radians(best_pitch_angles)


def optimize_sigma_arm(arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, fractional_contribution=const.fractional_contribution, h_spiral_arm=const.h_spiral_arm, sigma_arm=const.sigma_arm, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, theta_max_local = const.theta_max_local):
    """ Function to optimize the sigma_arm parameter. The function will try different values for sigma in the range +- 0.5 around the existing value, and calculate the chi-squared value for each value.
    The sigma_arm value that gives the lowest chi-squared value will be returned.
    
    Args:
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        fractional_contribution (list, optional): Fractional contribution of each spiral arm. Defaults to fractional_contribution.
        h_spiral_arm (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        rho_min_sagittarius (float, optional): The minimum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_min_sagittarius.
        rho_max_sagittarius (float, optional): The maximum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_max_sagittarius.
        sigma_devoid (float, optional): The dispersion of the devoid region of the Sagittarius-Carina spiral arm. Defaults to sigma_devoid.
        theta_max_local (float, optional): The maximum angle of the local arm. Defaults to theta_max_local.       
    Returns:
        float: The optimized sigma_arm value
    """
    logging.info("Optimizing sigma_arm")
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"sigma_arm_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data() # load the Firas data
    best_sigma = sigma_arm # initialize the best sigma to the existing sigma
    sigmas_to_check = np.linspace(sigma_arm - 0.5, sigma_arm + 0.5, 11) # check sigma_arm in the range +- 0.5
    sigmas_to_check = sigmas_to_check[sigmas_to_check > 0] # keep only positive non zero values
    chi_squared_min = np.inf
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    with open(filename_log, 'w') as file:
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, rho_min_sagittarius, rho_max_sagittarius, sigma_devoid, theta_max_local, Chi-squared\n')
        for sigma in sigmas_to_check:
            logging.info(f'Checking sigma_arm: {sigma}')
            sam.calc_modelled_intensity(readfile_effective_area=False, interpolate_all_arms=True, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h=h_spiral_arm, sigma_arm=sigma)
            # remember to recalculate the sagittarius-carina arm with the updated values for the devoid region
            interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[2], pitch_angle=pitch_angles[2], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
            calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
            # update the local arm with the new value for theta_max
            rho_max_local = const.rho_min_local * np.exp(np.tan(const.pitch_local) * (theta_max_local - const.theta_start_local))
            interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[4], pitch_angle=pitch_angles[4], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
            calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
            intensities_modelled = load_modelled_data()
            chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
            file.write(f'{np.degrees(arm_angles[0])}, {np.degrees(arm_angles[1])}, {np.degrees(arm_angles[2])}, {np.degrees(arm_angles[3])}, {np.degrees(pitch_angles[0])}, {np.degrees(pitch_angles[1])}, {np.degrees(pitch_angles[2])}, {np.degrees(pitch_angles[3])}, {fractional_contribution[0]}, {fractional_contribution[1]}, {fractional_contribution[2]}, {fractional_contribution[3]}, {h_spiral_arm}, {sigma}, {rho_min_sagittarius}, {rho_max_sagittarius}, {sigma_devoid}, {np.degrees(theta_max_local)}, {chi_squared_val}\n')
            if chi_squared_val < chi_squared_min:
                chi_squared_min = chi_squared_val
                best_sigma = sigma
    print('Best sigma_arm:', best_sigma)
    print('Best chi-squared:', chi_squared_min)
    return best_sigma


def optimize_spiral_arm_h(arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, fractional_contribution=const.fractional_contribution, h_spiral_arm=const.h_spiral_arm, sigma_arm=const.sigma_arm, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, theta_max_local = const.theta_max_local):
    """ Function to optimize the h_spiral_arm parameter. The function will try different values for h_spiral_arm in the range +- 0.5 around the existing value, and calculate the chi-squared value for each value.
    The h_spiral_arm value that gives the lowest chi-squared value will be returned.

    Args:
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        fractional_contribution (list, optional): Fractional contribution of each spiral arm. Defaults to fractional_contribution.
        h_spiral_arm (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        rho_min_sagittarius (float, optional): The minimum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_min_sagittarius.
        rho_max_sagittarius (float, optional): The maximum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_max_sagittarius.
        sigma_devoid (float, optional): The dispersion of the devoid region of the Sagittarius-Carina spiral arm. Defaults to sigma_devoid.
        theta_max_local (float, optional): The maximum angle of the local arm. Defaults to theta_max_local.
    Returns:
        float: The optimized h_spiral_arm value
    """
    logging.info("Optimizing h_spiral_arm")
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"h_spiral_arm_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data()
    best_h_spiral_arm = h_spiral_arm
    h_spiral_arm_to_check = np.linspace(h_spiral_arm - 0.5, h_spiral_arm + 0.5, 11)
    chi_squared_min = np.inf
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    with open(filename_log, 'w') as file:
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, rho_min_sagittarius, rho_max_sagittarius, sigma_devoid, theta_max_local, Chi-squared\n')
        for h in h_spiral_arm_to_check:
            logging.info(f'Checking h_spiral_arm: {h}')
            sam.calc_modelled_intensity(readfile_effective_area=False, interpolate_all_arms=True, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h=h, sigma_arm=sigma_arm)
            # remember to recalculate the sagittarius-carina arm with the updated values for the devoid region
            interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[2], pitch_angle=pitch_angles[2], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
            calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
            # update the local arm with the new value for theta_max
            rho_max_local = const.rho_min_local * np.exp(np.tan(const.pitch_local) * (theta_max_local - const.theta_start_local))
            interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[4], pitch_angle=pitch_angles[4], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
            calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
            intensities_modelled = load_modelled_data()
            chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
            file.write(f'{np.degrees(arm_angles[0])}, {np.degrees(arm_angles[1])}, {np.degrees(arm_angles[2])}, {np.degrees(arm_angles[3])}, {np.degrees(pitch_angles[0])}, {np.degrees(pitch_angles[1])}, {np.degrees(pitch_angles[2])}, {np.degrees(pitch_angles[3])}, {fractional_contribution[0]}, {fractional_contribution[1]}, {fractional_contribution[2]}, {fractional_contribution[3]}, {h}, {sigma_arm}, {rho_min_sagittarius}, {rho_max_sagittarius}, {sigma_devoid}, {np.degrees(theta_max_local)} {chi_squared_val}\n')
            if chi_squared_val < chi_squared_min:
                chi_squared_min = chi_squared_val
                best_h_spiral_arm = h
    print('Best h_spiral_arm:', best_h_spiral_arm)
    print('Best chi-squared:', chi_squared_min)
    return best_h_spiral_arm


def optimize_fractional_contribution_four_spiral_arms(arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, fractional_contribution=const.fractional_contribution, h_spiral_arm=const.h_spiral_arm, sigma_arm=const.sigma_arm, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, theta_max_local = const.theta_max_local):
    """ Function to optimize the fractional contribution of the spiral arms. The function will try different values for the fractional contribution in the range +- 0.1 around the existing value, and calculate the chi-squared value for each value.
    The fractional contribution that gives the lowest chi-squared value will be returned.

    Args:install
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        fractional_contribution (list, optional): Fractional contribution of each spiral arm. Defaults to fractional_contribution.
        h_spiral_arm (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        rho_min_sagittarius (float, optional): The minimum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_min_sagittarius.
        rho_max_sagittarius (float, optional): The maximum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_max_sagittarius.
        sigma_devoid (float, optional): The dispersion of the devoid region of the Sagittarius-Carina spiral arm. Defaults to sigma_devoid.
        theta_max_local (float, optional): The maximum angle of the local arm. Defaults to theta_max_local.
    Returns:
        np.array: The optimized fractional contribution of the spiral arms
    """
    if np.sum(fractional_contribution[:4]) != 0.99: # shall be 0.99 because 0.01 percentage of the N II intensity is being attributed to the local arm
        logging.warning(f"The fractional contributions do not sum to 0.99. The sum is {np.sum(fractional_contribution[:4])}. Exiting...")
        return
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"fractional_four_spiral_arms_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data() # load the Firas data
    best_fractional_contribution = fractional_contribution.copy() # initialize the best fractional contribution to the existing one
    fractional_contributions_to_check = fractional_contribution.copy() # initialize the fractional contributions to check to the existing one
    increments = [+0.01, -0.01]
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    with open(filename_log, 'w') as file:
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, rho_min_sagittarius, rho_max_sagittarius, sigma_devoid, theta_max_local, Chi-squared\n')
        logging.info(f'Checking fractional_contributions_to_check: {fractional_contributions_to_check}')
        # Calculate the chi-squared value for the initial fractional contribution. Remember to recalculate the spiral arms with the updated parameters. 
        sam.calc_modelled_intensity(readfile_effective_area=False, interpolate_all_arms=True, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=best_fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm) # only checking fractional contribution - nothing to interpolate as fractional contribution is just some scale factor
        # remember to recalculate the sagittarius-carina arm with the updated values for the devoid region
        interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[2], pitch_angle=pitch_angles[2], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
        calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
        # update the local arm with the new value for theta_max
        rho_max_local = const.rho_min_local * np.exp(np.tan(const.pitch_local) * (theta_max_local - const.theta_start_local))
        interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[4], pitch_angle=pitch_angles[4], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
        calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
        # and as the sagittarius-carina arm and local arm is updated, remember to also update all intensities
        sam.calc_modelled_intensity(readfile_effective_area=True, interpolate_all_arms=False, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=best_fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm)
        intensities_modelled = load_modelled_data()
        chi_squared_min = chi_squared(firas_intensity, firas_variance, intensities_modelled)
        file.write(f'{np.degrees(arm_angles[0])}, {np.degrees(arm_angles[1])}, {np.degrees(arm_angles[2])}, {np.degrees(arm_angles[3])}, {np.degrees(pitch_angles[0])}, {np.degrees(pitch_angles[1])}, {np.degrees(pitch_angles[2])}, {np.degrees(pitch_angles[3])}, {fractional_contributions_to_check[0]}, {fractional_contributions_to_check[1]}, {fractional_contributions_to_check[2]}, {fractional_contributions_to_check[3]}, {h_spiral_arm}, {sigma_arm}, {np.degrees(theta_max_local)}, {chi_squared_min}\n')
        print(f'first run. fractional_contribution: {best_fractional_contribution}, chi-squared-min: {chi_squared_min}')
        for i in range(len(fractional_contributions_to_check[:4]) - 1):
            for increment in increments:
                fractional_contributions_to_check[i] += increment
                for k in range(len(fractional_contributions_to_check[i+1:-1])): # avoid the local arm
                    fractional_contributions_to_check[i + 1 + k] -= increment
                    sam.calc_modelled_intensity(readfile_effective_area=True, interpolate_all_arms=False, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contributions_to_check, h=h_spiral_arm, sigma_arm=sigma_arm) # only checking fractional contribution - nothing to interpolate as fractional contribution is just some scale factor
                    intensities_modelled = load_modelled_data()
                    chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
                    logging.info(f'Checking fractional_contribution: {fractional_contributions_to_check}, chi_squared: {chi_squared_val}')
                    file.write(f'{np.degrees(arm_angles[0])}, {np.degrees(arm_angles[1])}, {np.degrees(arm_angles[2])}, {np.degrees(arm_angles[3])}, {np.degrees(pitch_angles[0])}, {np.degrees(pitch_angles[1])}, {np.degrees(pitch_angles[2])}, {np.degrees(pitch_angles[3])}, {fractional_contributions_to_check[0]}, {fractional_contributions_to_check[1]}, {fractional_contributions_to_check[2]}, {fractional_contributions_to_check[3]}, {h_spiral_arm}, {sigma_arm}, {rho_min_sagittarius}, {rho_max_sagittarius}, {sigma_devoid}, {np.degrees(theta_max_local)}, {chi_squared_val}\n')
                    if chi_squared_val < chi_squared_min:
                        chi_squared_min = chi_squared_val
                        best_fractional_contribution = fractional_contributions_to_check.copy()
                    fractional_contributions_to_check[i + 1 + k] += increment # reset the value
                fractional_contributions_to_check[i] -= increment # reset the value
    print('Best fractional contribution:', best_fractional_contribution)
    print('Best chi-squared:', chi_squared_min)
    return best_fractional_contribution
            

def optimize_fractional_contribution_four_spiral_arms_total(arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, fractional_contribution_original=const.fractional_contribution, h_spiral_arm=const.h_spiral_arm, sigma_arm=const.sigma_arm, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, theta_max_local = const.theta_max_local):
    """ Function to optimize the fractional contribution of the spiral arms. The function will call the function optimize_fractional_contribution_four_spiral_arms several times, each time with the updated values for the fractional contribution, until either the values converge or the function has been called ten times
    
    Args:
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        fractional_contribution_original (list, optional): Fractional contribution of each spiral arm. Defaults to fractional_contribution_original.
        h_spiral_arm (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        rho_min_sagittarius (float, optional): The minimum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_min_sagittarius.
        rho_max_sagittarius (float, optional): The maximum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_max_sagittarius.
        sigma_devoid (float, optional): The dispersion of the devoid region of the Sagittarius-Carina spiral arm. Defaults to sigma_devoid.
        theta_max_local (float, optional): The maximum angle of the local arm. Defaults to theta_max_local.
    Returns:
        np.array: The optimized fractional contribution of the spiral arms
    """
    fractional_contribution = fractional_contribution_original.copy() # copy to prevent editing the original
    if np.sum(fractional_contribution[:4]) != 0.99: # shall be 0.99 because 0.01 percentage of the N II intensity is being attributed to the local arm
        logging.warning(f"The fractional contributions do not sum to 0.99. The sum is {np.sum(fractional_contribution[:4])}. Exiting...")
        return
    # Must remember to interpolate the arms the first time, as the parameters given to the function may not be the same as used to make the current intensity files on disk
    sam.calc_modelled_intensity(readfile_effective_area=False, interpolate_all_arms=True, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm) # only checking fractional contribution - nothing to interpolate as fractional contribution is just some scale factor
    best_fractional_contribution = optimize_fractional_contribution_four_spiral_arms(arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h_spiral_arm=h_spiral_arm, sigma_arm=sigma_arm)
    count = 0 # count the number of iterations
    while (not all(a == b for a, b in zip(best_fractional_contribution, fractional_contribution))) and count < 3:
        fractional_contribution = best_fractional_contribution.copy()
        best_fractional_contribution = optimize_fractional_contribution_four_spiral_arms(arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h_spiral_arm=h_spiral_arm, sigma_arm=sigma_arm)
        count += 1
    print(f'The very best fractional contribution: {best_fractional_contribution}. Obtained after {count} iterations')
    return best_fractional_contribution
    

@ut.timing_decorator
def optimize_saggitarus_devoid_region(arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, fractional_contribution=const.fractional_contribution, h_spiral_arm=const.h_spiral_arm, sigma_arm=const.sigma_arm, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, theta_max_local = const.theta_max_local):
    """ Function to optimize the density of the devoid region of Sagittarius. The function will loop over all possible combinations of rho_min, rho_max and sigma_devoid for the Sagittarius arm, and calculate the chi-squared value for each combination.
    The rho_min, rho_max and sigma_devoid values that give the lowest chi-squared value will be returned.
    
    Args:
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        fractional_contribution (list, optional): Fractional contribution of each spiral arm. Defaults to fractional_contribution.
        h_spiral_arm (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        rho_min_sagittarius (float, optional): The minimum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_min_sagittarius.
        rho_max_sagittarius (float, optional): The maximum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_max_sagittarius.
        sigma_devoid (float, optional): The dispersion of the devoid region of the Sagittarius-Carina spiral arm. Defaults to sigma_devoid.
        theta_max_local (float, optional): The maximum angle of the local arm. Defaults to theta_max_local.
    Returns:
        float: The optimized rho_min value
        float: The optimized rho_max value
        float: The optimized sigma_devoid value
    """
    if settings.add_devoid_region_sagittarius == False:
        logging.warning("In settings.py, add_devoid_region_sagittarius is set to False. Exiting...") # should realy remove this, as we should optimize for parameters regardless of if the user wants to add it to the model or not
        return
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"saggitarius_devoid_region_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data()
    best_rho_min, best_rho_max, best_sigma_devoid = rho_min_sagittarius, rho_max_sagittarius, sigma_devoid
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 

    rho_mins_sag_to_check = np.arange(rho_min_sagittarius - 0.1, rho_min_sagittarius + 0.1 + 0.1, 0.1)
    rho_maxs_sag_to_check = np.arange(rho_max_sagittarius - 0.1, rho_max_sagittarius + 0.1 + 0.1, 0.1)
    rho_maxs_sag_to_check = rho_maxs_sag_to_check[rho_maxs_sag_to_check <= 7.5] # ensure that rho_max is less than 7.5 kpc. Do not want to go to the left of the sun
    sigmas_sag_to_check = np.arange(sigma_devoid - 0.01, sigma_devoid + 0.01, 0.01) # some rounding errors here, hence I removed the extra  + 0.01
    sigmas_sag_to_check = sigmas_sag_to_check[sigmas_sag_to_check >= 0] # ensure that sigma is greater than 0
    arm_index = 2 # index for the Sagittarius-Carina arm
    chi_squared_min = np.inf
    # Calculate the intensity for the initial values
    sam.calc_modelled_intensity(readfile_effective_area=False, interpolate_all_arms=True, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm)
    # update the local arm with the new value for theta_max
    rho_max_local = const.rho_min_local * np.exp(np.tan(const.pitch_local) * (theta_max_local - const.theta_start_local))
    interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[4], pitch_angle=pitch_angles[4], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
    calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=4, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
    with open(filename_log, 'w') as file:
        # i = Norma-Cygnus, j = Perseus, k = Sagittarius-Carina, l = Scutum-Crux
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, rho_min, rho_max, sigma_devoid, theta_max_local, Chi-squared\n')
        for rho_min in rho_mins_sag_to_check: 
            for rho_max in rho_maxs_sag_to_check: 
                for sigma_devoid in sigmas_sag_to_check:
                    if rho_min > rho_max:
                        continue
                    interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[arm_index], pitch_angle=pitch_angles[arm_index], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=arm_index, rho_min_sagittarius=rho_min, rho_max_sagittarius=rho_max, sigma_devoid=sigma_devoid)
                    calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=arm_index, rho_min_sagittarius=rho_min, rho_max_sagittarius=rho_max, sigma_devoid=sigma_devoid)
                    sam.calc_modelled_intensity(readfile_effective_area=True, interpolate_all_arms=False, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm)
                    intensities_modelled = load_modelled_data()
                    chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
                    logging.info(f'rho_min: {rho_min}, rho_max: {rho_max}, sigma: {sigma_devoid}, chi_squared_val: {chi_squared_val}')
                    file.write(f'{np.degrees(arm_angles[0])}, {np.degrees(arm_angles[1])}, {np.degrees(arm_angles[2])}, {np.degrees(arm_angles[3])}, {np.degrees(pitch_angles[0])}, {np.degrees(pitch_angles[1])}, {np.degrees(pitch_angles[2])}, {np.degrees(pitch_angles[3])}, {fractional_contribution[0]}, {fractional_contribution[1]}, {fractional_contribution[2]}, {fractional_contribution[3]}, {h_spiral_arm}, {sigma_arm}, {rho_min}, {rho_max}, {sigma_devoid}, {np.degrees(theta_max_local)}, {chi_squared_val}\n')
                    if chi_squared_val < chi_squared_min:
                        chi_squared_min = chi_squared_val
                        best_rho_min, best_rho_max, best_sigma_devoid = rho_min, rho_max, sigma_devoid
    print('Best Rho_min, Rho_max, sigma_devoid:', best_rho_min, best_rho_max, best_sigma_devoid)
    print('Best chi-squared:', chi_squared_min)
    return best_rho_min, best_rho_max, best_sigma_devoid


def optimize_local_arm_theta_max(arm_angles=const.arm_angles, pitch_angles=const.pitch_angles, fractional_contribution=const.fractional_contribution, h_spiral_arm=const.h_spiral_arm, sigma_arm=const.sigma_arm, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid, theta_max_local=const.theta_max_local):
    """ Function to optimize the theta_max parameter for the local arm. The function will try different values for theta_max in the range +- 0.1 around the existing value, and calculate the chi-squared value for each value.
    The theta_max value that gives the lowest chi-squared value will be returned.

    Args:
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.
        fractional_contribution (list, optional): Fractional contribution of each spiral arm. Defaults to fractional_contribution.
        h_spiral_arm (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        rho_min_sagittarius (float, optional): The minimum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_min_sagittarius.
        rho_max_sagittarius (float, optional): The maximum distance from the galactic center to the beginning of the devoid region of Sagittarius spiral arm. Defaults to rho_max_sagittarius.
        sigma_devoid (float, optional): The dispersion of the devoid region of the Sagittarius-Carina spiral arm. Defaults to sigma_devoid.
        theta_max_local (float, optional): The maximum angle of the local arm. Defaults to theta_max_local.
    """
    logging.info("Optimizing theta_max for the local arm")
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"local_arm_theta_max_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data()
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    best_theta_max = np.degrees(const.theta_max_local)
    theta_max_to_check = np.linspace(np.degrees(const.theta_max_local) - 20, np.degrees(const.theta_max_local) + 20, 41)
    print(best_theta_max, theta_max_to_check)
    # calculate the intensity for the updated parameters, and remember to recalculate the devoid region
    sam.calc_modelled_intensity(readfile_effective_area=False, interpolate_all_arms=True, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm)
    interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[2], pitch_angle=pitch_angles[2], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
    calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=2, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid)
    arm_index=4 # index for the local arm
    chi_squared_min = np.inf
    with open(filename_log, 'w') as file:
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, rho_min_sagittarius, rho_max_sagittarius, sigma_devoid, theta_max_local, Chi-squared\n')
        for theta_max in theta_max_to_check:
            logging.info(f'Checking theta_max: {theta_max}')
            rho_max_local = const.rho_min_local * np.exp(np.tan(const.pitch_local) * (np.radians(theta_max) - const.theta_start_local))
            interpolate_density_one_arm(h_spiral_arm=h_spiral_arm, arm_angle=arm_angles[arm_index], pitch_angle=pitch_angles[arm_index], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=arm_index, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
            calc_effective_area_one_arm(h=h_spiral_arm, sigma_arm=sigma_arm, arm_angles=arm_angles, pitch_angles=pitch_angles, arm_index=arm_index, rho_min_sagittarius=rho_min_sagittarius, rho_max_sagittarius=rho_max_sagittarius, sigma_devoid=sigma_devoid, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
            sam.calc_modelled_intensity(readfile_effective_area=True, interpolate_all_arms=False, calc_gum_cyg=False, arm_angles=arm_angles, pitch_angles=pitch_angles, fractional_contribution=fractional_contribution, h=h_spiral_arm, sigma_arm=sigma_arm)
            intensities_modelled = load_modelled_data()
            chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
            logging.info(f'theta_max_local: {theta_max}, chi_squared_val: {chi_squared_val}')
            file.write(f'{np.degrees(arm_angles[0])}, {np.degrees(arm_angles[1])}, {np.degrees(arm_angles[2])}, {np.degrees(arm_angles[3])}, {np.degrees(pitch_angles[0])}, {np.degrees(pitch_angles[1])}, {np.degrees(pitch_angles[2])}, {np.degrees(pitch_angles[3])}, {fractional_contribution[0]}, {fractional_contribution[1]}, {fractional_contribution[2]}, {fractional_contribution[3]}, {h_spiral_arm}, {sigma_arm}, {rho_min_sagittarius}, {rho_max_sagittarius}, {sigma_devoid}, {theta_max}, {chi_squared_val}\n')
            if chi_squared_val < chi_squared_min:
                best_theta_max = theta_max
                chi_squared_min = chi_squared_val
    print('Best theta_max_local', best_theta_max)
    print('Best chi-squared:', chi_squared_min)
    return np.radians(best_theta_max)


def run_tests(num_iterations=10):
    """
    Function to run all the optimization tests
    """
    # Load existing data
    best_arm_angles = const.arm_angles.copy()
    best_pitch_angles = const.pitch_angles.copy()
    best_sigma_arm = const.sigma_arm
    best_h_spiral_arm = const.h_spiral_arm
    best_fractional_contribution = const.fractional_contribution.copy()
    best_rho_min = const.rho_min_sagittarius
    best_rho_max = const.rho_max_sagittarius
    best_sigma_devoid = const.sigma_devoid
    best_theta_max = const.theta_max_local
    # calculate the gum-cygnus OBA:
    gum_cygnus.generate_gum_cygnus() # to ensure the files exist before running the optimization tests
    # Run the optimization tests
    for i in range(num_iterations):
        print(f'Iteration: {i + 1}')
        best_rho_min, best_rho_max, best_sigma_devoid = optimize_saggitarus_devoid_region(arm_angles=best_arm_angles, pitch_angles=best_pitch_angles, fractional_contribution=best_fractional_contribution, h_spiral_arm=best_h_spiral_arm, sigma_arm=best_sigma_arm, rho_min_sagittarius=best_rho_min, rho_max_sagittarius=best_rho_max, sigma_devoid=best_sigma_devoid, theta_max_local=best_theta_max)
        best_arm_angles = optimize_spiral_arm_start_angle(delta=1, arm_angles=best_arm_angles, pitch_angles=best_pitch_angles, fractional_contribution=best_fractional_contribution, h_spiral_arm=best_h_spiral_arm, sigma_arm=best_sigma_arm, rho_min_sagittarius=best_rho_min, rho_max_sagittarius=best_rho_max, sigma_devoid=best_sigma_devoid, theta_max_local=best_theta_max)
        best_pitch_angles = optimize_spiral_arm_pitch_angle(delta=0.1, arm_angles=best_arm_angles, pitch_angles=best_pitch_angles, fractional_contribution=best_fractional_contribution, h_spiral_arm=best_h_spiral_arm, sigma_arm=best_sigma_arm, rho_min_sagittarius=best_rho_min, rho_max_sagittarius=best_rho_max, sigma_devoid=best_sigma_devoid, theta_max_local=best_theta_max)
        best_sigma_arm = optimize_sigma_arm(arm_angles=best_arm_angles, pitch_angles=best_pitch_angles, fractional_contribution=best_fractional_contribution, h_spiral_arm=best_h_spiral_arm, sigma_arm=best_sigma_arm, rho_min_sagittarius=best_rho_min, rho_max_sagittarius=best_rho_max, sigma_devoid=best_sigma_devoid, theta_max_local=best_theta_max)
        best_h_spiral_arm = optimize_spiral_arm_h(arm_angles=best_arm_angles, pitch_angles=best_pitch_angles, fractional_contribution=best_fractional_contribution, h_spiral_arm=best_h_spiral_arm, sigma_arm=best_sigma_arm, rho_min_sagittarius=best_rho_min, rho_max_sagittarius=best_rho_max, sigma_devoid=best_sigma_devoid, theta_max_local=best_theta_max)
        best_fractional_contribution = optimize_fractional_contribution_four_spiral_arms(arm_angles=best_arm_angles, pitch_angles=best_pitch_angles, fractional_contribution=best_fractional_contribution, h_spiral_arm=best_h_spiral_arm, sigma_arm=best_sigma_arm, rho_min_sagittarius=best_rho_min, rho_max_sagittarius=best_rho_max, sigma_devoid=best_sigma_devoid, theta_max_local=best_theta_max)
        best_theta_max = optimize_local_arm_theta_max(arm_angles=best_arm_angles, pitch_angles=best_pitch_angles, fractional_contribution=best_fractional_contribution, h_spiral_arm=best_h_spiral_arm, sigma_arm=best_sigma_arm, rho_min_sagittarius=best_rho_min, rho_max_sagittarius=best_rho_max, sigma_devoid=best_sigma_devoid, theta_max_local=best_theta_max)
    print('Best spiral arm angles:', np.degrees(best_arm_angles))
    print('Best pitch angles:', np.degrees(best_pitch_angles))
    print('Best sigma_arm:', best_sigma_arm)
    print('Best h_spiral_arm:', best_h_spiral_arm)
    print('Best fractional contribution:', best_fractional_contribution)
    print('Best rho_min, rho_max, sigma_devoid:', best_rho_min, best_rho_max, best_sigma_devoid)
    print('Best theta_max_local:', np.degrees(best_theta_max))


def plot_bins_excluded_from_fiting(filename_output = f'{const.FOLDER_MODELS_GALAXY}/bins_excluded_from_fitting.pdf', filename_intensity_data = f'{const.FOLDER_GALAXY_DATA}/intensities_per_arm_b_max_5.npy'):
    """
    Function to plot the bins that are excluded from the fitting
    """
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # plot the modelled intensity
    #sam.calc_modelled_intensity(readfile_effective_area=False, interpolate_all_arms=True, calc_gum_cyg=True)
    intensities_per_arm = np.load(filename_intensity_data)
    intensities_total = np.sum(intensities_per_arm, axis=0)
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    if settings.add_gum_cygnus == True:
        try: # check if the gum-cygnus regions have been generated
            gum = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_gum.npy')
            cygnus = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_cygnus.npy')
            gum_cygnus = gum + cygnus
            intensities_total += gum_cygnus
        except: # if not, raise a warning
            logging.warning("The Gum and Cygnus regions were not included in the model. They may not have been generated. Skipping this part of the plot. Try calling the function 'gum' and 'cygnus' in gum_cygnus.py first")
    colors = sns.color_palette('bright', 7)
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_total, label="Total intensity", color=colors[6])
    bins_removed = []
    intensities_bins_removed = []
    for removed_bin in RANGES_TO_REMOVE_BIN_CENTERS:
        if removed_bin <=180:
            removed_bin += 2 * (90 - removed_bin)
        elif removed_bin > 180:
            removed_bin += 2 * (270 - removed_bin)
        removed_bin_intensity = line_flux[np.where((bin_centre_line_flux == removed_bin))]
        bins_removed.append(removed_bin)
        intensities_bins_removed.append(removed_bin_intensity[0])
    plt.bar(bins_removed, intensities_bins_removed, width=5, color='red', alpha=0.5, label='Bins excluded from fitting')
    
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    # Add parameter values as text labels
    fontsize_in_ax_text = 12
    plt.xlim(0, 360)
    plt.ylim(bottom=0)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text)
    plt.savefig(filename_output)
    plt.close()


def main():
    run_tests()
    plot_bins_excluded_from_fiting()


if __name__ == '__main__':
    main()
