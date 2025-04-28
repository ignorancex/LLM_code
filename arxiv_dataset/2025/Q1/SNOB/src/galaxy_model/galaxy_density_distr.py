import numpy as np
import logging
import os
logging.basicConfig(level=logging.INFO) 
import src.nii_intensities.spiral_arm_model as sam
import src.utilities.utilities as ut
import src.utilities.constants as const
from pathlib import Path


@ut.timing_decorator    
def monte_carlo_numpy_choise(density_distribution, NUM_ASC):
    """ Function to draw associations from the density distribution using numpy's choice method
    
    Args:
        density_distribution (np.array): The density distribution from which to draw the associations
        NUM_ASC (int): The number of associations to draw
    Returns:
        grid_index (np.array): The grid-indexes of the drawn associations
    """
    rng = np.random.default_rng()
    density_distribution /= np.sum(density_distribution) # normalize the density to unity
    grid_index = rng.choice(a=len(density_distribution), size=NUM_ASC, p=density_distribution, replace=False) #replace = False means that the same index cannot be drawn twice
    return grid_index


def interpolate_density(x_grid, y_grid, h=const.h_lyc, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles):
    """ Integrates the densities of the spiral arms over the entire galactic plane. The returned density is in units of kpc^-2. 
    
    Args:
        x_grid (2D np.array): Contains all the x-values for the grid
        y_grid (2D np.array): Contains all the y-values for the grid
        h (float, optional): Scale length of the disk. Defaults to h_default.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm_default.
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.

    Returns:
        interpolated_densities (3D np.array): Interpolated densities for each spiral arm along axis 0. Axis 1 and 2 are the densities with respect to the grid
    """
    logging.info("Interpolating the density distribution of the Milky Way")
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) #d_min: minimum distance from the spiral arm
    interpolated_densities = []
    rho_min_spiral_arm = const.rho_min_spiral_arm
    rho_max_spiral_arm = const.rho_max_spiral_arm
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = sam.spiral_arm_medians(arm_angles[i], pitch_angles[i], rho_min_spiral_arm[i], rho_max_spiral_arm[i])
        # generate the spiral arm points 
        x, y = sam.generate_spiral_arm_coordinates(rho, transverse_distances, theta, pitch_angles[i])
        # generate the spiral arm densities
        density_spiral_arm = sam.generate_spiral_arm_densities(rho, transverse_densities_initial, h, arm_index=i, transverse_distances=transverse_distances)
        # calculate interpolated density for the spiral arm
        interpolated_arm = sam.griddata((x, y), density_spiral_arm, (x_grid, y_grid), method='cubic', fill_value=0)
        interpolated_arm[interpolated_arm < 0] = 0 # set all negative values to 0
        # normalize the density to the highest value of the density
        interpolated_densities.append(interpolated_arm)
    logging.info("Interpolation done")
    return interpolated_densities


@ut.timing_decorator
def generate_coords_densities(plane=1000, transverse=20, half_edge=40, readfile_effective_area=True, read_data_from_file=True):
    """ Function to generate the coordinates and densities for the density distribution of the Milky Way

    Args:
        plane (int, optional): Number of points in the x and y direction. Defaults to 1000.
        transverse (int, optional): Number of points in the z direction. Defaults to 20.
        half_edge (int, optional): Half the length of the grid in x and y direction. Defaults to 25 kpc.
        readfile (bool, optional): Whether to read the effective area from a file. Defaults to True.
        
    Returns:
        x_grid (np.array): x-values of the grid
        y_grid (np.array): y-values of the grid
        z_grid (np.array): z-values of the grid
        uniform_spiral_arm_density_total (np.array): The uniform spiral arm density distribution
        emissivity (np.array): The modelled emissivity
    """
    # check if the folder exists, if not create it
    Path(const.FOLDER_GALAXY_DATA).mkdir(parents=True, exist_ok=True)
    if read_data_from_file == True:
        if not os.path.exists(f'{const.FOLDER_GALAXY_DATA}/sim_x_grid.npy'):
            logging.warning("Data for simulating the Galaxy has not been generated. Generating it now.")
            generate_coords_densities(plane=plane, transverse=transverse, half_edge=half_edge, readfile_effective_area=readfile_effective_area, read_data_from_file=False)
        logging.info("Reading the density distribution of the Milky Way from file")
        x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/sim_x_grid.npy')
        y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/sim_y_grid.npy')
        z_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/sim_z_grid.npy')
        uniform_spiral_arm_density = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/sim_uniform_spiral_arm_density.npy')
        emissivity = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/sim_emissivity.npy')
        return x_grid, y_grid, z_grid, uniform_spiral_arm_density, emissivity
    logging.info("Generating the coordinates and densities for the density distribution of the Milky Way")
    # Use slice to create slices programmatically
    x_grid, y_grid, z_grid = np.mgrid[slice(-half_edge, half_edge, plane*1j), slice(-half_edge, half_edge, plane*1j), slice(-0.5, 0.5, transverse*1j)]
    x_grid, y_grid, z_grid = x_grid.ravel(), y_grid.ravel(), z_grid.ravel() # ravel the grid
    uniform_spiral_arm_density = interpolate_density(x_grid, y_grid) # generate the density
    uniform_spiral_arm_density_total = np.sum(uniform_spiral_arm_density, axis=0) # sum up all the arms to get an array for the total galactic density
    uniform_spiral_arm_density_total = uniform_spiral_arm_density_total.ravel() # flatten the array
    uniform_spiral_arm_density_total /= uniform_spiral_arm_density_total.max() # normalize the density to unity
    # Either read or calculate the effective area per spiral arm
    if readfile_effective_area == True:
        try:
            effective_area = np.load(f'{const.FOLDER_GALAXY_DATA}/effective_area_per_spiral_arm.npy')
        except:
            logging.warning("File effective_area_per_spiral_arm.npy does not exist in the folder. Calculating the effective area per spiral arm")
            effective_area = sam.calc_effective_area_per_spiral_arm(readfile_effective_area=readfile_effective_area)
    elif readfile_effective_area == False:
        effective_area = sam.calc_effective_area_per_spiral_arm(readfile_effective_area=readfile_effective_area)
    else:
        raise ValueError("readfile must be either True or False")
    # Calculate the emissivity
    common_multiplication_factor = const.total_galactic_n_luminosity * ut.height_distribution(z_grid, sigma=const.sigma_height_distr) # common factor for the emissivity
    for i in range(len(uniform_spiral_arm_density)):
        uniform_spiral_arm_density[i] *= common_multiplication_factor * const.fractional_contribution[i] / (effective_area[i] * const.kpc**2) # scale each spiral arm with the common factor, the fractional contribution and the effective area in cm^2
    emissivity = np.sum(uniform_spiral_arm_density, axis=0)
    emissivity = emissivity.ravel()
    emissivity /= emissivity.max()
    # Save the data to file
    np.save(f'{const.FOLDER_GALAXY_DATA}/sim_x_grid.npy', x_grid)
    np.save(f'{const.FOLDER_GALAXY_DATA}/sim_y_grid.npy', y_grid)
    np.save(f'{const.FOLDER_GALAXY_DATA}/sim_z_grid.npy', z_grid)
    np.save(f'{const.FOLDER_GALAXY_DATA}/sim_uniform_spiral_arm_density.npy', uniform_spiral_arm_density_total) # drop total from the filename, as it will be understood that it is the total density and not per arm
    np.save(f'{const.FOLDER_GALAXY_DATA}/sim_emissivity.npy', emissivity)
    logging.info("Coordinates and densities for the density distribution of the Milky Way generated")
    return x_grid, y_grid, z_grid, uniform_spiral_arm_density_total, emissivity


def galaxy_density_distr_test():
    x_grid, y_grid, z_grid, uniform_spiral_arm_density, emissivity = generate_coords_densities(read_data_from_file=False, read_eff_area=True)
    grid_index = monte_carlo_numpy_choise(uniform_spiral_arm_density, 15000) # draw the associations


if __name__ == "__main__":
    galaxy_density_distr_test()
    logging.info("galaxy_density_distr.py is working correctly")