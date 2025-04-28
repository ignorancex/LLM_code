import numpy as np
import matplotlib.pyplot as plt
import src.utilities.utilities as ut
import logging
logging.basicConfig(level=logging.INFO)
import src.observational_data.firas_data as firas_data
import src.utilities.constants as const
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path


def calc_modelled_intensity(b_max = 5):
    logging.info("Running calc_modelled_intensity() for axisymmetric model")
    logging.info("Calculating coordinates")
    # check if the folder exists, if not create it
    Path(const.FOLDER_GALAXY_DATA).mkdir(parents=True, exist_ok=True)
    # Calculate coordinates    
    dr = 0.01   # increments in dr (kpc). For the spiral arm model, 0.01 kpc was used, but seems like 0.1 kpc is enough for the axisymmetric model
    dl = 0.2   # increments in dl (degrees)
    db = 0.1   # increments in db (degrees)
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(dr, const.r_s + const.rho_max_axisymmetric + dr, dr) #r_s + rho_max is the maximum distance from the Sun to the outer edge of the Galaxy
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    dl = np.radians(dl)
    # np.array with values for galactic latitude b in radians.
    latitudes = np.radians(np.arange(-b_max, b_max + db, db))
    db = np.radians(db)
    num_rads, num_longs, num_lats = len(radial_distances), len(longitudes), len(latitudes)
    # Create meshgrids. Effciciently creates 3D arrays with the coordinates for r, l and b
    radial_grid, long_grid, lat_grid = np.meshgrid(radial_distances, longitudes, latitudes, indexing='ij')
    latitudinal_cosinus = np.cos(lat_grid.ravel())
    height_distribution_values = ut.height_distribution(ut.z(radial_grid.ravel(), lat_grid.ravel()), sigma=const.sigma_height_distr)
    # Calculate the common multiplication factor for the modelled intensity. This factor ensures that, after the summation of the relative_density array further down, the result is the integrated intensity
    common_multiplication_factor =  dr * db * latitudinal_cosinus * height_distribution_values/ (4 * np.pi * np.radians(1) * const.a_d_axisymmetric * const.kpc**2) 
    rho = ut.rho(radial_grid.ravel(), long_grid.ravel(), lat_grid.ravel())
    logging.info("Coordinates calculated. Calculating modelled intensity")
    relative_density = ut.axisymmetric_disk_population(rho, const.h_axisymmetric) * common_multiplication_factor # 3D array with the modelled relative density for each value of r, l and b
    logging.info("Modelled intensity calculated. Removing values outside the bright H 2 regions and normalizing to the measured value at 30 degrees longitude")
    mask = (rho > const.rho_max_axisymmetric) | (rho < const.rho_min_axisymmetric)
    # Set values in intensities to zero where the mask is True
    relative_density[mask] = 0
    intensities = np.sum(relative_density.reshape(num_rads, num_longs, num_lats), axis=(0, 2)) # sum over the radial and latitudinal axis
    windowsize = np.radians(5) / dl
    intensities = ut.running_average(intensities, windowsize) * dl / np.radians(5)  # Calculate the running average of the intensities
    # Now normalize the modelled intensity to the measured value at 30 degrees longitude in the FIRAS data
    abs_diff = np.abs(longitudes - np.radians(30))  # Calculate the absolute differences between the 30 degrees longitude and all elements in longitudes
    closest_index = np.argmin(abs_diff) # Find the index of the element with the smallest absolute difference
    modelled_value_30_degrees = intensities[closest_index] # Retrieve the closest value from the integrated spectrum
    luminosity_axisymmetric = const.measured_nii_30_deg / modelled_value_30_degrees # Calculate the normalization factor
    logging.info("Normalization factor calculated: %s", luminosity_axisymmetric)
    intensities *= luminosity_axisymmetric # normalize the modelled emissivity to the measured value at 30 degrees longitude    
    # Note: for comparison reasons with Higdon and Lingenfelter I am not using a running average on the intensities, as it smoothen the spikes
    logging.info("Saving the modelled intensity and the normalization factor")
    np.save(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_luminosity.npy', luminosity_axisymmetric)
    np.save(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_intensities.npy', intensities) 
    np.save(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_longitudes.npy', longitudes) 
    return


def plot_axisymmetric():
    logging.info("Plotting the modelled intensity for the axisymmetric model")
    # check if the output folder exists, if not create it
    Path(const.FOLDER_MODELS_GALAXY).mkdir(parents=True, exist_ok=True)
    filename_output =  f'{const.FOLDER_MODELS_GALAXY}/axisymmetric_modelled_emissivity_h_2.4.pdf'
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_longitudes.npy')
    intensities = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_intensities.npy')
    luminosity = np.load(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_luminosity.npy')
    # plot the FIRAS data for the NII 205 micron line
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # plot the modelled intensity
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    #plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_axisymmetric} kpc & $\sigma_z$ = {const.sigma_height_distr} kpc', transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.text(0.02, 0.9, fr'N$\,$II Luminosity = {luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.text(0.02, 0.85, fr'{const.rho_min_axisymmetric:.2e} $\leq \rho \leq$ {const.rho_max_axisymmetric:.2e} kpc', transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.savefig(filename_output)
    plt.close()

def plot_axisymmetric_compare_rhos(rho_mins, rho_maxs):
    assert len(rho_mins) == len(rho_maxs), "rho_mins and rho_maxs must have the same length"
    logging.info("Plotting the modelled intensity for the axisymmetric model for different rhos")
    # check if the output folder exists, if not create it
    Path(const.FOLDER_MODELS_GALAXY).mkdir(parents=True, exist_ok=True)
    # plot the FIRAS data for the NII 205 micron line
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # plot the modelled intensities for different rhos
    for i in range(len(rho_mins)):
        const.rho_min_axisymmetric = rho_mins[i]
        const.rho_max_axisymmetric = rho_maxs[i]
        calc_modelled_intensity()
        intensities = np.load(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_intensities.npy')
        luminosity = np.load(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_luminosity.npy')
        plt.plot(np.linspace(0, 360, len(intensities)), intensities, label=f'{rho_mins[i]} - {rho_maxs[i]} kpc. NII Luminosity = {luminosity:.2e} erg/s')
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_axisymmetric} kpc & $\sigma_z$ = {const.sigma_height_distr} kpc', transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.legend(loc='upper right')
    filename_output =  f'{const.FOLDER_MODELS_GALAXY}/axisymmetric_modelled_emissivity_comp_rhos_norm_30_deg_2.pdf'
    plt.savefig(filename_output)
    plt.close()
    
    
def main():
    logging.basicConfig(level=logging.INFO) 
    calc_modelled_intensity(b_max=5)
    plot_axisymmetric()
    # Test the effect of different rhos
    rho_mins = [3, 3.2, 3.5]
    rho_maxs = [11, 11, 11]
    plot_axisymmetric_compare_rhos(rho_mins, rho_maxs)


if __name__ == "__main__":
    main()
