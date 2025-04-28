import numpy as np
import src.observational_data.firas_data as firas_data
import src.utilities.constants as const
import src.utilities.settings as settings
import src.nii_intensities.spiral_arm_model as sam
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from src.nii_intensities.chi_squared import interpolate_density_one_arm, calc_effective_area_one_arm
import seaborn as sns
import logging
logging.basicConfig(level=logging.INFO)


TOP = 2.2e-4 # y-axis limit for the plots. Used for the tests which have long label names
LINEWIDTH = 1 # linewidth for the plots
NUM_FITTED_PARAMETERS = 40 # number of fitted parameters in the model
linestyles = [(0, (3, 5, 1, 5, 1, 5)), (0, (1, 1)), (5, (10, 3)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1))] # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html for reference

"""
    This module contains functions to test the spiral arm model by varying the parameters of the model and plotting the resulting N II intensity. The tests are:
    - The start angle of the major arms
    - The pitch angle of the major arms
    - The fractional contribution of the major arms
    - The width of the spiral arms
    - The scale height of the spiral arms
    - Parameters for the devoid region of Sagittarius-Carina
    - Max angle of the local arm
    Finally, we calculate the reduced chi-squared value for the case without the local arm and the devoid region of Sagittarius-Carina included in the model,
    for the case where we add only the local arm and also for the case where both are included.
"""
def reduced_chi_squared(observational_data, observational_data_variance, modelled_data):
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
    chi_squared = np.sum(((observational_data - modelled_data) ** 2) / observational_data_variance)
    N = len(observational_data)  # number of data points
    degrees_of_freedom = N - NUM_FITTED_PARAMETERS
    if degrees_of_freedom <= 0:
        raise ValueError("The number of degrees of freedom must be positive.")
    reduced_chi_squared = chi_squared / degrees_of_freedom
    return reduced_chi_squared


def load_firas_data():
    """ Function to load the FIRAS data and variance, and expand the data to match the modelled data in number of points to facilitate the chi-squared calculation 

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

    Args: 
        filename_arm_intensities (str): The filename of the file containing the modelled intensities per arm
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


def plot_test_start_angle_major_arms(filename_output=f'{const.FOLDER_MODELS_GALAXY}/test_start_angle_major_arms.pdf'):
    """ Function to plot the resulting N II intensity for different start angles of the major arms
    
    Args:
        filename_output (str): The filename of the output plot
    """
    logging.info("Plotting the test for the start angle of the major arms")
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # FIRAS data to calculate the reduced chi-squared value:
    expanded_firas_intensity, expanded_firas_variance = load_firas_data()
    # parameters for the plot:
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    colors = sns.color_palette('bright', 5)
    arm_angles_to_check = [const.arm_angles,
                            np.array([const.arm_angles[0] + np.radians(10), const.arm_angles[1], const.arm_angles[2], const.arm_angles[3], const.arm_angles[4]]),
                            np.array([const.arm_angles[0], const.arm_angles[1] + np.radians(10), const.arm_angles[2], const.arm_angles[3], const.arm_angles[4]]),
                            np.array([const.arm_angles[0], const.arm_angles[1], const.arm_angles[2] + np.radians(10), const.arm_angles[3], const.arm_angles[4]]),
                            np.array([const.arm_angles[0], const.arm_angles[1], const.arm_angles[2], const.arm_angles[3] + np.radians(10), const.arm_angles[4]])]
    for i in range(len(arm_angles_to_check)):
        arm_angles = arm_angles_to_check[i]
        sam.calc_modelled_intensity(readfile_effective_area=False, arm_angles=arm_angles)
        intensities_total = load_modelled_data()
        reduced_chi_squared_value = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_total)
        label = f'$\\theta_{{\mathrm{{NC}}}}$ = {np.round(np.degrees(arm_angles[0]), 0)}°, $\\theta_{{\mathrm{{P}}}}$ = {np.round(np.degrees(arm_angles[1]), 0)}°, $\\theta_{{\mathrm{{SA}}}}$ = {np.round(np.degrees(arm_angles[2]), 0)}°, $\\theta_{{\mathrm{{SC}}}}$ = {np.round(np.degrees(arm_angles[3]), 0)}°. Reduced $\\chi^2$ = {reduced_chi_squared_value:.2f}'
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_total, label=label, color=colors[i], linestyle=linestyles[i], linewidth=LINEWIDTH)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    #plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    fontsize_in_ax_text = 12
    #plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_spiral_arm} kpc & $\sigma_{{\mathrm{{A}}}}$ = {const.sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=fontsize_in_ax_text, color='black')
    #plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=fontsize_in_ax_text, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0, top=2.2e-4)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text)
    plt.savefig(filename_output)
    plt.close()


def plot_test_pitch_angle_major_arms(filename_output=f'{const.FOLDER_MODELS_GALAXY}/test_pitch_angle_major_arms.pdf'):
    """ Function to plot the resulting N II intensity for different pitch angles of the major arms
    
    Args:
        filename_output (str): The filename of the output plot
    """
    logging.info("Plotting the test for the pitch angles of the major arms")
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # FIRAS data to calculate the reduced chi-squared value:
    expanded_firas_intensity, expanded_firas_variance = load_firas_data()
    # parameters for the plot:
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    colors = sns.color_palette('bright', 5)
    pitch_angles_to_check = [const.pitch_angles,
                            np.array([const.pitch_angles[0] + np.radians(2), const.pitch_angles[1], const.pitch_angles[2], const.pitch_angles[3], const.pitch_angles[4]]),
                            np.array([const.pitch_angles[0], const.pitch_angles[1] + np.radians(2), const.pitch_angles[2], const.pitch_angles[3], const.pitch_angles[4]]),
                            np.array([const.pitch_angles[0], const.pitch_angles[1], const.pitch_angles[2] + np.radians(2), const.pitch_angles[3], const.pitch_angles[4]]),
                            np.array([const.pitch_angles[0], const.pitch_angles[1], const.pitch_angles[2], const.pitch_angles[3] + np.radians(2), const.pitch_angles[4]])]
    for i in range(len(pitch_angles_to_check)):
        sam.calc_modelled_intensity(readfile_effective_area=False, pitch_angles=pitch_angles_to_check[i])
        intensities_total = load_modelled_data()
        reduced_chi_squared_value = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_total)
        label = f'$p_{{\mathrm{{NC}}}}$ = {np.round(np.degrees(pitch_angles_to_check[i][0]), 2)}°, $p_{{\mathrm{{P}}}}$ = {np.round(np.degrees(pitch_angles_to_check[i][1]), 2)}°, $p_{{\mathrm{{SA}}}}$ = {np.round(np.degrees(pitch_angles_to_check[i][2]), 2)}°, $p_{{\mathrm{{SC}}}}$ = {np.round(np.degrees(pitch_angles_to_check[i][3]), 2)}°. Reduced $\\chi^2$ = {reduced_chi_squared_value:.2f}'
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_total, label=label, color=colors[i], linestyle=linestyles[i], linewidth=LINEWIDTH)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    #plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    fontsize_in_ax_text = 12
    #plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_spiral_arm} kpc & $\sigma_{{\mathrm{{A}}}}$ = {const.sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=14, color='black')
    #plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0, top=2.2e-4)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text)
    plt.savefig(filename_output)
    plt.close()


def plot_test_fractional_contribution(filename_output=f'{const.FOLDER_MODELS_GALAXY}/test_fractional_contribution.pdf'):
    """ Function to plot the resulting N II intensity for different fractional contributions of the major arms
    
    Args:
        filename_output (str): The filename of the output plot
    """
    logging.info("Plotting the test for the fractional contribution of the major arms")
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # FIRAS data to calculate the reduced chi-squared value:
    expanded_firas_intensity, expanded_firas_variance = load_firas_data()
    # parameters for the plot:
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    colors = sns.color_palette('bright', 5)
    fractional_contribution_to_check = [const.fractional_contribution,
                                        np.array([0.18, 0.35, 0.18, 0.28, 0.01]),
                                        np.array([0.25, 0.24, 0.25, 0.25, 0.01]),
                                        np.array([0.15, 0.39, 0.15, 0.30, 0.01])]
    for i in range(len(fractional_contribution_to_check)):
        sam.calc_modelled_intensity(readfile_effective_area=False, fractional_contribution=fractional_contribution_to_check[i])
        intensities_total = load_modelled_data()
        reduced_chi_squared_value = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_total)
        label = f'$f_{{\mathrm{{NC}}}}$ = {fractional_contribution_to_check[i][0]}, $f_{{\mathrm{{P}}}}$ = {fractional_contribution_to_check[i][1]}, $f_{{\mathrm{{SA}}}}$ = {fractional_contribution_to_check[i][2]}, $f_{{\mathrm{{SC}}}}$ = {fractional_contribution_to_check[i][3]:.2f}. Reduced $\\chi^2$ = {reduced_chi_squared_value:.2f}'
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_total, label=label, color=colors[i], linestyle=linestyles[i], linewidth=LINEWIDTH)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    #plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    fontsize_in_ax_text = 12
    #plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_spiral_arm} kpc & $\sigma_{{\mathrm{{A}}}}$ = {const.sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=14, color='black')
    #plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0, top=2.0e-4)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text)
    plt.savefig(filename_output)
    plt.close()


def plot_test_sigma_arm(filename_output=f'{const.FOLDER_MODELS_GALAXY}/test_sigma_arm.pdf'):
    """ Function to plot the resulting N II intensity for different sigmas for the spiral arms
    
    Args:
        filename_output (str): The filename of the output plot
    """
    logging.info("Plotting the test for the width of the spiral arms")
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # FIRAS data to calculate the reduced chi-squared value:
    expanded_firas_intensity, expanded_firas_variance = load_firas_data()
    # parameters for the plot:
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    colors = sns.color_palette('bright', 5)
    sigma_arm_to_check = [0.25, 0.40, 0.50, 0.60, 0.75]
    for i in range(len(sigma_arm_to_check)):
        sam.calc_modelled_intensity(readfile_effective_area=False, sigma_arm=sigma_arm_to_check[i])
        intensities_total = load_modelled_data()
        reduced_chi_squared_value = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_total)
        label = f'$\sigma_{{\mathrm{{A}}}}$ = {sigma_arm_to_check[i]:.2f} kpc. Reduced $\\chi^2$ = {reduced_chi_squared_value:.2f}'
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_total, label=label, color=colors[i], linestyle=linestyles[i], linewidth=LINEWIDTH)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    #plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    fontsize_in_ax_text = 12
    #plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_spiral_arm} kpc & $\sigma_{{\mathrm{{A}}}}$ = {const.sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=14, color='black')
    #plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0, top=2.2e-4)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text)
    plt.savefig(filename_output)
    plt.close()


def plot_test_h_arm(filename_output=f'{const.FOLDER_MODELS_GALAXY}/test_h_arm.pdf'):
    """ Function to plot the resulting N II intensity for different scale lengths of the spiral arms
    
    Args:
        filename_output (str): The filename of the output plot
    """
    logging.info("Plotting the test for the scale height of the spiral arms")
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # FIRAS data to calculate the reduced chi-squared value:
    expanded_firas_intensity, expanded_firas_variance = load_firas_data()
    # parameters for the plot:
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    colors = sns.color_palette('bright', 5)
    h_arm_to_check = [1.8, 2.1, 2.4, 2.7, 3.0]
    for i in range(len(h_arm_to_check)):
        sam.calc_modelled_intensity(readfile_effective_area=False, h=h_arm_to_check[i])
        intensities_total = load_modelled_data()
        reduced_chi_squared_value = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_total)
        label = f'H$_\\rho$ = {h_arm_to_check[i]} kpc. Reduced $\\chi^2$ = {reduced_chi_squared_value:.2f}'
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_total, label=label, color=colors[i], linestyle=linestyles[i], linewidth=LINEWIDTH)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    #plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    fontsize_in_ax_text = 12
    #plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_spiral_arm} kpc & $\sigma_{{\mathrm{{A}}}}$ = {const.sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=14, color='black')
    #plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0, top=2.2e-4)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text, loc='upper right')
    plt.savefig(filename_output)
    plt.close()


def plot_test_devoid_region_sagittarius(filename_output=f'{const.FOLDER_MODELS_GALAXY}/test_devoid_region.pdf'):
    """ Function to plot the resulting N II intensity for different parameters of the devoid region in Sagittarius

    Args:
        filename_output (str): The filename of the output plot
    """
    logging.info("Plotting the test for the devoid region in Sagittarius")
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # FIRAS data to calculate the reduced chi-squared value:
    expanded_firas_intensity, expanded_firas_variance = load_firas_data()
    # parameters for the plot:
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    colors = sns.color_palette('bright', 5)
    rho_min_sagittarius_to_check = [5.1, 5.3, 5.5]
    rho_max_sagittarius_to_check = [7.0, 6.8, 7.3]
    sigma_devoid_to_check = [0.25, 0.3, 0.4]
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(const.sigma_arm) 
    # calculate initial intensities:
    sam.calc_modelled_intensity(readfile_effective_area=False)
    for i in range(len(rho_min_sagittarius_to_check)):
        interpolate_density_one_arm(arm_index=2, rho_min_sagittarius=rho_min_sagittarius_to_check[i], rho_max_sagittarius=rho_max_sagittarius_to_check[i], sigma_devoid=sigma_devoid_to_check[i], h_spiral_arm=const.h_spiral_arm, arm_angle=const.arm_angles[2], pitch_angle=const.pitch_angles[2], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial)
        calc_effective_area_one_arm(arm_index=2, rho_min_sagittarius=rho_min_sagittarius_to_check[i], rho_max_sagittarius=rho_max_sagittarius_to_check[i], sigma_devoid=sigma_devoid_to_check[i])
        sam.calc_modelled_intensity(readfile_effective_area=True, interpolate_all_arms=False, calc_gum_cyg=False)
        intensities_total = load_modelled_data()
        reduced_chi_squared_value = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_total)
        label = f'$\\rho_{{\mathrm{{min}}}}$ = {rho_min_sagittarius_to_check[i]}, $\\rho_{{\mathrm{{max}}}}$ = {rho_max_sagittarius_to_check[i]:.1f}, $\\sigma_{{\mathrm{{d}}}}$ = {sigma_devoid_to_check[i]}. Reduced $\\chi^2$ = {reduced_chi_squared_value:.2f}'
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_total, label=label, color=colors[i], linestyle=linestyles[i], linewidth=LINEWIDTH)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    #plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    fontsize_in_ax_text = 12
    #plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_spiral_arm} kpc & $\sigma_{{\mathrm{{A}}}}$ = {const.sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=14, color='black')
    #plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0, top=1.8e-4)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text)
    plt.savefig(filename_output)
    plt.close()


def plot_test_max_angle_local_arm(filename_output=f'{const.FOLDER_MODELS_GALAXY}/test_max_angle_local_arm.pdf'):
    """ Function to plot the resulting N II intensity for different maximum angles of the local arm

    Args:
        filename_output (str): The filename of the output plot
    """
    logging.info("Plotting the test for the maximum angle of the local arm")
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # FIRAS data to calculate the reduced chi-squared value:
    expanded_firas_intensity, expanded_firas_variance = load_firas_data()
    # parameters for the plot:
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    colors = sns.color_palette('bright', 5)
    theta_max_to_check = [const.theta_max_local, const.theta_max_local + np.radians(3), const.theta_max_local + np.radians(6)] # 110 +- 3 degrees
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(const.sigma_arm) 
    # calculate initial intensities:
    sam.calc_modelled_intensity(readfile_effective_area=False)
    interpolate_density_one_arm(h_spiral_arm=const.h_spiral_arm, arm_angle=const.arm_angles[2], pitch_angle=const.pitch_angles[2], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial, arm_index=2, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid)
    calc_effective_area_one_arm(arm_index=2, rho_min_sagittarius=const.rho_min_sagittarius, rho_max_sagittarius=const.rho_max_sagittarius, sigma_devoid=const.sigma_devoid)
    for i in range(len(theta_max_to_check)):
        rho_max_local = const.rho_min_local * np.exp(np.tan(const.pitch_local) * (theta_max_to_check[i] - const.theta_start_local))
        interpolate_density_one_arm(arm_index=4, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local, h_spiral_arm=const.h_spiral_arm, arm_angle=const.arm_angles[4], pitch_angle=const.pitch_angles[4], transverse_distances=transverse_distances, transverse_densities_initial=transverse_densities_initial)
        calc_effective_area_one_arm(arm_index=4, rho_min_spiral_arm=const.rho_min_local, rho_max_spiral_arm=rho_max_local)
        sam.calc_modelled_intensity(readfile_effective_area=True, interpolate_all_arms=False, calc_gum_cyg=False)
        intensities_total = load_modelled_data()
        reduced_chi_squared_value = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_total)
        label = f'$\\theta_{{\mathrm{{max}}}}$ = {np.round(np.degrees(theta_max_to_check[i]), 0):.0f}$^{{\circ}}$. Reduced $\\chi^2$ = {reduced_chi_squared_value:.2f}'
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_total, label=label, color=colors[i], linestyle=linestyles[i], linewidth=LINEWIDTH)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    #plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    fontsize_in_ax_text = 12
    #plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_spiral_arm} kpc & $\sigma_{{\mathrm{{A}}}}$ = {const.sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=14, color='black')
    #plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=14, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0, top=1.8e-4)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text)
    plt.savefig(filename_output)
    plt.close()


def chi_squared_with_and_without_local_arm_devoid_region():
    """ Function to calculate the chi-squared values with and without the local arm
    """
    settings.add_devoid_region_sagittarius = False
    settings.add_local_arm = False
    expanded_firas_intensity, expanded_firas_variance = load_firas_data()
    sam.calc_modelled_intensity(readfile_effective_area=False)
    intensities_no_local_or_devoid = load_modelled_data()
    reduced_chi_squared_no_local_or_devoid = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_no_local_or_devoid)
    settings.add_local_arm = True
    sam.calc_modelled_intensity(readfile_effective_area=False)
    intensities_local = load_modelled_data()
    reduced_chi_squared_value_local = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_local)
    settings.add_devoid_region_sagittarius = True
    sam.calc_modelled_intensity(readfile_effective_area=False)
    intensities_total = load_modelled_data()
    reduced_chi_squared_value_total = reduced_chi_squared(expanded_firas_intensity, expanded_firas_variance, intensities_total)
    logging.info(f"Reduced chi-squared value with no local arm or devoid region: {reduced_chi_squared_no_local_or_devoid:.2f}")
    logging.info(f"Reduced chi-squared value with local arm: {reduced_chi_squared_value_local:.2f}")
    logging.info(f"Reduced chi-squared value with local arm and devoid region: {reduced_chi_squared_value_total:.2f}")
    return


def main():
    # Important! Run src.nii_intensities.spiral_arm_model.py before running this script
    plot_test_start_angle_major_arms()
    plot_test_pitch_angle_major_arms()
    plot_test_fractional_contribution()
    plot_test_sigma_arm()
    plot_test_h_arm()
    plot_test_devoid_region_sagittarius()
    plot_test_max_angle_local_arm()
    chi_squared_with_and_without_local_arm_devoid_region()
    return


if __name__ == "__main__":
    main()
