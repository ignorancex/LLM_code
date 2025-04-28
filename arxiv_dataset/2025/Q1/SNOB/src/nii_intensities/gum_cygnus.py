import numpy as np
import matplotlib.pyplot as plt
import src.utilities.constants as const
import src.utilities.utilities as ut
from matplotlib.ticker import AutoMinorLocator
import src.observational_data.firas_data as firas_data
from scipy.interpolate import griddata
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)


def gaussian_distribution(l, mu, sigma):
    return np.exp(-0.5 * (l - mu)**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def spherical_to_cartesian(r, l, b):
    """
    Convert spherical coordinates to Cartesian.
    r: radius
    l: galactic longitude (in radians)
    b: galactic latitude (in radians)
    """
    rho = ut.rho(r, l, b)
    theta = ut.theta(r, l, b)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    z = ut.z(r, b)
    return x, y, z


def generate_uniform_sphere(radius, center_r, center_l, center_b, resolution=250):
    """ Generate a uniform sphere of points from the point of view from Earth.

    Args:
        radius: float, radius of the sphere in kpc
        center_r: float, distance from the Sun to the center of the sphere in kpc
        center_l: float, galactic longitude of the center of the sphere in radians
        center_b: float, galactic latitude of the center of the sphere in radians
        resolution: int, number of points in each dimension
    Returns:
        x_sphere, y_sphere, z_sphere: arrays, Cartesian coordinates of the points in the sphere
        density: array, density of the points in the sphere
    """
    # Define the range for r, l, and b based on the sphere's radius and center location
    # Create grids of points in spherical coordinates
    max_angular_extent = np.arctan2(radius, center_r) # assuming a perfect sphere
    r_min = center_r - radius
    r_max = center_r + radius
    l_min = center_l - max_angular_extent
    l_max = center_l + max_angular_extent
    b_min = center_b - max_angular_extent
    b_max = center_b + max_angular_extent
    r, r_step = np.linspace(r_min, r_max, resolution, retstep=True)
    l, l_step = np.linspace(l_min, l_max, resolution, retstep=True)
    b, b_step = np.linspace(b_min, b_max, resolution, retstep=True)
    r_grid, l_grid, b_grid = np.meshgrid(r, l, b, indexing='ij')
    # Convert grid points to Cartesian coordinates for distance calculation
    x_grid, y_grid, z_grid = spherical_to_cartesian(r_grid, l_grid, b_grid)
    # Calculate center Cartesian coordinates
    center_x, center_y, center_z = spherical_to_cartesian(center_r, center_l, center_b)
    # Compute the distance of each point from the center
    distances = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2 + (z_grid - center_z)**2)
    # Select points within the sphere's radius
    within_sphere = distances <= radius
    outside_sphere = distances > radius
    # Filter the points that are within the sphere
    r_sphere = r_grid[within_sphere]
    l_sphere = l_grid[within_sphere]
    b_sphere = b_grid[within_sphere]
    b_min, b_max = np.min(b_sphere), np.max(b_sphere)
    delta_b = b_max - b_min
    density = np.ones_like(r_grid) * r_step * b_step / (4 * np.pi * const.kpc**2 * np.radians(5) * delta_b)
    density[outside_sphere] = 0
    density = density.sum(axis=(0, 2))  # Sum up the array to have the longitude as the first axis
    return r_sphere, l, b_sphere, density


def intensity(longitudes, densities, central_long, window_size=5):
    dl = 0.1 # Keep this value at 0.1 in order for the resulting intensities to be able to be plotted together with the spiral arms
    bins = np.arange(0, 360 + dl, dl)
    central_bins = bins[:-1] + dl / 2
    central_bins = ut.rearange_data(central_bins)[:-1] / 2
    binned_intensity, bin_edges = np.histogram(np.degrees(longitudes), bins=bins, weights=densities)
    num_intensities_per_bin, _ = np.histogram(np.degrees(longitudes), bins=bins)
    num_intensities_per_bin[num_intensities_per_bin == 0] = 1
    binned_intensity /= num_intensities_per_bin
    rearanged_intensity = ut.rearange_data(binned_intensity)[:-1]
    window_size = window_size / dl
    rearanged_intensity = ut.running_average(rearanged_intensity, window_size) / window_size 
    fwhm = np.radians(7)
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian_density_weights = gaussian_distribution(np.radians(central_bins), central_long, std)
    rearanged_intensity *= gaussian_density_weights
    return rearanged_intensity


def calc_effective_area(radius):
    logging.info('Calculating the effective area of the Gum Nebula and Cygnus Loop')
    d_x = 70 / 3000 # distance between each interpolated point in the x direction. 70 kpc is the diameter of the Milky Way, 1000 is the number of points
    d_y = 70 / 3000 # distance between each interpolated point in the y direction. 70 kpc is the diameter of the Milky Way, 1000 is the number of points
    grid_x, grid_y = np.mgrid[-35:35:3000j, -35:35:3000j]
    theta = np.linspace(0, 2 * np.pi, 100)
    radial_points = np.linspace(0, radius, 100)
    theta_grid, radial_grid = np.meshgrid(theta, radial_points, indexing='ij')
    x = radial_grid * np.cos(theta_grid)
    y = radial_grid * np.sin(theta_grid)
    density = np.ones_like(x)
    interpolated_density = griddata((x.flatten(), y.flatten()), density.flatten(), (grid_x, grid_y), method='cubic', fill_value=0)
    interpolated_density[interpolated_density < 0] = 0 # set all negative values to 0
    effective_area = np.sum(interpolated_density) * d_x * d_y
    return effective_area


def cygnus():
    # check if the folder exists, if not create it
    Path(const.FOLDER_GALAXY_DATA).mkdir(parents=True, exist_ok=True)
    cygnus_distance = 1.45 # kpc
    cygnus_long = np.radians(80)
    cygnus_lat = np.radians(0)
    cygnus_radius = 0.075 # kpc
    effective_area_cyg = calc_effective_area(cygnus_radius)
    max_angular_extent = np.arctan2(cygnus_radius, cygnus_distance) * 2 # assuming a perfect sphere
    print(f'Cygnus max angular extent: {np.degrees(max_angular_extent)}')
    r_cygnus, l_cygnus, b_cygnus, density = generate_uniform_sphere(cygnus_radius, cygnus_distance, cygnus_long, cygnus_lat)
    intensity_cyg = intensity(l_cygnus, density * const.cygnus_nii_luminosity, cygnus_long, window_size=5) / effective_area_cyg
    np.save(f'{const.FOLDER_GALAXY_DATA}/intensities_cygnus.npy', intensity_cyg)
    return r_cygnus, l_cygnus, b_cygnus, intensity_cyg


def gum():
    # check if the folder exists, if not create it
    Path(const.FOLDER_GALAXY_DATA).mkdir(parents=True, exist_ok=True)
    # Gum parameters
    gum_distance = 0.33 # kpc
    gum_long = np.radians(262)
    gum_lat = np.radians(0)
    gum_radius = 0.030 # kpc
    effective_area_gum = calc_effective_area(gum_radius)
    max_angular_extent = np.arctan2(gum_radius, gum_distance) * 2 # assuming a perfect sphere
    print(f'Gum max angular extent: {np.degrees(max_angular_extent)}')
    r_gum, l_gum, b_gum, density = generate_uniform_sphere(gum_radius, gum_distance, gum_long, gum_lat)
    intensity_gum = intensity(l_gum, density * const.gum_nii_luminosity, gum_long) / effective_area_gum
    np.save(f'{const.FOLDER_GALAXY_DATA}/intensities_gum.npy', intensity_gum)
    return r_gum, l_gum, b_gum, intensity_gum


def generate_gum_cygnus():
    gum()
    cygnus()


def plot_modelled_intensity_gum_cygnus(filename_output = f'{const.FOLDER_MODELS_GALAXY}/gym_cygnus_test.pdf'):
    # plot the FIRAS data for the NII 205 micron line
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # gum, cygnus data
    _, _, _, cyg_intensity = cygnus()
    _, _, _, gum_intensity = gum()
    total_intensity = cyg_intensity + gum_intensity
    plt.plot(np.linspace(0, 360, len(total_intensity)), total_intensity, label='Local OBA')
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude $l$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)", fontsize=14)
    fontsize_in_ax_text = 12
    plt.text(0.02, 0.95, fr'$H_\rho$ = {const.h_spiral_arm} kpc & $\sigma_{{\mathrm{{A}}}}$ = {const.sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=fontsize_in_ax_text, color='black')
    plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=fontsize_in_ax_text, color='black')
    plt.xlim(0, 360)
    plt.ylim(bottom=0)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = fontsize_in_ax_text)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_modelled_intensity_gum_cygnus()
