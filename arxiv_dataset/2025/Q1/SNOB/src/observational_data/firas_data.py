import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
import logging
logging.basicConfig(level=logging.INFO) 
from src.utilities import utilities as ut
import src.utilities.constants as const
from pathlib import Path

def calc_hist_1d(data):
    """ Calculate the 1D histogram of the N+ line intensity vs Galactic longitude based on data shared by Fixsen. The data is given in the form of a histogram.
    
    Args:
        data: 2D np.array with the data. The first column contains the Galactic longitude, the second column contains the Galactic latitude, and the third column contains the intensity of the N+ line in units of MJy/sr.
        
    Returns:
        1D np.array with the histogram of the N+ line intensity vs Galactic longitude
    """
    # 1D binning of data
    long = data[:, 0]
    intensity = data[:, 2] # units of MJy/sr
    # set negative values in intensity to zero
    intensity[intensity < 0] = 0
    intensity *= 1e6 * 1e-26 * 1e9 * 1.463*1e12 # convert to nW/m^2/str. 1.463e12 is the frequency of the N+ line in Hertz
    intensity *= 1e-4 * 1e-9 *1e7  # convert from nW/m^2/str to erg/s/cm²/sr.
    bin_edges_long = np.arange(0, 362.5, 2.5) # will end at 360. No data is left out. 145 bin edges
    hist, _ = np.histogram(long, bins=bin_edges_long, weights=intensity) # if a longitude is in the bin, add the intensity to the bin
    hist_num_long_per_bin, _ = np.histogram(long, bins=bin_edges_long)
    # Rearange data to be plotted in desired format
    rearanged_hist = ut.rearange_data(hist)
    rearanged_hist_num_long_per_bin = ut.rearange_data(hist_num_long_per_bin)
    rearanged_hist_num_long_per_bin[rearanged_hist_num_long_per_bin == 0] = 1
    hist = rearanged_hist / rearanged_hist_num_long_per_bin
    return hist


def plot_data_from_fixsen():
    """ Plot the N+ line intensity vs Galactic longitude based on data shared by Fixsen. The data is given in the form of a histogram.

    Returns:
        None. Saves the plot.
    """
    # check if the output folder exists, if not create it
    Path(const.FOLDER_OBSERVATIONAL_PLOTS).mkdir(parents=True, exist_ok=True)
    # Load data from the text file
    data = np.loadtxt(f'{const.FOLDER_OBSERVATIONAL_DATA}/N+.txt')
    data = data[np.abs(data[:, 1]) <= 5]    
    hist = calc_hist_1d(data)
    # Create bin_edges
    bin_edges_central = np.arange(2.5, 360, 5)
    bin_edges = np.concatenate(([0], bin_edges_central, [360]))
    # Plot using stairs
    plt.figure(figsize=(10, 6))
    plt.stairs(values=hist, edges=bin_edges, fill=False, color='black')
    # Set up the plot
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # use scientific notation for the y-axis
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3))
    plt.xlabel('Galactic longitude (degrees)')
    plt.xlim(0, 360)
    plt.ylabel("Line intensity in erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$")
    plt.title("Estimated NII intensity of the Galactic disk based on data shared by Fixsen")
    # Save the plot
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/firas_estimate_from_fixsen.pdf')
    plt.close()


def create_bin_edges_from_central_values(central_values, bin_half_width):
    """ Create bin edges from the central values of the bins. The bin edges are used for plotting the histogram. Function used for the FIRAS data (.fits).
    
    Args:
        central_values: 1D np.array with the central values of the bins
        bin_half_width: float. The half-width of the bins
        
    Returns:
        1D np.array with the bin edges
    """
    first_edge = central_values[0]
    last_edge = central_values[-1]
    bin_edges = []
    for i in range(len(central_values) - 1):
            bin_edges.append(central_values[i + 1] - bin_half_width)
    bin_edges = np.concatenate(([first_edge], bin_edges, [last_edge]))
    return bin_edges


def firas_data_for_plotting():
    """ Extract the FIRAS data from the .fits file for the N+ line and prepare it for plotting. The data is given in the form of a histogram, with the bin edges and the bin centres.

    Returns:
        bin_edges_line_flux: 1D np.array with the bin edges for the line flux
        bin_centre_line_flux: 1D np.array with the bin centres for the line flux
        line_flux: 1D np.array with the line flux
        line_flux_error: 1D np.array with the error in the line flux
    """
    logging.info("Extracting the FIRAS data for the N+ line")
    fits_file = fits.open(f'{const.FOLDER_OBSERVATIONAL_DATA}/lambda_firas_lines_1999_galplane.fits') # .fits file retrieved from NASA's website on the FIRAS instrument
    #fits_file.info()
    # grab the data from the 12th HDU
    data_hdu = fits_file[12] 
    data = data_hdu.data
    # extract the data
    line_id = data['LINE_ID'] # plot this to see we are looking at the right line
    gal_lon = data['GAL_LON'][0] + 180 # add 180 to get only positive values, from 0 to 360. Contains the central values of the bins
    bin_edges_line_flux = create_bin_edges_from_central_values(gal_lon, bin_half_width=2.5) # create bin-edges for the plotting. bin_half_width is 2.5 degrees
    bin_centre_line_flux = np.concatenate(([gal_lon[0] + 2.5/2], gal_lon[1:-1], [gal_lon[-1] - 2.5/2])) # create bin-centres for the error-plotting
    line_flux = data['LINE_FLUX'][0] *1e-4 * 1e-9 *1e7  # convert from nW/m^2/str to erg/s/cm²/sr.
    line_flux_error = data['LINE_FLERR'][0] * 1e-4 * 1e-9 * 1e7 * 2  # convert from nW/m^2/str to erg/s/cm²/sr. Multiply by 2 to get 2-sigma error
    return bin_edges_line_flux, bin_centre_line_flux, line_flux[::-1], line_flux_error[::-1] # reverse the order of the data to match the longitudes


def plot_firas_nii_line():
    """ Plot the observed NII intensity of the Galactic disk from the FIRAS data. The data is given in the form of a histogram, together with 2-sigma error bars.
    
    Returns:
        None. Saves the plot.
    """
    # check if the output folder exists, if not create it
    Path(const.FOLDER_OBSERVATIONAL_PLOTS).mkdir(parents=True, exist_ok=True)
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data_for_plotting() # retrieve the data from the .fits file
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black') # plot the observed intensity
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1) # plot the error bars
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) # add minor ticks
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # use scientific notation for the y-axis
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Line intensity in erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$")
    plt.title("Observed NII intensity of the Galactic disk")
    plt.xlim(0, 360)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/firas_data_NII_line.pdf')
    plt.close()


def add_firas_data_to_plot(ax):
    """ Add the FIRAS data to an existing plot. ax's x-range should be 0 to 360. Used for adding observational data to the model plots.

    Args:
        ax: matplotlib axis object

    Returns:
        None
    """
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data_for_plotting()
    ax.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    ax.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)


def find_firas_intensity_at_central_long(long):
    """ Find the intensity of the N+ line at a given longitude in the FIRAS data. The intensity is given in erg/s/cm²/sr. Used for normalisation of the axisymmetric model. 

    Args:
        long: int, the longitude in degrees. Valid values are in the range -180 to 180, with increments in 5 degrees.

    Returns:
        float, the intensity of the N+ line at the given longitude.
    """
    try:
        fits_file = fits.open(f'{const.FOLDER_OBSERVATIONAL_DATA}/lambda_firas_lines_1999_galplane.fits')
    except:
        logging.error("The .fits file could not be opened. Check if the file is in the correct folder.")
        return
    data_hdu = fits_file[12] 
    data = data_hdu.data
    gal_lon = data['GAL_LON'][0] 
    line_flux = data['LINE_FLUX'][0] *1e-4 * 1e-9 *1e7  # convert from nW/m^2/str to erg/s/cm²/sr.
    if long not in gal_lon:
        print("The longitude is not in the data. Valid values are in the range -180 to 180, with increments in 5 degrees.")
        return
    index = np.where(gal_lon == long)
    logging.info(f'The intensity at longitude {long} degrees is {line_flux[index][0]} erg/s/cm²/sr.')
    return line_flux[index][0]


def main():
    plot_firas_nii_line()
    plot_data_from_fixsen()
    find_firas_intensity_at_central_long(30)


if __name__ == "__main__":
    main()
