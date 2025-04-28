import numpy as np
import matplotlib.pyplot as plt
import obs_utilities as obs_ut
import src.utilities.constants as const
from pathlib import Path

CATALOGUE = 'J/ApJS/165/338'
TABLE = 'table1'
FONTSIZE = 12

def plot_rrl_hii_data():
    """ Plot the data of HII regions from the VizieR catalogue J/ApJS/165/338
    Source paper for data: https://ui.adsabs.harvard.edu/abs/2006ApJS..165..338Q/abstract
    
    Returns:
        2D np.array with the data. The first column contains the Galactic longitude, the second column contains the Galactic latitude

    """
    # check if the output folder exists, if not create it
    Path(const.FOLDER_OBSERVATIONAL_PLOTS).mkdir(parents=True, exist_ok=True)
    tap_records = obs_ut.get_catalogue_data(CATALOGUE, TABLE, ['GLON', 'GLAT'])
    glat_data = tap_records['GLAT'].data
    glon_data = tap_records['GLON'].data
    bin_edges = np.arange(-4, 4+0.5, 0.5)
    binned_data, bin_edges = np.histogram(glat_data, bins=bin_edges)
    plt.bar(bin_edges[:-1], binned_data, width=0.5, align='edge')
    plt.xlabel('Galactic latitude $b$ (degrees)', fontsize=FONTSIZE)
    plt.ylabel('Frequency', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    #plt.title(f'Histogram of Galactic HII regions. Data from Quireza et al. (2006)\n{np.sum(binned_data)} HII regions in total')
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/HII_regions_histogram.pdf')
    print("Number of points with glat < abs(1): ", len(glat_data[np.abs(glat_data) < 1]))
    print("Number of points with glat > abs(4) in the original dataset:", len(glat_data[np.abs(glat_data) > 4]))
    print("Number of points with exactly glat = +1 or -1: ", len(glat_data[np.abs(glat_data) == 1]))
    print(glat_data[np.abs(glat_data) > 4])


if __name__ == "__main__":
    plot_rrl_hii_data()