import os
import sys
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from paths import (
    figures as figures_dir,
    data as data_dir,
    scripts as script_dir
)

__all__ = ['set_rcparams', 'teff_colorbar', 'prob_colorbar', 'age_colorbar',
           'get_galex_colormap']

def set_rcparams():
    tab = Table.read(os.path.join(data_dir,'rcparams.txt'), format='csv')
    for i in range(len(tab)):
        try:
            plt.rcParams[tab['key'][i]] = float(tab['val'][i])
        except ValueError:
            plt.rcParams[tab['key'][i]] = str(tab['val'][i])
    return

def teff_colorbar():
    """ Color map for Teff values. """
    colors = Table.read(os.path.join(data_dir,'results_Z-1.txt'), format='csv',
                        delimiter='&')

    c = [i[1:-3] for i in
         colors[colors['$\\log$(g)'] == 4.5]['Hex PHX \\\\ \\hline'][3:]]

    stars = LinearSegmentedColormap.from_list('stars', c)
    return stars

def prob_colorbar():
    """ Color map for probability values. """
    blues = LinearSegmentedColormap.from_list('gradient',
                                              ['#dbeaff',
                                               '#a9bbf9', '#7593ff',
                                               '#091540'])
    return blues

def age_colorbar(short=False):
    """ Color map for age values. """
    if short:
        dat = np.load(os.path.join(data_dir, 'parula_data.npy')[:60])
    else:
        dat = np.load(os.path.join(data_dir, 'parula_data.npy'))
    parula = LinearSegmentedColormap.from_list('parula', dat)
    return parula


def get_galex_colormap(channel='fuv'):
    """ Color maps used for GALEX data. """
    clist = ['#000000', '#1f363dff', '#284751ff', '#305865ff',
             '#40798cff', '#589197ff', '#70a9a1ff', '#87b5a2ff',
             '#9ec1a3ff', '#b7d1b3ff',
             '#cfe0c3ff', '#d7e6cd']
    ccmap = LinearSegmentedColormap.from_list('rossby', clist)

    plist = ['#f6e5cbff',
             '#f7c4a5ff', '#9e7682ff',
             '#605770ff', '#4d4861ff', '#0d1321']
    pcmap = LinearSegmentedColormap.from_list('rossby', plist).reversed()

    if channel=='fuv':
        return pcmap
    else:
        return ccmap
