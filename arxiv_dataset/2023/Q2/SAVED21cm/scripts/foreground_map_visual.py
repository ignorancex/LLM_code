import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import matplotlib

font = {'weight': 'normal', 'size': 10}

matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rc('font', **font)


def plot_healpy(data, min_val=None, max_val=None, hold=True, title="", cmap="jet", show=False, cbar=True, unit="",
                save=False, filename=""):
    hp.mollview(data, min=min_val, max=max_val, hold=hold, title=title, cmap=cmap, cbar=cbar, unit=unit)
    hp.graticule(color='w', dpar=30, dmer=60)
    # hp.projtext(178, 90, '0', lonlat=True)
    hp.projtext(145, 62, r'$60\degree$', lonlat=True, c="w", fontsize=9)
    hp.projtext(170, 32, r'$30\degree$', lonlat=True, c="w", fontsize=9)
    hp.projtext(176, 2, r'$0\degree$', lonlat=True, c="w", fontsize=9)
    hp.projtext(170, -58, r'$-60\degree$', lonlat=True, c="w", fontsize=9)
    hp.projtext(178, -28, r'$-30\degree$', lonlat=True, c="w", fontsize=9)
    
    hp.projtext(356, 2, r'$0\degree$', lonlat=True, c="w", fontsize=9)
    hp.projtext(56, 2, r'$60\degree$', lonlat=True, c="w", fontsize=9)
    hp.projtext(116, 2, r'$120\degree$', lonlat=True, c="w", fontsize=9)
    hp.projtext(236, 2, r'$240\degree$', lonlat=True, c="w", fontsize=9)
    hp.projtext(296, 2, r'$300\degree$', lonlat=True, c="w", fontsize=9)
    plt.text(-2.15, 0, r"$b$", c="k", rotation=90)
    plt.text(-0.02, -1.15, r"$l$", c="k")

    # hp.projtext(178, -90, 'i = 180', lonlat=True,
    #                 horizontalalignment='center')
    if show is True:
        plt.show()
    if save is True:
        plt.savefig(filename, bbox_inches="tight")


if __name__ == '__main__':
    haslam_408 = np.load('../Data/Foreground-maps/Haslam_map_dsds_smallsc_new.npy')
    plot_healpy(data=haslam_408, min_val=0, max_val=200, show=False, title='Haslam map at 408 MHz', cbar=False)
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    fig.colorbar(image, ax=ax, orientation='horizontal', aspect=50, extend='both', pad=0.03, label=r'$T\ (\rm K)$')
    plt.savefig('haslam_408.pdf', bbox_inches='tight')
