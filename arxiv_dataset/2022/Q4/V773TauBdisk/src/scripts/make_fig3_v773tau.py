import paths
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#from matplotlib import rc
#rc('text', usetex=True)

def V773Tau_nocoro(pic_7):

    colorscheme=plt.cm.gist_heat

    ano_font_size = 14
    tick_font_size =10

    scidata7 = fits.getdata(pic_7, ext=0)

    fig, ax7 = plt.subplots(1,1,figsize=(4,4))

    imnorm = scidata7 / np.max(scidata7)

    # clip to 1e-4 of peak value in image
    logim = np.ones_like(imnorm) * 1e-4
    logim[(imnorm>1e-4)] = imnorm[(imnorm>1e-4)]
    
    plt.imshow(np.log10(logim), origin='lower',
               extent=[-6.27027,6.27027,-6.27027,6.27027],
               cmap=plt.cm.gist_heat, vmin=-3, vmax=0,
               interpolation="nearest")

    ax7.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax7.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    ax7.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax7.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    ax7.tick_params(axis='both', labelsize=tick_font_size,
                    direction='in',
                    color='white')

    ax7.tick_params(which='major',
                    length=6)

    ax7.tick_params(which='minor',
                    length=3,
                    direction='in',
                    color='white')

    ax7.set_xlim([-0.5,0.5])
    ax7.set_ylim([-0.5,0.5])

    ax7.set_xlabel("$\mathbf{\Delta RA  \ [arcsec]}$", fontsize=12)
    ax7.set_ylabel("$\mathbf{\Delta Dec  \ [arcsec]}$", fontsize=12)

    ax7.yaxis.set_ticks_position('both')
    ax7.xaxis.set_ticks_position('both')

    ax7.annotate("$\mathbf{V\,773\,Tau}$", (0.05, 0.95),
                 xycoords="axes fraction", va="top", ha="left",
                 fontsize=ano_font_size, color="white")

    fig.savefig(paths.figures / 'fig3_v773tau.pdf', dpi=300, bbox_inches='tight')

    return

V773Tau_nocoro(paths.data / "V773_SPHERE_median.fits")
