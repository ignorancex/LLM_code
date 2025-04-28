import numpy as np

def acc(phi_2d, dx=1):
    '''
    Calculating the acceleration a = \nabla(phi).

    Parameters: phi_2d: the 2D gravitational potential array
                dx    : Array pixel length in unit of cm

    Returns:    acc   : the acceleration map array
    '''

    ay, ax = np.gradient(phi_2d, dx)

    ax2 = np.multiply(ax, ax)
    ay2 = np.multiply(ay, ay)
    acc = np.sqrt(ax2 + ay2)

    return acc




if __name__ == "__main__":
    from astropy.io import fits
    import sys

    fname = sys.argv[1]
    hdu = fits.open(fname)
    phi_2d = hdu[0].data

    acc = acc(phi_2d, 0.5e17)

    fits.writeto(sys.argv[1].replace('.fits', '.acc.fits'), acc, overwrite=True)

