import numpy as np

def phi_2d(data, dx=1, dy=1, H=0):
    '''
    Calculating gravitational potential of a 2D surface density map using Fast Fourier method (FFT)

    Parameters: data: Surface density array in unit of g/cm^2
                dx  : Array pixel length in unit of cm
                dy  : Array pixel width in unit of cm
                H   : Half-thickness of density layer in unit of cm

    Returns:    phi : Gravitational potential array of the surface density
    '''

    #physical constants
    G = 6.6743E-8
    pi = 3.141592653589793

    #Padding for periodic boundary condition
    ny, nx = data.shape[0]*2, data.shape[1]*2 
    sigma = np.zeros((ny, nx))
    sigma[0:data.shape[0], 0:data.shape[1]] = np.nan_to_num(data)

    sigma_k = np.fft.fft2(sigma)

    kx = np.fft.fftfreq(nx, dx/2/pi)
    kx = np.tile(kx, ny)
    kx = np.reshape(kx, (ny,nx))
    kx2 = kx*kx

    ky = np.fft.fftfreq(ny, dy/2/pi)
    ky = np.tile(ky, nx)
    ky = np.reshape(ky, (nx,ny))
    ky = ky.transpose(1,0)
    ky2 = ky*ky

    karray = np.sqrt(kx2 + ky2)
    karray[0][0] = (karray.min() + karray.max())/2

    #Poisson equation in Fourier space
    phi_k = -2*pi*G*sigma_k / (np.absolute(karray)*(1 + np.absolute(karray*H)))

    phi_k[0][0] = 0
    phi = np.fft.ifft2(phi_k)
    phi = phi[0:data.shape[0], 0:data.shape[1]].real

    return phi

if __name__ == "__main__":
    from astropy.io import fits
    import sys

    fname = sys.argv[1]
    hdu = fits.open(fname)
    data = hdu[0].data

    phi = phi_2d(data, 0.5e17, 0.5e17, 0)

    f = fits.PrimaryHDU(phi)
    f.writeto(sys.argv[1].replace('.fits', '.phi2d.fits'), overwrite=True)