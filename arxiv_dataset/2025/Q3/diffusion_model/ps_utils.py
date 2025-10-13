import numpy as np
from scipy.fftpack import fft2, ifft2
import numpy as np, sys, os, warnings

def apply_maxmin_normalization(maps):
    min_val = np.nanmin(maps)
    max_val = np.nanmax(maps)
    return (maps - min_val) / (max_val - min_val) 
    
def load_all_moments(filename, bandpass_centers, max_lines=-1):
    moments_data = np.load(filename)[:max_lines]
    moments = {}

    norms = [
        bandpass_centers,        # S2aa
        bandpass_centers,        # S2bb
        bandpass_centers,        # S2ab
        bandpass_centers,        # S3aaa
        bandpass_centers,        # S3bbb
        bandpass_centers,        # S3aab
        bandpass_centers,        # S3abb
        bandpass_centers**2,     # S4aaaa
        bandpass_centers**2,     # S4bbbb
        bandpass_centers**2,     # S4aaab
        bandpass_centers**2,     # S4aabb
        bandpass_centers**2      # S4abbb
    ]

    for i in range(12):
        label = f"moment_{i:02d}"
        moments[label] = [m / norms[i] for m in moments_data[:, :, i]]

    return moments
    
def get_peak_masks(tmap, mask_threshold_sigma_units = 10, mask_radius_pixel_units = 0, perform_apod = 1, mask_shape = 'circle', taper_radius_fac = 2. ):
    
    """
    Get masking using sigma clipping.

    Parameters
    ----------
    tmap: array
        flatsky map.
    mask_threshold_sigma_units: float
        threshold for sigma clipping.
        mask = np.where( (abs(tmap) > np.mean(tmap) + sigma * np.std(tmap)) )
    mask_radius_pixel_units: float
        punch bigger hole around each peak.
        Default is 0, in which case, only peak pixels satisfying the above condition are masked.
    perform_apod: boolean
        Apodise the binary mask.
        Default is True.
    mask_shape: str
        circle or a square mask.
        default is circle.
        Must be on of ['circle', 'square']
    taper_radius_fac: float
        radius for apodisation.
        taper_radius = taper_radius_fac * mask_radius.
        default is 2 (FIX ME: which works well but need to be explored further).

    Returns
    -------
    peak_mask: array, shape is tmap.shape.
        binary peak mask.
    mask: array shape is tmap.shape.
        binary or apodised mask with some finite radius around the peaks.
        Only computed when mask_radius_pixel_units>0.
        else, mask_radius_pixel_units == peak_mask
    """

    import scipy as sc
    import scipy.ndimage as ndimage

    assert mask_radius_pixel_units >=0

    peak_mask = np.ones( tmap.shape )
    inds = np.where( (abs(tmap) > abs(np.mean(tmap)) + mask_threshold_sigma_units * np.std(tmap)) )
    peak_mask[inds] = 0.


    if mask_radius_pixel_units > 0:
        x_grid, y_grid = np.indices( tmap.shape )

        assert mask_shape in ['circle', 'square']

        mask = np.ones( y_grid.shape )

        #imshow(mask); colorbar(); 
        #pick all the masked inds
        for (i,j) in zip(inds[0], inds[1]):

            y, x = x_grid[j, i], y_grid[j, i]
            
            #plot( y, x, 'ko', color = 'None', mec = 'black', ms = 10, mew = 1.)

            if mask_shape == 'circle':
                radius = np.sqrt( ((x-x_grid)**2. + (y-y_grid)**2.) )
                inds_to_mask = np.where((radius<=mask_radius_pixel_units))
            elif mask_shape == 'square':
                inds_to_mask = np.where( (abs(x-x_grid)<mask_radius_pixel_units) & (abs(y-y_grid)<mask_radius_pixel_units))

            mask[inds_to_mask[0], inds_to_mask[1]] = 0.
        #show(); sys.exit()

        taper_radius = mask_radius_pixel_units * taper_radius_fac #
        if perform_apod:
            ker=np.hanning(taper_radius)
            ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )
            mask=ndimage.convolve(mask, ker2d)
            mask/=mask.max()
    else:
        mask = peak_mask

    return peak_mask, mask


def boundary_apod_mask(x_grid, y_grid, mask_radius, perform_apod = True, mask_shape = 'circle', taper_radius_fac = 6.):

    """
    Interpolating a 1d power spectrum (cl) defined on multipoles (el) to 2D assuming azimuthal symmetry (i.e:) isotropy.

    Parameters
    ----------
    x_grid: array
        x grid of the map (like ra).
    y_grid: array
        y grid of the map (like dec).
    mask_radius: float
        mask radius in same units as x_grid and y_grid
    perform_apod: boolean
        Apodise the binary mask.
        Default is True.
    mask_shape: str
        circle or a square mask.
        default is circle.
        Must be on of ['circle', 'square']
    taper_radius_fac: float
        radius for apodisation.
        taper_radius = taper_radius_fac * mask_radius.
        default is 6 (FIX ME: which works well but need to be explored further).

    Returns
    -------
    mask: array, shape is x_grid.shape.
        binary or apodised mask.
    """

    assert mask_shape in ['circle', 'square']

    import scipy as sc
    import scipy.ndimage as ndimage

    mask = np.ones( y_grid.shape )

    if mask_shape == 'circle':
        radius = np.sqrt( (x_grid**2. + y_grid**2.) )
        inds_to_mask = np.where((radius<=mask_radius))
    elif mask_shape == 'sqaure':
        inds_to_mask = np.where( (abs(x_grid)<mask_radius) & (abs(y_grid)<mask_radius))

    mask[inds_to_mask[0], inds_to_mask[1]] = 0.

    taper_radius = mask_radius * taper_radius_fac #
    if perform_apod:
        ##imshow(mask); colorbar(); show(); sys.exit()
        ker=np.hanning(taper_radius)
        ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )
        mask=ndimage.convolve(mask, ker2d)
        mask/=mask.max()

    return mask

def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2d_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 

    return cl2d
def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None, minbin = 100, maxbin = 10000,):

    """
    map2cl module - get the power spectra of map/maps

    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    flatskymap1: map1 with dimensions (ny, nx)
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra

    binsize: el bins. computed automatically if None

    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2

    output:
    auto/cross power spectra: [el, cl, cl_err]
    """

    nx, ny, dx, dy = flatskymapparams
    dx_rad = np.radians(dx/60.)
    dy_rad = np.radians(dy/60.)

    lx, ly = get_lxly(flatskymapparams)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
            flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)

    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = minbin, maxbin = maxbin, to_arcmins = 0)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    return el, cl

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    lx, ly
    """

    nx, ny, dx, dy = flatskymapparams
    dx = np.radians(dx/60.)
    dy = np.radians(dy/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dy ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

def radial_profile(z, xy = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = 1):

    """
    get the radial profile of an image (both real and fourier space)
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(z.shape)
    else:
        x, y = xy

    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    if to_arcmins: radius *= 60.

    binarr=np.arange(minbin,maxbin,bin_size)
    radprf=np.zeros((len(binarr),3))

    hit_count=[]

    for b,bin in enumerate(binarr):
        ind=np.where((radius>=bin) & (radius<bin+bin_size))
        radprf[b,0]=(bin+bin_size/2.)
        hits = len(np.where(abs(z[ind])>0.)[0])

        if hits>0:
            radprf[b,1]=np.sum(z[ind])/hits
            radprf[b,2]=np.std(z[ind])
        hit_count.append(hits)

    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    errval=std_mean/(hit_count)**0.5
    radprf[:,2]=errval

    return radprf

def cl2map(flatskymapparams, cl, el = None):

    """
    cl2map module - creates a flat sky map based on the flatskymap parameters and the input power spectra.
    Look into make_gaussian_realisation for a more general code. 

    Parameters
    ----------
    flatskymyapparams: list
        [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    cl: array
        1d vector of Cl power spectra: temp / pol. power spectra

    el: array (optional)
        Multipole over which the signal / noise spectra are defined.
        Default is None and el will be np.arange( len(cl_signal) )

    Returns
    -------
    flatskymap: array
        flatskymap with the given underlying power spectrum cl.

    See Also
    -------
    make_gaussian_realisation
    """

    if el is None: el = np.arange(len(cl))

    nx, ny, dx, dx = flatskymapparams

    #get 2D cl
    cl2d = cl_to_cl2d(el, cl, flatskymapparams) 

    #pixel area normalisation
    dx_rad = np.radians(dx/60.)
    pix_area_norm = np.sqrt(1./ (dx_rad**2.))
    cl2d_sqrt_normed = np.sqrt(cl2d) * pix_area_norm

    #make a random Gaussian realisation now
    gauss_reals = np.random.randn(nx,ny)
    
    #convolve with the power spectra
    flatskymap = np.fft.ifft2( np.fft.fft2(gauss_reals) * cl2d_sqrt_normed).real
    flatskymap = flatskymap - np.mean(flatskymap)

    return flatskymap    