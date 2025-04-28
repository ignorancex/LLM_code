import sys
sys.path.append('../')
from setup import *
from astropy.wcs import WCS
from astropy.io import fits
import pyspeckit
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk, opening
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import argparse


def filtersolutions(parcube, errcube, npeaks, rmsmap=None, snratio=None, 
                    velinit=-np.inf, velend=np.inf, filter_negative=False, 
                    errorfrac=None, eps=1.e-6, filter_islands=False, 
                    minsizetrim=50, chansize=0.08, masked_chans=False, 
                    velinit_mask=None, velend_mask=None, widthfilter=False):
    """
    Replace the pixels in the fitted cube with np.nan where the fit is not
    good enough according to our criteria.

    The criteria that a pixel must have are:
    - The errors are not zero (less than eps)
    - The peak must not be negative in case filter_negatives is true
    - The error fraction is lower than errorfrac, if given
    - The central velocity value must be within the range [velinit,velend]
    - The peak value must be larger than rms times snratio, if given
    - If one parameter in a spectra is np.nan, all the spectra must be nan (sanity
    check)
    - NEW 10.5.22: the uncertainty in the central velocity must be smaller than the channel width
    - NEW 11.5.22: if we masked certain channels, the central velocity of the components must not be within those channels

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """
    # we first create all the masks we need
    
    # all errors must be non zero
    # note that eps must be larger than the velocity dispersion
    zeromask = np.zeros(np.shape(parcube[0]), dtype=int) # we need to do a plane mask
    for i in range(3*npeaks):
        zeromask += np.where(np.abs(errcube[i])<eps, 1, 0)
        
    # if a fraction is given, make sure all values have an error fraction less than that
    errormask = np.zeros(np.shape(parcube[0]), dtype=int)
    if errorfrac is not None:
        for i in range(3*npeaks):
            errormask += np.where(np.abs(errcube[i]/parcube[i]) > errorfrac, 1, 0)
            
    # if indicated, the velocity width must not be larger than the selection range
    widthmask = np.zeros(np.shape(zeromask), dtype=int)
    if widthfilter:
        for i in range(npeaks):
            widthmask += np.where(parcube[2+3*i] > (velend-velinit)/2.35, 1, 0)
        
    
    # if indicated, all values must be non-negative
    negativemask = np.zeros(np.shape(zeromask), dtype=int)
    if filter_negative:
        for i in range(3*npeaks):
            negativemask += np.where(parcube[i] < 0, 1, 0)
    
    # velocities must be within range
    velocitymask = np.zeros(np.shape(zeromask), dtype=int)
    for i in range(npeaks):
        velocitymask += np.where(parcube[1+3*i] < velinit, 1, 0) + \
        np.where(parcube[1+3*i] > velend, 1, 0)
    
    # the amplitude of the Gaussian must be above the snratio indicated
    peakmask = np.zeros(np.shape(zeromask), dtype=int)
    if snratio is not None and rmsmap is not None:
        for i in range(npeaks):
            snrmappeak = parcube[3*i] / rmsmap
            peakmask += np.where(snrmappeak < snratio, 1, 0)
    
    # all values of parameters and uncertainties must be not NaN
    nanmask = np.zeros(np.shape(zeromask), dtype=int)
    for i in range(3*npeaks):
        nanmask += np.where(np.isnan(parcube[i]), 1, 0) + np.where(np.isnan(errcube[i]), 1, 0)
    
    # central velocity uncertainty must be lower than
    centraluncertaintymask = np.zeros(np.shape(zeromask), dtype=int)
    for i in range(npeaks):
        velocitymask += np.where(errcube[1+3*i] > chansize, 1, 0)
        
    # if indicated, the central velocities must not be within masked channels
    maskedchansmask = np.zeros(np.shape(zeromask), dtype=int)
    if masked_chans:
        for i in range(npeaks):
            maskedchansmask += np.where(parcube[1+3*i] > velinit_mask, 1, 0) * np.where(parcube[1+3*i] < velend_mask, 1, 0)
    
    finalmask = zeromask + errormask + negativemask + widthmask + velocitymask + peakmask + nanmask + centraluncertaintymask + maskedchansmask
    
    parcubenew = parcube.copy()
    errcubenew = errcube.copy()
    parcubenew[np.where(np.repeat([finalmask], 3*npeaks, axis=0))] = np.nan
    errcubenew[np.where(np.repeat([finalmask], 3*npeaks, axis=0))] = np.nan   
    
    # eliminate isolated small islands of emission after the filter
    if filter_islands:        
        planemask = ~np.isnan(parcubenew[0])
        planemask = remove_small_objects(planemask, min_size=minsizetrim)
        smallmask = np.ones(np.shape(planemask), dtype=int) - planemask
        parcubenew[np.where(np.repeat([smallmask], 3*npeaks, axis=0))] = np.nan
        errcubenew[np.where(np.repeat([smallmask], 3*npeaks, axis=0))] = np.nan
    
    return parcubenew, errcubenew


def interpolatesolutions(solfilein, npeaks, mask=None):
    '''
    The solfilein must be a .fits file that contains one parameter per 
    plane and then one parameter uncertainty per plane.
    The shape must be [nplane, yy, xx]
    The mask must be 2 dimensional
    '''
    solcube = fits.getdata(solfilein)[:3*npeaks]
    if np.any(np.isnan(solcube)):
        solcube[np.where(np.isnan(solcube))] = 0
    solcubeshape = np.shape(solcube)
    yy, xx = np.indices(solcubeshape[1:])
    filledcube = solcube.copy()
    headersolcube = fits.getheader(solfilein)

    for i, plane in enumerate(solcube):
        indexknown = np.where(plane<1e-5, False, True)
        filledcube[i][~indexknown] = griddata((xx[indexknown], yy[indexknown]),
                                                  plane[indexknown],
                                                  (xx[~indexknown], yy[~indexknown])
                                                 )
        if mask is not None:
            filledcube[i][np.where(mask==0)] = np.nan
    return filledcube, headersolcube


def main(ngauss, snratio, minsize, initguesses=None, err_tol=0.5, starting_point=None, overwrite=False, verbose=False, filter_before_refit=True):
    imagefile = hc3n_10_9_cube + '.fits'
    # maskfile = fitdir + 'HC3N_10_9_mask'

    fitfile =  fitfilebase.format(ngauss) + '.fits' #fitdir + 'HC3N_10_9_{}G_fitparams.fits'.format(ngauss)
    fitfilefiltered = fitfilebase.format(ngauss) + '_filtered.fits' # fitdir + 'HC3N_10_9_{}G_fitparams_filtered.fits'.format(ngauss)
    newguessfile = fitfilebase.format(ngauss) + '_filtered_guesses.fits' # fitdir + 'HC3N_10_9_{}G_fitparams_filtered_guesses.fits'.format(ngauss)
    fitfile2 = fitfilebase.format(ngauss) + '_2.fits'  # fitdir + 'HC3N_10_9_{}G_fitparams_2.fits'.format(ngauss)
    fitfile2filtered = fitfilebase.format(ngauss) + '_2_filtered.fits' # fitdir + 'HC3N_10_9_{}G_fitparams_2_filtered.fits'.format(ngauss)
    if ngauss == 1:
        amp1_1Gfile = fitfilebase.format(ngauss) + '_amp1.fits'
        amp1_e_1Gfile = fitfilebase.format(ngauss) + '_eamp1.fits'
        v1_1Gfile = fitfilebase.format(ngauss) + '_vel1.fits'
        v1_e_1Gfile = fitfilebase.format(ngauss) + '_evel1.fits'
        sigma1_1Gfile = fitfilebase.format(ngauss) + '_sigma1.fits'
        sigma1_e_1Gfile = fitfilebase.format(ngauss) + '_esigma1.fits'

    
    cube = pyspeckit.Cube(imagefile)
    header = cube.header

    beamarea, beamarea_pix2 = beam_size(header)
    minsizetrim = minsize * beamarea_pix2
    chansize = np.abs(header['CDELT3'])
    print('Channel width: ', chansize)

    rmsmap = fits.getdata(rmsfile)
    snrmap, headerimage = fits.getdata(snrfile, header=True)
    
    limitedmin=[True, True, True]
    limitedmax=[False, True, True]
    minpars=[0, velrange[0], 0]
    maxpars=[0, velrange[1], velrange[1]-velrange[0]]
    
    if not os.path.exists(maskfile+'.fits'): # the mask is not rewritten
        hdcube = headerimage.copy()
        hdcube['BUNIT'] = ''
        planemask = (snrmap > snratio)
        # fits.writeto(maskfile+'_0.fits', planemask.astype(int), hdcube)
        # check the resulting mask map to see how much does the minimum size have to be and its connectivity
        # before applying this filter
        planemask = remove_small_objects(planemask, min_size=minsizetrim) # removes small islands of emission
        # fits.writeto(maskfile+'_1.fits', planemask.astype(int), hdcube)
        planemask = remove_small_holes(planemask, area_threshold=minsizetrim) # fills small holes inside the emission area
        # fits.writeto(maskfile+'_2.fits', planemask.astype(int), hdcube)
        planemask = closing(planemask, disk(1))
        planemask = remove_small_holes(planemask, area_threshold=6)
        fits.writeto(maskfile+'.fits', planemask.astype(int), hdcube)
        if verbose: print('Created Mask file')
    else:
        planemask = fits.getdata(maskfile+'.fits')
        if verbose: print('Loaded Mask file')
    
    initguess_master = [1, 8, 0.2, 0.8, 7, 0.2, 0.7, 6, 0.3]
        
    if initguesses is None:
        initguesses = initguess_master[:ngauss*3]
    if starting_point is None:
        starting_point = (194,264)
    
    if not os.path.exists(fitfile) or overwrite:
        cube.fiteach(fittype='gaussian',
                     # signal_cut=snratio,
                     guesses=initguesses,
                     errmap = rmsmap, 
                     maskmap = planemask,
                     limitedmin=limitedmin,
                     limitedmax=limitedmax,
                     minpars=minpars,
                     maxpars=maxpars,
                     use_neighbor_as_guess=True, 
                     start_from_point=starting_point,
                     verbose=False,
                     multicore=40)
        cube.write_fit(fitfile, overwrite=overwrite)
        fittedmodel = cube.get_modelcube()

    else:
        cube.load_model_fit(fitfile, npars=3, npeaks=ngauss, fittype='gaussian')

        
    if not filter_before_refit:
        if verbose: print('Not filtering before refit')
        newinitguess = fits.getdata(fitfile)[:3*ngauss]
    
    else:
        
        if not os.path.exists(fitfilefiltered) or overwrite:
            if verbose: print("Creating filtered version.") 
            parcube, errcube = filtersolutions(cube.parcube, cube.errcube, ngauss, 
                                               rmsmap=rmsmap, snratio=3, 
                                               velinit=velrange[0], velend=velrange[1], 
                                               filter_negative=True, errorfrac=err_tol, 
                                               filter_islands=True, chansize=chansize, widthfilter=True)
            cube.parcube = parcube
            cube.errcube = errcube
            cube.write_fit(fitfilefiltered, overwrite=overwrite) 
            fittedmodel = cube.get_modelcube()

        else:
            if verbose: print("Loading filtered version.")
            cube.load_model_fit(fitfilefiltered, npars=3, npeaks=ngauss, fittype='gaussian')
            fittedmodel = cube.get_modelcube()

        if not os.path.exists(newguessfile) or overwrite:
            if verbose: print("Interpolating previous solutions.")
            newinitguess, headerguess = interpolatesolutions(fitfilefiltered, ngauss, mask=planemask)
            fits.writeto(newguessfile, newinitguess, headerguess, overwrite=overwrite)

        else:
            if verbose: print("Interpolation exists. Loading.")
            newinitguess = fits.getdata(newguessfile)
        
    if not os.path.exists(fitfile2) or overwrite:
        print("Starting re-fit")
        cube.fiteach(fittype='gaussian',
                     guesses=newinitguess,
                     errmap = rmsmap, 
                     maskmap = planemask,
                     limitedmin=limitedmin,
                     limitedmax=limitedmax,
                     minpars=minpars,
                     maxpars=maxpars,
                     use_neighbor_as_guess=True, 
                     start_from_point=starting_point,
                     verbose=False,
                     verbose_level=1,
                     multicore=40)
        cube.write_fit(fitfile2, overwrite=overwrite)
        fittedmodel = cube.get_modelcube()

    else:
        if verbose: print("Fit 2 exists. Loading")
        cube.load_model_fit(fitfile2, npars=3, npeaks=ngauss, fittype='gaussian')
        
    if not os.path.exists(fitfile2filtered) or overwrite:
        if verbose: print("Creating filtered version.") 
        parcube, errcube = filtersolutions(cube.parcube, cube.errcube, ngauss, 
                                           rmsmap=rmsmap, snratio=3, 
                                           velinit=velrange[0], velend=velrange[1], 
                                           filter_negative=True, errorfrac=err_tol, 
                                           filter_islands=True, chansize=chansize, widthfilter=True)
        cube.parcube = parcube
        cube.errcube = errcube
        cube.write_fit(fitfile2filtered, overwrite=overwrite) #(fitfile2filtered)
        fittedmodel = cube.get_modelcube()

    else:
        if verbose: print("Loading filtered version.")
        cube.load_model_fit(fitfile2filtered, 3, fittype='gaussian')
        fittedmodel = cube.get_modelcube()
        parcube = cube.parcube
        errcube = cube.errcube

    headeramp = headerimage.copy()
    headerv = headerimage.copy()
    headersigma = headerimage.copy()
    headeramp['BUNIT'] = 'K'
    headerv['BUNIT'] = 'km s-1'
    headersigma['BUNIT'] = 'km s-1'

    if ngauss == 1:
        if not os.path.exists(amp1_1Gfile) or overwrite:
            fits.writeto(amp1_1Gfile, parcube[0], headeramp, overwrite=True)
            fits.writeto(v1_1Gfile, parcube[1], headerv, overwrite=True)
            fits.writeto(sigma1_1Gfile, parcube[2], headersigma, overwrite=True)
        if not os.path.exists(amp1_e_1Gfile) or overwrite:
            fits.writeto(amp1_e_1Gfile, errcube[0], headeramp, overwrite=True)
            fits.writeto(v1_e_1Gfile, errcube[1], headerv, overwrite=True)
            fits.writeto(sigma1_e_1Gfile, errcube[2], headersigma, overwrite=True)
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fits the HC3N 10-9 cube using the indicated number of Gaussian components')
    parser.add_argument('ngauss', type=int, help="Number of Gaussian components to fit")
    parser.add_argument('-g', '--guesses', type=int, nargs='+', default=None, help="Initial guesses for the fit. Length: 3 times ngauss")
    parser.add_argument('-s', '--snr', type=float, default=4.0, help="Minimal signal to noise ratio to fit.")
    parser.add_argument('-o', '--overwrite', action="store_true", help="Overwrite existing files.")
    parser.add_argument('-m', '--minsize', type=float, default=1.0, help="Minimum area to fit in beam sizes.")
    parser.add_argument('--startpoint', type=int, nargs=2, default=None, help="X and Y initial position to fit.")
    parser.add_argument('-f', '--filterbefore', action="store_false", help="Avoid filtering the first solutions to use as initial guesses for second fit.")
    parser.add_argument('-v', '--verbose', action="store_true", help="Activate verbose output.")

    args = parser.parse_args()
    
    initialguesses = args.guesses
    if (initialguesses is not None) and (len(initialguesses) != ngauss * 3): 
        raise ValueError('The initial guesses list has a length of {0} and should be {1}'.format(len(initialguesses), ngauss*3))
    if args.startpoint is not None:
        startpoint = (args.startpoint[0], args.startpoint[1])
    else:
        startpoint = None
    
    main(args.ngauss, args.snr, args.minsize, initguesses=initialguesses, starting_point=startpoint, overwrite=args.overwrite, verbose=args.verbose, filter_before_refit=args.filterbefore)
