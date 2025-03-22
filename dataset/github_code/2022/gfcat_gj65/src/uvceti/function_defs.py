"""
.. module:: function_defs
   :synopsis: Helper functions for the UV Ceti gPhoton notebook tutorials, used
       to re-create the data and figures used in the paper.

.. moduleauthor:: Chase Million, Scott W. Fleming
"""

import itertools
import os
from astropy.io import fits as pyfits
from astropy import wcs as pywcs
from astropy.stats import sigma_clip
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import gPhoton
from gPhoton import galextools as gt
from gPhoton.gphoton_utils import read_lc
from gPhoton.MCUtils import print_inline
from scipy.stats import anderson

def listdir_contains(directory, contains_str):
    """This function returns a sorted list of files that match the given
    pattern."""
    listd = os.listdir(directory)
    listd.sort()
    return ['{d}{f}'.format(d=directory, f=fn) for fn in listd if (
        contains_str in fn)]

def write_image(photon_file, band, imsz=[3600, 3600], overwrite=False,
                skypos=(24.76279, -17.94948)):
    """This function writes an image to disk based on the photon event .csv
    file."""
    fitsfilename = photon_file.replace('csv', 'fits')
    if not os.path.exists(fitsfilename) or overwrite:
        print('  Making image {f}'.format(f=fitsfilename))
        image, wcs = make_image(photon_file, band, imsz=imsz, skypos=skypos)
        write_fits(image, wcs, fitsfilename, band, overwrite=overwrite)

def write_movie(photon_file, band, tranges, stepsz=30, overwrite=False,
                pixsz=0.000416666666666667):
    """This function creates FITS cubes based on a photon event .csv file."""
    fitsfilename = photon_file.replace('.csv', '-{s}s.fits'.format(
        s=int(stepsz)))
    if not os.path.exists(fitsfilename) or overwrite:
        events = calibrate_photons(photon_file, band)
        for trange in tranges:
            print_inline(trange)
            image, wcs = make_image(photon_file, band, trange=trange,
                                    events=events, pixsz=pixsz)
            try:
                movie = np.append(movie, [image], axis=0)
            except:
                movie = [image]
        hdu = pyfits.PrimaryHDU(movie)
        hdulist = pyfits.HDUList([hdu])
        hdulist.writeto(fitsfilename)

def write_fits(image, wcs, fitsfilename, band, overwrite=False):
    """This function creates a FITS file out of a gPhoton image."""
    hdu = pyfits.PrimaryHDU()
    hdu.header['CDELT1'], hdu.header['CDELT2'] = wcs.wcs.cdelt
    hdu.header['CTYPE1'], hdu.header['CTYPE2'] = wcs.wcs.ctype
    hdu.header['CRPIX1'], hdu.header['CRPIX2'] = wcs.wcs.crpix
    hdu.header['CRVAL1'], hdu.header['CRVAL2'] = wcs.wcs.crval
    hdu.header['EQUINOX'], hdu.header['EPOCH'] = 2000., 2000.
    hdu.header['BAND'] = 1 if band == 'NUV' else 2
    hdu.header['VERSION'] = 'v{v}'.format(v=gPhoton.__version__)
    hdu = pyfits.PrimaryHDU(image)
    print('  Writing image to {f}'.format(f=fitsfilename))
    hdu.writeto(fitsfilename, overwrite=overwrite)

def make_image(photon_file, band, events=None, trange=None, imsz=[3600, 3600],
               skypos=(24.76279, -17.94948), pixsz=0.000416666666666667):
    """This function creates an image out of photon events."""
    if events is None:
        events = calibrate_photons(photon_file, band)
    if not trange:
        trange = (events['t'].loc[np.isfinite(events['ra'])].min(),
                  events['t'].loc[np.isfinite(events['ra'])].max())
    tix = np.where((events['t'] >= trange[0]) & (events['t'] < trange[1]))
    wcs = define_wcs(events, imsz=imsz, skypos=skypos, pixsz=pixsz)
    coo = list(zip(events.iloc[tix]['ra'], events.iloc[tix]['dec']))
    foc = wcs.sip_pix2foc(wcs.wcs_world2pix(coo, 1), 1)
    image, _, _ = np.histogram2d(foc[:, 1]-0.5, foc[:, 0]-0.5, bins=imsz,
                                 range=([[0, imsz[0]], [0, imsz[1]]]))
    return image, wcs

def calibrate_photons(photon_file, band, overwrite=False):
    """This function takes the raw photon events and produces calibrated
    photon events."""
    xfilename = '{d}{base}-x.csv'.format(d=photon_file[:-13],
                                         base=photon_file[-13:-4])
    if os.path.exists(xfilename) and not overwrite:
        return pd.read_csv(xfilename, index_col=None)
    data = pd.read_csv(photon_file, names=['t', 'x', 'y', 'xa', 'ya', 'q', 'xi',
                                           'eta', 'ra', 'dec', 'flags'])
    events = pd.DataFrame()
    flat, _ = gPhoton.cal.flat(band)
    col, row = gPhoton.curvetools.xieta2colrow(np.array(data['xi']),
                                               np.array(data['eta']), band)
    # Use only data that is on the detector.
    ix = np.where((col > 0) & (col < 800) & (row > 0) & (row < 800))
    events['t'] = pd.Series((np.array(data.iloc[ix]['t'])/1000.).byteswap().newbyteorder())
    events['ra'] = pd.Series((np.array(data.iloc[ix]['ra'])).byteswap().newbyteorder())
    events['dec'] = pd.Series((np.array(data.iloc[ix]['dec'])).byteswap().newbyteorder())
    events['flags'] = pd.Series((np.array(data.iloc[ix]['flags'])).byteswap().newbyteorder())
    events['col'] = pd.Series((col[ix]).byteswap().newbyteorder())
    events['row'] = pd.Series((row[ix]).byteswap().newbyteorder())
    flat = flat[np.array(events['col'], dtype='int16'),
                np.array(events['row'], dtype='int16')]
    events['flat'] = pd.Series((flat).byteswap().newbyteorder())
    scale = gt.compute_flat_scale(
        np.array(data.iloc[ix]['t'])/1000., band)
    events['scale'] = pd.Series((scale).byteswap().newbyteorder())
    response = np.array(events['flat'])*np.array(events['scale'])
    events['response'] = pd.Series((response).byteswap().newbyteorder())

    # Define the hotspot mask
    mask, maskinfo = gPhoton.cal.mask(band)

    events['mask'] = pd.Series(
        ((mask[np.array(col[ix], dtype='int64'),
              np.array(row[ix], dtype='int64')] == 0)).byteswap().newbyteorder())

    # Add the remaining photon list parameters back in for completeness.
    events['x'] = pd.Series((np.array(data.iloc[ix]['x'])).byteswap().newbyteorder())
    events['y'] = pd.Series((np.array(data.iloc[ix]['y'])).byteswap().newbyteorder())
    events['xa'] = pd.Series((np.array(data.iloc[ix]['xa'])).byteswap().newbyteorder())
    events['ya'] = pd.Series((np.array(data.iloc[ix]['ya'])).byteswap().newbyteorder())
    events['q'] = pd.Series((np.array(data.iloc[ix]['q'])).byteswap().newbyteorder())
    events['xi'] = pd.Series((np.array(data.iloc[ix]['xi'])).byteswap().newbyteorder())
    events['eta'] = pd.Series((np.array(data.iloc[ix]['eta'])).byteswap().newbyteorder())

    print('Writing {xf}'.format(xf=xfilename))
    events.to_csv(xfilename, index=None)

    return events

def define_wcs(events, imsz=[3200, 3200], skypos=None,
               pixsz=0.000416666666666667):
    """Generates a WCS for a gPhoton image.
    Note: The default for 'pixsz' is the same resolution as the GALEX intensity
    ("-int") maps."""
    if skypos is None:
        skypos = (events['ra'].min() + (events['ra'].max() -
                                        events['ra'].min())/2,
                  events['dec'].min() + (events['dec'].max()
                                         -events['dec'].min())/2)
    wcs = pywcs.WCS(naxis=2)
    wcs.wcs.cdelt = np.array([-pixsz, pixsz])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [(imsz[1]/2.) + 0.5, (imsz[0]/2.) + 0.5]
    wcs.wcs.crval = skypos
    return wcs

def make_lightcurve(photon_file, band, stepsz=30., skypos=(24.76279, -17.94948),
                    aper=gt.aper2deg(7), fixed_t0=False,
                    makefile=False, quiet=False, filetag='', lc_filename=None):
    """ Generate a light curve of a specific target. """
    if lc_filename is None:
        lc_filename = photon_file.replace('.csv',
                                          '-{stepsz}s{filetag}.csv'.format(
                                              stepsz=int(stepsz), filetag=filetag))
    if os.path.exists(lc_filename) and not makefile:
        if not quiet:
            print_inline('    Pre-exists, reading in file...')
        return pd.read_csv(lc_filename)
    else:
        if not quiet:
            print_inline('Generating {fn}'.format(fn=lc_filename))
    events = calibrate_photons(photon_file, band)
    # Below is a calculation of the re-centering, if desired.
    skypos_recentered = recenter(events, skypos=skypos)
    c1 = SkyCoord(ra=skypos[0]*u.degree, dec=skypos[1]*u.degree)
    c2 = SkyCoord(ra=skypos_recentered[0]*u.degree,
                  dec=skypos_recentered[1]*u.degree)
    if not quiet:
        print('Recentering aperture on [{ra}, {dec}]'.format(
            ra=skypos_recentered[0], dec=skypos_recentered[1]))
        print("Recenter shift (arcsec): " + str(c1.separation(c2).arcsec))
    ix = aper_photons(events, skypos=skypos_recentered, aper=aper)
    if len(ix[0]) == 0:
        return [], [], [], []
    trange = [np.floor(np.array(events['t'])[ix].min()),
              np.ceil(np.array(events['t'])[ix].max())]
    if fixed_t0:
        # Use this to force NUV and FUV to have the same bins
        trange[0] = fixed_t0
    expt = compute_exptime_array(np.array(events['t'].values), band, trange,
                                 stepsz, np.array(events['flags'].values))
    counts, tbins, detrads  = [], [], []
    col, row = np.array(events['col']), np.array(events['row'])
    detrad = np.sqrt((col-400)**2+(row-400)**2)
    for t0 in np.arange(trange[0], trange[1], stepsz):
        tix = np.where((np.array(events['t'])[ix] >= t0) &
                       (np.array(events['t']) < t0 + stepsz)[ix] &
                       (np.array(events['flags'], dtype='int16')[ix] == 0))
        tbins += [t0]
        detrads += [detrad[ix][tix].mean()]
        if len(tix[0]) == 0:
            counts += [0.]
        else:
            counts += [np.array(events['response'])[ix][tix].sum()]
    cps = np.array(counts) / np.array(expt)
    cps_err = np.sqrt(counts) / np.array(expt)
    lc = pd.DataFrame({'t0':tbins, 't1':list(np.array(tbins) + stepsz),
                       'cps':cps, 'cps_err':cps_err,
                       'flux':gt.counts2flux(cps, band),
                       'flux_err':gt.counts2flux(cps_err, band),
                       'counts':counts, 'expt':expt,
                       'detrad':detrads})
    lc['cps_apcorrected'] = apcorrect_cps(lc, band, aper=aper)
    lc['flux_apcorrected'] = gt.counts2flux(lc['cps_apcorrected'], band)
    lc.to_csv(lc_filename)
    return lc

def read_lightcurve(lc_file):
    """Reads a light curve file and returns the data as a Pandas DataFrame."""
    if os.path.isfile(lc_file):
        # Read in the light curve file.
        return read_lc(lc_file)
    else:
        raise IOError("Light curve file not found: " + lc_file)

def recenter(events, skypos=(24.76279, -17.94948),
             aper=gt.aper2deg(7), n_iters=5):
    """Given a position on the sky, iteratively recenter on the median photon
    position."""
    for i in np.arange(n_iters):
        # iterate to recenter on the star
        angsep = gPhoton.MCUtils.angularSeparation(skypos[0], skypos[1],
                                                   np.array(events['ra']),
                                                   np.array(events['dec']))
        ix = np.where((angsep <= aper) & (np.isfinite(angsep)) &
                      (np.array(events['flags'], dtype='int16') == 0))
        skypos = [np.median(np.array(events['ra'])[ix]),
                  np.median(np.array(events['dec'])[ix])]
    return skypos

def aper_photons(photon_data, skypos=(24.76279, -17.94948),
                 aper=gt.aper2deg(7)):
    """ Extract the events within the aperture. """
    image = photon_data
    angsep = gPhoton.MCUtils.angularSeparation(
        skypos[0], skypos[1], np.array(image['ra']), np.array(image['dec']))
    ix = np.where((angsep <= aper) & (np.isfinite(angsep)) &
                  (np.array(image['flags'], dtype='int16') == 0))
    return ix

def compute_exptime(times, band, trange, flags):
    """ Calculate the exposure time using the empirical formula from global
    countrate. """
    rawexpt = trange[1]-trange[0]

    tix = np.where((times >= trange[0]) & (times < trange[1]))
    if len(tix[0]) <= 1:
        return 0.

    # Calculate shutter
    tfix = np.where((times >= trange[0]) & (times < trange[1]) & (flags == 0))
    if len(tfix[0]) <= 1:
        return 0.
    rawexpt = max(times[tfix]) - min(times[tfix])
    t = np.sort(np.unique(times[tfix]))
    shutgap = 0.05
    if len(t) <= 1:
        return 0.
    ix = np.where(t[1:]-t[:-1] >= shutgap)
    shutter = np.array(t[1:]-t[:-1])[ix].sum()

    # Calculate deadtime
    band = 'FUV'
    model = {'NUV':[-0.000434730599193, 77.217817988],
             'FUV':[-0.000408075976406, 76.3000943221]}

    rawexpt = trange[1] - trange[0] - shutter
    gcr = len(times[tix]) / rawexpt
    feeclkratio = 0.966
    refrate = model[band][1] / feeclkratio
    scr = model[band][0] * gcr + model[band][1]
    deadtime = 1 - scr / feeclkratio / refrate

    return (rawexpt - shutter) * (1. - deadtime)

def compute_exptime_array(times, band, trange, stepsz, flags):
    """ Returns exposure times as an array based on global count rates. """
    return np.array(
        [compute_exptime(times, band, (t0, t0 + stepsz), flags) for t0 in
         np.arange(trange[0], trange[1], stepsz)])

def apcorrect_cps(lc, band, aper=gt.aper2deg(7)):
    """ Apply the aperture correction in units of linear counts-per-second.
    Aperture correction is linear in magnitude units, so convert the count rate
    into AB mag, correct it, and then convert it back.
    """
    return (gt.mag2counts(gt.counts2mag(lc['cps'].values, band) -
                          gt.apcorrect1(aper, band), band))

def find_flare_ranges(lc, sigma=3, quiescence=None):
    """ Identify the start and stop indexes of a flare event. The range will continue backwards and forwards
        in time from the peak until either the end of the visit, or a flux that is within 1-sigma of the INFF
        is found.
    """
    tranges = [[min(lc['t0']), max(lc['t1'])]]
    if not quiescence:
        q, q_err = get_inff(lc)
    else:
        q, q_err = quiescence
    flare_ranges = []
    for trange in tranges:
        # The range excludes those points that don't have good coverage in the time bin, based on 'expt'.
        # NOTE: This assumes a 30-second bin size!!
        ix = np.where((np.array(lc['t0'].values) >= trange[0]) &
                      (np.array(lc['t0'].values) <= trange[1]) &
                      (np.array(lc['expt'].values) >= 20.0) &
                      (np.array(lc['cps'].values) -
                       sigma*np.array(lc['cps_err'].values) >= q))[0]
        # Save the points that are 3-sigma above the INFF to return.
        fluxes_3sig = ix
        if not len(ix):
            # No flares were found
            continue
        # This chunk extends flares until they are indistinguishable from
        # INFF, which we define has having two sequential fluxes that are less than
        # 1-sigma above the INFF.
        temp_ix = []
        for ix_range in find_ix_ranges(ix):
            # Set extra_part = 0.0 for the original version from Chase that did not
            # take into account errors, otherwise this is set to require fluxes be
            # greater than 1-sigma from the INFF before it stops the range extension.
            # Going backwards.
            n_in_a_row = 0
            extra_part = lc.iloc[ix_range[0]]['cps_err']
            while (lc.iloc[ix_range[0]]['cps']-extra_part >= q and ix_range[0] > 0 or (n_in_a_row < 1 and ix_range[0] > 0)):
                extra_part = lc.iloc[ix_range[0]]['cps_err']
                if (lc.iloc[ix_range[0]]['cps']-extra_part < q):
                    n_in_a_row += 1
                else:
                    n_in_a_row = 0
                if (lc.iloc[ix_range[0]]['t0'] - lc.iloc[ix_range[0]-1]['t0'] >
                        1000):
                    break
                ix_range = [ix_range[0] - 1] + ix_range
            # Going forwards.
            n_in_a_row = 0
            extra_part = lc.iloc[ix_range[-1]]['cps_err']
            while (lc.iloc[ix_range[-1]]['cps']-extra_part >= q and ix_range[-1] != len(lc)-1 or (n_in_a_row < 1 and ix_range[-1] != len(lc)-1)):
                extra_part = lc.iloc[ix_range[-1]]['cps_err']
                if (lc.iloc[ix_range[-1]]['cps']-extra_part < q):
                    n_in_a_row += 1
                else:
                    n_in_a_row = 0
                if (lc.iloc[ix_range[-1]+1]['t0']-lc.iloc[ix_range[-1]]['t0'] >
                        1000):
                    break
                ix_range = ix_range + [ix_range[-1] + 1]
            temp_ix += ix_range
        ix = np.unique(temp_ix)
        flare_ranges += find_ix_ranges(list(np.array(ix).flatten()))
    return (flare_ranges, fluxes_3sig)

def refine_flare_ranges(lc, sigma=3., makeplot=True, flare_ranges=None):
    """ Identify the start and stop indexes of a flare event after
    refining the INFF by masking out the initial flare detection indexes. """
    if not flare_ranges:
        flare_ranges, _ = find_flare_ranges(lc, sigma=sigma)
    flare_ix = list(itertools.chain.from_iterable(flare_ranges))
    quiescience_mask = [False if i in flare_ix else True for i in
                        np.arange(len(lc['t0']))]
    quiescence = ((lc['cps'][quiescience_mask] *
                   lc['expt'][quiescience_mask]).sum() /
                  lc['expt'][quiescience_mask].sum())
    quiescence_err = (np.sqrt(lc['counts'][quiescience_mask].sum()) /
                      lc['expt'][quiescience_mask].sum())
    flare_ranges, flare_3sigs = find_flare_ranges(lc,
                                                  quiescence=(quiescence,
                                                              quiescence_err),
                                                  sigma=sigma)
    flare_ix = list(itertools.chain.from_iterable(flare_ranges))
    not_flare_ix = list(set([x for x in range(len(lc['t0']))]) - set(flare_ix))
    if makeplot:
        plt.figure(figsize=(15, 3))
        plt.plot(lc['t0']-min(lc['t0']), lc['cps'], '-k')
        plt.errorbar(lc['t0'].iloc[not_flare_ix]-min(lc['t0']),
                     lc['cps'].iloc[not_flare_ix],
                     yerr=1.*lc['cps_err'].iloc[not_flare_ix], fmt='ko')
        plt.errorbar(lc['t0'].iloc[flare_ix]-min(lc['t0']),
                     lc['cps'].iloc[flare_ix],
                     yerr=1.*lc['cps_err'].iloc[flare_ix], fmt='rs')
        plt.plot(lc['t0'].iloc[flare_3sigs]-min(lc['t0']),
                 lc['cps'].iloc[flare_3sigs],
                    'ro', fillstyle='none', markersize=20)
        where_badexpt = np.where(np.array(lc['expt']) < 20.)[0]
        plt.plot(lc['t0'].iloc[where_badexpt]-min(lc['t0']),
                 lc['cps'].iloc[where_badexpt],
                    'bo', fillstyle='none', markersize=20)
        plt.hlines(quiescence, lc['t0'].min()-min(lc['t0']),
                   lc['t0'].max()-min(lc['t0']))
        plt.show()
    return flare_ranges, quiescence, quiescence_err

def find_ix_ranges(ix, buffer=False):
    """ Finds indexes in the range. """
    foo, bar = [], []
    for n, i in enumerate(ix):
        if len(bar) == 0 or bar[-1] == i-1:
            bar += [i]
        else:
            if buffer:
                bar.append(min(bar)-1)
                bar.append(max(bar)+1)
            foo += [np.sort(bar).tolist()]
            bar = [i]
        if n == len(ix)-1:
            if buffer:
                bar.append(min(bar)-1)
                bar.append(max(bar)+1)
            foo += [np.sort(bar).tolist()]
    return foo

def get_inff(lc, clipsigma=3, quiet=True, band='NUV',
             binsize=30.):
    """ Calculates the Instantaneous Non-Flare Flux values. """
    sclip = sigma_clip(np.array(lc['cps']), sigma=clipsigma)
    inff = np.ma.median(sclip)
    inff_err = np.sqrt(inff*len(sclip)*binsize)/(len(sclip)*binsize)
    if inff and not quiet:
        print('Quiescent at {m} AB mag.'.format(m=gt.counts2mag(inff, band)))
    return inff, inff_err

# Alternative INFF calculation method, not used for the GJ 65 paper.
#def get_inff(lc, clipsigma=3, use_mcmc=False, quiet=True, band='NUV',
#             binsize=30.):
#    if anderson(lc['cps']).statistic < max(anderson(lc['cps']).critical_values):
#        return np.mean(lc['cps']), np.std(lc['cps'])
#    sclip = sigma_clip(lc['cps'].values,sigma_lower=3, sigma_upper=1)
#    quiescence = np.ma.median(sclip)
#    quiescence_err = np.sqrt(quiescence*len(sclip)*binsize)/(len(sclip)*binsize)
#    if quiescence and not quiet:
#        print('Quiescent at {m} AB Mag.'.format(m=counts2mag(quiescence,band)))
#    return quiescence, quiescence_err

def calculate_flare_energy(lc, frange, distance, binsize=30, band='NUV',
                           effective_widths={'NUV':729.94, 'FUV':255.45},
                           quiescence=None):
    """ Calculates the energy of a flare in erg. """
    if not quiescence:
        q, _ = get_inff(lc)
        # Convert to aperture-corrected flux
        q = gt.mag2counts(gt.counts2mag(q,band)-gt.apcorrect1(gt.aper2deg(6),band),band)
    else:
        q = quiescence[0]

    # Convert from parsecs to cm
    distance_cm = distance * 3.086e+18
    if 'cps_apcorrected' in lc.keys():
        # Converting from counts / sec to flux units.
        flare_flux = (np.array(gt.counts2flux(
            np.array(lc.iloc[frange]['cps_apcorrected']), band)) -
                      gt.counts2flux(q, band))
    else:
        # Really need to have aperture-corrected counts/sec.
        raise ValueError("Need aperture-corrected cps fluxes to continue.")
    # Zero any flux values where the flux is below the INFF so that we don't subtract from the total flux!
    flare_flux = np.array([0 if f < 0 else f for f in flare_flux])
    flare_flux_err = gt.counts2flux(np.array(lc.iloc[frange]['cps_err']), band)
    tbins = (np.array(lc.iloc[frange]['t1'].values) -
             np.array(lc.iloc[frange]['t0'].values))
    # Caluclate the area under the curve.
    integrated_flux = (binsize*flare_flux).sum()
    """
    GALEX effective widths from
    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GALEX/GALEX.NUV
    width = 729.94 A
    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GALEX/GALEX.FUV
    width = 255.45 A
    """
    # Convert integrated flux to a fluence using the GALEX effective widths.
    fluence = integrated_flux*effective_widths[band]
    fluence_err = (np.sqrt(((gt.counts2flux(lc.iloc[frange]['cps_err'], band) *
                             binsize)**2).sum())*effective_widths[band])
    energy = (4 * np.pi * (distance_cm**2) * fluence)
    energy_err = (4 * np.pi * (distance_cm**2) * fluence_err)
    return energy, energy_err

def is_left_censored(frange):
    """ Returns true if the light curve is cutoff on the left. """
    return 0 in frange

def is_right_censored(lc, frange):
    """ Returns true if the light curve is cutoff on the right. """
    return len(lc['t0'])-1 in frange

def peak_flux(lc, frange):
    """ Returns the peak flux in the light curve. """
    return lc['flux_apcorrected'][np.argmax(np.array(lc['cps'][frange].values))]

def peak_time(lc, frange, stepsz=30):
    """ Return the bin start time corresponding to peak flux. """
    return lc['t0'][np.argmax(np.array(lc['cps'][frange].values))] + stepsz/2

def is_peak_censored(lc, frange):
    """ Returns true is the peak flux is the first or last point in the light
    curve. """
    return ((np.argmax(np.array(lc['cps'][frange].values)) == 0) or
            (np.argmax(np.array(lc['cps'][frange].values)) == len(lc) - 1))
