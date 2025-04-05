from astropy.io import fits

# M Kenworthy adding FITS keywords and a WCS to a 2D image for CDS upload

fileout = 'naco1407.fit'

with fits.open('../naco_analysis/J1407_NACO_result.fits') as hdu1:

    hdu1.info()

    hdr = hdu1[0].header
    # http://simbad.u-strasbg.fr/simbad/sim-basic?Ident=V1400+Cen&submit=SIMBAD+search
    # ICRS coord. (ep=J2000) :	14 07 47.9296051120 -39 45 42.766332988

    from astropy import units as u
    from astropy.coordinates import Angle

    # input cooedinates
    ra  = Angle('14h07m47.92s')
    dec = Angle('-39d45m42.7s')
    print(ra.degree)
    print(dec.degree)
    
    hdr['DATE-OBS'] = '2019-03-01'
    hdr['INSTRUME'] = 'NAOS+CONICA'
    hdr['OBJECT']  = 'J1407'
    hdr['REFERENC'] = 'Kenworthy 2020 A&A'
    hdr['TELESCOP'] = 'ESO-VLT_U1'
    hdr['RA']       = ra.degree
    hdr['DEC']      = dec.degree
    hdr['EQUINOX']  = 2000.0
    hdr['EXPTIME']  = 5220
    hdr['FOV']      = 0.001
    hdr['FILTER']   = 'L_prime'
    hdr['WAVELMAX']  = 4105
    hdr['WAVELMIN']  = 3495
    hdr['WLENUNIT'] = 'nm'
    hdr['NAXIS']    = 2

    # now write in WCS....
    import numpy as np
    from astropy import wcs

    w = wcs.WCS(naxis=2)

    w.wcs.crpix = [51.0, 51.0] # location of star in pixels
    pscale = 0.02719 # pixel scale (arcsec/pixel)
    w.wcs.cdelt = np.array([-pscale/3600., pscale/3600.]) # NOTE the minus for correct RA!
    w.wcs.crval = [ra.degree, dec.degree]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    wcsheader = w.to_header()
    hdr.extend(wcsheader)
    print(wcsheader)

    hdu1.writeto(fileout)


from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

hdu = fits.open(fileout)[0]
wcs = WCS(hdu.header, naxis=2)

fig = plt.figure()
fig.add_subplot(111, projection=wcs)
plt.imshow(hdu.data, origin='lower', cmap=plt.cm.viridis)
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()

