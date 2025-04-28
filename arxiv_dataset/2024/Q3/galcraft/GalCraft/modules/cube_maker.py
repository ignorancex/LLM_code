import numpy as np
from astropy.io import fits


def write_cube(data_cube, params, x_edges, y_edges, new_wave, filepath, logger, cube_idx, velscale, version):
    '''
    Write the final data cube values into a fits file, with proper header information written.
    :param data_cube:
    :param params:
    :param x_edges:
    :param y_edges:
    :param new_wave:
    :param filepath:
    :param logger:
    :param cube_idx:
    :param velscale:
    :param version:
    :return:
    '''

    wave_range = params['ssp_params']['wave_range']
    # Do the wavelength cut
    if wave_range!=None:
        good_lam = (new_wave >= wave_range[0]) & (new_wave <= wave_range[1])
        new_wave = new_wave[good_lam]
        data_cube = data_cube[:, :, good_lam]

    # add noise on it
    if params['cube_params']['add_noise'] == True:
        logger.info('Adding noise on the flux with sn = %s' % params['cube_params']['sn'])
        noise = data_cube / params['cube_params']['sn']
        data_cube_n = np.random.normal(data_cube, noise)
    else:
        logger.info('No noise was added on the flux')
        data_cube_n = data_cube


    logger.info('Writing Headers...')
    # Write header
    hdr = fits.Header()
    hdr['MODEL'] = params['other_params']['model_name']
    hdr['DIST_kpc'] = params['oparams']['distance']
    hdr['l_deg'] = params['oparams']['l']
    hdr['b_deg'] = params['oparams']['b']
    hdr['theta_zx'] = params['oparams']['theta_zx']
    hdr['theta_yz'] = params['oparams']['theta_yz']
    hdr['XCOORD'] = params['cube_params']['x_coord']
    hdr['YCOORD'] = params['cube_params']['y_coord']
    hdr['SNR'] = params['cube_params']['sn']
    hdr['X_RES'] = params['cube_params']['spatial_resolution'][0]
    hdr['Y_RES'] = params['cube_params']['spatial_resolution'][1]
    hdr['NCPU'] = params['other_params']['ncpu']
    hdr['USELOSVD'] = params['cube_params']['use_losvd']
    hdr['USEEXTIN'] = params['cube_params']['use_extinc']
    hdr['DUSTFACT'] = params['cube_params']['extinc_factor']
    hdr['ADDNOISE'] = params['cube_params']['add_noise']
    hdr['BOOTSTRA'] = params['cube_params']['bootstrap_table']
    hdr['USEDIST'] = params['cube_params']['use_dist']
    hdr['VELSCALE'] = velscale
    hdr['VERSION'] = version

    hdr['SSP'] = params['ssp_params']['model']
    hdr['ISOCHRON'] = params['ssp_params']['isochrone']
    hdr['IMF'] = params['ssp_params']['imf']
    hdr['SLOPE'] = params['ssp_params']['slope']
    hdr['NOALPHA'] = params['ssp_params']['single_alpha']
    hdr['FWHMGAL'] = params['ssp_params']['FWHM_gal']
    hdr['INTERP'] = params['ssp_params']['spec_interpolator']

    hdr1 = fits.Header()
    hdr1['EXTNAME'] = 'DATA'
    hdr1['OBJECT'] = 'MWcube'
    hdr1['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom'
    # hdr1['LOSVD'] = params['cube_params']['use_losvd']
    # hdr1['RADESYS'] = 'ICRS'
    hdr1['CRPIX1'] = (x_edges.shape[0] - 1) / 2
    hdr1['CRPIX2'] = (y_edges.shape[0] - 1) / 2
    hdr1['CD1_1'] = -np.diff(x_edges)[0]
    hdr1['CD1_2'] = 0
    hdr1['CD2_1'] = 0
    hdr1['CD2_2'] = np.diff(y_edges)[0]
    hdr1['CUNIT1'] = 'deg     '
    hdr1['CUNIT2'] = 'deg     '
    if params['cube_params']['x_coord'] == 'ra' and params['cube_params']['y_coord'] == 'dec':
        hdr1['CTYPE1'] = 'RA---TAN'
        hdr1['CTYPE2'] = 'DEC--TAN'
    elif params['cube_params']['x_coord'] == 'l' and params['cube_params']['y_coord'] == 'b':
        hdr1['CTYPE1'] = 'GLON-TAN'
        hdr1['CTYPE2'] = 'GLAT-TAN'
    hdr1['CRVAL1'] = np.mean(x_edges)
    hdr1['CRVAL2'] = np.mean(y_edges)
    hdr1['CTYPE3'] = 'AWAV'
    hdr1['CUNIT3'] = 'Angstrom'
    hdr1['CD3_3'] = new_wave[1] - new_wave[0]
    hdr1['CRPIX3'] = 1
    hdr1['CRVAL3'] = new_wave[0]
    hdr1['CD1_3'] = 0
    hdr1['CD2_3'] = 0
    hdr1['CD3_1'] = 0
    hdr1['CD3_2'] = 0
    hdr1['ADDNOISE'] = params['cube_params']['add_noise']


    empty_primary = fits.PrimaryHDU(header=hdr)
    image_hdu1 = fits.ImageHDU(data_cube_n.T, header=hdr1)
    image_hdu3 = fits.ImageHDU(new_wave, name='WAVE')

    # Write file
    logger.info('Combining hduls...')
    if params['cube_params']['add_noise'] == True:
        hdr2 = hdr1.copy()
        hdr2['EXTNAME'] = 'ERROR'
        image_hdu2 = fits.ImageHDU(noise.T, header=hdr2)
        hdul = fits.HDUList([empty_primary, image_hdu1, image_hdu2, image_hdu3])
    else:
        hdul = fits.HDUList([empty_primary, image_hdu1, image_hdu3])
    hdul.writeto(filepath + 'data_cube_' + str(cube_idx) + '.fits', overwrite=True)
