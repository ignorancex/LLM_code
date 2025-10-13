from astropy.io import fits
import numpy as np
import os


def create_gaussian_source(x_center, y_center, amplitude, sigma_x, sigma_y, size):
    """
    Create a 2D Gaussian source
    """
    y, x = np.ogrid[:size[0], :size[1]]
    return amplitude * np.exp(-(
            (x - x_center) ** 2 / (2 * sigma_x ** 2) +
            (y - y_center) ** 2 / (2 * sigma_y ** 2)
    ))


def create_test_fits_mini():

    """
    Create two test FITS files with very similar WCS headers and SIP polynomial coefficients
    """
    # Create simple test images
    source_data = np.random.normal(100, 10, (100, 100))  # 100x100 image with Gaussian noise
    target_data = np.random.normal(100, 10, (100, 100))  # Same size as source image

    # Add Gaussian sources to source image
    source_sources = [
        (30, 30, 50, 3, 3),  # x, y, amplitude, sigma_x, sigma_y
        (70, 70, 30, 2, 2),
        (40, 60, 40, 2.5, 2.5),
        (20, 80, 35, 2, 2),
        (80, 20, 45, 3, 3),
    ]

    for x, y, amp, sig_x, sig_y in source_sources:
        source_data += create_gaussian_source(x, y, amp, sig_x, sig_y, source_data.shape)

    # Add corresponding Gaussian sources to target image (very slightly shifted)
    target_sources = [
        (31, 31, 50, 3, 3),  # Minimally shifted versions of source image sources
        (71, 71, 30, 2, 2),
        (41, 61, 40, 2.5, 2.5),
        (21, 81, 35, 2, 2),
        (81, 21, 45, 3, 3),
    ]

    for x, y, amp, sig_x, sig_y in target_sources:
        target_data += create_gaussian_source(x, y, amp, sig_x, sig_y, target_data.shape)

    # Create primary HDUs
    source_hdu = fits.PrimaryHDU(source_data)
    target_hdu = fits.PrimaryHDU(target_data)

    # Source WCS header
    source_hdu.header['CTYPE1'] = 'RA---TAN-SIP'  # RA with TAN projection and SIP distortions
    source_hdu.header['CTYPE2'] = 'DEC--TAN-SIP'  # DEC with TAN projection and SIP distortions
    source_hdu.header['CRVAL1'] = 150.0  # RA of reference pixel
    source_hdu.header['CRVAL2'] = 30.0  # DEC of reference pixel
    source_hdu.header['CRPIX1'] = 50.0  # X reference pixel
    source_hdu.header['CRPIX2'] = 50.0  # Y reference pixel
    source_hdu.header['CDELT1'] = -0.001  # Degree increment along x-axis
    source_hdu.header['CDELT2'] = 0.001  # Degree increment along y-axis
    source_hdu.header['PC1_1'] = 1.0
    source_hdu.header['PC1_2'] = 0.0
    source_hdu.header['PC2_1'] = 0.0
    source_hdu.header['PC2_2'] = 1.0

    # Target WCS header
    target_hdu.header['CTYPE1'] = 'RA---TAN-SIP'
    target_hdu.header['CTYPE2'] = 'DEC--TAN-SIP'
    target_hdu.header['CRVAL1'] = 150.1  # Slightly offset RA
    target_hdu.header['CRVAL2'] = 30.1  # Slightly offset DEC
    target_hdu.header['CRPIX1'] = 60.0
    target_hdu.header['CRPIX2'] = 60.0
    target_hdu.header['CDELT1'] = -0.001
    target_hdu.header['CDELT2'] = 0.001
    target_hdu.header['PC1_1'] = 0.9997  # Slight rotation
    target_hdu.header['PC1_2'] = -0.0244
    target_hdu.header['PC2_1'] = 0.0244
    target_hdu.header['PC2_2'] = 0.9997
    # SIP coefficients for source
    source_hdu.header['A_ORDER'] = 3
    source_hdu.header['B_ORDER'] = 3
    source_hdu.header['AP_ORDER'] = 3
    source_hdu.header['BP_ORDER'] = 3

    # Forward SIP coefficients
    source_hdu.header['A_0_2'] = 0.001
    source_hdu.header['A_0_3'] = 0.0002
    source_hdu.header['A_1_1'] = 0.0003
    source_hdu.header['A_1_2'] = 0.0001
    source_hdu.header['A_2_0'] = 0.0002
    source_hdu.header['A_2_1'] = 0.0004
    source_hdu.header['A_3_0'] = 0.0001

    source_hdu.header['B_0_2'] = 0.001
    source_hdu.header['B_0_3'] = 0.0002
    source_hdu.header['B_1_1'] = 0.0003
    source_hdu.header['B_1_2'] = 0.0001
    source_hdu.header['B_2_0'] = 0.0002
    source_hdu.header['B_2_1'] = 0.0004
    source_hdu.header['B_3_0'] = 0.0001

    # Inverse SIP coefficients
    source_hdu.header['AP_0_2'] = -0.001
    source_hdu.header['AP_0_3'] = -0.0002
    source_hdu.header['AP_1_1'] = 0.0
    source_hdu.header['AP_1_2'] = -0.0001
    source_hdu.header['AP_2_0'] = -0.0002
    source_hdu.header['AP_2_1'] = -0.0004
    source_hdu.header['AP_3_0'] = -0.0001

    source_hdu.header['BP_0_2'] = -0.001
    source_hdu.header['BP_0_3'] = -0.0002
    source_hdu.header['BP_1_1'] = 0.0
    source_hdu.header['BP_1_2'] = -0.0001
    source_hdu.header['BP_2_0'] = -0.0002
    source_hdu.header['BP_2_1'] = -0.0004
    source_hdu.header['BP_3_0'] = -0.0001



    # SIP coefficients for target (slightly different from source)
    target_hdu.header['A_ORDER'] = 3
    target_hdu.header['B_ORDER'] = 3
    target_hdu.header['AP_ORDER'] = 3
    target_hdu.header['BP_ORDER'] = 3

    # Forward SIP coefficients
    target_hdu.header['A_0_2'] = 0.0011
    target_hdu.header['A_0_3'] = 0.00022
    target_hdu.header['A_1_1'] = 0.00033
    target_hdu.header['A_1_2'] = 0.00012
    target_hdu.header['A_2_0'] = 0.00022
    target_hdu.header['A_2_1'] = 0.00042
    target_hdu.header['A_3_0'] = 0.00012

    target_hdu.header['B_0_2'] = 0.0011
    target_hdu.header['B_0_3'] = 0.00022
    target_hdu.header['B_1_1'] = 0.00033
    target_hdu.header['B_1_2'] = 0.00012
    target_hdu.header['B_2_0'] = 0.00022
    target_hdu.header['B_2_1'] = 0.00042
    target_hdu.header['B_3_0'] = 0.00012

    # Inverse SIP coefficients
    target_hdu.header['AP_0_2'] = -0.0011
    target_hdu.header['AP_0_3'] = -0.00022
    target_hdu.header['AP_1_1'] = 0.0
    target_hdu.header['AP_1_2'] = -0.00012
    target_hdu.header['AP_2_0'] = -0.00022
    target_hdu.header['AP_2_1'] = -0.00042
    target_hdu.header['AP_3_0'] = -0.00012

    target_hdu.header['BP_0_2'] = -0.0011
    target_hdu.header['BP_0_3'] = -0.00022
    target_hdu.header['BP_1_1'] = 0.0
    target_hdu.header['BP_1_2'] = -0.00012
    target_hdu.header['BP_2_0'] = -0.00022
    target_hdu.header['BP_2_1'] = -0.00042
    target_hdu.header['BP_3_0'] = -0.00012

    # Write files
    if not os.path.exists('data'):
        os.mkdir('data')
    source_hdu.writeto('data/test_source_mini.fits', overwrite=True)
    target_hdu.writeto('data/test_target_mini.fits', overwrite=True)

    return 'data/test_source_mini.fits', 'data/test_target_mini.fits'


def create_test_fits():
    """
    Create two test FITS files with WCS headers similar to real astronomical observations
    """
    # Image dimensions matching the header
    source_data = np.random.normal(200, 10, (4176, 6248))  # Matching NAXIS1, NAXIS2
    target_data = np.random.normal(200, 10, (4176, 6248))

    # Add Gaussian sources to source image
    source_sources = [
        (1000, 1000, 500, 50, 50),  # x, y, amplitude, sigma_x, sigma_y
        (3000, 2000, 300, 30, 30),
        (5000, 3000, 400, 40, 40),
    ]

    for x, y, amp, sig_x, sig_y in source_sources:
        source_data += create_gaussian_source(x, y, amp, sig_x, sig_y, source_data.shape)

    # Similar sources for target image (slightly shifted)
    target_sources = [
        (1010, 1010, 500, 50, 50),
        (3010, 2010, 300, 30, 30),
        (5010, 3010, 400, 40, 40),
    ]

    for x, y, amp, sig_x, sig_y in target_sources:
        target_data += create_gaussian_source(x, y, amp, sig_x, sig_y, target_data.shape)

    # Create primary HDUs
    source_hdu = fits.PrimaryHDU(source_data)
    target_hdu = fits.PrimaryHDU(target_data)

    # Copy standard header keywords from the provided example
    def set_standard_keywords(hdu):
        hdu.header['SIMPLE'] = True
        hdu.header['BITPIX'] = 16
        hdu.header['NAXIS'] = 2
        hdu.header['NAXIS1'] = 6248
        hdu.header['NAXIS2'] = 4176
        hdu.header['EXTEND'] = True
        hdu.header['BSCALE'] = 1
        hdu.header['BZERO'] = 32768
        hdu.header['DATE-UTC'] = '2025-01-24'
        hdu.header['EXPTIME'] = 10.0
        hdu.header['IMTYPE'] = 'light'
        hdu.header['TARGET'] = 'M42'
        hdu.header['FILTER'] = 'ha_left'

    set_standard_keywords(source_hdu)
    set_standard_keywords(target_hdu)

    # WCS Keywords for source image
    source_hdu.header['WCSAXES'] = 2
    source_hdu.header['CRPIX1'] = 3973.18865967
    source_hdu.header['CRPIX2'] = 963.187820435
    source_hdu.header['PC1_1'] = -0.000483329184925
    source_hdu.header['PC1_2'] = -0.000267373825915
    source_hdu.header['PC2_1'] = 0.000267249442999
    source_hdu.header['PC2_2'] = -0.000482917011768
    source_hdu.header['CDELT1'] = 1.0
    source_hdu.header['CDELT2'] = 1.0
    source_hdu.header['CUNIT1'] = 'deg'
    source_hdu.header['CUNIT2'] = 'deg'
    source_hdu.header['CTYPE1'] = 'RA---TAN-SIP'
    source_hdu.header['CTYPE2'] = 'DEC--TAN-SIP'
    source_hdu.header['CRVAL1'] = 83.8709374485
    source_hdu.header['CRVAL2'] = -4.6205010575
    source_hdu.header['RADESYS'] = 'FK5'
    source_hdu.header['EQUINOX'] = 2000.0

    # SIP coefficients for source
    source_hdu.header['A_ORDER'] = 3
    source_hdu.header['A_0_2'] = -1.52889718406E-07
    source_hdu.header['A_0_3'] = 3.74865498433E-11
    source_hdu.header['A_1_1'] = 5.60852447736E-07
    source_hdu.header['A_1_2'] = -1.83017213138E-10
    source_hdu.header['A_2_0'] = -5.75805990751E-07
    source_hdu.header['A_2_1'] = -1.11427481136E-11
    source_hdu.header['A_3_0'] = -2.47451650776E-10

    source_hdu.header['B_ORDER'] = 3
    source_hdu.header['B_0_2'] = 9.52712806455E-07
    source_hdu.header['B_0_3'] = -2.61907044449E-10
    source_hdu.header['B_1_1'] = -3.3148811631E-07
    source_hdu.header['B_1_2'] = -5.92177329209E-11
    source_hdu.header['B_2_0'] = 2.61136678113E-07
    source_hdu.header['B_2_1'] = -2.16911114561E-10
    source_hdu.header['B_3_0'] = 2.50869340434E-12

    # Inverse SIP coefficients
    source_hdu.header['AP_ORDER'] = 3
    source_hdu.header['AP_0_0'] = -0.00139834680668
    source_hdu.header['AP_0_1'] = -3.01854088991E-06
    source_hdu.header['AP_0_2'] = 1.5335763081E-07
    source_hdu.header['AP_0_3'] = -3.7352444321E-11
    source_hdu.header['AP_1_0'] = -2.56567087153E-07
    source_hdu.header['AP_1_1'] = -5.64599234945E-07
    source_hdu.header['AP_1_2'] = 1.84294661401E-10
    source_hdu.header['AP_2_0'] = 5.7927719309E-07
    source_hdu.header['AP_2_1'] = 1.16805040674E-11
    source_hdu.header['AP_3_0'] = 2.49099237306E-10

    source_hdu.header['BP_ORDER'] = 3
    source_hdu.header['BP_0_0'] = 0.00164004748175
    source_hdu.header['BP_0_1'] = 2.53558438377E-06
    source_hdu.header['BP_0_2'] = -9.55846742341E-07
    source_hdu.header['BP_0_3'] = 2.62580287344E-10
    source_hdu.header['BP_1_0'] = -3.41032541814E-06
    source_hdu.header['BP_1_1'] = 3.31732401539E-07
    source_hdu.header['BP_1_2'] = 6.0097471581E-11
    source_hdu.header['BP_2_0'] = -2.62388052293E-07
    source_hdu.header['BP_2_1'] = 2.18225292156E-10
    source_hdu.header['BP_3_0'] = -2.22157615323E-12

    # Target image WCS with more significant differences
    target_hdu.header.update(source_hdu.header)

    # Modify WCS parameters for the target image with more substantial changes
    target_hdu.header['CRVAL1'] = 84.1209374485  # Shifted RA by ~0.25 degrees
    target_hdu.header['CRVAL2'] = -4.3705010575  # Shifted DEC by ~0.25 degrees
    target_hdu.header['CRPIX1'] = 4073.18865967  # Shifted reference pixel
    target_hdu.header['CRPIX2'] = 1063.187820435  # Shifted reference pixel

    # Modify PC matrix more significantly
    target_hdu.header['PC1_1'] = -0.000583329184925
    target_hdu.header['PC1_2'] = -0.000367373825915
    target_hdu.header['PC2_1'] = 0.000367249442999
    target_hdu.header['PC2_2'] = -0.000582917011768

    # Unique SIP coefficients for target
    target_hdu.header['A_ORDER'] = 3
    target_hdu.header['A_0_2'] = -2.52889718406E-07
    target_hdu.header['A_0_3'] = 5.74865498433E-11
    target_hdu.header['A_1_1'] = 8.60852447736E-07
    target_hdu.header['A_1_2'] = -3.83017213138E-10
    target_hdu.header['A_2_0'] = -9.75805990751E-07
    target_hdu.header['A_2_1'] = -3.11427481136E-11
    target_hdu.header['A_3_0'] = -5.47451650776E-10

    target_hdu.header['B_ORDER'] = 3
    target_hdu.header['B_0_2'] = 1.952712806455E-06
    target_hdu.header['B_0_3'] = -5.61907044449E-10
    target_hdu.header['B_1_1'] = -6.3148811631E-07
    target_hdu.header['B_1_2'] = -1.092177329209E-10
    target_hdu.header['B_2_0'] = 5.61136678113E-07
    target_hdu.header['B_2_1'] = -4.16911114561E-10
    target_hdu.header['B_3_0'] = 5.50869340434E-12

    # Unique Inverse SIP coefficients
    target_hdu.header['AP_ORDER'] = 3
    target_hdu.header['AP_0_0'] = -0.00439834680668
    target_hdu.header['AP_0_1'] = -6.01854088991E-06
    target_hdu.header['AP_0_2'] = 4.5335763081E-07
    target_hdu.header['AP_0_3'] = -8.7352444321E-11
    target_hdu.header['AP_1_0'] = -5.56567087153E-07
    target_hdu.header['AP_1_1'] = -1.164599234945E-06
    target_hdu.header['AP_1_2'] = 4.84294661401E-10
    target_hdu.header['AP_2_0'] = 1.17927719309E-06
    target_hdu.header['AP_2_1'] = 3.16805040674E-11
    target_hdu.header['AP_3_0'] = 5.49099237306E-10

    target_hdu.header['BP_ORDER'] = 3
    target_hdu.header['BP_0_0'] = 0.00464004748175
    target_hdu.header['BP_0_1'] = 5.53558438377E-06
    target_hdu.header['BP_0_2'] = -2.55846742341E-06
    target_hdu.header['BP_0_3'] = 7.62580287344E-10
    target_hdu.header['BP_1_0'] = -8.41032541814E-06
    target_hdu.header['BP_1_1'] = 6.31732401539E-07
    target_hdu.header['BP_1_2'] = 1.50974715810E-10
    target_hdu.header['BP_2_0'] = -5.62388052293E-07
    target_hdu.header['BP_2_1'] = 4.18225292156E-10
    target_hdu.header['BP_3_0'] = -4.22157615323E-12

    # Write files
    if not os.path.exists('data'):
        os.mkdir('data')
    source_hdu.writeto('data/test_source.fits', overwrite=True)
    target_hdu.writeto('data/test_target.fits', overwrite=True)

    return 'data/test_source.fits', 'data/test_target.fits'


def create_test_fits_tiny():
    """
    Create two test FITS files with minimal WCS differences and extremely small SIP coefficients
    """
    # Create simple test images (50x50 for faster processing)
    source_data = np.random.normal(100, 5, (50, 50))
    target_data = np.random.normal(100, 5, (50, 50))

    # Add a few Gaussian sources to source image
    source_sources = [
        (15, 15, 50, 2, 2),  # x, y, amplitude, sigma_x, sigma_y
        (35, 35, 40, 2, 2),
        (25, 35, 45, 2, 2),
    ]

    for x, y, amp, sig_x, sig_y in source_sources:
        source_data += create_gaussian_source(x, y, amp, sig_x, sig_y, source_data.shape)

    # Add corresponding sources to target image (very slightly shifted)
    dx = 0.5
    dy = 0.5
    target_sources = [
        (15+dx, 15+dy, 50, 2, 2),  #
        (35+dx, 35+dy, 40, 2, 2),
        (25+dx, 35+dy, 45, 2, 2),
    ]

    for x, y, amp, sig_x, sig_y in target_sources:
        target_data += create_gaussian_source(x, y, amp, sig_x, sig_y, target_data.shape)

    # Create primary HDUs
    source_hdu = fits.PrimaryHDU(source_data)
    target_hdu = fits.PrimaryHDU(target_data)

    # Source WCS header - extremely simple
    source_hdu.header['CTYPE1'] = 'RA---TAN-SIP'
    source_hdu.header['CTYPE2'] = 'DEC--TAN-SIP'
    source_hdu.header['CRVAL1'] = 150.0  # RA of reference pixel
    source_hdu.header['CRVAL2'] = 30.0  # DEC of reference pixel
    source_hdu.header['CRPIX1'] = 25.0  # Center of image
    source_hdu.header['CRPIX2'] = 25.0  # Center of image
    source_hdu.header['CDELT1'] = -0.001  # Degree increment along x-axis
    source_hdu.header['CDELT2'] = 0.001  # Degree increment along y-axis
    source_hdu.header['PC1_1'] = 1.0  # Identity matrix - no rotation
    source_hdu.header['PC1_2'] = 0.0
    source_hdu.header['PC2_1'] = 0.0
    source_hdu.header['PC2_2'] = 1.0

    # Target WCS header - minimal differences
    target_hdu.header['CTYPE1'] = 'RA---TAN-SIP'
    target_hdu.header['CTYPE2'] = 'DEC--TAN-SIP'
    target_hdu.header['CRVAL1'] = 150.005  # Just 0.01 degree offset
    target_hdu.header['CRVAL2'] = 30.005  # Just 0.01 degree offset
    target_hdu.header['CRPIX1'] = 25.0 + dx  # Just 1 pixel offset
    target_hdu.header['CRPIX2'] = 25.0 + dy  # Just 1 pixel offset
    target_hdu.header['CDELT1'] = -0.001  # Same scale
    target_hdu.header['CDELT2'] = 0.001  # Same scale
    target_hdu.header['PC1_1'] = 0.9999  # Tiny rotation
    target_hdu.header['PC1_2'] = -0.0044
    target_hdu.header['PC2_1'] = 0.0044
    target_hdu.header['PC2_2'] = 0.9999

    # Add minimal SIP information
    # Source SIP - tiny coefficients
    source_hdu.header['A_ORDER'] = 2  # Lower order for simplicity
    source_hdu.header['B_ORDER'] = 2
    source_hdu.header['AP_ORDER'] = 2
    source_hdu.header['BP_ORDER'] = 2

    # Forward SIP - extremely small coefficients
    source_hdu.header['A_1_1'] = 1.0e-8  # Extremely tiny
    source_hdu.header['A_2_0'] = 5.0e-9
    source_hdu.header['B_1_1'] = 1.0e-8
    source_hdu.header['B_0_2'] = 5.0e-9

    # Inverse SIP - corresponding tiny coefficients
    source_hdu.header['AP_0_0'] = 0.0  # Zero constant term
    source_hdu.header['AP_1_1'] = -1.0e-8  # Opposite of forward
    source_hdu.header['AP_2_0'] = -5.0e-9
    source_hdu.header['BP_0_0'] = 0.0
    source_hdu.header['BP_1_1'] = -1.0e-8
    source_hdu.header['BP_0_2'] = -5.0e-9

    # Target has identical SIP coefficients
    target_hdu.header['A_ORDER'] = 2
    target_hdu.header['B_ORDER'] = 2
    target_hdu.header['AP_ORDER'] = 2
    target_hdu.header['BP_ORDER'] = 2

    target_hdu.header['A_1_1'] = 1.0e-8
    target_hdu.header['A_2_0'] = 5.0e-9
    target_hdu.header['B_1_1'] = 1.0e-8
    target_hdu.header['B_0_2'] = 5.0e-9

    target_hdu.header['AP_0_0'] = 0.0
    target_hdu.header['AP_1_1'] = -1.0e-8
    target_hdu.header['AP_2_0'] = -5.0e-9
    target_hdu.header['BP_0_0'] = 0.0
    target_hdu.header['BP_1_1'] = -1.0e-8
    target_hdu.header['BP_0_2'] = -5.0e-9

    # Write files
    if not os.path.exists('data'):
        os.mkdir('data')
    source_hdu.writeto('data/test_source_tiny.fits', overwrite=True)
    target_hdu.writeto('data/test_target_tiny.fits', overwrite=True)

    return 'data/test_source_tiny.fits', 'data/test_target_tiny.fits'

def main():
    source_file, target_file = create_test_fits()
    print(f"Created source file: {source_file}")
    print(f"Created target file: {target_file}")


if __name__ == '__main__':
    main()