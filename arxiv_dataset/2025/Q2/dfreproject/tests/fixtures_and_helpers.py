import pytest
import numpy as np
import torch
from astropy.wcs import WCS
from astropy.io import fits
import os
import tempfile


# Helper functions for creating test data

def create_test_fits_file(wcs_header=None, data=None, filename=None):
    """
    Create a temporary FITS file for testing.

    Parameters:
    -----------
    wcs_header : dict, optional
        WCS header information to include
    data : numpy.ndarray, optional
        Image data. If None, creates a 200x200 empty array
    filename : str, optional
        Output filename. If None, creates a temporary file

    Returns:
    --------
    str
        Path to the created FITS file
    """
    # Create default data if not provided
    if data is None:
        data = np.zeros((200, 200), dtype=np.float32)

    # Create default WCS header if not provided
    if wcs_header is None:
        wcs_header = {
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CRPIX1': 100.0,
            'CRPIX2': 100.0,
            'CRVAL1': 150.0,
            'CRVAL2': 30.0,
            'CDELT1': -0.001,
            'CDELT2': 0.001,
            'PC1_1': 1.0,
            'PC1_2': 0.0,
            'PC2_1': 0.0,
            'PC2_2': 1.0,
        }

    # Create a temporary file if no filename is provided
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
        filename = temp_file.name
        temp_file.close()

    # Create HDU with data and header
    primary_hdu = fits.PrimaryHDU(data=data)

    # Add WCS headers
    for key, value in wcs_header.items():
        primary_hdu.header[key] = value

    # Write to file and close
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(filename, overwrite=True)
    hdul.close()

    return filename


def create_wcs_with_sip(degree=3):
    """
    Create a WCS object with SIP distortion coefficients.

    Parameters:
    -----------
    degree : int, optional
        Degree of the SIP polynomial. Default is 3.

    Returns:
    --------
    astropy.wcs.WCS
        WCS object with SIP distortion
    """
    # Create base WCS header
    header = {
        'CTYPE1': 'RA---TAN-SIP',
        'CTYPE2': 'DEC--TAN-SIP',
        'CRPIX1': 100.0,
        'CRPIX2': 100.0,
        'CRVAL1': 150.0,
        'CRVAL2': 30.0,
        'CDELT1': -0.001,
        'CDELT2': 0.001,
        'PC1_1': 1.0,
        'PC1_2': 0.0,
        'PC2_1': 0.0,
        'PC2_2': 1.0,
        # Add SIP header keywords
        'A_ORDER': degree,
        'B_ORDER': degree,
        'AP_ORDER': degree,
        'BP_ORDER': degree,
    }

    # Add SIP coefficients
    # These are arbitrary small values for testing
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if i + j > 0:  # Skip the constant term
                header[f'A_{i}_{j}'] = 0.001 * i * j
                header[f'B_{i}_{j}'] = -0.001 * i * j
                # Add inverse coefficients
                header[f'AP_{i}_{j}'] = -0.001 * i * j
                header[f'BP_{i}_{j}'] = 0.001 * i * j

    return WCS(header, relax=True)


@pytest.fixture(scope="session")
def simple_wcs():
    """Fixture providing a simple TAN projection WCS."""
    header = {
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CRPIX1': 100.0,
        'CRPIX2': 100.0,
        'CRVAL1': 150.0,
        'CRVAL2': 30.0,
        'CDELT1': -0.001,
        'CDELT2': 0.001,
        'PC1_1': 1.0,
        'PC1_2': 0.0,
        'PC2_1': 0.0,
        'PC2_2': 1.0,
    }
    return WCS(header)


@pytest.fixture(scope="session")
def sip_wcs():
    """Fixture providing a WCS with SIP distortion."""
    return create_wcs_with_sip(degree=2)


@pytest.fixture(scope="session")
def rotated_wcs():
    """Fixture providing a rotated WCS."""
    header = {
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CRPIX1': 100.0,
        'CRPIX2': 100.0,
        'CRVAL1': 150.0,
        'CRVAL2': 30.0,
        'CDELT1': -0.001,
        'CDELT2': 0.001,
        # 30 degree rotation
        'PC1_1': 0.866,  # cos(30°)
        'PC1_2': -0.5,  # -sin(30°)
        'PC2_1': 0.5,  # sin(30°)
        'PC2_2': 0.866,  # cos(30°)
    }
    return WCS(header)


@pytest.fixture(scope="session")
def device():
    """Fixture providing device for PyTorch testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_test_grid(shape=(10, 10), start=(0, 0), end=(199, 199)):
    """
    Create a test grid of pixel coordinates.

    Parameters:
    -----------
    shape : tuple, optional
        Shape of the grid (default: 10x10)
    start : tuple, optional
        Starting coordinates (default: (0, 0))
    end : tuple, optional
        Ending coordinates (default: (199, 199))

    Returns:
    --------
    tuple(numpy.ndarray, numpy.ndarray)
        y and x coordinate arrays
    """
    x = np.linspace(start[0], end[0], shape[1])
    y = np.linspace(start[1], end[1], shape[0])
    X, Y = np.meshgrid(x, y)
    return Y, X