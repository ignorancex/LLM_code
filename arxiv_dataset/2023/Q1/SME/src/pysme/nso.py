# -*- coding: utf-8 -*-
# Get and read the NSO spectrum
import numpy as np

from . import large_file_storage as lfs


def load_solar_spectrum(ravel=True):
    """
    Load the solar spectrum

    The data is from the National Solar Observatory Atlas #1, which is an
    optical, full-disk, FTS spectrum of a relatively inactive Sun.
    Automatically retrieves the data from the SME Large File Server
    and stores it for local use.

    Returns
    -------
    wave : array
        Wavelength in Angström
    flux : array
        Normalized spectrum
    """
    server = lfs.setup_atmo()
    fname = server.get("nso.bin")

    # number of wavelength segments
    nseg = 1073
    # spectrum points per segment
    pts = 1024
    # the data is split into wavelength and flux
    # as defined here
    dtype = [("wave", f"({pts},)>f4"), ("flux", f"({pts},)>i2")]

    with open(fname, "rb") as f:
        # Read the wavelength of each segment from the header
        wseg = np.fromfile(f, dtype=">i2", count=nseg)
        data = np.fromfile(f, dtype=dtype)

    # Split the data
    wave = data["wave"]
    flux = data["flux"]
    # Add the base wavelength to each segment
    wave = wave + wseg[:, None]
    # Normalize spectrum (and change to double data type implicitly)
    flux = flux / 30_000

    # Unravel the spectra
    if ravel:
        wave = wave.ravel()
        flux = flux.ravel()

    return wave, flux
