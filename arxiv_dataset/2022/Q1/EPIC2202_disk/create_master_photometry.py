# This file is used to create a master photometry file and uses the AAVSO 
# format as this seems to contain lots of information. The different fields
# are:
# JD, Magnitude, Uncertainty, HQuncertainty, Band, Observer Code, Comment 
# Code(s), Comp Star 1, Comp Star 2, Charts,Comments, Transfomed, Airmass,
# Validation Flag, Cmag, Kmag, HJD, Star Name, Observer Affiliation, 
# Measurement Method, Grouping Method, ADS Reference, Digitizer, Credit

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord


def convert_asas_to_aavso(filename, grades='ABC'):
    '''
    This function converts asas *.lc files to aavso format. the asas light 
    curve files contain the following information: HJD - 2450000, 5 
    magnitudes and magnitude errors (different aperture sizes - we select the
    smallest aperture), the grade (from A - D), flag (integer, 0 is good), it 
    contains data per field (several different cameras), details for the 
    comparison star (ra, dec, class, 5 magnitudes, 5 magnitude errors, 5 
    nskips?), details for the target (ra, dec). The filename contains the 
    band.

    This means we get the following set-up:

    JD, magnitude0, magnitude_error0, , Band, ASAS - Field, , Comp Star 1, , ,
     , , , Flag, Cmag, , HJD, ASASSN-V J181654.06-202117.6, Astronomical 
    Observatory - University of Warsaw, , , , , 

    Parameters
    ----------
    filename : str
        name of the asas lightcurve (*.lc) file to convert
    grades : str
        which letters are acceptable for the grade of the data [default = 'ABC']

    Returns
    -------
    write_lines : list of str
        contains all the lines that are to be written to the master file
    '''
    # read the file and extract the lines
    f = open(filename, 'r')
    read_lines = f.readlines()
    f.close()
    # get band
    band = filename.split('_')[1][1].upper()
    # prepare the lines to write to the master file
    write_lines    = []
    line_template1 = '%.5f,%.3f,%.3f,,%s,ASAS - %s,,%s,,,,,,%s,%.3f,,,'
    line_template2 = 'EPIC 220208795,Astronomical '
    line_template3 = 'Observatory - University of Warsaw,,,,,\n'
    line_template  = line_template1 + line_template2 + line_template3
    # define target
    target = SkyCoord('18:16:54.06 -20:21:17.6', unit=(u.hourangle, u.deg))
    # read each line
    for line in read_lines:
        # first look at the comment lines that give certain information this is
        # organised by field so we define several patterns in the written lines
        # and then start writing lines afterwards
        if line[0] == '#':
            # get field
            if '#dataset=' in line:
                field = line.split()[-1]
            # get comparison coordinates - convert later to comparison name
            elif '#cra=' in line:
                cra = float(line.split()[1])
            elif '#cdec=' in line:
                cdec = float(line.split()[1])
            # get comparison magnitude and error
            elif '#cmag_0=' in line:
                cmag = float(line.split()[-1])
            # get target coordinates - convert later to comparison name
            elif '#ra=' in line:
                ra = float(line.split()[1])
            elif '#dec=' in line:
                dec = float(line.split()[1])
                # determine comparison star name from coordinates of target and
                # comparison star
                delta_ra       = ra - cra
                delta_dec      = dec - cdec
                comparison_ra  = target.ra.deg - delta_ra
                comparison_dec = target.dec.deg - delta_dec
                comparison_loc = SkyCoord(ra=comparison_ra, dec=comparison_dec,
                                          unit=u.deg)
                # convert to string
                comparison_name = comparison_loc.to_string('hmsdms')
                remove_chars = ['h', 'd', 'm', 's']
                for char in remove_chars:
                    comparison_name = comparison_name.replace(char, '')
                comparison_ra, comparison_dec = comparison_name.split()
                # ensure the right number of decimals
                comparison_ra  = '%.2f' % float(comparison_ra)
                comparison_dec = '%.1f' % float(comparison_dec)
                comparison_name = 'J%s%s' % (comparison_ra, comparison_dec)
        # this is where data is actually accumulated and lines are written
        else:
            data = line.split()
            jd        = float(data[0]) + 2450000
            magnitude = float(data[1])
            mag_error = float(data[6])
            grade     = data[11]
            flag      = data[-1]
            line_data = (jd, magnitude, mag_error, band, field, 
                         comparison_name, flag, cmag)
            write_line = line_template % line_data
            if grade in grades:
                write_lines.append(write_line)
    return write_lines     

def convert_asassn_to_aavso(filename):
    '''
    This function converts asas-sn *.csv files to aavso format. The assasn light
    curve files contain the following information: HJD, UT Date, Camera, FWHM,
    Limit, magnitude, magnitude error, flux (mJy), flux error (mJy), Filter

    This means we get the following 
    HJD, magnitude, magnitude error, , Filter, ASAS-SN - Camera, , , , , , , ,
    , , , , ASASSN-V J181654.06-202117.6, Ohio State University, , , , , 

    Parameters
    ----------
    filename : str
        name of the asas-sn lightcurve (*.csv) file to convert

    Returns
    -------
    write_lines : list of str
        contains all the lines that are to be written to the master file
    '''
    # load the data
    hjds, magnitudes, mag_errors = np.loadtxt(filename, usecols=(0, 5, 6),
                                           delimiter=',', skiprows=1).T
    cameras, bands = np.loadtxt(filename, usecols=(2, 9), delimiter=',',
                                skiprows=1, dtype=str).T
    data = (hjds, magnitudes, mag_errors, cameras, bands)
    # prepare the lines to write to the master file
    write_lines    = []
    line_template1 = '%.5f,%.3f,%.3f,,%s,ASAS-SN - %s,,,,,,,,,,,,EPIC 220208795'
    line_template2 = ',Ohio State University,,,,,\n'
    line_template  = line_template1 + line_template2
    # write the lines
    for hjd, magnitude, mag_error, camera, band in zip(*data):
        line_data  = (hjd, magnitude, mag_error, band, camera)
        write_line = line_template % line_data
        write_lines.append(write_line)
    return write_lines

def convert_atlas_to_aavso(filename):
    '''
    This function converts atlas *.dph files to aavso format. The atlas light
    curve files contain the following information: OBS (contains filter), MJD,
    RA, dec, magnitude, Type (star, galaxy...etc., must be 1 or 7), dfitmag 
    (magnitude error).

    This means we get the following 
    HJD, magnitude, magnitude error, , Filter, ATLAS, , , , , , , , , , , , 
    ASASSN-V J181654.06-202117.6, University of Hawaii, , , , , 

    Parameters
    ----------
    filename : str
        name of the atlas lightcurve (*.dph) file to convert

    Returns
    -------
    write_lines : list of str
        contains all the lines that are to be written to the master file
    '''
    # read the file and extract the lines
    f = open(filename, 'r')
    read_lines = f.readlines()
    f.close()
    # prepare the lines to write to the master file
    write_lines    = []
    line_template1 = '%.5f,%.3f,%.3f,,%s,ATLAS,,,,,,,,,,,,EPIC 220208795'
    line_template2 = ',University of Hawaii,,,,,\n'
    line_template  = line_template1 + line_template2
    # read lines
    for line in read_lines:
        # try because some lines are incomplete
        try:
            data      = line.split()
            band      = data[0][-1]
            jd        = float(data[1]) + 2400000.5
            magnitude = float(data[4])
            mag_error = float(data[10])
            type_flag = int(data[6])
            if (type_flag == 1) or (type_flag == 7):
                line_data  = (jd, magnitude, mag_error, band)
                write_line = line_template % line_data
                write_lines.append(write_line)
        # incomplete line
        except:
            pass
    return write_lines

def convert_evryscope_to_aavso(filename, snr_min=5):
    '''
    This function converts evryscope *.fits files to aavso format. The 
    Evryscope light curve files contain the following information: MJD, 
    magnitude, magnitude error, flux, flux error, SNR.

    This means we get the following 
    JD, magnitude, magnitude error, , g, Evryscope, , , , , , , , , , , , 
    ASASSN-V J181654.06-202117.6, University of North Carolina - Chapel Hill, 
    , , , , 

    Parameters
    ----------
    filename : str
        name of the evryscope lightcurve (*.fits) file to convert
    snr_min : float
        minimum value of the signal to noise ratio for the point to be accepted

    Returns
    -------
    write_lines : list of str
        contains all the lines that are to be written to the master file
    '''
    # extract the information
    fitsfile   = fits.open(filename)
    fitsdata   = fitsfile[1].data
    jds        = np.array(fitsdata['mjd']) + 2400000.5
    magnitudes = np.array(fitsdata['mag'])
    mag_errors = np.array(fitsdata['magerr'])
    snrs       = np.array(fitsdata['snr'])
    data       = (jds, magnitudes, mag_errors, snrs)
    # prepare the lines to write to the master file
    write_lines    = []
    line_template1 = '%.5f,%.3f,%.3f,,g,Evryscope,,,,,,,,,,,,EPIC 220208795'
    line_template2 = ',University of North Carolina'
    line_template3 = ' - Chapel Hill,,,,,\n'
    line_template  = line_template1 + line_template2 + line_template3
    # write the lines
    for jd, magnitude, mag_error, snr in zip(*data):
        # check that the magnitude is not a nan and that snr >= snr_min
        if (magnitude != 'nan') and (snr >= snr_min):
            line_data  = (jd, magnitude, mag_error)
            write_line = line_template % line_data
            write_lines.append(write_line)
    return write_lines
     
def convert_pobs_to_aavso(filename, dB=13, dV=12.4, dR=12.1, dI=11.8):
    '''
    This function converts pobs *.xlsx files to aavso format. The POBS light
    curve files contain the following information: JD - 2400000, Airmass, flux
    flux error, snr, coordinate + much more information on the target and 9 
    comparison stars separated into a sheet per filter.

    This means we get the following 
    JD, magnitude, magnitude error, , Filter, POBS, , ENSEMBLE, , , , , Airmass,
    , , , , ASASSN-V J181654.06-202117.6, Perth Observatory, , , , , 

    Parameters
    ----------
    filename : str
        name of the pobs lightcurve (*.xlsx) file to convert
    dB : float
        this is the shift needed for B band to line up approximately
    dV : float
        this is the shift needed for V band to line up approximately
    dR : float
        this is the shift needed for R band to line up approximately
    dI : float
        this is the shift needed for I band to line up approximately

    Returns
    -------
    write_lines : list of str
        contains all the lines that are to be written to the master file

    Notes
    -----
    We need to convert the flux to magnitude to be able to write this to the file
    which means we might need to apply a magnitude shift to the data
    '''
    # extract data
    bands  = ['B', 'V', 'R', 'I']
    shifts = [dB, dV, dR, dI]
    cols  = ['J.D.-2400000', 'rel_flux_T1', 'rel_flux_err_T1', 'AIRMASS']
    # prepare the lines to write to the master file
    write_lines    = []
    line_template1 = '%.5f,%.3f,%.3f,,%s,POBS,,ENSEMBLE,,,,,%.3f,,,,,'
    line_template2 = 'EPIC 220208795,Perth Observatory,,,,,\n'
    line_template  = line_template1 + line_template2
    # go through each sheet / band
    for sheet in range(4):
        # get the data for a sheet
        sheet_data  = pd.read_excel(filename, sheet_name=sheet, usecols=cols)
        jds         = sheet_data[cols[0]] + 2400000
        fluxes      = sheet_data[cols[1]]
        flux_errors = sheet_data[cols[2]] / np.nanmedian(fluxes)
        fluxes      = fluxes / np.nanmedian(fluxes)
        airmasses   = sheet_data[cols[3]]
        # conversion of flux to magnitude
        magnitudes = -2.5 * np.log10(fluxes) + shifts[sheet]
        error1 = np.abs(-2.5 * np.log10(fluxes - flux_errors))
        error2 = np.abs(-2.5 * np.log10(fluxes + flux_errors))
        mag_errors = error1
        mag_errors[mag_errors < error2] = error2[mag_errors < error2]
        # get data array
        data = (jds, magnitudes, mag_errors, airmasses)
        # write lines
        for jd, magnitude, mag_error, airmass in zip(*data):
            if np.isnan(magnitude) == False:
                line_data  = (jd, magnitude, mag_error, bands[sheet], airmass)
                write_line = line_template % line_data
            write_lines.append(write_line)
    return write_lines

def read_aavso(filename):
    '''
    This functions reads the lines from the aavso files (*.txt)

    Parameters
    ----------
    filename : str
        name of the aavso lightcurve (*.txt) file to read

    Returns
    -------
    read_lines : list of str
        contains all the lines in the aavso file
    '''
    # open and extract the data
    f = open(filename, 'r')
    read_lines = f.readlines()
    f.close()
    return read_lines

def create_master_file(output_filename, aavso_files=[], asas_files=[], 
                       asassn_files=[], atlas_files=[], evryscope_files=[],
                       pobs_files=[], grades='ABC', snr_min=5, dB=13, dV=12.4,
                       dR=12.1, dI=11.8):
    '''
    This function creates one master file from the asas, asas-sn, atlas,
    evryscope, pobs and the aavso

    Parameters
    ----------
    output_filename : str
        name of the master light curve file (*.txt)
    aavso_files : list of str
        list of aavso files to be added to the master light curve file (*.txt)
    asas_files : list of str
        list of asas files to be converted to aavso format (*.lc)
    asassn_files : list of str
        list of asassn files to be converted to aavso format (*.csv)
    atlas_files : list of str
        list of atlas files to be converted to aavso format (*.dph)
    evryscope_files : list of str
        list of evryscope files to be converted to aavso format (*.fits)
    pobs_files : list of str
        list of perth obs files to be converted to aavso format (*.xlsx)
    grades : str
        which grades are acceptable for ASAS data [default = 'ABC']
    snr_min : float
        minimum signal to noise ratio acceptable for Evryscope data
    dB : float
        B shift for POBS data (based on AAVSO)
    dV : float
        V shift for POBS data (based on AAVSO)
    dR : float
        R shift for POBS data (based on AAVSO)
    dI : float
        I shift for POBS data (based on AAVSO)

    Returns
    -------
    master_light_curve : list of str
        list containing all the individual lines of the master_light_curve file
    '''
    # all files
    all_files = [aavso_files, asas_files, asassn_files, atlas_files, 
                 evryscope_files, pobs_files]
    # all the read and convert functions
    functions = [read_aavso, convert_asas_to_aavso, convert_asassn_to_aavso,
                 convert_atlas_to_aavso, convert_evryscope_to_aavso,
                 convert_pobs_to_aavso]
    labels = ['AAVSO', 'ASAS', 'ASAS-SN', 'ATLAS', 'Evryscope', 'POBS']
    # create list for all lines
    print('Creating Master Light Curve File')
    master_lightcurve = []
    for files, function, label in zip(all_files, functions, labels):
        try:
            print('    Appending %s...' % label)
            if isinstance(files, list) == False:
                files = [files]
            for filename in files:
                if label == 'ASAS':
                    photometry = function(filename, grades)
                elif label == 'Evryscope':
                    photometry = function(filename, snr_min)
                elif label == 'POBS':
                    photometry = function(filename, dB, dV, dR, dI)
                else:
                    photometry = function(filename)
                master_lightcurve = master_lightcurve + photometry
        except:
            pass
    print('Saving Master Light Curve to %s' % output_filename)
    np.savetxt(output_filename, master_lightcurve, fmt='%s', newline='')
    return master_lightcurve

def load_aavso(filename, min_obs=10, flux=False):
    '''
    this function loads the aavso data
    
    Parameters
    ----------
    filename : str
        name of the file containing all the aavso data
    min_obs : ind
        the minimum number of observations to be considered
    flux : bool
        if true return fluxes, instead of magnitudes
        
    Returns
    -------
    times : list - float
        list of lists containing the times for each of the observers + bands
    magnitudes : list - float
        list of lists containing the magnitudes for each of the observers + bands
    errors : list - float
        list of lists containing the errors for each of the observers + bands
    IDs : list - str
        list of the observers + bands
    '''
    num_data = np.genfromtxt(filename, usecols=(0, 1, 2, 12), delimiter=',',
                             skip_header=1, filling_values=0).T
    all_times, all_mags, all_errors, all_airmasses = num_data
    str_data = np.loadtxt(filename, usecols=(4, 5), skiprows=1, delimiter=',',
                          dtype=str).T 
    all_bands, all_obs = str_data
    times = []
    magnitudes = []
    errors = []
    airmasses = []
    IDs = []
    # get unique observers
    observers = np.unique(all_obs)
    for observer in observers:
        obs_mask       = (all_obs == observer)
        obs_times      = all_times[obs_mask]
        obs_magnitudes = all_mags[obs_mask]
        obs_errors     = all_errors[obs_mask]
        obs_airmasses  = all_airmasses[obs_mask]
        obs_all_bands  = all_bands[obs_mask]
        obs_bands      = np.unique(obs_all_bands)
        for obs_band in obs_bands:
            band_mask       = (obs_all_bands == obs_band)
            if np.sum(band_mask) >= min_obs:
                band_times      = obs_times[band_mask]
                band_magnitudes = obs_magnitudes[band_mask]
                band_errors     = obs_errors[band_mask]
                band_airmasses  = obs_airmasses[band_mask]
                # appending
                times.append(band_times)
                magnitudes.append(band_magnitudes)
                errors.append(band_errors)
                airmasses.append(band_airmasses)
                IDs.append('%s %s' % (observer, obs_band))
    if flux == True:
        fluxes, flux_errors = magnitude_to_flux(magnitudes, errors)
        return times, fluxes, flux_errors, airmasses, IDs
    else:
        return times, magnitudes, errors, airmasses, IDs

def magnitude_to_flux(magnitudes, magnitude_errors):
    '''
    this function converts magnitude and magnitude errors to flux and flux errors
    
    Parameters
    ----------
    magnitudes : list of arrays
        contains magnitude data for each band
    magnitude_errors : list of arrays
        contains magnitude error data for each band
    
    Returns
    -------
    fluxes : list of arrays
        contains normalised flux data for each band
    flux_errors : list of arrays
        contains flux error data for each band
    '''
    fluxes = []
    flux_errors = []
    for magnitude, magnitude_error in zip(magnitudes, magnitude_errors):
        # determine flux
        flux = 10**(magnitude / -2.5)
        # determine the flux error for magnitude +- magnitude_error
        flux_m = 10**((magnitude + magnitude_error) / -2.5)
        flux_p = 10**((magnitude - magnitude_error) / -2.5)
        error_m = np.abs(flux - flux_m)
        error_p = np.abs(flux_p - flux)
        # determine the maximum error
        flux_error = error_p
        flux_error[flux_error < error_m] = error_m[flux_error < error_m]
        # normalise both
        flux_error /= flux
        flux /= np.nanmedian(flux)
        # append
        fluxes.append(flux)
        flux_errors.append(flux_error)
    return fluxes, flux_errors


if __name__ == '__main__':
    ''' update the master file if run '''
    # define file lists
    aavso     = []
    asas      = ['asas_3i_epicliz.lc', 'asas_4v_epicliz.lc', 'asas_3v_epicliz.lc','asas_Nv_epicliz.lc']
    asassn    = ['epic_liz.csv']
    atlas     = ['17.731554+0.314032.dphraw']
    evryscope = []
    pobs      = []
    # define extra parameters
    grades  = 'ABC'
    snr_min = 5.0
    dB      = 13.0
    dV      = 12.4
    dR      = 12.1
    dI      = 11.8
    # name of the output file
    out_file  = 'master_lightcurve.txt'
    _ = create_master_file(out_file, aavso, asas, asassn, atlas, evryscope,
                           pobs, grades, snr_min, dB, dV, dR, dI)
