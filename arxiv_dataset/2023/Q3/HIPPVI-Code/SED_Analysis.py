# Python script to assemble a complete UCD SED, fit BT-Settl and SM08 atmospheric models, and compute the bolometric luminosity
# Author: Aniket Sanghi
# Contact: asanghi@caltech.edu

# Load all packages 
import numpy as np
import pandas as pd
import warnings
import glob
import multiprocessing
from scipy.interpolate import interp1d
from astropy.io import fits
import os
import glob
warnings.filterwarnings("ignore")

#------------ FUNCTION DEFINITIONS -----------------#
def spec_phot_fit(object_name, model_spec, fit_spec, synth_phot, fit_phot, model_wave, fit_wave, fit_err, fit_phot_err):
    """Function to jointly fit optical photometry + NIR spectrum + MIR photometry

    Args:
        model_spec (numpy array): model atmosphere grid spectrum to be fit to measured Spex NIR spectrum
        fit_spec (numpy array): measured flux-calibrated spectrum of the object
        synth_phot (numpy array): synthetic photometry from model spectra
        fit_phot (numpy array): measured photometry of objects
        model_wave (numpy array): wavelength grid of model spectrum
        fit_wave (numpy array): wavelength grid of flux-calibrated spectrum of the object
        fit_err (numpy array): uncertainty in the measured flux-calibrated spectrum of the object
        fit_phot_err (numpy array): uncertainty in measured photometry

    Returns:
        numpy array: model fit error arrays
    """        
    
    # Find min-max ranges of model and fit spectrum wavelength grids
    max_model = np.max(model_wave)
    min_model = np.min(model_wave)
    max_fit = np.max(fit_wave)
    min_fit = np.min(fit_wave)
    
    # Determine range of wavelengths in the fit spectrum wavelength grid to use for interpolation of model spectrum
    # See Week 4 Notes for Pictorial Representation
    start_index = np.nan
    end_index = np.nan
    
    if((min_model < min_fit)):
        start_index = np.where(fit_wave == min_fit)[0][0]
    else:
        diff_min = min_model - wavelength
        start_index = np.where(np.sign(diff_min[:-1]) != np.sign(diff_min[1:]))[0][0] + 1
    
    if(max_model > max_fit):
        end_index = np.where(fit_wave == max_fit)[0][0]
    else:
        diff_max = max_model - wavelength
        end_index = np.where(np.sign(diff_max[:-1]) != np.sign(diff_max[1:]))[0][0]
    
    indices = np.arange(start_index, end_index+1)
        
    # Remove all points with S/N < -3 from fitting procedure
    neg_fit_indices = np.where(fit_spec/fit_err <= -3)[0]
    fit_spec[neg_fit_indices] = np.nan

    # Remove high S/N point for 2MASS J07290002-3954043 in K band (object specific issue)
    if(object_name == '2MASS J07290002-3954043'):
        temp_exclude_indices = np.where((fit_spec/fit_err > 5) & (fit_wave > 2.2))[0]
        fit_spec[temp_exclude_indices] = np.nan

    # Cut spectrum of WISE J105257.95-194250.2 long-ward of 2.35 microns (object specific issue)
    if(object_name == 'WISE J105257.95-194250.2'):
        temp_exclude_indices = np.where(fit_wave >= 2.35)[0]
        fit_spec[temp_exclude_indices] = np.nan

    # Remove telluric water absorption wavelength ranges from fitting procedure
    # 1.355 - 1.415 microns and 1.83 - 1.93 microns
    telluric_indices = np.where(((fit_wave <= 1.415) & (fit_wave >= 1.355)) | (((fit_wave <= 1.93) & (fit_wave >= 1.83))))[0]
    fit_spec[telluric_indices] = np.nan

    # Interpolate model spectrum to wavelength grid of fit spectrum
    f_model = interp1d(model_wave, model_spec, kind='cubic')
    model_spec_resamp = f_model(fit_wave[indices])
   
    # Calculate scale factor for minimum chi-sq
    spectra_sum_term_num = np.nansum((model_spec_resamp*fit_spec[indices])/(fit_err[indices]**2))
    photo_sum_term_num = np.nansum((fit_phot*synth_phot)/fit_phot_err**2)
    spectra_sum_term_denom = np.nansum((fit_spec[indices]**2)/(fit_err[indices]**2))
    photo_sum_term_denom = np.nansum((fit_phot**2)/fit_phot_err**2)
    
    min_chi_c_inv = (spectra_sum_term_num+photo_sum_term_num)/(spectra_sum_term_denom+photo_sum_term_denom)
    min_chi_c = 1/min_chi_c_inv # Based on fraction we actually need to invert before scaling
    
    # Scale spectrum and find chi-sq
    model_spec_scaled = model_spec_resamp*min_chi_c
    model_phot_scaled = synth_phot*min_chi_c

    # Full SED fit analysis
    chi_sq = np.nansum((((model_spec_scaled - fit_spec[indices])**2)/(fit_err[indices])**2)) + np.nansum(((model_phot_scaled - fit_phot)**2)/(fit_phot_err**2))
    dof = len(indices) + len(fit_phot) - 1
    reduced_chi_sq = chi_sq/dof
    rms_full_fit = np.sqrt(np.nansum((model_spec_scaled - fit_spec[indices])**2)/len(indices) + np.nansum((model_phot_scaled - fit_phot)**2)/len(fit_phot))

    # Spectrum fit analysis
    chi_sq_spec_only = np.nansum((((model_spec_scaled - fit_spec[indices])**2)/(fit_err[indices])**2))
    dof_spec_only = len(indices) - 1
    reduced_chi_sq_spec_only = chi_sq_spec_only/dof_spec_only
    rms_spec_only_fit = np.sqrt(np.nansum((model_spec_scaled - fit_spec[indices])**2)/len(indices))

    # Photometry fit analysis
    chi_sq_phot_only = np.nansum(((model_phot_scaled - fit_phot)**2)/(fit_phot_err**2))
    dof_phot_only = len(fit_phot) - 1
    reduced_chi_sq_phot_only = chi_sq_phot_only/dof_phot_only
    rms_phot_only_fit = np.sqrt(np.nansum((model_phot_scaled - fit_phot)**2)/len(fit_phot))

    return reduced_chi_sq, min_chi_c, reduced_chi_sq_phot_only, reduced_chi_sq_spec_only, rms_full_fit, rms_phot_only_fit, rms_spec_only_fit

def start_fit(object_name, model_spec_paths_compute, model_inf_fnames_compute, flux_app_cal, model_syn_flux_compute, flux_phot_nan_nir, flux_app_cal_err, flux_phot_nan_nir_err, wavelength):
    """Separate function to start the fitting procedure needed to implement parallelization

    Args:
        model_spec_paths_compute (numpy array): degraded model spectrum paths
        model_inf_fnames_compute (numpy array): infinite resolution model spectrum paths
        flux_app_cal (numpy array): measured flux-calibrated spectrum of the object
        model_syn_flux_compute (numpy array): synthetic photometry from model spectra
        flux_phot_nan_nir (numpy array): measured photometry of objects
        flux_app_cal_err (numpy array): uncertainty in measured flux-calibrated spectrum of the object
        flux_phot_nan_nir_err (numpy array): uncertainty in measured photometry
        wavelength (numpy array): wavelength grid of model spectrum

    Returns:
        numpy array: model fit error array
    """      

    # Takes care of dummy grid points
    if(('1400' in model_spec_paths_compute) and ('2.5' in model_spec_paths_compute) and ('BT-Settl' in model_spec_paths_compute)):
        return [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    # Load degraded resolution model spectrum
    model_data = np.loadtxt(model_spec_paths_compute, skiprows=1, delimiter=',')
    model_wavelength = model_data.T[0]*1e-4 # angstrom to micron
    model_flux_density = model_data.T[1] # W/m^2/um
    
    # Load corresponding infinite resolution spectra
    model_data_inf = np.genfromtxt(model_inf_fnames_compute)
    model_wavelength_inf = model_data_inf.T[0] # micron
    model_flux_density_inf = model_data_inf.T[1] # W/m^2/um
    
    # Do spectral fitting
    chi_sq, min_chi_c, chi_sq_phot_only, chi_sq_spec_only, rms_full_fit, rms_phot_only_fit, rms_spec_only_fit = spec_phot_fit(object_name=object_name, model_spec=model_flux_density, fit_spec=flux_app_cal, synth_phot=model_syn_flux_compute, fit_phot=flux_phot_nan_nir, model_wave=model_wavelength, fit_wave=wavelength, fit_err=flux_app_cal_err, fit_phot_err=flux_phot_nan_nir_err)

    sig_arr = [chi_sq, min_chi_c, chi_sq_phot_only, chi_sq_spec_only, rms_full_fit, rms_phot_only_fit, rms_spec_only_fit]
    return sig_arr

def get_photometry(df, phot_cols, index_obj):
    """Retrieve photometry from UCS

    Args:
        df (pandas dataframe): dataframe of UCS objects and data
        phot_cols (list of strings): name of the df col that stores object photometry
        index_obj (integer): index of object in df to retrieve photometry for

    Returns:
        numpy array: retrieved photometry
    """    

    phot_obj = []
    for i in range(len(phot_cols)):
        phot_obj.append(df[phot_cols[i]][index_obj])
    return phot_obj

if __name__ == '__main__': 
    #------------ MAIN START POINT -----------------#
    # NOTE: The following filepaths and formats are unique to this dataset and file directory set-up
    # Load UCS and CatWISE sheets
    df_ucs = pd.read_csv('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/Sample-Creation/UltracoolSheet - PRIVATE - Main.csv')
    names_2MASS_ucs = df_ucs['designation_2mass']
    names_ucs = df_ucs['name']

    # List to store the minimum reduced chi-sq for each target's fit: for entire fit and only optical + MIR photmetry fit
    reduced_chi_sq_for_best_fit = []
    reduced_chi_sq_for_best_phot_fit = []
    rms_full_fit_arr = []
    rms_phot_only_fit_arr = []

    for ind in range(len(names_ucs)): 
        object_name = names_ucs[ind]
        print(f'Starting index: {ind} and Object: {object_name}')

        # Object's position in UCS is same as current index
        index_obj_ucs = ind*1

        # Retrieve spectrum for object from dropbox files
        path_spec = df_ucs['HighestSNR_prism'][index_obj_ucs]
        if((type(path_spec) == float) or ('obs' in path_spec)):
            path_spec = 'path_does_not_exist'
        else:
            path_spec = '/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/SpeX-Prism-Spectra/' + path_spec[21:]
        print(path_spec)

        # ------------------- PHOTOMETRY RETRIEVAL ---------------------- #
        if(os.path.exists(path_spec)):
                    
            # Get photometry for object
            ucs_phot_cols = ['g_P1', 'r_P1', 'i_P1', 'z_P1', 'y_P1', 'J_2MASS', 'H_2MASS', 'Ks_2MASS', 
                            'J_MKO', 'H_MKO', 'K_MKO', 'W3', 'W4', 'ch1', 'ch2', 'J_ukidss', 'H_ukidss', 'K_ukidss', 'W1', 'W2']
            eff_wavelength = np.array([0.48, 0.62, 0.75, 0.87, 0.96, 1.24, 1.66, 2.16, 
                                    1.25, 1.64, 2.2, 11.56, 22.09, 3.51, 4.44, 1.2461, 1.6265, 2.1948, 3.35, 4.6]) # micron in ucs order. Source: Filter Profile Service (FPS) for Optical + Filippazzo+15 for NIR.
            filter_width = np.array([1053.08, 1252.41, 1206.62, 997.72, 638.98, 1624.32, 2509.40, 2618.87, 
                                    1516.76, 2845.78, 3242.19, 55055.23, 41016.80, 6836.18, 8649.92, 1431.53, 2772.34, 3178.83, 6626.42, 10422.66])*1e-4 # microns in ucs order. Source: Filter Profile Service.

            # Get apparent magnitudes 
            phot_obj_ucs = get_photometry(df_ucs, ucs_phot_cols, index_obj_ucs)
            phot_obj = phot_obj_ucs
            phot_obj = np.array(phot_obj, dtype=float)

            # Get errors in apparent magnitudes
            ucs_phot_cols_err = ['gerr_P1', 'rerr_P1', 'ierr_P1', 'zerr_P1', 'yerr_P1', 'Jerr_2MASS', 'Herr_2MASS', 'Kserr_2MASS', 
                                'Jerr_MKO', 'Herr_MKO', 'Kerr_MKO', 'W3err', 'W4err', 'ch1err', 'ch2err', 'Jerr_ukidss', 'Herr_ukidss', 'Kerr_ukidss', 'W1err', 'W2err']
            phot_obj_ucs_err = get_photometry(df_ucs, ucs_phot_cols_err, index_obj_ucs)
            phot_obj_err = phot_obj_ucs_err
            phot_obj_err = np.array(phot_obj_err, dtype=float)

            # Exclude PS1 photometry where the uncertainty was set to -999
            for ps1_indices_check in range(5):
                if(phot_obj_err[ps1_indices_check] == -999):
                    phot_obj[ps1_indices_check] = np.nan
                    phot_obj_err[ps1_indices_check] = np.nan

            # Exclude 2MASS photometry with contamination flag
            twomass_flag = df_ucs['Cflg_2MASS'][index_obj_ucs]
            mko_band_ref = ['ref_J_MKO', 'ref_H_MKO', 'ref_K_MKO']
            if(type(twomass_flag) == str):
                if(len(twomass_flag) == 3):
                    for flag_ind in range(3):
                        # +5 is based on the column ordering above J=5, H=6, Ks=7
                        if(twomass_flag[flag_ind] != '0'):
                            # Set the 2MASS photometry to zero for that band
                            phot_obj[flag_ind+5] = np.nan
                            phot_obj_err[flag_ind+5] = np.nan
                            
                            # Check if the MKO photometry needs to be discarded (MKO J=8, H=9, K=10)
                            # If uncertainties are equal then MKO is synthesized from 2MASS. All other cases (As checked in a separate test), the MKO photometry is smaller than all the 2MASS photometry (in which case it isn't synthesized from 2MASS)
                            if(df_ucs[mko_band_ref[flag_ind]][index_obj_ucs] == 'Best21'):
                                if(phot_obj_err[flag_ind+5] == phot_obj_err[flag_ind+8]):
                                    phot_obj[flag_ind+8] = np.nan
                                    phot_obj_err[flag_ind+8] = np.nan

            # Source: filter profile service
            # PS1 is AB system ZP
            zp_flux_vega = np.array([4.63e-9, 2.83e-9, 1.92e-9, 1.45e-9, 1.17e-9, 3.13e-10, 1.13e-10, 4.28e-11, 
                                    2.98e-10, 1.18e-10, 3.97e-11, 6.52e-14, 5.09e-15, 6.58e-12, 
                                    2.66e-12, 2.94412e-10, 1.1465e-10, 3.89743e-11, 8.18e-12, 2.42e-12])*1e-3*1e4 # erg/s/cm2/A -> W/m2/um (same order as ucs)
            vega_mag = 0

            # Sort by wavelength and designate optical, NIR, MIR bands with keywords for quick retrieval in later parts of code
            wave_sort = np.argsort(eff_wavelength)
            eff_wavelength = eff_wavelength[wave_sort]
            filter_width = filter_width[wave_sort]
            zp_flux_vega = zp_flux_vega[wave_sort]
            phot_obj = phot_obj[wave_sort]
            phot_obj_err = phot_obj_err[wave_sort]
            opt_indices = np.array([0, 1, 2, 3])
            nir_indices = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
            mir_indices = np.array([14, 15, 16, 17, 18, 19])
            twomass_indices = np.array([5, 10, 11])
            mko_indices = np.array([7, 9, 13])
            ukidss_indices = np.array([6, 8, 12])

            # If changing the contaminated 2MASS photometry results in no calibration photometry available, skip object.
            # Can remove this later since there already is a no IR photometry flag. Added for now to see how many objects drop out.
            if(len(np.where(np.isnan(phot_obj_err[nir_indices]))[0]) == len(phot_obj_err[nir_indices])):
                print('NO IR PHOTOMETRY FOR CALIBRATION')

            else:
                # Define flags for model fit - if flag is 0 that means no photometry available in that band region. Also can store flag information later for analysis.
                opt_phot_flag = 1
                nir_phot_flag = 1
                mir_phot_flag = 1
                if(all(np.isnan(phot_obj[opt_indices]))):
                    opt_phot_flag = 0
                if(all(np.isnan(phot_obj[nir_indices]))):
                    nir_phot_flag = 0
                if(all(np.isnan(phot_obj[mir_indices]))):
                    mir_phot_flag = 0
                
                # Load spectrum
                if(('.txt' in path_spec) or ('.dat' in  path_spec)):
                    data = np.loadtxt(path_spec, comments='#')
                    wavelength = data[:, 0]
                    flux = data[:, 1]
                    error = data[:, 2]

                elif('.fits' in path_spec):
                    hdulist = fits.open(path_spec)
                    wavelength = hdulist[0].data[0]
                    flux = hdulist[0].data[1]
                    error = hdulist[0].data[2]

                # Remove nans from data
                nan_indices = np.where(np.isnan(flux) == True)[0]
                wavelength = np.delete(wavelength, nan_indices)
                flux = np.delete(flux, nan_indices)
                error = np.delete(error, nan_indices)

                # Cut SED at edge points
                sed_cut_indices = np.where((wavelength >= 0.85) & (wavelength <= 2.45))[0]
                wavelength = wavelength[sed_cut_indices]
                flux = flux[sed_cut_indices]
                error = error[sed_cut_indices]

                # ---------------------- FLUX CALIBRATION OF SPECTRUM ---------------------- #
                # Get highest SNR photometry in NIR for spectrum calibration
                snr = 1.09/phot_obj_err[nir_indices]
                if(len(np.where(snr == np.nanmax(snr))[0])):
                    max_snr_index = np.where(snr == np.nanmax(snr))[0][0]

                    fit_phot = phot_obj[nir_indices][max_snr_index]
                    fit_phot_err = phot_obj_err[nir_indices][max_snr_index]
                    max_sn_perc_uncertainty = (fit_phot_err/fit_phot)*100
                    fit_wavelength = eff_wavelength[nir_indices][max_snr_index]
                    fit_filter_width = filter_width[nir_indices][max_snr_index]

                    # Get effective flux density for fit band
                    # First retrieve filter transmission profile
                    filters = ['PAN-STARRS_PS1.g', 'PAN-STARRS_PS1.r', 'PAN-STARRS_PS1.i', 'PAN-STARRS_PS1.z', 'PAN-STARRS_PS1.y', 
                            '2MASS_2MASS.J', '2MASS_2MASS.H', '2MASS_2MASS.Ks', 'MKO_NSFCam.J', 'MKO_NSFCam.H', 'MKO_NSFCam.K',
                            'WISE_WISE.W3', 'WISE_WISE.W4', 'Spitzer_IRAC.I1', 'Spitzer_IRAC.I2', 'UKIRT_UKIDSS.J', 'UKIRT_UKIDSS.H', 'UKIRT_UKIDSS.K', 'WISE_WISE.W1', 'WISE_WISE.W2']
                    filters_sorted = []
                    # Sort in wavelength order
                    for i in range(len(wave_sort)):
                        filters_sorted.append(filters[wave_sort[i]])
                    filters = filters_sorted

                    # Rearrange filter names from glob in order of wavelength
                    filter_fnames = glob.glob('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/Filter_Transmission/*.dat')
                    filter_fnames_sorted = []
                    for i in range(len(filters)):
                        for j in range(len(filter_fnames)):
                            if(filters[i] in filter_fnames[j]):
                                filter_fnames_sorted.append(filter_fnames[j])
                                break
                    filter_fnames = filter_fnames_sorted
                            
                    # Calculate synthetic flux density in all NIR bands
                    filter_fnames_opt = [filter_fnames[k] for k in opt_indices]
                    filter_fnames_nir = [filter_fnames[k] for k in nir_indices]
                    filter_fnames_mir = [filter_fnames[k] for k in mir_indices]
                    syn_flux_density_nir = []
                    for i in range(len(filter_fnames_nir)):
                        data_filter = np.genfromtxt(filter_fnames_nir[i])
                        wavelength_filter = data_filter.T[0]*1e-4 # angstrom to micron
                        throughput_filter = data_filter.T[1]

                        # Edge case where filter spans wavelength where there is no spectrum in NIR (J band). Set that synthetic magntiude to 0
                        # If that photometry also is highest S/N point then use the next highest S/N point
                        if(min(wavelength_filter) < min(wavelength)):
                            syn_flux_density_nir.append(np.nan)
                            if(max_snr_index == i):
                                # This approach is taken to find second max without running into NaN error
                                snr_temp = np.copy(snr)
                                snr_temp[max_snr_index] = np.nan
                                max_snr_index = np.where(snr_temp == np.nanmax(snr_temp))[0][0]
                        else:
                            # Next map spectrum to wavelength grid of filter transmission profile
                            f_spec_filt = interp1d(wavelength, flux, kind='cubic')
                            spec_flux_filt = f_spec_filt(wavelength_filter)

                            # Perform integration and normalization
                            syn_flux_density = np.trapz(y=spec_flux_filt*throughput_filter, x=wavelength_filter)/np.trapz(y=throughput_filter, x=wavelength_filter)
                            syn_flux_density_nir.append(syn_flux_density)
                        
                    syn_flux_density_nir = np.array(syn_flux_density_nir)

                    # Add a noise floor to the photometric uncertainties (choose point that highest S/N before noise floor was implemented)
                    # Don't replace NaNs
                    floor_indices = np.where((phot_obj_err < 0.01) & (~np.isnan(phot_obj_err)))[0]
                    phot_obj_err[floor_indices] = 0.01

                    # Convert apparent magnitude to flux density in all bands
                    # Pedagogical note: vega_mag is not vega's magnitude (because in AB System Vega magnitude is not 0) but in general is zero since comparison is with a zero magnitude star. zp_flux_vega is just the zero point flux of a given band in a given system and is not referred to as the vega zero point flux. Thus equations here are correct irrespective of magnitude system but the nomenclature used is wrong.
                    flux_phot = 10.0 ** (-0.4 * (phot_obj - vega_mag)) * zp_flux_vega
                    flux_phot_err = flux_phot*0.4*np.log(10)*phot_obj_err # From error propagation
                    
                    # Find Scaling Factor
                    # Single data point may have low SNR but when we do integration, the uncertainty is not significant
                    c_arr = []
                    c_err_arr = []
                    for fl in range(len(flux_phot[nir_indices])):
                        c = flux_phot[nir_indices][fl]/syn_flux_density_nir[fl]
                        c_err = flux_phot_err[nir_indices][fl]/syn_flux_density_nir[fl]
                        c_arr.append(c)
                        c_err_arr.append(c_err)

                    c_arr = np.array(c_arr)
                    c_err_arr = np.array(c_err_arr)

                    # # Max S/N calibration scale factor
                    # c_max = c_arr[max_snr_index]
                    # c_max_err = c_err_arr[max_snr_index]

                    # Weighted average
                    weights = 1/(c_err_arr**2)
                    c = np.nansum(c_arr*weights)/np.nansum(weights)
                    c_err = 1/np.sqrt(np.nansum(weights))
                    
                    # Create a noise floor for the scaling factor
                    if(c_err < 0.01*c):
                        c_err = 0.01*c
                
                    # Apparent Flux Calibrated Spectrum
                    flux_app_cal = flux*c
                    flux_app_cal_err = ((error*c)**2 + (flux*c_err)**2)**0.5
                
                    # Calculate percentage difference in photometry and spectra and significance
                    perc_diff = (np.abs(syn_flux_density_nir*c - flux_phot[nir_indices])/flux_phot[nir_indices])*100
                    sig_diff = (np.abs(syn_flux_density_nir*c - flux_phot[nir_indices])/flux_phot_err[nir_indices])
                    
                    # Apparent Bolometric Flux W/m^2
                    flux_app_bol_nir = np.trapz(y=flux_app_cal, x=wavelength)

                    # Monte Carlo uncalibrated flux and scaling factor for error estimate on flux_app_bol from integration
                    mc_size = 10000
                    sampled_flux = []
                    for i in range(len(flux)):
                        sampled_flux.append(np.random.normal(flux[i], error[i], mc_size))
                    sampled_flux = np.array(sampled_flux)

                    sampled_c = np.random.normal(c, c_err, mc_size)

                    # Compute bolometric apparent flux for each MC sampled case
                    sampled_bol_flux = []
                    for i in range(mc_size):
                        flux_bol_mc = np.trapz(y=sampled_flux[:, i]*sampled_c[i], x=wavelength)
                        sampled_bol_flux.append(flux_bol_mc)
                        
                    sampled_bol_flux = np.array(sampled_bol_flux)
                    flux_app_bol_nir_err = np.std(sampled_bol_flux, ddof=1)

                    # ------------------ MODEL ATMOSPHERE FITTING -------------------- #
                    # Note: R=75 OR R=37 OR R=150 => s=08, R=120 => s=05, R~200 => s=03
                    # First find resolution of spectrum to fit
                    R_spec = np.nan
                    s_spec = np.nan
                    if('R=' in path_spec):
                        R_spec = int(path_spec.split('_')[-2][2:])
                    elif('0.8prism' in path_spec):
                        s_spec = 0.8
                    elif('0.5prism' in path_spec):
                        s_spec = 0.5
                    elif('0.3prism' in path_spec):
                        s_spec = 0.3
                    elif(('.txt' in path_spec) or ('.dat' in path_spec)):
                        with open(path_spec) as file:
                            lines = file.readlines()
                            lines = [line.rstrip() for line in lines]

                        R_spec = int(lines[3].split(' ')[-1])
                    elif('.fits' in path_spec):
                        hdulist = fits.open(path_spec)
                        s_spec = float(hdulist[0].header['SLIT'][0:3])

                    # Get slit size
                    if((R_spec == 75) or (R_spec == 37) or (R_spec == 150) or (s_spec == 0.8)):
                        slit_spec = '08'
                    elif((R_spec == 120) or (s_spec == 0.5)):
                        slit_spec = '05'
                    elif((s_spec == 0.3) or (R_spec == 200)):
                        slit_spec = '03'

                    print(f'Spectral Resolution: {R_spec}')
                    print(f'Slit Size: {slit_spec}')
                        
                    # Retrive all model spectra filenames (Degraded R)
                    model_spec_paths = glob.glob(f'/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/model-set-degraded/*/R={slit_spec}/*.txt')
                    model_spec_paths = sorted(model_spec_paths) # Sorted in order of increasing temperature and for each temperature increasing log(g)

                    # Cut-off model at 3500 K 
                    model_spec_paths = model_spec_paths[:238]

                    # Load BT-Settl models (R = infinite)
                    # Cut-off model at 3500 K
                    model_inf_fnames = glob.glob('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/model-set/*/*')
                    model_inf_fnames = sorted(model_inf_fnames)[:238]

                    # Load the synthetic flux densities in all bands
                    # Cut-off model at 3500 K 
                    model_syn_flux = np.load('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/script-outputs/synthetic_flux_density_inf.npy')[:238]

                    # Set synthetic fluxes in NIR bands to NaN to exclude them since spectrum is available
                    model_syn_flux[:, nir_indices] = np.nan

                    # Set NIR photometry to NaN to avoid fitting to those (since spectra available)
                    flux_phot_nan_nir = np.copy(flux_phot)
                    flux_phot_nan_nir_err = np.copy(flux_phot_err)
                    flux_phot_nan_nir[nir_indices] = np.nan
                    flux_phot_nan_nir_err[nir_indices] = np.nan

                    # Find min chi-sq model fit
                    chi_sq_arr = []
                    chi_sq_arr_phot_only = []
                    chi_sq_arr_spec_only = []
                    rms_arr = []
                    rms_phot_only_arr = []
                    rms_spec_only_arr = []
                    min_chi_c_arr = []
                    pool_jobs = []
                    processes = 10
                    print(f'Number of CPUs requested = {processes}')
                    pool = multiprocessing.Pool(processes=processes)
                    print('Parallelizing spectral fitting ...')
                    # Fit each model grid and calculate chi-sq
                    for i in range(len(model_spec_paths)):
                        job = pool.apply_async(start_fit,(object_name, model_spec_paths[i], model_inf_fnames[i], flux_app_cal, model_syn_flux[i], flux_phot_nan_nir, flux_app_cal_err, flux_phot_nan_nir_err, wavelength))
                        pool_jobs.append(job)
                    
                    print('Fetching results ...')
                    for j, job in enumerate(pool_jobs):
                        result = job.get() 
                        chi_sq_arr.append(result[0])
                        min_chi_c_arr.append(result[1])
                        chi_sq_arr_phot_only.append(result[2])
                        chi_sq_arr_spec_only.append(result[3])
                        rms_arr.append(result[4])
                        rms_phot_only_arr.append(result[5])
                        rms_spec_only_arr.append(result[6])
                        
                    # Close pool to avoid accumulation in memory and file open error
                    pool.terminate()

                    # SAVE minimum chi-sq values to lists
                    best_fit_index = np.where(chi_sq_arr == np.min(chi_sq_arr))[0][0]

                    save_fit_error_analysis_arr = [np.min(chi_sq_arr), chi_sq_arr_phot_only[best_fit_index], chi_sq_arr_spec_only[best_fit_index], rms_arr[best_fit_index], rms_phot_only_arr[best_fit_index], rms_spec_only_arr[best_fit_index]]
                    np.save('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/model-fit-error-arrays/'+object_name+'.npy', save_fit_error_analysis_arr)

                    # MANUALLY ADDED INFINITE CHI-SQ FOR 1400 K logg 2.5 TO ENSURE RECTANGULAR GRID 
                    # Find model with minimum chi-square and the corresponding scale factor
                    min_chi_c_arr = np.array(min_chi_c_arr)
                    best_fit_index = np.where(chi_sq_arr == np.min(chi_sq_arr))[0][0]
                    best_fit_scale_factor = min_chi_c_arr[best_fit_index]

                    # Load best fit data
                    model_data = np.loadtxt(model_spec_paths[best_fit_index], skiprows=1, delimiter=',')
                    model_wavelength = model_data.T[0]*1e-4 # angstrom to micron
                    model_flux_density = model_data.T[1] # W/m^2/um

                    model_data_inf = np.genfromtxt(model_inf_fnames[best_fit_index])
                    model_wavelength_inf = model_data_inf.T[0] # micron
                    model_flux_density_inf = model_data_inf.T[1] # W/m^2/um

                    # Which photometry exists, only those synthetic photometry values should be plotted
                    synth_phot_indices = np.where(~np.isnan(flux_phot) & ~np.isnan(flux_phot_err))[0]

                    print(f'Chi-square for best-fit: {chi_sq_arr[best_fit_index]}')
                    print(f'Best Fit Model: {model_spec_paths[best_fit_index]}')

                    # Save what I need for plotting
                    np.save('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/plot_files/'+object_name+'.npy', [filter_fnames_opt, filter_fnames_nir, filter_fnames_mir, wavelength, flux_app_cal, flux_app_cal_err, eff_wavelength, flux_phot, flux_phot_err, opt_indices, nir_indices, mir_indices, twomass_indices, mko_indices, best_fit_index, max_snr_index, filter_width, model_wavelength, model_flux_density, best_fit_scale_factor, synth_phot_indices, model_syn_flux, R_spec, ukidss_indices, syn_flux_density_nir])

                    # Find cut-off of SpeX data and start integration from that point of infinite resolution spectrum
                    max_spex_wave = np.max(wavelength)
                    max_cut_index = np.where(np.abs(model_wavelength_inf - max_spex_wave) == np.min(np.abs(model_wavelength_inf - max_spex_wave)))[0][0]

                    # Apparent Bolometric Flux W/m^2 in MIR
                    flux_app_bol_mir = np.trapz(y=model_flux_density_inf[max_cut_index:]*best_fit_scale_factor, x=model_wavelength_inf[max_cut_index:])

                    # Find cut-off of SpeX data and start integration from that point of infinite resolution spectrum
                    min_spex_wave = np.min(wavelength)
                    min_cut_index = np.where(np.abs(model_wavelength_inf - min_spex_wave) == np.min(np.abs(model_wavelength_inf - min_spex_wave)))[0][0]

                    # Apparent Bolometric Flux W/m^2 in Optical
                    flux_app_bol_opt = np.trapz(y=model_flux_density_inf[:min_cut_index]*best_fit_scale_factor, x=model_wavelength_inf[:min_cut_index])

                    # Get parallax (there are cases where no DR3 parallax but formula column has value)
                    # plx_flag: 0 = Gaia DR3, 1 = plx formula column, 2 = distance formula column
                    plx_flag = 0
                    parallax = df_ucs['plx_Gaia'][index_obj_ucs]*1e-3 # mas to arcsecond
                    parallax_err = df_ucs['plxerr_Gaia'][index_obj_ucs]*1e-3 # mas to arcsecond
                    if(np.isnan(parallax)):
                        parallax = df_ucs['plx_formula'][index_obj_ucs]*1e-3 # mas to arcsecond
                        parallax_err = df_ucs['plxerr_formula'][index_obj_ucs]*1e-3 # mas to arcsecond
                        plx_flag = 1
                        
                    # Get distance 
                    distance = 1/parallax # pc
                    distance_error = parallax_err/(parallax**2)
                    
                    dist_source = ''
                    if(np.isnan(distance)):
                        distance = df_ucs['dist_formula'][index_obj_ucs] # pc
                        distance_error = df_ucs['disterr_formula'][index_obj_ucs]
                        dist_source = df_ucs['dist_formula_source'][index_obj_ucs]
                        plx_flag = 2
                        
                    # Bolometric flux
                    fbol_app = flux_app_bol_nir+flux_app_bol_mir+flux_app_bol_opt
                
                    # Convert to luminosity
                    distance_m = distance*3.086e16
                    distance_m_err = distance_error*3.086e16
                    L_bol = fbol_app*4*np.pi*(distance_m**2) # (pc -> m) Final answer in Watts
                    L_sun = 3.828e26 # Watts
                    log_L_bol_sun = np.log10(L_bol/L_sun)
                    L_bol_err = ((flux_app_bol_nir_err*4*np.pi*(distance_m**2))**2 + (8*np.pi*distance_m*fbol_app*distance_m_err)**2)**0.5
                    log_L_bol_sun_err = L_bol_err/(L_bol*np.log(10))
                    print(f'Bolometric Luminosity for {object_name}: {log_L_bol_sun} +- {log_L_bol_sun_err}')

                    # Extract logg and Teff value of best-fit model
                    fit_model_name = ''
                    if('ATMO' in model_inf_fnames[best_fit_index]):
                        logg_inf = model_inf_fnames[best_fit_index].split('_')[-3][2:]
                        teff_inf = model_inf_fnames[best_fit_index].split('_')[-4][1:]
                        fit_model_name = 'ATMO'
                    else:
                        logg_inf = model_inf_fnames[best_fit_index].split('_')[-2]
                        teff_inf = model_inf_fnames[best_fit_index].split('_')[-4]
                        fit_model_name = 'BT-Settl'
                
                    # Store flux calibrated spectrum + uncertainty, photometry + uncertainty, bolometric luminosity by object names, scaling factor for model fit, logg, teff of best fit model, apparent bolometric flux (in case parallax not known), chi-sq distribution of model fitting, store fitting flag information, flux in opt+ nir + mir, percentage and sigma difference in synthetic and calibrated NIR photometry, and the max s/n point's percentage uncertainty, slit size
                    np.save('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/data_products/'+object_name+'.npy', np.array([log_L_bol_sun, log_L_bol_sun_err, teff_inf, logg_inf, best_fit_scale_factor, fbol_app, wavelength, flux_app_cal, flux_app_cal_err, eff_wavelength, flux_phot, flux_phot_err, nir_phot_flag, mir_phot_flag, opt_phot_flag, chi_sq_arr, [flux_app_bol_opt, flux_app_bol_nir, flux_app_bol_mir], [perc_diff, sig_diff], max_sn_perc_uncertainty, slit_spec, plx_flag, dist_source, fit_model_name, c, c_err]))

                else:
                    print('NO IR PHOTOMETRY')

        else:
            print('Highest SNR Prism path does not exist!')

