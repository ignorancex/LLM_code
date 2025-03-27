# Python script to perform Bayesian Rejection Sampling with the SM08 and BHAC15 evolutionary model tracks given luminosity and age
# Author: Aniket Sanghi
# Contact: asanghi@caltech.edu

# Package Imports
import numpy as np
from scipy.interpolate import griddata, interp1d
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Perform rejection sampling
def rejection_sampling(log_age_obj, log_age_err_obj, min_t, max_t, min_l, max_l, log_age_low, log_age_high, logt_model, logl_model, log_r_rs_model, log_m_ms_model, log_teff_model, logg_model, field):
    """
    Args:
        log_age_obj (float): Mean log10(age) of object for a normal age distribution
        log_age_err_obj (float): 1 sigma uncertainty in log10(age) of object for a normal distribution
        min_t (float): Smallest age value in evolutionary model grid
        max_t (float): Largest age value in evolutionary model grid
        min_l (float): Smallest luminosity value in evolutionary model grid
        max_l (float): Largest luminosity value in evolutionary model grid
        log_age_low (float): Lower bound log10(age) for uniform distribution
        log_age_high (float): Upper bound log10(age) for uniform distribution
        logt_model (numpy array): log(age) values for evolutionary model tracks
        logl_model (numpy array): log(luminosity) values for evolutionary model tracks
        log_r_rs_model (numpy array): log(radius) values for evolutionary model tracks
        log_m_ms_model (numpy array): log(mass) values for evolutionary model tracks
        log_teff_model (numpy array): log(teff) values for evolutionary model tracks
        logg_model (numpy array): log(gravity) values for evolutionary model tracks
        field (int): Indicator variable for youth category

    Returns:
        interp_vals_rad, interp_vals_mass, interp_vals_teff, interp_vals_g (numpy arrays): Interpolated radii, masses, temperatures, and gravities at sampled age and luminosity input values
    """

    # Sample ~1e6 values
    nsamp = 1e6

    # Sample ~1e6 logt values
    # Uniform Age Distribution
    if(~np.isnan(log_age_low)):
        if(log_age_low < min_t):
            min_t_samp = min_t*1
        else:
            min_t_samp = log_age_low*1
        if(log_age_high > max_t):
            max_t_samp = max_t*1
        else:
            max_t_samp = log_age_high*1
        samp_t = np.random.uniform(low=min_t_samp, high=max_t_samp, size=int(nsamp))
        
    # Field Age Distribution
    elif(field == 1):
        if(np.log10(10e6) < min_t):
            min_t_samp = min_t*1
        else:
            min_t_samp = np.log10(10e6)*1
        if(np.log10(10e9) > max_t):
            max_t_samp = max_t*1
        else:
            max_t_samp = np.log10(10e9)*1
        samp_t = np.random.uniform(low=min_t_samp, high=max_t_samp, size=int(nsamp))
        nsamp = len(samp_t)

    # FLD-G Age Distribution
    elif(field == 2):
        if(np.log10(300e6) < min_t):
            min_t_samp = min_t*1
        else:
            min_t_samp = np.log10(10e6)*1
        if(np.log10(10e9) > max_t):
            max_t_samp = max_t*1
        else:
            max_t_samp = np.log10(10e9)*1
        samp_t = np.random.uniform(low=min_t_samp, high=max_t_samp, size=int(nsamp))
        nsamp = len(samp_t)

    # Otherwise sample full age range of models
    else:
        samp_t = np.random.uniform(low=min_t, high=max_t, size=int(nsamp))
        
    # Sample ~1e6 logl values
    samp_l = np.random.uniform(low=min_l, high=max_l, size=int(nsamp))

    # Compute chi-sq for samples
    if((~np.isnan(log_age_low)) or (field == 1) or (field == 2)):
        # Uniform, Field, or FLD-G Distribution
        chi_sq = ((samp_l-log_L_bol_sun)/log_L_bol_sun_err)**2
    else:
        # Normal Age Distribution
        chi_sq = ((samp_l-log_L_bol_sun)/log_L_bol_sun_err)**2 + ((samp_t-log_age_obj)/log_age_err_obj)**2

    # Convert to probability normalized by minimum chi-sq sample
    p = np.exp(-0.5*(chi_sq-np.nanmin(chi_sq)))

    # Multiply by probability of drawing sample from (truncated) DL17 for (FLD-G) Field objects
    if((field == 1) or (field == 2)):
        find_samp_t = (10**samp_t)/1e9
        p_dl17 = np.copy(samp_t)
        if(field == 1):
            p_dl17[np.where((find_samp_t >= 0) & (find_samp_t < 0.15))[0]] = 0.081
        p_dl17[np.where((find_samp_t >= 0.15) & (find_samp_t < 1))[0]] = 0.2
        p_dl17[np.where((find_samp_t >= 1) & (find_samp_t < 2))[0]] = 0.161
        p_dl17[np.where((find_samp_t >= 2) & (find_samp_t < 3))[0]] = 0.119
        p_dl17[np.where((find_samp_t >= 3) & (find_samp_t < 5))[0]] = 0.166
        p_dl17[np.where((find_samp_t >= 5) & (find_samp_t < 7))[0]] = 0.129
        p_dl17[np.where((find_samp_t >= 7) & (find_samp_t <= 10))[0]] = 0.144
        p = p*p_dl17/np.nanmax(p*p_dl17)

    # Draw ~1e6 samples from [0, 1] uniformly for rejection test
    samples = np.random.uniform(low=0, high=1, size=int(nsamp))
    accepted_indices = np.where(samples <= p)[0]
    accepted_l = samp_l[accepted_indices]
    accepted_t = samp_t[accepted_indices]
        
    # Create xi array (x = log1, y = logt) for interpolation procedure
    xi = []
    for i in range(len(accepted_l)):
        xi.append((accepted_l[i], accepted_t[i]))

    # Create points array (x = logl, y = logt): Radius, mass, teff, logg are the values array
    points = []
    for i in range(len(logt_model)):
        points.append((logl_model[i], logt_model[i]))
        
    # Interpolate model at accepted ages and luminosities
    interp_vals_rad = griddata(points=points, values=log_r_rs_model, xi=xi, method='linear')
    interp_vals_mass = griddata(points=points, values=log_m_ms_model, xi=xi, method='linear')
    interp_vals_teff = griddata(points=points, values=log_teff_model, xi=xi, method='linear')
    interp_vals_g = griddata(points=points, values=logg_model, xi=xi, method='linear')

    return interp_vals_rad, interp_vals_mass, interp_vals_teff, interp_vals_g

# Perform calculations of final fundamental parameter values
def property_calculation(interp_vals_rad, interp_vals_mass, interp_vals_teff, interp_vals_g):
    """_summary_

    Args:
        interp_vals_rad (numpy array): Interpolated radius value
        interp_vals_mass (numpy array): Interpolated mass value
        interp_vals_teff (numpy array): Interpolated temperature value
        interp_vals_g (numpy array): Interpolated gravity value

    Returns:
        r_jup_measure, r_jup_measure_err, m_jup_measure, m_jup_measure_err, teff_measure, teff_measure_err, logg_measure, logg_measure_err: Median parameter measurements and corresponding uncertainties
    """

    # Convert to RJup
    r_rs = 10**(interp_vals_rad)
    r_jup = r_rs*6.96e10/6.9911e9 # values in cm
    r_jup_measure = np.nanmedian(r_jup)
    r_jup_measure_err = np.nanstd(r_jup, ddof=1)

    # Convert to MJup
    m_ms = 10**(interp_vals_mass)
    m_jup = m_ms/0.000954265748 
    m_jup_measure = np.nanmedian(m_jup)
    m_jup_measure_err = np.nanstd(m_jup, ddof=1)
    
    # Convert to linear K
    teff_k = 10**(interp_vals_teff)
    teff_measure = np.nanmedian(teff_k)
    teff_measure_err = np.nanstd(teff_k, ddof=1)

    logg_measure = np.nanmedian(interp_vals_g)
    logg_measure_err = np.nanstd(interp_vals_g, ddof=1)

    return r_jup_measure, r_jup_measure_err, m_jup_measure, m_jup_measure_err, teff_measure, teff_measure_err, logg_measure, logg_measure_err

# NOTE: The following filepaths and formats are unique to this dataset and file directory set-up
# Retrieve all data products
new_uncertainties = np.load('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/script-outputs/total_uncertainty_sorted.npy')
df_ages = pd.read_csv('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/Sample-Creation/UltracoolSheet - PRIVATE - AgeValues.csv')
df_ucs = pd.read_csv('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/Sample-Creation/UltracoolSheet - PRIVATE - Main.csv')
names_ucs = df_ucs['name']

# Load all data
fnames = glob.glob('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/data_products/*.npy')
fnames = sorted(fnames)
with open('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/Sample-Creation/final_sample.txt') as f:
    lines = [line.rstrip('\n') for line in f]

# Get Names from file names
object_name_arr = []
fnames_final = []
for i in range(len(fnames)):
    unjoined = fnames[i].split('/')[-1].split('.')[:-1]
    joined = unjoined[0]
    for j in range(1, len(unjoined)):
        joined += '.'+unjoined[j]
    if(joined in lines):
        object_name_arr.append(joined)
        fnames_final.append(fnames[i])
fnames = fnames_final

radius_arr = []
mass_arr = []
teff_arr = []
logg_arr = []
age_string = []

# Loop through all objects
for index in range(len(fnames)): # len(fnames)
    obj_search = object_name_arr[index]

    # Find object in UCS
    index_obj_ucs = np.nan
    for k in range(len(names_ucs)):
        if(obj_search in names_ucs[k]):
            index_obj_ucs = k*1
            break

    ########## AGE FORMATTING ##########
    agevalue_string = df_ucs['age_category'][index_obj_ucs].split('?')[0].split('!')[0]
    
    find_index = np.nan
    for s_index in range(len(df_ages['name'])):
        if(agevalue_string == df_ages['name'][s_index]):
            find_index = s_index
            break

    age_Gyr	= df_ages['age_Gyr'][s_index]
    age_upp_err	= df_ages['age_upp_err'][s_index]
    age_low_err	= df_ages['age_low_err'][s_index]
    age_range_type = df_ages['age_range_type'][s_index]

    # Object Properties
    # Gaussian Distributed Age: Set to NaN for now
    age_yr_gauss = np.nan
    age_err_yr_gauss = np.nan
    log_age_obj = np.nan
    log_age_err_obj = np.nan
    
    # Uniformly Distributed Age (Range provided or YMG asymmetric uncertainties converted to a range)
    age_low = np.nan
    age_high = np.nan
    log_age_low = np.nan
    log_age_high = np.nan

    # For model limits
    highest_age = np.nan
    lowest_age = np.nan

    # Set field flag based on object (0 = Not Field; 1 = Field; 2 = FLD-G => use for DL17 cut-off determination)
    field = 0
    if('companion' in agevalue_string):
        age_string.append('\\nodata')

    elif('Field' in agevalue_string):
        field = 1
        lowest_age = np.log10(10e6) # yr
        highest_age = np.log10(10e9) # yr
        age_string.append('DL17')

    elif('FLD-G' in agevalue_string):
        field = 2
        lowest_age = np.log10(300e6) # yr
        highest_age = np.log10(10e9) # yr
        age_string.append('Truncated DL17')

    # Uniform age distributions
    elif(('uniform' in age_range_type) or ('Uniform' in age_range_type) or (('normal' in age_range_type) and (age_upp_err != age_low_err)) or (('Normal' in age_range_type) and (age_upp_err != age_low_err))):

        # Deal with cases where 0-10 Gyr is age range by Deac14b
        if((age_Gyr-age_low_err)*1e9 < 10e6):
            age_low = 10e6
        else:
            age_low = (age_Gyr-age_low_err)*1e9

        age_high = (age_Gyr+age_upp_err)*1e9
        log_age_low = np.log10(age_low) 
        log_age_high = np.log10(age_high) 
        lowest_age = log_age_low*1
        highest_age = log_age_high*1
        age_print = format(round(age_low/1e9, 3), '.3f')
        age_err_print = format(round(age_high/1e9, 3), '.3f')
        age_string.append(f'{age_print}-{age_err_print} Gyr')

    # Normal age distributions
    elif(('normal' in age_range_type or 'Normal' in age_range_type) and age_upp_err==age_low_err):
        age_yr_gauss = age_Gyr*1e9
        age_err_yr_gauss = age_upp_err*1e9
        highest_age = np.log10(age_yr_gauss + age_err_yr_gauss)
        lowest_age = np.log10(age_yr_gauss - age_err_yr_gauss)
        log_age_obj = np.log10(age_yr_gauss) 
        log_age_err_obj = age_err_yr_gauss/(age_yr_gauss*np.log(10))
        age_print = format(round(age_yr_gauss/1e9, 3), '.3f')
        age_err_print = format(round(age_err_yr_gauss/1e9, 3), '.3f')
        age_string.append(f'${age_print} \pm {age_err_print}$ Gyr')

    # Single-value: Treated as normal with 5% uncertainty
    elif('single_value' in age_range_type):
        age_yr_gauss = age_Gyr*1e9
        age_err_yr_gauss = 0.05*age_Gyr*1e9
        highest_age = np.log10(age_yr_gauss + age_err_yr_gauss)
        lowest_age = np.log10(age_yr_gauss - age_err_yr_gauss)
        log_age_obj = np.log10(age_yr_gauss) 
        log_age_err_obj = age_err_yr_gauss/(age_yr_gauss*np.log(10))
        age_print = format(round(age_yr_gauss/1e9, 3), '.3f')
        age_err_print = format(round(age_err_yr_gauss/1e9, 3), '.3f')
        age_string.append(f'${age_print} \pm {age_err_print}$ Gyr')

    else:
        age_string.append('\\nodata')

    # Load luminosity measurement
    data = np.load(fnames[index], allow_pickle=True)
    log_L_bol_sun = data[0] 
    log_L_bol_sun_err = new_uncertainties[index]
    
    # For model limits
    highest_lbol = log_L_bol_sun + log_L_bol_sun_err
    lowest_lbol = log_L_bol_sun - log_L_bol_sun_err

    if(highest_age > 10):
        highest_age = 10

    if(age_string[-1] == '\\nodata'):
        print(f'No age available for Object {index}: {obj_search}')
        radius_arr.append([np.nan, np.nan, np.nan, np.nan])
        mass_arr.append([np.nan, np.nan, np.nan, np.nan])
        teff_arr.append([np.nan, np.nan, np.nan, np.nan])
        logg_arr.append([np.nan, np.nan, np.nan, np.nan])

    else:
        # Load grids
        data_BHAC = np.genfromtxt('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/evo-model/BHAC15_tracks+structure.txt', comments='#')

        # Load data for BHAC grid
        logt_BHAC = data_BHAC[:, 1] # yr
        logl_BHAC = data_BHAC[:, 3] # Lbol/Lsun
        min_t_BHAC = np.nanmin(logt_BHAC)
        max_t_BHAC = np.nanmax(logt_BHAC)
        min_l_BHAC = np.nanmin(logl_BHAC)
        max_l_BHAC = np.nanmax(logl_BHAC)

        # Get Radius, Mass, Teff, logg
        log_r_rs_BHAC = np.log10(data_BHAC[:, 5])
        log_m_ms_BHAC = np.log10(data_BHAC[:, 0])
        log_teff_BHAC = np.log10(data_BHAC[:, 2])
        logg_BHAC = data_BHAC[:, 4] # logg cgs

        # Define boundary of model: Only care about bottom and right edge for BHAC. There are no objects that will break the top edge, the ages are not young enough to break left edge. 
        logt_separated = []
        logl_separated = []
        temp_t = []
        temp_l = []
        for i in range(1, len(log_m_ms_BHAC)):
            if(log_m_ms_BHAC[i] == log_m_ms_BHAC[i-1]):
                temp_t.append(logt_BHAC[i])
                temp_l.append(logl_BHAC[i])
            else:
                logt_separated.append(temp_t)
                logl_separated.append(temp_l)
                temp_t = []
                temp_l = []
                temp_t.append(logt_BHAC[i])
                temp_l.append(logl_BHAC[i])

        # Bottom edge 
        bot_edge_t = logt_separated[0]
        bot_edge_l = logl_separated[0]
        bot_edge_t = np.array(bot_edge_t)
        bot_edge_l = np.array(bot_edge_l)
    
        # Load data for SM08 grid
        data_SM08 = np.genfromtxt('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/evo-model/hybrid_isochrones.dat', comments='#')
        logt_SM08 = np.log10(data_SM08[:, 0]*1e9) # yr
        logl_SM08 = data_SM08[:, 3] # Lbol/Lsun
        min_t_SM08 = np.nanmin(logt_SM08)
        max_t_SM08 = np.nanmax(logt_SM08)
        min_l_SM08 = np.nanmin(logl_SM08)
        max_l_SM08 = np.nanmax(logl_SM08)

        # Get Get Radius, Mass, Teff, logg
        log_r_rs_SM08 = np.log10(data_SM08[:, 5])
        log_m_ms_SM08 = np.log10(data_SM08[:, 1])
        log_teff_SM08 = np.log10(data_SM08[:, 2])
        logg_SM08 = data_SM08[:, 4] # logg cgs

        # SM08 Organization
        log_m_ms_SM08 = np.array(log_m_ms_SM08)
        logt_SM08 = np.array(logt_SM08)
        logl_SM08 = np.array(logl_SM08)

        sort_indices = np.argsort(log_m_ms_SM08)

        log_m_ms_SM08_sorted = log_m_ms_SM08[sort_indices]
        logt_SM08_sorted = logt_SM08[sort_indices]
        logl_SM08_sorted = logl_SM08[sort_indices]

        logt_separated_SM08 = []
        logl_separated_SM08 = []
        temp_t_SM08 = []
        temp_l_SM08 = []
        for i in range(1, len(log_m_ms_SM08_sorted)):
            if(log_m_ms_SM08_sorted[i] == log_m_ms_SM08_sorted[i-1]):
                temp_t_SM08.append(logt_SM08_sorted[i])
                temp_l_SM08.append(logl_SM08_sorted[i])
            else:
                temp_t_SM08 = np.array(temp_t_SM08)
                temp_l_SM08 = np.array(temp_l_SM08)
                sort_separated_indices = np.argsort(temp_t_SM08)
                logt_separated_SM08.append(temp_t_SM08[sort_separated_indices])
                logl_separated_SM08.append(temp_l_SM08[sort_separated_indices])
                temp_t_SM08 = []
                temp_l_SM08 = []
                temp_t_SM08.append(logt_SM08_sorted[i])
                temp_l_SM08.append(logl_SM08_sorted[i])

        # Top edge SM08
        top_edge_t_SM08 = logt_separated_SM08[-1]
        top_edge_l_SM08 = logl_separated_SM08[-1]
        
        # Perform rejection sampling
        interp_vals_rad_BHAC, interp_vals_mass_BHAC, interp_vals_teff_BHAC, interp_vals_g_BHAC = rejection_sampling(log_age_obj, log_age_err_obj, min_t_BHAC, max_t_BHAC, min_l_BHAC, max_l_BHAC, log_age_low, log_age_high, logt_BHAC, logl_BHAC, log_r_rs_BHAC, log_m_ms_BHAC, log_teff_BHAC, logg_BHAC, field)

        interp_vals_rad_SM08, interp_vals_mass_SM08, interp_vals_teff_SM08, interp_vals_g_SM08 = rejection_sampling(log_age_obj, log_age_err_obj, min_t_SM08, max_t_SM08, min_l_SM08, max_l_SM08, log_age_low, log_age_high, logt_SM08, logl_SM08, log_r_rs_SM08, log_m_ms_SM08, log_teff_SM08, logg_SM08, field)

        # Perform fundamental parameter calculations
        r_jup_measure_BHAC, r_jup_measure_err_BHAC, m_jup_measure_BHAC, m_jup_measure_err_BHAC, teff_measure_BHAC, teff_measure_err_BHAC, logg_measure_BHAC, logg_measure_err_BHAC = property_calculation(interp_vals_rad_BHAC, interp_vals_mass_BHAC, interp_vals_teff_BHAC, interp_vals_g_BHAC)

        r_jup_measure_SM08, r_jup_measure_err_SM08, m_jup_measure_SM08, m_jup_measure_err_SM08, teff_measure_SM08, teff_measure_err_SM08, logg_measure_SM08, logg_measure_err_SM08 = property_calculation(interp_vals_rad_SM08, interp_vals_mass_SM08, interp_vals_teff_SM08, interp_vals_g_SM08)
        
        # Make sure extrapolations did not occur
        f_int = interp1d(bot_edge_t, bot_edge_l)
        if(lowest_age > bot_edge_t[-1] and lowest_lbol > bot_edge_l[-1]):
            print('Outside bottom track age range but above floor')

        elif(lowest_age > bot_edge_t[-1] and lowest_lbol < bot_edge_l[-1]):
            print('Outside bottom track age range but below floor')
            r_jup_measure_BHAC = np.nan
            r_jup_measure_err_BHAC = np.nan
            m_jup_measure_BHAC = np.nan
            m_jup_measure_err_BHAC = np.nan
            teff_measure_BHAC = np.nan
            teff_measure_err_BHAC = np.nan
            logg_measure_BHAC = np.nan
            logg_measure_err_BHAC = np.nan

        elif(f_int(lowest_age) > lowest_lbol):
            r_jup_measure_BHAC = np.nan
            r_jup_measure_err_BHAC = np.nan
            m_jup_measure_BHAC = np.nan
            m_jup_measure_err_BHAC = np.nan
            teff_measure_BHAC = np.nan
            teff_measure_err_BHAC = np.nan
            logg_measure_BHAC = np.nan
            logg_measure_err_BHAC = np.nan
            
        f_int = interp1d(top_edge_t_SM08, top_edge_l_SM08)
        if(highest_age < top_edge_t_SM08[0] and highest_lbol < top_edge_l_SM08[0]):
            print('Outside top track age range but below floor')

        elif(highest_age < top_edge_t_SM08[0] and highest_lbol > top_edge_l_SM08[0]):
            if(index != 56 and index != 586 and index != 626):
                print('Outside top track age range but above floor')
                r_jup_measure_SM08 = np.nan
                r_jup_measure_err_SM08 = np.nan
                m_jup_measure_SM08 = np.nan
                m_jup_measure_err_SM08 = np.nan
                teff_measure_SM08 = np.nan
                teff_measure_err_SM08 = np.nan
                logg_measure_SM08 = np.nan
                logg_measure_err_SM08 = np.nan

        elif(f_int(highest_age) < highest_lbol):
            r_jup_measure_SM08 = np.nan
            r_jup_measure_err_SM08 = np.nan
            m_jup_measure_SM08 = np.nan
            m_jup_measure_err_SM08 = np.nan
            teff_measure_SM08 = np.nan
            teff_measure_err_SM08 = np.nan
            logg_measure_SM08 = np.nan
            logg_measure_err_SM08 = np.nan
        
        # Append Final Parameter Values
        radius_arr.append([r_jup_measure_BHAC, r_jup_measure_err_BHAC, r_jup_measure_SM08, r_jup_measure_err_SM08])
        mass_arr.append([m_jup_measure_BHAC, m_jup_measure_err_BHAC, m_jup_measure_SM08, m_jup_measure_err_SM08])
        teff_arr.append([teff_measure_BHAC, teff_measure_err_BHAC, teff_measure_SM08, teff_measure_err_SM08])
        logg_arr.append([logg_measure_BHAC, logg_measure_err_BHAC, logg_measure_SM08, logg_measure_err_SM08])

        print(f'Object {index}: {obj_search} Complete!')
    
# Save fundamental parameters
np.save('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/rejection-sampling/physical_param_evo_prop.npy', [object_name_arr, radius_arr, mass_arr, teff_arr, logg_arr])

# Save age strings for table creation purposes
with open('/Users/asanghi/Documents/Undergraduate-Projects/IfA_REU/fund-prop-analysis/script-outputs/age_strings.txt', 'w') as f:
    for i, line in enumerate(age_string):
        f.write(f"{line}\n")