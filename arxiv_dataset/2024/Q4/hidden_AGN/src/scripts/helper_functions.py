import numpy as np
from astropy.table import Table, join
import astropy.units as u
from progress.bar import Bar
import paths
from astropy.stats import median_absolute_deviation as apy_mad
import os
from astropy.io import fits
from scipy.optimize import curve_fit
## cosmology to match Kondapally and Cochrane
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

## error propagation functions
def add_sub_error( eX, eY ):
    result = np.sqrt( np.power( eX, 2. ) + np.power( eY, 2. ) )
    return( result )

def mult_div_error( resval, vals, e_vals ):
    result = resval * np.sqrt( np.sum( np.power( e_vals/vals, 2. ) ) )
    return( result )

def radio_power( obs_flux_Jy, redshift, spectral_index=-0.8):
    flux_cgs = obs_flux_Jy * 1e-23
    DL = cosmo.luminosity_distance(redshift).value * 3.086e24 ## convert to cm
    power_cgs = 4. * np.pi * np.power(DL,2.) * flux_cgs / np.power( 1+redshift, 1.+spectral_index )
    ## convert to W/Hz
    power = power_cgs * 1e-7
    return( power )

def trapezoid_rule( yvals, dx ):
    first_term = dx / 2. * yvals[0]
    last_term = dx / 2. * yvals[-1]
    middle_terms = dx * np.nansum(yvals[1:-2])
    add_terms = first_term + middle_terms + last_term
    return( add_terms )

def trapezoid_error( e_yvals, dx ):
    const = 0.5 * dx
    errs = add_sub_error( e_yvals[0:-1], e_yvals[1:] )
    individual_errs = const * errs
    overall_err = np.sqrt( np.sum( np.power( individual_errs, 2. ) ) )
    return( overall_err )

def midpoint_rule( yvals, dx ):
    sum_vals = np.nansum( yvals * dx )
    return( sum_vals )

def log10_when_zeros( vals ):
    zero_idx = np.where( vals <= 0 )[0]
    vals[zero_idx] = 1.
    new_vals = np.log10(vals)
    new_vals[zero_idx] = 0.
    return(new_vals)

def get_values( x, y, dy, cutoff=0.7, useidx=True ):
    if useidx:
        non_zero = np.where( np.logical_and( y != 0.0, np.isfinite(y) ) )[0]
        xvals = x[non_zero]
        yvals = y[non_zero]
        dyvals = dy[non_zero]
    else:
        xvals = x
        yvals = y
        dyvals = dy
    ## first find the values that are valid
    n_hidx = np.where( dyvals < cutoff )[0]
    ## the ends will be where the problems are
    hidx = []
    if not n_hidx[0] == 0:
        ## start of the list
        hidx.append(np.arange(0,n_hidx[0]+1))
    if not n_hidx[-1] == len(xvals)-1:
        ## end of the list
        hidx.append(np.arange(n_hidx[-1],len(xvals)))
    return( xvals, yvals, dyvals, n_hidx, hidx )
    

######################################################################
## Functions for calculating brightness temp

def get_tb_information( mycat, im_weight=0.5, maj_lim=0.4, min_lim=0.3, T_e=1e4, alpha=-0.7, ref_freqs=np.array([0.003,0.01,0.03,0.1,0.3,1,3])*1e9, freqs_GHz=np.arange( 1e-3, 1e2, 1e-3 ), use_z=True ):
    freqs = freqs_GHz*1e9  ## Hz

    ## step 1: sizes
    beam_solid_angle = get_beam_sizes( mycat, im_weight=im_weight, maj_lim=maj_lim, min_lim=min_lim )
    ## step 2: flux density per solid angle in units of mJy per arcsec squared
    compact_flux_per_SA_peak = mycat['Peak_flux']*1e3 / beam_solid_angle
    compact_flux_per_SA_total = mycat['Total_flux']*1e3 / beam_solid_angle

    if use_z:
        print('Using redshift information to find T_b-identified AGN')
        ## use redshift information redshift information to find high T_b sources
        peak_agn = find_agn_withz( compact_flux_per_SA_peak, mycat['Z_BEST'], T_e=T_e, rf_array=ref_freqs, freq=144e6, alpha=alpha)
        total_agn = find_agn_withz( compact_flux_per_SA_total, mycat['Z_BEST'], T_e=T_e, rf_array=ref_freqs, freq=144e6, alpha=alpha)
    else:
        ## High T_b sources, assuming nu_0 = 3 GHz
        peak_agn = find_agn( compact_flux_per_SA_peak, T_e=T_e, rf_array=ref_freqs, freq=144e6, alpha=alpha)
        total_agn = find_agn( compact_flux_per_SA_total, T_e=T_e, rf_array=ref_freqs, freq=144e6, alpha=alpha)

    agn_final = np.unique( np.concatenate((peak_agn,total_agn)))
    print('Number of identified AGN: ', len(agn_final), ' which is ', len(agn_final)/len(mycat)*100, ' percent of sources.')
    print(len(peak_agn), 'are identified via peak brightness')
    print(len(total_agn), 'are identified via total brightness')

    ## add some information to the table
    ## column for peak (=1) or total (=2) T_b identification
    tb_from = np.zeros(len(mycat))
    tb_from[total_agn] = 2.
    tb_from[peak_agn] = 1. ## overwrites total
    ## add single column for flux density per solid angle 
    compact_flux_per_SA = np.copy(compact_flux_per_SA_total)
    compact_flux_per_SA[peak_agn] = compact_flux_per_SA_peak[peak_agn]

    ## add columns to table ... 
    mycat.add_column( compact_flux_per_SA, name='flux_per_SA' )
    mycat.add_column( tb_from, name='tb_from' )

    ## return the flux density per solid angle and whether it was identified using peak or total
    return( mycat )

def get_beam_solid_angle( theta1, theta2 ):
    ## for a 2D Gaussian: https://www.cv.nrao.edu/~sransom/web/Ch3.html eqn 3.118
    result =  np.pi / (4. * np.log(2.)) * theta1 * theta2 
    return( result )

def get_beam_sizes( mycat, im_weight=0.5, maj_lim=0.4, min_lim=0.3 ):
    ## where the deconvolved resolution is smaller than the limiting resolution, use the limiting resolution
    ## See Radcliffe et al. (2018) Eqn. 6 (also Lobanov 2005)
    lim_factor = np.power( 2., 2-0.5*im_weight ) * np.sqrt( np.log(2.)/np.pi * np.log( mycat['Peak_flux']/mycat['Isl_rms'] / ( mycat['Peak_flux']/mycat['Isl_rms'] -1. ) ) )
    maj_lim_theory = maj_lim * lim_factor / 60. / 60. ## convert to degrees
    min_lim_theory = min_lim * lim_factor / 60. / 60. ## convert to degrees
    maj_idx = np.where( mycat['DC_Maj'] < maj_lim_theory )[0]
    min_idx = np.where( mycat['DC_Min'] < min_lim_theory )[0]
    beam_solid_angle = get_beam_solid_angle( mycat['DC_Maj']*60.*60., mycat['DC_Min']*60.*60. )
    return(beam_solid_angle)

def find_agn( flux_per_SA, T_e=1e4, rf_array=np.array([0.003,0.01,0.03,0.1,0.3,1,3])*1e9, freq=144e6, alpha=-0.8 ):
    ## predict brightness temps at frequency
    flux_per_SA_rfs = []
    for rf in rf_array:        
        tau_norm = 1. 
        opt_depth_term = 1. - np.exp( - tau_norm * ( freq / rf )**-2.1 )
        tb = T_e * opt_depth_term * ( 1. + 10. * (freq/1e9)**(0.1+alpha) ) 
        mJyperasec2 = tb * (freq)**2. / 1.38e24 * 1e3
        flux_per_SA_rfs.append(mJyperasec2)
    maxval = np.max(np.array(flux_per_SA_rfs)) 
    idx = np.where(flux_per_SA >= maxval)[0]
    return( idx )

def find_agn_withz( flux_per_SA_vec, redshift_vec, T_e=1e4, rf_array=np.array([0.003,0.01,0.03,0.1,0.3,1,3])*1e9, freq=144e6, alpha=-0.8 ):
    idx = []
    for i in np.arange(0,len(flux_per_SA_vec)):
        flux_per_SA = flux_per_SA_vec[i]
        if np.isfinite(flux_per_SA):
            redshift = redshift_vec[i]    
            ## predict brightness temps at frequency
            flux_per_SA_rfs = []
            for rf in rf_array:        
                tau_norm = 1. 
                opt_depth_term = 1. - np.exp( - tau_norm * ( freq / rf )**-2.1 )
                tb = T_e * opt_depth_term * ( 1. + 10. * (freq/1e9)**(0.1+alpha) ) 
                mJyperasec2 = tb * (freq/(1.+redshift))**2. / 1.38e24 * 1e3
                flux_per_SA_rfs.append(mJyperasec2)
            maxval = np.max(np.array(flux_per_SA_rfs))
            if flux_per_SA >= maxval:
                idx.append(i)
    return( idx )

######################################################################
## Functions for doing star formation / AGN separation

def calculate_Lrad_from_sfr_mass( sfr, Mstar ):
    ## mass is given in log10(mass)
    ## Smith et al 2021:
    ## log10(L150) = (0.9+-0.01)*log10(SFR)+(0.33+-0.04)*log10(M/10^10)+22.22+-0.02
    logLrad = 0.9 * sfr + 0.33 * np.log10( np.power( 10., Mstar) / np.power( 10., 10. ) ) + 22.22
    return(logLrad)

def calculate_SFR_from_Lrad_mass( Lrad, Mstar ):
    ## mass is given in log10(mass)
    ## Smith et al 2021:
    ## log10(SFR) = ( log10(L150) - (22.22+-0.02) - (0.33+-0.04)*log10(M/10^10) ) / (0.9+-0.01)

    ## to avoid errors, don't calculate where Lrad = 0 or where Mstar < 0
    avoid_idx = np.where( np.logical_or( Lrad <= 0, Mstar <= 1 ) )[0]
    Lrad[avoid_idx] = 1.
    Mstar[avoid_idx] = 1.
    logsfr = ( np.log10(Lrad) - 22.22 - 0.33*np.log10( np.power( 10., Mstar) / np.power( 10., 10. ) ) ) / 0.9
    ## re-set the nans
    logsfr[avoid_idx] = np.nan
    return(logsfr)

def Lrad_sfr_relation( xvals, a, b, c ):
    sfr, mass = xvals
    ## Smith et al 2021:
    ## log10(L150) = (0.9+-0.01)*log10(SFR)+(0.33+-0.04)*log10(M/10^10)+22.22+-0.02
    ## log10(SFR) = ( log10(L150) - (22.22+-0.02) - (0.33+-0.04)*log10(M/10^10) ) / (0.9+-0.01)
    yvals = a * sfr + b * mass + c
    return(yvals)

def linear( x, m, b ):
    y = m * x + b
    return(y)

def find_ridgeline( sfr, lrad, min_num=1000, sfrmin=-0.8, nbins=29 ):
    ## log10 (L144 [W Hz−1 ]) = 22.24 + 1.08×log10 (SFR [M yr−1 ])  # from Best et al. 2023
    sfrbins = sfrmin + np.arange(0,nbins)*0.2 
    lrad_bins = np.zeros(len(sfrbins))
    n_sfrbin = np.zeros(len(sfrbins)) 

    ## select objects which lie in small range in SFR, widen if number of objects is too small
    for i in np.arange(0,nbins):
        offset = 0.1
        idx = np.where( np.logical_and( sfr >= sfrbins[i]-offset, sfr <= sfrbins[i]+offset ) )[0]
        if len(idx) < min_num:
            offset = 0.2
            idx = np.where( np.logical_and( sfr >= sfrbins[i]-offset, sfr <= sfrbins[i]+offset ) )[0]
            if len(idx) < min_num:
                offset = 0.3
                idx = np.where( np.logical_and( sfr >= sfrbins[i]-offset, sfr <= sfrbins[i]+offset ) )[0]
                if len(idx) < min_num:
                    offset = 0.4
                    idx = np.where( np.logical_and( sfr >= sfrbins[i]-offset, sfr <= sfrbins[i]+offset ) )[0]
                else:
                    print('bin is still too small.')
        n_sfrbin[i] = len(idx)
        ## get the SFR and radio luminosity for the bin; note here that these should both be log10 values
        tmp_vals = lrad[idx] #- sfr[idx]
        ## now go through Lrad 
        nlradbins = 1200
        check_lrad = 18.0 + 0.01*np.arange(0,nlradbins)
        lrad_count = np.zeros(len(check_lrad))
        #print(len(idx))
        for j in np.arange(0,len(check_lrad)):
            idx1 = np.where( np.logical_and( tmp_vals >= check_lrad[j]-0.05, tmp_vals <= check_lrad[j]+0.05 ) )[0]
            lrad_count[j] = len(idx1)
        #print(np.sum(lrad_count))
        max_idx = np.where( lrad_count == np.max(lrad_count) )[0]
        #print(max_idx)
        lrad_bins[i] = check_lrad[int(np.median(max_idx))]
    ## derive fit parameters
    ## linear fit with: lrad = param0 + param1*sfr
    #nonzero = np.where( n_sfrbin > 0 )[0]
    #popt = curve_fit( linear, sfrbins[nonzero], lrad_bins[nonzero], p0=[0,0] )
    return( sfrbins, lrad_bins )
    

def do_SFR_AGN_separation( mycat ):
    ## will need SED classifications
    SFG_idx = np.where( mycat['Overall_class'] == 'SFG' )[0]
    not_SFG_idx = np.asarray( [ i for i in np.arange(0,len(mycat)) if i not in SFG_idx ] )
    
    RG_idx = np.where( np.logical_or( mycat['Overall_class'] == 'HERG', mycat['Overall_class'] == 'LERG' ) )[0]
    RQ_idx = np.where( mycat['Overall_class'] == 'RQAGN' )[0]
    AGN_idx = np.union1d(RG_idx, RQ_idx)
    not_AGN_idx = np.asarray( [ i for i in np.arange(0,len(mycat)) if i not in AGN_idx ] )

    ## doing it this way includes unclassified ... 
    #AGN_idx = np.asarray( [ i for i in np.arange(0,len(mycat)) if i not in SFG_idx ])

    ## ONLY VALID FOR T_b IDENTIFIED AND UNRESOLVED
    tb_idx = np.where( np.logical_and( mycat['tb_from'] > 0.0, mycat['Resolved'] == 'U' ) )[0]
    not_excess_idx = np.where( mycat['Radio_excess'] < 0.7 )[0]
    combined_idx = np.intersect1d(tb_idx,not_excess_idx)
    ## separate star formation and AGN flux densities
    ## initialise column for AGN
    ## radio excess: total_flux_dr
    agn_flux = np.copy( mycat['Total_flux_dr'] )
    e_agn_flux = np.copy( mycat['E_Total_flux_dr'] )
    ## set the agn_flux and e_agn_flux to zero for non-agn
    agn_flux[not_AGN_idx] = 0.
    e_agn_flux[not_AGN_idx] = 0.

    ## tb_from > 0, unresolved, not radio excess: peak_flux
    agn_flux[combined_idx] = mycat['Peak_flux'][combined_idx]
    e_agn_flux[combined_idx] = mycat['E_Peak_flux'][combined_idx]
    ## initialise column for SF
    ## tb_from = 0: total_flux_dr
    sf_flux = np.copy( mycat['Total_flux_dr'] )
    e_sf_flux = np.copy( mycat['Total_flux_dr'] )
    ## set the sf_flux and e_agn_flux to zero for AGN
    sf_flux[not_SFG_idx] = 0.
    e_sf_flux[not_SFG_idx] = 0.

    ## tb_from > 0, unresolved, not radio excess: total_flux_dr - peak_flux
    ## NOTE: if unclassified sources are T_b identified AGN, this will add them back in!
    sf_flux[combined_idx] = mycat['Total_flux_dr'][combined_idx] - mycat['Peak_flux'][combined_idx]
    e_sf_flux[combined_idx] = add_sub_error( mycat['E_Total_flux_dr'][combined_idx], mycat['E_Peak_flux'][combined_idx])
    mycat.add_column( agn_flux, name='AGN_flux' )
    mycat.add_column( e_agn_flux, name='E_AGN_flux' )
    mycat.add_column( sf_flux, name='SF_flux' )
    mycat.add_column( e_sf_flux, name='E_SF_flux' )
    return( mycat )

######################################################################
## Functions for calculating RLFs

def get_RLFs( vmaxes, zmin, zmax, lmin=20.5, lmax=27, dl=0.3, si=-0.7 ):
    ## get indices for galaxy identifications
    SFG_idx = np.where( vmaxes['Overall_class'] == 'SFG' )[0]
    Cochrane_SFG_idx = np.where( np.logical_or( vmaxes['Overall_class'] == 'SFG', vmaxes['Overall_class'] == 'RQAGN' ) )[0]
    ## set the SF galaxy index to match Cochrane's selection
    SFG_idx = Cochrane_SFG_idx

    RG_idx = np.where( np.logical_or( vmaxes['Overall_class'] == 'HERG', vmaxes['Overall_class'] == 'LERG' ) )[0]
    RQ_idx = np.where( vmaxes['Overall_class'] == 'RQAGN' )[0]
    AGN_idx = np.union1d(RG_idx, RQ_idx)
    ## set the AGN galaxy index to match Kondapally's selection
    AGN_idx = RG_idx

    ## RLF bins
    redshift_bins = np.array([zmin,zmax])
    lum_bins = np.arange( lmin, lmax, dl ) # + np.log10( np.power( (144./1400.), si ) )

    ## get radio luminosities
    Lrad = radio_power( vmaxes['Total_flux_dr'], vmaxes['Z_BEST'], spectral_index=si )
    agn_Lrad = radio_power( vmaxes['AGN_flux'], vmaxes['Z_BEST'], spectral_index=si )
    sf_Lrad = radio_power( vmaxes['SF_flux'], vmaxes['Z_BEST'], spectral_index=si )
    ## calculate the SFR
    sfrs = calculate_SFR_from_Lrad_mass( sf_Lrad, vmaxes['Mass_cons'] )

    ## take the log
    log10_Lrad = np.log10(Lrad)
    agn_log10_Lrad = log10_when_zeros(agn_Lrad)
    sf_log10_Lrad = log10_when_zeros(sf_Lrad)

    ## by galaxy
    gal_sfg_log10_Lrad = log10_Lrad[SFG_idx]
    gal_rg_log10_Lrad = log10_Lrad[AGN_idx]
    gal_sfg_vmax = vmaxes['vmax'][SFG_idx]
    gal_rg_vmax = vmaxes['vmax'][AGN_idx]


    ## calculate RLFs
    rhos = []
    agn_rhos = []
    sf_rhos = []
    gal_sfg_rhos = []
    gal_rg_rhos = []
    for i in np.arange(1,len(lum_bins)):
        ## change base from 10 to e to match units in Kondapally and Cochrane
        delta_log_L = (lum_bins[i] - lum_bins[i-1]) * np.log(10.)
        lum_idx = np.where(np.logical_and( log10_Lrad >= lum_bins[i-1], log10_Lrad < lum_bins[i] ) )[0]
        agn_lum_idx = np.where(np.logical_and( agn_log10_Lrad >= lum_bins[i-1], agn_log10_Lrad < lum_bins[i] ) )[0]
        sf_lum_idx = np.where(np.logical_and( sf_log10_Lrad >= lum_bins[i-1], sf_log10_Lrad < lum_bins[i] ) )[0]
        gal_sfg_idx = np.where( np.logical_and( gal_sfg_log10_Lrad >= lum_bins[i-1], gal_sfg_log10_Lrad < lum_bins[i] ) )[0]
        gal_rg_idx = np.where( np.logical_and( gal_rg_log10_Lrad >= lum_bins[i-1], gal_rg_log10_Lrad < lum_bins[i] ) )[0]
        if len(lum_idx) > 0:
            rho = np.log10( np.sum( 1. / vmaxes['vmax'][lum_idx] ) / delta_log_L ) 
        else:
            rho = 0
        if len(agn_lum_idx) > 0:
            agn_rho = np.log10( np.sum( 1. / vmaxes['agn_vmax'][agn_lum_idx] ) / delta_log_L ) 
        else:
            agn_rho = 0
        if len(sf_lum_idx) > 0:
            sf_rho = np.log10( np.sum( 1. / vmaxes['sf_vmax'][sf_lum_idx] ) / delta_log_L ) 
        else:
            sf_rho = 0
        if len(gal_sfg_idx) > 0:
            gal_sf_rho = np.log10( np.sum( 1. / gal_sfg_vmax[gal_sfg_idx] ) / delta_log_L )
        else:
            gal_sf_rho = 0
        if len(gal_rg_idx) > 0:
            gal_rg_rho = np.log10( np.sum( 1. / gal_rg_vmax[gal_rg_idx] ) / delta_log_L )
        else:
            gal_rg_rho = 0

        rhos.append(rho)
        agn_rhos.append(agn_rho)
        sf_rhos.append(sf_rho)
        gal_sfg_rhos.append(gal_sf_rho)
        gal_rg_rhos.append(gal_rg_rho)

    lum_func = np.asarray(rhos)
    agn_lum_func = np.asarray(agn_rhos)
    sf_lum_func = np.asarray(sf_rhos)
    gal_agn_lum_func = np.asarray(gal_rg_rhos)
    gal_sf_lum_func = np.asarray(gal_sfg_rhos)
    return( lum_bins, lum_func, agn_lum_func, sf_lum_func, gal_agn_lum_func, gal_sf_lum_func )

def random_resample( agn_lum_func, sf_lum_func, gal_agn_lum_func, gal_sf_lum_func, vmaxes, zmin, zmax, lmin=20.5, lmax=27, dl=0.3, si=-0.7, nsamp=1000 ):

    ## randomly re-sample 1000 times
    print('randomly resampling to get uncertainties')
    nsamp = 1000
    agn_lfs = np.zeros((len(agn_lum_func),nsamp))
    sf_lfs = np.zeros((len(agn_lum_func),nsamp))
    g_agn_lfs = np.zeros((len(agn_lum_func),nsamp))
    g_sf_lfs = np.zeros((len(agn_lum_func),nsamp))
    fracs = np.zeros(nsamp)

    for i in np.arange(0,nsamp):
        np.random.seed(i)
        idx = np.random.randint(low=0,high=len(vmaxes),size=len(vmaxes))
        fracs[i] = float(len(np.unique(idx))) / float(len(vmaxes))
        lb, lf, agn_lf, sf_lf, g_agn_lf, g_sf_lf = get_RLFs( vmaxes[idx], zmin, zmax, lmin=lmin, lmax=lmax, dl=dl, si=si )
        agn_lfs[:,i] = agn_lf
        sf_lfs[:,i] = sf_lf
        g_agn_lfs[:,i] = g_agn_lf
        g_sf_lfs[:,i] = g_sf_lf


    ## as I'm sampling with replacement, need to scale by the size of the resulting random sample
    tmp = np.tile( agn_lum_func, (nsamp,1) ).transpose()
    e_agn_lum_func = np.std( np.sqrt(fracs)*(agn_lfs-tmp),axis=1)
    tmp = np.tile( sf_lum_func, (nsamp,1) ).transpose()
    e_sf_lum_func = np.std( np.sqrt(fracs)*(sf_lfs-tmp),axis=1)
    tmp = np.tile( gal_agn_lum_func, (nsamp,1) ).transpose()
    e_gal_agn_lum_func = np.std( np.sqrt(fracs)*(g_agn_lfs-tmp), axis=1 )
    tmp = np.tile( gal_sf_lum_func, (nsamp,1) ).transpose()
    e_gal_sf_lum_func = np.std(np.sqrt(fracs)*(g_sf_lfs-tmp),axis=1)

    ## replace small values with 0.03
    e_agn_lum_func[np.where(e_agn_lum_func < 0.03)[0]] = 0.03
    e_sf_lum_func[np.where(e_sf_lum_func < 0.03)[0]] = 0.03
    e_gal_agn_lum_func[np.where(e_gal_agn_lum_func < 0.03)[0]] = 0.03
    e_gal_sf_lum_func[np.where(e_gal_sf_lum_func < 0.03)[0]] = 0.03

    return( e_agn_lum_func, e_sf_lum_func, e_gal_agn_lum_func, e_gal_sf_lum_func )


######################################################################
## Functions for calculating vmaxes

def effective_area( rms_file ):
    tmp = os.path.basename( rms_file )
    outfile = paths.data / tmp.replace( '.fits', '_effective_areas.fits' ) 
    print(outfile)
    if os.path.exists( outfile ):
        t = Table.read( outfile, format='fits' )
        noise_bins = t['noise_bin']
        eff_area_deg2 = t['eff_area_deg2']
    else:
        ## Effective areas as function of flux density 
        rms_image = fits.open( rms_file )
        rms = rms_image[0].data.flatten()
        ## pixel scales -- this is in degrees
        pixscale = np.abs(rms_image[0].header['CDELT1'])
        pixarea = np.power(pixscale, 2.)
        ## first get where this is finite
        rms_idx = np.where(np.isfinite(rms))
        ## get cumulative distribution of effective areas
        rms = rms[rms_idx]  ## cuts down on size of rms to make it faster
        noise_min = np.min(rms)
        noise_max = np.max(rms)
        binsize = 1. * np.power( 10., -(len(str(1./noise_min).split('.')[0])+1) )
        noise_bins = np.arange(noise_min, noise_max, binsize)
        npix = []
        for nb in noise_bins:
            tmp_idx = np.where(rms <= nb)
            npix.append(len(tmp_idx[0]))
        eff_area_deg2 = np.asarray(npix) * pixarea
        t = Table()
        t.add_column( noise_bins, name='noise_bin' )
        t.add_column( eff_area_deg2, name='eff_area_deg2' )
        t.write( outfile, format='fits' )
    return( noise_bins, eff_area_deg2 )

def get_source_vmax( power, scaling_factor, power_to_flux_factor, V_DLs, completeness, fluxdens, solid_angle, noise_bins, noise_min, sigma_cut = 5. ):
    ## find the flux at all possible redshifts
    flux_at_z = power_to_flux_factor * power 
    ## find the maximum redshift at which it can be observed       
    noise_diff = np.where( flux_at_z >= sigma_cut*noise_min )[0]
    ## if this is in the middle of the redshift bin, use an index to limit to max redshift
    if len(noise_diff) < len(flux_at_z):
        flux_at_z = power_to_flux_factor[noise_diff] * power 
        V_DLs = V_DLs[noise_diff]
    ## initialise some empty lists
    solid_angle_at_z = []
    completeness_at_z = []
    ## loop over the fluxes at different redshift
    for j in np.arange(0,len(flux_at_z)):
        ## find the solid angle for which this S(z) is observable
        idx = np.where(flux_at_z[j] >= sigma_cut*noise_bins)[0]
        solid_angle_at_z.append(solid_angle[np.max(idx)])
        ## find the completeness correction for this S(z), using the scaling factor to find the completeness for the galaxy, not the activity
        comp_check = flux_at_z[j] * scaling_factor - fluxdens
        idx = np.where( np.abs(comp_check) == np.min( np.abs(comp_check) ) )[0]
        completeness_at_z.append(completeness[idx[0]])
    ## calculate theta_z
    theta_sz = np.asarray(solid_angle_at_z) * np.asarray(completeness_at_z)
    integrand = V_DLs * theta_sz
    vmax = np.nansum( integrand )
    return(vmax)

def get_vmax( lotss, field, col_suffix='', zmin=0.003, zmax=0.3, dz=0.0001, si=-0.8, sigma_cut=5., rms_image='', cochrane='', kondapally='', test=False ):
    
    nsrc = len(lotss)
    print('Starting with {:s} sources in the area of coverage'.format(str(nsrc)))

    ## to select the rested flux density columns
    fluxcol = 'Total_flux{:s}'.format(col_suffix)
    fluxerrcol = 'E_Total_flux{:s}'.format(col_suffix)

    ## remove things with no redshifts, i.e. those which are masked
    good_z = np.where( np.logical_not(lotss['Z_BEST'].mask) )[0]
    lotss = lotss[good_z]
    ## remove the mask to avoid warnings later on
    lotss['Z_BEST'] = np.ma.getdata(lotss['Z_BEST'])

    nsrc = len(lotss)
    print('Continuing with {:s} sources with unmasked redshifts.'.format(str(nsrc)))

    ## limit to the redshift range requested
    z_idx = np.where( np.logical_and( lotss['Z_BEST'] >= zmin, lotss['Z_BEST'] <= zmax  ) )[0]
    lotss = lotss[z_idx]

    nsrc = len(lotss)
    print('Continuing with {:s} sources in the redshift range of {:s} to {:s}'.format(str(nsrc),str(zmin),str(zmax)))

    if test:
        ## randomly select 200 sources
        test_idx = np.random.choice(np.arange(0,len(lotss)),200)
        lotss = lotss[test_idx]
        nsrc = len(lotss)
        print('Test is true, continuing with {:s} sources.'.format(str(nsrc)))        

    ## get solid angle
    noise_bins, eff_area_deg2 = effective_area( rms_image )
    eff_area_sr = eff_area_deg2 * np.power( ( np.pi / 180. ), 2. )
    solid_angle = eff_area_sr / ( 4. * np.pi )
    ## convert the noise bins from Jy to W/Hz/m^2
    noise_bins = noise_bins * 1e-26
    noise_min = np.min(noise_bins)

    ## calculate intrinsic powers: first convert from Jy to W/Hz/m^2
    fluxes = lotss[fluxcol] * 1e-26
    flux_errors = lotss[fluxerrcol] * 1e-26
    agn_fluxes = lotss['AGN_flux'] * 1e-26
    agn_flux_errors = lotss['E_AGN_flux'] * 1e-26
    sf_fluxes = lotss['SF_flux'] * 1e-26
    sf_flux_errors = lotss['E_SF_flux'] * 1e-26
    redshifts = lotss['Z_BEST']
    ## calculate the power
    tmp = cosmo.luminosity_distance(redshifts)
    ## convert to metres so units are consistent
    DL = tmp.to(u.m).value
    source_factors = 4. * np.pi * np.power( DL, 2. ) / np.power((1.+redshifts), (1+si))
    power = source_factors * fluxes
    power_error = source_factors * flux_errors
    agn_power = source_factors * agn_fluxes
    agn_power_error = source_factors * agn_flux_errors
    sf_power = source_factors * sf_fluxes
    sf_power_error = source_factors * sf_flux_errors

    ## calculate distances and volumes 
    nbins = int(round((zmax+dz-zmin)/dz))
    zbins = np.arange(0,nbins)*dz+zmin
    tmp = cosmo.comoving_volume(zbins)  ## these are in Mpc3
    V_DLs = tmp[1:] - tmp[0:-1] ## length of one per bin
    zbin_cens = np.arange(zmin+0.5*dz,zmax,dz)
    tmp = cosmo.luminosity_distance(zbin_cens)
    DLs = tmp.to(u.m).value

    ## keep some calculations out of the loop
    DL_fac = 4. * np.pi * np.power( DLs, 2. )
    redshift_kcorr = np.power( (1.+zbin_cens), (1+si) ) 
    power_to_flux_factor = redshift_kcorr / DL_fac 

    ## now loop over every source to get the vmax
    vmaxes = []
    agn_vmaxes = []
    sf_vmaxes = []
    tmp = field.capitalize()
    pb = Bar('Processing', max=len(lotss))
    for i in np.arange(0,len(lotss)):
        if lotss['Overall_class'][i] == 'SFG':
            field_idx = [ i for i,cn in enumerate(cochrane.colnames) if tmp[0] in cn ]
            colname = np.asarray(cochrane.colnames)[field_idx][0]
            completeness = cochrane[colname]
            completeness_fluxDens = cochrane['FluxDensity_mJy']*1e-29
        else:
            field_idx = [ i for i,cn in enumerate(kondapally.colnames) if tmp[0] in cn ]
            colname = np.asarray(kondapally.colnames)[field_idx][0]
            completeness = kondapally[colname]
            completeness_fluxDens = kondapally['FluxDensity_mJy']*1e-29
        try:
            completeness = completeness.filled(0)
        except:
            completeness = completeness
        ## overall vmax
        vmax = get_source_vmax( power[i], 1.0, power_to_flux_factor, V_DLs, completeness, completeness_fluxDens, solid_angle, noise_bins, noise_min, sigma_cut = sigma_cut )
        if agn_power[i] == 0:
            agn_vmax = 0.*np.power(u.Mpc,3)
        elif agn_power[i] != power[i]:
            agn_vmax = get_source_vmax( agn_power[i], power[i]/agn_power[i], power_to_flux_factor, V_DLs, completeness, completeness_fluxDens, solid_angle, noise_bins, noise_min, sigma_cut = sigma_cut )
        else:
            agn_vmax = vmax
        if sf_power[i] == 0.:
            sf_vmax = 0.*np.power(u.Mpc,3)
        elif sf_power[i] != power[i]:
            sf_vmax = get_source_vmax( sf_power[i], power[i]/sf_power[i], power_to_flux_factor, V_DLs, completeness, completeness_fluxDens, solid_angle, noise_bins, noise_min, sigma_cut = sigma_cut )                
        else:
            sf_vmax = vmax
        vmaxes.append(vmax)
        agn_vmaxes.append(agn_vmax)
        sf_vmaxes.append(sf_vmax)
      
        pb.next()
    pb.finish()

    lotss.add_column( vmaxes, name='vmax' )
    lotss.add_column( agn_vmaxes, name='agn_vmax' )
    lotss.add_column( sf_vmaxes, name='sf_vmax' )
    return( lotss )
