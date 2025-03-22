import pylab as pl
import numpy as np
import random
from astropy.io import fits
import datetime
import pymysql
import os
import pandas as pd
import warnings
import scipy.special as ss
warnings.filterwarnings("ignore")
from astropy.io import fits
import matplotlib.pyplot as pl
import pickle
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

"""
Python routine to redden a spectra
"""
from natastro import utils
from natastro.pycasso2.reddening import calc_redlaw
import numpy as np
#-------------------------INPUTS-------------------------#
#--------------------------------------------------------#

def redden_tau(HaHb):
    """
    We apply redenning by calculate the Balmer decrement and using it in conjunction
    with a redenning law.
    F_0(lam) = F_obs(lam)*exp(tau_Lam)
    haHb  -- Balmer Decrement
    """
    #Take Cardelli, Clayton & Mathis (1989) attenuation law with $R_V = 3.1$
    qHb, qHa = calc_redlaw([4861, 6563], 'CCM', R_V=3.1)
    B = 2.87  # Standard balmer decrement value
    tauV = (qHb - qHa)**-1 * np.log(HaHb / B)
    return tauV


def redden_line(line, flux_init, tau):
    line = 10*line  # nm -> Ang
    q_l = calc_redlaw([line], 'CCM', R_V=3.1)[0]
    tau_l = tau*q_l
    flux_red = flux_init/np.exp(tau_l)
    return flux_red


# ------------------------------- FUNCTIONS --------------------------#
"""
Calculate the Flux of an emission line for SITELLE given the amplitude, broadening,
resolution, and wavenumber
"""
def ampToFlux(ampl, broad, res, wvn):
    """
    ampl - amplitude
    broad - broadening
    res - spectral resolution
    wvn - wavenumber of emission line (nm)
    """
    num = np.sqrt(2*np.pi)*ampl*broad
    den = ss.erf((2*(1e7/wvn)*broad)/(1.20671*res))
    flux = num/den
    return flux

def fluxToAmp(flux, broad, res, wvn):
    num = ss.erf((2*(1e7/wvn)*broad)/(1.20671*res))*flux
    den = np.sqrt(2*np.pi)*broad
    ampl = num/den
    return ampl

line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803,
              'SII6716': 671.647, 'SII6731': 673.085, 'OII3726': 372.603,
              'OII3729': 372.882, 'OIII4959': 495.891, 'OIII5007': 500.684,
              'Hbeta': 486.133}
# ------------------- IMPORTS -----------------------------------------#
# Set Input Parameters
name = '10k-red-broad'
output_dir = '/home/carterrhea/Dropbox/CFHT/Analysis-Paper2/SyntheticData/'+name+'/'
plot_dir = '/home/carterrhea/Dropbox/CFHT/Analysis-Paper2/SyntheticData/Plots/'+name+'/'
# Set observation parameters
resolution_SN3 = 5000
resolution_SN2 = 1000
vel_num = 2000  # Number of Velocity Values Sampled
broad_num = 1000  # Number of Broadening Values Sampled
num_syn = 50000  # Number of Synthetic Spectra
n_threads = 4  # Number of threads to use for computation
#-----------------------------------------------------------------------------------#
# Randomize values
# Sample velocity
vel_ = np.random.uniform(-200,200,vel_num)
# Sample broadening
broad_ = np.random.uniform(10,50,broad_num)
# Same resolution
res_SN3 = np.random.uniform(resolution_SN3-200, resolution_SN3, 200)
res_SN2 = np.random.uniform(resolution_SN2-100, resolution_SN2, 100)

# Now we need to get our emission lines for each filter
lines_sn3 = ['Halpha', 'NII6583', 'NII6548', 'SII6716', 'SII6731']
lines_sn3 = ['OIII5007', 'OIII4959', 'Hbeta']
lines_sn3 = ['OII3726', 'OII3729']
# Set fitting function
fit_function = 'sincgauss'
print('# -- Connecting to 3MdB -- #')
# We must alo get our flux values from 3mdb
# First we load in the parameters needed to login to the sql database
MdB_HOST='3mdb.astro.unam.mx'
MdB_USER='OVN_user'
MdB_PASSWD='oiii5007'
MdB_PORT='3306'
MdB_DBs='3MdBs'
MdB_DBp='3MdB'
MdB_DB_17='3MdB_17'
# Now we connect to the database
co = pymysql.connect(host=MdB_HOST, db=MdB_DB_17, user=MdB_USER, passwd=MdB_PASSWD)
print('# -- Obtaining Amplitudes -- #')
# Now we get the lines we want
print('# -- HII --#')
HII_ampls = pd.read_sql("select H__1_656281A as ha, N__2_654805A as n1, N__2_658345A as n2, \
                  S__2_673082A  as s2, S__2_671644A as s1, O__2_372603A as O2, O__3_500684A as O3, O__3_495891A as O3_2, \
                  H__1_486133A as hb, O__2_372881A as O2_2  \
                  from tab_17 \
                  where ref = 'BOND'"
                    , con=co)
print('# -- PNe --#')
PNe_ampls = pd.read_sql("select H__1_656281A as ha, O__3_500684A as O3, N__2_658345A as n2, N__2_654805A as n1, \
                  S__2_673082A  as s2, S__2_671644A as s1, H__1_486133A as hb, O__1_630030A as O1,  O__3_495891A as O3_2,   \
                  O__2_372603A as O2, O__2_372881A as O2_2 \
                  from tab_17 \
                  where ref = 'PNe_2020'"
                    , con=co)
co = pymysql.connect(host=MdB_HOST, db=MdB_DBs, user=MdB_USER, passwd=MdB_PASSWD)
print('# -- SNR --#')
SNR_ampls = pd.read_sql("""SELECT shock_params.shck_vel AS shck_vel,
                         shock_params.mag_fld AS mag_fld,
                         emis_VI.NII_6548 AS n1,
                         emis_VI.NII_6583 AS n2,
                         emis_VI.HI_6563 AS ha,
                         emis_VI.OIII_5007 AS O3,
                         emis_VI.OIII_4959 as O3_2,
                         emis_VI.HI_4861 AS hb,
                         emis_VI.SII_6716 AS s1,
                         emis_VI.SII_6731 AS s2,
                         emis_VI.OII_3726 AS O2,
                         emis_VI.OII_3729 as O2_2
                         FROM shock_params
                         INNER JOIN emis_VI ON emis_VI.ModelID=shock_params.ModelID
                         INNER JOIN abundances ON abundances.AbundID=shock_params.AbundID
                         WHERE emis_VI.model_type='shock'
                         AND abundances.name='Allen2008_Solar'
                         ORDER BY shck_vel, mag_fld;""", con=co)
# We will now filter out values that are non representative of our SIGNALS sample
print('# -- Starting Creation of Spectra -- #')
## We now can model the lines. For the moment, we will assume all lines have the same velocity and broadening
# Do this for randomized combinations of vel_ and broad_
ct = 0
#for spec_ct in range(num_syn):
def create_spectrum(spec_ct, ampls, ampls_str):
    spectrum = None  # Intialize
    pick_new = True
    # Randomly select velocity and broadening parameter and theta
    velocity = random.choice(vel_)
    broadening = random.choice(broad_)
    resolution_sn3 = random.choice(res_SN3)
    resolution_sn2 = random.choice(res_SN2)  # SN2 and SN1 since they have the same resoution
    # Randomly Select a M3db simulation
    while pick_new:
        sim_num = random.randint(0,len(ampls)-1)
        sim_vals = ampls.iloc[sim_num]
        # Redden Spectra
        Balmer_dec = np.random.uniform(2,6)  # Randomly select Balmer Decrement Value
        # Calculate tau value
        tauV = redden_tau(Balmer_dec)
        # Calculate Halpha redenned for comparisons

    # Calculate flux and line amplitudes for each line
    Ha_red_flux = redden_line(line_dict['Halpha'], ampToFlux(sim_vals['ha'], broadening, resolution_sn3, line_dict['Halpha']), tauV)
    Ha_red_ampl = fluxToAmp(Ha_red_flux, broadening, resolution_sn3, line_dict['Halpha'])
    n1_red_flux = redden_line(line_dict['NII6548'], ampToFlux(sim_vals['n1'], broadening, resolution_sn3, line_dict['NII6548']), tauV)
    n1_red_ampl = fluxToAmp(n1_red_flux, broadening, resolution_sn3, line_dict['NII6548'])
    n2_red_flux = redden_line(line_dict['NII6583'], ampToFlux(sim_vals['n2'], broadening, resolution_sn3, line_dict['NII6583']), tauV)
    n2_red_ampl = fluxToAmp(n2_red_flux, broadening, resolution_sn3, line_dict['NII6583'])
    s1_red_flux = redden_line(line_dict['SII6716'], ampToFlux(sim_vals['s1'], broadening, resolution_sn3, line_dict['SII6716']), tauV)
    s1_red_ampl = fluxToAmp(s1_red_flux, broadening, resolution_sn3, line_dict['SII6716'])
    s2_red_flux = redden_line(line_dict['SII6731'], ampToFlux(sim_vals['s2'], broadening, resolution_sn3, line_dict['SII6731']), tauV)
    s2_red_ampl = fluxToAmp(s2_red_flux, broadening, resolution_sn3, line_dict['SII6731'])
    O2_red_flux = redden_line(line_dict['OII3726'], ampToFlux(sim_vals['O2'], broadening, resolution_sn2, line_dict['OII3726']), tauV)
    O2_red_ampl = fluxToAmp(O2_red_flux, broadening, resolution_sn2, line_dict['OII3726'])
    O2_2_red_flux = redden_line(line_dict['OII3729'], ampToFlux(sim_vals['O2_2'], broadening, resolution_sn2, line_dict['OII3729']), tauV)
    O2_2_red_ampl = fluxToAmp(O2_2_red_flux, broadening, resolution_sn2, line_dict['OII3729'])
    O3_red_flux = redden_line(line_dict['OIII5007'], ampToFlux(sim_vals['O3'], broadening, resolution_sn2, line_dict['OIII5007']), tauV)
    O3_red_ampl = fluxToAmp(O3_red_flux, broadening, resolution_sn2, line_dict['OIII5007'])
    O3_2_red_flux = redden_line(line_dict['OIII4959'], ampToFlux(sim_vals['O3_2'], broadening, resolution_sn2, line_dict['OIII4959']), tauV)
    O3_2_red_ampl = fluxToAmp(O3_2_red_flux, broadening, resolution_sn2, line_dict['OIII4959'])
    Hb_red_flux = redden_line(line_dict['Hbeta'], ampToFlux(sim_vals['hb'], broadening, resolution_sn2, line_dict['Hbeta']), tauV)
    Hb_red_ampl = fluxToAmp(Hb_red_flux, broadening, resolution_sn2, line_dict['Hbeta'])

    # We can now create all of the spectra
    spectrum_axis_sn3, spectrum_sn3 = Spectrum(lines_sn3, fit_function,
         [Ha_red_ampl, n2_red_ampl, n1_red_ampl, s1_red_ampl, s2_red_ampl],
         velocity, broadening, 'SN3', resolution_sn3, SNR).create_spectrum()
    spectrum_axis_sn2, spectrum_sn2 = Spectrum(lines_sn2, fit_function,
         [O3_red_ampl, O3_2_red_ampl, Hb_red_ampl],
         velocity, broadening, 'SN2', resolution_sn2, SNR).create_spectrum()
    spectrum_axis_sn1, spectrum_sn1 = Spectrum(lines_sn1, fit_function,
          [O2_red_ampl, O2_2_red_ampl],
          velocity, broadening, 'SN1', resolution_sn2, SNR).create_spectrum()

    # Now normalize to the max value. This must be done for later comparison with real data
    # Since we are normalizing the spectra to the max value
    spec_max = np.max(spectrum_SN3)
    if np.max(spectrum_SN2) > spec_max: spec_max = np.max(spectrum_SN2)
    if np.max(spectrum_SN1) > spec_max: spec_max = np.max(spectrum_SN1)
    spectrum_SN3 = np.array([spec_/spec_max for spec_ in spectrum_SN3])
    spectrum_SN2 = np.array([spec_/spec_max for spec_ in spectrum_SN2])
    spectrum_SN1 = np.array([spec_/spec_max for spec_ in spectrum_SN1])
    # Gather information to make Fits file
    col1 = fits.Column(name='Wavenumber', format='E', array=spectrum_axis_SN3)
    col2 = fits.Column(name='Flux', format='E', array=spectrum_SN3)
    cols = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(cols)
    # Header info
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Carter Rhea'
    hdr['COMMENT'] = "Synthetic Spectrum SN3 Number: %i"%spec_ct
    hdr['TIME'] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr['VELOCITY'] = velocity
    hdr['BROADEN'] = broadening
    hdr['THETA'] = theta
    hdr['SIM'] = ampls_str
    hdr['SIM_NUM'] = spec_ct  # sim_num
    hdr['RES'] = resolution_sn3
    hdr['SNR'] = SNR
    hdr['Halpha'] = Ha_red_flux
    hdr['NII6548'] = n1_red_flux
    hdr['NII6583'] = n2_red_flux
    hdr['SII6716'] = s1_red_flux
    hdr['SII6731'] = s2_red_flux
    hdr['tauV'] = tauV
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary, hdu])
    hdul.writeto(output_dir+ampls_str+'/Spectrum_SN3_%i.fits'%spec_ct, overwrite=True)


    #SN2 FITS
    # Gather information to make Fits file
    col1 = fits.Column(name='Wavenumber', format='E', array=spectrum_axis_SN2)
    col2 = fits.Column(name='Flux', format='E', array=spectrum_SN2)
    cols = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(cols)
    # Header info
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Carter Rhea'
    hdr['COMMENT'] = "Synthetic Spectrum SN2 Number: %i"%spec_ct
    hdr['TIME'] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr['VELOCITY'] = velocity
    hdr['BROADEN'] = broadening
    hdr['THETA'] = theta
    hdr['SIM'] = ampls_str
    hdr['SIM_NUM'] = spec_ct  # sim_num
    hdr['RES'] = resolution_sn2
    hdr['OIII5007'] = O3_red_flux
    hdr['OIII4959'] = O3_2_red_flux
    hdr['Hbeta'] = Hb_red_flux
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary, hdu])
    hdul.writeto(output_dir+ampls_str+'/Spectrum_SN2_%i.fits'%spec_ct, overwrite=True)


    #SN1 FITS
    # Gather information to make Fits file
    col1 = fits.Column(name='Wavenumber', format='E', array=spectrum_axis_SN1)
    col2 = fits.Column(name='Flux', format='E', array=spectrum_SN1)
    cols = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(cols)
    # Header info
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Carter Rhea'
    hdr['COMMENT'] = "Synthetic Spectrum SN1 Number: %i"%spec_ct
    hdr['TIME'] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr['VELOCITY'] = velocity
    hdr['BROADEN'] = broadening
    hdr['THETA'] = theta
    hdr['SIM'] = ampls_str
    hdr['SIM_NUM'] = spec_ct  # sim_num
    hdr['RES'] = resolution_sn2
    hdr['OII3726'] = O2_red_flux
    hdr['OII3729'] = O2_2_red_flux
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary, hdu])
    hdul.writeto(output_dir+ampls_str+'/Spectrum_SN1_%i.fits'%spec_ct, overwrite=True)


    # save figure
    if spec_ct%1000 == 0:
        fig, axs = pl.subplots(3,1, figsize=(10,8))
        axs[0].plot([1e7/spec for spec in spectrum_axis_SN1], spectrum_SN1)
        axs[1].plot([1e7/spec for spec in spectrum_axis_SN2], spectrum_SN2)
        axs[2].plot([1e7/spec for spec in spectrum_axis_SN3], spectrum_SN3)
        print('Ha/Hb initial: '+str(sim_vals['ha']/sim_vals['hb']))
        print('Ha/Hb: '+str(Ha_red_flux/Hb_red_flux))
        pl.savefig(output_dir+ampls_str+'_'+str(spec_ct)+'.png')
        pl.clf()



for ampl, ampl_str in zip([HII_ampls, PNe_ampls, SNR_ampls],['BOND', 'PNe_2020', 'SNR_2008']):
    print("# -- Creating %s Spectra -- #"%ampl_str)
    if not os.path.exists(output_dir+ampl_str):
        os.makedirs(output_dir+ampl_str)
    #for spec_ct in tqdm(range(num_syn)):
    #    create_spectrum(spec_ct, ampl, ampl_str)
    Parallel(n_jobs=n_threads)(delayed(create_spectrum)(spec_ct, ampl, ampl_str) for spec_ct in tqdm(range(num_syn)))
