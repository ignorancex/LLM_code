# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import numpy as np
from astropy.io import ascii, fits
from astropy import units as u
from astropy.table import Table


def radio_qualityfilter(data, factor=5):
    data = data[data['f'] > 0]
    data = data[data['df'] > 0]
    data = data[data['f']/data['df'] > factor]
    return data


# ## Loading radio data

# ### 3C 273

filename = "./data/3c273/radio/J1229+0203.csv"
rlc = ascii.read(filename, format='csv', names=['tc', 'f', 'df'])
rlc.meta = {
                'source' : '3C 273',
                'title' : 'OVRO (15$\,$GHz)',
                'f_units' : 'Jy',
                'freq' : np.array([1.5E10, 1.5E10]) * u.Hz,
                'file' : filename,
                'timerange' : [np.min(rlc['tc']), np.max(rlc['tc'])]
            }
rlc['tc'].unit = u.day
rlc['f'].unit = u.Jy
rlc['df'].unit = u.Jy
radio_3c273_orig = radio_qualityfilter(rlc)

# ### Mrk 421

filename = "./data/mrk421/radio/J1104+3812.csv"
rlc = ascii.read(filename, format='csv', names=['tc', 'f', 'df'])
rlc.meta = {
                'source' : 'Mrk 421',
                'title' : 'OVRO (15$\,$GHz)',
                'f_units' : 'Jy',
                'freq' : np.array([1.5E10, 1.5E10]) * u.Hz,
                'file' : filename,
                'timerange' : [np.min(rlc['tc']), np.max(rlc['tc'])]
            }
rlc['tc'].unit = u.day
rlc['f'].unit = u.Jy
rlc['df'].unit = u.Jy
radio_mrk421_orig = radio_qualityfilter(rlc)

# ### Mrk 501

filename = "./data/mrk501/radio/J1653+3945.csv"
rlc = ascii.read(filename, format='csv', names=['tc', 'f', 'df'])
rlc.meta = {
                'source' : 'Mrk 501',
                'title' : 'OVRO (15$\,$GHz)',
                'f_units' : 'Jy',
                'freq' : np.array([1.5E10, 1.5E10]) * u.Hz,
                'file' : filename,
                'timerange' : [np.min(rlc['tc']), np.max(rlc['tc'])]
            }
rlc['tc'].unit = u.day
rlc['f'].unit = u.Jy
rlc['df'].unit = u.Jy
radio_mrk501_orig = radio_qualityfilter(rlc)

# ## Loading GeV data

# ### 3C 273

# +
#filename = './data/3c273/fermi/private_analysis/3day/gal-exgal-fixed-and-sources-fixed.fits'
#csv_filename = './data/3c273/fermi/3c273_3day_gal-exgal-fixed-and-sources-fixed.csv'
#data = fits.open(filename)
#m = np.logical_and(np.logical_or(data[1].data['tmin_mjd'] > 0, data[1].data['tmax_mjd'] > 0), ~np.isnan(data[1].data['flux100']))
#data_fermi = {'tc' : (data[1].data['tmax_mjd'][m] + data[1].data['tmin_mjd'][m])/2, 'f' : data[1].data['flux100'][m], 'df' : data[1].data['flux100_err'][m], 'TS' : data[1].data['ts'][m], 'UL95' : data[1].data['flux100_ul95'][m]}
#Table(data_fermi).write(csv_filename, format='csv', overwrite=True)

#filename = './data/3c273/fermi/private_analysis/7day/gal-exgal-fixed-and-sources-fixed.fits'
#csv_filename = './data/3c273/fermi/3c273_7day_gal-exgal-fixed-and-sources-fixed.csv'
#data = fits.open(filename)
#m = np.logical_and(np.logical_or(data[1].data['tmin_mjd'] > 0, data[1].data['tmax_mjd'] > 0), ~np.isnan(data[1].data['flux100']))
#data_fermi = {'tc' : (data[1].data['tmax_mjd'][m] + data[1].data['tmin_mjd'][m])/2, 'f' : data[1].data['flux100'][m], 'df' : data[1].data['flux100_err'][m], 'TS' : data[1].data['ts'][m], 'UL95' : data[1].data['flux100_ul95'][m]}
#Table(data_fermi).write(csv_filename, format='csv', overwrite=True)

filename = "./data/3c273/fermi/3c273_3days.csv"
glc = ascii.read(filename, format='csv', names=['tc', 'f', 'df', 'TS', 'UL95'])
glc.meta = {
                'source' : '3C 273',
                'title' : 'Fermi (0.1-300$\,$GeV)',
                'f_units' : 'ph/cm$^2$/s',
                'freq' : np.array([100*1000, 300*1.0E6])*2.418E17 * u.Hz,
                'binsize' : 3 * u.day,
                'file' : filename,
                'timerange' : [np.min(glc['tc']), np.max(glc['tc'])]
            }
glc['tc'].unit = u.day
glc['f'].unit = u.ph / u.cm ** 2 / u.s
glc['df'].unit = u.ph / u.cm ** 2 / u.s
glc['UL95'].unit = u.ph / u.cm ** 2 / u.s
gev_3c273 = glc
# -

# ### Mrk 421

# +
#import pandas as pd
#data = pd.read_csv("../data/mrk421/fermi/private_analysis/1day/1_energy_bin_(100MeV-300GeV)/data.dat", delim_whitespace=True, usecols=[0, 1, 2, 5, 6], names=["tc", "f", "df", "TS", "UL95"], skiprows=1)
#Table(data.to_numpy()).write("../data/mrk421/fermi/mrk421_1day.csv", format='csv', names=['tc', 'f', 'df', 'TS', 'UL95'])

filename = "./data/mrk421/fermi/mrk421_1day.csv"
glc = ascii.read(filename, format='csv', names=['tc', 'f', 'df', 'TS', 'UL95'])
glc.meta = {
                'source' : 'Mrk 421',
                'title' : 'Fermi (0.1-300$\,$GeV)',
                'f_units' : 'ph/cm$^2$/s',
                'freq' : np.array([100*1000, 300*1.0E6])*2.418E17 * u.Hz,
                'binsize' : 1 * u.day,
                'file' : filename,
                'timerange' : [np.min(glc['tc']), np.max(glc['tc'])]
            }
glc['tc'].unit = u.day
glc['f'].unit = u.ph / u.cm ** 2 / u.s
glc['df'].unit = u.ph / u.cm ** 2 / u.s
glc['UL95'].unit = u.ph / u.cm ** 2 / u.s
gev_mrk421 = glc
# -

# ### Mrk 501

# +
#data = fits.open("../data/mrk501/fermi/private_analysis/7day_NEW_binned/gal-exgal-fixed-and-sources-fixed.fits")
#m = np.logical_and(np.logical_or(data[1].data['tmin_mjd'] > 0, data[1].data['tmax_mjd'] > 0), ~np.isnan(data[1].data['flux100']))
#data_fermi = {'tc': (data[1].data['tmin_mjd'][m] + data[1].data['tmax_mjd'][m])/2, 'f' : data[1].data['flux100'][m], 'df' : data[1].data['flux100_err'][m], 'TS' : data[1].data['ts'][m], 'UL95' : data[1].data['flux100_ul95'][m]}
#Table(data_fermi).write("../data/mrk501/fermi/mrk501_7days.csv", format='csv', overwrite=True)

#data = fits.open("./data/mrk501/fermi/private_analysis/7day_NEW_binned/gal-exgal-fixed-and-sources-fixed.fits")
#m = np.logical_and(np.logical_or(data[1].data['tmin_mjd'] > 0, data[1].data['tmax_mjd'] > 0), ~np.isnan(data[1].data['flux1000']))
#data_fermi = {'tc': (data[1].data['tmin_mjd'][m] + data[1].data['tmax_mjd'][m])/2, 'f' : data[1].data['flux1000'][m], 'df' : data[1].data['flux1000_err'][m], 'TS' : data[1].data['ts'][m], 'UL95' : data[1].data['flux100_ul95'][m]}
#Table(data_fermi).write("./data/mrk501/fermi/mrk501_7days.csv", format='csv', overwrite=True)

#data = fits.open("./data/fermi/private_analysis/7day_NEW_binned/gal-exgal-fixed-and-sources-fixed.fits")
#data_fermi = {
#                'ts': data[1].data['tmin_mjd'], 
#                'te' : data[1].data['tmax_mjd'], 
#                'f' : data[1].data['flux1000'], 
#                'df' : data[1].data['flux1000_err'], 
#                'TS' : data[1].data['ts'], 
#                'UL95' : data[1].data['flux1000_ul95'],               
#                'N': data[1].data['src_norm'], 
#                'dN': data[1].data['src_norm_err'], 
#                'alpha': data[1].data['src_alpha'], 
#                'dalpha': data[1].data['src_alpha_err'], 
#                'beta': data[1].data['src_beta'], 
#                'dbeta': data[1].data['src_beta_err'],
#                'Eb': data[1].data['src_Eb'], 
#                'dEb': data[1].data['src_Eb_err']
#             }
#create_and_save_data('fermi7d1g', data_fermi, freqHz=np.array([1000*1000, 300*1.0E6])*2.418E17, title="Fermi (1-300$\,$GeV) 7d", units="ph/cm$^2$/s")




filename = "./data/mrk501/fermi/mrk501_7days.csv"
glc = ascii.read(filename, format='csv', names=['tc', 'f', 'df', 'TS', 'UL95'])
glc.meta = {
                'source' : 'Mrk 501',
                'title' : 'Fermi (0.1-300$\,$GeV)',
                'f_units' : 'ph/cm$^2$/s',
                'freq' : np.array([1000*1000, 300*1.0E6])*2.418E17 * u.Hz,
                'binsize' : 7 * u.day,
                'file' : filename,
                'timerange' : [np.min(glc['tc']), np.max(glc['tc'])]
            }
glc['tc'].unit = u.day
glc['f'].unit = u.ph / u.cm ** 2 / u.s
glc['df'].unit = u.ph / u.cm ** 2 / u.s
glc['UL95'].unit = u.ph / u.cm ** 2 / u.s
gev_mrk501 = glc


# -
# ## Additional routines

def rebin_lc(ds, like_ds, bin_size):
    data = ds[['tc','f', 'df']]
    roi = [np.min(like_ds['tc']), np.max(like_ds['tc'])]
    step = bin_size/2
    t_f = np.arange(roi[0], roi[1], bin_size)
    
    f_t = []
    f_f = []
    f_df = []
    for t in t_f:
        d = np.logical_and(data['tc'] >= t - step, data['tc'] < t + step)
        if len(data['tc'][d]) > 0:
            _m = np.mean(data['f'][d])
            _dm = 0
            _l = len(data['tc'][d])
            if _l == 1:
                _dm = data['df'][d][0]
            else:
                _dm = np.sqrt(np.sum(data['df'][d]**2))/_l
            if (_dm) > 0:
                f_t.append(t)
                f_f.append(_m)
                f_df.append(_dm)
            else:
                print("Skipping MJD %s point due to zero uncertainty provided" % t)
    f_t = np.array(f_t)
    f_f = np.array(f_f)
    f_df = np.array(f_df)

    _t = Table({'tc' : f_t, 'f' : f_f, 'df' : f_df})
    _t['tc'].unit = ds['tc'].unit
    _t['f'].unit = ds['f'].unit
    _t['df'].unit = ds['df'].unit
    _t.meta = ds.meta    
    return _t


def coincidence_checker(data1, data2, bin_size=30):
    step = bin_size / 2
    mask1 = []
    mask2 = []
    offsets1 = []
    offsets2 = []

    for _t in data1['tc']:
        _m = np.abs(data2['tc'] - _t) < 1   
        if np.count_nonzero(_m) == 0:
            mask1.append(False)
        else:
            offsets1.append(np.max(data2['tc'][_m] - _t))
            if np.count_nonzero(_m) > 1:
                print("There might be wrong binning, at least two points of data #2 found in one bin of data #1")
                mask1.append(False)
            else:
                mask1.append(True)
            
    for _t in data2['tc']:
        _m = np.abs(data1['tc'] - _t) < 1  
        if np.count_nonzero(_m) == 0:
            mask2.append(False)
        else:
            offsets2.append(np.max(data1['tc'][_m] - _t))
            if np.count_nonzero(_m) > 1:
                print("There might be wrong binning, at least two points of data #1 found in one bin of data #2")
                mask2.append(False)
            else:
                mask2.append(True)
    if __name__ == '__main__':
        print("%s: data #1 and #2 offset: %f±%f" % (data1.meta['source'], np.mean(offsets1), np.std(offsets1)))
    return np.array(mask1), np.array(mask2)


# ## Rebinning radio LCs according to GeV binning

# +
radio_3c273 = rebin_lc(radio_3c273_orig, gev_3c273, bin_size=3)
radio_mrk421 = rebin_lc(radio_mrk421_orig, gev_mrk421, bin_size=1)
radio_mrk501 = rebin_lc(radio_mrk501_orig, gev_mrk501, bin_size=7)

coincidence_mask_3c273 = coincidence_checker(radio_3c273, gev_3c273, bin_size=3)
coincidence_mask_mrk421 = coincidence_checker(radio_mrk421, gev_mrk421, bin_size=1)
coincidence_mask_mrk501 = coincidence_checker(radio_mrk501, gev_mrk501, bin_size=7)
# -

# ## LCs preview


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def simple_plot(data1, data2, data3):
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15,12))
        ax1.set_title(data1.meta['source'])
        ax1.errorbar(data1['tc'], data1['f'], yerr=data1['df'], linestyle='none', ecolor='grey', marker='.', color='black')
        ax2.errorbar(data2['tc'], data2['f'], yerr=data2['df'], linestyle='none', ecolor='grey', marker='.', color='black')
        ax3.errorbar(data3['tc'], data3['f'], yerr=data3['df'], linestyle='none', ecolor='grey', marker='.', color='black')
        
        ax1.set_ylabel(f"{data1.meta['title']}\n$f$ [{data1.meta['f_units']}]")
        ax2.set_ylabel(f"{data2.meta['title']}\n$f$ [{data2.meta['f_units']}]")
        ax3.set_ylabel(f"{data3.meta['title']}\n$f$ [{data3.meta['f_units']}]")
        
        ax3.set_xlabel('$t$ [MJD]')

    simple_plot(gev_3c273, radio_3c273, radio_3c273_orig)    
    simple_plot(gev_mrk421, radio_mrk421, radio_mrk421_orig)
    simple_plot(gev_mrk501, radio_mrk501, radio_mrk501_orig)














