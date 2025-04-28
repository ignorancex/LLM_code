# Copyright (c) 2024, Markus Gaug
# All rights reserved.
# 
# This source code is licensed under the GNU General Public License v3.0 found in the
# LICENSE file in the root directory of this source tree. 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.optimize import minimize

from setup_helper import SetUp
from sun_helper import AltAzSun, plot_anual_sol
from filter_helper import *
from plot_helper import * 
from likelihood_helper import * 
from extremes_helper import *
from coverage_helper import *
from naoi_helper import *
from not_helper import *
from precipitation_helper import *
from wind_helper import *
from diurnal_helper import *

resultdir = 'Results'

h5file_long  = 'WS2003-24_long.h5'
# from: https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv
naoi_file = 'norm.daily.nao.cdas.z500.19500101_current.csv'
not_file = 'NOT_2003_2023.h5'

is_naoi = False
is_NOT  = False

# Names of variables provided by WS
name_temperature = 'temperature'
name_humidity    = 'humidity'
name_pressure    = 'pressure'
name_ws_current  = 'windSpeedCurrent'
name_ws_gust     = 'windGust'
name_ws_average  = 'windSpeedAverage'
name_wdir_current= 'windDirection'
name_wdir_average= 'windDirectionAverage'    


coverage_cut = 50.
coverage_cut_for_daily_samples = 10
coverage_cut_for_monthly_samples = 80
day_coverage_for_samples = 85  # in percent
day_coverage_for_diurnal = 95  # in percent

#wind_binning = np.concatenate((np.arange(0.,104.,2.)-1.,105.2 + 4*(np.arange(0,10)-0.5) + 0.3*(np.arange(0,10)-0.5)**2),axis=None)
wind_binning = np.concatenate((np.arange(0.,104.,4.)-0.5,105.2 + 4*(np.arange(0,10)-0.5) + 0.3*(np.arange(0,10)-0.5)**2),axis=None)

WS_start      = '2003-01-01 00:00:01'
WS_relocation = '2004-03-01 00:00:01'
new_WS        = '2007-03-20 00:00:01'
new_model     = '2017-04-10 00:00:01'
old_model     = '2023-01-16 00:00:01'
NOT_end_of_5min = '2007-02-02 12:00:00'
NOT_end_of_data = '2019-12-31 23:59:59'

mpl.rcParams['agg.path.chunksize'] = 10000 
pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 1500)
pd.set_option('max_seq_items',500)

# ML stuff
name_mu              = ['A', 'B', 'C', 'phi_mu',          'sigma0' ]
name_mu_nob          = ['A', 'C',      'phi_mu',          'sigma0' ]
name_mu_dphim        = ['A', 'B', 'C', 'phi_mu', 'dphim', 'sigma0' ]
name_mu_dCm          = ['A', 'B', 'C', 'phi_mu', 'dCm',   'sigma0' ]
name_mu_offset       = ['A', 'B', 'C', 'phi_mu',          'sigma0', 'offset' ]
name_mu2             = ['A', 'B', 'C', 'phi_mu', 'D', 'phi_mu2','sigma0' ]
name_mu2_sig2        = ['A', 'B', 'C', 'phi_mu', 'D', 'phi_mu2','sigma0', 'E', 'phi_sig' ]    
name_mu2_offset      = ['A', 'B', 'C', 'phi_mu', 'D', 'phi_mu2','sigma0' ,                'offset' ]
name_mu2_sig2_offset = ['A', 'B', 'C', 'phi_mu', 'D', 'phi_mu2','sigma0', 'E', 'phi_sig', 'offset' ]    
bounds_mu              = ((0, None), (None, None), (0, None), (0, None), (0.1,None))
bounds_mu_nob          = ((0, None),               (0, None), (0, None), (0.1,None))
bounds_mu_dphim        = ((0, None), (None, None), (0, None), (0, None), (None,None),             (0.1,None))
bounds_mu_dCm          = ((0, None), (None, None), (0, None), (0, None), (None,None),             (0.1,None))
bounds_mu_offset       = ((0, None), (None, None), (0, None), (0, None), (0.1,None), (None, None))
bounds_mu2             = ((0, None), (None, None), (None, None), (0, None), (0, None), (0, None), (0.1,None))
bounds_mu2_sig2        = ((0, None), (None, None), (None, None), (0, None), (0, None), (0, None), (0.1,None), (-10.,None), (0, 12.))
bounds_mu2_offset      = ((0, None), (None, None), (None, None), (0, None), (0, None), (0, None), (0.1,None),                        (None,None))
bounds_mu2_sig2_offset = ((0, None), (None, None), (None, None), (0, None), (0, None), (0, None), (0.1,None), (-10.,None), (0, 12.), (None,None))

method='L-BFGS-B'
tol=1e-9

hum_dev_start = 80
def hum_dev(x,*p):
    return -p[0]*(x-hum_dev_start)-p[1]*(x-hum_dev_start)**2
#return p[0]-p[1]*(x-hum_dev_start)-p[2]*(x-hum_dev_start)**2

def wet_bulb_temperature(T,RH):
    return T*np.arctan(0.151977*np.power(RH + 8.313659,0.5)) + np.arctan(T + RH) - np.arctan(RH - 1.676331) + 0.00391838*np.power(RH,1.5)*np.arctan(0.023101 * RH) - 4.686035

def plot_downtime() -> None:

    dfff = dff[(dff['temperature_reliable']==True) & (dff['humidity_reliable']==True) & (dff['wind_reliable']==True)]

    dfff, coverage = apply_coverage(dfff,debug=False)

    cuts_dict = { name_humidity : [ '<', 90, '%' ],
                  name_ws_average : [ '<', 36, 'km/h' ] }
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    print ('entering in downtime calculation', flush=True)
    plot_mensual_downtime(dfff,cuts_dict, coverage_cut=80)
    plt.savefig('{:s}/Downtime_mensual{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')
    plt.show()

    cuts_dict = { name_humidity : [ '<', 90, '%' ],
                  name_ws_average : [ '<', 36, 'km/h' ] }
    #'windGust' : [ '<', 40, 'km/h'] }

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_downtime_vs_windgust(dfff,cuts_dict,36.,70.,coverage_cut=80,windgust_waitingtime=10, color=cmap(0))
    plot_downtime_vs_windgust(dfff,cuts_dict,36.,70.,coverage_cut=80,windgust_waitingtime=20, color=cmap(1))
    plot_downtime_vs_windgust(dfff,cuts_dict,36.,70.,coverage_cut=80,windgust_waitingtime=30, color=cmap(2))
    plt.xlabel('Maximum wind gust (km/h)', fontsize=26)
    plt.ylabel('Mean downtime (%)',fontsize=26)
    plt.legend(loc='best')
    plt.savefig('{:s}/Downtime_windgust{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')
    plt.show()
    
    
def plot_temperature_gradient() -> None:

    dfff = dff[(dff['temperature_reliable']==True)]

    dfff, coverage = apply_coverage(dfff,debug=False)

    mask = (dfff['Tgradient10R']>0.32)
    print ('Strong Tgradients: ', dfff.loc[mask,name_temperature])
    mask = (dfff['Tgradient10R']<-0.4)
    print ('Negative Tgradients: ', dfff.loc[mask,name_temperature])

    dfff_p = dff[dff['pressure_reliable']==True]
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    h,p = plot_profile(dfff['pressure'],dfff['Tgradient5R'],nbins=50)
    plt.xlabel('Pressure (mbar)')
    plt.ylabel('Temperature change rate (ºC/min)',fontsize=22)
    plt.savefig('{:s}/Tgradient5R_corr_pressure{:s}.pdf'.format(resultdir,tits))

    num = 20
    grad_thr = 0.5

    print ('TGRADIENT1 TE: ',dfff.loc[ dfff['Tgradient5R'] > grad_thr, name_temperature].head(n=num))
 
    grad_thr = -0.5

    print ('TGRADIENT5 TE: ',dfff.loc[ dfff['Tgradient1R'] < grad_thr, name_temperature].head(n=num))
    print ('TGRADIENT5 PR: ',dfff.loc[ dfff['Tgradient1R'] < grad_thr, name_pressure].head(n=num))
    
    mask_r = (dfff['humidity_reliable']==True) # & (Filtre_Humidity(dfff,False)))
    dfff_r = dfff[mask_r]
    mask = (dfff_r['Rgradient10R']<-4.0)
    print ('Negative Rgradients: ', dfff_r.loc[mask,name_humidity])
    mask = (dfff_r['Tgradient10R']<-0.4)    
    print ('Negative Tgradients: ', dfff_r.loc[mask,name_temperature])    
    mask = (dfff_r['Tgradient10R']>0.33)    
    print ('Positive Tgradients: ', dfff_r.loc[mask,name_temperature])    
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    #print ('TEST SHIFT ', s,': ', dfff_p['Rgradient5R'].shift(s,freq='T').head(n=20))
    h,p = plot_profile(dfff_r['Rgradient10R'],dfff_r['Tgradient10R'],nbins=30, is_shifted=False)
    plt.xlabel('RH change rate (%/min)')
    plt.ylabel(r'Temperature change rate (ºC/min)',fontsize=22)
    plt.savefig('{:s}/Tgradient10R_corr_Rgradient10R_shift0T{:s}.pdf'.format(resultdir,tits))
    
    for s in np.arange(2,15,1):
        
        plt.figure(figsize = (10,5), constrained_layout = True)
        #print ('TEST SHIFT ', s,': ', dfff_p['Rgradient5R'].shift(s,freq='T').head(n=20))
        h,p = plot_profile(dfff_r['Rgradient10R'].shift(s,freq='T'),dfff_r['Tgradient10R'],nbins=30, is_shifted=True)
        plt.xlabel('RH change rate (%/min)')
        plt.ylabel(r'Temperature change rate (ºC/min)',fontsize=22)
        plt.savefig('{:s}/Tgradient10R_corr_Rgradient10R_shift{:d}T{:s}.pdf'.format(resultdir,s,tits))
    
    for s in np.arange(2,120,10):
        plt.figure(figsize = (10,5), constrained_layout = True)
        #print ('TEST SHIFT ', s,': ', dfff_p['Rgradient5R'].shift(s,freq='T').head(n=20))
        h,p = plot_profile(dfff['pressure'].shift(s,freq='T'),dfff['Tgradient10R'],nbins=50, is_shifted=True)
        plt.xlabel('pressure (mbar)')
        plt.ylabel(r'Temperature change rate (ºC/min)',fontsize=22)
        plt.savefig('{:s}/Tgradient10R_corr_pressure_shift{:d}T{:s}.pdf'.format(resultdir,s,tits))

    for s in np.arange(2,120,10):
        plt.figure(figsize = (10,5), constrained_layout = True)
        #print ('TEST SHIFT ', s,': ', dfff_p['Rgradient5R'].shift(s,freq='T').head(n=20))
        h,p = plot_profile(dfff['Pgradient10R'].shift(s,freq='T'),dfff['Tgradient10R'],nbins=50, is_shifted=True)
        plt.xlabel('Pressure change rate (mbar/min)')
        plt.ylabel(r'Temperature change rate (ºC/min)',fontsize=23)
        plt.savefig('{:s}/Tgradient10R_corr_Pgradient5R_shift{:d}T{:s}.pdf'.format(resultdir,s,tits))

def plot_DTR() -> None:

    #plt.rcParams.update(params)

    is_offset = True
    is_averaging = True
    avg_n = 5
    tits = '_5avg_r'
    freq_n = '11min'
    offset_from_temperature = 0.051   # result of offset analysis from plot-temperature
    
    dfff = dff[(dff['temperature_reliable']==True)]

    dfff, coverage = apply_coverage(dfff,debug=False)

    dfn = dfff[dfff['coverage'] > coverage_cut_for_daily_samples]

    mask2 = ((dfn.index > new_model) & (dfn.index < old_model))
    dfn_offset = dfn
    dfn_offset.loc[mask2,name_temperature] = dfn_offset.loc[mask2,name_temperature].add(-1.*offset_from_temperature)

    mask = (dfn.index > WS_relocation)
    df_s = dfn[mask].shift(8,freq='H')  # calculate spread from 8:00 to 8:00
    df_s = df_s[df_s['coverage']>coverage_cut_for_daily_samples]
    mask_daily = (df_s['mjd'].resample('D').count().dropna() > day_coverage_for_diurnal/100*expected_data_per_day)

    mask_offset = (dfn_offset.index > WS_relocation)
    df_s_off = dfn_offset[mask_offset].shift(8,freq='H')  # calculate spread from 8:00 to 8:00
    df_s_off = df_s_off[df_s_off['coverage']>coverage_cut_for_daily_samples]
    mask_daily_off = (df_s_off['mjd'].resample('D').count().dropna() > day_coverage_for_diurnal/100*expected_data_per_day)

    mask_NOT = ((dfn.index > new_WS) & (dfn.index < NOT_end_of_data))
    df_s_NOT = dfn[mask_NOT].shift(0.5,freq='D')
    df_s_NOT = df_s_NOT[df_s_NOT['coverage']>coverage_cut_for_daily_samples]
    mask_daily_NOT = (df_s_NOT['mjd'].resample('D').count().dropna() > day_coverage_for_diurnal/100*expected_data_per_day)
    
    if is_averaging: 
        diu_s, mjd_s, hum_s = average_diurnal_df_rolling(df_s,mask_daily,name_temperature,name_humidity,freq=freq_n)
        diu_s_off, mjd_s_off, hum_s_off = average_diurnal_df_rolling(df_s_off,mask_daily_off,name_temperature,name_humidity,freq=freq_n)
        diu_s_NOT, mjd_s_NOT, hum_s_NOT = average_diurnal_df_rolling(df_s_NOT,mask_daily_NOT,name_temperature,name_humidity,freq=freq_n)                
        #diu_s, mjd_s, hum_s = average_diurnal_df(df_s,mask_daily,name_temperature,name_humidity,n=avg_n)
        #diu_s_off, mjd_s_off, hum_s_off = average_diurnal_df(df_s_off,mask_daily_off,name_temperature,name_humidity,n=avg_n)
        #diu_s_NOT, mjd_s_NOT, hum_s_NOT = average_diurnal_df(df_s_NOT,mask_daily_NOT,name_temperature,name_humidity,n=avg_n)                
    else:
        mjd_s = df_s['mjd'].resample('D').mean()
        hum_s = df_s[name_humidity].resample('D').mean()    
        diu_s = df_s[name_temperature].resample('D').max().dropna()-df_s[name_temperature].resample('D').min().dropna()    
        mjd_s = mjd_s[mask_daily]
        diu_s = diu_s[mask_daily]
        hum_s = hum_s[mask_daily]

        mjd_s_off = df_s_off['mjd'].resample('D').mean()
        hum_s_off = df_s_off[name_humidity].resample('D').mean()    
        diu_s_off = df_s_off[name_temperature].resample('D').max().dropna()-df_s_off[name_temperature].resample('D').min().dropna()    
        mjd_s_off = mjd_s_off[mask_daily_off]
        diu_s_off = diu_s_off[mask_daily_off]
        hum_s_off = hum_s_off[mask_daily_off]

        mjd_s_NOT = df_s_NOT['mjd'].resample('D').mean()
        hum_s_NOT = df_s_NOT[name_humidity].resample('D').mean()            
        diu_s_NOT = df_s_NOT[name_temperature].resample('D').max().dropna()-df_s_NOT[name_temperature].resample('D').min().dropna()    

        mjd_s_NOT = mjd_s_NOT[mask_daily_NOT]
        hum_s_NOT = hum_s_NOT[mask_daily_NOT]        
        diu_s_NOT = diu_s_NOT[mask_daily_NOT]
        
    mjd_m = mjd_s.resample('M').mean().dropna()
    diu_m = diu_s.resample('M').mean().dropna()
    hum_m = hum_s.resample('M').mean().dropna()

    mjd_m_NOT = mjd_s_NOT.resample('M').mean().dropna()
    diu_m_NOT = diu_s_NOT.resample('M').mean().dropna()
    hum_m_NOT = hum_s_NOT.resample('M').mean().dropna()

    init_diu = np.array([8.2, 0., 0.9, 5.9, 0.7])
    init_diu_nob = np.array([8.2, 0.9, 5.9, 0.7])
    init_diu_dphim = np.array([8.2, 0., 0.9, 5.9, 0., 0.7])
    init_diu_dCm  = np.array([8.2, 0., 0.9, 5.9, 0., 0.7])
    init_diu_offset = np.array([8.2, 0., 0.9, 5.9, 0.7, 0.])    

    diu_lik_daily = Likelihood_Wrapper(loglike,mu,
                                      name_mu,init_diu,bounds_mu,
                                      'Eq. (2)')
    diu_lik_daily.setargs_seq(mjd_s,diu_s)
    diu_lik_daily.like_minimize(method=method,tol=tol)
    diu_lik_daily.chi_square_ndf()
    residuals = diu_lik_daily.full_residuals()

    plt.figure(figsize = (10,5), constrained_layout = True)
    h,p = plot_profile(hum_s,residuals,nbins=50,maximum=100.5)
    popt, pcov = curve_fit(hum_dev, h[h>hum_dev_start-2],p[h>hum_dev_start-2], [0.05,0.001])
    plt.plot(h[h>hum_dev_start-2],hum_dev(h[h>hum_dev_start-2],*popt),'--',linewidth=4,color='r',label=r'fit: -%.3f$\cdot$(RH-80)-%0.4f$\cdot$(RH-80)$^{2}$' % tuple(popt))
    plt.xlim(0.,100.)
    plt.legend(loc='best')
    plt.xlabel('Relative humidity (%)')
    plt.ylabel(r'DTR fit residuals')
    plt.savefig('{:s}/DiurnalTemperature_corr_d_humidity{:s}.pdf'.format(resultdir,tits))

    print ('HUM FIT RES.:',popt)
    hum_p0 = popt[0]
    hum_p1 = popt[1]
    
    diu_lik_daily_hum = Likelihood_Wrapper(loglike_hum,mu_hum,
                                           name_mu,init_diu,bounds_mu,
                                           'Eq. (2), w/hum corr.')
    diu_lik_daily_hum.setargs_seq_hum(mjd_s,diu_s,hum_s,hum_p0,hum_p1)
    diu_lik_daily_hum.like_minimize(method=method,tol=tol)

    diu_lik_daily_hum_offset = Likelihood_Wrapper(loglike_hum,mu_hum,
                                                  name_mu,init_diu,bounds_mu,
                                                  'Eq. (2), w/hum corr. and temperature offset')
    diu_lik_daily_hum_offset.setargs_seq_hum(mjd_s_off,diu_s_off,hum_s_off,hum_p0,hum_p1) 
    diu_lik_daily_hum_offset.like_minimize(method=method,tol=tol)

    mask1_diu = ((mjd_s.index > new_WS) & ((mjd_s.index < new_model) | (mjd_s.index > old_model)))
    mask2_diu = ((mjd_s.index > new_model) & (mjd_s.index < old_model))    
    
    diu_lik_daily_hum_fulloffset = Likelihood_Wrapper(loglike_hum_2sets,mu_hum,
                                                      name_mu_offset,init_diu_offset,bounds_mu_offset,
                                                      'Eq. (2), w/hum corr. and DTR offset')
    diu_lik_daily_hum_fulloffset.setargs_seq_hum2(mjd_s,diu_s,hum_s,mask1_diu,mask2_diu,hum_p0,hum_p1)
    diu_lik_daily_hum_fulloffset.like_minimize(method=method,tol=tol)

    diu_lik_daily_nob = Likelihood_Wrapper(loglike_nob,mu_nob,
                                           name_mu_nob,init_diu_nob,bounds_mu_nob,
                                           'Eq. (2), b=0, all data')
    diu_lik_daily_nob.setargs_seq(mjd_s,diu_s)
    diu_lik_daily_nob.like_minimize(method=method,tol=tol)

    diu_lik_mean = Likelihood_Wrapper(loglike,mu,
                                      name_mu,init_diu,bounds_mu,
                                      'Eq. (2), monthly means')
    diu_lik_mean.setargs_seq(mjd_m,diu_m)
    diu_lik_mean.like_minimize(method=method,tol=tol)

    #diu_lik_dpm = Likelihood_Wrapper(loglike_dphim,mu_dphim,
    #                                 name_mu_dphim,init_diu_dphim,bounds_mu_dphim,
    #                                 'Eq. (2), monthly medians, w/ phase shift')    
    #diu_lik_dpm.setargs_seq(mjd_m,diu_m)
    #diu_lik_dpm.like_minimize(method=method,tol=tol)

    diu_lik_dCm = Likelihood_Wrapper(loglike_dCm,mu_dCm,
                                     name_mu_dCm,init_diu_dCm,bounds_mu_dCm,
                                     'Eq. (4), monthly means')    
    diu_lik_dCm.setargs_seq(mjd_m,diu_m)
    diu_lik_dCm.like_minimize(method=method,tol=tol)

    diu_lik_dCm_daily = Likelihood_Wrapper(loglike_dCm,mu_dCm,
                                           name_mu_dCm,init_diu_dCm,bounds_mu_dCm,
                                           'Eq. (4)')    
    diu_lik_dCm_daily.setargs_seq(mjd_s,diu_s)
    diu_lik_dCm_daily.like_minimize(method=method,tol=tol)
    diu_lik_dCm_daily.chi_square_ndf()

    residuals_dCm = diu_lik_dCm_daily.full_residuals()

    plt.figure(figsize = (10,5), constrained_layout = True)
    h,p = plot_profile(hum_s,residuals_dCm,nbins=50,maximum=100.5)
    popt, pcov = curve_fit(hum_dev, h[h>hum_dev_start-2],p[h>hum_dev_start-2], [0.05,0.001])
    plt.plot(h[h>hum_dev_start-2],hum_dev(h[h>hum_dev_start-2],*popt),'--',linewidth=4,color='r',label=r'fit: -%.3f$\cdot$(RH-80)-%0.4f$\cdot$(RH-80)$^{2}$' % tuple(popt))
    plt.xlim(0.,100.)
    plt.legend(loc='best')
    plt.xlabel('Relative humidity (%)')
    plt.ylabel(r'DTR fit residuals')
    plt.savefig('{:s}/DiurnalTemperature_corr_dCm_humidity{:s}.pdf'.format(resultdir,tits))

    print ('HUM FIT dCm RES.:',popt)
    hum_dCm_p0 = popt[0]
    hum_dCm_p1 = popt[1]
    
    diu_lik_dCm_daily_hum = Likelihood_Wrapper(loglike_dCm_hum,mu_dCm_hum,
                                           name_mu_dCm,init_diu_dCm,bounds_mu_dCm,
                                           'Eq. (4), w/hum corr.')    
    diu_lik_dCm_daily_hum.setargs_seq_hum(mjd_s,diu_s,hum_s,hum_dCm_p0,hum_dCm_p1)
    diu_lik_dCm_daily_hum.like_minimize(method=method,tol=tol)
    diu_lik_dCm_daily_hum.chi_square_ndf()    

    diu_lik_dCm_NOT_daily = Likelihood_Wrapper(loglike_dCm,mu_dCm,
                                               name_mu_dCm,init_diu_dCm,bounds_mu_dCm,
                                               'Eq. (4), only 2007-2019')    
    diu_lik_dCm_NOT_daily.setargs_seq(mjd_s_NOT,diu_s_NOT)
    diu_lik_dCm_NOT_daily.like_minimize(method=method,tol=tol)

    diu_lik_dCm_NOT = Likelihood_Wrapper(loglike_dCm,mu_dCm,
                                         name_mu_dCm,init_diu_dCm,bounds_mu_dCm,
                                         'Eq. (4), monthly means 2007-2019')    
    diu_lik_dCm_NOT.setargs_seq(mjd_m_NOT,diu_m_NOT)
    diu_lik_dCm_NOT.like_minimize(method=method,tol=tol)

    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_profile(hum_s,residuals_dCm,nbins=50,maximum=100.5)
    plt.xlim(0.,100.)    
    plt.xlabel('Relative humidity (%)')
    plt.ylabel(r'DTR fit residuals')
    plt.savefig('{:s}/DiurnalTemperature_corr_dCm_humidity_hum{:s}.pdf'.format(resultdir,tits))

    if is_naoi:
        plt.figure()
        naoi_correlate(df_naoi,diu_lik_daily.full_residuals(),color='r')
        naoi_profile(df_naoi,diu_lik_daily.full_residuals(),nbins=12)
        plt.savefig('{:s}/DiurnalTemperature_corr_d_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')
        
        plt.figure()
        naoi_correlate(df_naoi,diu_lik_mean.full_residuals(),color='r')
        naoi_profile(df_naoi,diu_lik_mean.full_residuals(),nbins=12)
        plt.savefig('{:s}/DiurnalTemperature_corr_m_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_diurnal_spread(dfff, name_temperature, coverage)
    #plot_diurnal_fit_results(mjd_s,diu_lik_dCm_daily,is_daily=True,color='r')
    #plot_diurnal_fit_results(mjd_s_NOT,diu_lik_dCm_NOT_daily,is_daily=True,color='b')
    #plot_diurnal_fit_results(mjd_m,diu_lik_dCm,is_daily=False,color='orange')
    #if is_offset:
    #    plot_diurnal_fit_results(mjd_s_off,diu_lik_daily_hum_offset,df_hum=hum_s_off,is_daily=True,color='green',p0=hum_p0,p1=hum_p1)
    #    tits = tits + '_offset'
    #else:
    plot_diurnal_fit_results(mjd_s,diu_lik_dCm_daily_hum,df_hum=hum_s,is_daily=True,color='r',p0=hum_p0,p1=hum_p1)    
    #plot_diurnal_fit_results(mjd_m,diu_lik_dCm_hum,is_daily=False,color='violet')
    #plot_diurnal_fit_results(mjd_m_NOT,diu_lik_dCm_NOT,is_daily=False,color='violet')
    #plt.xlabel('Year')
    plt.ylabel('DTR (ºC)', fontsize=18)
    #plt.gca().yaxis.set_tick_params(labelsize=18)    
    plt.savefig('{:s}/Temperaturadiurnal_sencer{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    tits = tits.replace('_offset','')

    #2D profile of diurnal temperature increase and amplitude increase
    plt.figure(figsize = (8,7), constrained_layout = True)
    plt.xlabel(r'$b$ (DTR increase) (ºC/10y)', fontsize = 25)
    plt.ylabel(r'$\Delta C_{m}$ (DTR seasonal ampltiude increase) (ºC/10y)', fontsize = 22)
    diu_lik_dCm_daily.profile_likelihood_2d(1,4,chi2=14,NN=60,method=method,tol=1e-8,clabel=r'$D(b,\Delta C_{m})$')
    ax = plt.gca()
    ax.set_xlim(-0.05,0.3)
    ax.set_ylim(0,1.0)
    plt.savefig('{:s}/DiurnalTemperature_profiled_daily_contour{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    #2D profile of diurnal temperature increase and amplitude increase
    plt.figure(figsize = (8,7), constrained_layout = True)
    plt.xlabel(r'$b$ (DTR increase) (ºC/10y)', fontsize = 25)
    plt.ylabel(r'$\Delta C_{m}$ (DTR seasonal ampltiude increase) (ºC/10y)', fontsize = 22)
    diu_lik_dCm_daily_hum.profile_likelihood_2d(1,4,chi2=12,NN=60,method=method,tol=1e-8,clabel=r'$D(b,\Delta C_{m})$')    
    ax = plt.gca()
    ax.set_xlim(-0.05,0.3)
    ax.set_ylim(0,1.0)
    plt.savefig('{:s}/DiurnalTemperature_profiled_dailyhum_contour{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    # Profile the diurnal temperature increase parameter now
    NN   = 50
    chi2 = 10

    plt.clf()
    plt.xlabel(r'$\Delta C_{m}$ (seasonal oscillation ampltiude increase) (ºC/10y)', fontsize = 22)
    plt.ylabel(r'$D(b)$', fontsize = 25)
    diu_lik_dCm_daily.profile_likelihood(4,chi2=11,NN=NN,method=method,col='g')
    #diu_lik_dCm_NOT_daily.profile_likelihood(4,chi2=6,NN=NN,method=method,col='b')
    diu_lik_dCm.profile_likelihood(4,chi2=8,NN=NN,method=method,col='orange')
    diu_lik_dCm_daily_hum.profile_likelihood(4,chi2=8,NN=NN,method=method,col='deepskyblue')
    if (is_offset):
        diu_lik_daily_hum_offset.profile_likelihood(4,chi2=8,NN=NN,method=method,col='cyan')
        diu_lik_daily_hum_fulloffset.profile_likelihood(4,chi2=8,NN=NN,method=method,col='k')
        tits = tits + '_offset'                        
    #diu_lik_dCm_NOT.profile_likelihood(4,chi2=11,NN=NN,method=method,col='violet')
    plt.savefig('{:s}/DiurnalTemperatureSeasonal_b_profiled{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    tits = tits.replace('_offset','')
    
    # Profile the diurnal temperature increase parameter now
    plt.clf()
    plt.xlabel(r'$b$ (DTR increase) (ºC/10y)', fontsize = 24)    
    plt.ylabel(r'$D(b)$', fontsize = 25)
    diu_lik_daily.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g')
    diu_lik_dCm_daily.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='violet')
    diu_lik_dCm_daily_hum.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='deepskyblue')
    diu_lik_mean.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='orange')
    plt.plot([], [], ' ', label="Robustness tests:")        
    if (is_offset):
        diu_lik_daily_hum_offset.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='cyan',alpha=0.2)
        diu_lik_daily_hum_fulloffset.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='k',alpha=0.2)
        tits = tits + '_offset'                
    plt.savefig('{:s}/DiurnalTemperature_b_profiled{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    tits = tits.replace('_offset','')
    
    #2D profile of diurnal temperature increase and amplitude increase
    #plt.figure()
    #plt.xlabel(r'$b$ (diurnal temperature amplitude increase) (ºC/10y)', fontsize = 24)
    #plt.ylabel(r'$\Delta C_{m}$ (seasonal oscillation ampltiude increase) (ºC/10y)', fontsize = 22)
    #diu_lik_dCm.profile_likelihood_2d(1,4,chi2=20,NN=25,method=method,tol=1e-7,clabel=r'$D(b,\Delta C_{m})$')    
    #plt.savefig('{:s}/DiurnalTemperature_profiled_contour.pdf', bbox_inches='tight')

    #plt.figure(figsize = (10,5), constrained_layout = True)
    #plot_diurnal(dfff,name_temperature)
    #plt.xlabel('Solar Altitude')
    #plt.ylabel('Temperature (ºC)')
    #plt.savefig('Temperatura_diurnal.png', bbox_inches='tight')
    #plt.show()

def plot_datacount() -> None:

    dfff = dff[(dff['temperature_reliable']==True)]

    dfff, coverage = apply_coverage(dfff,debug=False)

    fig = plt.figure(figsize = (12,5), constrained_layout = True)
    plt.clf()
    #plt.figure(figsize = (10,5), constrained_layout = True)#    fig = plt.figure()    plt.clf()
    subfig = fig.subfigures(1, 2, wspace=0.02, width_ratios=[2.75,1.])
    ax = subfig[0].subplots(1, 1)
    ax.plot(coverage, color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k')
    #plot_coverage(coverage)
    #ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('Coverage (%)', fontsize=18)

    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(ynums) #np.arange(start, end, 365.))
    ax.set_xlim([ynums[0],ynums[-1]+365.])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("         %Y"))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.set_ylim([-1.,101.])    
    ymin, ymax = ax.get_ylim()
    ax.vlines(pd.to_datetime(WS_exchange_dates),ymin,ymax,colors='r',linestyles='dashed')

    ax = subfig[1].subplots(1, 1)
    ax.hist(coverage.array, bins=10, color='steelblue')
    ax.set_xlabel('Coverage (%)', fontsize=18)
    ax.set_ylabel('Number of months', fontsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.set_xlim([0.,100.])
    plt.savefig('{:s}/Data_Count{:s}.pdf'.format(resultdir,tits))
    plt.show()
    
    
    
def plot_temperature() -> None:
    
    is_offset = False
    is_coauthor_tests = True
    tits = ''
    
    dfff = dff[(dff['temperature_reliable']==True)]

    dfff, coverage = apply_coverage(dfff,debug=False)

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual(dfff, name_temperature, 'Temperature (ºC)')
    plt.savefig('{:s}/Temperatura_mensual.pdf')
    plt.show()
    
    plt.clf()
    plot_mensual_distributions(dfff, name_temperature, 'Temperature - monthly mean (ºC)', 0.1)
    plt.savefig('Temperatura_distributions.png', bbox_inches='tight')
    plt.show()
    
    dfn = dfff[dfff['coverage'] > coverage_cut_for_daily_samples]
    dfn_H = dfff[dfff['coverage'] > coverage_cut_for_monthly_samples]
    mask = (dfn.index > WS_relocation)
    mask_NOT = ((dfn.index > new_WS) & (dfn.index < NOT_end_of_data))
    mask_H = (dfn_H.index > WS_relocation)
    mask1 = ((dfn.index > new_WS) & ((dfn.index < new_model) | (dfn.index > old_model))  & (dfn[name_temperature]>-11))
    mask2 = ((dfn.index > new_model) & (dfn.index < old_model) & (dfn[name_temperature]>-11))    

    mask_hahn = (mask & ((dfn.index < '2017-04-10 00:00:01') | (dfn.index > '2019-06-30 23:59:59'))) # email from 07/02/2024
    mask_longo = (mask & ((dfn.index < '2013-03-01 00:00:01') | (dfn.index > '2019-02-01 23:59:59'))) # email from 17/04/2024
    mask_dorner = (mask & (dfn.index > new_WS) &
                   ((dfn.index < '2013-01-01 00:00:01') | (dfn.index > '2014-12-31 23:59:59')) &
                   (dfn.index < '2019-06-01 00:00:01') ) # email from 05/04/2024  
    mask_schmuck = (mask & 
                    ((dfn.index < '2007-01-01 00:00:01') | (dfn.index > '2008-12-31 23:59:59')) &
                    ((dfn.index < '2017-01-01 00:00:01') | (dfn.index > '2020-12-31 23:59:59'))  ) # email from 05/04/2024  
    
    # Take a guess at initial βs
    init_temp = np.array([10.6, 0., 6.5, 6.9, 3.0])
    init_temp_dphim = np.array([10.6, 0., 6.5, 6.9, 0., 3.0])
    init_temp_dCm  = np.array([10.6, 0., 6.5, 6.9, 0., 3.0])    
    init_temp_Haslebacher = np.array([10.6, 0., 6.5, 4.2, 1.9])
    init_temp_offset = np.array([6.3, 0., 6., 7., 3.2, 0.])
    
    temp_lik_mean   = Likelihood_Wrapper(loglike,mu,
                                         name_mu,init_temp,bounds_mu,
                                         'Eq. (2), daily means')
    temp_lik_mean.setargs_df(dfn,name_temperature,mask,
                             is_daily=True,is_median=False,day_coverage=day_coverage_for_samples)

    temp_lik_median = Likelihood_Wrapper(loglike,mu,
                                         name_mu,init_temp,bounds_mu,
                                         'Eq. (2), daily medians')
    temp_lik_median.setargs_df(dfn,name_temperature,mask,
                               is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)    

    # Tests suggested from different coauthors based on NOT-MAGIC correlation plots
    temp_lik_median_hahn = Likelihood_Wrapper(loglike,mu,
                                              name_mu,init_temp,bounds_mu,
                                              'Eq. (2), daily medians, excl. 01/2017-06/2019')
    temp_lik_median_hahn.setargs_df(dfn,name_temperature,mask_hahn,
                                    is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)    

    temp_lik_median_longo = Likelihood_Wrapper(loglike,mu,
                                              name_mu,init_temp,bounds_mu,
                                              'Eq. (2), daily medians, excl. 03/2013-02/2019')
    temp_lik_median_longo.setargs_df(dfn,name_temperature,mask_longo,
                                    is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)    

    temp_lik_median_dorner = Likelihood_Wrapper(loglike,mu,
                                              name_mu,init_temp,bounds_mu,
                                              'Eq. (2), daily medians, excl. <03/2007 & (2013-2014) & >06/2019')
    temp_lik_median_dorner.setargs_df(dfn,name_temperature,mask_dorner,
                                    is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)    

    temp_lik_median_schmuck = Likelihood_Wrapper(loglike,mu,
                                              name_mu,init_temp,bounds_mu,
                                              'Eq. (2), daily medians, excl. (2007-2008) & (2017-2020)')
    temp_lik_median_schmuck.setargs_df(dfn,name_temperature,mask_schmuck,
                                    is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)    

    temp_lik_median_NOT = Likelihood_Wrapper(loglike,mu,
                                             name_mu,init_temp,bounds_mu,
                                             'Eq. (2), daily medians, 2007-2019')
    temp_lik_median_NOT.setargs_df(dfn,name_temperature,mask_NOT,
                                   is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)        
    temp_lik_median_dphim = Likelihood_Wrapper(loglike_dphim,mu_dphim,
                                               name_mu_dphim,init_temp_dphim,bounds_mu_dphim,
                                               'Eq. (2), daily medians, w/ phase shift')    
    temp_lik_median_dphim.setargs_df(dfn,name_temperature,mask,
                                     is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)    
    temp_lik_median_dCm = Likelihood_Wrapper(loglike_dCm,mu_dCm,
                                               name_mu_dCm,init_temp_dCm,bounds_mu_dCm,
                                               'Eq. (2), daily medians, w/ amplitude increase')    
    temp_lik_median_dCm.setargs_df(dfn,name_temperature,mask,
                                   is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)    
    temp_lik_Haslebacher = Likelihood_Wrapper(loglike_Haslebacher,mu_Haslebacher,
                                              name_mu,init_temp_Haslebacher,bounds_mu,
                                              'Eq. (19H), monthly means')
    temp_lik_Haslebacher.setargs_df(dfn_H,name_temperature,mask_H,
                                    is_daily=False,is_median=False)

    temp_lik_median_offset = Likelihood_Wrapper(loglike_2sets,mu,
                                                name_mu_offset,init_temp_offset,bounds_mu_offset,
                                                'Eq. (2), daily medians w/ offset')
    temp_lik_median_offset.setargs_df2(dfn,name_temperature,mask1,mask2,
                                       is_daily=True,is_median=True,day_coverage=day_coverage_for_samples)



    
    temp_lik_mean.like_minimize(method=method,tol=tol)
    temp_lik_median.like_minimize(method=method,tol=tol)
    if is_coauthor_tests:
        temp_lik_median_hahn.like_minimize(method=method,tol=tol)
        temp_lik_median_longo.like_minimize(method=method,tol=tol)
        temp_lik_median_dorner.like_minimize(method=method,tol=tol)
        temp_lik_median_schmuck.like_minimize(method=method,tol=tol)
    
    temp_lik_median_NOT.like_minimize(method=method,tol=tol)    
    temp_lik_median_dphim.like_minimize(method=method,tol=tol)
    temp_lik_median_dCm.like_minimize(method=method,tol=tol)        
    temp_lik_Haslebacher.like_minimize(method=method,tol=tol)
    temp_lik_median_offset.like_minimize(method=method,tol=tol)

    H = temp_lik_mean.approx_hessian(eps=1e-5,only_diag=False)
    H_inv = np.linalg.inv(H)
    print ('Approximate inverse Hessian: ',H_inv)
    print ('Approximate parameter uncertainties:', np.sqrt(np.diag(H_inv)))
    
    H = temp_lik_median.approx_hessian(eps=1e-5,only_diag=False)
    H_inv = np.linalg.inv(H)
    print ('Approximate inverse Hessian (median): ',H_inv)
    print ('Approximate parameter uncertainties (median):', np.sqrt(np.diag(H_inv)))

    plt.figure()
    mask_sun_alt = ((dfn['sun_alt'] > 38) & (dfn['sun_alt'] < 44))
    mask_sun_az = ((dfn['sun_az'] > 190) & (dfn['sun_az'] < 220))
    mask_sun_azalt = mask_sun_alt & mask_sun_az & (dfn.index > '2014-02-04 00:00') & (dfn.index < '2014-02-05 00:00')
    #plot_profile(dfn.loc[mask_sun,'sun_az'],dfn.loc[mask_sun,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),nbins=100)
    plt.plot(dfn.loc[mask_sun_alt,'sun_az'],dfn.loc[mask_sun_alt,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun_alt,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),'.',color='r')
    #plt.plot(dfn.loc[mask_sun,'sun_az'],dfn.loc[mask_sun,name_temperature],'.',color='r')
    plt.savefig('SunAz_Temperature.pdf')
    
    plt.clf()
    #plot_profile(dfn.loc[mask_sun,'sun_az'],dfn.loc[mask_sun,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),nbins=100)
    plt.plot(dfn.loc[mask_sun_az,'sun_alt'],dfn.loc[mask_sun_az,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun_az,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),'.',color='r')
    #plt.plot(dfn.loc[mask_sun,'sun_az'],dfn.loc[mask_sun,name_temperature],'.',color='r')
    plt.savefig('SunAlt_Temperature.pdf')
    
    plt.clf()
    #plot_profile(dfn.loc[mask_sun,'sun_az'],dfn.loc[mask_sun,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),nbins=100)
    plt.plot(dfn.loc[mask_sun_azalt].index,dfn.loc[mask_sun_azalt,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun_azalt,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),'.',color='r')
    #plt.plot(dfn.loc[mask_sun,'sun_az'],dfn.loc[mask_sun,name_temperature],'.',color='r')
    plt.savefig('SunAltAz_Temperature.pdf')
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff, name_temperature, coverage)
    #plt.xlabel('Year')
    plt.ylabel('Temperature (ºC)', fontsize=18)
    if is_offset:
        plot_historic_fit_results(dfn,mask1,temp_lik_median_offset,is_daily=True, day_coverage=day_coverage_for_samples,
                                  color='red',is_sigma2=False,is_offset=True)
        plot_historic_fit_results(dfn,mask2,temp_lik_median_offset,is_daily=True, day_coverage=day_coverage_for_samples,
                                  color='orange',is_sigma2=False,is_offset=True,offset=temp_lik_median_offset.res.x[-1])
        tits = tits + '_offset'        
    else:
        plot_historic_fit_results(dfn,mask,temp_lik_median,is_daily=True, day_coverage=day_coverage_for_samples)
    #plot_historic_fit_results(dfn_H,mask_H,temp_lik_Haslebacher,is_daily=False,color='orange')    
    plt.savefig('Temperatura_sencer{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Temperature (ºC)')
    temp_lik_median.plot_residuals()
    temp_lik_Haslebacher.plot_residuals(color='orange')
    plt.savefig('{:s}/Temperatura_residual{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    residuals = temp_lik_median.red_residuals()
    chi2_measured = temp_lik_median.chi_square_ndf()
    residuals_mean = temp_lik_mean.red_residuals()
    chi2_measured_mean = temp_lik_mean.chi_square_ndf()
    residuals_H = temp_lik_Haslebacher.red_residuals()
    chi2_measured_H = temp_lik_Haslebacher.chi_square_ndf()
    residuals_NOT = temp_lik_median_NOT.red_residuals()
    chi2_measured_NOT = temp_lik_median_NOT.chi_square_ndf()

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals,bins=40,label='median')
    a[0].hist(residuals_H,bins=40,label='Haslebacher')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals**2,bins=40,label='median')
    a[1].hist(residuals_H**2,bins=40,label='Haslebacher')    
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured),fontsize=20)
    a[1].legend(loc='best')    
    
    plt.savefig('{:s}/Temperatura_residuals_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    chi2_limit_rob_temperature = 3
    mask_chi2 = np.where(residuals**2 < chi2_limit_rob_temperature)
    temp_lik_median_rob = Likelihood_Wrapper(loglike,mu,
                                             name_mu,init_temp,bounds_mu,
                                             r'Eq. (2), daily medians ($\chi^{2}<$'+'{:d})'.format(chi2_limit_rob_temperature))
    temp_lik_median_rob.setargs_df(dfn,name_temperature,mask,
                                   is_daily=True,is_median=True,
                                   day_coverage=day_coverage_for_samples,np_mask=mask_chi2)    
    temp_lik_median_rob.like_minimize(method=method,tol=1e-9)    

    residuals_rob = temp_lik_median_rob.red_residuals()
    chi2_measured_rob = temp_lik_median_rob.chi_square_ndf()

    if is_naoi:    
        plt.figure()
        naoi_correlate(df_naoi,temp_lik_mean.full_residuals(),color='r')
        naoi_profile(df_naoi,temp_lik_mean.full_residuals(),nbins=12)    
        plt.savefig('{:s}/Temperature_corr_d_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

        plt.clf()
        naoi_correlate(df_naoi,temp_lik_Haslebacher.full_residuals(),color='orange')
        naoi_profile(df_naoi,temp_lik_Haslebacher.full_residuals(),nbins=12)        
        plt.savefig('{:s}/Temperature_corr_m_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

        plt.clf()
        naoi_correlate(df_naoi,temp_lik_Haslebacher.Y.resample('Y').mean(),color='b')
        #naoi_profile(df_naoi,temp_lik_Haslebacher.Y.resample('Y').mean(),nbins=12)            
        plt.legend(loc='best')    
        plt.savefig('{:s}/Temperature_corr_y_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')


    # Profile the temperature increase parameter now
    plt.figure()
    plt.xlabel(r'$b$ (temperature increase) (ºC/10y)', fontsize = 25)
    plt.ylabel(r'$D(b)$', fontsize = 25)

    NN   = 20
    chi2 = 10
    temp_lik_mean.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='b')
    temp_lik_median.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g')
    temp_lik_Haslebacher.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='orange')
    # Create empty plot with blank marker containing the extra label
    plt.plot([], [], ' ', label="Robustness tests:")    
    temp_lik_median_rob.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='r',alpha=0.2)
    temp_lik_median_NOT.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='darkviolet',alpha=0.2)
    if is_offset:
        temp_lik_median_offset.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='cyan',alpha=0.2)
        tits = tits + '_offset'
    if is_coauthor_tests:
        temp_lik_median_hahn.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='paleturquoise')
        temp_lik_median_longo.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='mediumturquoise')
        temp_lik_median_dorner.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='lightcyan')
        temp_lik_median_schmuck.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='lightseagreen')
        tits = tits + '_coauthors'
    plt.savefig('{:s}/Temperature_b_profiled{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    tits = tits.replace('_offset','')
    tits = tits.replace('_coauthors','')    
    # Profile the temperature phase shift parameter now
    #plt.figure()
    #plt.xlabel(r'$\Delta\phi_{x}$ (phase increase) (days/10y)', fontsize = 25)
    #plt.ylabel(r'$D(\Delta\phi_{x})$', fontsize = 25)

    #chi2 = 6    
    #NN   = 25
    #temp_lik_median_dphim.profile_likelihood(4,chi2=chi2,NN=NN,method=method,col='r')

    #plt.savefig('Temperature_dphim_profiled.pdf', bbox_inches='tight')

    # 2D profile of temperature increase and phase shift
    #plt.figure()
    #plt.xlabel(r'$b$ (temperature increase) (ºC/10y)', fontsize = 25)
    #plt.ylabel(r'$\Delta\phi_{x}$ (phase increase) (days/10y)', fontsize = 25)
    #temp_lik_median_dphim.profile_likelihood_2d(1,4,chi2=chi2,NN=NN,method=method)    
    #plt.savefig('Temperature_profiled_contour.pdf', bbox_inches='tight')
    
    plt.hist(dfff[name_temperature], density = True, bins = 70, color = 'steelblue')
    plt.xlabel('Temperature (ºC)', fontsize = 30)
    plt.ylabel('Probability / ºC', fontsize = 30)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.savefig('{:s}/Histogram_temperature{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.hist(dfff[name_temperature], density = True, bins = 45, color = 'steelblue',log = True)
    plt.xlabel('Temperature (ºC)', fontsize = 30)
    plt.ylabel('Probability / ºC', fontsize = 30)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.savefig('{:s}/Histogram_logtemperature{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    #Distribucions anuals i mensuals:
    plt.clf()
    plot_hist(dfff, name_temperature, 1.,'Temperature (ºC)','Probability / ºC', 'ºC', xoff=0.08, coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_temperature{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plot_hist(dfff, name_temperature, 1.,'Temperature (ºC)','Probability / ºC', 'ºC', xoff=0.08, is_night=True, coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_temperature_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()


        
def plot_temperature_not() -> None:

    #plt.rcParams.update(params)

    df_not = df_not[df_not['Tgradient1']<3]
    print (df_not.head())

    pd.set_option('display.max_columns', 20)
    #print ('TGRADIENT1 TD: ',df_not[df_not['Tgradient1']>3].head(n=100))

    name_temperature_save = name_temperature
    name_humidity_save = name_humidity
    name_temperature = 'TempInAirDegC'
    name_humidity = 'Humidity'
    day_coverage_for_not = 80  # a bit lower because of early samples showing coverage of only 85%

    mask_5min = ((df_not.index < NOT_end_of_5min) & (df_not[name_temperature]>-11) & (df_not[name_temperature]<35))
    mask_1min = ((df_not.index > NOT_end_of_5min) & (df_not[name_temperature]>-11) & (df_not[name_temperature]<35))

    dfff_5min, coverage_5min = apply_coverage(df_not[mask_5min],data_spacing=5,debug=False)
    dfff_1min, coverage_1min = apply_coverage(df_not[mask_1min],data_spacing=1,debug=False)

    dfff = pd.concat([dfff_5min, dfff_1min])
    coverage = pd.concat([coverage_5min, coverage_1min])
    print ('AFTER all', dfff[dfff.index>'2003-09-03 00:00:00'])    
    
    plt.figure(figsize = (10,5), constrained_layout = True)#    fig = plt.figure()
    subfig = fig.subfigures(1, 2, wspace=0.02, width_ratios=[2.75,1.])
    ax = subfig[0].subplots(1, 1)
    ax.plot(coverage, color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k')
    ax.set_ylabel('Coverage (%)', fontsize=18)

    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(ynums) #np.arange(start, end, 365.))
    ax.set_xlim([ynums[0],ynums[-1]+365.])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("  %Y"))
    ax.xaxis.set_tick_params(labelsize=14)
    plt.xticks(ha='left')

    ax = subfig[1].subplots(1, 1)
    ax.hist(coverage.array, bins=10, color='steelblue')
    ax.set_xlabel('Coverage (%)', fontsize=18)
    ax.set_ylabel('Number of months', fontsize=18)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)    
    plt.savefig('{:s}/Data_Count_NOT{:s}.pdf'.format(resultdir,tits))
    plt.show()
        
    dfn   = dfff[dfff['coverage'] > coverage_cut_for_daily_samples]
    dfn_H = dfff[dfff['coverage'] > coverage_cut_for_monthly_samples]
    mask   = (dfn[name_temperature]>-11)
    mask_H = (dfn[name_temperature]>-11)

    # Take a guess at initial βs
    init_temp       = np.array([6.3, 0., 6., 7., 3.2])
    init_temp_dphim = np.array([6.3, 0., 6., 7., 0., 3.2])
    init_temp_dCm   = np.array([6.3, 0., 6., 7., 0., 3.2])    
    init_temp_Haslebacher = np.array([6.3, 0., 6., 4.2, 2.])
    
    temp_lik_mean   = Likelihood_Wrapper(loglike,mu,
                                         name_mu,init_temp,bounds_mu,
                                         'Eq. (2), daily means')
    temp_lik_mean.setargs_df(dfn,name_temperature,mask,
                             is_daily=True,is_median=False,day_coverage=day_coverage_for_not)
    temp_lik_median = Likelihood_Wrapper(loglike,mu,
                                         name_mu,init_temp,bounds_mu,
                                         'Eq. (2), daily medians')
    temp_lik_median.setargs_df(dfn,name_temperature,mask,
                               is_daily=True,is_median=True,day_coverage=day_coverage_for_not)

    
    temp_lik_median_dphim = Likelihood_Wrapper(loglike_dphim,mu_dphim,
                                               name_mu_dphim,init_temp_dphim,bounds_mu_dphim,
                                               'Eq. (2), daily medians, w/ phase shift')    
    temp_lik_median_dphim.setargs_df(dfn,name_temperature,mask,
                                     is_daily=True,is_median=True,day_coverage=day_coverage_for_not)    
    temp_lik_median_dCm = Likelihood_Wrapper(loglike_dCm,mu_dCm,
                                               name_mu_dCm,init_temp_dCm,bounds_mu_dCm,
                                               'Eq. (2), daily medians, w/ amplitude increase')    
    temp_lik_median_dCm.setargs_df(dfn,name_temperature,mask,
                                   is_daily=True,is_median=True,day_coverage=day_coverage_for_not)    
    temp_lik_Haslebacher = Likelihood_Wrapper(loglike_Haslebacher,mu_Haslebacher,
                                              name_mu,init_temp_Haslebacher,bounds_mu,
                                              'Haslebacher et al.\'s Eq. (19), monthly means')
    temp_lik_Haslebacher.setargs_df(dfn_H,name_temperature,mask_H,
                                    is_daily=False,is_median=False)
    
    temp_lik_mean.like_minimize(method=method,tol=tol)
    temp_lik_median.like_minimize(method=method,tol=tol)
    temp_lik_median_dphim.like_minimize(method=method,tol=tol)
    temp_lik_median_dCm.like_minimize(method=method,tol=tol)        
    temp_lik_Haslebacher.like_minimize(method=method,tol=tol)

    plt.figure()
    mask_sun_alt = ((dfn['sun_alt'] > 38) & (dfn['sun_alt'] < 44))
    mask_sun_az = ((dfn['sun_az'] > 190) & (dfn['sun_az'] < 220))
    mask_sun_azalt = mask_sun_alt & mask_sun_az 
    plt.plot(dfn.loc[mask_sun_alt,'sun_az'],dfn.loc[mask_sun_alt,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun_alt,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),'.',color='r')
    plt.savefig('{:s}/SunAz_Temperature_NOT{:s}.pdf'.format(resultdir,tits))
    
    plt.clf()
    plt.plot(dfn.loc[mask_sun_az,'sun_alt'],dfn.loc[mask_sun_az,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun_az,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),'.',color='r')
    plt.savefig('{:s}/SunAlt_Temperature_NOT{:s}.pdf'.format(resultdir,tits))
    
    plt.clf()
    plt.plot(dfn.loc[mask_sun_azalt].index,dfn.loc[mask_sun_azalt,name_temperature]-temp_lik_median.mu_func(temp_lik_median.res.x[0:-1],np.array(dfn.loc[mask_sun_azalt,'mjd'].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year).values.astype(float))),'.',color='r')
    plt.savefig('{:s}/SunAltAz_Temperature_NOT{:s}.pdf'.format(resultdir,tits))
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff[mask_1min], name_temperature, coverage_1min)
    plot_historic(dfff[mask_5min], name_temperature, coverage_5min)    
    plt.ylabel('Temperature (ºC)', fontsize=18)
    plot_historic_fit_results(dfn,(dfn.index < NOT_end_of_5min),temp_lik_median,is_daily=True, day_coverage=day_coverage_for_not * 2/5)
    plot_historic_fit_results(dfn,(dfn.index > NOT_end_of_5min),temp_lik_median,is_daily=True, day_coverage=day_coverage_for_not * 2/1)
    plt.savefig('{:s}/Temperatura_sencer_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Temperature (ºC)')
    temp_lik_median.plot_residuals()
    temp_lik_Haslebacher.plot_residuals(color='orange')
    plt.savefig('{:s}/Temperatura_residuals_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    residuals = temp_lik_median.red_residuals()
    chi2_measured = temp_lik_median.chi_square_ndf()
    residuals_mean = temp_lik_mean.red_residuals()
    chi2_measured_mean = temp_lik_mean.chi_square_ndf()
    residuals_H = temp_lik_Haslebacher.red_residuals()
    chi2_measured_H = temp_lik_Haslebacher.chi_square_ndf()

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals,bins=40,label='median')
    a[0].hist(residuals_H,bins=40,label='Haslebacher')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals**2,bins=40,label='median')
    a[1].hist(residuals_H**2,bins=40,label='Haslebacher')    
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured),fontsize=20)
    a[1].legend(loc='best')    
    
    plt.savefig('{:s}/Temperatura_residuals_hist_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    chi2_limit_rob_temperature = 3
    mask_chi2 = np.where(residuals**2 < chi2_limit_rob_temperature)
    temp_lik_median_rob = Likelihood_Wrapper(loglike,mu,
                                             name_mu,init_temp,bounds_mu,
                                             r'Eq. (2), daily medians ($\chi^{2}<$'+'{:d})'.format(chi2_limit_rob_temperature))
    temp_lik_median_rob.setargs_df(dfn,name_temperature,mask,
                                   is_daily=True,is_median=True,
                                   day_coverage=day_coverage_for_not,np_mask=mask_chi2)    
    temp_lik_median_rob.like_minimize(method=method,tol=1e-9)    

    residuals_rob = temp_lik_median_rob.red_residuals()
    chi2_measured_rob = temp_lik_median_rob.chi_square_ndf()

    if is_naoi:
        plt.figure()
        naoi_correlate(df_naoi,temp_lik_mean.full_residuals(),color='r')
        naoi_profile(df_naoi,temp_lik_mean.full_residuals(),nbins=12)    
        plt.savefig('Temperature_corr_d_NAOI_NOT.pdf',bbox_inches='tight')

        plt.clf()
        naoi_correlate(df_naoi,temp_lik_Haslebacher.full_residuals(),color='orange')
        naoi_profile(df_naoi,temp_lik_Haslebacher.full_residuals(),nbins=12)        
        plt.savefig('Temperature_corr_m_NAOI_NOT.pdf',bbox_inches='tight')

        plt.clf()
        naoi_correlate(df_naoi,temp_lik_Haslebacher.Y.resample('Y').mean(),color='b')
        plt.legend(loc='best')    
        plt.savefig('Temperature_corr_y_NAOI_NOT.pdf',bbox_inches='tight')


    # Profile the temperature increase parameter now
    plt.figure()
    plt.xlabel(r'$b$ (temperature increase) (ºC/10y)', fontsize = 25)
    plt.ylabel(r'$D(b)$', fontsize = 25)
    NN   = 20
    chi2 = 10
    temp_lik_mean.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='b')
    temp_lik_median.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g')
    temp_lik_median_rob.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='r')
    temp_lik_Haslebacher.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='orange')    

    plt.savefig('{:s}/Temperature_b_profiled_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    plt.hist(dfff[name_temperature], density = True, bins = 70, color = 'steelblue')
    plt.xlabel('Temperature (ºC)', fontsize = 30)
    plt.ylabel('Probability / ºC', fontsize = 30)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.savefig('{:s}/Histogram_temperature_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.hist(dfff[name_temperature], density = True, bins = 45, color = 'steelblue',log = True)
    plt.xlabel('Temperature (ºC)', fontsize = 30)
    plt.ylabel('Probability / ºC', fontsize = 30)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.savefig('{:s}/Histogram_logtemperature_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    #Distribucions anuals i mensuals:
    plt.clf()
    plot_hist(dfff, name_temperature, 1.,'Temperature (ºC)','Probability / ºC', 'ºC', xoff=0.08, coverage_cut=coverage_cut)
    plt.savefig('{:s}/Hist_temperature_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plot_hist(dfff, name_temperature, 1.,'Temperature (ºC)','Probability / ºC', 'ºC', xoff=0.08, is_night=True, coverage_cut=coverage_cut)
    plt.savefig('{:s}/Hist_temperature_night_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()


    plt.clf()
    plot_mensual(dfff, name_temperature, 'Temperature (ºC)')
    plt.savefig('{:s}/Temperatura_mensual_NOT{:s}.pdf'.format(resultdir,tits))
    plt.show()
    
    plt.clf()
    plot_mensual_distributions(dfff, name_temperature, 'Temperature - monthly mean (ºC)', 0.1)
    plt.savefig('{:s}/Temperatura_distributions_NOT{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    init_diu = np.array([8.2, 0., 0.9, 5.9, 0.7])
    init_diu_nob = np.array([8.2, 0.9, 5.9, 0.7])
    init_diu_dphim = np.array([5., 0., 0.9, 5., 0., 2.3])
    init_diu_dCm  = np.array([5., 0., 0.6, 4.1, 0., 2.3])    
    
    mask_5min = (dfn.index < NOT_end_of_5min)
    mask_1min = (dfn.index > NOT_end_of_5min)

    df_s1  = dfn[mask_5min].shift(8,freq='H')  # calculate spread from 8:00 to 8:00
    df_s1  = df_s1[df_s1['coverage']>coverage_cut_for_daily_samples]
    mjd_s1 = df_s1['mjd'].resample('D').mean()
    hum_s1 = df_s1[name_humidity].resample('D').mean()    
    diu_s1 = df_s1[name_temperature].resample('D').max().dropna()-df_s1[name_temperature].resample('D').min().dropna()    

    expected_data = 60*24/5
    mask_daily = (df_s1['mjd'].resample('D').count().dropna() > day_coverage_for_diurnal/100*expected_data)
    mjd_s1 = mjd_s1[mask_daily]
    diu_s1 = diu_s1[mask_daily]
    hum_s1 = hum_s1[mask_daily]

    df_s2 = dfn[mask_1min].shift(8,freq='H')  # calculate spread from 8:00 to 8:00
    #df_s = dfn[mask]
    df_s2  = df_s2[df_s2['coverage']>coverage_cut_for_daily_samples]
    mjd_s2 = df_s2['mjd'].resample('D').mean()
    hum_s2 = df_s2[name_humidity].resample('D').mean()    
    diu_s2 = df_s2[name_temperature].resample('D').max().dropna()-df_s2[name_temperature].resample('D').min().dropna()    

    expected_data = 60*24/1
    mask_daily = (df_s2['mjd'].resample('D').count().dropna() > day_coverage_for_diurnal/100*expected_data)
    mjd_s2 = mjd_s2[mask_daily]
    diu_s2 = diu_s2[mask_daily]
    hum_s2 = hum_s2[mask_daily]

    df_s  = pd.concat([df_s1,df_s2])
    mjd_s = pd.concat([mjd_s1,mjd_s2])
    diu_s = pd.concat([diu_s1,diu_s2])
    hum_s = pd.concat([hum_s1,hum_s2])
    
    mjd_m = mjd_s.resample('M').mean().dropna()
    diu_m = diu_s.resample('M').mean().dropna()
    hum_m = hum_s.resample('M').mean().dropna()
        
    diu_lik_daily = Likelihood_Wrapper(loglike,mu,
                                      name_mu,init_diu,bounds_mu,
                                      'Eq. (2), all data')
    diu_lik_daily.setargs_seq(mjd_s,diu_s)
    diu_lik_daily.like_minimize(method=method,tol=tol)
    diu_lik_daily.chi_square_ndf()

    diu_lik_daily_hum = Likelihood_Wrapper(loglike_hum,mu_hum,
                                           name_mu,init_diu,bounds_mu,
                                           'Eq. (2), all data, w/hum correction')
    diu_lik_daily_hum.setargs_seq_hum(mjd_s,diu_s,hum_s)
    diu_lik_daily_hum.like_minimize(method=method,tol=tol)

    diu_lik_daily_nob = Likelihood_Wrapper(loglike_nob,mu_nob,
                                           name_mu_nob,init_diu_nob,bounds_mu_nob,
                                           'Eq. (2), b=0, all data')
    diu_lik_daily_nob.setargs_seq(mjd_s,diu_s)
    diu_lik_daily_nob.like_minimize(method=method,tol=tol)

    diu_lik_mean = Likelihood_Wrapper(loglike,mu,
                                      name_mu,init_diu,bounds_mu,
                                      'Eq. (2), monthly means')
    diu_lik_mean.setargs_seq(mjd_m,diu_m)
    diu_lik_mean.like_minimize(method=method,tol=tol)

    diu_lik_dCm = Likelihood_Wrapper(loglike_dCm,mu_dCm,
                                     name_mu_dCm,init_diu_dCm,bounds_mu_dCm,
                                     'Eq. (2), monthly means, w/ amplitude increase')    
    diu_lik_dCm.setargs_seq(mjd_m,diu_m)
    diu_lik_dCm.like_minimize(method=method,tol=tol)

    diu_lik_dCm_daily = Likelihood_Wrapper(loglike_dCm,mu_dCm,
                                           name_mu_dCm,init_diu_dCm,bounds_mu_dCm,
                                           'Eq. (2), all data, w/ amplitude increase')    
    diu_lik_dCm_daily.setargs_seq(mjd_s,diu_s)
    diu_lik_dCm_daily.like_minimize(method=method,tol=tol)
    diu_lik_dCm_daily.chi_square_ndf()
    
    diu_lik_dCm_daily_hum = Likelihood_Wrapper(loglike_dCm_hum,mu_dCm_hum,
                                           name_mu_dCm,init_diu_dCm,bounds_mu_dCm,
                                           'Eq. (2), all data, w/ amplitude increase and hum corr')    
    diu_lik_dCm_daily_hum.setargs_seq_hum(mjd_s,diu_s,hum_s)
    diu_lik_dCm_daily_hum.like_minimize(method=method,tol=tol)
    diu_lik_dCm_daily_hum.chi_square_ndf()    

    t_max = df_s[name_temperature].resample('D').max().dropna()
    t_min = df_s[name_temperature].resample('D').min().dropna()
    residuals = diu_lik_daily.full_residuals()
    print ('RESIDUALS: ', residuals[residuals>7])

    plt.figure(figsize = (10,5), constrained_layout = True)
    h,p = plot_profile(hum_s,residuals,nbins=50,maximum=100.5)
    popt, pcov = curve_fit(hum_dev, h[h>hum_dev_start-2],p[h>hum_dev_start-2], [0.05,0.001])
    plt.plot(h[h>hum_dev_start-2],hum_dev(h[h>hum_dev_start-2],*popt),'--',linewidth=4,color='r',label=r'fit: -%.3f$\cdot$(RH-80)-%0.4f$\cdot$(RH-80)$^{2}$' % tuple(popt))
    plt.xlim(0.,100.)
    plt.legend(loc='best')
    plt.xlabel('Relative humidity (%)')
    plt.ylabel(r'DTR fit residuals')
    plt.savefig('{:s}/DiurnalTemperature_corr_d_humidity_NOT{:s}.pdf'.format(resultdir,tits))
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    residuals = diu_lik_dCm_daily_hum.full_residuals()    
    #plt.plot(hum_s,diu_lik_daily_nob.full_residuals(),'.',color='violet')
    plot_profile(hum_s,diu_lik_dCm_daily_hum.full_residuals(),nbins=50,maximum=100.5)
    plt.xlim(0.,100.)    
    plt.xlabel('Relative humidity (%)')
    plt.ylabel(r'DTR fit residuals')
    plt.savefig('{:s}/DiurnalTemperature_corr_d_humidity_hum_NOT{:s}.pdf'.format(resultdir,tits))

    if is_naoi:
        plt.figure()
        naoi_correlate(df_naoi,diu_lik_daily.full_residuals(),color='r')
        naoi_profile(df_naoi,diu_lik_daily.full_residuals(),nbins=12)
        plt.savefig('DiurnalTemperature_corr_d_NAOI_NOT.pdf',bbox_inches='tight')

        plt.figure()
        naoi_correlate(df_naoi,diu_lik_mean.full_residuals(),color='r')
        naoi_profile(df_naoi,diu_lik_mean.full_residuals(),nbins=12)
        plt.savefig('DiurnalTemperature_corr_m_NAOI_NOT.pdf',bbox_inches='tight')

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_diurnal_spread(dfff_5min, name_temperature, coverage)
    plot_diurnal_spread(dfff_1min, name_temperature, coverage)    
    #plot_diurnal_fit_results(mjd_s,diu_lik_dCm_daily,is_daily=True,color='r')
    #plot_diurnal_fit_results(mjd_s_NOT,diu_lik_dCm_NOT_daily,is_daily=True,color='b')
    #plot_diurnal_fit_results(mjd_m,diu_lik_dCm,is_daily=False,color='orange')
    plot_diurnal_fit_results(mjd_s,diu_lik_dCm_daily_hum,df_hum=hum_s,is_daily=True,color='r')    
    #plot_diurnal_fit_results(mjd_m,diu_lik_dCm_hum,is_daily=False,color='violet')
    #plot_diurnal_fit_results(mjd_m_NOT,diu_lik_dCm_NOT,is_daily=False,color='violet')
    #plt.xlabel('Year')
    plt.ylabel('DTR (ºC)', fontsize=18)
    plt.savefig('{:s}/Temperaturadiurnal_sencer_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    #2D profile of diurnal temperature increase and amplitude increase
    plt.figure(figsize = (8,7), constrained_layout = True)
    plt.xlabel(r'$b$ (DTR increase) (ºC/10y)', fontsize = 25)
    plt.ylabel(r'$\Delta C_{m}$ (DTR seasonal ampltiude increase) (ºC/10y)', fontsize = 22)
    diu_lik_dCm_daily.profile_likelihood_2d(1,4,chi2=14,NN=60,method=method,tol=1e-8,clabel=r'$D(b,\Delta C_{m})$')
    ax = plt.gca()
    ax.set_xlim(0,0.4)
    ax.set_ylim(0,1.0)
    plt.savefig('{:s}/DiurnalTemperature_profiled_daily_contour_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    #2D profile of diurnal temperature increase and amplitude increase
    plt.figure(figsize = (8,7), constrained_layout = True)
    plt.xlabel(r'$b$ (DTR increase) (ºC/10y)', fontsize = 25)
    plt.ylabel(r'$\Delta C_{m}$ (DTR seasonal ampltiude increase) (ºC/10y)', fontsize = 22)
    diu_lik_dCm_daily_hum.profile_likelihood_2d(1,4,chi2=12,NN=60,method=method,tol=1e-8,clabel=r'$D(b,\Delta C_{m})$')    
    ax = plt.gca()
    ax.set_xlim(0,0.4)
    ax.set_ylim(0,1.0)
    plt.savefig('{:s}/DiurnalTemperature_profiled_dailyhum_contour_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    # Profile the diurnal temperature increase parameter now
    NN   = 50
    chi2 = 10

    plt.clf()
    plt.xlabel(r'$\Delta C_{m}$ (seasonal oscillation ampltiude increase) (ºC/10y)', fontsize = 22)
    plt.ylabel(r'$D(b)$', fontsize = 25)
    diu_lik_dCm_daily.profile_likelihood(4,chi2=11,NN=NN,method=method,col='g')
    diu_lik_dCm.profile_likelihood(4,chi2=8,NN=NN,method=method,col='orange')
    diu_lik_dCm_daily_hum.profile_likelihood(4,chi2=8,NN=NN,method=method,col='deepskyblue')
    plt.savefig('{:s}/DiurnalTemperatureSeasonal_b_profiled_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    
    # Profile the diurnal temperature increase parameter now
    plt.clf()
    plt.xlabel(r'$b$ (diurnal $\Delta T$ increase) (ºC/10y)', fontsize = 24)    
    plt.ylabel(r'$D(b)$', fontsize = 25)
    diu_lik_daily.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g')
    diu_lik_dCm_daily.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='violet')
    diu_lik_dCm_daily_hum.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='deepskyblue')
    diu_lik_mean.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='orange')
    plt.savefig('{:s}/DiurnalTemperature_b_profiled_NOT{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    num = 20
    grad_thr = 1

    print ('TGRADIENT1 TD: ',dfff.loc[ dfff['Tgradient1R'] > grad_thr, 'diff1'].head(n=num))
    #print ('TGRADIENT5 TE: ',dfff.loc[ dfff['Tgradient5'] > grad_thr, name_temperature].head(n=num))
    #print ('TGRADIENT5 TED: ',dfff.loc[ dfff['Tgradient5'] > grad_thr, 'Tgradient5'].head(n=num))
    print ('TGRADIENT1 TED: ',dfff.loc[ dfff['Tgradient1R'] > grad_thr, 'Tgradient1R'].head(n=num))
    #print ('TDIFF1: ', dfff.loc[ dfff['Tgradient1R'] > grad_thr,'Tdiff1'].head(n=num))
    #print ('TGRADIENT5R TED: ',dfff.loc[ dfff['Tgradient5'] > grad_thr, 'Tgradient5R'].head(n=num))
    #print ('TGRADIENT1R TED: ',dfff.loc[ dfff['Tgradient5'] > grad_thr, 'Tgradient1R'].head(n=num))
 
    grad_thr = -1

    print ('TGRADIENT1 TD: ',dfff.loc[ dfff['Tgradient1R'] < grad_thr, 'diff1'].head(n=num))
    #print ('TGRADIENT5 TE: ',dfff.loc[ dfff['Tgradient5'] < grad_thr, 'temperature'].head(n=num))
    #print ('TGRADIENT5 TED: ',dfff.loc[ dfff['Tgradient5'] < grad_thr, 'Tgradient5'].head(n=num))
    print ('TGRADIENT1 TED: ',dfff.loc[ dfff['Tgradient1R'] < grad_thr, 'Tgradient1R'].head(n=num))
    #print ('TDIFF1: ', dfff.loc[ dfff['Tgradient1R'] < grad_thr,'Tdiff1'].head(n=num))
    #print ('TGRADIENT5R TED: ',dfff.loc[ dfff['Tgradient5'] < grad_thr, 'Tgradient5R'].head(n=num))
    #print ('TGRADIENT1R TED: ',dfff.loc[ dfff['Tgradient5'] < grad_thr, 'Tgradient1R'].head(n=num))

    gradients = [1,5]
    
    for i in gradients:
    
        var = 'Tgradient' + str(i) 

        #Distribucions anuals i mensuals:
        plt.clf()
        plot_hist(dfff, var, 0.01,'Temperature change rate (ºC/min)','Probability / (ºC/min)', 'ºC/min', xoff=0.08, coverage_cut=coverage_cut)
        #plt.plot(x,y, color= 'steelblue')
        plt.savefig('{:s}/Hist_'+var+'{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
        plt.show()

        plt.clf()
        plot_hist(dfff, var, 0.01,'Temperature change rate (ºC/min)','Probability / (ºC/min)', 'ºC/min', xoff=0.08, is_night=True, coverage_cut=coverage_cut)
        #plt.plot(x,y, color= 'steelblue')
        plt.savefig('{:s}/Hist_'+var+'_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
        plt.show()
        
        plt.clf()
        plot_mensual(dfff, var, 'Temperature change rate (ºC/min)')
        plt.savefig(resultdir+'/'+var+'_mensual{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize = (10,5), constrained_layout = True)
        plot_historic(dfff, var, coverage)
        #plt.xlabel('Year')
        plt.ylabel('Temperature change rate (ºC/min)',fontsize=18)
        plt.savefig(resultdir+'/'+var+'_sencer{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()
        
    name_temperature = name_temperature_save 
    name_humidity    = name_humidity_save 

def plot_DP() -> None:

    dfff = dff[((dff['humidity_reliable']==True) & (dff['temperature_reliable']==True))]

    mask_highRH = (dfff[name_humidity] > 70)
    mask_lowDP  = ((dfff[name_temperature]-dfff['DP']) < 5)
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    h,p = plot_profile(dfff[name_humidity],dfff[name_temperature]-dfff['DP'],nbins=50)
    plt.xlabel('Relative Humidity (%)', fontsize=22)
    plt.ylabel('Temperature minus dew point (ºC)',fontsize=22)
    plt.legend(loc='best')
    plt.savefig('{:s}/DP_vs_Humidity_corr{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    plt.figure(figsize = (10,5), constrained_layout = True)
    h,p = plot_profile(dfff.loc[mask_highRH,name_humidity],dfff.loc[mask_highRH,name_temperature]-dfff.loc[mask_highRH,'DP'],nbins=50)
    plt.xlabel('Relative Humidity (%)', fontsize=22)
    plt.ylabel('Temperature minus dew point (ºC)',fontsize=22)
    plt.legend(loc='best')
    plt.savefig('{:s}/DP_vs_HighHumidity_corr{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    plt.figure(figsize = (10,5), constrained_layout = True)
    h,p = plot_profile(dfff[name_temperature]-dfff['DP'],dfff[name_humidity],nbins=50)
    plt.ylabel('Relative Humidity (%)', fontsize=22)
    plt.xlabel('Temperature minus dew point (ºC)',fontsize=22)
    plt.legend(loc='best')
    plt.savefig('{:s}/Humidity_vs_DP_corr{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    plt.figure(figsize = (10,5), constrained_layout = True)
    h,p = plot_profile(dfff.loc[mask_lowDP,name_temperature]-dfff.loc[mask_lowDP,'DP'],dfff.loc[mask_lowDP,name_humidity],nbins=50)
    plt.ylabel('Relative Humidity (%)', fontsize=22)
    plt.xlabel('Temperature minus dew point (ºC)',fontsize=22)
    plt.legend(loc='best')
    plt.savefig('{:s}/Humidity_vs_LowDP_corr{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    
def plot_humidity() -> None:

    #plt.rcParams.update(params)

    tits = '_2024'
    
    is_offset = False
    is_coauthor_tests = True

    if is_NOT:
        plt.figure()
        #not_correlate(df_not['Humidity'],dff['humidity'],color='r')
        not_profile(df_not['Humidity'].resample('D').median(),dff[name_humidity].resample('D').median(),nbins=25)    
        plt.savefig('{:s}/Humidity_corr_NOT{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

        plt.figure(figsize = (10,5), constrained_layout = True)
        not_profile_time(df_not,dff,'Humidity',name_humidity)
        plt.ylabel('RH (MAGIC) - RH (NOT)')
        plt.savefig('{:s}/Humidity_corr_NOT_time{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

        plt.figure(figsize = (10,5), constrained_layout = True)
        not_profile_time(df_not[df_not['Humidity']<90],dff[dff[name_humidity]<90],'Humidity',name_humidity,85,print_threshold=50)
        plt.ylabel('RH (MAGIC) - RH (NOT)')
        plt.savefig('{:s}/Humidity_corr_NOT_time_lowhum{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

        plt.figure(figsize = (10,5), constrained_layout = True)
        not_profile_time(df_not,dff[dff['temperature_reliable']==True],'TempInAirDegC',name_temperature,85,print_threshold=10)
        plt.ylabel('Temp (MAGIC) - Temp (NOT)')
        plt.savefig('{:s}/Temperature_corr_NOT_time{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    day_coverage_for_humidity = 70
    
    if (is_20to22):
        dfff = dff[(dff['humidity_reliable']==True) | (dff.index.year==2020) | (dff.index.year==2021) | (dff.index.year==2022) | (dff.index.year==2023) ]
        #tits = '_with20to22'
        dfff_no = dff[(dff['humidity_reliable']==True) & (dff.index.year!=2020) & (dff.index.year!=2021) & (dff.index.year!=2022) ]
    else:
        dfff = dff[(dff['humidity_reliable']==True)]
        #tits = ''        

    dfff, coverage = apply_coverage(dfff,debug=False)
    dfn = dfff[(dfff['coverage'] > coverage_cut_for_daily_samples)]
    dfn.loc[(dfn[name_humidity]<2), name_humidity] = 1
    #mask = ((dfn.index > WS_relocation) & (dfn[name_humidity]<90))
    mask = (dfn[name_humidity]<90)
    mask1 = (((dfn.index < new_model) | (dfn.index > old_model))  & (dfn[name_humidity]<90))
    mask2 = ((dfn.index > new_model) & (dfn.index < old_model) & (dfn[name_humidity]<90))    
    mask_no2023 = ((dfn.index > WS_relocation) & (dfn.index < '2023-01-01 00:00:01') & (dfn[name_humidity]<90))

    mask_hahn = (mask &
                 ((dfn.index < '2007-01-01 00:00:01') | (dfn.index > '2009-01-01 23:59:59')) & 
                 ((dfn.index < '2021-01-01 00:00:01') | (dfn.index > '2023-01-01 00:00:01')) ) # email from 07/02/2024  
    
    mask_longo = (mask &
                 ((dfn.index < '2010-10-01 00:00:01') | (dfn.index > '2017-01-01 23:59:59')))  # email from 14/04/2024  
    
    mask_dorner = (mask &
                 ((dfn.index < '2009-01-01 00:00:01') | (dfn.index > '2010-12-31 23:59:59')) & 
                 (dfn.index > new_WS)   )  # email from 05/04/2024  
    
    mask_schmuck = (mask &
                    ((dfn.index < '2009-01-01 00:00:01') | (dfn.index > '2010-12-31 23:59:59')) &
                    ((dfn.index < '2015-01-01 00:00:01') | (dfn.index > '2016-12-31 23:59:59')) &
                    ((dfn.index < '2021-01-01 00:00:01') | (dfn.index > '2022-12-31 23:59:59'))  )  # email from 05/04/2024  
    
    if (is_20to22):
        dfff_no, coverage_no = apply_coverage(dfff_no,debug=False)
        dfn_no = dfff_no[(dfff_no['coverage'] > coverage_cut_for_daily_samples)]
        dfn_no.loc[(dfn_no[name_humidity]<2), name_humidity] = 1
        #mask = ((dfn.index > WS_relocation) & (dfn[name_humidity]<90))
        mask_no = (dfn_no[name_humidity]<90)
    
    dfn_H = dfff[(dfff['coverage'] > coverage_cut_for_monthly_samples)]
    dfn_H.loc[(dfn[name_humidity]<2), name_humidity] = 1
    mask_H = ((dfn_H.index > WS_relocation)  & (dfn_H[name_humidity]<90))
    
    init_mu = np.array([30., 0., 6.5, 9.9, 10.0])    
    hum_lik_mean   = Likelihood_Wrapper(loglike,mu,name_mu,init_mu,bounds_mu,'Eq. (2), Daily means')
    hum_lik_mean.setargs_df(dfn,name_humidity,mask,
                            is_daily=True,is_median=False,day_coverage=day_coverage_for_humidity)
    hum_lik_mean.like_minimize(method=method,tol=tol)
    H = hum_lik_mean.approx_hessian(eps=1e-5,only_diag=False)
    H_inv = np.linalg.inv(H)
    print ('Approximate inverse Hessian: ',H_inv)
    print ('Approximate parameter uncertainties:', np.sqrt(np.diag(H_inv)))

    init_mu2 = np.array([31.5, 0.,25.4, 1.2, 33., 7.6, 26.3])
    hum_lik_mean2   = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily means')
    hum_lik_mean2.setargs_df(dfn,name_humidity,mask,
                             is_daily=True,is_median=False,day_coverage=day_coverage_for_humidity)
    hum_lik_mean2.like_minimize(method=method,tol=tol)

    hum_lik_median2   = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily medians')
    hum_lik_median2.setargs_df(dfn,name_humidity,mask,
                               is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
    hum_lik_median2.like_minimize(method=method,tol=tol)

    hum_lik_median2_hahn = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily medians, excl. 01/2017-01/2017 & 2021-2022 ')
    hum_lik_median2_hahn.setargs_df(dfn,name_humidity,mask_hahn,
                                    is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
    hum_lik_median2_longo = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily medians, excl. 10/2010-01/2017')
    hum_lik_median2_longo.setargs_df(dfn,name_humidity,mask_longo,
                                    is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
    hum_lik_median2_dorner = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily medians, excl. <03/2007 & (2009-2010)')
    hum_lik_median2_dorner.setargs_df(dfn,name_humidity,mask_dorner,
                                    is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
    hum_lik_median2_schmuck = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily medians, excl. (2009-2010) & (2015-2016) & (2021-2022)')
    hum_lik_median2_schmuck.setargs_df(dfn,name_humidity,mask_schmuck,
                                    is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
    if is_coauthor_tests:
        hum_lik_median2_hahn.like_minimize(method=method,tol=tol)
        hum_lik_median2_longo.like_minimize(method=method,tol=tol)
        hum_lik_median2_dorner.like_minimize(method=method,tol=tol)
        hum_lik_median2_schmuck.like_minimize(method=method,tol=tol)        

    init_mu4 = np.array([24.5, 0.,20.8, 3.8, 26., 6.9, 17.1, -2.6, 3.1])
    hum_lik_median4   = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_mu4,bounds_mu2_sig2,
                                           'Daily medians, seas. spreads')
    hum_lik_median4.setargs_df(dfn,name_humidity,mask,
                               is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
    hum_lik_median4.like_minimize(method=method,tol=tol)

    init_mu2_offset = np.array([31.5, 0.,25.4, 1.2, 33., 7.6, 26.3,0.])
    hum_lik_median2_offset = Likelihood_Wrapper(loglike2_2sets,mu2,name_mu2_offset,init_mu2_offset,bounds_mu2_offset,
                                                    'Daily medians w/ offset')
    hum_lik_median2_offset.setargs_df2(dfn,name_humidity,mask1, mask2,
                               is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
    hum_lik_median2_offset.like_minimize(method=method,tol=tol)

    init_mu4_offset = np.array([24.5, 0.,20.8, 3.8, 26., 6.9, 17.1, -2.6, 3.1, 0.])
    hum_lik_median4_offset = Likelihood_Wrapper(loglike4_2sets,mu2,name_mu2_sig2_offset,init_mu4_offset,bounds_mu2_sig2_offset,
                                                    'Daily medians, seas. spreads w/ offset')
    hum_lik_median4_offset.setargs_df2(dfn,name_humidity,mask1, mask2,
                               is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
    hum_lik_median4_offset.like_minimize(method=method,tol=tol)

    if (is_20to22):
        hum_lik_median2_no = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily medians, w/o 2020-2022')
        hum_lik_median2_no.setargs_df(dfn_no,name_humidity,mask_no,
                                   is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
        hum_lik_median2_no.like_minimize(method=method,tol=tol)

        hum_lik_median4_no = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_mu4,bounds_mu2_sig2,
                                                  'Daily medians, seas. spreads, w/o 2020-2022')
        hum_lik_median4_no.setargs_df(dfn_no,name_humidity,mask_no,is_daily=True,is_median=True,day_coverage=day_coverage_for_humidity)
        hum_lik_median4_no.like_minimize(method=method,tol=tol)

    init_month = np.array([28., 0.,20., 3.7, 24., 7., 17.])
    hum_lik_month  = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_month,bounds_mu2,'Monthly means')
    hum_lik_month.setargs_df(dfn_H,name_humidity,mask_H,
                             is_daily=False,is_median=False)
    hum_lik_month.like_minimize(method=method,tol=tol)
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Relative Humidity (%)')
    hum_lik_mean.plot_residuals()
    plt.savefig('{:s}/Humidity_residuals{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Relative Humidity (%)')
    hum_lik_mean2.plot_residuals()
    plt.savefig('{:s}/Humidity_residuals2{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    residuals_mean = hum_lik_mean.red_residuals()
    chi2_measured_mean = hum_lik_mean.chi_square_ndf()
    residuals_mean2 = hum_lik_mean2.red_residuals()
    chi2_measured_mean2 = hum_lik_mean2.chi_square_ndf()
    residuals_median2 = hum_lik_median2.red_residuals()
    chi2_measured_median2 = hum_lik_median2.chi_square_ndf()
    residuals_median4 = hum_lik_median4.red_residuals()
    chi2_measured_median4 = hum_lik_median4.chi_square_ndf()
    residuals_median4_offset = hum_lik_median4_offset.red_residuals()
    chi2_measured_median4_offset = hum_lik_median4_offset.chi_square_ndf()
    residuals_median2_offset = hum_lik_median2_offset.red_residuals()
    chi2_measured_median2_offset = hum_lik_median2_offset.chi_square_ndf()
    residuals_month = hum_lik_month.red_residuals()
    chi2_measured_month = hum_lik_month.chi_square_ndf()

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median4,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median4**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median4),fontsize=20)
    a[1].legend(loc='best')    
    plt.savefig('{:s}/Humidity_residuals_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median4_offset,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median4_offset**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median4_offset),fontsize=20)
    a[1].legend(loc='best')    
    plt.savefig('{:s}/Humidity_residuals_hist_offset{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    if is_naoi:
        plt.figure()
        naoi_correlate(df_naoi,hum_lik_mean.full_residuals(),color='r')
        naoi_profile(df_naoi,hum_lik_mean.full_residuals(),nbins=12)    
        plt.savefig('{:s}/Humidity_corr_d_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    # Profile the humidity increase parameter now
    plt.figure()
    plt.xlabel(r'$b$ (humidity increase) (%/10y)', fontsize = 25)
    plt.ylabel(r'$D(b)$', fontsize = 25)

    NN   = 28
    if (is_20to22):
        chi2 = 34
    else:
        chi2 = 35
    hum_lik_mean2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='b')        
    hum_lik_median2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g')    
    hum_lik_month.profile_likelihood(1,chi2=25,NN=NN,method=method,col='orange')
    # Create empty plot with blank marker containing the extra label
    plt.plot([], [], ' ', label="Robustness tests:")        
    #hum_lik_mean.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='darkviolet',alpha=0.2)
    if is_20to22:
        hum_lik_median4.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='r',alpha=0.2)
        hum_lik_median2_no.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='cyan',alpha=0.2)
        if is_offset:
            hum_lik_median2_offset.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='k',alpha=0.2)
            #hum_lik_median4_offset.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='k',alpha=0.2)     
            tits = tits + '_offset'
    else:
        hum_lik_month.profile_likelihood(1,chi2=23,NN=NN,method=method,col='orange')
    if is_coauthor_tests:
        hum_lik_median2_hahn.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='paleturquoise')
        hum_lik_median2_longo.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='mediumturquoise')
        hum_lik_median2_dorner.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='lightcyan')
        hum_lik_median2_schmuck.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='lightseagreen')
        tits = tits + '_coauthors'
        
    plt.savefig('{:s}/Humidity_b_profiled{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    tits = tits.replace('_offset','')
    tits = tits.replace('_coauthors','')
    
    plt.clf()
    plt.hist(dfff[name_humidity], density = True, bins = 100)
    plt.xlabel('Relative humidity (%)')
    plt.ylabel('Probability / %')
    #plt.xlim(0,100)
    plt.savefig('{:s}/Histogram_humidity{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dfff.index, dfff[name_humidity])
    plt.ylabel('Relative humidity (%)')
    plt.savefig('{:s}/Histogram_lowhumidity_vsTime{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
        
    plt.clf()
    plt.hist(dfff[name_humidity], density = True, bins = 100, log=True)
    plt.xlabel('Relative humidity (%)')
    plt.ylabel('Probability / %')
    #plt.xlim(0,100)
    plt.savefig('{:s}/Histogram_loghumidity{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_hist(dfff, name_humidity, 1.,'Relative humidity (%)','Probability / %', '\%', xoff=0., loc='right', coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_humidity{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_hist(dfff, name_humidity, 1.,'Relative humidity (%)','Probability / %', '\%', xoff=0., loc='right', is_night=True, coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_humidity_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
 
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual(dfff, name_humidity,'RH (%)', fullfits=True)
    plt.savefig('{:s}/Humidity_mensual_fullfits{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual(dfff, name_humidity,'Relative humidity (%)')
    plt.savefig('{:s}/Humidity_mensual{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual(dfn, name_humidity,'Relative humidity (%)')
    plt.savefig('{:s}/Humidity_mensual_low{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual_diurnal(dfn, name_humidity,r'$\Delta$ RH (%)', min_coverage=coverage_cut, day_coverage=day_coverage_for_samples, is_lombardi=True)
    plt.ylim([-20,20])
    plt.savefig('{:s}/Humidity_mensual_diurnal{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
        
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_sunalt(dfn[dfn[name_humidity]<90], name_humidity,r'Median relative humidity (%)', join_months=True)
    plt.ylim([10., 50.])
    plt.savefig('{:s}/Humidity_diurnal{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual_distributions(dfn, name_humidity, 'Relative Humidity - monthly mean (ºC)', 0.1)
    plt.savefig('{:s}/Humidity_distributions{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff, name_humidity, coverage)
    plt.ylabel('Relative Humidity (%)')
    #plt.xlabel('Year')
    plt.savefig('{:s}/Humidity_sencer{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff, name_humidity, coverage)
    plt.ylabel('Relative Humidity (%)')
    #plt.xlabel('Year')
    #plot_historic_fit_results(dfn,mask,hum_lik_median,is_daily=True, day_coverage=day_coverage_for_samples)
    #plot_historic_fit_results(dfn,mask,hum_lik_median4,is_daily=True, day_coverage=day_coverage_for_samples,color='orange',is_sigma2=True)
    if is_offset:
        plot_historic_fit_results(dfn,mask1,hum_lik_median4_offset,is_daily=True, day_coverage=day_coverage_for_humidity,
                                  color='red',is_sigma2=True,is_offset=True)
        plot_historic_fit_results(dfn,mask2,hum_lik_median4_offset,is_daily=True, day_coverage=day_coverage_for_humidity,
                                  color='orange',is_sigma2=True,is_offset=True,offset=hum_lik_median4_offset.res.x[-1])
        tits = tits + '_offset'        
    else:
        plot_historic_fit_results(dfn,mask,hum_lik_median4,is_daily=True, day_coverage=day_coverage_for_humidity,color='red',is_sigma2=True)
    plt.savefig('{:s}/Humidity_sencer_fits{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    tits = tits.replace('_offset','')
    

    gradients = [1,5,10]
    
    for i in gradients:
    
        var = 'Rgradient' + str(i) + 'R'

        #Distribucions anuals i mensuals:
        plt.clf()
        plot_hist(dfff, var, 0.5,'Humidity change rate (%/min)','Probability / (%/min)', '%/min', xoff=0.08, coverage_cut=coverage_cut)
        #plt.plot(x,y, color= 'steelblue')
        plt.savefig(resultdir+'/Hist_'+var+'{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()

        plt.clf()
        plot_hist(dfff, var, 0.5,'Humidity change rate (%/min)','Probability / (%/min)', '%/min', xoff=0.08, is_night=True, coverage_cut=coverage_cut)
        #plt.plot(x,y, color= 'steelblue')
        plt.savefig(resultdir+'/Hist_'+var+'_night{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()
        
        plt.clf()
        plot_mensual(dfff, var, 'Humidity change rate (%/min)')
        plt.savefig(resultdir+'/'+var+'_mensual{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()
    
        plt.figure(figsize = (10,5), constrained_layout = True)
        plot_historic(dfff, var, coverage)
        #plt.xlabel('Year')
        plt.ylabel('Humidity change rate (%/min)')
        plt.savefig(resultdir+'/'+var+'_sencer{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()

    dffff = dfff[(dfff['temperature_reliable']==True)]

    maskRvsTa = ((dffff['Rgradient1R'] > 0.1) | (dffff['Rgradient1R'] < -0.1))
    maskRvsTb = ((dffff['Tgradient1R'] > 0.01) | (dffff['Tgradient1R'] < -0.01))
    maskRvsT  = (maskRvsTa & maskRvsTb)
    maskStorm = (((dffff['windSpeedAverage'] > 25.)) & maskRvsT)
    maskNoStorm = (((dffff['windSpeedAverage'] < 20.)) & maskRvsT)

    plt.clf()
    plt.plot(dffff.loc[maskRvsT,'Rgradient1R'], dffff.loc[maskRvsT,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient1_vs_Rgradient1{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskStorm,'Rgradient1R'], dffff.loc[maskStorm,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient1_vs_Rgradient1_storm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskNoStorm,'Rgradient1R'], dffff.loc[maskNoStorm,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient1_vs_Rgradient1_nostorm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    
    maskRvsT = (((dffff['Rgradient5R'] > 0.1) | (dffff['Rgradient5R'] < -0.1)) & ((dffff['Tgradient5R'] > 0.01) | (dffff['Tgradient5R'] < -0.01)))
    maskStorm = (((dffff['windSpeedAverage'] > 25.)) & maskRvsT)
    maskNoStorm = (((dffff['windSpeedAverage'] < 20.)) & maskRvsT)

    plt.clf()
    plt.plot(dffff.loc[maskRvsT,'Rgradient5R'], dffff.loc[maskRvsT,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Rgradient5{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskStorm,'Rgradient5R'], dffff.loc[maskStorm,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Rgradient5_storm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskNoStorm,'Rgradient5R'], dffff.loc[maskNoStorm,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Rgradient5_nostorm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    dfff_n = dff[(dff['humidity_reliable']==False)]
    dfff_t = dfff_n[(dfff_n['temperature_reliable']==False)]  #Filtre_Temperatures(dfff_n)

    plt.clf()
    plt.plot(dfff_t[name_humidity], dfff_t[name_temperature]-dfff_t['DP'],'ro',linestyle='None')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Temperature minus dew point (ºC)')
    plt.savefig('{:s}/TemperatureDP_vs_Humidity_baddata{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    dfff_h = dfff_t[(dfff_t[name_humidity]>99.5)]

    plt.clf()
    plt.hist(dfff_h[name_temperature]-dfff_h['DP'],bins = 100, log=True)
    plt.xlabel('Temperature minus dew point (ºC)')
    plt.savefig('{:s}/Temperature_vs_DP_highHum_baddata{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff[name_humidity],  dffff[name_temperature]-dffff['DP'],'ro',linestyle='None')
    plt.xlabel('Humidity (%/min)')
    plt.ylabel('Temperature - dew point (ºC)')
    plt.savefig('{:s}/TemperatureDP_vs_Humidity{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.expand_frame_repr', False)
    #print ('DEW POINT HIGH HUM: ',dfff_h.loc[ (dfff_h['temperature']-dfff_h['DP']) > 10., 'humidity'].head(n=300))
    #print ('DEW POINT HIGH HUM: ',dfff_h.loc[ (dfff_h['temperature']-dfff_h['DP']) > 10., 'DP'].head(n=300))

def plot_humidity_not() -> None:

    #plt.rcParams.update(params)

    df_not = df_not[df_not['Tgradient1']<3]
    print (df_not.head())

    pd.set_option('display.max_columns', 20)

    name_temperature_save = name_temperature
    name_humidity_save = name_humidity
    name_temperature = 'TempInAirDegC'
    name_humidity = 'Humidity'
    day_coverage_for_not = 80  # a bit lower because of early samples showing coverage of only 85%

    mask_5min = ((df_not.index < NOT_end_of_5min) & (df_not[name_temperature]>-11) & (df_not[name_temperature]<35))
    mask_1min = ((df_not.index > NOT_end_of_5min) & (df_not[name_temperature]>-11) & (df_not[name_temperature]<35))

    dfff_5min, coverage_5min = apply_coverage(df_not[mask_5min],data_spacing=5,debug=False)
    dfff_1min, coverage_1min = apply_coverage(df_not[mask_1min],data_spacing=1,debug=False)

    dfff = pd.concat([dfff_5min, dfff_1min])
    coverage = pd.concat([coverage_5min, coverage_1min])
    print ('AFTER all', dfff)    
        
    is_offset = False
    tits = '_NOT'        
    
    dfn = dfff[(dfff['coverage'] > coverage_cut_for_daily_samples)]
    dfn.loc[(dfn[name_humidity]<2), name_humidity] = 1
    dfn_H = dfff[(dfff['coverage'] > coverage_cut_for_monthly_samples)]
    dfn_H.loc[(dfn[name_humidity]<2), name_humidity] = 1

    #mask = ((dfn.index > WS_relocation) & (dfn['humidity']<90))
    mask = (dfn[name_humidity]<90)
    mask_H = ((dfn_H.index > WS_relocation) & (dfn_H[name_humidity]<90))


    init_mu = np.array([30., 0., 6.5, 9.9, 10.0])    
    hum_lik_mean   = Likelihood_Wrapper(loglike,mu,name_mu,init_mu,bounds_mu,'Eq. (2), daily means')
    hum_lik_mean.setargs_df(dfn,name_humidity,mask,
                            is_daily=True,is_median=False,day_coverage=day_coverage_for_not)
    hum_lik_mean.like_minimize(method=method,tol=tol)
    
    init_mu2 = np.array([30., 0.,6., 7., 10., 7.6, 26.3])
    hum_lik_mean2   = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Eq. (5), daily means')
    hum_lik_mean2.setargs_df(dfn,name_humidity,mask,
                             is_daily=True,is_median=False,day_coverage=day_coverage_for_not)
    hum_lik_mean2.like_minimize(method=method,tol=tol)

    hum_lik_median2   = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Eq. (5), daily medians')
    hum_lik_median2.setargs_df(dfn,name_humidity,mask,
                               is_daily=True,is_median=True,day_coverage=day_coverage_for_not)
    hum_lik_median2.like_minimize(method=method,tol=tol)

    init_mu4 = np.array([24.5, 0.,20.8, 3.8, 26., 6.9, 17.1, -2.6, 3.1])    
    hum_lik_median4   = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_mu4,bounds_mu2_sig2,
                                           'Eq. (5), daily medians, seas. spreads')
    hum_lik_median4.setargs_df(dfn,name_humidity,mask,
                               is_daily=True,is_median=True,day_coverage=day_coverage_for_not)
    hum_lik_median4.like_minimize(method=method,tol=tol)

    #init_mu4_offset = np.array([24.5, 0.,20.8, 3.8, 26., 6.9, 17.1, -2.6, 3.1, 0.])
    #hum_lik_median4_offset = Likelihood_Wrapper(loglike4_2sets,mu2,name_mu2_sig2_offset,init_mu4_offset,bounds_mu2_sig2_offset,
    #                                                'Daily medians, seas. spreads with offset')
    #hum_lik_median4_offset.setargs_df2(dfn,name_humidity,mask1, mask2,
    #                           is_daily=True,is_median=True,day_coverage=day_coverage_for_not)
    #hum_lik_median4_offset.like_minimize(method=method,tol=tol)
    
    init_month = np.array([36.5, 0.,-25.4, 11.25, 33., 10.8, 12.9])
    hum_lik_month  = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_month,bounds_mu2,
                                        'Eq. (5), monthly means')
    hum_lik_month.setargs_df(dfn_H,name_humidity,mask_H,
                             is_daily=False,is_median=False)
    hum_lik_month.like_minimize(method=method,tol=tol)
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Relative Humidity (%)')
    hum_lik_mean.plot_residuals()
    plt.savefig('{:s}/Humidity_residuals{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Relative Humidity (%)')
    hum_lik_mean2.plot_residuals()
    plt.savefig('{:s}/Humidity_residuals2{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    residuals_mean = hum_lik_mean.red_residuals()
    chi2_measured_mean = hum_lik_mean.chi_square_ndf()
    residuals_mean2 = hum_lik_mean2.red_residuals()
    chi2_measured_mean2 = hum_lik_mean2.chi_square_ndf()
    residuals_median2 = hum_lik_median2.red_residuals()
    chi2_measured_median2 = hum_lik_median2.chi_square_ndf()
    residuals_median4 = hum_lik_median4.red_residuals()
    chi2_measured_median4 = hum_lik_median4.chi_square_ndf()
    #residuals_median4_offset = hum_lik_median4_offset.red_residuals()
    #chi2_measured_median4_offset = hum_lik_median4_offset.chi_square_ndf()
    residuals_month = hum_lik_month.red_residuals()
    chi2_measured_month = hum_lik_month.chi_square_ndf()

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median4,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median4**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median4),fontsize=20)
    a[1].legend(loc='best')    
    plt.savefig('{:s}/Humidity_residuals_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median4,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median4**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median4),fontsize=20)
    a[1].legend(loc='best')    
    plt.savefig('{:s}/Humidity_residuals_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    if is_naoi:
        plt.figure()
        naoi_correlate(df_naoi,hum_lik_mean.full_residuals(),color='r')
        naoi_profile(df_naoi,hum_lik_mean.full_residuals(),nbins=12)    
        plt.savefig('{:s}/Humidity_corr_d_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    # Profile the humidity increase parameter now
    plt.figure()
    plt.xlabel(r'$b$ (humidity increase) (%/10y)', fontsize = 25)
    plt.ylabel(r'$D(b)$', fontsize = 25)

    NN   = 28
    chi2 = 35
    hum_lik_mean.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g',add_sigma=0.8)
    hum_lik_mean2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='deepskyblue',add_sigma=0.8)        
    hum_lik_median2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='b',add_sigma=0.8)    
    hum_lik_median4.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='r',add_sigma=0.8)
    hum_lik_month.profile_likelihood(1,chi2=23,NN=NN,method=method,col='orange')        
    plt.savefig('{:s}/Humidity_b_profiled{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    
    plt.clf()
    plt.hist(dfff[name_humidity], density = True, bins = 100)
    plt.xlabel('Relative humidity (%)')
    plt.ylabel('Probability / %')
    #plt.xlim(0,100)
    plt.savefig('{:s}/Histogram_humidity{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dfff.index, dfff[name_humidity])
    plt.ylabel('Relative humidity (%)')
    plt.savefig('{:s}/Histogram_lowhumidity_vsTime{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
        
    plt.clf()
    plt.hist(dfff[name_humidity], density = True, bins = 100, log=True)
    plt.xlabel('Relative humidity (%)')
    plt.ylabel('Probability / %')
    #plt.xlim(0,100)
    plt.savefig('{:s}/Histogram_loghumidity{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_hist(dfff, name_humidity, 1.,'Relative humidity (%)','Probability / %', '\%', xoff=0., loc='right', coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_humidity{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_hist(dfff, name_humidity, 1.,'Relative humidity (%)','Probability / %', '\%', xoff=0., loc='right', is_night=True, coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_humidity_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
 
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual(dfff, name_humidity,'RH (%)', fullfits=True)
    plt.savefig('{:s}/Humidity_mensual_fullfits{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual(dfff, name_humidity,'Relative humidity (%)')
    plt.savefig('{:s}/Humidity_mensual{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual(dfn, name_humidity,'Relative humidity (%)')
    plt.savefig('{:s}/Humidity_mensual_low{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual_diurnal(dfn, name_humidity,r'$\Delta$ RH (%)', min_coverage=coverage_cut, day_coverage=day_coverage_for_samples, is_lombardi=True)
    plt.ylim([-20,20])
    plt.savefig('{:s}/Humidity_mensual_diurnal{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
        
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_sunalt(dfn[dfn[name_humidity]<90], name_humidity,r'Median relative humidity (%)', join_months=True)
    plt.ylim([10., 50.])
    plt.savefig('{:s}/Humidity_diurnal{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_mensual_distributions(dfn, name_humidity, 'Relative Humidity - monthly mean (ºC)', 0.1)
    plt.savefig('{:s}/Humidity_distributions{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff[mask_1min], name_humidity, coverage_1min)
    plot_historic(dfff[mask_5min], name_humidity, coverage_5min)
    plt.ylabel('Relative Humidity (%)')
    #plt.xlabel('Year')
    plt.savefig('{:s}/Humidity_sencer{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff[mask_1min], name_humidity, coverage_1min)
    plot_historic(dfff[mask_5min], name_humidity, coverage_5min)
    plt.ylabel('Relative Humidity (%)')
    #plt.xlabel('Year')
    #plot_historic_fit_results(dfn,mask,hum_lik_median2,is_daily=True, day_coverage=day_coverage_for_samples)
    #plot_historic_fit_results(dfn,mask,hum_lik_median4,is_daily=True, day_coverage=day_coverage_for_samples,color='orange',is_sigma2=True)
    plot_historic_fit_results(dfn,(dfn.index < NOT_end_of_5min),hum_lik_median2,is_daily=True, day_coverage=day_coverage_for_not * 2/5)
    plot_historic_fit_results(dfn,(dfn.index > NOT_end_of_5min),hum_lik_median2,is_daily=True, day_coverage=day_coverage_for_not * 2/1)
    #plot_historic_fit_results(dfn,mask,hum_lik_median4,is_daily=True, day_coverage=day_coverage_for_not,color='red',is_sigma2=True)
    plt.savefig('{:s}/Humidity_sencer_fits{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    gradients = [1,5,10]
    
    for i in gradients:
    
        var = 'Rgradient' + str(i) + 'R'

        #Distribucions anuals i mensuals:
        plt.clf()
        plot_hist(dfff, var, 0.5,'Humidity change rate (%/min)','Probability / (%/min)', '%/min', xoff=0.08, coverage_cut=coverage_cut)
        #plt.plot(x,y, color= 'steelblue')
        plt.savefig(resultdir+'/Hist_'+var+'{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()

        plt.clf()
        plot_hist(dfff, var, 0.5,'Humidity change rate (%/min)','Probability / (%/min)', '%/min', xoff=0.08, is_night=True, coverage_cut=coverage_cut)
        #plt.plot(x,y, color= 'steelblue')
        plt.savefig(resultdir+'/Hist_'+var+'_night{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()
        
        plt.clf()
        plot_mensual(dfff, var, 'Humidity change rate (%/min)')
        plt.savefig(resultdir+'/'+var+'_mensual{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()
    
        plt.figure(figsize = (10,5), constrained_layout = True)
        plot_historic(dfff, var, coverage)
        #plt.xlabel('Year')
        plt.ylabel('Humidity change rate (%/min)')
        plt.savefig(resultdir+'/'+var+'_sencer{:s}.pdf'.format(tits), bbox_inches='tight')
        plt.show()

    dffff = dfff[(dfff['temperature_reliable']==True)]

    maskRvsTa = ((dffff['Rgradient1R'] > 0.1) | (dffff['Rgradient1R'] < -0.1))
    maskRvsTb = ((dffff['Tgradient1R'] > 0.01) | (dffff['Tgradient1R'] < -0.01))
    maskRvsT  = (maskRvsTa & maskRvsTb)
    maskStorm = (((dffff['windSpeedAverage'] > 25.)) & maskRvsT)
    maskNoStorm = (((dffff['windSpeedAverage'] < 20.)) & maskRvsT)

    plt.clf()
    plt.plot(dffff.loc[maskRvsT,'Rgradient1R'], dffff.loc[maskRvsT,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient1_vs_Rgradient1{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskStorm,'Rgradient1R'], dffff.loc[maskStorm,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient1_vs_Rgradient1_storm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskNoStorm,'Rgradient1R'], dffff.loc[maskNoStorm,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient1_vs_Rgradient1_nostorm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    
    maskRvsT = (((dffff['Rgradient5R'] > 0.1) | (dffff['Rgradient5R'] < -0.1)) & ((dffff['Tgradient5R'] > 0.01) | (dffff['Tgradient5R'] < -0.01)))
    maskStorm = (((dffff['windSpeedAverage'] > 25.)) & maskRvsT)
    maskNoStorm = (((dffff['windSpeedAverage'] < 20.)) & maskRvsT)

    plt.clf()
    plt.plot(dffff.loc[maskRvsT,'Rgradient5R'], dffff.loc[maskRvsT,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Rgradient5{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskStorm,'Rgradient5R'], dffff.loc[maskStorm,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Rgradient5_storm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskNoStorm,'Rgradient5R'], dffff.loc[maskNoStorm,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Humidity change rate (%/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Rgradient5_nostorm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    dfff_n = dff[(dff['humidity_reliable']==False)]
    dfff_t = dfff_n[(dfff_n['temperature_reliable']==False)]  #Filtre_Temperatures(dfff_n)

    plt.clf()
    plt.plot(dfff_t[name_humidity], dfff_t[name_temperature]-dfff_t['DP'],'ro',linestyle='None')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Temperature minus dew point (ºC)')
    plt.savefig('{:s}/TemperatureDP_vs_Humidity_baddata{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    dfff_h = dfff_t[(dfff_t[name_humidity]>99.5)]

    plt.clf()
    plt.hist(dfff_h[name_temperature]-dfff_h['DP'],bins = 100, log=True)
    plt.xlabel('Temperature minus dew point (ºC)')
    plt.savefig('{:s}/Temperature_vs_DP_highHum_baddata{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff[name_humidity],  dffff[name_temperature]-dffff['DP'],'ro',linestyle='None')
    plt.xlabel('Humidity (%/min)')
    plt.ylabel('Temperature - dew point (ºC)')
    plt.savefig('{:s}/TemperatureDP_vs_Humidity{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    name_temperature = name_temperature_save 
    name_humidity    = name_humidity_save 
    
def plot_pressure_not() -> None:

    name_pressure_save = name_pressure
    name_pressure = 'PressureHPA'
    coverage_cut_for_not = 0
    
    #dfff, coverage = apply_coverage(df_not,debug=False)
    dfff, coverage = apply_coverage(df_not,debug=False)
    dfff = dfff[(dfff['PressureHPA']>730)]   # one outlier at 680 hPa
    dfn = dfff[(dfff['coverage'] > coverage_cut_for_not)] 

    print ('NOT: ',dfn.head())
    
    day_coverage_for_pressure = 40
    #day_coverage_for_pressure = 0

    mask = (dfn.index < '2022-11-01 00:00:01') 

    tits = '_NOT'

    init_press2 = np.array([772., 0., 5.5, 6.6, 4.0, 9.1, 3.1])
    press_lik_mean2   = Likelihood_Wrapper(loglike2,mu2,
                                           name_mu2,init_press2,bounds_mu2,
                                           'Daily means')
    press_lik_mean2.setargs_df(dfn,name_pressure,mask,
                               is_daily=True,is_median=False,day_coverage=day_coverage_for_pressure)
    press_lik_mean2.like_minimize(method=method,tol=tol)

    press_lik_median2   = Likelihood_Wrapper(loglike2,mu2,
                                             name_mu2,init_press2,bounds_mu2,
                                             'Daily medians')
    press_lik_median2.setargs_df(dfn,name_pressure,mask,
                                 is_daily=True,is_median=True,day_coverage=day_coverage_for_pressure)
    press_lik_median2.like_minimize(method=method,tol=tol)

    init_press4 = np.array([772., 0., 5.2, 6.6, 3.7, 9.0, 3.1, -1.0, 4.0])
    press_lik_median4   = Likelihood_Wrapper(loglike4,mu2,
                                             name_mu2_sig2,init_press4,bounds_mu2_sig2,
                                             'Daily medians, seas. spreads')

    press_lik_median4.setargs_df(dfn,name_pressure,mask,
                                 is_daily=True,is_median=True,day_coverage=day_coverage_for_pressure)
    press_lik_median4.like_minimize(method=method,tol=tol)
    
    dfn_H = dfff[(dfff['coverage'] > 0)]
    mask_H = (dfn_H.index > WS_relocation)
    
    init_month = np.array([772., 0.,5.4,6.6,4.0, 8.9, 1.6])
    press_lik_month  = Likelihood_Wrapper(loglike2,mu2,
                                          name_mu2,init_month,bounds_mu2,
                                          'Monthly means')
    press_lik_month.setargs_df(dfn_H,name_pressure,mask_H,
                               is_daily=False,is_median=False)
    press_lik_month.like_minimize(method=method,tol=tol)
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Atmospheric Pressure (hPa)')
    press_lik_mean2.plot_residuals()
    plt.savefig('{:s}/Pressure_residuals{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Atmospheric Pressure (hPa)')
    press_lik_median2.plot_residuals()
    plt.savefig('{:s}/Pressure_residuals2{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    residuals_mean2 = press_lik_mean2.red_residuals()
    chi2_measured_mean2 = press_lik_mean2.chi_square_ndf()
    residuals_median2 = press_lik_median2.red_residuals()
    chi2_measured_median2 = press_lik_median2.chi_square_ndf()
    residuals_median4 = press_lik_median4.red_residuals()
    chi2_measured_median4 = press_lik_median4.chi_square_ndf()
    residuals_month = press_lik_month.red_residuals()
    chi2_measured_month = press_lik_month.chi_square_ndf()

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median2,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median2**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median2),fontsize=20)
    a[1].legend(loc='best')    
    
    plt.savefig('{:s}/Pressure_residuals_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    if is_naoi:
        plt.figure()
        naoi_correlate(df_naoi,press_lik_median2.full_residuals(),color='r')
        naoi_profile(df_naoi,press_lik_median2.full_residuals(),nbins=12)    
        plt.savefig('{:s}/Pressure_corr_d_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    plt.figure()
    plt.xlabel(r'$b$ (atm. pressure increase) (%/10y)', fontsize = 25)
    plt.ylabel(r'$D(b)$', fontsize = 25)    
    NN   = 28
    chi2 = 35
    press_lik_mean2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='deepskyblue')        
    press_lik_median2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='b')
    press_lik_median4.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='r')    
    press_lik_month.profile_likelihood(1,chi2=23,NN=NN,method=method,col='orange')        
    plt.savefig('{:s}/Pressure_b_profiled{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    
    
    plt.clf()
    plt.hist(dfff[name_pressure], density = True, bins = 100)
    plt.xlabel('Pressure (mbar)')
    plt.ylabel('Probability / mbar')
    plt.savefig('{:s}/Histogram_pressure{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.hist(dfff[name_pressure], density = True, bins = 100, log=True)
    plt.xlabel('Pressure (mbar)'   , fontsize = 30)
    plt.ylabel('Probability / mbar', fontsize = 30)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.savefig('{:s}/Histogram_logpressure{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    #plt.clf()
    #plot_hist(dfff,name_pressure, 1.,'Atmospheric Pressure (hPa)','Probability / hPa', ' hPa',xoff=0., coverage_cut=coverage_cut_for_not)
    ##plt.plot(x,y, color= 'steelblue')
    #plt.savefig('{:s}/Hist_pressure{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    #plt.show()
    
    #plt.clf()
    #plot_hist(dfff,name_pressure, 1.,'Atmospheric Pressure (hPa)','Probability / hPa', ' hPa',xoff=0., is_night=True, coverage_cut=coverage_cut_for_not)
    ##plt.plot(x,y, color= 'steelblue')
    #plt.savefig('{:s}/Hist_pressure_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    #plt.show()

    plt.clf()
    plot_mensual(dfff, name_pressure, 'Atmospheric pressure (hPa)')
    plt.savefig('{:s}/Pressure_mensual{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff, name_pressure, coverage, min_coverage=0)
    plt.ylabel('Atmospheric pressure (hPa)')
    #plt.xlabel('Year')
    #plot_historic_fit_results(dfn,mask,press_lik_median2,is_daily=True, day_coverage=day_coverage_for_pressure,color='red',is_sigma2=False)
    plot_historic_fit_results(dfn,mask,press_lik_median4,is_daily=True, day_coverage=day_coverage_for_pressure,color='red',is_sigma2=True)        
    plt.savefig('{:s}/Pressure_sencer{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    name_pressure = name_pressure_save
    
def plot_pressure() -> None:

    dfff = dff[(dff['pressure_reliable']==True)]
    dfff_h = dff[((dff['pressure_reliable']==True) | (dff.index < new_WS))]
    
    dfff, coverage = apply_coverage(dfff,debug=False)
    dfff_h, coverage_h = apply_coverage(dfff_h,debug=False)
    dfn = dfff[(dfff['coverage'] > coverage_cut_for_daily_samples)]

    day_coverage_for_pressure = 90

    mask  = ((dfn.index > new_WS) & (dfn[name_pressure]>750))    
    mask1 = ((dfn.index > new_WS) & ((dfn.index < new_model) | (dfn.index > old_model))  & (dfn[name_pressure]>750))
    mask2 = ((dfn.index > new_model) & (dfn.index < old_model) & (dfn[name_pressure]>750))    
    #mask = ((dfn.index > WS_relocation)  & (dfn[name_pressure]>750))    

    tits = ''

    init_mu2 = np.array([786., 0., 5.6, 6.6, 4.2, 9.0, 3.1])
    press_lik_mean2 = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily means')
    press_lik_mean2.setargs_df(dfn,name_pressure,mask,
                               is_daily=True,is_median=False,day_coverage=day_coverage_for_pressure)
    press_lik_mean2.like_minimize(method=method,tol=tol)
    #H = press_lik_mean2.approx_hessian(eps=1e-5,only_diag=False)
    #H_inv = np.linalg.inv(H)
    #print ('Approximate inverse Hessian: ',H_inv)
    #print ('Approximate parameter uncertainties:', np.sqrt(np.diag(H_inv)))
    
    press_lik_median2   = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily medians')
    press_lik_median2.setargs_df(dfn,name_pressure,mask,
                                 is_daily=True,is_median=True,day_coverage=day_coverage_for_pressure)
    press_lik_median2.like_minimize(method=method,tol=tol)

    init_mu2_sig2 = np.array([786., 0., 5.6, 6.6, 4.2, 9.0, 3.1, -1.0, 4.0])
    press_lik_median4   = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_mu2_sig2,bounds_mu2_sig2,
                                             'Daily medians, seas. spreads')
    press_lik_median4.setargs_df(dfn,name_pressure,mask,
                                 is_daily=True,is_median=True,day_coverage=day_coverage_for_pressure)
    press_lik_median4.like_minimize(method=method,tol=tol)

    init_mu2_offset = np.array([786., 0., 5.6, 6.6, 4.2, 9.0, 3.1, 0.])
    press_lik_median2_offset = Likelihood_Wrapper(loglike2_2sets,mu2,name_mu2_offset,init_mu2_offset,bounds_mu2_offset,
                                                  'Daily medians with offset')
    press_lik_median2_offset.setargs_df2(dfn,name_pressure,mask1, mask2, 
                                 is_daily=True,is_median=True,day_coverage=day_coverage_for_pressure)
    press_lik_median2_offset.like_minimize(method=method,tol=tol)

    press_lik_mean2_offset = Likelihood_Wrapper(loglike2_2sets,mu2,name_mu2_offset,init_mu2_offset,bounds_mu2_offset,
                                                  'Daily means with offset')
    press_lik_mean2_offset.setargs_df2(dfn,name_pressure,mask1, mask2, 
                                       is_daily=True,is_median=False,day_coverage=day_coverage_for_pressure)
    press_lik_mean2_offset.like_minimize(method=method,tol=tol)

    init_mu2_sig2_offset = np.array([786., 0., 5.6, 6.6, 4.2, 9.0, 3.1, -1.0, 4.0, 0.])
    press_lik_median4_offset = Likelihood_Wrapper(loglike4_2sets,mu2,name_mu2_sig2_offset,init_mu2_sig2_offset,bounds_mu2_sig2_offset,
                                                  'Daily medians, seas. spreads with offset')
    press_lik_median4_offset.setargs_df2(dfn,name_pressure,mask1,mask2,
                                         is_daily=True,is_median=True,day_coverage=day_coverage_for_pressure)
    press_lik_median4_offset.like_minimize(method=method,tol=tol)

    press_lik_mean4_offset = Likelihood_Wrapper(loglike4_2sets,mu2,name_mu2_sig2_offset,init_mu2_sig2_offset,bounds_mu2_sig2_offset,
                                                  'Daily means, seas. spreads with offset')
    press_lik_mean4_offset.setargs_df2(dfn,name_pressure,mask1,mask2,
                                         is_daily=True,is_median=False,day_coverage=day_coverage_for_pressure)
    press_lik_mean4_offset.like_minimize(method=method,tol=tol)

    press_lik_median41   = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_mu2_sig2,bounds_mu2_sig2,
                                              'Daily medians, seas. spreads (2007-2017)')
    press_lik_median41.setargs_df(dfn,name_pressure,mask1,
                                 is_daily=True,is_median=True,day_coverage=day_coverage_for_pressure)
    press_lik_median41.like_minimize(method=method,tol=tol)

    press_lik_median42   = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_mu2_sig2,bounds_mu2_sig2,
                                              'Daily medians, seas. spreads (2017-2023)')
    press_lik_median42.setargs_df(dfn,name_pressure,mask2,
                                 is_daily=True,is_median=True,day_coverage=day_coverage_for_pressure)
    press_lik_median42.like_minimize(method=method,tol=tol)
    
    dfn_H = dfff[(dfff['coverage'] > coverage_cut_for_monthly_samples)]
    mask_H = ((dfn_H.index > new_WS)  & (dfn_H[name_pressure]>750))
    
    init_month = np.array([786., 0.,5.4,6.6,4.0, 8.9, 1.6])
    press_lik_month  = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_month,bounds_mu2,'Monthly means')
    press_lik_month.setargs_df(dfn_H,name_pressure,mask_H,
                               is_daily=False,is_median=False)
    press_lik_month.like_minimize(method=method,tol=tol)
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Atmospheric Pressure (hPa)')
    press_lik_mean2.plot_residuals()
    plt.savefig('{:s}/Pressure_residuals{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Atmospheric Pressure (hPa)')
    press_lik_median2.plot_residuals()
    plt.savefig('{:s}/Pressure_residuals2{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()


    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Atmospheric Pressure (hPa)')
    press_lik_median4_offset.plot_residuals(is_sigma2=True)
    plt.savefig('{:s}/Pressure_residuals4_offset{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    residuals_mean2 = press_lik_mean2.red_residuals()
    chi2_measured_mean2 = press_lik_mean2.chi_square_ndf()
    residuals_median2 = press_lik_median2.red_residuals()
    chi2_measured_median2 = press_lik_median2.chi_square_ndf()
    residuals_median4 = press_lik_median4.red_residuals(is_sigma2=True)
    chi2_measured_median4 = press_lik_median4.chi_square_ndf(is_sigma2=True)
    residuals_median2_offset = press_lik_median2_offset.red_residuals()
    chi2_measured_median2_offset = press_lik_median2_offset.chi_square_ndf()
    residuals_mean2_offset = press_lik_mean2_offset.red_residuals()
    chi2_measured_mean2_offset = press_lik_mean2_offset.chi_square_ndf()
    residuals_median4_offset = press_lik_median4_offset.red_residuals(is_sigma2=True)
    chi2_measured_median4_offset = press_lik_median4_offset.chi_square_ndf(is_sigma2=True)
    residuals_mean4_offset = press_lik_mean4_offset.red_residuals(is_sigma2=True)
    chi2_measured_mean4_offset = press_lik_mean4_offset.chi_square_ndf(is_sigma2=True)
    residuals_median41 = press_lik_median41.red_residuals(is_sigma2=True)
    chi2_measured_median41 = press_lik_median41.chi_square_ndf(is_sigma2=True)
    residuals_median42 = press_lik_median42.red_residuals(is_sigma2=True)
    chi2_measured_median42 = press_lik_median42.chi_square_ndf(is_sigma2=True)
    residuals_month = press_lik_month.red_residuals()
    chi2_measured_month = press_lik_month.chi_square_ndf()

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median4,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median4**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median4),fontsize=20)
    a[1].legend(loc='best')    
    plt.savefig('{:s}/Pressure_residuals_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median4_offset,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median4_offset**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median4_offset),fontsize=20)
    a[1].legend(loc='best')    
    plt.savefig('{:s}/Pressure_residuals_hist_offset{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    if is_naoi:
        plt.figure()
        naoi_correlate(df_naoi,press_lik_median2.full_residuals(),color='r')
        naoi_profile(df_naoi,press_lik_median2.full_residuals(),nbins=12)    
        plt.savefig('{:s}/Pressure_corr_d_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')

    plt.figure()
    plt.xlabel(r'$b$ (atm. pressure increase) (%/10y)', fontsize = 25)
    plt.ylabel(r'$D(b)$', fontsize = 25)    
    NN   = 28
    chi2 = 35
    press_lik_mean2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='b')        
    press_lik_median2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g')
    press_lik_month.profile_likelihood(1,chi2=23,NN=NN,method=method,col='orange')        
    plt.plot([], [], ' ', label="Robustness tests:")        
    press_lik_median4.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='r',alpha=0.2)
    press_lik_median2_offset.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='cyan')
    press_lik_median4_offset.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='k')    
    #press_lik_median41.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='tomato')
    #press_lik_median42.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='seagreen')    
    plt.savefig('{:s}/Pressure_b_profiled{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    
    
    plt.clf()
    plt.hist(dfff[name_pressure], density = True, bins = 100)
    plt.xlabel('Pressure (mbar)')
    plt.ylabel('Probability / mbar')
    plt.savefig('{:s}/Histogram_pressure{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.hist(dfff[name_pressure], density = True, bins = 100, log=True)
    plt.xlabel('Pressure (mbar)'   , fontsize = 30)
    plt.ylabel('Probability / mbar', fontsize = 30)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.savefig('{:s}/Histogram_logpressure{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plot_hist(dfff, name_pressure, 1.,'Atmospheric Pressure (hPa)','Probability / hPa', ' hPa',xoff=0., coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_pressure{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plot_hist(dfff, name_pressure, 1.,'Atmospheric Pressure (hPa)','Probability / hPa', ' hPa',xoff=0., is_night=True, coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_pressure_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plot_mensual(dfff, name_pressure, 'Atmospheric pressure (hPa)')
    plt.savefig('{:s}/Pressure_mensual{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff_h, name_pressure, coverage_h)
    plt.ylabel('Atmospheric pressure (hPa)')
    #plt.xlabel('Year')
    #plot_historic_fit_results(dfn,mask,press_lik_median2,is_daily=True, day_coverage=day_coverage_for_pressure,color='red',is_sigma2=False)
    #plot_historic_fit_results(dfn,mask,press_lik_median4,is_daily=True, day_coverage=day_coverage_for_pressure,color='red',is_sigma2=True)
    plot_historic_fit_results(dfn,mask1,press_lik_median2_offset,is_daily=True, day_coverage=day_coverage_for_pressure,
                              color='red',is_sigma2=False,is_offset=True)
    plot_historic_fit_results(dfn,mask2,press_lik_median2_offset,is_daily=True, day_coverage=day_coverage_for_pressure,
                              color='red',is_sigma2=False,is_offset=True,offset=press_lik_median2_offset.res.x[-1])
    #plot_historic_fit_results(dfn,mask1,press_lik_median41,is_daily=True, day_coverage=day_coverage_for_pressure,color='tomato',is_sigma2=True)
    #plot_historic_fit_results(dfn,mask2,press_lik_median42,is_daily=True, day_coverage=day_coverage_for_pressure,color='seagreen',is_sigma2=True)            
    plt.savefig('{:s}/Pressure_sencer{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    dfff['Pdiff1'] = dfff[name_pressure].diff(1)

    num = 30
    grad_thr = 0.3

    print ('PGRADIENT1 TD: ',dfff.loc[ dfff['Pgradient1R'] > grad_thr, 'diff1'].head(n=num))
    #print ('TGRADIENT5 TE: ',dfff.loc[ dfff['Tgradient5'] > grad_thr, 'temperature'].head(n=num))
    #print ('TGRADIENT5 TED: ',dfff.loc[ dfff['Tgradient5'] > grad_thr, 'Tgradient5'].head(n=num))
    print ('PGRADIENT1 TED: ',dfff.loc[ dfff['Pgradient1R'] > grad_thr, 'Pgradient1R'].head(n=num))
    print ('PDIFF1: ', dfff.loc[ dfff['Pgradient1R'] > grad_thr,'Pdiff1'].head(n=num))
    #print ('TGRADIENT5R TED: ',dfff.loc[ dfff['Tgradient5'] > grad_thr, 'Tgradient5R'].head(n=num))
    #print ('TGRADIENT1R TED: ',dfff.loc[ dfff['Tgradient5'] > grad_thr, 'Tgradient1R'].head(n=num))
 
    grad_thr = -0.3

    print ('PGRADIENT1 TD: ',dfff.loc[ dfff['Pgradient1R'] < grad_thr, 'diff1'].head(n=num))
    #print ('TGRADIENT5 TE: ',dfff.loc[ dfff['Tgradient5'] < grad_thr, 'temperature'].head(n=num))
    #print ('TGRADIENT5 TED: ',dfff.loc[ dfff['Tgradient5'] < grad_thr, 'Tgradient5'].head(n=num))
    print ('PGRADIENT1 TED: ',dfff.loc[ dfff['Pgradient1R'] < grad_thr, 'Pgradient1R'].head(n=num))
    print ('PDIFF1: ', dfff.loc[ dfff['Pgradient1R'] < grad_thr,'Pdiff1'].head(n=num))
    #print ('TGRADIENT5R TED: ',dfff.loc[ dfff['Tgradient5'] < grad_thr, 'Tgradient5R'].head(n=num))
    #print ('TGRADIENT1R TED: ',dfff.loc[ dfff['Tgradient5'] < grad_thr, 'Tgradient1R'].head(n=num))
 


    for i in np.arange(1,6):
    
        var = 'Pgradient' + str(i) + 'R'

        #Distribucions anuals i mensuals:
        plt.clf()
        plot_hist(dfff, var, 0.01,'Pressure change rate (hPa/min)','Probability / (hPa/min)', 'hPa/min', xoff=0.08, coverage_cut=coverage_cut)
        #plt.plot(x,y, color= 'steelblue')
        plt.savefig(resultdir+'/Hist_'+var+'.pdf', bbox_inches='tight')
        plt.show()

        plt.clf()
        plot_hist(dfff, var, 0.01,'Pressure change rate (hPa/min)','Probability / (hPa/min)', 'hPa/min', xoff=0.08, is_night=True, coverage_cut=coverage_cut)
        #plt.plot(x,y, color= 'steelblue')
        plt.savefig(resultdir+'/Hist_'+var+'_night.pdf', bbox_inches='tight')
        plt.show()
        
        plt.clf()
        plot_mensual(dfff, var, 'Pressure change rate (hPa/min)')
        plt.savefig(resultdir+'/'+var+'_mensual.pdf', bbox_inches='tight')
        plt.show()
    
        plt.figure(figsize = (10,5), constrained_layout = True)
        plot_historic(dfff, var, coverage)
        #plt.xlabel('Year')
        plt.ylabel('Pressure change rate (hPa/min)')
        plt.savefig(resultdir+'/'+var+'_sencer.pdf', bbox_inches='tight')
        plt.show()

    dffff = Filtre_Temperatures(dfff)

    maskTvsP = (((dffff['Pgradient1R'] > 0.01) | (dffff['Pgradient1R'] < -0.01)) & ((dffff['Tgradient1R'] > 0.01) | (dffff['Tgradient1R'] < -0.01)))
    maskStorm = (((dffff['windSpeedAverage'] > 25.)) & maskTvsP)
    maskNoStorm = (((dffff['windSpeedAverage'] < 20.)) & maskTvsP)

    plt.clf()
    plt.plot(dffff.loc[maskTvsP,'Pgradient1R'], dffff.loc[maskTvsP,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Pressure change rate (hPa/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient_vs_Pgradient{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskStorm,'Pgradient1R'], dffff.loc[maskStorm,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Pressure change rate (hPa/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient1_vs_Pgradient1_storm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskNoStorm,'Pgradient1R'], dffff.loc[maskNoStorm,'Tgradient1R'],'ro',linestyle='None')
    plt.xlabel('Pressure change rate (hPa/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient1_vs_Pgradient1_nostorm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    maskTvsP = (((dffff['Pgradient5R'] > 0.01) | (dffff['Pgradient5R'] < -0.01)) & ((dffff['Tgradient5R'] > 0.01) | (dffff['Tgradient5R'] < -0.01)))
    maskStorm = (((dffff['windSpeedAverage'] > 25.)) & maskTvsP)
    maskNoStorm = (((dffff['windSpeedAverage'] < 20.)) & maskTvsP)

    plt.clf()
    plt.plot(dffff.loc[maskTvsP,'Pgradient5R'], dffff.loc[maskTvsP,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Pressure change rate (hPa/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Pgradient5{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskStorm,'Pgradient5R'], dffff.loc[maskStorm,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Pressure change rate (hPa/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Pgradient5_storm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.plot(dffff.loc[maskNoStorm,'Pgradient5R'], dffff.loc[maskNoStorm,'Tgradient5R'],'ro',linestyle='None')
    plt.xlabel('Pressure change rate (hPa/min)')
    plt.ylabel('Temperature change rate (ºC/min)')
    plt.savefig('{:s}/Tgradient5_vs_Pgradient5_nostorm{:s}.png'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    
def plot_wind() -> None:

    is_offset = True
    tits = '_2024'
    
    print ('wind binning: ', wind_binning)
    
    dfff = dff[(dff['wind_reliable']==True)]

    mask_ti = ((dfff[name_ws_current] > 0.) & (dfff['windTI']>1) & (dfff[name_ws_gust] > 0))
    pd.set_option('display.max_columns', 50)    
    print (dfff[mask_ti].head(n=100))
    
    #dfff[name_ws_current].rolling('10T',center=True,min_periods=1).agg(calc_ti)
    
    #pd.set_option('display.max_columns', 30)    
    #print('HERE', dfff[dfff.index > '2005-11-01'].head(n=20))
        
    dfff, coverage = apply_coverage(dfff,debug=False)
    dfn = dfff[(dfff['coverage'] > coverage_cut_for_daily_samples)]
    dfn_H = dfff[(dfff['coverage'] > coverage_cut_for_monthly_samples)]

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Wind Direction Average (deg.)')
    plt.hist(dfff[name_wdir_average])
    plt.savefig('{:s}/WindDirectionAverage{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    day_coverage_for_windSpeedAverage = 70
    alisio_limit = 50
    
    mask = (dfn[name_ws_average]<alisio_limit)  # select only Alisio winds
    mask_H = (dfn[name_ws_average]<alisio_limit)  # select only Alisio winds
    mask1 = ((dfn.index > new_WS) & ((dfn.index < new_model) | (dfn.index > old_model))  & (dfn[name_ws_average]<alisio_limit))
    mask2 = ((dfn.index > new_model) & (dfn.index < old_model) & (dfn[name_ws_average]<alisio_limit))    

    init_mu = np.array([11.8, 0., 1., 0.4, 4.3])    
    wind_lik_mean   = Likelihood_Wrapper(loglike,mu,name_mu,init_mu,bounds_mu,'Eq. (2), daily means')
    wind_lik_mean.setargs_df(dfn,name_ws_average,mask,
                             is_daily=True,is_median=False,day_coverage=day_coverage_for_windSpeedAverage)
    wind_lik_mean.like_minimize(method=method,tol=tol)

    init_mu_offset = np.array([11.8, 0., 1., 0.4, 4.3, 0.])    
    wind_lik_mean_offset = Likelihood_Wrapper(loglike_2sets,mu,
                                              name_mu_offset,init_mu_offset,bounds_mu_offset,
                                              'Eq. (2), daily means w/ offset')
    wind_lik_mean_offset.setargs_df2(dfn,name_ws_average,mask1,mask2,
                                     is_daily=True,is_median=True,day_coverage=day_coverage_for_windSpeedAverage)
    wind_lik_mean_offset.like_minimize(method=method,tol=tol)

    
    init_mu2 = np.array([11.6, 0.,1.3, 11.1, 1.1, 11.8, 4.5])
    wind_lik_mean2   = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily means')
    wind_lik_mean2.setargs_df(dfn,name_ws_average,mask,
                              is_daily=True,is_median=False,day_coverage=day_coverage_for_windSpeedAverage)
    wind_lik_mean2.like_minimize(method=method,tol=tol)

    init_mu2_offset = np.array([11.6, 0.,1.3, 11.1, 1.1, 11.8, 4.5, 0.])    
    wind_lik_median2_offset   = Likelihood_Wrapper(loglike2_2sets,mu2,name_mu2_offset,init_mu2_offset,bounds_mu2_offset,'Daily medians, w/ offset')
    wind_lik_median2_offset.setargs_df2(dfn,name_ws_average,mask1,mask2,
                                is_daily=True,is_median=True,day_coverage=day_coverage_for_windSpeedAverage)
    wind_lik_median2_offset.like_minimize(method=method,tol=tol)

    wind_lik_median2   = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_mu2,bounds_mu2,'Daily medians')
    wind_lik_median2.setargs_df(dfn,name_ws_average,mask,
                                is_daily=True,is_median=True,day_coverage=day_coverage_for_windSpeedAverage)
    wind_lik_median2.like_minimize(method=method,tol=tol)

    init_mu4 = np.array([12.5, 0.,1.1, 11.4, 1.1, 11.3, 5.4, 1.2, 9.3])
    wind_lik_median4   = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_mu4,bounds_mu2_sig2,
                                            'Daily medians, seas. spreads')
    wind_lik_median4.setargs_df(dfn,name_ws_average,mask,
                                is_daily=True,is_median=True,day_coverage=day_coverage_for_windSpeedAverage)
    wind_lik_median4.like_minimize(method=method,tol=tol)

    init_month = np.array([12, 0.,1.2, 11.1, 1., 11.8, 1.2])
    wind_lik_month  = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_month,bounds_mu2,
                                         'Monthly means')
    wind_lik_month.setargs_df(dfn_H,name_ws_average,mask_H,
                             is_daily=False,is_median=False)
    wind_lik_month.like_minimize(method=method,tol=tol)
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Wind Speed Average (km/h)')
    wind_lik_mean.plot_residuals()
    plt.savefig('{:s}/WindSpeed_residuals{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.xlabel('Year')
    plt.ylabel('Wind Speed Average (km/h)')
    wind_lik_mean2.plot_residuals()
    plt.savefig('{:s}/WindSpeed_residuals2{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    residuals_mean = wind_lik_mean.red_residuals()
    chi2_measured_mean = wind_lik_mean.chi_square_ndf()
    residuals_mean2 = wind_lik_mean2.red_residuals()
    chi2_measured_mean2 = wind_lik_mean2.chi_square_ndf()
    residuals_median2 = wind_lik_median2.red_residuals()
    chi2_measured_median2 = wind_lik_median2.chi_square_ndf()
    residuals_median2_offset = wind_lik_median2_offset.red_residuals()
    chi2_measured_median2_offset = wind_lik_median2_offset.chi_square_ndf()
    residuals_median4 = wind_lik_median4.red_residuals(is_sigma2=True)
    chi2_measured_median4 = wind_lik_median4.chi_square_ndf(is_sigma2=True)
    residuals_month = wind_lik_month.red_residuals()
    chi2_measured_month = wind_lik_month.chi_square_ndf()

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median2,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median2**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median2),fontsize=20)
    a[1].legend(loc='best')    
    plt.savefig('{:s}/WindSpeed_residuals2_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    plt.figure()
    fig,a = plt.subplots(1,2)
    a = a.ravel()
    a[0].hist(residuals_median4,bins=40,label='mean')
    a[0].set_yscale('log')
    a[0].legend(loc='best')
    a[1].hist(residuals_median4**2,bins=40,label='mean')
    a[1].set_yscale('log')
    a[1].text(6,1000.,r'$\chi^{2}/NDF$='+'{:.2f}'.format(chi2_measured_median4),fontsize=20)
    a[1].legend(loc='best')    
    plt.savefig('{:s}/WindSpeed_residuals4_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')

    #plt.figure()
    #naoi_correlate(df_naoi,wind_lik_mean.full_residuals(),color='r')
    #naoi_profile(df_naoi,wind_lik_mean.full_residuals(),nbins=12)    
    #plt.savefig('{:s}/WindSpeed_corr_d_NAOI{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')


    # Profile the wind increase parameter now
    plt.figure()
    plt.xlabel(r'$b$ (average wind speed increase) ((km/h)/10y)', fontsize = 25)
    plt.ylabel(r'$D(b)$', fontsize = 25)

    NN   = 28
    chi2 = 25
    #wind_lik_mean.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g')
    wind_lik_mean2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='b')        
    wind_lik_median2.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='g')
    wind_lik_month.profile_likelihood(1,chi2=23,NN=NN,method=method,col='orange')
    plt.plot([], [], ' ', label="Robustness tests:")        
    wind_lik_median4.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='r',alpha=0.2)
    if (is_offset):
        wind_lik_median2_offset.profile_likelihood(1,chi2=chi2,NN=NN,method=method,col='cyan',alpha=0.2)
        tits = tits + '_offset'                
    plt.savefig('{:s}/WindSpeed_b_profiled{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    tits = tits.replace('_offset','')    
    
    #Distribucions del vent
    plt.clf()
    plot_hist(dfff, name_ws_current, 1.,'Instantaneous wind speed (km/h)','Probability / (km/h)', ' km/h', xoff=0., loc='right', coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_windSpeed.pdf{:s}'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plot_hist(dfff, name_ws_current, 1.,'Instantaneous wind speed (km/h)','Probability / (km/h)', ' km/h', xoff=0.,loc='right',is_night=True, coverage_cut=coverage_cut)
    #plt.plot(x,y, color= 'steelblue')
    plt.savefig('{:s}/Hist_windSpeed_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.hist(dfff[name_ws_current], bins = 100, log = 'True')
    plt.xlabel('Wind Speed (km/h)')
    plt.savefig('{:s}/Histogram_windSpeed{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.hist(dfff[name_ws_average], bins = 100)
    plt.xlabel('Wind Speed Average (km/h)')
    plt.savefig('{:s}/Histogram_windAverage{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
        
    plt.clf()
    plt.plot(dfff.index, dfff[name_ws_average])
    plt.xlabel('Wind Speed Average (km/h)')
    plt.savefig('{:s}/Histogram_windAverage_vsTime{:s}.png'.format(tits), bbox_inches='tight')
    plt.show()
        
    plt.figure(figsize = (10,5), constrained_layout = True)
    v10, v50, v475 = plot_windspeed(dfff,name_ws_average,name_ws_gust, unit='km/h',wbins=wind_binning, sampling_freq_min=1./data_spacing_minutes)
    plt.ylim([1e-4,5e4])
    plt.xlim([-0.5,v475*1.05])
    plt.vlines(v10,1e-4,1e-2,colors='darkturquoise')
    plt.vlines(v50,1e-4,2e-3,colors='cadetblue')
    plt.vlines(v475,1e-4,2e-4,colors='darkslategrey')
    plt.hlines(1e-2,v10,v10+10,colors='darkturquoise')
    plt.hlines(2e-3,v50,v50+10,colors='cadetblue')
    plt.hlines(2e-4,v475,v475+10,colors='darkslategrey')
    plt.text(v10,2e-2,'1/10y',fontsize=15,color='darkturquoise')
    plt.text(v50,5e-3,'1/50y',fontsize=15,color='cadetblue')
    plt.text(v475,5e-4,'1/475y',fontsize=15,color='darkslategrey')
    plt.legend(loc='best', fontsize=18)
    plt.savefig('Histogram_windGust.pdf{:s}'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure()    
    plot_mensual_wind(dfff,fullfits=False) #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/WindSpeed_mensual.pdf{:s}'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    mask_ti = ((dfff[name_ws_current] > 0.) & (dfff.index > WS_relocation))
    ti_fit_idx = 9
    ti_fit_il  = -10
    h,p = plot_profile(dfff.loc[mask_ti,name_ws_average],dfff.loc[mask_ti,'windTI'],nbins=50)
    popt, pcov = curve_fit(lambda x,*p: p[0]+p[1]*x, h[ti_fit_idx:ti_fit_il], p[ti_fit_idx:ti_fit_il], p0=[0.2,0.2/80])
    plt.plot(h[ti_fit_idx:ti_fit_il],popt[0] + popt[1]*h[ti_fit_idx:ti_fit_il],'-',lw=2,label=r'TI=%.3f+%.4f$\cdot$(wind speed)' % tuple(popt),color='r')
    plt.xlabel('Wind Speed Average (km/h)')
    plt.ylabel('Turbulence Index',fontsize=22)
    plt.legend(loc='best')
    plt.savefig('{:s}/WindTI_corr_WindSpeed{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    mask_ti_s = ((dfff[name_ws_current] > 0.) & (dfff.index > WS_relocation) & (dfff[name_ws_average]>20.))   
    h,p = plot_profile(dfff.loc[mask_ti_s,name_wdir_average],dfff.loc[mask_ti_s,'windTI'].add(-1.*popt[0]).add(-1*popt[1]*dfff.loc[mask_ti_s,name_ws_average]),nbins=30,  markercolor='b', label='Wind speed average > 20 km/h')
    mask_ti_s = ((dfff[name_ws_current] > 0.) & (dfff.index > WS_relocation) & (dfff[name_ws_average]>40.))   
    h,p = plot_profile(dfff.loc[mask_ti_s,name_wdir_average],dfff.loc[mask_ti_s,'windTI'].add(-1.*popt[0]).add(-1*popt[1]*dfff.loc[mask_ti_s,name_ws_average]),nbins=30, max_color='orange', markercolor='tomato', facecolor='C1',label='Wind speed average > 40 km/h')
    mask_ti_s = ((dfff[name_ws_current] > 0.) & (dfff.index > WS_relocation) & (dfff[name_ws_average]>60.))      
    h,p = plot_profile(dfff.loc[mask_ti_s,name_wdir_average],dfff.loc[mask_ti_s,'windTI'].add(-1.*popt[0]).add(-1*popt[1]*dfff.loc[mask_ti_s,name_ws_average]),nbins=30, max_color='mediumorchid', markercolor='darkviolet', facecolor='m',label='Wind speed average > 60 km/h')
    #popt, pcov = curve_fit(lambda x,*p: p[0]+p[1]*x, h[ti_fit_idx:], p[ti_fit_idx:], p0=[0.2,0.2/80])
    #plt.plot(h[10:],popt[0] + popt[1]*h[ti_fit_idx:],'-',lw=2,label=r'TI=%.3f+%.4f$\cdot$(wind speed)' % tuple(popt),color='r')
    # MAGIC 1
    plt.gca().add_patch(patches.Rectangle((76,-0.25),93-76, 0.5, linewidth=1, edgecolor='forestgreen', facecolor='forestgreen',alpha=0.3))
    plt.text(76-10,0.26,'MAGIC-I',fontsize=18,color='forestgreen')    
    plt.gca().add_patch(patches.Rectangle((162,-0.25),178-162, 0.5, linewidth=1, edgecolor='forestgreen', facecolor='forestgreen',alpha=0.3))
    plt.text(162-15,0.26,'MAGIC-II',fontsize=18,color='forestgreen')        
    plt.gca().add_patch(patches.Rectangle((198,-0.25),224-198, 0.5, linewidth=1, edgecolor='forestgreen', facecolor='forestgreen',alpha=0.3))
    plt.text(198-1,0.26,'LIDAR',fontsize=18,color='forestgreen')            
    plt.gca().add_patch(patches.Rectangle((238,-0.25),259-238, 0.5, linewidth=1, edgecolor='forestgreen', facecolor='forestgreen',alpha=0.3))
    #plt.gca().add_patch(patches.Rectangle((238,-0.25),247-238, 0.5, linewidth=1, edgecolor='forestgreen', facecolor='forestgreen',alpha=0.3))
    plt.text(238-2,0.26,'LST-1',fontsize=18,color='forestgreen')                
    #    plt.gca().add_patch(patches.Rectangle((259,-0.25),262-259, 0.5, linewidth=1, edgecolor='forestgreen', facecolor='forestgreen',alpha=0.3))
    #    plt.text(259-20,0.35,'Camera Tower',fontsize=18,color='forestgreen')                
    plt.xlabel('Wind Direction Average (deg.)')
    plt.ylabel('Normalized TI',fontsize=22)
    plt.ylim([-0.25,1.5])
    plt.xlim([0.,360.])
    plt.legend(loc='best')
    plt.savefig('{:s}/WindTInorm_corr_WindDir{:s}.pdf'.format(resultdir,tits),bbox_inches='tight')
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    mask_ti = ((dfff[name_ws_current] > 0.) & (dfff.index > WS_relocation))
    h,p = plot_profile(dfff.loc[mask_ti,'windTI'],dfff.loc[mask_ti,name_wdir_current],nbins=50)
    plt.xlabel('Turbulence Index',fontsize=22)
    plt.ylabel('Wind Direction',fontsize=22)
    plt.savefig('{:s}/WindTI_corr_WindDir{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    mask_ti = ((dfff[name_ws_current] > 0.) & (dfff.index > WS_relocation))
    h,p = plot_profile(dfff.loc[mask_ti,name_ws_average],dfff.loc[mask_ti,name_wdir_current],nbins=50)
    plt.xlabel('Wind Speed Average (km/h)')
    plt.ylabel('Wind Direction',fontsize=22)
    plt.savefig('{:s}/WindDir_corr_WindSpeed{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    
    plt.figure()    
    plot_mensual(dfff[mask_ti],'windTI',ytit='Turbulence Index') #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/WindTI_mensual{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic(dfff[mask_ti], 'windTI', coverage)
    #plt.xlabel('Year')
    plt.ylabel('Turbulence Index')
    plt.savefig('{:s}/WindTI_sencer{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plot_historic_wind(dfff, coverage, min_coverage=30) #, name_ws_current)
    if is_offset:
        plot_historic_fit_results(dfn,mask1,wind_lik_mean_offset,is_daily=True, day_coverage=day_coverage_for_windSpeedAverage,
                                  color='red',is_sigma2=False,is_offset=True)
        plot_historic_fit_results(dfn,mask2,wind_lik_mean_offset,is_daily=True, day_coverage=day_coverage_for_windSpeedAverage,
                                  color='orange',is_sigma2=False,is_offset=True,offset=wind_lik_mean_offset.res.x[-1])
        tits = tits + '_offset'        
    else:
        plot_historic_fit_results(dfn,mask,wind_lik_median4,is_daily=True, day_coverage=day_coverage_for_windSpeedAverage,color='red',is_sigma2=True)    
    #plt.xlabel('Year')
    plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/WindSpeed_sencer{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    tits = tits.replace('_offset','')    

    #Roses de vent
    dfn = dfff[dfff['coverage'] > coverage_cut]
    alisio_limit = 70
    pw_ti = 2
    pw_full = 5
    pw_strong = 0.7
    pw_ti_strong = 2

    plt.figure()
    mask_ti = ((dfn[name_ws_current] > 0.) & (dfn.index > WS_relocation))
    plot_windrose(dfn[mask_ti],'windTI',name_wdir_current,
                  'Turbulence Index, full sample',pw=pw_ti,leg_tit='TI',form='.3f',min_scale=0.)
    plt.savefig('{:s}/windrose_TI_normal{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure()
    mask_ws = ((dfn[name_ws_current] > 0.) & (dfn.index > WS_relocation))    
    plot_windrose(dfn[mask_ws],name_ws_current,name_wdir_current,
                  'Instantaneous wind speed, full sample',pw=pw_full)
    plt.savefig('{:s}/windrose_normal{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()


    plt.clf()
    df_fort = dfn[dfn[name_ws_current] > alisio_limit]
    plot_windrose(df_fort,name_ws_current,name_wdir_current,
                  'Instantaneous wind speed > {:d} km/h'.format(alisio_limit), pw=pw_strong)
    plt.savefig('{:s}/windrose_extrem{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()


    plt.clf()
    plot_windrose(df_fort,'windTI',name_wdir_current,
                  'Instantaneous wind speed > {:d} km/h'.format(alisio_limit), pw=pw_ti_strong,
                  leg_tit='TI',form='.3f',min_scale=0.1)
    plt.savefig('{:s}/windrose_TI_extrem{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    dfn = dfff[((dfff['coverage'] > coverage_cut) & (dfff['sun_alt'] < -12.) & (dfff[name_ws_current] > 0.))]

    plt.figure()
    mask_ti = ((dfn[name_ws_current] > 0.)  & (dfn.index > WS_relocation))
    plot_windrose(dfn[mask_ti],'windTI',name_wdir_current,
                  'Turbulence Index, full sample',pw=pw_ti,leg_tit='TI',form='.3f',min_scale=0.)
    plt.savefig('{:s}/windrose_TI_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plot_windrose(dfn,name_ws_current,name_wdir_current,
                  'Instantaneous wind speed, night time only',pw=pw_full)
    plt.savefig('{:s}/windrose_normal_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    ws_min = 70
    df_fort = dfn[dfn[name_ws_current] > ws_min]
    plot_windrose(df_fort,name_ws_current,name_wdir_current,
                  'Instantaneous wind speed > {:d} km/h, night time only'.format(ws_min),pw=pw_strong)
    plt.savefig('{:s}/windrose_extrem_night{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()


def plot_huracans() -> None:

    dfff = Filtre_Wind(dff)

    year_start = 2004
    year_end   = 2023
    years = np.arange(year_start,year_end+1)

    for y in years:
        print ('Year: ', dfff.loc[(dfff['Year']==y) & (dfff['diff1']>pd.Timedelta(1,'day')),'diff1'].head(n=200))

    gaps   = []
    Storms = []
    months = range(1,13)
    for m in months:
        gaps.append(count_gaps_month(dfff,m,year_start,year_end))

    plt.clf()
    plt.plot(months, gaps, marker = 'o', linestyle = 'none', color = 'steelblue', markersize = 8,
             markerfacecolor = 'k', markeredgecolor = 'k')
    #plt.xlabel('Month', fontsize=20)
    plt.ylabel('Data gaps (days)',    fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)    
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(np.arange(1,13), months_n,ha='center')
    plt.savefig('{:s}/Gaps_days.pdf')
    plt.show()

    plt.clf()
    test_cuts = np.arange(50,110,10)
    for test_cut in test_cuts:
        Storms = []
        for m in months:
            Storms.append(extremes_month(dfff,name_ws_gust, test_cut, m, year_start, year_end))
        plt.plot(months, np.array(Storms), marker = 'o', linestyle = 'none', markersize = 10, markeredgecolor = 'k', 
                 label='threshold={:d} km/h'.format(resultdir,test_cut))  # markerfacecolor = 'white', 
    #plt.xlabel('Month', fontsize=20)
    plt.ylabel('Number of storms exceeding threshold', fontsize=20)
    plt.legend(loc='best', fontsize=18)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(np.arange(1,13), months_n,ha='center')
    plt.savefig('{:s}/Mensual_storms_all{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    test_cuts = np.arange(60,100,10)
    for test_cut in test_cuts:
        weights = calc_all_weights(dfff,name_ws_gust, test_cut, year_start, year_end)
        print ('years', years, ' Weights', weights)
        plt.plot(years, weights, marker = 'o', linestyle = 'none', markersize = 7, #markeredgecolor = 'k', 
                 label='threshold={:d} (km/h)'.format(test_cut))  # markerfacecolor = 'white', 
    #plt.xlabel('Month', fontsize=20)
    plt.ylabel('Weights due to data gaps', fontsize=18)
    plt.legend(loc='best', fontsize=18)
    plt.ylim(-0.05,1.05)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=15)
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(years, years,ha='center')
    plt.savefig('{:s}/Yearly_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    test_cut = 115
    k = np.array(extremes(dfff, name_ws_gust, test_cut, year_start, year_end))
    plot_profile_likelihood(k)
    plt.title('Wind Gust threshold: {:d} km/h'.format(test_cut), fontsize=20)
    plt.savefig('{:s}/likelihood_cut_{:d}{:s}.pdf'.format(resultdir,test_cut,tits), bbox_inches='tight')
    plt.show()           

    weights_for_p0 = np.array(calc_all_weights(dfff,name_ws_gust,test_cut, year_start, year_end))

    plot_profile_likelihood(k,weights_for_p0)
    plt.title('Wind Gust threshold: {:d} km/h'.format(test_cut), fontsize=20)
    plt.savefig('{:s}/likelihood_cut_{:d}_weights{:s}.pdf'.format(resultdir,test_cut,tits), bbox_inches='tight')
    plt.show()           
    
    test_cut = 100
    k = np.array(extremes(dfff, name_ws_gust, test_cut, year_start, year_end))
    plot_profile_likelihood(k)
    plt.title('Wind Gust threshold: {:d} km/h'.format(test_cut), fontsize=20)
    plt.savefig('{:s}/likelihood_cut_{:d}{:s}.pdf'.format(resultdir,test_cut,tits), bbox_inches='tight')
    plt.show()           

    weights_for_p0 = np.array(calc_all_weights(dfff,name_ws_gust,test_cut, year_start, year_end))
    
    plot_profile_likelihood(k,weights_for_p0)
    plt.title('Wind Gust threshold: {:d} km/h'.format(test_cut), fontsize=20)
    plt.savefig('{:s}/likelihood_cut_{:d}_weights{:s}.pdf'.format(resultdir,test_cut,tits), bbox_inches='tight')
    plt.show()           

    print ('FINSIHED LIKELIHOOD')


    for cut in np.arange(50,110,10):
        weights = np.array(calc_all_weights(dfff,name_ws_gust,cut, year_start, year_end))        
        k = extremes(dfff, name_ws_gust, cut, year_start, year_end)
        plot_extremes(k, year_start, year_end,weights)
        plt.ylabel('Number of storms exceeding {:d} km/h'.format(cut), fontsize = 20)
        plt.savefig('{:s}/Huracans_cut_{:d}{:s}.pdf'.format(resultdir,cut,tits), bbox_inches='tight')
        plt.show()                   
            
    TSs, p0s, alphas, alphas_sup1, alphas_inf1, alphas_sup2, alphas_inf2 = [], [], [], [], [], [], []
    cuts = range(50,115,5)

    for cut in cuts:

        weights = np.array(calc_all_weights(dfff,name_ws_gust,cut, year_start, year_end))                
        
        k = np.array(extremes(dfff, name_ws_gust, cut, year_start, year_end))
        ts, p0, alpha = TS(k,weights)
        TSs.append(ts)
        p0s.append(p0)
        alphas.append(alpha)
        conf_int1 = confidence_intervals(k, chi2=1, weights=weights)
        alphas_sup1.append(conf_int1[0]) 
        alphas_inf1.append(conf_int1[1])

        conf_int2 = confidence_intervals(k, chi2=4, weights=weights)
        alphas_sup2.append(conf_int2[0]) 
        alphas_inf2.append(conf_int2[1])

    plt.clf()
    plt.scatter(cuts, alphas, s = 20)
    plt.fill_between(cuts, alphas_sup2, alphas_inf2, color='seagreen', alpha=0.3)
    plt.fill_between(cuts, alphas_sup1, alphas_inf1, color='tomato', alpha=0.3)
    plt.ylim(-0.6,0.3)
    plt.xlabel('Minimum wind gust (km/h)', fontsize = 26)
    plt.ylabel(r'Yearly increase of storms', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/alphas{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.scatter(cuts, np.array(alphas)/np.array(p0s), s = 20)
    plt.fill_between(cuts, np.array(alphas_sup2)/np.array(p0s), np.array(alphas_inf2)/np.array(p0s), color='seagreen', alpha=0.3)
    plt.fill_between(cuts, np.array(alphas_sup1)/np.array(p0s), np.array(alphas_inf1)/np.array(p0s), color='tomato', alpha=0.3)
    #plt.ylim(-0.3,0.65)
    plt.xlabel('Minimum wind gust (km/h)', fontsize = 26)
    plt.ylabel(r'Yearly occurrence probability increase', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/alphas{:s}_rel..format(resultdir,tits)pdf', bbox_inches='tight')
    plt.show()
    
    
    plt.clf()
    plt.plot(cuts, TSs, '.' , markersize=20)
    plt.ylabel(r'$\sqrt{TS}$', fontsize = 26)
    plt.xlabel('Minimum wind gust (km/h)', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/TSs{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.plot(cuts, p0s, '.', markersize=20, color='gold')
    plt.xlabel('Minimum wind gust (km/h)', fontsize = 26)
    plt.ylabel(r'Mean yearly occurrence', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/p0s{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
            
    TSs, p0s, alphas, alphas_sup1, alphas_inf1, alphas_sup2, alphas_inf2 = [], [], [], [], [], [], []

    for cut in cuts:
        k = np.array(extremes(dfff, name_ws_gust, cut, year_start, year_end))
        ts, p0, alpha = TS(k,weights_for_p0)
        TSs.append(ts)
        p0s.append(p0)
        alphas.append(alpha)
        conf_int1 = confidence_intervals(k, chi2=1, weights=weights_for_p0)
        alphas_sup1.append(conf_int1[0]) 
        alphas_inf1.append(conf_int1[1])

        conf_int2 = confidence_intervals(k, chi2=4, weights=weights_for_p0)
        alphas_sup2.append(conf_int2[0]) 
        alphas_inf2.append(conf_int2[1])

    plt.clf()
    plt.scatter(cuts, alphas, s = 20)
    plt.fill_between(cuts, alphas_sup2, alphas_inf2, color='seagreen', alpha=0.3)
    plt.fill_between(cuts, alphas_sup1, alphas_inf1, color='tomato', alpha=0.3)
    plt.ylim(-0.6,0.3)
    plt.xlabel('Threshold wind gust (km/h)', fontsize = 26)
    plt.ylabel(r'Yearly increase of events', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/alphas_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.scatter(cuts, np.array(alphas)/np.array(p0s), s = 20)
    plt.fill_between(cuts, np.array(alphas_sup2)/np.array(p0s), np.array(alphas_inf2)/np.array(p0s), color='seagreen', alpha=0.3)
    plt.fill_between(cuts, np.array(alphas_sup1)/np.array(p0s), np.array(alphas_inf1)/np.array(p0s), color='tomato', alpha=0.3)
    #plt.ylim(-0.3,0.65)
    plt.xlabel('Threshold wind gust (km/h)', fontsize = 26)
    plt.ylabel(r'Yearly occurrence probability increase', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/alphas_rel_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    
    plt.clf()
    plt.plot(cuts, TSs, '.' , markersize=20)
    plt.ylabel(r'$\sqrt{TS}$', fontsize = 26)
    plt.xlabel('Threshold wind gust (km/h)', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/TSs_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.plot(cuts, p0s, '.', markersize=20, color='gold')
    plt.xlabel('Threshold wind gust (km/h)', fontsize = 26)
    plt.ylabel(r'Mean yearly occurrence', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/p0s_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
            

def plot_snow() -> None:

    #plt.rcParams.update(params)

    if (is_20to22):
        dfff = dff[(dff['humidity_reliable']==True) | (dff.index.year==2020) | (dff.index.year==2021) | (dff.index.year==2022) | (dff.index.year==2023) ]
        tits = '_with20to22'
    else:
        dfff = dff[(dff['humidity_reliable']==True)]
        tits = ''        

    dfff = dfff[dfff['humidity']>90] 
        
    plt.clf()
    plt.plot(dfff['WB'],dfff['WBD'])
    plt.savefig('{:s}/Histogram_WB2{:s}.png'.format(resultdir,tits))

    plt.clf()
    plt.hist(Tmin(dfff['humidity']))
    plt.savefig('{:s}/Histogram_Tmin{:s}.pdf'.format(resultdir,tits))

    plt.clf()
    plt.plot(dfff['temperature'],dfff['WB']-Tmin(dfff['humidity']),color='b')
    plt.plot(dfff['temperature'],dfff['WB']-Tmax(dfff['humidity']),color='r')
    plt.plot(dfff['temperature'],dfff['WBD']-Tmin(dfff['humidity']),color='indigo')
    plt.plot(dfff['temperature'],dfff['WBD']-Tmax(dfff['humidity']),color='orange')
    plt.savefig('{:s}/Histogram_WB-Tmin{:s}.png'.format(resultdir,tits))

    plt.clf()
    plt.plot(dfff['temperature'],dfff['WB'], color='g')
    plt.plot(dfff['temperature'],Tmin(dfff['humidity']), color='b')
    plt.plot(dfff['temperature'],Tmax(dfff['humidity']), color='r')
    plt.savefig('{:s}/Histogram_WB{:s}.png'.format(resultdir,tits))

        
    dfff, coverage = apply_coverage(dfff,debug=False)

    mask_snow  = (dfff['WBD']<=Tmin(dfff['humidity']))
    mask_sleet = ((dfff['WBD']>Tmin(dfff['humidity'])) & (dfff['WBD']<=Tmax(dfff['humidity'])))
    
    year_start = 2004
    year_end   = 2023
    years = np.arange(year_start,year_end+1)

    bins = 17
    min_h = 6/60.
    max_h = 260
    bin_edges = np.squeeze(np.logspace(np.log10(min_h),np.log10(max_h), num=bins+1, endpoint=True))
    #bin_edges = bin_edges[0]
    #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    length_tot_sn = []
    length_tot_sl = []
    
    plt.clf()
    fig,ax = plt.subplots(2,1, sharex=True)
    print ('COUNTING EXTREMES')
    
    for y in years:
        length_arr_sn = count_extremes_length(dfff[mask_snow], 'humidity',y, 90, False)
        if (len(length_arr_sn)>1):
            arr = np.array(length_arr_sn)
            ax[0].violinplot(arr[arr<3], positions=[y])
            ax[1].violinplot(arr[arr>=3], positions=[y])
        else:
            continue
        length_tot_sn.extend(length_arr_sn)

    length_arr_sn = np.squeeze(np.array(length_tot_sn))

    ax[0].set_ylabel('Precipitation duration (<3 hours)',fontsize=16)
    ax[1].set_ylabel('Precipitation duration (>3 hours)',fontsize=16)
    ax[0].yaxis.set_tick_params(labelsize=18)
    ax[1].yaxis.set_tick_params(labelsize=18)
    ax[0].xaxis.set_tick_params(labelsize=15)    
    plt.xticks(years, years,ha='center')
    plt.savefig('{:s}/Length_hours_snow{:s}.pdf'.format(resultdir,tits))
    plt.show()

    plt.clf()
    fig,ax = plt.subplots(2,1, sharex=True)
    print ('COUNTING EXTREMES')
    
    for y in years:
        length_arr_sl = count_extremes_length(dfff[mask_sleet], 'humidity',y, 90, False)
        if (len(length_arr_sl)>1):
            arr = np.array(length_arr_sl)
            ax[0].violinplot(arr[arr<3], positions=[y])
            ax[1].violinplot(arr[arr>=3], positions=[y])
        else:
            continue
        length_tot_sl.extend(length_arr_sl)

    length_arr_sl = np.squeeze(np.array(length_tot_sl))

    ax[0].set_ylabel('Precipitation duration (<3 hours)',fontsize=16)
    ax[1].set_ylabel('Precipitation duration (>3 hours)',fontsize=16)
    ax[0].yaxis.set_tick_params(labelsize=18)
    ax[1].yaxis.set_tick_params(labelsize=18)
    ax[0].xaxis.set_tick_params(labelsize=15)    
    plt.xticks(years, years,ha='center')
    plt.savefig('{:s}/Length_hours_sleet{:s}.pdf'.format(resultdir,tits))
    plt.show()

def plot_rainy_periods() -> None:

    #plt.rcParams.update(params)

    if (is_20to22):
        dfff = dff[(dff['humidity_reliable']==True) | (dff.index.year==2020) | (dff.index.year==2021) | (dff.index.year==2022) | (dff.index.year==2023) ]
        tits = '_with20to22'
    else:
        dfff = dff[(dff['humidity_reliable']==True)]
        tits = ''        
        
    dfff, coverage = apply_coverage(dfff,debug=False)
    
    mask_snow  = (dfff['WBD']<=Tmin(dfff['humidity']))
    mask_snow_cover = ((dfff['WBD']<=Tmin(dfff['humidity'])) & (dfff['temperature']<=0))
    mask_sleet = ((dfff['WBD']>Tmin(dfff['humidity'])) & (dfff['WBD']<=Tmax(dfff['humidity'])))
    mask_rain  = (dfff['WBD']>Tmax(dfff['humidity']))
    
    year_start = 2004
    year_end   = 2023
    years = np.arange(year_start,year_end+1)

    for y in years:
        print ('GAPS Year: ', dfff.loc[(dfff['Year']==y) & (dfff['diff1']>pd.Timedelta(1,'day')),'diff1'].head(n=200))

    bins = 17
    min_h = 6/60.
    max_h = 260
    bin_edges = np.squeeze(np.logspace(np.log10(min_h),np.log10(max_h), num=bins+1, endpoint=True))
    #bin_edges = bin_edges[0]
    #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    length_tot    = []
    length_tot_sn = []  # snow fall
    length_tot_sc = []  # snow cover
    length_tot_sl = []  # sleet
    length_tot_ra = []  # rain
    
    plt.clf()
    fig,ax = plt.subplots(2,1, sharex=True)
    print ('COUNTING EXTREMES')
    
    for y in years:
        length_arr = count_extremes_length(dfff,name_humidity,y, 90, False)
        if (len(length_arr)>1):
            arr = np.array(length_arr)
            ax[0].violinplot(arr[arr<3], positions=[y])
            ax[1].violinplot(arr[arr>=3], positions=[y])
        else:
            continue
        length_tot.extend(length_arr)
    length_arr = np.squeeze(np.array(length_tot))

    print ('TOTAL EXTREMES ALL: ',length_arr.sum())
    
    print ('COUNTING EXTREMES SNOW')
    
    for y in years:
        length_arr_sn = count_extremes_length(dfff[mask_snow], 'humidity',y, 90, False)
        if (len(length_arr_sn)>1):
            arr_sn = np.array(length_arr_sn)
            #ax[0].violinplot(arr[arr<3], positions=[y])
            #ax[1].violinplot(arr[arr>=3], positions=[y])
        else:
            continue
        length_tot_sn.extend(length_arr_sn)
    length_arr_sn = np.squeeze(np.array(length_tot_sn))

    print ('TOTAL EXTREMES SNOW: ',length_arr_sn.sum())
    
    for y in years:
        length_arr_sc = count_extremes_length(dfff[mask_snow_cover], 'humidity',y, 90, False)
        if (len(length_arr_sc)>1):
            arr_sc = np.array(length_arr_sc)
            #ax[0].violinplot(arr[arr<3], positions=[y])
            #ax[1].violinplot(arr[arr>=3], positions=[y])
        else:
            continue
        length_tot_sc.extend(length_arr_sc)
    length_arr_sc = np.squeeze(np.array(length_tot_sc))

    print ('TOTAL EXTREMES SNOW COVER: ',length_arr_sc.sum())
    
    for y in years:
        length_arr_sl = count_extremes_length(dfff[mask_sleet], 'humidity',y, 90, False)
        if (len(length_arr_sl)>1):
            arr_sl = np.array(length_arr_sl)
            #ax[0].violinplot(arr[arr<3], positions=[y])
            #ax[1].violinplot(arr[arr>=3], positions=[y])
        else:
            continue
        length_tot_sl.extend(length_arr_sl)
    length_arr_sl = np.squeeze(np.array(length_tot_sl))

    print ('TOTAL EXTREMES SLEET: ',length_arr_sl.sum())
    
    for y in years:
        length_arr_ra = count_extremes_length(dfff[mask_rain], 'humidity',y, 90, False)
        if (len(length_arr_ra)>1):
            arr_ra = np.array(length_arr_ra)
            #ax[0].violinplot(arr[arr<3], positions=[y])
            #ax[1].violinplot(arr[arr>=3], positions=[y])
        else:
            continue
        length_tot_ra.extend(length_arr_ra)
    length_arr_ra = np.squeeze(np.array(length_tot_ra))

    print ('TOTAL EXTREMES RAIN: ',length_arr_ra.sum())
    
    ax[0].set_ylabel('Precipitation duration (<3 hours)',fontsize=16)
    ax[1].set_ylabel('Precipitation duration (>3 hours)',fontsize=16)
    ax[0].yaxis.set_tick_params(labelsize=18)
    ax[1].yaxis.set_tick_params(labelsize=18)
    ax[0].xaxis.set_tick_params(labelsize=15)    
    plt.xticks(years, years,ha='center')
    plt.savefig('{:s}/Length_hours_rains{:s}.pdf'.format(resultdir,tits))
    plt.show()

    sun_bins = 9

    plt.clf()
    plot_mensual_rain_sun(dfff, 'humidity',
                          hum_threshold=90, lengthcut=3, direction='<',nbins=sun_bins,join_months=True, weights=None) #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/Length_hours_rains_month_sun_short{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    weights = weights_gaps_months(dfff, year_min=2003, year_max=2023, gap_min_hours=3, is_rain=True, is_20to22=is_20to22)
    print ('WEIGHTS MONTH: ', weights)
    plot_mensual_rain_sun(dfff, 'humidity',
                          hum_threshold=90, lengthcut=3, direction='<',nbins=sun_bins,join_months=True, weights=weights) #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/Length_hours_rains_month_sun_short_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plot_mensual_rain_sun(dfff, 'humidity',
                          hum_threshold=90, lengthcut=3, direction='<',nbins=sun_bins,join_months=True, weights=weights,plot_tot=True) #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/Length_tothours_rains_month_sun_short_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    sun_bins = 6
    plot_mensual_rain_sun(dfff, 'humidity',
                          hum_threshold=90, lengthcut=3, direction='>',nbins=sun_bins,join_months=True) #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/Length_hours_rains_month_sun_long{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    weights = weights_gaps_months(dfff, year_min=2003, year_max=2023, gap_min_hours=24, is_rain=True, is_20to22=is_20to22)
    print ('WEIGHTS MONTH: ', weights)    
    plot_mensual_rain_sun(dfff, 'humidity',
                          hum_threshold=90, lengthcut=3, direction='>',nbins=sun_bins,join_months=True, weights=weights) #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/Length_hours_rains_month_sun_long_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plot_mensual_rain_sun(dfff, 'humidity',
                          hum_threshold=90, lengthcut=3, direction='>',nbins=sun_bins,join_months=True, weights=weights, plot_tot=True) #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/Length_tothours_rains_month_sun_long_weights{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    ax,ax2 = plot_mensual_rain(dfff, 'humidity', hum_threshold=90) #, name_ws_current)
    plot_mensual_rain(dfff[mask_snow], 'humidity',hum_threshold=90,
                      color1='forestgreen',color2='seagreen',color3='paleturquoise',ax=ax,ax2=ax2) #, name_ws_current)
    #plot_mensual_rain(dfff[mask_sleet], 'humidity', hum_threshold=90,color1='firebrick',color2='lightcoral',ax=ax,ax2=ax2) #, name_ws_current)
    #plt.xlabel('Month')
    #plt.ylabel('Wind Speed (km/h)')
    plt.savefig('{:s}/Length_hours_rains_month{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    #months = np.arange(1,13)    
    #plt.clf()
    #fig,ax = plt.subplots(2,1, sharex=True)    
    #for m in months:
    #    length_arr_m = count_extremes_length_month(dfff,'humidity',m, 90, False)
    #    if (len(length_arr_m)>1):
    #        arr = np.array(length_arr)
    #        ax[0].violinplot(arr[arr<3], positions=[m])
    #        ax[1].violinplot(arr[arr>=3], positions=[m])
    #    else:
    #        continue
    #ax[0].set_ylabel('Length of rain (<3 hours)',fontsize=16)
    #ax[1].set_ylabel('Length of rain (>3 hours)',fontsize=16)
    #ax[0].yaxis.set_tick_params(labelsize=18)
    #ax[1].yaxis.set_tick_params(labelsize=18)
    #ax[0].xaxis.set_tick_params(labelsize=15)    
    #plt.xticks(months, months_n,ha='center')
    #plt.savefig('{:s}/Length_hours_rains_month.pdf')
    #plt.show()

    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.hist(length_arr,    bins=bin_edges,log=False,align='mid',histtype='step', edgecolor='steelblue',lw=3, label='total precipitation', linestyle='solid')
    #plt.hist(length_arr_sl, bins=bin_edges,log=False,align='mid',histtype='step', facecolor='tomato',lw=3, label='sleet')
    plt.hist(length_arr_sn, bins=bin_edges,log=False,align='mid',histtype='step', edgecolor='seagreen',lw=3, label='snowfall', linestyle='solid')
    plt.hist(length_arr_sc, bins=bin_edges,log=False,align='mid',histtype='step', edgecolor='orange',lw=3, label='snowcover', linestyle='solid')
    #plt.hist(length_arr_ra, bins=bin_edges,log=False,align='mid',histtype='step', facecolor='gold',lw=3, label='rain')
    plt.xscale('log')
    plt.xlabel('Precipitation duration (hours)',fontsize=26)
    plt.ylabel('Counts/(0.2 dex)',fontsize=26)

    # Create new legend handles but use the colors from the existing ones
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    plt.legend(handles=new_handles, labels=labels,loc='best')
    plt.savefig('Length_hours_rains_hist{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize = (10,5), constrained_layout = True)
    plt.hist(length_arr,    weights=length_arr, bins=bin_edges,log=False,
             align='mid',histtype='step', edgecolor='steelblue',lw=3, label='total precipitation', linestyle='solid')
    #plt.hist(length_arr_sl, bins=bin_edges,log=False,align='mid',histtype='step', facecolor='tomato',lw=3, label='sleet')
    plt.hist(length_arr_sn, weights=length_arr_sn, bins=bin_edges,log=False,
             align='mid',histtype='step', edgecolor='seagreen',lw=3, label='snowfall', linestyle='solid')
    plt.hist(length_arr_sc, weights=length_arr_sc,
             bins=bin_edges,log=False,align='mid',histtype='step', edgecolor='orange',lw=3, label='snowcover', linestyle='solid')
    #plt.hist(length_arr_ra, bins=bin_edges,log=False,align='mid',histtype='step', facecolor='gold',lw=3, label='rain')
    plt.xscale('log')
    plt.xlabel('Precipitation duration (hours)',fontsize=26)
    plt.ylabel('Accumulated duration (hours)/(0.2 dex)',fontsize=26)

    # Create new legend handles but use the colors from the existing ones
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    plt.legend(handles=new_handles, labels=labels,loc='upper left')
    plt.savefig('{:s}/Length_hours_rains_hist_duration{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    gaps   = []
    months = range(1,13)
    for m in np.arange(1,13):
        gaps.append(count_gaps_month(dfff,m,year_start,year_end))

    plt.clf()
    plt.plot(months, gaps, marker = 'o', linestyle = 'none', color = 'steelblue', markersize = 8,
             markerfacecolor = 'k', markeredgecolor = 'k')
    #plt.xlabel('Month', fontsize=20)
    plt.ylabel('Data gaps (days)',    fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)    
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(np.arange(1,13), months_n,ha='center')
    plt.savefig('{:s}/Gaps_days_rains{:s}.pdf'.format(resultdir,tits))
    plt.show()

    plt.clf()
    test_cuts = np.arange(60,100,5)
    for test_cut in test_cuts:
        rains = []
        for m in np.arange(1,13):
            rains.append(extremes_month(dfff,'humidity', test_cut, m, year_start, year_end))
        plt.plot(months, np.array(rains), marker = 'o', linestyle = 'none', markersize = 10, markeredgecolor = 'k', 
                 label='threshold={:d} (%)'.format(test_cut))  # markerfacecolor = 'white', 
    #plt.xlabel('Month', fontsize=20)
    plt.ylabel('Number of periods exceeding humidity threshold', fontsize=20)
    plt.legend(loc='best', fontsize=18)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(np.arange(1,13), months_n,ha='center')
    plt.savefig('{:s}/Mensual_rains_all{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    plt.clf()
    test_cuts = np.arange(60,100,5)
    for test_cut in test_cuts:
        weights = calc_all_weights(dfff,'humidity', test_cut, year_start, year_end, is_rain=True,lengthcut=3,direction='>', is_20to22=is_20to22)
        print ('years', years, ' Weights', weights)
        plt.plot(years, weights, marker = 'o', linestyle = 'none', markersize = 7, #markeredgecolor = 'k', 
                 label='threshold={:d} (%)'.format(test_cut))  # markerfacecolor = 'white', 
    #plt.xlabel('Month', fontsize=20)
    plt.ylabel('Weights due to data gaps', fontsize=18)
    plt.legend(loc='best', fontsize=18)
    plt.ylim(-0.05,1.05)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=15)
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(years, years,ha='center')
    plt.savefig('{:s}/Yearly_weights_rains{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

    for test_cut in [98, 95, 90, 80]:
    
        k = np.array(extremes_with_lengthcut(dfff, 'humidity', test_cut, year_start, year_end,3,'>'))

        print ('LONG EXTREMES for test cut ',test_cut, ': ', k)
        
        plot_profile_likelihood(k)
        #plt.title('Relative humidity threshold: {:d} (%)'.format(test_cut), fontsize=20)
        plt.savefig('{:s}/likelihood_cut_{:d}_rains_long{:s}.pdf'.format(resultdir,test_cut,tits), bbox_inches='tight')
        plt.show()           
        
        weights_for_p0 = np.array(calc_all_weights(dfff,'humidity', test_cut,
                                                   year_start, year_end,
                                                   is_rain=True, lengthcut=3, direction='>',is_20to22=is_20to22))
        print ('years', years, ' Weights LONG', weights_for_p0)
        plot_profile_likelihood(k,weights_for_p0)
        #plt.title('Relative humidity threshold: {:d} (%)'.format(test_cut), fontsize=20)
        plt.savefig('{:s}/likelihood_cut_{:d}_weights_rains_long{:s}.pdf'.format(resultdir,test_cut,tits), bbox_inches='tight')
        plt.show()           
        
        k = np.array(extremes_with_lengthcut(dfff, 'humidity', test_cut, year_start, year_end,3,'<'))

        print ('SHORT EXTREMES for test cut ', test_cut,': ', k)
        
        plot_profile_likelihood(k)
        #plt.title('Relative humidity threshold: {:d} (%)'.format(test_cut), fontsize=20)
        plt.savefig('{:s}/likelihood_cut_{:d}_rains_short{:s}.pdf'.format(resultdir,test_cut,tits), bbox_inches='tight')
        plt.show()           
        
        weights_for_p0 = np.array(calc_all_weights(dfff,'humidity', test_cut,
                                                   year_start, year_end,
                                                   is_rain=True, lengthcut=3, direction='<',is_20to22=is_20to22))
        print ('years', years, ' Weights SHORT', weights_for_p0)        
        plot_profile_likelihood(k,weights_for_p0)
        #plt.title('Relative humidity threshold: {:d} (%)'.format(test_cut), fontsize=20)
        plt.savefig('{:s}/likelihood_cut_{:d}_weights_rains_short{:s}.pdf'.format(resultdir,test_cut,tits), bbox_inches='tight')
        plt.show()           
        

    print ('FINSIHED LIKELIHOOD')

    cuts = np.arange(80,100,5)
    
    for cut in cuts:
        weights = np.array(calc_all_weights(dfff,'humidity',cut, year_start, year_end,
                                            is_rain=True, lengthcut=3, direction='>',is_20to22=is_20to22))                
        k = extremes_with_lengthcut(dfff, 'humidity', cut, year_start, year_end,lengthcut=3, direction='>')
        plot_extremes(k, year_start, year_end, weights)
        plt.ylabel('Number of periods exceeding humidity threshold of {:d} %'.format(cut), fontsize = 20)
        plt.savefig('{:s}/Rains_cut_{:d}_long{:s}.pdf'.format(resultdir,cut,tits), bbox_inches='tight')
        plt.show()                   
            
    for cut in cuts:
        weights = np.array(calc_all_weights(dfff,'humidity',cut, year_start, year_end,
                                            is_rain=True, lengthcut=3, direction='>',is_20to22=is_20to22))                
        k = extremes_with_lengthcut(dfff, 'humidity', cut, year_start, year_end,lengthcut=3, direction='<')
        plot_extremes(k, year_start, year_end,weights)
        plt.ylabel('Number of periods exceeding humidity threshold of {:d} %'.format(cut), fontsize = 20)
        plt.savefig('{:s}/Rains_cut_{:d}_short{:s}.pdf'.format(resultdir,cut,tits), bbox_inches='tight')
        plt.show()                   
            
    TSs, p0s, alphas, alphas_sup1, alphas_inf1, alphas_sup2, alphas_inf2 = [], [], [], [], [], [], []

    for cut in cuts:

        weights = np.array(calc_all_weights(dfff,'humidity',cut, year_start, year_end,
                                            is_rain=True,lengthcut=3, direction='<',is_20to22=is_20to22))
        
        k = np.array(extremes_with_lengthcut(dfff, 'humidity', cut, year_start, year_end,lengthcut=3, direction='<'))
        ts, p0, alpha = TS(k, weights)
        TSs.append(ts)
        p0s.append(p0)
        alphas.append(alpha)
        conf_int1 = confidence_intervals(k, chi2=1, weights=weights)
        alphas_sup1.append(conf_int1[0]) 
        alphas_inf1.append(conf_int1[1])

        conf_int2 = confidence_intervals(k, chi2=4, weights=weights)
        alphas_sup2.append(conf_int2[0]) 
        alphas_inf2.append(conf_int2[1])

    plt.clf()
    plt.scatter(cuts, alphas, s = 20)
    plt.fill_between(cuts, alphas_sup2, alphas_inf2, color='seagreen', alpha=0.3)
    plt.fill_between(cuts, alphas_sup1, alphas_inf1, color='tomato', alpha=0.3)
    plt.ylim(-0.3,0.7)
    plt.xlabel('Relative humidity minimum (%)', fontsize = 20)
    plt.ylabel(r'Yearly increase of short periods of precipitation', fontsize = 19)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/alphas_rains_short{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.scatter(cuts, np.array(alphas)/np.array(p0s), s = 20)
    plt.fill_between(cuts, np.array(alphas_sup2)/np.array(p0s), np.array(alphas_inf2)/np.array(p0s), color='seagreen', alpha=0.3)
    plt.fill_between(cuts, np.array(alphas_sup1)/np.array(p0s), np.array(alphas_inf1)/np.array(p0s), color='tomato', alpha=0.3)
    #plt.ylim(-0.3,0.65)
    plt.xlabel('Threshold for relative humidity (%)', fontsize = 20)
    plt.ylabel(r'Yearly occurrence probability increase', fontsize = 19)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/alphas_rel_rains_short{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    
    plt.clf()
    plt.plot(cuts, TSs, '.' , markersize=20)
    plt.ylabel(r'$\sqrt{TS}$', fontsize = 20)
    plt.xlabel('Threshold relative humidity (%)', fontsize = 20)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/TSs_rains_short{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.plot(cuts, p0s, '.', markersize=20, color='gold')
    plt.xlabel('Threshold relative humidity (%)', fontsize = 20)
    plt.ylabel(r'Mean yearly occurrence', fontsize = 18)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/p0s_rains_short{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()


    TSs, p0s, alphas, alphas_sup1, alphas_inf1, alphas_sup2, alphas_inf2 = [], [], [], [], [], [], []

    for cut in cuts:

        weights = np.array(calc_all_weights(dfff,'humidity',cut, year_start, year_end,
                                            is_rain=True,lengthcut=3, direction='>',is_20to22=is_20to22))
        
        k = np.array(extremes_with_lengthcut(dfff, 'humidity', cut, year_start, year_end,lengthcut=3, direction='>'))
        ts, p0, alpha = TS(k, weights)
        TSs.append(ts)
        p0s.append(p0)
        alphas.append(alpha)
        conf_int1 = confidence_intervals(k, chi2=1, weights=weights)
        alphas_sup1.append(conf_int1[0]) 
        alphas_inf1.append(conf_int1[1])

        conf_int2 = confidence_intervals(k, chi2=4, weights=weights)
        alphas_sup2.append(conf_int2[0]) 
        alphas_inf2.append(conf_int2[1])

    plt.clf()
    plt.scatter(cuts, alphas, s = 20)
    plt.fill_between(cuts, alphas_sup2, alphas_inf2, color='seagreen', alpha=0.3)
    plt.fill_between(cuts, alphas_sup1, alphas_inf1, color='tomato', alpha=0.3)
    plt.ylim(-0.3,0.7)
    plt.xlabel('Relative humidity minimum (%)', fontsize = 20)
    plt.ylabel(r'Yearly increase of long periods of precipitation', fontsize = 19)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/alphas_rains_long{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.scatter(cuts, np.array(alphas)/np.array(p0s), s = 20)
    plt.fill_between(cuts, np.array(alphas_sup2)/np.array(p0s), np.array(alphas_inf2)/np.array(p0s), color='seagreen', alpha=0.3)
    plt.fill_between(cuts, np.array(alphas_sup1)/np.array(p0s), np.array(alphas_inf1)/np.array(p0s), color='tomato', alpha=0.3)
    #plt.ylim(-0.3,0.65)
    plt.xlabel('Threshold for relative humidity (%)', fontsize = 20)
    plt.ylabel(r'Yearly occurrence probability increase', fontsize = 19)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/alphas_rel_rains_long{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    
    plt.clf()
    plt.plot(cuts, TSs, '.' , markersize=20)
    plt.ylabel(r'$\sqrt{TS}$', fontsize = 20)
    plt.xlabel('Threshold relative humidity (%)', fontsize = 20)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/TSs_rains_long{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.plot(cuts, p0s, '.', markersize=20, color='gold')
    plt.xlabel('Threshold relative humidity (%)', fontsize = 20)
    plt.ylabel(r'Mean yearly occurrence', fontsize = 18)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/p0s_rains_long{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
            
    TSs, p0s, alphas, alphas_sup1, alphas_inf1, alphas_sup2, alphas_inf2 = [], [], [], [], [], [], []
    lengths = np.arange(3,110,3)

    cut = 90
    
    for length in lengths:

        weights = np.array(calc_all_weights(dfff,'humidity',cut, year_start, year_end,
                                            is_rain=True,lengthcut=length, direction='>',is_20to22=is_20to22))
        
        k = np.array(extremes_with_lengthcut(dfff, 'humidity', cut, year_start, year_end,lengthcut=length, direction='>'))

        print ('EXTREMES for length cut ', length,': ', k)

        ts, p0, alpha = TS(k, weights)

        print ('TS, p0, alpha:',ts,p0,alpha)
        
        TSs.append(ts)
        p0s.append(p0)
        alphas.append(alpha)
        conf_int1 = confidence_intervals(k, chi2=1, weights=weights)
        print ('CONF INT1:',conf_int1)        
        alphas_sup1.append(conf_int1[0]) 
        alphas_inf1.append(conf_int1[1])

        conf_int2 = confidence_intervals(k, chi2=4, weights=weights)
        print ('CONF INT2:',conf_int2)                
        alphas_sup2.append(conf_int2[0]) 
        alphas_inf2.append(conf_int2[1])


    plt.clf()
    plt.scatter(lengths, alphas, s = 20)
    plt.fill_between(lengths, alphas_sup2, alphas_inf2, color='seagreen', alpha=0.3)
    plt.fill_between(lengths, alphas_sup1, alphas_inf1, color='tomato', alpha=0.3)
    plt.ylim(-1.2,1.2)
    plt.xlabel('Threshold length of continued precipitation (h)', fontsize = 26)
    plt.ylabel('Yearly increase of events', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/alphas_weights_rains{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.scatter(lengths, np.array(alphas)/np.array(p0s), s = 20)
    plt.fill_between(lengths, np.array(alphas_sup2)/np.array(p0s), np.array(alphas_inf2)/np.array(p0s), color='seagreen', alpha=0.3)
    plt.fill_between(lengths, np.array(alphas_sup1)/np.array(p0s), np.array(alphas_inf1)/np.array(p0s), color='tomato', alpha=0.3)
    #plt.ylim(-0.3,0.65)
    plt.xlabel('Threshold length of continued precipitation (h)', fontsize = 26)
    plt.ylabel(r'Yearly occurrence prob. increase', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/alphas_rel_weights_rains{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    
    plt.clf()
    plt.plot(lengths, TSs, '.' , markersize=20)
    plt.ylabel(r'$\sqrt{TS}$', fontsize = 20)
    plt.xlabel('Threshold length of continued precipitation (h)', fontsize = 20)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.savefig('{:s}/TSs_weights_rains{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()
    
    plt.clf()
    plt.plot(lengths, p0s, '.', markersize=20, color='gold')
    plt.xlabel('Threshold length of continued precipitation (h)', fontsize = 26)
    plt.ylabel(r'Mean yearly occurrence', fontsize = 26)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.savefig('{:s}/p0s_weights_rains{:s}.pdf'.format(resultdir,tits), bbox_inches='tight')
    plt.show()

def load_data() -> None: 
    
    if is_naoi:
        df_naoi = naoi_read(naoi_file,WS_start)    

    if is_NOT:
        df_not = pd.read_hdf(not_file)

    dff = pd.read_hdf(h5file_long)
        
    print('DUPLICATES: ',dff[dff['is_dup'] == True].head(n=100))
    print('LOW HUMIDITY: ',dff[dff['humidity']<0.1])
    print('MJD not valid: ',dff[dff['mjd'].isnull()])
    #dff = dff[dff['is_dup']==False]
        
if __name__ == '__main__':
    SetUp()

    load_data()
    
    #fig = plt.figure(figsize = (10,5), constrained_layout = True)

    #plot_datacount()
    #plot_downtime()
    #plot_temperature()
    #plot_DP()
    #plot_DTR()
    #plot_temperature_not()
    #plot_snow()
    #plot_rainy_periods()
    #plot_humidity()
    #plot_humidity_not()
    #plot_pressure_not()
    #plot_pressure()    
    #plot_wind()
    #plot_huracans()

