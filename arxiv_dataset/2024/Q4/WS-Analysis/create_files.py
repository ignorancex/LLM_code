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

from setup_helper import SetUp
from sun_helper import AltAzSun, plot_anual_sol
from filter_helper import *
from plot_helper import mjd_corrector, plot_historic, plot_historic_fit_results
from likelihood_helper import * 
from not_helper import not_read
from naoi_helper import *
from wind_helper import calc_ti, winddiraverage
from coverage_helper import expected_data_per_day, apply_coverage

jsonfile     = 'Data/WS2003-23.json'
h5file_short = 'Data/WS2003-23_short.h5'
h5file_long  = 'Data/WS2003-23_long.h5'
h5file_corr  = 'Data/WS2003-23_corr.h5'
# from: https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv
naoi_file = 'norm.daily.nao.cdas.z500.19500101_current.csv'
#not_file = 'NOT_2003_2023.csv'
#h5file_not = 'NOT_2003_2023.h5'

is_json  = False
invertcuts = False
# This option is needed once to create the long file,
# Beware, it is very slow and may take several hours,
# because of the slow astropy sun coordinates calculator
create_long = True
create_corr = False
create_not = False

WS_start      = '2003-01-01 00:00:01'
WS_relocation = '2004-03-01 00:00:01'
new_WS        = '2007-03-20 00:00:01'
new_model     = '2017-04-10 00:00:01'
old_model     = '2023-01-16 00:00:01'
NOT_end_of_data = '2019-12-31 23:59:59'

pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 500)
pd.set_option('max_seq_items',500)

if __name__ == '__main__':
    SetUp()

    if (is_json):
        df = pd.read_json(jsonfile, convert_dates = True, orient = 'index')
        pd.to_datetime(df.index)

        df = df.drop(columns=['fBits','mjd','tempSensor','tngDust','tngSeeing'])
        df = df.sort_index()

        df['pressure_reliable'] = True
        df['temperature_reliable'] = True
        df['humidity_reliable'] = True
        df['wind_reliable'] = True                

        # correction of pressure from old location to new one (12m height difference)
        df.loc[(df.index < WS_relocation),'pressure'] = df.loc[(df.index < WS_relocation),'pressure'].mul(np.exp(0.012/8.4))
        df.loc[(df.index < WS_relocation),'pressure_reliable'] = False
        
        # calculation of 10min. wind speed average for those data that have none
        mask_w = (df['windSpeedAverage']<0)
        df.loc[mask_w,'windSpeedAverage'] = df[mask_w].rolling('10T',center=True,min_periods=1)['windSpeedCurrent'].mean()

        # correction of negative wind directions
        mask_w = ((df['windDirection']<0) & (df['windDirection']>=-180))
        df.loc[mask_w,'windDirection'] = df.loc[mask_w,'windDirection'] + 360
        mask_w = ((df['windDirectionAverage']<0) & (df['windDirectionAverage']>=-180))        
        df.loc[mask_w,'windDirectionAverage'] = df.loc[mask_w,'windDirectionAverage'] + 360

        # calculation of 10min. wind direction average for those data that have none
        df.loc[((df['windSpeedCurrent'] == 0) & (df['windDirectionAverage'] == -100179) & (df['windSpeedAverage'] == 0) & (df['windGust'] == 0)), 'wind_reliable'] = False
        mask_w = (df['windDirectionAverage']<-200)
        df.loc[mask_w,'windDirectionAverage'] = df[mask_w].rolling('10T',center=True,min_periods=1)['windDirection'].apply(winddiraverage)

        df.loc[(df['temperature'] < -50), 'temperature'] = np.nan
        df.loc[(df['pressure'] < -1), 'pressure'] = np.nan
        df.loc[(df['humidity'] < -1), 'humidity'] = np.nan
        df.loc[(df['windSpeedCurrent'] < -1), 'windSpeedCurrent'] = np.nan
        df.loc[(df['windSpeedAverage'] < -1), 'windSpeedAverage'] = np.nan
        df.loc[(df['windGust'] < -1), 'windGust'] = np.nan
        df.loc[(df['windDirection'] < -1), 'windDirection'] = np.nan
        df.loc[(df['windDirectionAverage'] < -1), 'windDirectionAverage'] = np.nan
        df.loc[(df['windSpeedCurrent'] < -1), 'wind_reliable'] = False
        
        df.loc[(df['rain'] < 0), 'rain'] = np.nan                                
        
        df = Filtre_BrokenStation(df,False)
        df = Filtre_Pressure(df, False)
        df.loc[~(Filtre_Temperatures(df, False)),'temperature_reliable'] = False
        mask_w = ((df['windSpeedCurrent'] == 0) & (df['temperature'] < 1.0) & (df.index < new_WS))
        df.loc[mask_w,'windSpeedCurrent'] = np.nan  # frozen anemometers
        df.loc[mask_w,'windSpeedAverage'] = np.nan  # frozen anemometers
        df.loc[mask_w,'windGust'] = np.nan  # frozen anemometers
        df.loc[mask_w,'windDirection'] = np.nan  # frozen anemometers
        df.loc[mask_w,'windDirectionAverage'] = np.nan  # frozen anemometers

        #mask_h = ((df.index > RH_drift_start) & (df.index < RH_drift_end))
        #df.loc[mask_h,'humidity_reliable'] = False # gradual increase of humidity, compared with other stations
        df.loc[~(Filtre_Humidity(df, False)),'humidity_reliable'] = False        
        
        df.info()

        store = pd.HDFStore(h5file_short)
        store['dff'] = df
        store.close()
        print ('Short file', h5file_short, ' successfully created')
        exit(0)

    if create_long:

        dff = pd.read_hdf(h5file_short)
        
        dff['Year'] = dff.index.year                                
        dff['Month'] = dff.index.month 
        dff = AltAzSun(dff)
        #plot_anual_sol(dff)

        # convert JD to MJD, and subtract then 55000,
        # in order to maintain sufficient precision, if the variable
        # is stored as float. 
        dff['mjd'] = dff.index.to_julian_date()-2400000.5-mjd_corrector
        
        dff['DP'] = np.nan
        dff['DP_reliable'] = True

        # calculation of dew point according to August-Roche-Magnus approximation
        # (Buck, A. Journal of Applied Meteorology. 20 (12): 1527–1532
        b1 = 17.368   # 
        c1 = 238.88   # degC
        b2 = 17.966   # 
        c2 = 247.15   # degC
        maskg0 = (dff['temperature'] > 0)
        maskl0 = (dff['temperature'] <= 0)
        gamma1 = np.log(dff.loc[maskg0,'humidity'].mul(0.01)) + dff.loc[maskg0,'temperature'].mul(b1) / (dff.loc[maskg0,'temperature']+c1)
        gamma2 = np.log(dff.loc[maskl0,'humidity'].mul(0.01)) + dff.loc[maskl0,'temperature'].mul(b2) / (dff.loc[maskl0,'temperature']+c2)
        dff.loc[maskg0,'DP'] = gamma1.mul(c1) / (b1 - gamma1)
        dff.loc[maskl0,'DP'] = gamma2.mul(c2) / (b2 - gamma2)
        #mask_h = ((dff.index > RH_drift_start) & (dff.index < RH_drift_end))
        #dff.loc[mask_h,'DP_reliable'] = False # gradual increase of humidity, compared with other stations
        dff.loc[~(Filtre_Temperatures(dff, False)),'DP_reliable'] = False
        dff.loc[~(Filtre_Humidity(dff, False)),'DP_reliable'] = False

        # calculation of wet bulb temperature according to the Stull approximation
        # (Stull, R. Journal of Applied Meteorology and Climatology 50 (2011) 2267
        dff['WB'] = np.nan
        dff['WB_reliable'] = True
        #dff.loc[mask_h,'WB_reliable'] = False # gradual increase of humidity, compared with other stations
        dff.loc[~(Filtre_Temperatures(dff, False)),'WB_reliable'] = False
        dff.loc[~(Filtre_Humidity(dff, False)),'WB_reliable'] = False

        a1 = 0.151977
        a2 = 8.313659
        a3 = 1.676331
        a4 = 0.00391838
        a5 = 0.023101
        a6 = 4.686035

        at1 = dff.loc[(dff['WB_reliable']==True),'temperature']*np.arctan(a1*np.power(dff.loc[(dff['WB_reliable']==True),'humidity']+a2,0.5))
        at2 = np.arctan(dff.loc[(dff['WB_reliable']==True),'temperature']+dff.loc[(dff['WB_reliable']==True),'humidity'])
        at3 = np.arctan(dff.loc[(dff['WB_reliable']==True),'humidity']-a3)
        at4 = a4*np.power(dff.loc[(dff['WB_reliable']==True),'humidity'],1.5)*np.arctan(a5*dff.loc[(dff['WB_reliable']==True),'humidity'])
        dff.loc[dff['WB_reliable']==True,'WB'] = at1 + at2 - at3 + at4 - a6
        
        # calculation of wet bulb temperature according to the Ding approximation
        # B. Ding, K. Yang, J. Qin, L. Wang, Y. Chen, and X. He.
        # The dependence of precipitation types on surface elevation and meteorological conditions and its parameterization.
        # Journal of Hydrology, 513:154– 163, 2014.

        dff['WBD'] = np.nan
        dff['WBD_reliable'] = True
        #dff.loc[mask_h,'WB_reliable'] = False # gradual increase of humidity, compared with other stations
        dff.loc[~(Filtre_Temperatures(dff, False)),'WBD_reliable'] = False
        dff.loc[~(Filtre_Humidity(dff, False)),'WBD_reliable'] = False

        a1 = 0.000643
        a2 = 6.1078
        a3 = 17.27
        a4 = 237.3

        esat = a2*np.exp(dff.loc[(dff['WBD_reliable']==True),'temperature']*a3/(dff.loc[(dff['WBD_reliable']==True),'temperature']+a4))
        desatdT = esat * ( a3 / (dff.loc[(dff['WBD_reliable']==True),'temperature']+a4) - a3 * (dff.loc[(dff['WBD_reliable']==True),'temperature']) / np.power((dff.loc[(dff['WBD_reliable']==True),'temperature']+a4),2))
        dff.loc[dff['WBD_reliable']==True,'WBD'] = dff.loc[(dff['WBD_reliable']==True),'temperature'] - esat * (1- 0.01*dff.loc[(dff['WBD_reliable']==True),'humidity']) / ( a1 * dff.loc[(dff['WBD_reliable']==True),'pressure'] + desatdT)
        
        mask_ti = (dff['wind_reliable'] == True)
        dff['windTI'] = np.nan
        dff.loc[mask_ti,'windTI'] = dff.loc[mask_ti,'windSpeedCurrent'].rolling('10T',center=True,min_periods=1).agg(calc_ti)
        
        # create new df entries containing time differences for between up to 5 rows behind
        dff['diff1']  = dff.index.to_series().diff(1).fillna(pd.Timedelta(seconds=0))
        dff['diff5']  = dff.index.to_series().diff(5).fillna(pd.Timedelta(seconds=0))
        dff['diff10'] = dff.index.to_series().diff(10).fillna(pd.Timedelta(seconds=0))        

        # less than 1min time differences are marked as duplicates
        dff['is_dup'] = False
        mask_dup = (dff['diff1']>pd.Timedelta(0,'m')) & (dff['diff1']<=pd.Timedelta(1,'m'))
        dff.loc[mask_dup,'is_dup'] = True
        
        # gradients defined as difference w.r.t previous row(s), divided by time difference
        dff['Tgradient1'] = dff['temperature'].diff(1) / dff['diff1'].dt.total_seconds().div(60)
        dff['Tgradient5'] = dff['temperature'].diff(5) / dff['diff5'].dt.total_seconds().div(60)
        dff['Tgradient10'] = dff['temperature'].diff(10) / dff['diff10'].dt.total_seconds().div(60)        

        # smooth temperature with a rolling window to 7 min.
        dff['temperatureR'] = dff.rolling('7T',center=True,min_periods=1)['temperature'].mean()

        # temperature gradients from smoothing
        dff['Tgradient1R']  = dff['temperatureR'].diff(1) / dff['diff1'].dt.total_seconds().div(60)
        dff['Tgradient5R']  = dff['temperatureR'].diff(5) / dff['diff5'].dt.total_seconds().div(60)
        dff['Tgradient10R'] = dff['temperatureR'].diff(10)/ dff['diff10'].dt.total_seconds().div(60)        

        # masks to set the gradients after gaps to nan
        # expect two minutes
        mask1 = ((dff['diff1']  > pd.Timedelta(1,'min'))  & (dff['diff1']<pd.Timedelta(3,'min')))
        # expect 10 minutes        
        mask5 = ((dff['diff5']  > pd.Timedelta(7,'min'))  & (dff['diff5']<pd.Timedelta(12,'min')))
        # expecte 20 minutes
        mask10= ((dff['diff10'] > pd.Timedelta(15,'min')) & (dff['diff10']<pd.Timedelta(25,'min')))        
        
        dff.loc[~mask1,'Tgradient1']   = np.nan
        dff.loc[~mask5,'Tgradient5']   = np.nan
        dff.loc[~mask10,'Tgradient10'] = np.nan        

        dff.loc[~mask1,'Tgradient1R']   = np.nan
        dff.loc[~mask5,'Tgradient5R']   = np.nan
        dff.loc[~mask10,'Tgradient10R'] = np.nan        
            
        # humidity gradients

        # gradients defined as difference w.r.t previous row(s), divided by time difference
        dff['Rgradient1']  = dff['humidity'].diff(1)  / dff['diff1'].dt.total_seconds().div(60)
        dff['Rgradient5']  = dff['humidity'].diff(5)  / dff['diff5'].dt.total_seconds().div(60)
        dff['Rgradient10'] = dff['humidity'].diff(10) / dff['diff10'].dt.total_seconds().div(60)        

        # smooth humidity with a rolling window to 7 min.
        dff['humidityR'] = dff.rolling('7T',center=True,min_periods=1)['humidity'].mean()

        # humidity gradients from smoothing
        dff['Rgradient1R']  = dff['humidityR'].diff(1) / dff['diff1'].dt.total_seconds().div(60)
        dff['Rgradient5R']  = dff['humidityR'].diff(5) / dff['diff5'].dt.total_seconds().div(60)
        dff['Rgradient10R'] = dff['humidityR'].diff(10)/ dff['diff10'].dt.total_seconds().div(60)        

        dff.loc[~mask1,'Rgradient1']   = np.nan
        dff.loc[~mask5,'Rgradient5']   = np.nan
        dff.loc[~mask10,'Rgradient10'] = np.nan        

        dff.loc[~mask1,'Rgradient1R']   = np.nan
        dff.loc[~mask5,'Rgradient5R']   = np.nan
        dff.loc[~mask10,'Rgradient10R'] = np.nan        
            
        # pressure gradients

        # gradients defined as difference w.r.t previous row(s), divided by time difference
        dff['Pgradient1']  = dff['pressure'].diff(1)  / dff['diff1'].dt.total_seconds().div(60)
        dff['Pgradient5']  = dff['pressure'].diff(5)  / dff['diff5'].dt.total_seconds().div(60)
        dff['Pgradient10'] = dff['pressure'].diff(10) / dff['diff10'].dt.total_seconds().div(60)        

        # smooth pressure with a rolling window to 7 min.
        dff['pressureR'] = dff.rolling('7T',center=True,min_periods=1)['pressure'].mean()

        # pressure gradients from smoothing
        dff['Pgradient1R']  = dff['pressureR'].diff(1)  / dff['diff1'].dt.total_seconds().div(60)
        dff['Pgradient5R']  = dff['pressureR'].diff(5)  / dff['diff5'].dt.total_seconds().div(60)
        dff['Pgradient10R'] = dff['pressureR'].diff(10) / dff['diff10'].dt.total_seconds().div(60)        

        dff.loc[~mask1,'Pgradient1']   = np.nan
        dff.loc[~mask5,'Pgradient5']   = np.nan
        dff.loc[~mask10,'Pgradient10'] = np.nan        

        dff.loc[~mask1,'Pgradient1R']   = np.nan
        dff.loc[~mask5,'Pgradient5R']   = np.nan
        dff.loc[~mask10,'Pgradient10R'] = np.nan        
            
        dff.info()

        store = pd.HDFStore(h5file_long)
        store['dff'] = dff
        store.close()
        print ('Long file', h5file_long, ' successfully created')
        exit(0)

    if (create_corr):

        df_naoi = naoi_read(naoi_file)
        #print ('NAOI: ', df_naoi.head())
        dff = pd.read_hdf(h5file_long)
        #print ('DFF: ', dff.head())
        dfff = dff[((dff['temperature_reliable']==True) & (dff.index > WS_relocation))]
        dfff, coverage = apply_coverage(dfff,debug=False)

        dff_h = dff[((dff['humidity_reliable']==True) & (dff.index > WS_relocation))]
        dff_h, coverage_h = apply_coverage(dff_h,debug=False)
        
        dff_p = dff[((dff['pressure_reliable']==True) & (dff.index > WS_relocation) & (dff['pressure']>750))]
        dff_p, coverage_p = apply_coverage(dff_p,debug=False)

        dff_w = dff[((dff['wind_reliable']==True) & (dff.index > WS_relocation))]        
        dff_w, coverage_w = apply_coverage(dff_w,debug=False)

        dff_n = df_naoi[(df_naoi.index > WS_relocation)]
        dff_n['mjd'] = dff_n.index.to_julian_date()-2400000.5-mjd_corrector
        dff_n = dff_n[dff_n['nao_index_cdas'].notnull()]
        dff_n, coverage_n = apply_coverage(dff_n,data_spacing=60*24, debug=False)
        
        # for temperatures
        dfn = dfff[dfff['coverage'] > 10]        
        mask = ((dfn.index > WS_relocation) & (dfn['humidity_reliable']==True))

        # for humidity
        dfn_h = dff_h[(dff_h['coverage'] > 10)]
        dfn_h.loc[(dfn_h['humidity']<2), 'humidity'] = 1
        mask_h = (dfn_h['humidity']<90)

        # for pressure
        dfn_p = dff_p[(dff_p['coverage'] > 10)]
        mask1 = ((dfn_p.index > new_WS)    & ((dfn_p.index < new_model) | (dfn_p.index > old_model)))
        mask2 = ((dfn_p.index > new_model) & (dfn_p.index < old_model))    

        # for wind
        dfn_w = dff_w[(dff_w['coverage'] > 10)]        
        mask_w = (dfn_w['windSpeedAverage']<40)  # select only Alisio winds
        
        # for naoi
        dfn_n = dff_n.shift(12,freq='H')
        mask_n = (dfn_n.index > WS_relocation)
        
        df_s = dfn[mask].shift(8,freq='H')  # calculate spread from 8:00 to 8:00
        mjd_s = df_s['mjd'].resample('D').mean()
        hum_s = df_s['humidity'].resample('D').mean()    
        diu_s = df_s['temperature'].resample('D').max().dropna()-df_s['temperature'].resample('D').min().dropna()    

        mask_daily = (df_s['mjd'].resample('D').count().dropna() > 95./100*expected_data_per_day)
        mjd_s = mjd_s[mask_daily]
        diu_s = diu_s[mask_daily]
        hum_s = hum_s[mask_daily]
        
        name_temp     = ['A', 'B', 'C', 'phi_mu', 'sigma0' ]
        name_mu2      = ['A', 'B', 'C', 'phi_mu', 'D', 'phi_mu2','sigma0' ]        
        name_mu2_sig2 = ['A', 'B', 'C', 'phi_mu', 'D', 'phi_mu2','sigma0', 'E', 'phi_sig' ]
        name_mu2_sig2_offset = ['A', 'B', 'C', 'phi_mu', 'D', 'phi_mu2','sigma0', 'E', 'phi_sig', 'offset' ]            
        init_temp = np.array([10.6, 0., 6.5, 6.9, 3.0])
        init_diu = np.array([8.2, 0., 0.9, 5.9, 0.7])
        init_mu4 = np.array([24.5, 0.,20.8, 3.8, 26., 6.9, 17.1, -2.6, 3.1])
        init_mu2_sig2_offset = np.array([786., 0., 5.2, 6.6, 3.7, 9.0, 3.1, -1.0, 4.0, 0.])        
        init_wind = np.array([12.5, 0.,1.1, 11.4, 1.1, 11.3, 5.4, 1.2, 9.3])
        init_naoi = np.array([0.,0.,0.3, 1., 0.1, 10, 0.8])
        
        bounds_naoi     = ((None, None), (None, None), (0, None), (0, None), (0, None), (0, None), (0.01,None))        
        bounds_temp     = ((0, None), (None, None), (0, None),    (0, None), (0.1,None))        
        bounds_mu2_sig2 = ((0, None), (None, None), (None, None), (0, None), (0, None), (0, None), (0.1,None), (-10.,None), (0, 12.))
        bounds_mu2_sig2_offset = ((0, None), (None, None), (None, None), (0, None), (0, None), (0, None), (0.1,None), (-10.,None), (0, 12.), (None,None))

        method='L-BFGS-B'
        tol=1e-9

        naoi_lik_median2 = Likelihood_Wrapper(loglike2,mu2,name_mu2,init_naoi,bounds_naoi,
                                             'NAOI, daily values')
        naoi_lik_median2.setargs_seq(dfn_n['mjd'],dfn_n['nao_index_cdas'])
        naoi_lik_median2.like_minimize(method=method,tol=tol)
        residuals_naoi = naoi_lik_median2.full_residuals()        

        print ('residuals NAOI:', residuals_naoi)
        
        plt.figure(figsize = (10,5), constrained_layout = True)
        plot_historic(dfn_n,'nao_index_cdas',coverage_n)
        plt.ylabel('NAOI index', fontsize=18)
        plot_historic_fit_results(dfn_n,mask_n,naoi_lik_median2,is_daily=False)
        #plot_historic_fit_results(dfn_H,mask_H,temp_lik_Haslebacher,is_daily=False,color='orange')    
        plt.savefig('NAOI_sencer.pdf', bbox_inches='tight')
        plt.show()

        
        diu_lik_daily_hum = Likelihood_Wrapper(loglike_hum,mu_hum,
                                               name_temp,init_diu,bounds_temp,
                                               'Eq. (2), all data, w/hum correction')
        diu_lik_daily_hum.setargs_seq_hum(mjd_s,diu_s,hum_s)
        diu_lik_daily_hum.like_minimize(method=method,tol=tol)
        residuals_diu = diu_lik_daily_hum.full_residuals().shift(12,freq='H')        

        print ('residuals DTR:', residuals_diu)
        
        temp_lik_median = Likelihood_Wrapper(loglike,mu,name_temp,init_temp,bounds_temp,
                                           'Eq. (2), daily medians')
        temp_lik_median.setargs_df(dfn,'temperature',mask,
                                   is_daily=True,is_median=True,day_coverage=85)
        temp_lik_median.like_minimize(method=method,tol=tol)
        residuals_temp = temp_lik_median.full_residuals().shift(12,freq='H')        

        print ('residuals TEMP:', residuals_temp)
        
        hum_lik_median4   = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_mu4,bounds_mu2_sig2,
                                               'Daily medians, seas. spreads')
        hum_lik_median4.setargs_df(dfn_h,'humidity',mask_h,
                                   is_daily=True,is_median=True,day_coverage=70)
        hum_lik_median4.like_minimize(method=method,tol=tol)
        residuals_hum = hum_lik_median4.full_residuals(is_sigma2=True).shift(12,freq='H')        

        print ('residuals HUM:', residuals_hum)
        
        press_lik_median4_offset = Likelihood_Wrapper(loglike4_2sets,mu2,name_mu2_sig2_offset,init_mu2_sig2_offset,bounds_mu2_sig2_offset,
                                                      'Daily medians, seas. spreads with offset')
        press_lik_median4_offset.setargs_df2(dfn_p,'pressure',mask1,mask2,
                                             is_daily=True,is_median=True,day_coverage=40)
        press_lik_median4_offset.like_minimize(method=method,tol=tol)
        residuals_press = press_lik_median4_offset.full_residuals(is_sigma2=True,is_offset=True).shift(12,freq='H')        

        print ('residuals PRESS:', residuals_press)
        
        wind_lik_median4   = Likelihood_Wrapper(loglike4,mu2,name_mu2_sig2,init_wind,bounds_mu2_sig2,
                                                'Eq. (5), daily medians, seas. spreads')
        wind_lik_median4.setargs_df(dfn_w,'windSpeedAverage',mask_w,
                                    is_daily=True,is_median=True,day_coverage=70)
        wind_lik_median4.like_minimize(method=method,tol=tol)
        residuals_wind = wind_lik_median4.full_residuals(is_sigma2=True).shift(12,freq='H')        

        print ('residuals wind:', residuals_wind)
        
        naoi_res,  res_naoi = dfn_n.align(residuals_naoi, join='right', axis=0)#.dropna()
        naoi_temp, res_temp = naoi_res.align(residuals_temp, join='left', axis=0)#.dropna()        
        naoi_diu,  res_diu  = naoi_temp.align(residuals_diu,  join='left', axis=0)#.dropna()
        naoi_hum,  res_hum  = naoi_diu.align(residuals_hum,  join='left', axis=0)#.dropna()
        naoi_pres, res_pres  = naoi_hum.align(residuals_press,  join='left', axis=0)#.dropna()
        naoi_wind, res_wind  = naoi_pres.align(residuals_wind,  join='left', axis=0)#.dropna()
        
        print ('NAOI RES: ',naoi_res)
        print ('NAOI TEMP: ',naoi_temp)
        print ('NAOI DIU: ', naoi_diu)
        #res['naoi'] = naoi['nao_index_cdas']
        print ('RES NAOI', res_naoi)        
        print ('RES TEMP', res_temp)
        print ('RES DIU', res_diu)

        
        df = pd.DataFrame(data={'NAOI' : naoi_temp['nao_index_cdas'].values,
                                'NAOI residuals' : res_naoi.values, 
                                'Temp. residuals': res_temp.values,
                                'DTR residuals': res_diu.values,
                                'Hum. residuals': res_hum.values,
                                'Press. residuals': res_pres.values,
                                'Wind residuals': res_wind.values }, index=res_temp.index)
        print ('FINAL',df[df.index > '2004-03-28'].head())

        plt.figure()
        naoi_correlate(df['NAOI'],df['DTR residuals'],color='r')
        naoi_profile(df,df['DTR residuals'],nbins=12,arg='NAOI')
        plt.savefig('NAOI_corr_TDR.pdf',bbox_inches='tight')
        plt.show()

        plt.clf()
        naoi_correlate(df['NAOI'],df['Temp. residuals'],color='r')
        naoi_profile(df,df['Temp. residuals'],nbins=12,arg='NAOI')
        plt.savefig('NAOI_corr_Temp.pdf',bbox_inches='tight')
        plt.show()

        plt.clf()
        naoi_correlate(df['NAOI'],df['Hum. residuals'],color='r')
        naoi_profile(df,df['Hum. residuals'],nbins=12,arg='NAOI')
        plt.savefig('NAOI_corr_Hum.pdf',bbox_inches='tight')
        plt.show()

        plt.clf()
        naoi_correlate(df['NAOI'],df['Press. residuals'],color='r')
        naoi_profile(df,df['Press. residuals'],nbins=12,arg='NAOI')
        plt.savefig('NAOI_corr_Press.pdf',bbox_inches='tight')
        plt.show()

        plt.clf()
        naoi_correlate(df['NAOI'],df['Wind residuals'],color='r')
        naoi_profile(df,df['Wind residuals'],nbins=12,arg='NAOI')
        plt.savefig('NAOI_corr_Wind.pdf',bbox_inches='tight')
        plt.show()

        plt.figure()
        naoi_correlate(df['NAOI residuals'],df['DTR residuals'],color='r')
        naoi_profile(df,df['DTR residuals'],nbins=12,arg='NAOI residuals')
        plt.savefig('NAOIres_corr_TDR.pdf',bbox_inches='tight')
        plt.show()

        plt.clf()
        naoi_correlate(df['NAOI residuals'],df['Temp. residuals'],color='r')
        naoi_profile(df,df['Temp. residuals'],nbins=12,arg='NAOI residuals')
        plt.savefig('NAOIres_corr_Temp.pdf',bbox_inches='tight')
        plt.show()

        plt.clf()
        naoi_correlate(df['NAOI residuals'],df['Hum. residuals'],color='r')
        naoi_profile(df,df['Hum. residuals'],nbins=12,arg='NAOI residuals')
        plt.savefig('NAOIres_corr_Hum.pdf',bbox_inches='tight')
        plt.show()

        plt.clf()
        naoi_correlate(df['NAOI residuals'],df['Press. residuals'],color='r')
        naoi_profile(df,df['Press. residuals'],nbins=12,arg='NAOI residuals')
        plt.savefig('NAOIres_corr_Press.pdf',bbox_inches='tight')
        plt.show()

        plt.clf()
        naoi_correlate(df['NAOI residuals'],df['Wind residuals'],color='r')
        naoi_profile(df,df['Wind residuals'],nbins=12,arg='NAOI residuals')
        plt.savefig('NAOIres_corr_Wind.pdf',bbox_inches='tight')
        plt.show()


        store = pd.HDFStore(h5file_corr)
        store['df'] = df
        store.close()
        print ('Corr file', h5file_corr, ' successfully created')
        exit(0)
        
    if (create_not):

        df_not = not_read(not_file)
        print ('NOT: ', df_not.head())

        mask_broken = ((df_not['Humidity'] == 0) & (df_not['WindDirectionDeg'] == 0) & (df_not['TempInAirDegC'] == 0))
        df_not = df_not[~mask_broken]
        mask_broken = ((df_not['Humidity'] == 0) & (df_not['WindDirectionDeg'] == 0) & (df_not['WindSpeedMS'] == 0))
        df_not = df_not[~mask_broken]
        mask_broken = ((df_not['Humidity'] == 0) & (df_not['WindDirectionDeg'] == 180) & (df_not['WindSpeedMS'] == 0))
        df_not = df_not[~mask_broken]
        mask_broken = ((df_not['WindDirectionDeg'] == 0) & (df_not['WindSpeedMS'] == 60))
        df_not = df_not[~mask_broken]
        mask_broken = ((df_not['PressureHPA'] < 700))
        df_not = df_not[~mask_broken]
        mask_broken = ((df_not['Humidity'] == 33.3) & (df_not['WindDirectionDeg'] == 0) & (df_not['WindSpeedMS'] == 5.5) & (df_not['TempInAirDegC'] == 10))
        df_not = df_not[~mask_broken] 
        mask_broken = ((df_not['Humidity'] == 13.4) & (df_not['WindDirectionDeg'] == 0) & (df_not['WindSpeedMS'] == 0)  & (df_not['TempInAirDegC'] == 8.8))
        df_not = df_not[~mask_broken]
        mask_broken = ((df_not['Humidity'] == 73.7) & (df_not['WindDirectionDeg'] == 110) & (df_not['WindSpeedMS'] == 6.7)  & (df_not['TempInAirDegC'] == 2))
        df_not = df_not[~mask_broken]
        mask_broken = ((df_not['Humidity'] == 60.5) & (df_not['WindDirectionDeg'] == 305) & (df_not['WindSpeedMS'] == 3.1)  & (df_not['TempInAirDegC'] == 2.4))
        df_not = df_not[~mask_broken]
        mask_broken = ((df_not['PressureHPA'] == 770) & (df_not['WindDirectionDeg'] == 90) & (df_not['WindSpeedMS'] == 1)  & (df_not['TempInAirDegC'] == 1))
        df_not = df_not[~mask_broken]


        # create new df entries containing time differences for between up to 5 rows behind
        df_not['diff1']  = df_not.index.to_series().diff(1).fillna(pd.Timedelta(seconds=0))
        df_not['diff5']  = df_not.index.to_series().diff(5).fillna(pd.Timedelta(seconds=0))

        mask_dup = (df_not['diff1']<pd.Timedelta(30,'s'))
        df_not = df_not[~mask_dup]

        # gradients defined as difference w.r.t previous row(s), divided by time difference
        df_not['Tgradient1'] = df_not['TempInAirDegC'].diff(1) / df_not['diff1'].dt.total_seconds().div(60)
        df_not['Tgradient5'] = df_not['TempInAirDegC'].diff(5) / df_not['diff5'].dt.total_seconds().div(60)

        df_not = df_not[df_not['Tgradient1']<3]   # remove these outliers
        
        # humidity gradients

        # gradients defined as difference w.r.t previous row(s), divided by time difference
        df_not['Rgradient1']  = df_not['Humidity'].diff(1)  / df_not['diff1'].dt.total_seconds().div(60)
        df_not['Rgradient5']  = df_not['Humidity'].diff(5)  / df_not['diff5'].dt.total_seconds().div(60)

        # gradients defined as difference w.r.t previous row(s), divided by time difference
        df_not['Pgradient1']  = df_not['PressureHPA'].diff(1)  / df_not['diff1'].dt.total_seconds().div(60)
        df_not['Pgradient5']  = df_not['PressureHPA'].diff(5)  / df_not['diff5'].dt.total_seconds().div(60)

        mask_stuck = ((df_not['Tgradient1'] == 0) & (df_not['Pgradient1'] == 0) & (df_not['Rgradient1'] == 0))
        df_not = df_not[~mask_stuck]
        
        df_not['Year'] = df_not.index.year                                
        df_not['Month'] = df_not.index.month 
        df_not['mjd'] = df_not.index.to_julian_date()-2400000.5-mjd_corrector
        df_not = AltAzSun(df_not)
        
        store = pd.HDFStore(h5file_not)
        store['df'] = df_not
        store.close()
        print ('Corr file', h5file_not, ' successfully created')
        exit(0)
