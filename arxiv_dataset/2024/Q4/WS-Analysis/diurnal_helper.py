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
import pandas as pd
import numpy as np

def average_diurnal_df(df_s,mask,name_temperature,name_humidity,n=5):

    # check if n is odd
    n = int(n)
    if not n%2:
        print ('provided number of entries for diurnal averageing is even, reduce from ',n,' to ',n-1)
        n = n-1
        
    # Group the df by days and retrieve the id (in this case the *exact* datetime) of the daily maxima and minima
    diu_idxmax = df_s.groupby(pd.Grouper(freq = 'D')).apply(lambda x: x[name_temperature].idxmax() if len(x) else np.nan).dropna()
    diu_idxmin = df_s.groupby(pd.Grouper(freq = 'D')).apply(lambda x: x[name_temperature].idxmin() if len(x) else np.nan).dropna()

    diu_maxm = df_s.loc[(df_s.index.isin(diu_idxmax.values)),name_temperature]
    diu_minm = df_s.loc[(df_s.index.isin(diu_idxmin.values)),name_temperature]    

    # quite slow ... could be programmed more efficiently
    for i in np.arange(-n//2+1,n//2+1):
        if i: 
            diu_maxm = diu_maxm + df_s.shift(i).loc[(df_s.index.isin(diu_idxmax.values)),name_temperature]
            diu_minm = diu_minm + df_s.shift(i).loc[(df_s.index.isin(diu_idxmin.values)),name_temperature]                           

    diu_maxm = diu_maxm * (1./n)
    diu_minm = diu_minm * (1./n)

    mjd_s = df_s['mjd'].resample('D').mean()
    hum_s = df_s[name_humidity].resample('D').mean()        
    
    diu_sm = (diu_maxm.resample('D').mean()-diu_minm.resample('D').mean()).dropna()
    mjd_sm = mjd_s[(mjd_s.index.isin(diu_sm.index))]
    hum_sm = hum_s[(hum_s.index.isin(diu_sm.index))]    

    #print ('AVERAGING DIU M: ',diu_sm)
    #print ('AVERAGING MJD M: ',mjd_sm)
    #print ('AVERAGING HUM M: ',hum_sm)

    return diu_sm[mask], mjd_sm[mask], hum_sm[mask]
    
def average_diurnal_df_rolling(df_s,mask,name_temperature,name_humidity,freq='10min'):

    diu_s = df_s[name_temperature].rolling(freq,min_periods=3).mean()
        
    # Group the df by days and retrieve the id (in this case the *exact* datetime) of the daily maxima and minima
    diu_idxmax = diu_s.groupby(pd.Grouper(freq = 'D')).apply(lambda x: x.idxmax() if len(x) else np.nan).dropna()
    diu_idxmin = diu_s.groupby(pd.Grouper(freq = 'D')).apply(lambda x: x.idxmin() if len(x) else np.nan).dropna()

    diu_maxm = diu_s[(diu_s.index.isin(diu_idxmax.values))]
    diu_minm = diu_s[(diu_s.index.isin(diu_idxmin.values))]    

    mjd_s = df_s['mjd'].resample('D').mean()
    hum_s = df_s[name_humidity].resample('D').mean()        
    
    diu_sm = (diu_maxm.resample('D').mean()-diu_minm.resample('D').mean()).dropna()
    mjd_sm = mjd_s[(mjd_s.index.isin(diu_sm.index))]
    hum_sm = hum_s[(hum_s.index.isin(diu_sm.index))]    

    #print ('AVERAGING DIU M: ',diu_sm)
    #print ('AVERAGING MJD M: ',mjd_sm)
    #print ('AVERAGING HUM M: ',hum_sm)

    return diu_sm[mask], mjd_sm[mask], hum_sm[mask]
    
    
    
