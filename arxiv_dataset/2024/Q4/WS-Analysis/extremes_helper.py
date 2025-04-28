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

dict_gaps = {}
new_gaps_counter = np.zeros(13)
add_gaps_counter = np.zeros(13)  # 13 to use month as index
add_gaps_counter[3]   = 25       # 2004-03-25 15:01:20   25 days 06:57:30
new_gaps_counter[3]   = 25
add_gaps_counter[10]  = 10
new_gaps_counter[10]  = 10
add_gaps_counter[11]  = 30
new_gaps_counter[11]  = 30       # 122 d
add_gaps_counter[12]  = 31       # 92 d
new_gaps_counter[12]  = 31       # 92 d
#new_gaps_counter[10] += 9        # 131 d
dict_gaps['2004'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
add_gaps_counter[1]  += 31       # 61 d
new_gaps_counter[1]  += 31       # 61 d
add_gaps_counter[2]  += 29       # 30 d
new_gaps_counter[2]  += 29       # 30 d
dict_gaps['2005'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2006'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
add_gaps_counter[3]  += 11      # 2005-03-01 15:03:50   131 days 17:36:10
new_gaps_counter[3]  += 11      # 2005-03-01 15:03:50   131 days 17:36:10
add_gaps_counter[5]  += 18       # 32 d
new_gaps_counter[5]  += 18       # 32 d
add_gaps_counter[6]  += 14       # 2007-06-14 22:43:40   32 days 19:39:50
new_gaps_counter[6]  += 14       # 2007-06-14 22:43:40   32 days 19:39:50
add_gaps_counter[8]  += 31       # 61 d
new_gaps_counter[8]  += 31       # 61 d
add_gaps_counter[9]  += 30       # 2007-10-01 00:00:14   61 days 00:01:06
new_gaps_counter[9]  += 30       # 2007-10-01 00:00:14   61 days 00:01:06
dict_gaps['2007'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
add_gaps_counter[9]  += 13      # 2007-10-01 00:00:14   61 days 00:01:06
new_gaps_counter[9]  += 13      # 2007-10-01 00:00:14   61 days 00:01:06
add_gaps_counter[10] += 10      # 2007-10-01 00:00:14   61 days 00:01:06
new_gaps_counter[10] += 10      # 2007-10-01 00:00:14   61 days 00:01:06
dict_gaps['2008'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
add_gaps_counter[2]  += 22
new_gaps_counter[2]  += 22
add_gaps_counter[3]  += 11      # 2007-10-01 00:00:14   61 days 00:01:06
new_gaps_counter[3]  += 11      # 2007-10-01 00:00:14   61 days 00:01:06
add_gaps_counter[9]  += 14      # 2007-10-01 00:00:14   61 days 00:01:06
new_gaps_counter[9]  += 14      # 2007-10-01 00:00:14   61 days 00:01:06
dict_gaps['2009'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2010'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
add_gaps_counter[6]  += 14      # 2007-10-01 00:00:14   61 days 00:01:06
new_gaps_counter[6]  += 14      # 2007-10-01 00:00:14   61 days 00:01:06
dict_gaps['2011'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
add_gaps_counter[11] += 28      # 2007-10-01 00:00:14   61 days 00:01:06
new_gaps_counter[11] += 28      # 2007-10-01 00:00:14   61 days 00:01:06
dict_gaps['2012'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2013'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2014'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2015'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2016'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2017'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2018'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
add_gaps_counter[7]  += 20      # Year:  2019-07-20 00:00:01   21 days 00:02:00
new_gaps_counter[7]  += 20      # Year:  2019-07-20 00:00:01   21 days 00:02:00
dict_gaps['2019'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2020'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2021'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
add_gaps_counter[4]  += 22       # Year:  2022-05-01 00:00:01   22 days 00:02:00
new_gaps_counter[4]  += 22       # Year:  2022-05-01 00:00:01   22 days 00:02:00
dict_gaps['2022'] = new_gaps_counter
new_gaps_counter = np.zeros(13)
dict_gaps['2023'] = new_gaps_counter

gaps_counter_threshold = 5  # days between automatic counter and explicit one (from dict_gaps)

mdays = [0., 31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.]

def count_extremes(df, what, year, cut):

    df = df[(df['Year'] == year) & (df[what] > cut)]
    df = df[what].resample('D').max().dropna()
    dates = df.index
    date_int = np.array([d.toordinal() for d in dates])
    date_int = np.insert(date_int,0,0)
    
    huracans = 0
    for d in (range(len(date_int)-1)):
        if (date_int[d+1]-date_int[d]) != 1: 
            huracans += 1
    
    return huracans

def count_extremes_length(df, what, year, cut, verbose=True):

    mask = ((df['Year'] == year) & (df[what] > cut))
    df_tmp = df.loc[mask,'mjd']
    list_of_df = np.split(df_tmp,np.flatnonzero(df_tmp.index.to_series().diff(1) > pd.Timedelta(7,'m')))

    tdiffs_h = []

    for df_i in list_of_df:
        if (verbose):
            print (df_i.index)
        if (len(df_i.index)<2):
            continue
        t_0 = df_i.index[0]
        t_1 = df_i.index[-1]

        tdiff = t_1 - t_0
        tdiff_h = tdiff.days*24 + tdiff.seconds/3600.
        if (verbose or tdiff_h>100):
            print ("t_0: ", t_0, " t_1: ", t_1, " tdiff: ", tdiff,
                   " tdiff2: ",tdiff/np.timedelta64(1, 'D'),
                   " tdiff D: ", tdiff.days, " tdiff seconds: ", tdiff.seconds,
                   " tdiff hours: ", tdiff_h)

        tdiffs_h.append(tdiff_h)

    return tdiffs_h
            
def count_extremes_length_month(df, what, month, cut, lengthcut=0, direction='>', verbose=False):

    mask = ((df['Month'] == month) & (df[what] > cut))
    df_tmp = df.loc[mask,'sun_alt']
    list_of_df = np.split(df_tmp,np.flatnonzero(df_tmp.index.to_series().diff(1) > pd.Timedelta(7,'m')))

    tdiffs_h = []
    sun_alt  = []

    for df_i in list_of_df:
        if (verbose):
            print (df_i.index)
        if (len(df_i.index)<2):
            continue
        t_0 = df_i.index[0]
        t_1 = df_i.index[-1]

        tdiff = t_1 - t_0
        tdiff_h = tdiff.days*24 + tdiff.seconds/3600.
        if (verbose or tdiff_h>100):
            print ("t_0: ", t_0, " t_1: ", t_1, " tdiff: ", tdiff,
                   " tdiff2: ",tdiff/np.timedelta64(1, 'D'),
                   " tdiff D: ", tdiff.days, " tdiff seconds: ", tdiff.seconds,
                   " tdiff hours: ", tdiff_h)

        if (tdiff_h > lengthcut) and (direction == '>'):
            tdiffs_h.append(tdiff_h)
            #sun_alt.append(0.5*(df_i.values[0]+df_i.values[-1]))
            sun_alt.append(df_i.values[int(len(df_i)*0.5)])            
        if (tdiff_h < lengthcut) and (direction == '<'):
            tdiffs_h.append(tdiff_h)
            #sun_alt.append(0.5*(df_i.values[0]+df_i.values[-1]))
            sun_alt.append(df_i.values[int(len(df_i)*0.5)])
        
    return tdiffs_h, sun_alt

def count_extremes_with_lengthcut(df, what, year, cut, lengthcut=3, direction='>', verbose=False):

    mask = ((df['Year'] == year) & (df[what] > cut))
    df_tmp = df.loc[mask,'mjd']
    list_of_df = np.split(df_tmp,np.flatnonzero(df_tmp.index.to_series().diff(1) > pd.Timedelta(7,'m')))

    rains = 0

    for df_i in list_of_df:
        if (verbose):
            print (df_i.index)
        if (len(df_i.index)<2):
            continue
        t_0 = df_i.index[0]
        t_1 = df_i.index[-1]

        tdiff = t_1 - t_0
        tdiff_h = tdiff.days*24 + tdiff.seconds/3600.
        if (verbose or tdiff_h>100):
            print ("t_0: ", t_0, " t_1: ", t_1, " tdiff: ", tdiff,
                   " tdiff2: ",tdiff/np.timedelta64(1, 'D'),
                   " tdiff D: ", tdiff.days, " tdiff seconds: ", tdiff.seconds,
                   " tdiff hours: ", tdiff_h)

        if (tdiff_h > lengthcut) and (direction == '>'):
          rains += 1  
        if (tdiff_h < lengthcut) and (direction == '<'):
          rains += 1  

    return rains
            
def count_extremes_month(df, what, month, year, cut):
    df = df[(df['Year'] == year) & (df['Month'] == month) & (df[what] > cut)]
    df = df[what].resample('D').max()
    df = df.dropna()
    dates = df.index
    date_int = np.array([d.toordinal() for d in dates])
    date_int = np.insert(date_int,0,0)
    
    huracans = 0
    for d in (range(len(date_int)-1)):
        if (date_int[d+1]-date_int[d]) != 1: 
            huracans += 1
    
    return huracans

def count_extremes_month_with_lengthcut(df, what, month, year, cut, lengthcut=3, direction='>', verbose=False):

    mask = ((df['Year'] == year) & (df['Month'] == month) & (df[what] > cut))
    df_tmp = df.loc[mask,'mjd']
    list_of_df = np.split(df_tmp,np.flatnonzero(df_tmp.index.to_series().diff(1) > pd.Timedelta(7,'m')))

    rains = 0

    for df_i in list_of_df:
        if (verbose):
            print (df_i.index)
        if (len(df_i.index)<2):
            continue
        t_0 = df_i.index[0]
        t_1 = df_i.index[-1]

        tdiff = t_1 - t_0
        tdiff_h = tdiff.days*24 + tdiff.seconds/3600.
        if (verbose or tdiff_h>100):
            print ("t_0: ", t_0, " t_1: ", t_1, " tdiff: ", tdiff,
                   " tdiff2: ",tdiff/np.timedelta64(1, 'D'),
                   " tdiff D: ", tdiff.days, " tdiff seconds: ", tdiff.seconds,
                   " tdiff hours: ", tdiff_h)

        if (tdiff_h > lengthcut) and (direction == '>'):
          rains += 1  
        if (tdiff_h < lengthcut) and (direction == '<'):
          rains += 1  

    return rains
            

def extremes(df, what, cut, year_min=2004, year_max=2023):
    huracans = []
    years = np.arange(year_min,year_max+1)
    for y in years:
        huracans.append(count_extremes(df, what, y, cut))
    return huracans

def extremes_with_lengthcut(df, what, cut, year_min=2004, year_max=2023, lengthcut=3, direction='>'):
    rains = []
    years = np.arange(year_min,year_max+1)
    for y in years:
        rains.append(count_extremes_with_lengthcut(df, what, y, cut,lengthcut,direction))
    return rains

    
def extremes_month(df, what, cut, month, year_min=2004, year_max=2023):
    huracans = 0
    years = np.arange(year_min,year_max+1)
    for a in years:
        huracans += count_extremes_month(df, what, month, a, cut)
    return huracans

def extremes_month_with_lengthcut(df, what, cut, month, year_min=2004, year_max=2023, lengthcut=3, direction='>'):
    rains = 0
    years = np.arange(year_min,year_max+1)
    for a in years:
        rains += count_extremes_month_with_lengthcut(df, what, month, a, cut, lengthcut, direction)
    return rains

def count_gaps(df, year):
    df = df[(df['Year'] == year) & (df['diff1'].dt.total_seconds() < gaps_counter_threshold*24*3600.)]
    return df['diff1'].dt.total_seconds().sum()/3600./24.

def count_gaps_year_month(df, year, month, is_rain=False, is_20to22=True):
    df = df[(df['Year'] == year) & (df['Month'] == month)]
    from_dict = dict_gaps[str(year)]

    if (is_rain):
        # additional data gaps due to humidity sensor:
        if not is_20to22: 
            if ((year == 2020) or (year==2021) or (year==2022)):
                from_dict[month] = 30.5   # FIXME
            if ((year == 2023) and (month==1)):
                from_dict[month] = 15.5  
        if ((year == 2014) and (month==11)):
            from_dict[month] = 7  
        if ((year == 2014) and (month==12)):
            from_dict[month] = 31  
        if ((year == 2015) and (month==1)):
            from_dict[month] = 27.5
    
    print ('from dict year: ', year, ' month: ',month,' diff1: ', df['diff1'].dt.total_seconds().sum()/3600./24., ' dict: ', from_dict[month])
    return df['diff1'].dt.total_seconds().sum()/3600./24. + from_dict[month]

def count_gaps_month(df, month, year_min=2003, year_max=2023, gap_min_hours=24, is_rain=False, is_20to22=True, verbose=False):

    df = df[df['diff1']>pd.Timedelta(gap_min_hours,'hours')]
    df = df[df['Month'] == month] 

    if (verbose):
        print ('Month: ', df['diff1'].head(n=200))

    gaps = 0
    years = np.arange(year_min,year_max+1)
    for y in years:
        if (is_rain and (y == 2020) or (y==2021) or (y==2022)):
            continue
        g = count_gaps(df, y)
        if (g>gaps_counter_threshold):   # treat these separately in add_gaps_counters
            continue
        if (verbose):
            print ('GAPS year: ', a, ' Gaps: ', g)
        gaps += g
    return gaps + add_gaps_counter[month]

def weights_gaps_months(df, year_min=2003, year_max=2023, gap_min_hours=24, is_rain=False, is_20to22=True, verbose=False):

    gaps = np.zeros(13)
    for m in np.arange(13):
        gaps[m] = count_gaps_month(df, m, year_min, year_max, gap_min_hours, is_rain, is_20to22)

    days_tot = np.zeros(13)
    for y in np.arange(year_min, year_max+1):
        if (is_rain and (y==2020) or (y==2021) or (y==2022)):
            continue
        for m in np.arange(1,13):
            if m==2 and not y%4:
                days_tot[m] += 29
            else:
                days_tot[m] += mdays[m]            

    return days_tot/(days_tot-gaps)

def weight_for_year(df, year, storm_weights, is_rain=False, is_20to22=True):

    df = df[(df['diff1']>pd.Timedelta(1,'day')) & (df['diff1']<pd.Timedelta(gaps_counter_threshold,'day'))]

    weight_tot = 0.
    for m in np.arange(1,13):
        gaps_days = count_gaps_year_month(df, year, m, is_rain, is_20to22)
        print ('Gaps for year ',year,' month ',m,' =', gaps_days)
        if m==2 and not year%4:
            weight_tot += (29 - gaps_days)/29. * storm_weights[m]
        else:
            weight_tot += (mdays[m] - gaps_days)/mdays[m] * storm_weights[m]

    if (is_rain and not is_20to22 and ((year == 2020) or (year==2021) or (year==2022))):
        weight_tot = 0.
            
    return weight_tot

def calc_storm_weights(df, arg, cut, year_start, year_end):

    Storms_month = []
    Storms_month.append(0)  # dummy zero in order to be able to call with month as index later
    storms_tot = 0

    months = np.arange(1,13)
    for m in months:
        storms = extremes_month(df,arg, cut, m, year_start, year_end)
        Storms_month.append(storms)
        storms_tot += storms

    for m in months:
        if (storms_tot == 0.):
            Storms_month[m] = 0.
        else:
            Storms_month[m] = Storms_month[m]/storms_tot

    return Storms_month

def calc_rain_weights(df, arg, cut, year_start, year_end, lengthcut=3, direction='>'):

    rains_month = []
    rains_month.append(0)  # dummy zero in order to be able to call with month as index later
    rains_tot = 0

    months = np.arange(1,13)
    for m in months:
        rains = extremes_month_with_lengthcut(df,arg, cut, m, year_start, year_end,lengthcut, direction)
        rains_month.append(rains)
        rains_tot += rains

    for m in months:
        if (rains_tot == 0.):
            rains_month[m] = 0.
        else:
            rains_month[m] = rains_month[m]/rains_tot

    return rains_month

def calc_all_weights(df, arg, cut, year_start, year_end, is_rain=False,lengthcut=3, direction='>', is_20to22=True):

    if (is_rain):
        weights = calc_rain_weights(df,arg,cut,year_start,year_end, lengthcut, direction)
    else:
        weights = calc_storm_weights(df,arg,cut,year_start,year_end)

    all_weights = []
    for year in np.arange(year_start,year_end+1):
        all_weights.append(weight_for_year(df, year, weights,is_rain,is_20to22))

    return all_weights

