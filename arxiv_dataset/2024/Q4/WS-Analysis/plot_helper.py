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
from scipy.stats import median_abs_deviation as mad
from scipy.optimize import leastsq
import windrose
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
from prettytable import PrettyTable
import itertools
import re
import functools
from likelihood_helper import mu, mu_Haslebacher, Likelihood_Wrapper
from coverage_helper import expected_data_per_day, data_spacing_minutes
from extremes_helper import count_extremes_length_month

# matplotlib year start  (mpl uses mdate in float mode, starting from 1970/01/01, counted in days
ynums = np.array([12053., 12053.+365., 12053+365.+366.])   # 2003, 2004, 2005 
for i in np.arange(0,4):
    ynums = np.concatenate((ynums, np.arange(ynums[-1]+365,ynums[-1]+4*365,365)))  # add 2006, 2007, 2008 
    ynums = np.append(ynums,ynums[-1]+366.)  # add year after leap year
ynums = np.concatenate((ynums, np.arange(ynums[-1]+365,ynums[-1]+3*365,365))) # add next three years
#ynums = np.append(ynums,ynums[-1]+61.) # add two more months 

hum_binning = np.concatenate((np.array([-0.0001,2.00001]),np.arange(3.00001,101.00001)))
months_n = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# from sun_helper, pre-calculated
hours_per_month = np.array([321.3161521218717, 277.47642676874864, 281.9175944700837, 244.27680787816644, 224.05919694900513, 200.7800819389522, 214.7642215155065, 240.15215742588043, 260.72605580091476, 296.2006981037557, 305.57729030773044, 325.59119640290737])
# from sun_helper, with moon taken into account
hours_per_month = np.array([168.94, 162.40, 157.38, 131.99, 116.09, 103.23, 101.94, 110.41, 122.16, 140.92, 149.94, 166.60])
mjd_corrector  = 55000   # used to bring back all mjd to precision required with a float data member
mjd_start_2003 = 52640   # MJD of 1/1/2003
number_days_year = 365.2422 # average number of days per calendar year

#WS_exchange_dates = np.array(['2003-09-27', '2004-03-26', '2005-03-01', '2007-03-28', '2009-03-12', '2010-12-19', '2011-01-26', '2011-06-27', '2012-12-05', '2015-01-27', '2017-04-10', '2019-07-20', '2020-12-11', '2023-01-16'])
WS_exchange_dates = np.array(['2003-09-27', '2004-03-26', '2005-03-01', '2007-03-28', '2009-03-12', '2010-12-19', '2011-01-26', '2011-06-27', '2012-12-05', '2015-01-27', '2017-04-10', '2019-07-20', '2023-01-16'])

gausfunc = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
gauserrf = lambda p, x, y: (y - gausfunc(p, x))

sinfunc  = lambda p, x: p[0]+p[1]*np.sin(2*np.pi/12*(x-p[2]))
sinerrf  = lambda p, x, y: (y - sinfunc(p, x))
# ((cos(x)+1)/2)^2  varies between 0 and 1, with mean of 0.375.
# Note that another representation of this function is (cos(x/2)^4)
cosexpf    = lambda p, x: p[0]-2*0.375*p[1]+2*p[1]*((np.cos(2*np.pi/12*(x-p[2])-np.pi/2)+1)/2)**2  
cosexperrf = lambda p, x, y: (y - cosexpf(p, x))

asinfunc = lambda p, x: p[0]+2*p[1]*np.sin(2*np.pi/12*(x-p[2]))/(2-np.cos(2*np.pi/12*(x-p[2])))
asinerrf = lambda p, x, y: y - asinfunc(p,x)

dsinfunc = lambda p, x: p[0]+p[1]*np.sin(2*np.pi/12*(x-p[2]))+p[3]*np.sin(2*np.pi/12*(x-p[4]))
dsinerrf = lambda p, x, y: y - dsinfunc(p,x)

sinfunc2  = lambda p, x: cosexpf(p[0:3],x)+p[3]*np.sin(2*np.pi/12*(x-p[4]))
                                 #p[0]+(p[1]*(np.sin(2*np.pi/12*(x-p[2]))))
sinerrf2  = lambda p, x, y: (y - sinfunc2(p, x))
# ((cos(x)+1)/2)^2  varies between 0 and 1, with mean of 0.375.
# Note that another representation of this function is (cos(x/2)^4)


def weighted_median(values, weights, quantiles):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def disjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

def plot_mensual(df_sencer,arg, ytit, fullfits=False):

    months = range(1,13)
    Medianes = []
    Sup = []
    Inf = []
    Mins = []
    Maxs = []

    for mes in months: 
        df = df_sencer[df_sencer['Month'] == mes]
        
        df_sup = df[df[arg] > df[arg].median()]
        df_inf = df[df[arg] < df[arg].median()]

        Medianes.append(df[arg].median())
        #Sup.append(df[arg].median() + df_sup[arg].mad())
        #Inf.append(df[arg].median() - df_inf[arg].mad())
        Sup.append(df[arg].median() + (df_sup[arg] - df[arg].median()).median())
        Inf.append(df[arg].median() - (df[arg].median() - df_inf[arg]).median())

        Mins.append(df[arg].min())
        Maxs.append(df[arg].max())

    plt.plot(months, Medianes, marker = 'o', linestyle = 'none', color = 'steelblue', markersize = 7,
             markerfacecolor = 'white', markeredgecolor = 'k')
    plt.fill_between(months, Sup, Inf, alpha = 0.3)

    if (arg == 'temperature' or arg == 'pressure' or 'gradient' in arg): 
        plt.plot(months,Mins,color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k', linestyle='solid')
    plt.plot(months,Maxs,color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k', linestyle='solid')
    #plt.xlabel('Month', fontsize=20)
    plt.ylabel(ytit,    fontsize=26)

    if (arg == 'pressure' or arg == 'PressureHPA'):
        if (fullfits == True):
            init = [ 788., 6.1, 4.53]
            out = leastsq(sinerrf, init, args=(np.array(months), np.array(Medianes)))
            c = out[0]
            x = np.arange(1,12.1,0.1)
            #print ('MEDIANES',Medianes)
            plt.plot(x, sinfunc(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\sin(\frac{2\pi}{12}\cdot(m-%.1f))$ (Haslebacher et al.)' %(c[0],c[1],c[2]),color='r')
            init = [ 788., 6.1, 4.53]
            out = leastsq(cosexperrf, init, args=(np.array(months), np.array(Medianes)))
            c = out[0]
            x = np.arange(1,12.1,0.1)
            #print ('MEDIANES',Medianes)
            plt.plot(x, cosexpf(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\{\ [\cos(\frac{2\pi}{12}\cdot(m-%.1f))+1]^{2}/2-0.75\ \}$ (new formula)' %(c[0],c[1],c[2]),color='orange')
            
        init=  [ 788., 5.6,  4.1,  4.2,  9.6 ]
        out = leastsq(sinerrf2, init, args=(np.array(months), np.array(Medianes)))
        c = out[0]
        print ('Mensual fit result: ', c)
        x = np.arange(1,12.1,0.1)
        #print ('MEDIANES',Medianes)
        plt.plot(x, sinfunc2(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\{\sin(\frac{2\pi}{12}\cdot(m-%.1f))\}^{2}+%.1f\cdot\{\cos(\frac{2\pi}{12}\cdot(m-%.1f))\}$' %(c[0],c[1],c[2],c[3],c[4]),color='violet',lw=3, linestyle='dashed')

        if (fullfits == True):        
            plt.legend(loc='best')
    
    if (arg == 'temperature'):
        init = [ 11., 6.1, 4.53]
        out = leastsq(sinerrf, init, args=(np.array(months), np.array(Medianes)))
        c = out[0]
        x = np.arange(1,12.1,0.1)
        #print ('MEDIANES',Medianes)
        plt.plot(x, sinfunc(out[0], x),label=r'$\widetilde{T}=%.1f^\circ C+%.1f^\circ C\cdot\sin(\frac{2\pi}{12}\cdot(m-%.1f))$ (Eq. 19H)' %(c[0],c[1],c[2]),color='r')
        init = [ 11., 6.1, 4.53]
        out = leastsq(cosexperrf, init, args=(np.array(months), np.array(Medianes)))
        c = out[0]
        x = np.arange(1,12.1,0.1)
        #print ('MEDIANES',Medianes)
        plt.plot(x, cosexpf(out[0], x),label=r'$\widetilde{T}=%.1f^\circ C+%.1f^\circ C\cdot\{\ [\cos(\frac{2\pi}{12}\cdot(m-%.1f))+1]^{2}/2-0.75\ \}$ (Eq. 2)' %(c[0],c[1],c[2]),color='orange')
        plt.ylim([-10.,40.])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),fontsize=20)
    
    if (arg == 'humidity'):
        if (fullfits == True):
            init = [ 31., 9., 8.7]
            out = leastsq(sinerrf, init, args=(np.array(months), np.array(Medianes)))
            c = out[0]
            x = np.arange(1,12.1,0.1)
            #print ('MEDIANES',Medianes)
            plt.plot(x, sinfunc(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\sin(\frac{2\pi}{12}\cdot(m-%.1f))$ (Haslebacher et al.)' %(c[0],c[1],c[2]),color='r')
            init = [ 31., 9., 3.3]
            out = leastsq(cosexperrf, init, args=(np.array(months), np.array(Medianes)))
            c = out[0]
            x = np.arange(1,12.1,0.1)
            #print ('MEDIANES',Medianes)
            plt.plot(x, cosexpf(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\{\ [\cos(\frac{2\pi}{12}\cdot(m-%.1f))+1]^{2}/2-0.75\ \}$ (new formula)' %(c[0],c[1],c[2]),color='orange')
            #init = [ 31., 11., 8., 4., -2.]
            #out = leastsq(dsinerrf, init, args=(np.array(months), np.array(Medianes)))
            #c = out[0]
            #x = np.arange(1,12.1,0.1)
            #plt.plot(x, dsinfunc(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\{\sin(\frac{2\pi}{12}\cdot(m-%.1f))+%.1f\cdot\sin(\frac{2\pi}{12}\cdot(m-%.1f)$' %(c[0],c[1],c[2],c[3],c[4]),color='violet')
        init=  [ 35.86490607,  26.88409446,  3.64347761,  33.79520937,  7.25303265 ]
        out = leastsq(sinerrf2, init, args=(np.array(months), np.array(Medianes)))
        c = out[0]
        x = np.arange(1,12.1,0.1)
        #print ('MEDIANES',Medianes)
        plt.plot(x, sinfunc2(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\{\sin(\frac{2\pi}{12}\cdot(m-%.1f))\}^{2}+%.1f\cdot\{\cos(\frac{2\pi}{12}\cdot(m-%.1f))\}$' %(c[0],c[1],c[2],c[3],c[4]),color='violet',lw=3, linestyle='dashed')
        #init=  [ 35.86490607,  26.88409446,  3.64347761,  33.79520937,  7.25303265 ]        
        #plt.plot(x, sinfunc2(init, x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\{\sin(\frac{2\pi}{12}\cdot(m-%.1f))\}^{2}+%.1f\cdot\{\cos(\frac{2\pi}{12}\cdot(m-%.1f))\}$' %(init[0],init[1],init[2],init[3],init[4]),color='g')        
        if (fullfits == True):
            plt.legend(loc='best')
    
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.xticks(np.arange(1,13), months_n,ha='center')

def get_downtimes(df,arg_dict,year_min=2003,year_max=2023,windgust=200,windgust_waitingtime=20):

    pd.set_option('display.max_columns', 50)        
    downtimes = []
    for year in np.arange(year_min,year_max+1):
        
        dfy = df[df['Year']==year]
        expected_counts = dfy['mjd'].count()
        #print (expected_counts)
        if expected_counts == 0:
            continue

        bad_df  = dfy
        good_df = dfy
        filtered_df = dfy
        masks_bad = []
        
        for key, value in arg_dict.items():
            sign = value[0]
            cut  = value[1]
            if (sign == '>'):
                filtered_df = filtered_df[filtered_df[key]>cut]
                masks_bad.append((bad_df[key]<cut))
            elif (sign == '<'):
                filtered_df = filtered_df[filtered_df[key]<cut]
                masks_bad.append((bad_df[key]>cut))
            else:
                print ('Unrecognized sign of cut: ',sign)
                return None

        filtered_df = filtered_df[filtered_df['temperature']>(filtered_df['DP']+2)]

        masks_bad.append((bad_df['temperature']<(bad_df['DP']+2)))
        masks_bad.append((bad_df['windGust']>windgust))
        
        #bad_df_filtered1 = bad_df[(bad_df['temperature']<(bad_df['DP']+2))]
        #print ('bad_df_filtered1: ',bad_df_filtered1)
        
        #print ('masks_bad', masks_bad)

        # mark all bad periods, including wind gusts 
        mask_bad = disjunction(*masks_bad)
        bad_df_filtered = bad_df[mask_bad]
        #print ('mask_bad: ', mask_bad)
        #print ('bad_df_filtered: ',bad_df_filtered)

        # calculate time differences from one bad event to the next one
        bad_df_filtered['index_s'] = bad_df_filtered.index.to_series().shift(-1)
        bad_df_filtered['Bdiff1']  = bad_df_filtered['index_s']-bad_df_filtered.index.to_series()
        #print ('bad df: ',bad_df_filtered)
        bad_gusts = bad_df_filtered[bad_df_filtered['windGust']>windgust]        
        #print ('gust diffs: ',bad_gusts)

        # now replace the actual Bdiff with the maximum waiting time if that is exceeded
        gust_times = bad_gusts['Bdiff1'].where(bad_gusts['Bdiff1'] < pd.Timedelta(windgust_waitingtime,'m'),pd.Timedelta(windgust_waitingtime,'m'))
        # and remove those Bdiff's that a only 2 or less data spacings, since these would not cause additional time loss 
        gust_times = gust_times[gust_times > pd.Timedelta(2*data_spacing_minutes+1,'m')]
        #print ('gust times: ',gust_times)

        count_loss_from_wind_gusts = gust_times.sum()/pd.Timedelta(data_spacing_minutes,'m')
        #print ('gust times sum: ',gust_times.sum())
        #print ('count loss: ',count_loss_from_wind_gusts)
        
        found_counts = filtered_df['mjd'].count() - count_loss_from_wind_gusts

        if (found_counts < 0):
            found_counts = 0
        
        downtimes.append((expected_counts-found_counts)/expected_counts*100)

    return downtimes

def get_downtime_from_windgust(df_sencer,arg_dict,windgust,coverage_cut=50,sun_alt=-12,year_min=2003,year_max=2023,windgust_waitingtime=20):

    dfn = df_sencer[((df_sencer['coverage'] > coverage_cut) & (df_sencer['sun_alt']<sun_alt))]

    months = range(1,13)

    weights = np.array(hours_per_month)
    sum_weights= weights.sum()

    downtime_mean = 0.
    
    for m in months: 
        downtimes = get_downtimes(dfn[dfn['Month'] == m],arg_dict,year_min,year_max,windgust=windgust,windgust_waitingtime=windgust_waitingtime)
        downtimes_arr = np.array(downtimes)
        downtime_mean += downtimes_arr.mean() * weights[m-1]

    downtime_mean = downtime_mean / sum_weights 
        
    print ('downtime mean for windgust: ',windgust,'=',downtime_mean)
    return downtime_mean

def plot_downtime_vs_windgust(df_sencer,arg_dict,windgustmin,windgustmax,coverage_cut=50,sun_alt=-12,year_min=2003,year_max=2023,windgust_waitingtime=20, color='k'):

    windgusts = np.arange(windgustmin,windgustmax,2)

    downtimes = []
    
    for wg in windgusts:
        downtimes.append(get_downtime_from_windgust(df_sencer,arg_dict,wg,coverage_cut,sun_alt,year_min,year_max,windgust_waitingtime=windgust_waitingtime))

    tit = '{:.0f} min'.format(windgust_waitingtime)
    plt.plot(windgusts, np.array(downtimes), marker = 'o', linestyle = 'none', color = 'steelblue', markersize = 7,
             markerfacecolor = 'white', markeredgecolor = color,label=tit)

    uptime_s = ''
    for key, value in arg_dict.items():
        uptime_s += key+value[0]+' '+str(value[1])+' '+value[2]+'; '
    # replace only the last occurrence of ';'
    uptime_s = uptime_s[::-1].replace(';','', 1)[::-1]
    # replace capital letters by lower case and white space
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', uptime_s)
    uptime_s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()
    #uptime_s = ' '.join(l.lower() for l in re.findall('[A-Z][^A-Z]*', uptime_s))
    uptime_s = uptime_s+r';$T>T_{DP}+2^{\circ}C$'
    uptime_s = uptime_s.replace('humidity','RH')

    plt.title(uptime_s, fontsize=20, loc='right')     
    
    
def plot_mensual_downtime(df_sencer,arg_dict,coverage_cut=50,sun_alt=-12,year_min=2003,year_max=2023, windgust=40):

    dfn = df_sencer[((df_sencer['coverage'] > coverage_cut) & (df_sencer['sun_alt']<sun_alt))]

    months = range(1,13)
    Medianes = []
    Sup = []
    Inf = []
    Mins = []
    Maxs = []

    for mes in months: 

        downtimes = get_downtimes(dfn[dfn['Month'] == mes],arg_dict,year_min,year_max,windgust=windgust)
        downtimes_arr = np.array(downtimes)
        median = np.median(downtimes_arr)
        
        downtimes_sup = downtimes_arr[downtimes_arr > median]
        downtimes_inf = downtimes_arr[downtimes_arr < median]

        Medianes.append(median)
        Sup.append(median + np.median(downtimes_sup - median))
        Inf.append(median - np.median(median - downtimes_inf))

        Mins.append(np.min(downtimes_arr))
        Maxs.append(np.max(downtimes_arr))

    uptime_s = ''
    for key, value in arg_dict.items():
        uptime_s += key+value[0]+' '+str(value[1])+' '+value[2]+'; '
    # replace only the last occurrence of ';'
    uptime_s = uptime_s[::-1].replace(';','', 1)[::-1]
    # replace capital letters by lower case and white space
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', uptime_s)
    uptime_s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()
    #uptime_s = ' '.join(l.lower() for l in re.findall('[A-Z][^A-Z]*', uptime_s))
    uptime_s = uptime_s+r';$T>T_{DP}+2^{\circ}C$'
    uptime_s = uptime_s.replace('humidity','RH')
    uptime_s = uptime_s.replace('km/h ;','km/h; wind gust < {:.0f} km/h;'.format(windgust))
    
    tit = 'Sun alt. < ' + str(sun_alt)

    plt.title(uptime_s, fontsize=20, loc='right')    
    ax = plt.gca()
    ax.plot(months, Medianes, marker = 'o', linestyle = 'none', color = 'steelblue', markersize = 7,
             markerfacecolor = 'white', markeredgecolor = 'k') #,label=tit)
    ax.fill_between(months, Sup, Inf, alpha = 0.3)

    ax.plot(months,Mins,color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k', linestyle='solid')
    ax.plot(months,Maxs,color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k', linestyle='solid')
    ax.set_ylabel('Weather downtime (%)',fontsize=26)

    ax2 = ax.twinx()    
    ax2.plot(months,hours_per_month,color='tomato',marker='.', linestyle='dashed')
    ax2.yaxis.set_tick_params(labelsize=26, color='tomato')
    ax2.set_ylabel('Astron. night time (h/month)', fontsize=26,color='tomato')
    ax2.set_ylim([-180./60,180.])
    ax2.tick_params('y',colors='tomato')
    ax2.grid(None)

    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=22)

    ax.set_ylim([-1.,60.])
    
    plt.xticks(np.arange(1,13), months_n,ha='center')
    #plt.legend(loc='best')


def plot_mensual_diurnal(df, arg, ytit, min_coverage=50, day_coverage=80, is_lombardi=False):

    # centering of time averages, see https://stackoverflow.com/questions/47395119/center-datetimes-of-resampled-time-series
    h_shift = 4
    #print ('BEFORE SHIFT:', df.head(n=10))

    if (is_lombardi):
        df_s = df.shift(h_shift, freq='H')
    else:
        df_s = df.shift(0.5, freq='D')

    #print ('AFTER SHIFT:', df_s.head(n=10))        
    df_s = df_s[df_s['coverage']>min_coverage]
    
    months = range(1,13)
    Medianes = []
    Sup = []
    Inf = []
    Mins = []
    Maxs = []

    for mes in months: 
        df_m = df_s[df_s['Month'] == mes]

        mask_day   = ((df_m.index.hour > 10+h_shift) & (df_m.index.hour < 16+h_shift))
        mask_night = ((df_m.index.hour > -2+h_shift) & (df_m.index.hour < 4+h_shift))        

        if (is_lombardi):
            diurnal_m = df_m.loc[mask_night,arg].resample('D').mean().dropna()-df_m.loc[mask_day,arg].resample('D').mean().dropna()
        else:
            diurnal_m = df_m[arg].resample('D').max().dropna()-df_m[arg].resample('D').min().dropna()
        diurnal_m = diurnal_m[df_m[arg].resample('D').count().dropna() > day_coverage/100*expected_data_per_day]
        
        diurnal_sup = diurnal_m[diurnal_m > diurnal_m.median()]
        diurnal_inf = diurnal_m[diurnal_m < diurnal_m.median()]

        Medianes.append(diurnal_m.median())
        #Sup.append(diurnal_m[arg].median() + diurnal_m_sup[arg].mad())
        #Inf.append(diurnal_m[arg].median() - diurnal_m_inf[arg].mad())
        Sup.append(diurnal_m.median() + (diurnal_sup - diurnal_m.median()).median())
        Inf.append(diurnal_m.median() - (diurnal_m.median() - diurnal_inf).median())

        Mins.append(diurnal_m.min())
        Maxs.append(diurnal_m.max())

    plt.plot(months, Medianes, marker = 'o', linestyle = 'none', color = 'steelblue', markersize = 7,
             markerfacecolor = 'white', markeredgecolor = 'k')
    plt.fill_between(months, Sup, Inf, alpha = 0.3)

    if not is_lombardi:
        plt.plot(months,Mins,color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k', linestyle='solid')
        plt.plot(months,Maxs,color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k', linestyle='solid')
    #plt.xlabel('Month', fontsize=20)
    plt.ylabel(ytit,    fontsize=26)

    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    plt.xticks(np.arange(1,13), months_n,ha='center')

    
def plot_mensual_distributions(dfn,arg, ytit,resolution):

    months = range(1,13)
    Means = []
    Mins = []
    Maxs = []

    min_tot = 9999.;
    max_tot = -9999.;
    
    for mes in months: 
        df = dfn[dfn['Month'] == mes]
        mean = df[arg].mean()

        if (df[arg].max()-mean > max_tot):
            max_tot = df[arg].max()-mean
        
        if (df[arg].min()-mean < min_tot):
            min_tot = df[arg].min()-mean
        
    binning = pd.interval_range(start=min_tot, end=max_tot, freq=resolution) # int(np.rint((arg_max-arg_min)/resolution))

    init  = [8000., 0., 5.]
    if (arg == 'humidity'):
        init  = [8., 0., 15.]
    
    f, (ax, ax2) = plt.subplots(1, 2)

    sigmas = []
    
    for mes in months: 
        df   = dfn[dfn['Month'] == mes]
        mean = df[arg].mean()
        #df['bin'] = pd.cut(df[arg]-mean,bins=binning)
        grouper = df.groupby(pd.Grouper(freq='D'))
        print ('GROUPER:',grouper)
        print ('grouper mean:',grouper[arg].mean())        
        #df_day  = df[arg].resample('D').mean()
        #bins   = df.groupby(df.cut(df[arg].mean()-mean,bins=binning))
        #bins   = df_day.groupby(df_day.cut(df_day[arg]-mean,bins=binning))
        print ('grouper cut:',pd.cut(grouper[arg].mean()-mean,bins=binning))
        bins    = grouper[arg].mean().groupby(pd.cut(grouper[arg].mean()-mean,bins=binning))
        
        #plt.plot(binning.mid,df.groupby('bin').bin.count())
        color = next(ax._get_lines.prop_cycler)['color']
        
        ax.plot(binning.mid,bins.count(),'.',color=color)

        #print ('HERE',binning.mid, ' NOW:',bins['temperature'].count().values)
        
        out = leastsq(gauserrf, init, args=(binning.mid, bins.count().values))
        #out = leastsq(gauserrf, init, args=(binning.mid, bins['temperature'].count().values))
        c = out[0]
        sigmas.append(c[2])
        ax.plot(binning.mid, gausfunc(c, binning.mid),label=r'$%s\ A=%.0f\ \mu=%.2f\ \sigma=%.2f$' %(months_n[mes-1],c[0],c[1],abs(c[2])),color=color)
        
    ax.legend(loc='best',fontsize=12)
    ax.set_xlabel(ytit,fontsize=16)

    print ('MES:',months, ' sigmas:',sigmas)
    
    ax2.plot(months,sigmas,color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k')
    #plt.xlabel('Month', fontsize=20)

    init = [ 4.3, 0.5, 1.9]

    if (arg == 'humidity'):
        init = [ 19.3, -6.5, 3.4]    
    out = leastsq(asinerrf, init, args=(np.array(months), np.array(sigmas)))
    c = out[0]
    x = np.arange(1,12.1,0.1)
    #print ('MEDIANES',Medianes)
    ax2.plot(x, asinfunc(out[0], x),label=r'$\sigma(T)=%.1f+%.1f\cdot2\{\ \sin(\frac{2\pi}{12}\cdot(m-%.1f)) / [2-\cos(\frac{2\pi}{12}\cdot(m-%.1f))]$' %(c[0],c[1],c[2],c[2]),color='orange')
    ax2.legend(loc='best',fontsize=10)

    ax2.set_ylabel(r'$\sigma$ (%s)'%(ytit),fontsize=16)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_tick_params(labelsize=15)
    ax2.xaxis.set_tick_params(labelsize=12)
    plt.xticks(np.arange(1,13), months_n,ha='center')
    

def plot_mensual_wind(df_sencer,fullfits=True):
    months = range(1,13)
    Medianes = []
    Sup = []
    Inf = []
    Mins = []
    Maxs = []
    arg  = 'windSpeedAverage'
    for m in months: 
        df = df_sencer[df_sencer['Month'] == m]
        
        df_sup = df[df[arg] > df[arg].median()]
        df_inf = df[df[arg] < df[arg].median()]

        Medianes.append(df[arg].median())
        #Sup.append(df[arg].median() + df_sup[arg].mad())
        #Inf.append(df[arg].median() - df_inf[arg].mad())
        Sup.append(df[arg].median() + (df_sup[arg] - df[arg].median()).median())
        Inf.append(df[arg].median() - (df[arg].median() - df_inf[arg]).median())

        Mins.append(df['windSpeedCurrent'].min())
        Maxs.append(df['windGust'].max())

        
    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    ax2.plot(months, Medianes, marker = 'o', linestyle = 'none', color = 'steelblue', markersize = 8,
             markerfacecolor = 'white', markeredgecolor = 'k')
    ax2.fill_between(months, Sup, Inf, alpha = 0.3)

    if (fullfits == True):
        init = [ 12., 6.1, 4.53]
        out = leastsq(sinerrf, init, args=(np.array(months), np.array(Medianes)))
        c = out[0]
        x = np.arange(1,12.1,0.1)
        print ('sinfunc',c)
        ax2.plot(x, sinfunc(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\sin(\frac{2\pi}{12}\cdot(m-%.1f))$ (Haslebacher et al.)' %(c[0],c[1],c[2]),color='r')

        init = [ 12., 6.1, 4.53]
        out = leastsq(cosexperrf, init, args=(np.array(months), np.array(Medianes)))
        c = out[0]
        x = np.arange(1,12.1,0.1)
        print ('cosexpf',c)
        ax2.plot(x, cosexpf(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\{\ [\cos(\frac{2\pi}{12}\cdot(m-%.1f))+1]^{2}/2-0.75\ \}$ (new formula)' %(c[0],c[1],c[2]),color='orange')

    init=  [ 12.,  1.2,  3.,  2.3,  10. ]
    out = leastsq(sinerrf2, init, args=(np.array(months), np.array(Medianes)))
    c = out[0]
    x = np.arange(1,12.1,0.1)
    print ('wind sinfunc2',c)
    ax2.plot(x, sinfunc2(out[0], x),label=r'$\widetilde{T}=%.1f+%.1f\cdot\{\sin(\frac{2\pi}{12}\cdot(m-%.1f))\}^{2}+%.1f\cdot\{\cos(\frac{2\pi}{12}\cdot(m-%.1f))\}$' %(c[0],c[1],c[2],c[3],c[4]),color='violet',lw=2, linestyle='dashed')
    #ax2.legend(loc='best',fontsize=10)
    
    ax.plot(months,Maxs,color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k')

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(70., 160.)  # outliers only
    ax2.set_ylim(5., 21.)  # most of the data

    ax.yaxis.set_tick_params(labelsize=18)
    ax2.yaxis.set_tick_params(labelsize=18)
    ax2.xaxis.set_tick_params(labelsize=18)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False, length=0)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.xticks(np.arange(1,13), months_n,ha='center')
    #ax2.set_xlabel('Month', fontsize=30)
    ax2.set_ylabel('Av. wind speed (km/h)', fontsize=15)
    ax.set_ylabel('Max. wind gust (km/h)', fontsize=15)

def plot_mensual_rain(df, arg, hum_threshold=90,
                      color1='steelblue',color2='lightskyblue',color3='white',
                      ax=None, ax2=None):

    months = range(1,13)
    Medianes = []
    Sup = []
    Inf = []
    Mins = []
    Maxs = []

    for m in months: 

        length_rains, _ = count_extremes_length_month(df,arg,m, hum_threshold)

        if (len(length_rains) > 0):
            arr = np.array(length_rains)
            med = np.median(arr)
            
            df_sup = arr[arr > med]
            df_inf = arr[arr < med]
            
            Medianes.append(med)
            Sup.append(med + np.median(df_sup - med))
            Inf.append(med - np.median(med - df_inf))
            
            Mins.append(arr.min())
            Maxs.append(arr.max())
        else:
            Medianes.append(-1.)
            Sup.append(-1.)
            Inf.append(-1.)
            
            Mins.append(-1.)
            Maxs.append(-1.)
            
    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    if ax is None:
        f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    ax2.plot(months, Medianes, marker = 'o', linestyle = 'none', color = color1, markersize = 5,
             markerfacecolor = color3, markeredgecolor = 'k')
    ax2.fill_between(months, Sup, Inf, color=color1, alpha = 0.3)

    ax.plot(months,Maxs,color = color2, marker = '.', markersize = 0,lw=2,linestyle='solid')
    #ax.plot(months,Maxs,color = color2, marker = '.', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k')

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(3.1, 300.)  # outliers only
    ax.set_yscale('log')
    ax2.set_ylim(0., 3.1)  # most of the data

    ax.yaxis.set_tick_params(labelsize=18)
    ax2.yaxis.set_tick_params(labelsize=18)
    ax2.xaxis.set_tick_params(labelsize=18)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False, length=0)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.xticks(np.arange(1,13), months_n,ha='center')
    #ax2.set_xlabel('Month', fontsize=30)
    ax2.set_ylabel('Precipitation duration (h)', fontsize=18)
    ax.set_ylabel('Precipitation duration (h)', fontsize=18)

    return ax,ax2

def plot_mensual_rain_sun(df,arg,hum_threshold=90, lengthcut=3, direction='<', nbins=6, join_months=False, weights=None, plot_tot=False, verbose=True):

    months = range(1,13)
    Medianes = []
    Sup = []
    Inf = []
    Mins = []
    Maxs = []

    bin_edges = np.linspace(-90,90,nbins+1)
    suns = []
    lengths = []

    months_j = ['Jan/Mar-May/Dec', 'Feb', 'Jun/Aug', 'Jul', 'Sep', 'Oct-Nov']
    months_j = ['Jan/Nov/Dec', 'Feb/Mar/Oct', 'Apr/May/Jun', 'Jul/Aug/Sep']
    idx = 0
    
    total_years = (df.index[-1]-df.index[0]).total_seconds()/(3600*24*number_days_year)
    print ('TOTAL YEARS: ',total_years)

    month_names = []
    if weights is None:
        weights = np.ones(13)/total_years
    else:
        weights = weights/total_years
        
    for m in months: 

        if (join_months and m==1):
            length_rains, sun_alts = count_extremes_length_month(df,arg,m, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.array(sun_alts)
            if plot_tot:
                wess = arr * weights[m]
            else:
                wess = np.ones_like(sun_alts) * weights[m]
                
            length_rains, sun_alts = count_extremes_length_month(df,arg,m+10, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.concatenate((sun,np.array(sun_alts)))
            if plot_tot:
                wes = arr * weights[m+10]
            else:
                wes = np.ones_like(sun_alts) * weights[m+10]
            wess = np.concatenate((wess,wes))

            length_rains, sun_alts = count_extremes_length_month(df,arg,m+11, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.concatenate((sun,np.array(sun_alts)))
            if plot_tot:
                wes = arr * weights[m+11]
            else:
                wes = np.ones_like(sun_alts) * weights[m+11]
            wess = np.concatenate((wess,wes))
            
            suns.append(sun)
            lengths.append(wess)

            #if (m != 7) and (m!=8):
            month_names.append(months_j[idx])
        elif (join_months and m==11):
            continue
        elif (join_months and m==12):
            continue
        elif (join_months and m==2):
            length_rains, sun_alts = count_extremes_length_month(df,arg,m, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.array(sun_alts)
            if plot_tot:
                wess = arr * weights[m]
            else:
                wess = np.ones_like(sun_alts) * weights[m]
            
            length_rains, sun_alts = count_extremes_length_month(df,arg,m+1, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.concatenate((sun,np.array(sun_alts)))
            if plot_tot:
                wes = arr * weights[m+1]
            else:
                wes = np.ones_like(sun_alts) * weights[m+1]
            wess = np.concatenate((wess,wes))
            
            length_rains, sun_alts = count_extremes_length_month(df,arg,10, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.concatenate((sun,np.array(sun_alts)))
            if plot_tot:
                wes = arr * weights[10]
            else:
                wes = np.ones_like(sun_alts) * weights[10]
            wess = np.concatenate((wess,wes))
            
            suns.append(sun)
            lengths.append(wess)

            #if (m != 7) and (m!=8):
            month_names.append(months_j[idx])
        elif (join_months and m==3):
            continue
        elif (join_months and m==10):
            continue
        elif (join_months and m==4):
            length_rains, sun_alts = count_extremes_length_month(df,arg,m, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.array(sun_alts)
            if plot_tot:
                wess = arr * weights[m]
            else:
                wess = np.ones_like(sun_alts) * weights[m]

            length_rains, sun_alts = count_extremes_length_month(df,arg,m+1, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.concatenate((sun,np.array(sun_alts)))
            if plot_tot:
                wes = arr * weights[m+1]
            else:
                wes = np.ones_like(sun_alts) * weights[m+1]
            wess = np.concatenate((wess,wes))

            length_rains, sun_alts = count_extremes_length_month(df,arg,m+2, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.concatenate((sun,np.array(sun_alts)))
            if plot_tot:
                wes = arr * weights[m+2]
            else:
                wes = np.ones_like(sun_alts) * weights[m+2]
            wess = np.concatenate((wess,wes))
            
            suns.append(sun)
            lengths.append(wess)

            #if (m != 7) and (m!=8):
            month_names.append(months_j[idx])
        elif (join_months and m==5):
            continue
        elif (join_months and m==6):
            continue
        elif (join_months and m==7):
            length_rains, sun_alts = count_extremes_length_month(df,arg,m, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.array(sun_alts)
            if plot_tot:
                wess = arr * weights[m]
            else:
                wess = np.ones_like(sun_alts) * weights[m]
                
            length_rains, sun_alts = count_extremes_length_month(df,arg,m+1, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.concatenate((sun,np.array(sun_alts)))
            if plot_tot:
                wes = arr * weights[m+1]
            else:
                wes = np.ones_like(sun_alts) * weights[m+1]
            wess = np.concatenate((wess,wes))
            
            length_rains, sun_alts = count_extremes_length_month(df,arg,m+2, hum_threshold,
                                                                 lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.concatenate((sun,np.array(sun_alts)))
            if plot_tot:
                wes = arr * weights[m+2]
            else:
                wes = np.ones_like(sun_alts) * weights[m+2]
            wess = np.concatenate((wess,wes))
            
            suns.append(sun)
            lengths.append(wess)

            #if (m != 7) and (m!=8):
            month_names.append(months_j[idx])
        elif (join_months and m==8):
            continue
        elif (join_months and m==9):
            continue
        else:

            length_rains, sun_alts = count_extremes_length_month(df,arg,m, hum_threshold,
                                                             lengthcut=lengthcut, direction=direction)
            arr = np.array(length_rains)
            sun = np.array(sun_alts)
            if plot_tot:
                wess = arr * weights[m]
            else:
                wess = np.ones_like(sun_alts) * weights[m]

            suns.append(sun)
            lengths.append(wess)
            
            if (join_months):
                month_names.append(months_j[idx])
            else:
                month_names.append(months_n[m-1])

        idx = idx+1

    #ax.yaxis.set_tick_params(labelsize=18)
    #ax2.yaxis.set_tick_params(labelsize=18)
    #ax2.xaxis.set_tick_params(labelsize=18)

    #fig,ax = plt.subplots(1,1)
    #cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #cycle.append(tuple(mcolors.hex2color(mcolors.cnames["crimson"])))
    #cycle.append(tuple(mcolors.hex2color(mcolors.cnames["indigo"])))
    #print ("COLOR CYCLE: ", cycle)
    #plt.hist(suns, bins=nbins, histtype='bar', label=month_names, color=cycle[0:len(month_names)])

    #suns = np.array(suns)
    #tots = np.array(lengths)/total_years
    
    plt.hist(suns, bins=bin_edges, weights=lengths, histtype='bar', label=month_names)
    if (plot_tot):
        plt.ylabel('Hours/year/{:d}º'.format(int(180/nbins)),fontsize=22)
    else:
        #plt.hist(suns, bins=bin_edges, weights=np.ones(suns.shape)/total_years,histtype='bar', label=month_names)
        plt.ylabel('Counts/year/{:d}º'.format(int(180/nbins)),fontsize=22)
    
    plt.xlabel('Sun altitude (º)',fontsize=22)
    plt.legend(loc='best', fontsize=10)
    
def plot_hist(df, arg, resolution, xtit, ytit, sunit, xoff=0.1, loc='left', is_night=False, coverage_cut=50):

    #res = df['months'].resample('M').mean().mask(coverage < coverage_cut)   
    mdays = [0., 31., 28.25, 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.]
    tit = arg

    dfn = df[df['coverage'] > coverage_cut]
    if is_night:
        dfn = dfn[dfn['sun_alt'] < -12.]
        tit = 'night time only'
    else:
        tit = 'full sample'

    if arg == 'humidity':
        arg_min = -0.00000001
        arg_max = 100.00000001
    else:
        arg_min = dfn[arg].min()-resolution
        arg_max = dfn[arg].max()+resolution
    binning = pd.interval_range(start=arg_min, end=arg_max, freq=resolution) # int(np.rint((arg_max-arg_min)/resolution))
    #print ('bins: ', binning)
    binning_fine = pd.interval_range(start=arg_min, end=arg_max, freq=resolution*0.1) # int(np.rint((arg_max-arg_min)/resolution))
    #print ('bins fine: ', binning_fine)

    if arg == 'humidity':
        print (hum_binning)
        binning = pd.IntervalIndex.from_arrays(hum_binning[0:-1],hum_binning[1:])

    Months = range(1,13)
    for month in Months: 
        freq      = pd.cut(dfn.loc[dfn['Month'] == month, arg],bins=binning, ordered=True).value_counts(normalize=True)
        freq_fine = pd.cut(dfn.loc[dfn['Month'] == month, arg],bins=binning_fine, ordered=True).value_counts(normalize=True)
        if (month == 1):
            freq_tot      = freq * mdays[month]/365.25
            freq_tot_fine = freq_fine * mdays[month]/365.25
        else:
            freq_tot = freq_tot + freq * mdays[month]/365.25
            freq_tot_fine = freq_tot_fine + freq_fine * mdays[month]/365.25

    #print ("Bins: ", binning.mid)
    #print ("FREQ: ", freq_tot)
    #print ('SUM:', freq_tot.sum())
    #print ('SUM FINE:', freq_tot_fine.sum())
    #print ('MEAN:', np.average(binning.mid, weights=freq_tot.values))
    mean = np.average(binning_fine.mid,weights=freq_tot_fine.values)
    #print ('MEAN FINE:', mean)
    std  = np.sqrt(np.cov(binning_fine.mid,aweights=freq_tot_fine.values))

    quantiles  = weighted_median(binning_fine.mid,freq_tot_fine.values,[0.05,0.25,0.5,0.75,0.95])

    x = PrettyTable()
    x.add_rows(
        [
            [ "Data points:", str(len(dfn.index)) ],
            [ "Mean:",r' $\bf{{\tt{{{0:>5.1f}{1:}}}}}$'.format(mean, sunit) ],
            [ "Std. Dev:", r' $\bf{{{0:>5.1f}{1:}}}$'.format(std, sunit) ],
            [ "Max:", '{0:>5.1f}{1:}'.format(dfn[arg].max(), sunit) ],
            [ "Min:", '{0:>5.1f}{1:}'.format(dfn[arg].min(), sunit) ],
            [ "5%:", '{0:>5.1f}{1:}'.format(quantiles[0], sunit) ],
            [ "25%:", '{0:>5.1f}{1:}'.format(quantiles[1], sunit) ],
            [ "Median:", r' $\bf{{{0:>5.1f}{1:}}}$'.format(quantiles[2], sunit) ],
            [ "75%:", '{0:>5.1f}{1:}'.format(quantiles[3], sunit) ],
            [ "95%:", '{0:>5.1f}{1:}'.format(quantiles[4], sunit) ]
        ]
    )
    x.align = "l"

    s = f'{"Data points: ":<14}'+ '{0:}'.format(str(len(dfn.index))) + '\n' + \
        r'{0:<13} $\bf{{{1:>5.1f}{2:}}}$'.format("Mean:",mean, sunit) + '\n'+ \
        r'{0:<13} $\bf{{{1:>5.1f}{2:}}}$'.format("Std. dev.:",std, sunit) + '\n'+ \
        r'{0:<13} $\tt{{{1:>5.1f}{2:}}}$'.format("Abs. maximum:", dfn[arg].max(), sunit) + '\n'+ \
        r'{0:<13} $\tt{{{1:>5.1f}{2:}}}$'.format("Abs. minimum:", dfn[arg].min(), sunit) + '\n'+ \
        r'{0:<13} $\tt{{{1:>5.1f}{2:}}}$'.format("5%  quartile:", quantiles[0], sunit) + '\n' + \
        r'{0:<13} $\tt{{{1:>5.1f}{2:}}}$'.format("25% quartile:", quantiles[1], sunit) + '\n'+ \
        r'{0:<13} $\bf{{{1:>5.1f}{2:}}}$'.format("Median:", quantiles[2], sunit) + '\n'+ \
        r'{0:<13} $\tt{{{1:>5.1f}{2:}}}$'.format("75% quartile:", quantiles[3], sunit) + '\n'+ \
        r'{0:<13} $\tt{{{1:>5.1f}{2:}}}$'.format("95% quartile:", quantiles[4], sunit) 
    #'{0:<10} {1:.1f} {2:4}\n'.format("Bin size: ", resolution, sunit) + \

    if (arg in "gradient"):
        s = f'{"Data points: ":<14}'+ '{0:}'.format(str(len(dfn.index))) + '\n' + \
            r'{0:<13} $\bf{{{1:>5.2f}{2:}}}$'.format("Mean:",mean, sunit) + '\n'+ \
            r'{0:<13} $\bf{{{1:>5.2f}{2:}}}$'.format("Std. dev.:",std, sunit) + '\n'+ \
            r'{0:<13} $\tt{{{1:>5.2f}{2:}}}$'.format("Abs. maximum:", dfn[arg].max(), sunit) + '\n'+ \
            r'{0:<13} $\tt{{{1:>5.2f}{2:}}}$'.format("Abs. minimum:", dfn[arg].min(), sunit) + '\n'+ \
            r'{0:<13} $\tt{{{1:>5.2f}{2:}}}$'.format("5%  quartile:", quantiles[0], sunit) + '\n' + \
            r'{0:<13} $\tt{{{1:>5.2f}{2:}}}$'.format("25% quartile:", quantiles[1], sunit) + '\n'+ \
            r'{0:<13} $\bf{{{1:>5.2f}{2:}}}$'.format("Median:", quantiles[2], sunit) + '\n'+ \
            r'{0:<13} $\tt{{{1:>5.2f}{2:}}}$'.format("75% quartile:", quantiles[3], sunit) + '\n'+ \
            r'{0:<13} $\tt{{{1:>5.2f}{2:}}}$'.format("95% quartile:", quantiles[4], sunit) 


    #return binning.mid, freq_tot.values

    #fig, ax1 = plt.subplots()
    #pd.Series(freq_tot.values, index=binning.mid).plot(color='steelblue')
    #plt.plot(binning.mid,freq_tot.values, '-', color='steelblue')
    values = freq_tot.values
    if arg == 'humidity':
        values[0] = values[0]/2   # renormalize for bigger humidity bin width in first bin
    plt.step(binning.mid,values, where='mid', lw=3, color='steelblue')
    plt.xlabel(xtit, fontsize=30)
    plt.ylabel(ytit, fontsize=30)
    plt.title(tit, fontsize=20, loc='right')
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ymax_int = np.ceil(ymax*100)/100.
    ax.set_ylim(-0.005*ymax_int,1.005*ymax_int)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin-xoff*(xmax-xmin),xmax)
    #plt.annotate(s, 0.05, 0.3), 
    
    if arg == 'humidity':
        ax.set_xlim(0.,100.)
    if arg == 'windSpeedCurrent':
        ax.set_xlim(0.,xmax)

    ax2 = ax.twinx()
    #pd.Series(freq_tot_fine.values, index=binning_fine.mid).cumsum().plot(color='k')
    ax2.plot(binning_fine.mid, freq_tot_fine.cumsum().values, '-', lw=3, color='k')
    ax2.set_ylabel('cumulative frequency', fontsize=30)
    ax2.set_ylim(-0.005,1.005)
    ax2.yaxis.set_ticks(np.arange(0., 1.1, 0.1))
    ax2.yaxis.set_tick_params(labelsize=25)
    ax2.grid(axis='y', color='k', linestyle='--', linewidth=0.5)

    print ('TABLE')
    print ( x.get_string(header=False, border=False))

    #mono = {'family' : 'monospace'} 
    mono = {'family' : 'DejaVu Sans Mono'} 
    #mono = {'family' : 'Liberation Mono'} 
    #mono = {'family' : 'FreeMono'} 
    if loc == 'left':
        plt.text(0.03, 0.97, s,# x.get_string(header=False, border=False, rightpaddingwidth=0,leftpaddingwidth=0),
                 fontsize=12, fontdict=mono, #weight='bold',
                 bbox=dict(boxstyle="round", fc="white", ec="gray"),
                 transform=ax.transAxes,
                 horizontalalignment='left', verticalalignment='top')
    elif loc == 'right':
        plt.text(0.68, 0.75, s,# x.get_string(header=False, border=False, rightpaddingwidth=0,leftpaddingwidth=0),
                 fontsize=12, fontdict=mono, #weight='bold',
                 bbox=dict(boxstyle="round", fc="white", ec="gray"),
                 transform=ax.transAxes,
                 horizontalalignment='left', verticalalignment='top')
        
def plot_historic(df, arg, coverage, min_coverage=50):

    t_avg = 'M'
    # centering of time averages, see https://stackoverflow.com/questions/47395119/center-datetimes-of-resampled-time-series
    #df_tmp = df.shift(0.5, freq=t_avg)
    # this solution does not work and results in:
    # TypeError: unsupported operand type(s) for *: 'float' and 'pandas._libs.tslibs.offsets.MonthEnd'
    #
    # Use the alternative offset argument of resample
    df_tmp = df[arg].resample(t_avg, offset='15D')
    plt.plot(df_tmp.max().dropna().mask(coverage < min_coverage), color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k',linestyle='solid')
    plt.plot(df_tmp.median().dropna().mask(coverage < min_coverage), color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k', linestyle='solid')
    plt.fill_between(df_tmp.median().dropna().index, 
                     df_tmp.median().dropna().mask(coverage < min_coverage)+df_tmp.agg(lambda x: mad(x)).dropna().mask(coverage < min_coverage), 
                     df_tmp.median().dropna().mask(coverage < min_coverage)-df_tmp.agg(lambda x: mad(x)).dropna().mask(coverage < min_coverage), 
                     alpha = 0.3, color='#1f77b4')  # require always default color
    if (arg == 'temperature' or arg == 'pressure' or arg == 'PressureHPA' or arg == 'TempInAirDegC' or 'gradient' in arg): 
        plt.plot(df_tmp.min().dropna().mask(coverage < min_coverage), marker = '.', color = 'lightskyblue', markersize = 0, linestyle='solid')
    if arg == 'humidity':
        plt.plot(df_tmp.min().dropna().mask(coverage < min_coverage).replace(0,1).replace(2,1), marker = '.', color = 'lightskyblue', markersize = 0,linestyle='solid')
    #plt.plot(df_tmp.max()*(coverage.array > 50.), color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k')
    #plt.plot(df_tmp.median()*(coverage.array > 50.), color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k')
    #plt.fill_between(df_tmp.median().index, 
    #                 df_tmp.median()*(coverage.array > 50.)+df_tmp.agg(lambda x: mad(x))*(coverage.array > 50.), 
    #                 df_tmp.median()*(coverage.array > 50.)-df_tmp.agg(lambda x: mad(x))*(coverage.array > 50.), 
    #                 alpha = 0.3)
    #if arg == 'temperature': 
    #    plt.plot(df_tmp.min()*(coverage.array > 50.), marker = '.', color = 'lightskyblue', markersize = 0)
 
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(ynums) #np.arange(start, end, 365.))

    if (arg == 'pressure1'):     
        ax.set_xlim([ynums[1],ynums[-1]+365.])
    else:
        ax.set_xlim([ynums[0],ynums[-1]+365.])        
    ax.xaxis.set_major_formatter(mdates.DateFormatter("  %Y"))
    ax.yaxis.set_tick_params(labelsize=16)    
    plt.xticks(ha='left',fontsize=12)
    #print ('AXIS START: ', start, ' END: ', end)
    #print ('ticks: ', ynums)

def plot_historic_fit_results(df, mask, like_wrapper, is_daily=True, day_coverage=85, color='r',is_sigma2=False,is_offset=False,offset=0.):

    if (is_daily):
        df_tmp = df.loc[mask,'mjd'].resample('D').mean().dropna()
        mask_daily = (df.loc[mask,'mjd'].resample('D').count().dropna() > day_coverage/100*expected_data_per_day)        
        df_tmp = df_tmp[mask_daily]
    else:
        df_tmp = df.loc[mask,'mjd'].resample('M').mean().dropna()        
    # see discussion of splitting df into consecutive indices on
    # https://stackoverflow.com/questions/56257329/how-to-split-a-dataframe-based-on-consecutive-index
    
    #print ('DIFF TMP:', df.index.to_series().diff(1))
    if (is_daily):
        list_of_df = np.split(df_tmp,np.flatnonzero(df_tmp.index.to_series().diff(1) != '1 days'))
    else:
        # welcome to the fun of pandas...
        list_of_df = np.split(df_tmp,np.flatnonzero(((df_tmp.index.to_series().diff(1) != '28 days')
                                                     & (df_tmp.index.to_series().diff(1) != '29 days')
                                                     & (df_tmp.index.to_series().diff(1) != '30 days')
                                                     & (df_tmp.index.to_series().diff(1) != '31 days'))))        
        #print ('list: ',list_of_df)


    idxlast = -1
    if is_sigma2:
        idxlast = -3
    if is_offset:
        idxlast = idxlast-1
        
    for df_i in list_of_df:
        df_i_tmp = df_i.add(mjd_corrector-mjd_start_2003).mul(12/number_days_year)
        print ('INDEX:', df_i_tmp.index)
        #print ('MONTH:', df_i_tmp.values.astype(float))
        #print ('VALUES:',like_wrapper.mu_func(like_wrapper.res.x[0:-1],
        #                                      np.array(df_i_tmp.values.astype(float)),
        #                                      verb=False))

        #print ('PARAMS:', temp_lik_median.res.x[0:4])
        plt.plot(df_i_tmp.index,
                 like_wrapper.mu_func(like_wrapper.res.x[0:idxlast],
                                      np.array(df_i_tmp.values.astype(float)),
                                      verb=False) + offset,
                 color=color,linestyle='solid')


            
def plot_profile(df_x, df_y, nbins=50, minimum=None, maximum=None, is_shifted=False, max_color='lightskyblue', markercolor='white', facecolor='C0', label=''):

    #mask = df_x.notnull()
    #df = pd.DataFrame({'x' : df_x[mask] , 'y' : df_y[mask]})

    if (is_shifted):
        df = pd.DataFrame({'x' : df_x , 'y' : df_y.reindex(df_x.index, method='pad')})        
    else:
        df = pd.DataFrame({'x' : df_x , 'y' : df_y})

    print ('DF before mask: ', df.head(n=20))    
    mask = (df['x'].notnull() & df['y'].notnull())
    df = df[mask]

    print ('DF after mask: ', df.head(n=20))
    
    # multiplication needed because later np.digitize has no option 'rightandleft'
    if not minimum:
        minimum = df['x'].min()
    if not maximum:
        maximum = df['x'].max()

    print ('minimum:', minimum)
    print ('maximum:', maximum)
        
    # numerical security 
    minimum = minimum - 0.0001*(maximum-minimum)
    maximum = maximum + 0.0001*(maximum-minimum)    
    
    print ('new minimum:', minimum)
    print ('new maximum:', maximum)
        
    bin_edges = np.linspace(minimum,maximum, nbins+1)  

    print ('edges: ',bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    df['bin'] = np.digitize(df['x'], bins=bin_edges)
    print ('df_x:', df['x'])    
    print ('dfbin:', df['bin'])

    # Now have to (manually) remove empty bins from bin_centers
    binned = df.groupby(['bin'])['bin'].count()
    print ('binned: ', binned)
    print ('bin_centers before:', bin_centers)
    print ('bin_centers after', bin_centers[binned.index-1])    
    bin_centers = bin_centers[binned.index-1]    

    binned = df.groupby('bin')
    #print ('median:',binned['y'].count())    
    #print ('mad:',binned['y'].agg(lambda x: mad(x)))

    plt.plot(bin_centers,binned['y'].agg('max'),
             color = max_color, marker = 'o',markersize=0, linestyle='solid')
    plt.plot(bin_centers,binned['y'].agg('median'),
             color = max_color, marker = 'o', markerfacecolor = markercolor, markersize = 5,
             markeredgecolor = 'k', label=label)
    plt.fill_between(bin_centers,
                     binned['y'].agg('median')+binned['y'].agg(lambda x: mad(x)),
                     binned['y'].agg('median')-binned['y'].agg(lambda x: mad(x)),                     
                     alpha = 0.3, facecolor=facecolor)
    plt.plot(bin_centers,binned['y'].agg('min'),
             color = max_color, marker = 'o', markersize = 0,linestyle='solid')    

    return bin_centers, binned['y'].agg('median')
    
    
def plot_diurnal(df, arg):

    #dff = resample_hourly(df,arg) 
    
    cols = list(mcolors.CSS4_COLORS.keys())

    months = range(1,13)
    for month in months:
        # centering of time averages, see https://stackoverflow.com/questions/47395119/center-datetimes-of-resampled-time-series        
        dff = df[df['Month'] == month].shift(0.5, freq='H')
        #plt.plot(dff['sun_alt'].resample('H').median(),
        plt.plot(dff[arg].resample('H').median().index.hour,
                 dff[arg].resample('H').median(),
                 color = 'steelblue', marker = 'o', markerfacecolor = cols[month], markeredgecolor = 'k')
        #plt.fill_between(dff['sun_alt'].resample('H').median(),
        #                 dff[arg].resample('H').median(),
        #                 +dff[arg].resample('H').agg(lambda x: mad(x)),
        #                 dff[arg].resample('H').median(),
        #                 -dff[arg].resample('H').agg(lambda x: mad(x)),
        #                 alpha=0.3,color=cols[month])


def plot_sunalt(df, arg, ytit, join_months=False):

    #dff = resample_hourly(df,arg) 

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #cols = list(mcolors.CSS4_COLORS.keys())

    months = range(1,13)

    sun_alts = []

    minimum = df['sun_alt'].min()
    maximum = df['sun_alt'].max()

    #mask_daily = (df['mjd'].resample('D').count().dropna() > day_coverage/100*expected_data_per_day)        
    #df_tmp = df[mask_daily]
    
    nbins = 18
    bin_edges = np.linspace(minimum,maximum, nbins+1)      
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    months_j = ['Jan/Mar-May/Dec', 'Feb', 'Jun/Aug', 'Jul', 'Sep', 'Oct-Nov']
    idx = 0
    
    for month in months:
        # centering of time averages, see https://stackoverflow.com/questions/47395119/center-datetimes-of-resampled-time-series        

        if (join_months and month==1):
            dff = df[(df['Month'] == 1) | (df['Month'] == 12) | (df['Month'] == 3) | (df['Month'] == 4) | (df['Month'] == 5)]
        elif (join_months and month==3):
            continue
        elif (join_months and month==4):
            continue
        elif (join_months and month==5):
            continue
        elif (join_months and month==12):
            continue
        elif (join_months and month==10):
            dff = df[(df['Month'] == month) | (df['Month'] == month+1)]
        elif (join_months and month==11):
            continue
        elif (join_months and month==6):
            dff = df[(df['Month'] == month) | (df['Month'] == 8)]
        elif (join_months and month==8):
            continue
        else:
            dff = df[df['Month'] == month]
        
        if (join_months):
            label=months_j[idx]
            col = cols[idx]
        else:
            label=months_n[month-1]
            col = cols[month-1]

        idx = idx+1
        
        #plt.plot(dff['sun_alt'].resample('H').median(),

        sun_meds = []
        arg_meds = []
        arg_infs = []
        arg_sups = []
        
        for i in np.arange(nbins):
            mask = ((dff['sun_alt'] > bin_edges[i]) & (dff['sun_alt'] < bin_edges[i+1]))

            df_arg = dff.loc[mask,arg]
            if (df_arg.count() < 1000):
                continue
            #print ("Count: ", df_arg.count())

            sun_meds.append(dff.loc[mask,'sun_alt'].median())

            
            median = df_arg.median()
            arg_meds.append(median)

            df_sup = df_arg[df_arg > median]
            df_inf = df_arg[df_arg < median]
            
            arg_infs.append(median - (median - df_inf).median())
            arg_sups.append(median + (df_sup-median).median())

            
        plt.plot(sun_meds,arg_meds, 's', color = 'steelblue', marker = 'o', markerfacecolor=col, markeredgecolor = 'k', label=label)
        #plt.fill_between(sun_meds, arg_sups, arg_infs, alpha = 0.3, color=cols[month])        

    plt.legend(loc='upper left', fontsize=18)
    plt.ylabel(ytit,    fontsize=25)
    plt.xlabel('Sun altitude (º)',fontsize=25)
        
def plot_coverage(coverage):
    plt.plot(coverage, color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k')
    #plt.plot(df['index'].resample('M').agg(lambda x: monthrange(x.year,x.month)[1]), color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k')
    
def plot_historic_wind(df, coverage, min_coverage=50):

    t_avg = 'M'
    # centering of time averages, see https://stackoverflow.com/questions/47395119/center-datetimes-of-resampled-time-series
    #df_tmp = df.shift(0.5, freq=t_avg)
    # this solution does not work and results in:
    # TypeError: unsupported operand type(s) for *: 'float' and 'pandas._libs.tslibs.offsets.MonthEnd'
    #
    # Use the alternative offset argument of resample
    df_gust = df['windGust'].resample(t_avg, offset='15D')
    df_av   = df['windSpeedAverage'].resample(t_avg, offset='15D')
    ## centering of time averages, see https://stackoverflow.com/questions/47395119/center-datetimes-of-resampled-time-series        
    ##df_tmp = df.shift(0.5, freq='M')
    plt.plot(df_gust.max().mask(coverage < min_coverage), 
             color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k',linestyle='solid')
    plt.plot(df_av.median().mask(coverage < min_coverage), 
             color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k', linestyle='solid')
    plt.fill_between(df_av.median().index, 
                     df_av.median().mask(coverage < min_coverage)
                     +df_av.agg(lambda x: mad(x)).mask(coverage < min_coverage), 
                     df_av.median().mask(coverage < min_coverage)
                     -df_av.agg(lambda x: mad(x)).mask(coverage < min_coverage), 
                     alpha = 0.3, color='#1f77b4')

    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(ynums) #np.arange(start, end, 365.))
    ax.set_xlim([ynums[0],ynums[-1]+365.])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("  %Y"))
    ax.yaxis.set_tick_params(labelsize=16)    
    plt.xticks(ha='left',fontsize=12)
    
def plot_diurnal_spread(df, arg, coverage, min_coverage=50, day_coverage=80):

    # centering of time averages, see https://stackoverflow.com/questions/47395119/center-datetimes-of-resampled-time-series        
    df_s = df.shift(0.5, freq='D')
    #print ('df_s:',df_s)
    df_s = df_s[df_s['coverage']>min_coverage]
    
    diurnal_s = df_s[arg].resample('D').max().dropna()-df_s[arg].resample('D').min().dropna()
    diurnal_s = diurnal_s[df_s[arg].resample('D').count().dropna() > day_coverage/100*expected_data_per_day]

    print ('DIURNAL:', diurnal_s[diurnal_s.index > '2007-09-01'].head(n=100))

    df_tmp = diurnal_s.resample('M', offset='15D')
    
    #print ('DIURNAL1:', df_tmp)    
    
    plt.plot(df_tmp.max().dropna(), color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k',linestyle='solid')
    plt.plot(df_tmp.median().dropna(), color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k')
    plt.fill_between(df_tmp.median().dropna().index, 
                     df_tmp.median().dropna()+df_tmp.agg(lambda x: mad(x)).dropna(), 
                     df_tmp.median().dropna()-df_tmp.agg(lambda x: mad(x)).dropna(), 
                     alpha = 0.3)
    plt.plot(df_tmp.min().dropna(), marker = '.', color = 'lightskyblue', markersize = 0,linestyle='solid')
 
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(ynums) #np.arange(start, end, 365.))
    ax.set_xlim([ynums[0],ynums[-1]+365.])
    ax.yaxis.set_tick_params(labelsize=16)    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("  %Y"))
    plt.xticks(ha='left',fontsize=12)

def plot_diurnal_fit_results(df_mjd,like_wrapper,df_hum=None,is_daily=True,color='r',p0=0,p1=0):

    # see discussion of splitting df into consecutive indices on
    # https://stackoverflow.com/questions/56257329/how-to-split-a-dataframe-based-on-consecutive-index
    if (is_daily):
        list_of_df = np.split(df_mjd,np.flatnonzero(df_mjd.index.to_series().diff(1) != '1 days'))
        if (df_hum is not None):
            list_of_hum = np.split(df_hum,np.flatnonzero(df_hum.index.to_series().diff(1) != '1 days'))
    else:
        list_of_df = np.split(df_mjd,np.flatnonzero(((df_mjd.index.to_series().diff(1) != '28 days')
                                                     & (df_mjd.index.to_series().diff(1) != '29 days')
                                                     & (df_mjd.index.to_series().diff(1) != '30 days')
                                                     & (df_mjd.index.to_series().diff(1) != '31 days'))))
        if (df_hum is not None):        
            list_of_hum = np.split(df_mjd,np.flatnonzero(((df_hum.index.to_series().diff(1) != '28 days')
                                                         & (df_hum.index.to_series().diff(1) != '29 days')
                                                         & (df_hum.index.to_series().diff(1) != '30 days')
                                                         & (df_hum.index.to_series().diff(1) != '31 days'))))

    #print ('list: ',list_of_df)
    for ii, df_i in np.ndenumerate(list_of_df):
        df_i_tmp = df_i.add(mjd_corrector-mjd_start_2003).mul(12/number_days_year)
        #print ('INDEX:', df_i_tmp.index)
        #print ('MONTH:', df_i_tmp.values)
        #print ('PARAMS:', temp_lik_median.res.x[0:4])
        if (df_hum is None):                
            plt.plot(df_i_tmp.index,
                     like_wrapper.mu_func(like_wrapper.res.x[0:-1],
                                          np.array(df_i_tmp.values.astype(float)),
                                          verb=False),
                     color=color)
        else:
            df_i_hum = list_of_hum[ii[0]]
            plt.plot(df_i_tmp.index,
                     like_wrapper.mu_func(like_wrapper.res.x[0:-1],
                                          np.array(df_i_tmp.values.astype(float)),
                                          np.array(df_i_hum.values.astype(float)),
                                          p0,p1,verb=False),
                     color=color)
                        
            
