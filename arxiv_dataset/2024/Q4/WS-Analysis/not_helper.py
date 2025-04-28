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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import median_abs_deviation as mad
from coverage_helper import expected_data_per_day
from plot_helper import ynums, WS_exchange_dates



def not_read(not_file, start_date='2003-01-01 00:00:01'):

    dateparse = lambda d,t: datetime.strptime(d+" "+t, '%Y-%m-%d %H:%M:%S')
    df_not = pd.read_csv(not_file,sep='\s+',parse_dates={'datetime' : ['Date','Time']},date_parser=dateparse,index_col='datetime')
    #print('DUPLICATES NAOI: ',df_naoi[(df_naoi['diff1']>pd.Timedelta(0,'m')) & (df['diff1']<=pd.Timedelta(1,'m'))]
    df_not = df_not[(df_not.index > start_date)]
    print (df_not)

    return df_not
    
def not_correlate(df_not, df_x, color='r'):

    s1, s2 = df_x.align(df_not, join='inner', axis=0)#.dropna()
    print ("S1_d",s1.values,"S2_d",s2.values)
    a = np.squeeze(np.array(s2.values))
    b = np.array(s1.values)
    # necessary because of possible gaps in NAOI
    b = b[~np.isnan(a)]
    a = a[~np.isnan(a)]
    #print ('nans:', a[np.isnan(a)],' infs: ',a[np.isinf(a)], ' i: ',b[np.isinf(b)])
    r = pearsonr(a,b)[0]
    s = spearmanr(a,b)[0]
    k = kendalltau(a,b)[0]
    plt.plot(a,b,'.',color=color,label=rf'Pearson $r$:{r:.3f}, Spearman $\rho$:{s:.3f}, Kenall $\tau$:{k:.3f}')


def not_profile(df_not, df_x, nbins=50,minimum=None, maximum=None):

    #mask = df_not.notnull()
    
    df = pd.DataFrame({'x' : df_not , 'y' : df_x})

    #print ('NOT HERE: ',df.head())

    mask = (df['x'].notnull() & (df['y'].notnull()))
    df = df[mask]
    
    # multiplication needed because later np.digitize has no option 'rightandleft'
    if not minimum:
        minimum = df_not.min()#.values.astype(float)
    if not maximum:
        maximum = df_not.max()#.values.astype(float)

    # numerical security 
    minimum = minimum - 0.0001*(maximum-minimum)
    maximum = maximum + 0.0001*(maximum-minimum)    

    bin_edges = np.squeeze(np.linspace(minimum,maximum, nbins+1))

    #bin_edges = bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df['bin'] = np.digitize(df['x'], bins=bin_edges)
    binned = df.groupby('bin')

    mask = binned['y'].agg('median').notnull()
    
    plt.plot(bin_centers,binned['y'].agg('max'),
             color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k')
    plt.plot(bin_centers,binned['y'].agg('median'),
             color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k')
    plt.fill_between(bin_centers,
                     binned['y'].agg('median')+binned['y'].agg(lambda x: mad(x)),
                     binned['y'].agg('median')-binned['y'].agg(lambda x: mad(x)),                     
                     alpha = 0.3)
    plt.plot(bin_centers,binned['y'].agg('min'),
             color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k')    

    plt.xlabel('NOT')
    plt.legend(loc='best')


def not_profile_time(df_not, df_x, arg_not, arg, day_coverage=85, print_threshold=None, plot_exchange_dates=True):

    #mask = df_not.notnull()
    df_tmp = df_x['mjd'].resample('D',offset='12h').mean().dropna()
    mask_daily = (df_x['mjd'].resample('D',offset='12h').count().dropna() > day_coverage/100*expected_data_per_day)
    df_tmp = df_tmp[mask_daily]

    df = pd.DataFrame({'x' : df_not[arg_not].resample('D',offset='12h').median(), 'y' : df_x[arg].resample('D',offset='12h').median()})

    #print ('NOT HERE: ',df.head())

    mask = (df['x'].notnull() & (df['y'].notnull()))
    df = df[mask]

    df_tmp = (df['y']-df['x']).resample('M', offset='15D')
    
    if print_threshold is not None:
        print ('PROFILE TIME with threshold ',print_threshold)
        print (df_tmp.max()[df_tmp.max().values > print_threshold].head(n=200))
        idxs = df_tmp.max()[df_tmp.max().values > print_threshold].index
        for idx in idxs:
            mask = ((df.index > idx) & (df.index < (idx + pd.DateOffset(1))))
            print ('NOT: ',df.loc[mask,'x'])
            print ('MAGIC: ',df.loc[mask,'y'])
        

    plt.plot(df_tmp.max(), color = 'lightskyblue', marker = 'o', markersize = 0, markerfacecolor = 'white', markeredgecolor = 'k')
    plt.plot(df_tmp.median(), color = 'steelblue', marker = 'o', markerfacecolor = 'white', markeredgecolor = 'k')
    plt.fill_between(df_tmp.median().index, 
                     df_tmp.median()+df_tmp.agg(lambda x: mad(x)), 
                     df_tmp.median()-df_tmp.agg(lambda x: mad(x)), 
                     alpha = 0.3)
    plt.plot(df_tmp.min(), marker = '.', color = 'lightskyblue', markersize = 0)

    ymin, ymax = plt.gca().get_ylim()
    plt.vlines(pd.to_datetime(WS_exchange_dates),ymin,ymax, colors='r')
    
    ax = plt.gca()
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(ynums) #np.arange(start, end, 365.))
    ax.set_xlim([ynums[0],ynums[-1]+365.])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("  %Y"))
    ax.yaxis.set_tick_params(labelsize=16)    
    plt.xticks(ha='left',fontsize=12)
    
