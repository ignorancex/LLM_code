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
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import median_abs_deviation as mad

def naoi_read(naoi_file, start_date='2003-01-01 00:00:01'):

    df_naoi = pd.read_csv(naoi_file,parse_dates={'date' : ['year','month','day']},index_col='date')
    #print('DUPLICATES NAOI: ',df_naoi[(df_naoi['diff1']>pd.Timedelta(0,'m')) & (df['diff1']<=pd.Timedelta(1,'m'))]
    df_naoi = df_naoi[(df_naoi.index > start_date)]
    print (df_naoi)

    return df_naoi
    
def naoi_correlate(df_naoi, df_x, color='r'):

    s1, s2 = df_x[df_x.notnull()].align(df_naoi[df_naoi.notnull()], join='inner', axis=0)#.dropna()
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


def naoi_profile(df_naoi, df_x, nbins=50,arg='nao_index_cdas',minimum=None, maximum=None):

    mask = df_naoi[arg].notnull()
    
    df = pd.DataFrame({'x' : df_naoi.loc[mask,arg] , 'y' : df_x[mask]})

    # multiplication needed because later np.digitize has no option 'rightandleft'
    if not minimum:
        minimum = df_naoi[arg].min()
    if not maximum:
        maximum = df_naoi[arg].max()

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

    plt.xlabel('NAO index')
    plt.legend(loc='best')
