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
import re
import windrose
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from scipy.stats import exponweib
from scipy.optimize import fsolve

def winddiraverage(x):
    ph = x/180*np.pi
    Ds = np.sin(ph)
    Dc = np.cos(ph)

    wd0 = 180/np.pi*np.arctan2(Ds.mean(),Dc.mean())
    mean_wd = np.where(wd0<0,wd0+360,wd0)
    return mean_wd  

def weibull(v,k,c):
    '''Weibull distribution for wind speed v with shape parameter k and scale parameter c'''
    return (k / c) * (v / c)**(k-1) * np.exp(-(v/c)**k)

def _propagate_error(out, fct, x, chi2=1.,epsilon=1e-3, debug=False):
    """Evaluate error for a given function with uncertainty propagation.
    
        Parameters
        ----------
        fct : Function to estimate the error.
        epsilon : float
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error.
        **kwargs : dict
            Keyword arguments.
    
        Returns
        -------
        f_cov : Error of the given function.
        """
    pars = out[0]
    cov  = out[1]
    print ('COV=',cov)
    eps  = np.sqrt(np.diag(cov)) * epsilon

    n, f_0 = len(pars), fct(x,*pars)

    if debug is True:
        print ('cov: ', cov)
        print ('diag:: ', np.diag(cov))    
        print ('eps: ', eps)
        print ('f_0: ',f_0)
    shape = (n, len(np.atleast_1d(f_0)))
    df_dp = np.zeros(shape)

    for idx, par in enumerate(pars):
        if eps[idx] == 0:
            continue

        pars[idx] += eps[idx]
        df = fct(x,*pars) - f_0
        df_dp[idx] = df * np.sqrt(chi2) / eps[idx]

        if debug is True:
            print ('idx: ', idx)
            print ('pars: ', pars[idx])
            print ('df: ', df)
            print ('df_dp: ', df_dp)        
        pars[idx] -= eps[idx]

    f_cov = df_dp.T @ cov @ df_dp

    f_err = np.sqrt(np.diagonal(f_cov))

    if debug is True:    
        print ('f_cov: ', f_cov)
        print ('diag: ', np.diagonal(f_cov))    
        print ('f_err rel.:', f_err/f_0)
    
    return f_err


def _k_from_v_k(v,k,weights=None):
    #df = df[(~df.isnull()) & (df>0))]
    if weights is None:
        k = 1./ ( ((v**k*np.log(v)).sum()/(v**k).sum()) - (np.log(v)).sum()/len(v) )
    else:
        sum1 = (v**k*np.log(v)*weights).sum()
        sum2 = (v**k*weights).sum()
        sum3 = (np.log(v)*weights).sum()
        sum4 = weights.sum()
        #print ('sum1', sum1, 'sum2', sum2, 'sum3', sum3, 'sum4', sum4)
        k = 1./ ( sum1/sum2 - sum3/sum4 )        
    #print ('k try: ', k)
    return k

# estimate Weibull wind parameters from Likelihood solution of Stevens and Smulders, Wind Engineering 3 (1979) 132 
def estimate_k_c_from_df(df,arg):

    mask = ((df[arg]>0) & (~df[arg].isnull()))
    s = df.loc[mask,arg]
    v = s.values
    
    k_est = fsolve(lambda k: k - _k_from_v_k(v,k), (2)) # initialize k with 2, according to Stevens & Smulders

    print ('k_est: ',k_est)

    c_est = np.power((v**k_est).sum()/len(v),1/k_est)
    
    print ('c_est', c_est) 
    return k_est, c_est


# estimate Weibull wind parameters from Likelihood solution of Stevens and Smulders, Wind Engineering 3 (1979) 132
# from a distribution of v with weight w of each entry
def estimate_k_c_from_distribution(v,w):

    mask = np.where((~np.isnan(w)) & (~np.isinf(w)) & (v>0)) 
    v = v[mask]
    w = w[mask]    

    #print ('v=',v, 'w=',w)
    
    k_est = fsolve(lambda k: k - _k_from_v_k(v,k,w), (2)) # initialize k with 2, according to Stevens & Smulders

    print ('k_est: ',k_est)

    c_est = np.power((v**k_est*w).sum()/w.sum(),1/k_est)
    
    print ('c_est', c_est) 
    return k_est, c_est

def exp_decay(v,a,c):
    return a*np.exp(-v/c)

# exp_decay integrated from v_lim to infinity
def exp_decay_integrated(v_lim,a,c):
    return a*c*np.exp(-v_lim/c)

# solve exp_decay_integrated == recurrence_time (in years)
def find_recurrence(a,c,recurrence_time):
    # exp(-v_lim/c) = T/(a*c)
    # v_lim = -c * ln (T/(a*c))
    return -c * np.log(1./(recurrence_time*a*c))
    
def _mtch(s,form='.0f'):
    fs = '{:s}'.format(form)
    fs = '{:'+fs+'}'
    print (s.group(0), ' form=',form,' fs=',fs)
    f = float(s.group(0))
    return fs.format(f)

def plot_windrose(df,arg_speed,arg_dir,tit,pw=0.5,leg_tit="Wind Speed (km/h)",form='.0f',min_scale=1):

    dff = df[(df[arg_speed].notnull() & (df[arg_dir].notnull()))]
    
    ax = windrose.WindroseAxes.from_ax()
    # now plot the probabilities to find a given wind direction, with the wind speed as color code,
    # weight the higher speeds with a larger binning
    wsmax = max(dff[arg_speed])
    ax.bar(dff[arg_dir], dff[arg_speed],  # wd és la direcció del vent, i ws és la velocitat
           normed=True,bins=np.linspace(min_scale,wsmax**(1/pw),10)**pw,opening=0.8,edgecolor='white')  
    ax.set_title(tit, position=(0.5, 1.1),fontsize = 25)
    ax.set_legend()
    leg = ax.legend(title=leg_tit, bbox_to_anchor=(-0.47,-0.05),fontsize = 20)
    #get legend text
    leg_texts = leg.get_texts()
    leg.get_title().set_fontsize(20)    
    #change legend text
    for j,lab in enumerate(leg_texts):
        print (lab)
        #re.sub('[\d\.\d]+',mtch,s)
        leg_texts[j].set_text(re.sub('[\d\.\d]+',lambda s: _mtch(s,form=form),lab.get_text().replace('[','').replace(')','').replace(':','-').replace('126.0 - inf','>126.0').replace('1.800 - inf','>1.800')))


def plot_windspeed(df, arg_av, arg_gust, wbins, unit='(km/h)',sampling_freq_min=0.5):

    #print ('Wind bins:', wbins)
    
    mdays = [0., 31., 28.25, 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.]

    mask_gust = ((df[arg_gust]>0) & (~df[arg_gust].isnull()))
    mask_av   = ((df[arg_av]>0) & (~df[arg_av].isnull()))
    Months = range(1,13)
    for month in Months: 
        freq_gust = pd.cut(df.loc[(df['Month'] == month) & mask_gust, arg_gust],bins=wbins, ordered=True).value_counts(normalize=True)
        freq_av   = pd.cut(df.loc[(df['Month'] == month) & mask_av,   arg_av],  bins=wbins, ordered=True).value_counts(normalize=True)
        freq_gust_var = pd.cut(df.loc[(df['Month'] == month) & mask_gust, arg_gust],bins=wbins, ordered=True).value_counts(normalize=False) / np.power(df.loc[(df['Month'] == month) & mask_gust, arg_gust].count(),2)
        freq_av_var   = pd.cut(df.loc[(df['Month'] == month) & mask_av,   arg_av],  bins=wbins, ordered=True).value_counts(normalize=False) / np.power(df.loc[(df['Month'] == month) & mask_av,arg_av].count(),2)

        print ('MONTH=',month)
        print ('freq_gust:',freq_gust)
        print ('freq_gust_var:',freq_gust_var)
        
        if (month == 1):
            freq_tot_gust = freq_gust * mdays[month]/365.25
            freq_tot_av   = freq_av   * mdays[month]/365.25
            freq_tot_gust_var = freq_gust_var * (mdays[month]/365.25)**2
            freq_tot_av_var   = freq_av_var   * (mdays[month]/365.25)**2
        else:
            freq_tot_gust = freq_tot_gust + freq_gust * mdays[month]/365.25
            freq_tot_av   = freq_tot_av   + freq_av   * mdays[month]/365.25
            freq_tot_gust_var = freq_tot_gust_var + freq_gust_var * (mdays[month]/365.25)**2
            freq_tot_av_var   = freq_tot_av_var  + freq_av_var   * (mdays[month]/365.25)**2

    # convert the frequencies into densities (see also https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)
    # The new units are now P/year/(km/h)
    conv_per_min_per_year = sampling_freq_min * 60 * 24 * 365.25 
    values_gust = freq_tot_gust.values / np.sum(freq_tot_gust.values) / np.diff(wbins) * conv_per_min_per_year
    values_av   = freq_tot_av.values  / np.sum(freq_tot_av.values) / np.diff(wbins) * conv_per_min_per_year
    values_gust_err = np.sqrt(freq_tot_gust_var.values)/ np.sum(freq_tot_gust.values) / np.diff(wbins) * conv_per_min_per_year
    values_av_err   = np.sqrt(freq_tot_av_var.values)  / np.sum(freq_tot_av.values) / np.diff(wbins) * conv_per_min_per_year

    print ('values_gust: ',values_gust, ' values_gust_err: ', values_gust_err)

    freq_tot_gust_var = pd.cut(df.loc[mask_gust, arg_gust],bins=wbins, ordered=True).value_counts(normalize=False) / np.power(df.loc[mask_gust, arg_gust].count(),2)
    freq_tot_av_var   = pd.cut(df.loc[mask_av,   arg_av],  bins=wbins, ordered=True).value_counts(normalize=False) / np.power(df.loc[mask_av,arg_av].count(),2)
    values_gust_err = np.sqrt(freq_tot_gust_var.values)/ np.sum(freq_tot_gust.values) / np.diff(wbins) * conv_per_min_per_year
    values_av_err   = np.sqrt(freq_tot_av_var.values)  / np.sum(freq_tot_av.values) / np.diff(wbins) * conv_per_min_per_year

    print (' new values_gust_err: ', values_gust_err)
    
    bin_centers = 0.5 * (wbins[:-1] + wbins[1:])
    #plt.step(bin_centers,values_gust, where='mid', lw=3, color='steelblue', alpha=0.6, linestyle='solid')
    plt.errorbar(bin_centers,values_gust, yerr=values_gust_err,fmt='.',marker='o',markersize=5,ls='none', lw=2, color='steelblue', alpha=1,elinewidth=3)
    #plt.scatter(xval, yval, s = 5 * yval, marker = "h", color = "r")

    
    #k_est, c_est = estimate_k_c_from_df(df,arg_gust)
    k_est, c_est = estimate_k_c_from_distribution(bin_centers,values_gust)    
    
    # Fit Weibull function, some explanation below
    #params = exponweib.fit(df.loc[mask_gust,arg_gust], floc=0, f0=1)
    #shape = params[1]
    #scale = params[3]
    #print ('Weibull shape:',shape)
    #print ('Weibull scale:',scale)
    
    #out = leastsq(lambda p,x,y: y-weibull(x,p[0],p[1])*conv_per_min_per_year,[k_est,c_est],args=(bin_centers,values_gust))
    #c = out[0]    
    #print ('Weibull shape:',c[0])
    #print ('Weibull scale:',c[1])

    #cnts, bin_edges, _ = plt.hist(df[arg_gust].dropna(),density=True, bins=wbins, log=True, alpha=0.6)

    # Using all params and the stats function
    #plt.plot(bin_centers,weibull(bin_centers,*c)*conv_per_min_per_year,lw=2,label='Weibull')
    plt.plot(bin_centers,weibull(bin_centers,k_est,c_est)*conv_per_min_per_year,label=r'Gusts: Weibull c=%.1f %s, k=%.1f' % (c_est, unit, k_est),lw=2,color='deepskyblue',linestyle='dashed')

    weibull_fit_tradewinds = weibull(bin_centers,k_est,c_est)*conv_per_min_per_year
    diff_gust = np.log10(values_gust)-np.log10(weibull_fit_tradewinds)
    print ('Diff Gust: ',diff_gust)

    diff_start_for_fit = 0.7
    idx = np.where((diff_gust > diff_start_for_fit) & (~np.isnan(diff_gust)) & (~np.isinf(diff_gust)))[0]-2
    #if idx[0] < 0:
    #idx = 0
    k_est_s, c_est_s = estimate_k_c_from_distribution(bin_centers[idx[0]:],values_gust[idx[0]:]-weibull_fit_tradewinds[idx[0]:])
    print ('New Weibull with: ',k_est_s,', ',c_est_s)
    print ('From: ',bin_centers[idx[0]:], ' and: ',values_gust[idx[0]:]-weibull_fit_tradewinds[idx[0]:])
    print ('Weibull residuals:',(weibull(bin_centers[idx[0]:],k_est_s,c_est_s)-weibull(bin_centers[idx[0]:],k_est,c_est)))
    frac_storms = values_gust[idx[0]+2:].sum()/values_gust.sum()
    print ('Frac storms: ', frac_storms)
    print ('Converted: ', (weibull(bin_centers[idx[0]:],k_est_s,c_est_s)-weibull(bin_centers[idx[0]:],k_est,c_est))*conv_per_min_per_year)
    print ('Frac. storms: ', (weibull(bin_centers[idx[0]:],k_est_s,c_est_s)-weibull(bin_centers[idx[0]:],k_est,c_est))*conv_per_min_per_year*frac_storms)
    #plt.plot(bin_centers[idx[0]-8:],(weibull(bin_centers[idx[0]-8:],k_est_s,c_est_s)-weibull(bin_centers[idx[0]-8:],k_est,c_est))*conv_per_min_per_year*frac_storms,label=r'Storm gusts: Weibull c=%.1f %s, k=%.1f' % (c_est_s, unit, k_est_s),lw=2,color='teal',linestyle='dashed')

    diff_start = 2.3
    idx = np.where((diff_gust > diff_start) & (~np.isnan(diff_gust)) & (~np.isinf(diff_gust))  & (values_gust_err>0))[0]

    print ('idx: ',idx)
    print ('diff_start', diff_start)
    
    for i in np.arange(-2,len(bin_centers[idx[0]:])//2-1):

        id_new = idx[0]+i
        if (id_new < 0):
            print ('id_new ',id_new, ' continue')
            id_new = 0
            continue
        print ('i: ', i, ' Idx[0]: ',idx[0],' id_new: ',id_new)
        print (' Centered at: ',bin_centers[id_new])
    
        #out = leastsq(lambda p,x,y,err: (np.log(y)-np.log(exp_decay(x,p[0],p[1])))*y/err,[k_est * conv_per_min_per_year,c_est],args=(bin_centers[id_new:],values_gust[id_new:],values_gust_err[id_new:]),full_output=True)
        popt,pcov = curve_fit(exp_decay,bin_centers[id_new:-1],values_gust[id_new:-1],sigma=values_gust_err[id_new:-1],p0=[k_est * conv_per_min_per_year,c_est])
        out = [popt,pcov]        
        c = out[0]    
        a = np.linspace(bin_centers[id_new],210.,150)
        fs = exp_decay(a,c[0],c[1])
        errs = _propagate_error(out,exp_decay,a)

        # manual reduced chi2:
        devs = (values_gust[id_new:-1]-exp_decay(bin_centers[id_new:-1],*popt))/values_gust_err[id_new:-1]
        ndf  = len(devs) - len(popt)        
        s_sq = np.sum(devs**2) / ndf
        print ('devs for i=',i,': ',devs)
        print ('CHI2/NDF for i=',i,': ',s_sq)    

        if ((s_sq < 2.) and (ndf > 6)):
            plt.fill_between(a,fs-errs,fs+errs,color='thistle',alpha=0.4)

        # exp_decay integrated yields:  
        v10 = find_recurrence(c[0],c[1],10.)
        v50 = find_recurrence(c[0],c[1],50.)
        v475 = find_recurrence(c[0],c[1],475.)

        print ('Found wind speed gusts for 10, 50, 475 y:', v10, v50, v475)
        
    #diff_start = 1.7
    idx = np.where((diff_gust > diff_start) & (~np.isnan(diff_gust)) & (~np.isinf(diff_gust)) & (values_gust_err>0))[0]
    print ('Idx: ',idx[0])
    
    #out = leastsq(lambda p,x,y: np.log(y)-np.log(exp_decay(x,p[0],p[1])),[k_est * conv_per_min_per_year,c_est],args=(bin_centers[idx[0]:],values_gust[idx[0]:]),full_output=True)
    #out = leastsq(lambda p,x,y,err: (np.log(y)-np.log(exp_decay(x,p[0],p[1])))*y/err,[k_est * conv_per_min_per_year,c_est],args=(bin_centers[idx[0]:],values_gust[idx[0]:],values_gust_err[idx[0]:]),full_output=True)
    popt,pcov = curve_fit(exp_decay,bin_centers[idx[0]:-1],values_gust[idx[0]:-1],sigma=values_gust_err[idx[0]:-1],p0=[k_est * conv_per_min_per_year,c_est])
    out = [popt,pcov]        
    c_gust = out[0]    

    print ('OUT:',out)
    print ('OUT[0]:',out[0])
    print ('OUT[1]:',out[1])

    # manual reduced chi2: 
    devs = (values_gust[idx[0]:-1]-exp_decay(bin_centers[idx[0]:-1],c_gust[0],c_gust[1]))/values_gust_err[idx[0]:-1]
    s_sq = np.sum(devs**2) / ( len(devs) - len(popt))
    print ('devs:',devs)
    print ('CHI2/NDF:',s_sq)    
    
    plt.plot(bin_centers[idx[0]:-1],exp_decay(bin_centers[idx[0]:-1],c_gust[0],c_gust[1]),'--',lw=2,color='darkmagenta',label=r'Gusts: P = %.1g$\cdot\exp(-v/$%.1f%s)' % (c_gust[0],c_gust[1],unit))

    # exp_decay integrated yields:  
    v10_gust = find_recurrence(c_gust[0],c_gust[1],10.)
    v50_gust = find_recurrence(c_gust[0],c_gust[1],50.)
    v475_gust = find_recurrence(c_gust[0],c_gust[1],475.)

    ferr = _propagate_error(out, exp_decay, v10_gust, chi2=s_sq)
    print ('f_err for 10.', ferr)
    ferr = _propagate_error(out, exp_decay, v50_gust, chi2=s_sq)
    print ('f_err for 50.', ferr)
    ferr = _propagate_error(out, exp_decay, v475_gust, chi2=s_sq)
    print ('f_err for 475.', ferr)

    a = np.linspace(bin_centers[idx[0]],210.,150)
    fs = exp_decay(a,c_gust[0],c_gust[1])
    errs = _propagate_error(out,exp_decay,a)

    print ('a=',a)
    print ('fs=',fs)
    print ('errs=',errs)
    print ('flow=',fs-errs)
    print ('fup=',fs+errs)
    
    print ('lower err for 10.', np.interp(exp_decay(v10_gust,c_gust[0],c_gust[1]), np.flip(fs-errs), np.flip(a)))
    print ('uppr err for 10.', np.interp(exp_decay(v10_gust,c_gust[0],c_gust[1]), np.flip(fs+errs), np.flip(a)))
    print ('lower err for 50.', np.interp(exp_decay(v50_gust,c_gust[0],c_gust[1]), np.flip(fs-errs), np.flip(a)))
    print ('uppr err for 50.', np.interp(exp_decay(v50_gust,c_gust[0],c_gust[1]), np.flip(fs+errs), np.flip(a)))
    print ('lower err for 475.', np.interp(exp_decay(v475_gust,c_gust[0],c_gust[1]), np.flip(fs-errs), np.flip(a)))
    print ('uppr err for 475.', np.interp(exp_decay(v475_gust,c_gust[0],c_gust[1]), np.flip(fs+errs), np.flip(a)))
    
    plt.fill_between(a,fs-errs,fs+errs,color='darkmagenta',alpha=0.3)
    
    plt.plot(np.array([bin_centers[-1],v50,v475]),
             exp_decay(np.array([bin_centers[-1],v50,v475]),c_gust[0],c_gust[1]),'-.',lw=2,color='darkmagenta')    
    
    print ('Found wind speed gusts for 10, 50, 475 y:', v10, v50, v475)
    print ('Found wind speed gusts at:', exp_decay(v10,c_gust[0],c_gust[1]),
           exp_decay(v50,c_gust[0],c_gust[1]),exp_decay(v475,c_gust[0],c_gust[1]))
    
    #k_est, c_est = estimate_k_c_from_df(df,arg_av)
    k_est, c_est = estimate_k_c_from_distribution(bin_centers,values_av)
    
    # Fit Weibull function, some explanation below
    #mask = ((df[arg_av]>0) & (~df[arg_av].isnull()))
    #params = exponweib.fit(df.loc[mask,arg_av], floc=0, f0=1)
    #shape = params[1]
    #scale = params[3]
    #print ('Weibull shape:',shape)
    #print ('Weibull scale:',scale)
    
    #plt.step(bin_centers,values_av, where='mid', lw=3, color='orange', alpha=0.6, linestyle='solid')
    idx = np.where((values_av > 0) & (values_av_err > 0))[0]
    plt.errorbar(bin_centers[idx],values_av[idx], yerr=values_av_err[idx],
                 fmt='.',ls='none', marker='o',markersize=4, lw=2, color='orange', alpha=0.6,elinewidth=3)
    
    #cnts, bin_edges, _ = plt.hist(df[arg_av], density=True, bins=wbins, log=True, alpha=0.6)
    #out = leastsq(lambda p,x,y: y-weibull(x,p[0],p[1])*conv_per_min_per_year,[k_est,c_est],args=(bin_centers,values_av))
    #c = out[0]    
    #print ('Weibull shape:',c[0])
    #print ('Weibull scale:',c[1])
    #plt.plot(bin_centers,weibull(bin_centers,c[0],c[1])*conv_per_min_per_year,'--',color='orange',label=r'scale: %.1f shape: %.1f' % tuple(c))

    # Using all params and the stats function
    plt.plot(bin_centers,weibull(bin_centers,k_est,c_est)*conv_per_min_per_year,label=r'Average: Weibull c=%.1f %s, k=%.1f' % (c_est,unit,k_est),lw=2,color='gold',linestyle='dashed')

    diff_av = np.log10(values_av)-np.log10(weibull(bin_centers,k_est,c_est)*conv_per_min_per_year)
    print ('Diff Aver: ',diff_av)
    diff_start = 0.7
    idx = np.where((diff_av > diff_start) & (~np.isnan(diff_av)) & (~np.isinf(diff_av)) & (values_av > 0) & (values_av_err > 0))[0]
    print ('Idx: ',idx)

    for i in np.arange(-5,len(bin_centers[idx[0]:])//2-5):

        id_new = idx[0]+i
        if (id_new < 0):
            print ('id_new ',id_new, ' continue')
            id_new = 0
            continue

        v_av = values_av[id_new:]
        print ('v_av=',v_av)
        idx_new = np.where((values_av[id_new:] > 0) & (values_av_err[id_new:]>0))[0] + id_new
        print ('i: ', i, ' Idx[0]: ',idx[0],' id_new: ',idx_new)
        print (' Bin centers: ',bin_centers[idx_new])
        print (' value_av: ',values_av[idx_new])
        print (' value_av err: ',values_av_err[idx_new])        
    
        #out = leastsq(lambda p,x,y,err: (np.log(y)-np.log(exp_decay(x,p[0],p[1])))*y/err,[c_gust[0]*10,c_gust[1]/2],args=(bin_centers[idx_new],values_av[idx_new],values_av_err[idx_new]),full_output=True)
        popt,pcov = curve_fit(exp_decay,bin_centers[idx_new],values_av[idx_new],sigma=values_av_err[idx_new],p0=[c_gust[0]*10,c_gust[1]/2])
        out = [popt,pcov]
        c = out[0]
        a = np.linspace(bin_centers[id_new],140.,150)
        fs = exp_decay(a,c[0],c[1])
        print ('a=',a, 'c=',c)
        errs = _propagate_error(out,exp_decay,a)

        # manual reduced chi2:
        devs = (values_av[idx_new]-exp_decay(bin_centers[idx_new],*popt))/values_av_err[idx_new]
        ndf  = len(devs) - len(popt)
        s_sq = np.sum(devs**2) / ndf
        print ('devs for i=',i,': ',devs)
        print ('CHI2/NDF for i=',i,': ',s_sq)    

        if ((s_sq < 2.) and (ndf > 6)):
            plt.fill_between(a,fs-errs,fs+errs,color='khaki',alpha=0.3)

        # exp_decay integrated yields:  
        v10 = find_recurrence(c[0],c[1],10.)
        v50 = find_recurrence(c[0],c[1],50.)
        v475 = find_recurrence(c[0],c[1],475.)

        print ('Found average wind speeds for 10, 50, 475 y:', v10, v50, v475)

    #out = leastsq(lambda p,x,y: np.log(y)-np.log(exp_decay(x,p[0],p[1])),[c[0]/2,c[1]/1.5],args=(bin_centers[idx[0]:],values_av[idx[0]:]))
    #out = leastsq(lambda p,x,y: y-exp_decay(x,p[0],p[1]),[c[0]*5,c[1]/2],args=(bin_centers[idx[0]:],values_av[idx[0]:]))
    popt,pcov = curve_fit(exp_decay,bin_centers[idx],values_av[idx],sigma=values_av_err[idx],p0=[c_gust[0]*10,c_gust[1]/2])
    out = [popt,pcov]
    c_av = out[0]
    print ('c for gust: ',c_gust)
    print ('c for av: ',c_av)

    # manual reduced chi2: 
    devs = (values_av[idx]-exp_decay(bin_centers[idx],c_av[0],c_av[1]))/values_av_err[idx]
    s_sq = np.sum(devs**2) / ( len(devs) - len(popt))
    print ('devs:',devs)
    print ('CHI2/NDF:',s_sq)    
    
    plt.plot(bin_centers[idx],exp_decay(bin_centers[idx],c_av[0],c_av[1]),'-.',lw=2,color='salmon',label=r'Average: P = %.1g$\cdot\exp(-v/$%.1f%s)' % (c_av[0],c_av[1],unit))

    # exp_decay integrated yields:  
    v10_av = find_recurrence(c_av[0],c_av[1],10.)
    v50_av = find_recurrence(c_av[0],c_av[1],50.)
    v475_av = find_recurrence(c_av[0],c_av[1],475.)

    ferr = _propagate_error(out, exp_decay, v10_av, chi2=s_sq)
    print ('f_err for 10.', ferr)
    ferr = _propagate_error(out, exp_decay, v50_av, chi2=s_sq)
    print ('f_err for 50.', ferr)
    ferr = _propagate_error(out, exp_decay, v475_av, chi2=s_sq)
    print ('f_err for 475.', ferr)

    a = np.linspace(bin_centers[idx[0]],140.,150)
    fs = exp_decay(a,c_av[0],c_av[1])
    errs = _propagate_error(out,exp_decay,a)

    print ('a=',a)
    print ('fs=',fs)
    print ('errs=',errs)
    print ('flow=',fs-errs)
    print ('fup=',fs+errs)
    
    print ('lower err for 10.', np.interp(exp_decay(v10_av,c_av[0],c_av[1]), np.flip(fs-errs), np.flip(a)))
    print ('uppr err for 10.', np.interp(exp_decay(v10_av,c_av[0],c_av[1]), np.flip(fs+errs), np.flip(a)))
    print ('lower err for 50.', np.interp(exp_decay(v50_av,c_av[0],c_av[1]), np.flip(fs-errs), np.flip(a)))
    print ('uppr err for 50.', np.interp(exp_decay(v50_av,c_av[0],c_av[1]), np.flip(fs+errs), np.flip(a)))
    print ('lower err for 475.', np.interp(exp_decay(v475_av,c_av[0],c_av[1]), np.flip(fs-errs), np.flip(a)))
    print ('uppr err for 475.', np.interp(exp_decay(v475_av,c_av[0],c_av[1]), np.flip(fs+errs), np.flip(a)))

    plt.fill_between(a,fs-errs,fs+errs,color='goldenrod',alpha=0.3)
    
    print ('Found average wind speeds for 10, 50, 475 y:', v10_av, v50_av, v475_av)
    print ('Found average wind speeds at:', exp_decay(v10,c_av[0],c_av[1]),
           exp_decay(v50,c_av[0],c_av[1]),exp_decay(v475,c_av[0],c_av[1]))
    
    plt.yscale('log')
    plt.xlabel('Wind Speed (%s)' % (unit))
    plt.ylabel('Occurrence / year / (km/h)')

    return v10_gust, v50_gust, v475_gust

def calc_ti(x):

    #print ('x=',x)
    v = x.values
    #print ('v=',v)
    if (np.any(np.isnan(v))):
        #print ('Nan:',v)
        return np.nan
    N = len(v)
    #print ('N=',N)
    if (N < 5):
        #print ('<5:',v)
        return np.nan

    if not np.all(v):
        #print ('SOME ZEROS FOUND: ',v)
        return np.nan
    
    mean  = np.convolve(v, np.ones((N,))/N,mode='valid')[0]
    mean2 = np.convolve(v**2, np.ones((N,))/N,mode='valid')[0]
    var   = mean2-mean**2
    if (var < 0):
        print ('VAR<0',v)
        print ('mean, mean2, var', mean, mean2, var)
        return np.nan

    ti = np.sqrt(var)/mean

    if (ti > 1.):
        print ('TI=',ti,', values=',v, ', mean=',mean, ', mean2=',mean2) 
    
    return ti
    
