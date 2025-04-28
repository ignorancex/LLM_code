import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle

# =============================================================================
# #                         FUNCTIONS
# =============================================================================


def extract_peaks(power, periods, min_height):
    '''
    Identify and select best peaks from periodogram:
        - Remove "identical periods"
        - 1-day derived periods
    where periodogram is --> ls = LombScargle(time, flux)
    must import --> from scipy.signal import find_peaks
    Parameters
    ----------
    power : 1-d array
        Output from --> freq, powers = ls.autopowers()
    periods : 1-d array
        1 / freq
    min_height : float
        min_height = float(ls.false_alarm_level(0.1)) # required peak height to attain any given false alarm probability


    Returns
    -------
    df_best_peaks : TYPE
        DESCRIPTION.

    '''
    # identify and extract peaks
    # print('Extracting peaks with scipy.find_peaks...')
    inds, peaks = find_peaks(power, height=min_height)

    
    
    df_peaks = pd.DataFrame(np.transpose([periods[inds], peaks['peak_heights']]), columns=['periods', 'height'])
    df_peaks.sort_values(by='height', inplace=True, ascending=False)
    
    if len(df_peaks) > 1:
        df_peaks['unique'] = False
        df_peaks.loc[df_peaks.height==max(df_peaks.height), 'unique'] = True # first one already in
        best_periods = np.array([df_peaks.periods.iloc[0]])
        eps = 0.1
        for period in df_peaks.periods[1:]:
            diff = np.min(abs(best_periods - period))
            if diff > eps:
                best_periods = np.append(best_periods, period)
                df_peaks.loc[df_peaks.periods==period, 'unique'] = True
                # print('Append ', period)
            # print(period, diff)
    else:
        df_peaks['unique'] = True
    
    # remove one-day derived signals    
    df_peaks['oneday'] = df_peaks.periods -1. 
    df_peaks.loc[df_peaks.oneday<0.1, 'unique'] = False
    ##################
    # print('Selecting best peaks...')
    df_best_peaks = pd.DataFrame(df_peaks[df_peaks.unique==True])
    df_best_peaks = df_best_peaks.drop(labels=['unique', 'oneday'], axis=1)
    return df_best_peaks

def mask_eclipse(df, eclipse_time=[5250, 5550]):
    '''
    Add boolean column for in-eclipse (True) and off-eclipse (False)
    '''
    df['eclipse'] = (df.time>eclipse_time[0])&(df.time<eclipse_time[1])
    print('{:} in-eclipse points...'.format(len(df[df['eclipse']==True])))
    
    return df

def load_skypatrol(file, ax=None, show=True):
    df_sp = pd.read_csv(file)
    df_sp['time'] = df_sp.HJD - 2450000
    df_sp['mag_err'] = df_sp['mag_err'].replace(99.99, np.nan)
    df_sp = df_sp.dropna()


    for cam in df_sp.Camera.unique():
        df_cam = df_sp[df_sp['Camera']==cam]
        med_flux = df_cam['flux(mJy)'].median()
        df_sp.loc[df_sp.Camera==cam, 'flux'] = df_cam['flux(mJy)'] / med_flux
        df_sp.loc[df_sp.Camera==cam, 'flux_err'] = df_cam['flux_err'] / med_flux

    df_sp_list = []
    for f in df_sp.Filter.unique():
        temp = df_sp[df_sp.Filter==f]
        if show == True:
            ax.errorbar(temp.time, temp.flux, temp.flux_err, fmt='x', label=f, alpha=0.35)
        temp = df_sigma_clip(temp, 'flux_err', sigma=2)
        temp = df_sigma_clip(temp, 'flux', sigma=2)
        if show == True:
            ax.errorbar(temp.time, temp.flux, temp.flux_err, fmt='.', label=f+'-clipped', alpha=0.75)
        temp.insert(0, 'survey', 'ASAS-SN-'+f)
        df_sp_list.append(temp)
    df_sp = pd.concat(df_sp_list)
    df_sp = mask_eclipse(df_sp)
    return df_sp


def df_sigma_clip(df, col, sigma=3.):
    if len(df) > 2:
        median = np.nanmedian(df[col])
        std = np.std(df[col])
        len_in = len(df)
        df = df[abs(df[col] - median) < sigma*std] # clip
        len_out = len(df)
        print('Sigma-clipping column:"{:}": {:} points discarded at {:}-sigma'.format(col, len_in - len_out, sigma))
    else:
        print('Empty array!!!')
        return None
    return df

def myperiodogram(df, min_period, max_period, ax=None, **kwargs):
    ls = LombScargle(df.time, df.flux, df.flux_err)
    frequencies, power = ls.autopower(minimum_frequency=(1./max_period), 
                                      maximum_frequency=(1/min_period),
                                      samples_per_peak=40)
    # convert to periods
    periods = 1/frequencies
    # extract best peaks
    df_best_peaks = extract_peaks(power, periods, 0.00)
    if ax != None:
        ax.plot(periods, power,'-', **kwargs)
        ax.set_xlim(left=1.)
    return df_best_peaks


def subtract_short_periods(df, ncycles):

    ls_p = lsc_cycle(df.time, df.flux-1.0, df.flux_err, ncycles, max_period=10., show=False)

    ls_df = pd.DataFrame(ls_p, columns=['amplitude', 'period', 'phase']) # store values
    print('Building stellar variation model...')
    stellar_variation = 1. + sines(df.time, ls_df.amplitude, ls_df.period, ls_df.phase)
    df_corr = df.copy()
    df_corr['flux'] = df.flux / stellar_variation
    df_corr['flux_err'] = df.flux_err / stellar_variation
    return df_corr
def lsc_cycle(time, ls_flux, flux_err, ncycles=4, max_period=100., show=True):
    ls_p = []
    if show == True:
        fig, ax = plt.subplots(ncycles, figsize=(8,16))
        fig.tight_layout()

    for i in range(ncycles):
        print(i)
        if show == True:
            ls_p1b,ls2_flux = lsc_model(time, ls_flux, flux_err,max_period=max_period, ax=ax[i])
        else:
            ls_p1b,ls2_flux = lsc_model(time, ls_flux, flux_err,max_period=max_period, ax=None)
        ls_p.append(ls_p1b)
        ls_flux = ls2_flux
    return np.array(ls_p)


def subtracted_periodogram(df, n = 6, ax=None):
    df_corr = subtract_short_periods(df, ncycles=n)
    
    axins = ax.inset_axes([0.03, 0.50, 0.47, 0.47])

    min_period, max_period = 1.05, 95.

    dicts = [{'c':'gray', 'alpha':0.75, 'label':df.survey.iloc[0]},
             {'c':'green', 'alpha':0.85, 'label':df.survey.iloc[0]+'-Sub'}]
    
    for ax_i in [ax, axins]:
        _ = myperiodogram(df, min_period, max_period, ax=ax_i, **dicts[0])  
        _ = myperiodogram(df_corr, min_period, max_period, ax=ax_i, **dicts[1]) 

    df_peak68 = myperiodogram(df_corr, min_period=67, max_period=70, ax=None) 
    period_68 = np.round(df_peak68.iloc[0].periods,2)
    print('P_68 = {:.4f} +- {:.4f} d'.format(period_68, 1/len(df_corr)))
    ax.text(s=str(period_68), x=period_68-2, y=df_peak68.iloc[0].height+0.0045)
    ax.legend(fontsize=9)
    ax.set(ylabel='Power[-]', xlim=(30, max_period))
    ax.set_ylim(bottom=0.)

    # sub region of the original image
    x1, x2, y1, y2 = 2.25, 3.5, *ax.get_ylim()
    axins.set(xlim = (x1, x2), ylim=(y1,y2))
    axins.set_yticklabels('')
    axins.set_xticks(np.arange(1,3.51,0.50))
    return ax


def sines(time, amplitudes, periods, phases):
    '''
    function that returns the sum of several sinusoids
    '''
    trend = 0
    for amplitude, period, phase in zip(amplitudes, periods, phases):
        sine = amplitude * np.sin(2 * np.pi * time / period + phase)
        trend += sine
    return trend

def plot_folded(time, flux, error, Pb, flux_fit, ax=None):
    # extract best fit
    amp, prd, phs = Pb
    
    phase = (time % prd) / prd    
    sort = np.argsort(phase)
    ###### flip the sine if needed ######
    fun1 = sines(time, [amp], [prd], [phs])
    fun2 = sines(time, [amp], [prd], [phs+np.pi])
    
    area1 = sum((fun1-flux_fit)**2)
    area2 = sum((fun2-flux_fit)**2)
    
    # Update the value of "phs" is the blue area is smaller
    if  area1 == np.min([area1, area2]):
        pass
    else:
        phs += np.pi
    
    Pb = [amp, prd, phs] # CORRECT VALUES
    
    if ax != None:
        ax.set(ylabel='Flux [-]')
       
        ax.errorbar(phase.iloc[sort], flux.iloc[sort], yerr=error, fmt='.', color='k', alpha=0.7)
        ax.plot(phase.iloc[sort], flux_fit[sort],'g-', lw=4, alpha=0.7, label='P = {:.2f} d'.format(prd))
        ax.legend(ncol=2)
    return Pb


def lsc_model(time, flux, error, ax=None, min_period=0.1, 
                            max_period=4, **kwargs):
    # create the periodogram
    ls = LombScargle(time, flux, error)
    frequencies, power = ls.autopower(minimum_frequency=(1./max_period), 
                                          maximum_frequency=(1/min_period), 
                                          samples_per_peak=40)
    # convert to periods
    periods = 1/frequencies

    height = 0.0
    df_best_peaks = extract_peaks(power, periods, min_height=height)
    period = df_best_peaks.periods.iloc[0]
    # fit the sinusoid
    flux_fit = ls.model(time, 1/period)
    residuals = flux - flux_fit
    t0, t1, t2 = ls.model_parameters(1/period)
    # convert theta parameters to amplitude and phase
    amplitude = np.hypot(t1, t2)
    phase = -np.arctan(t1 / t2) + np.pi/2
    Pb = (amplitude, period, phase)

    # GET THE CORRECT "Pb" values by checking the areas
    Pb = plot_folded(time, flux, error, Pb, flux_fit, ax=ax)
    print('{:.6f} sin(2 pi time / {:.4f} + {:.4f})'.format(*Pb))
    
    return Pb, residuals