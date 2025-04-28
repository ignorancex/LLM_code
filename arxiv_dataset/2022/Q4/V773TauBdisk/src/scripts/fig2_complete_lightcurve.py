# Dario Gonzalez Picos
# picos@mail.strw.leidenuniv.nl
# Last update June 2022

'''
--- Script to generate figure 2 of the paper ---
Load calibrated photometry to build a complete light curve with data
from all ground-based surveys used in this work
'''
import paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_dir = '../data/'
out_dir = '../plots/'

plt.style.use(paths.data / 'texstyle') # my tex template


# Load DataFrame with photometry from each survey
# SWASP and HATNET data are excluded from the analysis
print('Reading complete light curve...')
df = pd.read_csv(paths.data / 'v773tau_complete_lightcurve.csv')

    
print('SURVEY \t  N_Points \n-------------------')
[print('{:8} \t {:}'.format(s, df[df.survey==s].size)) for s in df.survey.unique()]

def plot_inset(df, eclipse_time, save_name='fig2.pdf'):
    '''
    Generate figure with full light curve, highlighting each survey and 
    zooming-in on the eclipse

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with photometry and labels.
    eclipse_time : list
        list containing star and end of eclipse (in HJD-2450000).
    ax : plt.axis, optional
        axis to display figure on.

    Returns
    -------
    ax : plt.axis
        axes with plotted data and settings.

    '''
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    fig, ax = plt.subplots(1, figsize=(9,4))
    # inset axes....
    axins = ax.inset_axes([0.55, 0.10, 0.47, 0.57])
    
    for survey in df.survey.unique():
        df_survey = df[df.survey==survey]
        ax.errorbar(df_survey.time, df_survey.flux, df_survey.flux_err, 
                    fmt='.', label=survey, alpha=.75, ms=4)
        axins.errorbar(df_survey.time, df_survey.flux, df_survey.flux_err, 
                fmt='.', alpha=.75, ms=3.5)
    
    
    # sub region of the original image
    x1, x2, y1, y2 = eclipse_time[0], eclipse_time[-1], 0.0, 1.05
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    # axins.set_xticklabels('')
    axins.set_yticklabels('')
    
    # ax.indicate_inset_zoom(axins, edgecolor="black", alpha=.4, zorder=4.99)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='black', alpha=.7, zorder=4.99)
        
    ax.axhline(y=1.0, ls='--', color='black', alpha=0.1)  
    axins.axhline(y=1.0, ls='--', color='black', alpha=0.1) 
    ax.set(xlabel='HJD -2450000', ylabel='Flux [-]', title='V773Tau')
    ax.set_xticks(np.arange(2500,9501,500))
    ax.legend()
    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    return ax

eclipse_time = [5250, 5650]
_ = plot_inset(df, eclipse_time, save_name=paths.figures/'fig2_full_lightcurve.pdf')
