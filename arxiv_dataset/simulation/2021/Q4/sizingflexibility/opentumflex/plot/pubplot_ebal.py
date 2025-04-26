"""
__author__ = "Babu Kumaran Nalini"
__copyright__ = "2021 TUM-EWK"
__credits__ = []
__license__ = "GPL v3.0"
__version__ = "1.0"
__maintainer__ = "Babu Kumaran Nalini"
__email__ = "babu.kumaran-nalini@tum.de"
__status__ = "Development"
"""
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
import math

def pubplot_ebal(ems={}, save=''):
    # figure properties
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True,  gridspec_kw={'height_ratios': [3,1]})
    # fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(7, 3), gridspec_kw={'height_ratios': [3,0]}) # No soc plot
    plt.rc('font', family='serif')
       
    # Initialize
    ts = ems['time_data']['time_slots']  
    N = len(ts)
    ind = np.arange(N)  
    indplus1 = np.arange(N+1)  
    ts_hr = pd.to_datetime(ts).strftime('%H:%M')
    width = 1  
    
    # Get optimization results
    opt_res = ems['optplan'].copy()
    for param in opt_res:
        opt_res[param] = np.array(opt_res[param])
    
    # plots
    p2 = ax1.bar(ind, opt_res['PV_power'], width, bottom=opt_res['bat_output_power'],
            color='goldenrod', align='edge')
    p3 = ax1.bar(ind, opt_res['bat_output_power'], width, color='indianred', align='edge')
    p4 = ax1.bar(ind, -opt_res['bat_input_power'], width, color='indianred', align='edge')
    p5 = ax1.bar(ind, opt_res['grid_import'], width,
            bottom=opt_res['bat_output_power'] + opt_res['PV_power'],
            color='grey', align='edge')
    p6 = ax1.bar(ind, -opt_res['grid_export'], width, bottom=-opt_res['bat_input_power'],
            color='darkseagreen', align='edge')
    p9 = ax1.step(indplus1, np.append(opt_res['Last_elec'], 0), linewidth=2, where='post', color='k')
    
    # Highlight flex steps
    if 'flex_start' in ems['reoptim']:
        time_series = list(ems['time_data']['time_slots'])
        hglt_start = time_series.index(ems['reoptim']['flex_start'])
        hglt_end = hglt_start + ems['reoptim']['flex_steps']                
        ax1.axvspan(ind[hglt_start], ind[hglt_end], color='crimson', alpha=0.2, hatch='/')   
        ax2.axvspan(ind[hglt_start], ind[hglt_end], color='crimson', alpha=0.2, hatch='/')     

    # xticks
    ax1.axhline(linewidth=2, color="black")
    idx_plt = (np.linspace(0, N, 12, endpoint=False)).astype(int)
    
    # plot properties
    ax1.grid(color='lightgrey', linewidth=0.75)
    ax1.margins(x=0)
    
    # adjust the y-axis limit
    bottom, top = ax1.get_ylim()
    if top >= abs(bottom):
        ax1.set_ylim(-(top + 0.5), top + 0.5)
    else:
        ax1.set_ylim(bottom - 0.5, -(bottom - 0.5))   
    
    # labels 
    ax1.set_ylabel('Electrical load [kWh]', fontsize=10)
    
    # legend selector
    selector = {
                np.count_nonzero(opt_res['PV_power']): [p2[0], 'PV'],
                np.count_nonzero(opt_res['bat_output_power'] + opt_res['bat_input_power']): [p3[0], 'BES/TES'],
                np.count_nonzero(opt_res['grid_import']): [p5[0], 'Grid_import'],
                np.count_nonzero(opt_res['grid_export']): [p6[0], 'Grid_export'],
                }
    legend_entries = [p9[0]]
    labels_entries = ['Load']
    for count_nonzero, legend_entry in selector.items():
        if count_nonzero > 0:
            legend_entries.append(legend_entry[0])
            labels_entries.append(legend_entry[1])
    ax1.legend(legend_entries, labels_entries, prop={'size': 10}, 
               bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", ncol=4, frameon=False)
    # ax1.legend(legend_entries, labels_entries,
    #            prop={'size': 10}, bbox_to_anchor=(1.01, 0), loc='lower left', frameon=False)
    # ax1.legend(frameon=False)
    
    # ax1 ticks
    ax1.set_xticks(ind[idx_plt])
    ax1.set_xticklabels(ts_hr[idx_plt])
    ax1.tick_params(axis='both', labelsize=10)
    
    # SOC plots
    init_SOC = [ems['devices']['bat']['initSOC']]
    ax2.plot(indplus1, init_SOC+ems['optplan']['bat_SOC']) 
    
    # ax2 ticks
    ax2.set_xticks(ind[idx_plt])
    ax2.set_xticklabels(ts_hr[idx_plt])
    ax2.set_ylim([0, 100])
    ax2.tick_params(axis='both', labelsize=10)
    
    # ax2 properties
    ax2.margins(x=0)
    ax2.grid(color='lightgrey', linewidth=0.75)
    
    # labels 
    ax2.set_ylabel('BES-SOC %', fontsize=10)
    fig.align_ylabels((ax1, ax2))  
    
    # For no SOC plot
    # fig.delaxes(ax2)
    
    # figure settings
    plt.tight_layout()
    plt.show()
    
    # save figure
    path = r'C:/Modelling/Publication/ProsumerFlex/pubdata/'
    if save: plt.savefig(path + save + '_ebal.pdf', format='pdf')