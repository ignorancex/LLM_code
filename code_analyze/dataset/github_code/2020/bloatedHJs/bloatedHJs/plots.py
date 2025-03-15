'''
This module takes samples from MCMC as input
and plots several diagnostic plots:
- trace plots 
- corner plots 
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import corner

def trace_plot(samples, labels, outdir=None, outfile=None, save_plots=False, show_traces=True):
    ii = samples.shape[2]

    fig, axes = plt.subplots(ii, 1, sharex=True, figsize=(8, 9))
    plt.suptitle('{0}'.format(outfile))
    
    for i in range(ii):
        axes[i].plot(samples[:, :, i].T, color="k", alpha=0.4)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(labels[i])

    if save_plots:
        plt.savefig(outdir+'{0}_traces.pdf'.format(outfile))

    if show_traces:
        plt.show()

def corner_plot(samples, data, derived_params=False, selected_columns=None, \
    outdir=None, outfile=None, save_plots=True, show_plots=True, **kwargs):
    '''
    This function makes the cornet plot of:
        - the fitted parameters
        - the derived parameters 
        OR
        - columns of choice entered by the user 

    Args:
        samples          : the samples to do the corner plot
        data             : (dict) the observed parameters as a dictionary 
                            to plot the truths values
        derived_params   : (boolean) True to make a corner plot of the derived parameters:
                            Tint, Prcb, and Trcb
                            if False, a corner plot of the fitted parameters:
                            Lumi, Epsilon, Zhomo, Teq, Mp, Rp
        selected_columns : custom corner plot of the selected columns
        outdir           : the absolute path to where the files are saved
        outfile          : the name of the output pdf file 
    '''
    if selected_columns is None:
        if not derived_params:
            columns    = ['Lumi', 'Epsilon', 'Zhomo', 'Teq', 'Mp', 'Rp']
            truths     = [-1, -1, -1, data['Teq'], data['Mp'], data['Rp']]
            outfile    = outfile + '_fittedparams'
        if derived_params:
            columns    = ['Tint', 'Prcb', 'Trcb']
            truths     = [-1, -1, -1]
            outfile    = outfile + '_derivedparams'
    else:
        columns = selected_columns
        outfile = outfile

    X = samples[columns]

    if selected_columns is None:
        corner.corner(X, labels=columns, quantiles=[0.16,0.5,0.84], show_titles=True, truths=truths, **kwargs)
    else:
        corner.corner(X, labels=columns, quantiles=[0.16,0.5,0.84], show_titles=True, **kwargs)

    if save_plots is True:
        plt.savefig(outdir+'{0}_corner.pdf'.format(outfile), bbox_inches='tight', dpi=400)

    if show_plots:
        plt.show()
    else:
        plt.close()







