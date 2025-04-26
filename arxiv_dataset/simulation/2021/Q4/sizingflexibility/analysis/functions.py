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
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def annuity_factor(roi, n_inv):
    return roi/(1-math.pow(1+roi,-n_inv))

def invest_cost(dev, size, yr):
    if dev == 'pv':
        if yr == 0:
            c_inv = 1200       
        elif yr == 10:
            c_inv = 800
        c_inv = c_inv*size
    elif dev == 'bat':
        if yr == 0:
            c_inv = 900       
        elif yr == 10:
            c_inv = 600
        c_inv = c_inv*size
    return c_inv

def price_predict(typ, yr):
    if typ == 'import':
        prc = 0.3189*pow(1.015, yr)
    elif typ == 'export':
        if yr == 0:
            prc = 0.0683
        elif yr == 10:
            prc = 0.02
    return prc

def get_flex(lst):
    tot_flex = 0
    num_flex = 0
    for val in lst:
        if abs(val) > 0:
            tot_flex += abs(val)
            num_flex += 1
    if num_flex > 0:
        avg_flex = tot_flex/num_flex
    else:
        avg_flex = 0
    return avg_flex

def plt_contour(c_net, yr, save=''):
    x = np.arange(len(c_net))
    y = np.arange(len(c_net))
    Y, X = np.meshgrid(y, x)
    values = c_net.to_numpy()
    
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()
    plt.contourf(X, Y, values, 20, cmap='RdGy')
    plt.colorbar(label='Annual mean electricity price €/kWh')
    
    # mark point
    if yr == 0:
        ax.plot(x[3], y[1], '-ko')
        plt.text(x[3], y[1]-0.25, '0.2265')
    elif yr == 10:
        ax.plot(x[2], y[1], '-ko')
        plt.text(x[2], y[1]-0.25, '0.202')
    
    # add titles
    plt.xlabel('PV size kWp/MWh')
    plt.ylabel('BES size (days of backup) kWh/kWh')
    # if yr == 0:
    #     plt.title('Scenario: Present')
    # elif yr == 5:
    #     plt.title('Scenario: Short-term')
    # elif yr == 10:
    #     plt.title('Scenario: Long-term')
    
    # change labels
    xlabels = [0, 0.5, 1, 1.5, 2, 2.5]
    ylabels = [0, 0.5, 1, 1.5, 2, 2.5]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(xlabels)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ylabels)
    plt.tight_layout()
    
    # save plot
    if save: plt.savefig(r'pubs/'+save+'_'+str(yr)+'.svg', format='svg')
    
def plt_tflex(t_flex, yr, typ='imshow', save=''):
    x = np.arange(len(t_flex))
    y = np.arange(len(t_flex))
    Y, X = np.meshgrid(y, x)
    values = t_flex.to_numpy()
    
    # label init
    xlabels = [0, 0.5, 1, 1.5, 2, 2.5]
    ylabels = [0, 0.5, 1, 1.5, 2, 2.5]
    
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()    
    
    if typ == 'contour':
        plt.contourf(X, Y, values, 20, cmap='RdGy')
        plt.colorbar(label='Mean flexibility (kWh)')    
    elif typ == 'imshow':
        values = values.transpose()
        im = ax.imshow(values, cmap='copper_r', origin='lower')        
        # create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.get_yaxis().labelpad = 10
        cbar.set_label('Avg. daily flexibility (kWh)', rotation=90)
        # embed values
        for i in range(len(t_flex)):
            for j in range(len(t_flex)):
                text = ax.text(j, i, values[i, j], ha="center", va="center", color="w")
               
    # add titles
    plt.xlabel('PV size kWp/MWh')
    plt.ylabel('BES size (days of backup) kWh/kWh')
    # if yr == 0:
    #     plt.title('Scenario: Present')
    # elif yr == 5:
    #     plt.title('Scenario: Short-term')
    # elif yr == 10:
    #     plt.title('Scenario: Long-term')
    
    # change labels    
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    plt.tight_layout()
    
    # save plot
    if save: plt.savefig(r'pubs/'+save+'_'+str(yr)+'.svg', format='svg')
    
def plt_contour_anal(matx, yr, save=''):
    x = np.linspace(36,365,4,dtype=int)
    y = np.linspace(0.04,0.16,4)
    Y, X = np.meshgrid(y, x)
    values = matx.to_numpy()
    
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()
    plt.contourf(X, Y, values, 20, cmap='RdGy')
    plt.colorbar(label='Annual mean electricity price €/kWh')
    
    # add titles
    plt.xlabel('Number of flexibility service')
    plt.ylabel('Avg. flexibility profit €/kWh')
    # if yr == 0:
    #     plt.title('Scenario: Present')
    # elif yr == 5:
    #     plt.title('Scenario: Short-term')
    # elif yr == 10:
    #     plt.title('Scenario: Long-term')
    
    # change labels
    # xlabels = list(x)
    # ylabels = list(y)
    # ax.set_xticks(ax.get_xticks())
    # ax.set_xticklabels(xlabels)
    # ax.set_yticks(ax.get_yticks())
    # ax.set_yticklabels(ylabels)
    plt.tight_layout()
    
    # save plot
    if save: plt.savefig(r'pubs/'+save+'_'+str(yr)+'.png', format='png')

def plt_imshow_anal(matx_amep, matx_size, yr, save=''):
    x = np.array([50, 100, 200, 365])
    if yr == 0:
        y = np.array([0.04, 0.08, 0.12, 0.16])
    elif yr == 10:
        y = np.array([0.02, 0.06, 0.12, 0.16])
    Y, X = np.meshgrid(y, x)
    values = matx_amep.to_numpy()
    
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()
    
    values = values.transpose()
    im = ax.imshow(values, cmap='copper_r', origin='lower')    
    
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=4)
    
    # create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Annual mean electricity price (€/kWh)', rotation=90)
    
    # embed values
    for i in range(len(x)):
        for j in range(len(y)):
            text = ax.text(j, i, matx_size.iloc[i, j], ha="center", va="center", color="w")
    
    # change ticks
    xlabels = list(x)
    ylabels = list(y)
    ax.set_xticks(ax.get_xticks()[1:5])
    ax.set_xticklabels(xlabels)
    ax.set_yticks(ax.get_yticks()[1:5])
    ax.set_yticklabels(ylabels)
    
    # add labels
    plt.xlabel('Number of flexibility service executed')
    plt.ylabel('Avg. flexibility profit in €/kWh')
    
    # save plot
    if save: plt.savefig(r'pubs/'+save+'_'+str(yr)+'.svg', format='svg')
    
def nearest_val(matx, yr):
    if yr == 0:
        matx.iloc[0,0] = 'PV:1.5\nBES:0.5'
        matx.iloc[0,1] = 'PV:1.5\nBES:0.5'
        matx.iloc[0,2] = 'PV:1.5\nBES:0.5'
        matx.iloc[0,3] = 'PV:2.0\nBES:0.5'
        matx.iloc[1,0] = 'PV:1.5\nBES:0.5'
        matx.iloc[1,1] = 'PV:2.0\nBES:0.5'
        matx.iloc[1,2] = 'PV:2.0\nBES:0.5'
        matx.iloc[1,3] = 'PV:2.0\nBES:0.5'
        matx.iloc[2,0] = 'PV:2.0\nBES:0.5'
        matx.iloc[2,1] = 'PV:2.0\nBES:0.5'
        matx.iloc[2,2] = 'PV:2.5\nBES:0.5'
        matx.iloc[2,3] = 'PV:2.5\nBES:0.5'
        matx.iloc[3,0] = 'PV:2.5\nBES:0.5'
        matx.iloc[3,1] = 'PV:2.5\nBES:0.5'
        matx.iloc[3,2] = 'PV:2.5\nBES:0.5'
        matx.iloc[3,3] = 'PV:2.5\nBES:0.5'
    if yr == 10:
        matx.iloc[0,0] = 'PV:1.0\nBES:0.5'
        matx.iloc[0,1] = 'PV:1.0\nBES:0.5'
        matx.iloc[0,2] = 'PV:1.0\nBES:0.5'
        matx.iloc[0,3] = 'PV:1.5\nBES:0.5'
        matx.iloc[1,0] = 'PV:1.0\nBES:0.5'
        matx.iloc[1,1] = 'PV:1.0\nBES:0.5'
        matx.iloc[1,2] = 'PV:1.5\nBES:0.5'
        matx.iloc[1,3] = 'PV:1.5\nBES:0.5'
        matx.iloc[2,0] = 'PV:1.0\nBES:0.5'
        matx.iloc[2,1] = 'PV:1.5\nBES:0.5'
        matx.iloc[2,2] = 'PV:1.0\nBES:1.0'
        matx.iloc[2,3] = 'PV:1.0\nBES:1.0'
        matx.iloc[3,0] = 'PV:1.5\nBES:0.5'
        matx.iloc[3,1] = 'PV:1.0\nBES:1.0'
        matx.iloc[3,2] = 'PV:1.5\nBES:1.5'
        matx.iloc[3,3] = 'PV:1.5\nBES:1.5'
    
    return matx