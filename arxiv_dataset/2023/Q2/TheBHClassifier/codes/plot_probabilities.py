### Andrea Antonelli 10/26/2022

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import argparse


# User inputs
# ---------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Input parameters')
parser.add_argument('-event' ,'--event', type=str ,metavar=' ', default='GW190521', help='Choose a LVK event.')

args = parser.parse_args()
event =  args.event


# Create dataframe with sum of channels for probability renormalization.
# ---------------------------------------------------------------------------------------------------------------------------


modelname =['s0.0_M9.0_r1.0',
            's0.2_M5.3_r1.0',
            's0.2_M6.0_r1.0',
            's0.2_M9.0_r0.5',
            's0.2_M9.0_r1.0',
            's0.2_M9.0_r2.0',
            's0.5_M9.0_r1.0'] 

modellabel = [r'$S_{0.0}M_{9.0}R_{1.0}$',
              r'$S_{0.2}M_{5.3}R_{1.0}$',
              r'$S_{0.2}M_{6.0}R_{1.0}$',
              r'$S_{0.2}M_{9.0}R_{0.5}$',
              r'$S_{0.2}M_{9.0}R_{1.0}$',
              r'$S_{0.2}M_{9.0}R_{2.0}$',
              r'$S_{0.5}M_{9.0}R_{1.0}$'] 

#'A', 'B','C','D','E','F','G','H','I','J']
df = pd.DataFrame()
for model in modelname:
    with open('results/'+event+'/results_'+event+'__'+model+'.pickle', 'rb') as handle:
        
        df = df.append(pickle.load(handle), ignore_index=True)
        
df['sum_channels'] = df['channel_1'] + df['channel_2'] + df['channel_3'] + df['channel_4']





df_channel = pd.DataFrame({r'3bb': (df['channel_3']/df['sum_channels']).values,
                   r'exc': (df['channel_1']/df['sum_channels']).values,
                   r'cap': (df['channel_2']/df['sum_channels']).values,
                   r'ZLK':(df['channel_4']/df['sum_channels']).values}, index=modellabel)


# Set the plotting options.
# ---------------------------------------------------------------------------------------------------------------------------


viridis = cm.get_cmap('viridis', 12)
sns.set_theme(style="white")
params = {'axes.labelsize': 22,
          'font.family': 'serif',
          'axes.titlesize':20,
          'font.serif': 'Computer Modern Raman',
          'font.size': 20,
          'axes.linewidth': 1,
          'legend.fontsize': 15,
          'xtick.labelsize': 23,
          'xtick.top': False,
          'ytick.labelsize': 20,
          'ytick.right': False,
          'axes.grid' : True,
          'text.usetex': True,
          'savefig.dpi' : 100,
          'lines.markersize' : 14,
          'axes.formatter.limits' : (-3,3)}

mpl.rcParams.update(params)


# Plot the most likely formation channel of the event.
# ---------------------------------------------------------------------------------------------------------------------------


ax = df_channel.plot.bar(rot=0, stacked=True, 
                         color =(viridis(0.0),viridis(0.25),viridis(0.5),viridis(0.9)))
plt.xticks(rotation=90)
plt.title('Formation channels of '+ event)
plt.ylabel('Probability')
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig('paper/figures/'+event+'_form_channels.pdf')
plt.show()


# Plot the most likely generations of the black holes forming the event..
# ---------------------------------------------------------------------------------------------------------------------------

df_gen = pd.DataFrame({r'2g+2g': df['p_2g2g'].values,
                   r'1g+2g': df['p_1g2g'].values,
                   r'1g+1g': df['p_1g1g'].values}, index=modellabel)

ax = df_gen.plot.bar(rot=0, stacked=True,
                     color =(viridis(0.3),viridis(0.6),viridis(0.9)))
plt.xticks(rotation=90)
plt.title('Black-hole generations of '+event)
plt.ylabel('Probability')
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig('paper/figures/'+event+'_BH_gens.pdf')
plt.show()
