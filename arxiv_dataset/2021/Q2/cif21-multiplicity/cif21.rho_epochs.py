# %%
# RHO CALCULATOR
# Obtains the separation as a function of time spanning several epochs.
# Cifuentes et al. (2021)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
rc('text', usetex=False)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Data

filename = 'KO6AB'

df_ = pd.read_csv(filename+'.csv')
df = df_.loc[df_['INCLUDE'] == True]

# rho & theta
# As defined by J. Smolinski and W. Osborn (2006RMxAC..25...65S)

RA_A = np.deg2rad(df['RA_A'])
RA_B = np.deg2rad(df['RA_B'])
DE_A = np.deg2rad(df['DE_A'])
DE_B = np.deg2rad(df['DE_B'])

deltaRA = RA_B - RA_A
deltaDE = DE_B - DE_A

rho = np.rad2deg(np.arccos(np.cos(deltaRA*np.cos(DE_A))*np.cos(deltaDE)))*3600
theta = np.rad2deg(np.arctan(deltaRA*np.cos(DE_A)/deltaDE)) + 180

# Definitions: Julian Date and Modified Julian Date

Epoch = 2015.0
JD = (Epoch - 2000)*365.25 + 2451545.0
MJD = JD - 2400000.5

# Plotting

# Variables

x = df['JD']
y = rho
yerr = df['RHO_ERROR']

# Fit

fit = np.polyfit(x, y, 1)
p = np.poly1d(np.polyfit(x, y, 1))
t = np.linspace(min(x), max(x), 1000)

# Labels

xlabel = r'JD [d]'
ylabel = r'$\rho$ [arcsec]'

# Sizes

figsize = (12, 10)
pointsize = 8
tickssize = 22
labelsize = 22
legendsize = 18
cblabsize = 18
linewidth = 1.5

# Canvas & Colours

fig, ax = plt.subplots(figsize=figsize)

# Plots

plt.errorbar(x, y, yerr=yerr, c='b', ms=pointsize,
             capsize=6, marker='o', linestyle='none')
ax.plot(t, p(t), 'r-')

# Axes: range & scale

# ax.set_xlim(,)
# ax.set_ylim(,)

# Axes: labels & legend

ax.set_xlabel(xlabel, size=labelsize)
ax.set_ylabel(ylabel, size=labelsize)

# Axes: ticks

ax.tick_params(axis='x', labelsize=tickssize, direction='in',
               top=True, labeltop=False, which='both')
ax.tick_params(axis='y', labelsize=tickssize, direction='in',
               right=True, labelright=False, which='both')
ax.tick_params('both', length=10, width=1, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
ax.minorticks_on()
ax.xaxis.set_tick_params(which='minor', bottom=True, top=True)
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.xaxis.get_offset_text().set_fontsize(labelsize)

# Show & Save

plt.savefig(filename+'.png', bbox_inches='tight')
