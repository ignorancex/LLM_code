# %%

# PLOTS
# Cifuentes et al. (2021)

# ∆PA vs. mu ratio
# d2 vs. d1
# MG vs. G-J
# Ug vs. Mtotal

# First: this cell loads the data and the common params.
# Then: each cell must be run separately

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
rc('text', usetex=False)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams.update({'errorbar.capsize': 4})

# Data

filename = 'cif21.multiplicity'
df = pd.read_csv(filename+'.csv')

# Sizes

figsize = (12, 10)
pointsize = 80
tickssize = 22
labelsize = 22
legendsize = 18
cblabsize = 18
linewidth = 1.5

# %%
# Plot: ∆PA vs. mu ratio

filename = 'cif21_deltaPA_muratio'

# Variables
# Physical pairs according to the criteria by Montes et al. 2018.


physical_AB = df.loc[(df['muratio_AB'] < 0.15) & (df['deltaPA_AB'] < 15)]
physical_AC = df.loc[(df['muratio_AC'] < 0.15) & (df['deltaPA_AC'] < 15)]
visual_AB = df.loc[(df['deltaPA_AB'] > 15) & (df['muratio_AB'] > 0.25)]
visual_AC = df.loc[(df['deltaPA_AC'] > 15) & (df['muratio_AB'] > 0.25)]
mixed_AB = df.loc[(df['muratio_AB'] > 0.15) & (
    df['muratio_AB'] < 0.25) & (df['deltaPA_AB'] < 15)]
mixed_AC = df.loc[(df['muratio_AC'] > 0.15) & (
    df['muratio_AC'] < 0.25) & (df['deltaPA_AC'] < 15)]

x_AB_phys = physical_AB['muratio_AB']
y_AB_phys = physical_AB['deltaPA_AB']
x_AB_vis = visual_AB['muratio_AB']
y_AB_vis = visual_AB['deltaPA_AB']
x_AB_mix = mixed_AB['muratio_AB']
y_AB_mix = mixed_AB['deltaPA_AB']

x_AC_phys = physical_AC['muratio_AC']
y_AC_phys = physical_AC['deltaPA_AC']
x_AC_vis = visual_AC['muratio_AC']
y_AC_vis = visual_AC['deltaPA_AC']
x_AC_mix = mixed_AC['muratio_AC']
y_AC_mix = mixed_AC['deltaPA_AC']

# Labels

xlabel = r'$\mu$ ratio'
ylabel = r'$\Delta PA$ [deg]'

# Canvas & Colours

fig, ax = plt.subplots(figsize=figsize)

# Plots

ax.scatter(x_AB_phys, y_AB_phys, edgecolor='b', c='none', s=pointsize,
           marker='o', label='Physical AB')
ax.scatter(x_AB_vis, y_AB_vis, c='r', s=pointsize,
           marker='x', label='Visual AB')
ax.scatter(x_AB_mix, y_AB_mix, c='orange', s=pointsize,
           marker='x', label='Mixed AB')

ax.scatter(x_AC_phys, y_AC_phys, edgecolor='b', c='none', s=pointsize,
           marker='o', label='Physical AC')
ax.scatter(x_AC_vis, y_AC_vis, c='r', s=pointsize,
           marker='x', label='Visual AC')
ax.scatter(x_AC_mix, y_AC_mix, c='orange', s=pointsize,
           marker='x', label='Mixed AC')

ax.axvline(0.15, color='black', linestyle='dashed', linewidth=linewidth)
ax.axvline(0.25, color='black', linestyle='dashed', linewidth=linewidth)
ax.axhline(15, color='black', linestyle='dashed', linewidth=linewidth)

# Axes: range & scale

ax.loglog()
ax.set_ylim(0, 100)

# Axes: ticks

ax.tick_params(axis='x', labelsize=tickssize, direction='in',
               top=True, labeltop=False, which='both')
ax.tick_params(axis='y', labelsize=tickssize, direction='in',
               right=True, labelright=False, which='both')
ax.tick_params('both', length=10, width=1, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
ax.minorticks_on()
ax.xaxis.set_tick_params(which='minor', bottom=True, top=True)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

# Axes: labels & legend

ax.set_xlabel(xlabel, size=labelsize)
ax.set_ylabel(ylabel, size=labelsize)

# Show & Save

plt.savefig(filename+'.png', bbox_inches='tight')

# %%
# Plot: d2 vs. d1

filename = 'cif21_d2_d1'

# Variables

# Physical and visual AB and AC pairs computed in previous cell.

AB_phys = df.loc[np.abs(df['d_A'] - df['d_B'])/df['d_A'] < 0.10]
AC_phys = df.loc[np.abs(df['d_A'] - df['d_C'])/df['d_A'] < 0.10]
AB_vis = df.loc[np.abs(df['d_A'] - df['d_B'])/df['d_A'] >= 0.10]
AC_vis = df.loc[np.abs(df['d_A'] - df['d_C'])/df['d_A'] >= 0.10]

x_AB_phys = AB_phys['d_A']
y_AB_phys = AB_phys['d_B']
x_AC_phys = AC_phys['d_A']
y_AC_phys = AC_phys['d_C']
x_AB_vis = AB_vis['d_A']
y_AB_vis = AB_vis['d_B']
x_AC_vis = AC_vis['d_A']
y_AC_vis = AC_vis['d_C']

xp = np.logspace(0, 3, 1000)

# Labels

xlabel = r'd$_A$ [pc]'
ylabel = r'd$_{B, C}$ [pc]'

# Canvas & Colours

fig, ax = plt.subplots(figsize=figsize)

# Plots

ax.plot(xp, xp, 'k--', zorder=0)
ax.plot(xp, xp*1.15, 'k--', zorder=0)
ax.plot(xp, xp*0.85, 'k--', zorder=0)

ax.scatter(x_AB_phys, y_AB_phys, edgecolor='b', c='none',  s=pointsize,
           marker='o', label='Physical AB')
ax.scatter(x_AB_vis, y_AB_vis, c='r',  s=pointsize,
           marker='x', label='Visual AB')
ax.scatter(x_AC_phys, y_AC_phys, edgecolor='b', c='none',  s=pointsize,
           marker='o', label='Physical AC')
ax.scatter(x_AC_vis, y_AC_vis, c='r',  s=pointsize,
           marker='x', label='Visual AC')

# Axes: range & scale

ax.loglog()
ax.set_xlim(9, 1000)
ax.set_ylim(9, 1000)

# Axes: ticks

ax.tick_params(axis='x', labelsize=tickssize, direction='in',
               top=True, labeltop=False, which='both')
ax.tick_params(axis='y', labelsize=tickssize, direction='in',
               right=True, labelright=False, which='both')
ax.tick_params('both', length=10, width=1, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
ax.minorticks_on()
ax.xaxis.set_tick_params(which='minor', bottom=True, top=True)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

# Axes: labels & legend

ax.set_xlabel(xlabel, size=labelsize)
ax.set_ylabel(ylabel, size=labelsize)

# Show & Save

plt.savefig(filename+'.png', bbox_inches='tight')

# %%
# Plot: MG vs. G-J

filename = 'cif21_MG_GJ'

# Add columns to dataframe

df['MG'] = df['phot_g_mean_mag'] - 5 * np.log10(1000/df['parallax']) + 5
df['GJ'] = df['phot_g_mean_mag'] - df['Jmag']

# (Optional, example) Separate elements with string in the comments

# string = 'New'
# new_var = df[Comments.str.contains(string, regex=True)]['phot_g_mean_mag']

# Variables

x = df['GJ']
y = df['MG']

# Labels

xlabel = r'$G-J$ [mag]'
ylabel = r'M$_G$ [mag]'

# Canvas & Colours

fig, ax = plt.subplots(figsize=figsize)

# Plots

ax.scatter(x, y, c='none', edgecolor='b', s=pointsize, marker='o')

# Plot labels

labels_out = ['LSPM J1633+0311S', 'HD 134494', 'PYC J07311+4556']
coords_out_ = [[1.6, 14.7], [1.65, 1.0], [3.6, 10.6]]

for i, type in enumerate(labels_out):
    ax.text(coords_out_[i][0], coords_out_[i][1], type, fontsize=16)

# Axes: range & scale

ax.invert_yaxis()

# Axes: ticks

ax.tick_params(axis='x', labelsize=tickssize, direction='in',
               top=True, labeltop=False, which='both')
ax.tick_params(axis='y', labelsize=tickssize, direction='in',
               right=True, labelright=False, which='both')
ax.tick_params('both', length=10, width=1, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
ax.minorticks_on()
ax.xaxis.set_tick_params(which='minor', bottom=True, top=True)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

# Axes: labels & legend

ax.set_xlabel(xlabel, size=labelsize)
ax.set_ylabel(ylabel, size=labelsize)

# Show & Save

plt.savefig(filename+'.png', bbox_inches='tight')

# %%
# Plot: Ug vs. Mtotal

filename = 'cif21_Ug_M'

# Calculate binding energy -Ug*
# '*' denotes the approximation r ~ s.


def Ug(M1, M2, rho, d_pc):
    G = 6.67408*1E-11  # m3 kg−1 s−2 (CODATA 2018)
    Msol = 1.98847*1E30  # kg
    GM = 1.3271244*1E20  # m3 s-2 (IAU 2015 Resolution B3)
    s = rho * d_pc  # AU
    AU = 149597870700  # m/AU (IAU 2012 Resolution B2)
    Ug = - G * (M1*M2 * Msol**2)/(s*AU)  # kg2 m2 s-2 = kg J (1 J = kg m2 s-2)
    return Ug

# Variables


M_AB = df['Mass_A'] + df['Mass_B']
M_AC = df['Mass_A'] + df['Mass_C']

Ug_AB = Ug(df['Mass_A'], df['Mass_B'], df['rho_AB'], df['d_A'])
Ug_AC = Ug(df['Mass_A'], df['Mass_C'], df['rho_AC'], df['d_A'])

x_AB = M_AB
y_AB = -Ug_AB

x_AC = M_AC
y_AC = -Ug_AC

# Labels

xlabel = r'$\mathcal{M}_{\rm total}$ [$\mathcal{M}_{\rm sol}$]'
ylabel = r'$-$U$_g^{*}$ [J]'

# Canvas & Colours

fig, ax = plt.subplots(figsize=figsize)

# Plots

ax.scatter(x_AB, y_AB, c='none', edgecolor='b', s=pointsize, marker='o')
ax.scatter(x_AC, y_AC, c='none', edgecolor='b', s=pointsize, marker='o')

# Plot labels

labels_out = ['LP 209-28 (KO 6AB)']
coords_out_ = [[0.66, 2.1E33]]

for i, type in enumerate(labels_out):
    ax.text(coords_out_[i][0], coords_out_[i][1], type, fontsize=16)

# Axes: range & scale

ax.loglog()
ax.set_xlim(0, 10)

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

# Show & Save

plt.savefig(filename+'.png', bbox_inches='tight')
