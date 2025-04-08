import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import matplotlib
import matplotlib.gridspec as gridspec
import colorcet as cc
import os
"""
this script plots figure 2
written by Yangcheng Luo
"""

# set up figure directory 
figdir = "./figs/"
os.makedirs(figdir, exist_ok=True)

h = 6.62607015e-34
c = 299792458.
k = 1.380649e-23

# Create a figure
fig = plt.figure(figsize=(9, 6))

# Create a GridSpec with 4 rows and 2 columns
gs = gridspec.GridSpec(4, 2, figure=fig)

# The left column spans 1/2 of the width and the two subplots each take 2 rows (half of 4 rows)
ax1 = fig.add_subplot(gs[0:2, 0:1])  # Top left, spans 2 rows
ax2 = fig.add_subplot(gs[2:4, 0:1])  # Bottom left, spans 2 rows

# The right column spans 1/2 of the width and 4 subplots in 4 rows
ax3 = fig.add_subplot(gs[0:1, 1:2])  # 1st row right
ax4 = fig.add_subplot(gs[1:2, 1:2])  # 2nd row right
ax5 = fig.add_subplot(gs[2:3, 1:2])  # 3rd row right
ax6 = fig.add_subplot(gs[3:4, 1:2])  # 4th row right

dat = nc.Dataset('./out/fig2_inner.nc', 'r')

it_end = 379490
it_start_phase = 376865
it_start_t = 371525


t = dat['t-hat'][:]
T_surf = dat['T_surf'][:]
tau_sw = dat['tau_sw'][:]
tau_lw = dat['tau_lw'][:]
T_b_midIR = dat['T_4p5'][:]
T_b_visib = dat['T_0p5'][:]
T_surf_0 = dat['T_surf_0'][:]
tau_sw_0 = dat['tau_sw_0'][:]
T_cloud_up = dat['T_cloud_up'][:]
T_cloud_down = dat['T_cloud_down'][:]
A = dat['A'][:]

ax1.plot(T_surf[t >= 1], tau_sw[t >= 1], color='gray', linestyle='--', linewidth=1, zorder=0)
scatter = ax1.scatter(T_surf[it_start_phase:it_end], tau_sw[it_start_phase:it_end], c=t[it_start_phase:it_end] - t[it_start_phase], s=3, cmap=cc.cm.CET_C2, zorder=1)
cbar1 = fig.colorbar(scatter, ax=ax1)
cbar1.set_label('nondimensionalized time $t/d$')
ax1.scatter(T_surf_0, tau_sw_0, marker='^', c='k')
ax1.set_ylim(bottom=0)
ax1.set_xlabel(r'surface temperature $T_{\mathrm{surf}}$ (K)')
ax1.set_ylabel(r'cloud shortwave optical depth $\tau_{\mathrm{SW}}$')
ax1.text(0.9, 0.9, 'A', transform=ax1.transAxes, size=12, weight='bold')

T_b_midIR_0 = dat['T_4p5_0'][:]
alpha = dat['alpha'][:]
A_0 = 1 - 1/(1 + alpha*tau_sw_0)
T_star = dat['T55cncA'][:]
wavelength = 500e-9
B_T_star_500nm = 2*h*c**2/wavelength**5/(np.exp(h*c/(wavelength*k*T_star)) - 1)
R_star = dat['R55cncA'][:]
r = dat['a55cnce'][:]
B_T_surf_500nm_0 = 2*h*c**2/wavelength**5/(np.exp(h*c/(wavelength*k*T_surf_0)) - 1)
irradiance = A_0*B_T_star_500nm*R_star**2/r**2 + (1 - A_0)*B_T_surf_500nm_0
T_b_visible_0 = h*c/np.log((irradiance/(2*h*c**2)*wavelength**5)**(-1) + 1)/wavelength/k

ax2.plot(T_b_midIR[t >= 1], T_b_visib[t >= 1], color='gray', linestyle='--', linewidth=1, zorder=0)
scatter = ax2.scatter(T_b_midIR[it_start_phase:it_end], T_b_visib[it_start_phase:it_end], c=t[it_start_phase:it_end] - t[it_start_phase], s=3, cmap=cc.cm.CET_C2, zorder=1)
cbar2 = fig.colorbar(scatter, ax=ax2)
cbar2.set_label('nondimensionalized time $t/d$')
ax2.scatter(T_b_midIR_0, T_b_visible_0, marker='^', c='k')
ax2.set_xlabel(r'4.5 $\mathrm{\mu}$m brightness temperature $T_{\mathrm{b,4.5}}$ (K)')
ax2.set_ylabel(r'0.5 $\mathrm{\mu}$m brightness temperature $T_{\mathrm{b,0.5}}$ (K)')
ax2.text(0.9, 0.9, 'B', transform=ax2.transAxes, size=12, weight='bold')

ax3.plot(t[it_start_t:it_end] - t[it_start_t], T_surf[it_start_t:it_end], color='k', label='$T_{\mathrm{surf}}$')
ax3.plot(t[it_start_t:it_end] - t[it_start_t], T_cloud_up[it_start_t:it_end], color=(230/255, 140/255, 0/255), label='$T_{\mathrm{cloud,↑}}$')
ax3.plot(t[it_start_t:it_end] - t[it_start_t], T_cloud_down[it_start_t:it_end], color='#009E73', label='$T_{\mathrm{cloud,↓}}$')
ax3.set_xlim([0, t[it_end] - t[it_start_t]])
ax3.set_xlabel('$t/d$')
ax3.set_ylabel('temperature (K)')
ax3.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1.07))
ax3.text(0.025, 0.1, 'C', transform=ax3.transAxes, size=12, weight='bold')

ax4.plot(t[it_start_t:it_end] - t[it_start_t], A[it_start_t:it_end], color='k')
ax4.set_xlim([0, t[it_end] - t[it_start_t]])
ax4.set_ylim([0, 1])
ax4.set_xlabel('$t/d$')
ax4.set_ylabel('cloud albedo $A$')
ax4.text(0.025, 0.1, 'D', transform=ax4.transAxes, size=12, weight='bold')

ax41 = ax4.twinx()
ax41.plot(t[it_start_t:it_end] - t[it_start_t], tau_sw[it_start_t:it_end], color=(230/255, 140/255, 0/255))
ax41.set_ylim([0, np.amax(tau_sw[it_start_t:it_end])*1.1])
ax41.set_ylabel(r'$\tau_{SW}$', color=(230/255, 140/255, 0/255))
ax41.tick_params(axis='y', colors=(230/255, 140/255, 0/255), labelcolor=(230/255, 140/255, 0/255))

ax42 = ax4.twinx()
ax42.spines['right'].set_position(('axes', 1.2))
ax42.plot(t[it_start_t:it_end] - t[it_start_t], tau_lw[it_start_t:it_end], color='#009E73')
ax42.set_ylim([0, np.amax(tau_sw[it_start_t:it_end])*1.1])
ax42.set_ylabel(r'$\tau_{LW}$', color='#009E73')
ax42.tick_params(axis='y', colors='#009E73', labelcolor='#009E73')

wavelength = 4.5e-6
irrad_MIR_surface_contrib = 2*h*c**2/wavelength**5/(np.exp(h*c/(wavelength*k*T_surf)) - 1)*np.exp(-tau_lw)
irrad_MIR_cloud_contrib = 2*h*c**2/wavelength**5/(np.exp(h*c/(wavelength*k*T_cloud_up)) - 1)*(1 - np.exp(-tau_lw))
T_b_MIR_surface_only = h*c/np.log((irrad_MIR_surface_contrib/(2*h*c**2)*wavelength**5)**(-1) + 1)/wavelength/k
T_b_MIR_cloud_only = h*c/np.log((irrad_MIR_cloud_contrib/(2*h*c**2)*wavelength**5)**(-1) + 1)/wavelength/k

ax5.plot(t[it_start_t:it_end] - t[it_start_t], T_b_midIR[it_start_t:it_end], color='k', linewidth=3.5, label='total')
ax5.plot(t[it_start_t:it_end] - t[it_start_t], T_b_MIR_surface_only[it_start_t:it_end], color=(230/255, 140/255, 0/255), linewidth=1.2, label='surf. em.')
ax5.plot(t[it_start_t:it_end] - t[it_start_t], T_b_MIR_cloud_only[it_start_t:it_end], color='#009E73', linewidth=1.2, label='cloud em.')
ax5.set_xlim([0, t[it_end] - t[it_start_t]])
ax5.set_ylim(bottom=700)
ax5.set_xlabel('$t/d$')
ax5.set_ylabel('$T_{\mathrm{b,4.5}}$ (K)')
ax5.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1.07))
ax5.text(0.025, 0.1, 'E', transform=ax5.transAxes, size=12, weight='bold')

wavelength = 500e-9
irrad_visible_reflection_contrib = A*B_T_star_500nm*R_star**2/r**2
irrad_visible_surface_emission_contrib = (1 - A)*2*h*c**2/wavelength**5/(np.exp(h*c/(wavelength*k*T_surf)) - 1)
T_b_visible_reflection_only = h*c/np.log((irrad_visible_reflection_contrib/(2*h*c**2)*wavelength**5)**(-1) + 1)/wavelength/k
T_b_visible_surface_emission_only = h*c/np.log((irrad_visible_surface_emission_contrib/(2*h*c**2)*wavelength**5)**(-1) + 1)/wavelength/k

ax6.plot(t[it_start_t:it_end] - t[it_start_t], T_b_visib[it_start_t:it_end], color='k', linewidth=3.5, label='total')
ax6.plot(t[it_start_t:it_end] - t[it_start_t], T_b_visible_reflection_only[it_start_t:it_end], color=(230/255, 140/255, 0/255), linewidth=1.2, label='refl. starlight')
ax6.plot(t[it_start_t:it_end] - t[it_start_t], T_b_visible_surface_emission_only[it_start_t:it_end], color='#009E73', linewidth=1.2, label='surf. em.')
ax6.set_xlim([0, t[it_end] - t[it_start_t]])
ax6.set_ylim(bottom=2200)
ax6.set_xlabel('$t/d$')
ax6.set_ylabel('$T_{\mathrm{b,0.5}}$ (K)')
ax6.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1.07))
ax6.text(0.025, 0.1, 'F', transform=ax6.transAxes, size=12, weight='bold')

plt.tight_layout()
plt.savefig(figdir+'figure2.pdf', bbox_inches='tight')
plt.close()