
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


tp_in = np.array([3850,3850,3850,3850,3850,4200,4200,4200,4200,4200,4200])
ts_in = np.array([3025, 3225, 3425, 3625, 3800, 3225, 3425, 3625, 3825, 4025, 4175])

tp_out = np.array([3849,3851,3855,3813,3850,4199, 4195,4200,4195,4214,4188])
tp_out_err = np.array([0.2, 1, 2, 3, 2, 1, 2, 2, 5, 4, 13])
ts_out = np.array([3025,3211,3405,3705,3801,3323,3453,3626,3834,4044,4187])
ts_out_err = np.array([20, 7, 25, 8, 1, 36, 17, 6, 9, 4, 13])

rin = np.array([0.31, 0.43, 0.61, 0.78, 0.95, 0.35, 0.49, 0.63, 0.79, 0.94, 0.98])
rout = np.array([0.31, 0.43, 0.61, 0.75, 0.95, 0.33, 0.49, 0.63, 0.78, 0.94, 0.97])
rerr = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

deltak = np.array([3.283,2.386,1.437,0.713,0.149,3.085,2.130,1.412,0.772,0.244,0.058])

lgp_in = np.array([4.76, 4.76, 4.76, 4.76, 4.76, 4.67, 4.67, 4.67, 4.67, 4.67, 4.67])
lgs_in = np.array([5.16, 5.06, 4.96, 4.87, 4.79, 5.06, 4.96, 4.87, 4.77, 4.70, 4.68])

lgp_out = np.array([4.76, 4.76, 4.74, 4.71, 4.98, 4.71, 4.71, 4.67, 4.68, 4.77, 4.68])
lgp_err = np.array([0.003, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.1])
lgs_out = np.array([4.91, 5.13, 5.13, 5.00, 4.52, 4.62, 4.89, 4.84, 4.71, 4.55, 4.68])
lgs_err = np.array([0.04, 0.015, 0.03, 0.005, 0.01, 0.09, 0.08, 0.04, 0.03, 0.04, 0.1])

fig, ax = plt.subplots()
ax.errorbar(rin, rin-rout, yerr = rerr, color = 'k', linestyle = 'None', capthick = 1, capsize = 4, elinewidth = 1.5)
c = ax.scatter(rin, rin-rout, s = 200, marker = '.', c = deltak, cmap = plt.cm.plasma, edgecolor = 'k', zorder = 2)
plt.colorbar(c).set_label(size = 14, label = r'$\Delta K$ (mag)')
ax.set_xlabel(r'(R2/R1)$_{in}$', fontsize = 14)
ax.set_ylabel(r'(R2/R1)$_{in}$ - (R2/R1)$_{out}$', fontsize = 14)
ax.axhline(0, label = 'Exact Recovery', linestyle = '--', color = 'k', linewidth = 1.5, zorder = 1)

plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = "large", direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')

ax.legend(loc = 'best', fontsize = 14)
plt.tight_layout()
plt.savefig('synth_rad_resid.pdf')
plt.close()


fig, ax = plt.subplots()
ax.errorbar(tp_in - tp_out, ts_in - ts_out, xerr = tp_out_err, yerr = ts_out_err, color = 'k', linestyle = 'None', capthick = 1, capsize = 4, elinewidth = 1.5)
c = ax.scatter(tp_in - tp_out, ts_in - ts_out, s = 200, marker = '.', c = deltak, cmap = plt.cm.plasma, edgecolor = 'k', zorder = 2)
fig.colorbar(c).set_label(size = 14, label = r'$\Delta K$ (mag)')

ax.axvline(0, label = r'$\Delta T_{pri} = 0$', linestyle = '-', linewidth = 1.5, color = 'k', zorder = 1  )
ax.axhline(0, label = r'$\Delta T_{sec} = 0$', linestyle = '--', linewidth = 1.5, color = 'k', zorder = 1  )

ax.set_xlabel(r'$T_{pri, in} - T_{pri,out}$ (K)', fontsize = 14)
ax.set_ylabel(r'$T_{sec, in} - T_{sec, out}$ (K)', fontsize = 14)

ax.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = "large", direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.legend(loc = 'lower left', fontsize = 14)
plt.tight_layout()


# cax, kw = mpl.colorbar.make_axes([a for a in ax.flat])
# plt.colorbar(c, cax=cax, **kw).set_label(label = r'$\Delta T$ (K)', size = 14)

plt.savefig('synth_teff_comp2.pdf')
plt.close()

fig, ax = plt.subplots()
ax.errorbar(lgp_in - lgp_out, lgs_in - lgs_out, xerr = lgp_err, yerr = lgs_err, color = 'k', linestyle = 'None', capthick = 1, capsize = 4, elinewidth = 1.5)
c = ax.scatter(lgp_in - lgp_out, lgs_in - lgs_out, s = 200, marker = '.', c = deltak, cmap = plt.cm.plasma, edgecolor = 'k', zorder = 2)
fig.colorbar(c).set_label(size = 14, label = r'$\Delta K$ (mag)')

ax.axvline(0, label = r'$\Delta \log(g)_{pri} = 0$', linestyle = '-', linewidth = 1.5, color = 'k', zorder = 1  )
ax.axhline(0, label = r'$\Delta \log(g)_{sec} = 0$', linestyle = '--', linewidth = 1.5, color = 'k', zorder = 1  )

ax.set_xlabel(r'$\log(g)_{pri, in} - \log(g)_{pri,out}$ (dex)', fontsize = 14)
ax.set_ylabel(r'$\log(g)_{sec, in} - \log(g)_{sec, out}$ (dex)', fontsize = 14)

ax.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = "large", direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.legend(loc = 'lower left', fontsize = 14)
plt.tight_layout()


# cax, kw = mpl.colorbar.make_axes([a for a in ax.flat])
# plt.colorbar(c, cax=cax, **kw).set_label(label = r'$\Delta T$ (K)', size = 14)

plt.savefig('synth_logg_comp2.pdf')

