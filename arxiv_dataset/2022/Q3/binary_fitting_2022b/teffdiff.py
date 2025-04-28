import numpy as np
import matplotlib.pyplot as plt

m15_teff = np.array([4093, 3622, 3700, 4132, 4246, 3443, 3800, 3955])

k22_pt = np.array([4475, 3739, 3875, 4422, 4379, 3506, 3900, 4197])
k22_st = np.array([3711, 3274, 3488, 3945, 4129, 3371, 3688, 3718])

hot_teff = np.array([4471, 3738, 3892, 4442, 4392, 3498, 3913, 4183])

deltakep = [0.795, 1.874, 0.842, 0.501, 0.379, 1.306, 0.280, 0.741]
deltak = [0.018, 1.163,0.069,0.011,0.099,0.932,-0.001, 0.245]

print('avg primary teff change: ', np.mean(hot_teff-m15_teff), r'$\pm$', np.std(hot_teff - m15_teff), ', max difference:', max(hot_teff-m15_teff), ', min difference: ', min(hot_teff-m15_teff))
print('avg secondary teff change: ', np.mean(m15_teff - k22_st), r'$\pm$', np.std(m15_teff - k22_st), ', max difference:', max(m15_teff - k22_st), ', min difference: ', min(m15_teff - k22_st))

# k22c_pt = np.array([4504, 3958, 4455, 4494, 3468, 3966, 4305])
# k22c_st = np.array([3714, 3505, 3967, 4036, 3776, 3706, 3711])
# k22c_rr = np.array([1.27, 1.11, 1.15, 1.1, 0.48, 1.09, 1.12])

k22_rr = np.array([1.23, 0.73, 1.07, 1.12, 1.05, 0.66, 1.09, 1.05])

# print('avg pri teff change ', np.median(k22c_pt - k22_pt), 'sec teff', np.median(k22c_st - k22_st), 'radius ratio', np.median(k22c_rr - k22_rr))


fix, ax = plt.subplots()
ax.scatter(m15_teff, k22_st - m15_teff, marker = '.', s = 100, color = 'darkorange', label = 'Secondary', zorder = 1)
ax.scatter(m15_teff, k22_pt - m15_teff, marker = '.', s = 100, color = 'darkblue', label = 'Primary', zorder = 1)
for n, t in enumerate(k22_pt):
	ax.plot([m15_teff[n], m15_teff[n]], [k22_st[n]-m15_teff[n], t - m15_teff[n]], color = 'k', linewidth = 2, zorder = 0.5)
ax.axhline(0, label = 'No difference', linestyle = '--', color = 'k')


plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'T$_{eff}$ (M13; K)', fontsize = 14)
ax.set_ylabel('{}'.format(r'T$_{eff}$ residual (this work - M13; K)'), fontsize = 14)
ax.legend(loc = 'best', fontsize = 12)
plt.tight_layout()
plt.savefig('teff_diff.pdf')



fix, ax = plt.subplots()
ax.scatter(deltak, k22_st-m15_teff, marker = '.', s = 100, color = 'darkorange', label = 'Secondary', zorder = 1)
ax.scatter(deltak, k22_pt-m15_teff, marker = '.', s = 100, color = 'darkblue', label = 'Primary', zorder = 1)
for n, t in enumerate(k22_pt):
	ax.plot([deltak[n], deltak[n]], [k22_st[n]-m15_teff[n], t-m15_teff[n]], color = 'k', linewidth = 2, zorder = 0.5)
ax.axhline(0, label = 'No difference', linestyle = '--', color = 'k')
plt.minorticks_on()
plt.gca().invert_xaxis()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'$\Delta$Kep (mag)', fontsize = 14)
ax.set_ylabel('{}'.format(r'T$_{eff}$ residual (this work - M13; K)'), fontsize = 14)
ax.legend(loc = 'best', fontsize = 12)
plt.tight_layout()
plt.savefig('deltak_diff.pdf')
