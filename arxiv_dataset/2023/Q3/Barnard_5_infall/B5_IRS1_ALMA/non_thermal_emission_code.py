# Checking the KDE of the 1 Gaussian fit


from scipy import stats

r_proj, v_los = get_vc_r(fitfile2sigmafilteredVlsr, 'region_kde2.reg', ra_yso*u.deg, dec_yso*u.deg, 302*u.pc)
# x is projected distance
xmin = 0
xmax = 2000
# y is velocity lsr
ymin = 9
ymax = 10.5
v_lsr = 10.2 *u.km/u.s

xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
# we select only those who are not nan
gd_vlos = np.isfinite(r_proj*v_los)
values = np.vstack([r_proj[gd_vlos].value, v_los[gd_vlos].value])
kernel = stats.gaussian_kde(values)
zz = np.reshape(kernel(positions).T, xx.shape)
zz /= zz.max()  # normalization of probability

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.contourf(xx, yy, zz, cmap='Greys', levels=np.arange(0.1, 1.2, 0.1), vmin=0., vmax=1.1)
ax.axhline(v_lsr.value, color='k')
ax.set_xlabel('Projected distance (au)')
ax.set_ylabel(r"V$_{lsr}$ (km s$^{-1}$)")
ax.set_ylim([ymin, ymax])
ax.set_xlim([xmin,xmax])

# fig.savefig('B5IRS1_H2COa_ALMA_gaussfit_west_KDE.pdf', bbox_inches='tight')

# Level of non-thermal emission in the streamer

# First we observe the fitted sigma v
gc = aplpy.FITSFigure(fits.PrimaryHDU(cube.parcube[2], hdcube))
gc.show_colorscale(cmap='inferno', vmin=0, vmax=0.31)
gc.add_colorbar()
gc.set_nan_color(0.85*np.array((1,1,1)))
gc.add_beam(color='k')

# We deconvolve by the spectral resolution
Delta_chan = header['cdelt3']
corrfactor = 2
sigma_v = np.sqrt(cube.parcube[2]**2 - (Delta_chan*corrfactor/np.sqrt(8*np.log(2)))**2)
# it is a really small correction

# we calculate the thermal velocity dispersion of H2CO and subtract it from the adjusted sigma
from astropy.constants import k_B, m_p
mu_H2CO = 30.01056 # amu
mu = 2.37 # gas average weight
T_k = 9.7 * u.K
sigma_thermal = np.sqrt(k_B * T_k / (mu_H2CO * m_p)).to(u.km/u.s)
print(sigma_thermal)
soundspeed = np.sqrt(k_B * T_k / (mu * m_p)).to(u.km/u.s)
print(soundspeed)
sigma_nt = np.sqrt(sigma_v**2 - sigma_thermal.value**2) * u.km/u.s
gc = aplpy.FITSFigure(fits.PrimaryHDU(sigma_nt, hdcube), figsize=(4,4))
gc.show_colorscale(cmap='inferno', vmin=0, vmax=0.31)
gc.add_colorbar()
gc.set_nan_color(0.85*np.array((1,1,1)))
gc.add_beam(color='k')
gc.recenter(ra_yso, dec_yso, 1300/dist_B5.value/3600)
gc.show_contour(fits.PrimaryHDU(sigma_nt.value, hdcube), levels=[soundspeed.value], colors='w')

Mach_s = sigma_nt.value / soundspeed.value
gc = aplpy.FITSFigure(fits.PrimaryHDU(Mach_s, hdcube), figsize=(4,4))
gc.show_colorscale(cmap='RdYlBu_r', vmin=0, vmax=2)
gc.add_colorbar()
gc.set_nan_color(0.85*np.array((1,1,1)))
gc.add_beam(color='k')
gc.recenter(ra_yso, dec_yso, 1300/dist_B5.value/3600)

if not os.path.exists(fitfile2sigmafilteredSigmaNT):
    print("Saving non thermal velocity dispersion (sigma filtered)")
    hdcube['BUNIT'] = 'km s-1'
    fits.writeto(fitfile2sigmafilteredSigmaNT, sigma_nt.value, hdcube)
    
if not os.path.exists(fitfile2sigmafilteredMachs):
    print("Saving Mach number (sigma filtered)")
    hdcube['BUNIT'] = 'km s-1'
    fits.writeto(fitfile2sigmafilteredMachs, Mach_s, hdcube)
	
	
# is there a gradient in sigma_NT?

from scipy import stats

regionfile = 'region_kde2.reg'
# We calculate the sigma_nt per projected distance
r_proj, sigma_nt_proj = get_vc_r(fitfile2sigmafilteredSigmaNT, regionfile, ra_yso*u.deg, dec_yso*u.deg, dist_B5)
# create the grid for the kernel distribution
# x is projected distance
xmin = 0
xmax = 1600
# y is sigma_nt
ymin = 0
ymax = maxsigma
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
# we select only those who are not nan
gd_vlos = np.isfinite(r_proj*sigma_nt_proj)
values = np.vstack([r_proj[gd_vlos].value, sigma_nt_proj[gd_vlos].value])
# we calculate the kernel distribution
kernel = stats.gaussian_kde(values)
zz = np.reshape(kernel(positions).T, xx.shape)
zz /= zz.max()  # normalization of probability

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.contourf(xx, yy, zz, cmap='Greys', levels=np.arange(0.1, 1.2, 0.1), vmin=0., vmax=1.1)
ax.axhline(soundspeed.value, color='k', ls=':')
ax.set_ylim([ymin,ymax])
ax.set_xlim([xmin,xmax])
ax.set_xlabel('Projected distance (au)')
ax.set_ylabel(r"$\sigma_{NT}$ (km s$^{-1}$)")
ax.annotate('Sound speed', (1000, soundspeed.value+0.01), size=12)