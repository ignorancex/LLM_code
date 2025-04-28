import sys
sys.path.append('../B5_NOEMA_30m')

# import pyregion
from astropy.coordinates import SkyCoord, FK5
from B5setup import *
from astropy.io import fits
from astropy.wcs import WCS
import velocity_tools.stream_lines as SL
import astropy.units as u
from scipy import stats
import pickle
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import numpy as np

# Constants
M_star = 0.2 * u.Msun # from our PV diagram analysis
M_disk = 0.032 * u.Msun # corrected from 240 to 302 pc distance from Zapata et al 2014
M_env = 0.33 * u.Msun # From Brassfield and Bourke 2011
Mtot = M_star + M_disk + M_env
v_lsr = 10.2 * u.km/u.s
# inclination and position angle given by the outflow (i=0 is edge on, PA=0 lays on west)
inc = (13) * u.deg
PA_ang = (157+90)*u.deg
# exploration angles
# inc = (43) * u.deg
# PA_ang = (157+90)*u.deg
B5_c = SkyCoord(ra_yso*u.deg, dec_yso*u.deg, frame='fk5')
B5_ref = B5_c.skyoffset_frame()

# Filenames
folder = 'gaussfit_H2CO/analysis_central_chans_masked/'
imagename = folder + 'components_redshifted_envelope_Tpeak.fits'
vcname = folder + 'components_redshifted_envelope_vlsr.fits'
# regionfile = folder + 'region_kde2.reg'

# Define the figure where the widgets will be
fig = plt.figure(figsize=(10, 7))
plt.subplots_adjust(left=0.1, bottom=0.45)

# Open the image plane
hdu = fits.open(imagename)
header = hdu[0].header
wcs = WCS(header)

# Plot the image plane in one of the axes
ax = fig.add_subplot(121, projection=wcs)
imageplane = ax.imshow(hdu[0].data, vmin=0, vmax=13, origin='lower', cmap='Greys')
hdu.close()
fig.colorbar(imageplane, ax=ax)
ax.set_autoscale_on(False)
ax.plot(ra_yso, dec_yso, transform=ax.get_transform('fk5'), marker='*',
        color='red')
ax.set_xlabel('Right Ascension (J2000)')
ax.set_ylabel('Declination (J2000)')
# We add the region of the KDE to the plot
# regstreamer = pyregion.open(regionfile)
# r2 = regstreamer.as_imagecoord(header)
# patch_list, artist_list = r2.get_mpl_patches_texts()
# for p in patch_list:
#     ax.add_patch(p)
# for a in artist_list:
#     ax.add_artist(a)

# We add the axes to the image plane
x_b = np.array([1, 0, 0])*1e3/dist_B5.value
y_b = np.array([0, 0, 0])*1e3/dist_B5.value
z_b = np.array([0, 0, 1])*1e3/dist_B5.value
nx_b, ny_b, nz_b = SL.rotate_xyz(x_b, y_b, z_b, inc=inc, pa=PA_ang)
# new axes with color
my_axis_new = SkyCoord(-nx_b*u.arcsec, nz_b*u.arcsec, frame=B5_ref).transform_to(FK5)
if ny_b[-1] > 0:
    new_ax_color = 'red'
else:
    new_ax_color = 'blue'
ax.plot(my_axis_new.ra[1:], my_axis_new.dec[1:], transform=ax.get_transform('fk5'),
        color=new_ax_color)
ax.plot(my_axis_new.ra[0:2], my_axis_new.dec[0:2], transform=ax.get_transform('fk5'),
        color='red')

# Plot the velocity plane in the other axis
ax3 = fig.add_subplot(122)
ax3.set_xlabel('Projected distance (au)')
ax3.set_ylabel(r"V$_{lsr}$ (km s$^{-1}$)")

# Obtain the offset radius from IRS1 and the v_lsr for each
# pixel in the streamer region
r_proj, v_los = get_vc_r(vcname, ra_yso*u.deg, dec_yso*u.deg, dist_B5.value)
# create the grid for the kernel distribution
# x is projected distance
xmin = 0
xmax = 2000
# y is velocity lsr
ymin = 10
ymax = 11
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
# we select only those who are not nan
gd_vlos = np.isfinite(r_proj*v_los)
values = np.vstack([r_proj[gd_vlos].value, v_los[gd_vlos].value])
# we calculate the kernel distribution
kernel = stats.gaussian_kde(values)
zz = np.reshape(kernel(positions).T, xx.shape)
zz /= zz.max()  # normalization of probability
# We plot in the corresponding axis
ax3.contourf(xx, yy, zz, cmap='Greys', levels=np.arange(0.1, 1.2, 0.1), vmin=0., vmax=1.1)
ax3.axhline(v_lsr.value, color='k')
ax3.set_ylim([ymin,ymax])
ax3.set_xlim([xmin,xmax])

# Here we calculate the streamline model
def get_streamer(mass, r0, theta0, phi0, omega0, v_r0, inc, PA):
	r_c = SL.r_cent(mass, omega0, r0)
	(x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(mass=mass, r0=r0, theta0=theta0, phi0=phi0, omega=omega0, v_r0=v_r0, inc=inc, pa=PA, rmin=r_c) #, deltar=10*u.au)
	# we obtain the distance of each point in the sky
	d_sky_au = np.sqrt(x1**2 + z1**2)
	# Stream line into arcsec
	dra_stream = -x1.value / dist_B5.value
	ddec_stream = z1.value / dist_B5.value
	fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec, frame=B5_ref).transform_to(FK5)
	velocity = v_lsr + vy1
	return fil, d_sky_au, velocity

# Initial parameters
# tiny red
# r0 = 1510*u.au
# theta0 = 24 * u.deg  # rotate clockwise
# phi0 = -84 * u.deg
# v_r0 = 0.0 * u.km/u.s
# omega0 = 3.1e-13 / u.s
#blue
r0 = 2810 * u.au
theta0 = 101 * u.deg  # rotate clockwise
phi0 = 65 * u.deg
v_r0 = 0.0 * u.km/u.s
omega0 = 2.98e-13 / u.s
#red
r0 = 2810 * u.au # could also be  1010
theta0 = 22 * u.deg  # rotate clockwise, could be 24
phi0 = 242 * u.deg # could be   276
v_r0 = 0.0 * u.km/u.s
omega0 = 2.98e-13 / u.s

# Parameter steps
delta_theta0 = 1
delta_phi0 = 1
delta_r0 = 10
delta_omega0 = 1.0e-15
delta_v_r0 = 0.05

# We calculate the initial streamer
fil0, dsky0, velo0 = get_streamer(Mtot, r0, theta0, phi0, omega0, v_r0, inc, PA_ang)
r_c = SL.r_cent(Mtot,omega0,r0)
annotation = ax3.annotate(r'$r_c = {}$'.format(np.round(r_c,0)), (0.6, 0.1), xycoords='axes fraction', size=12)
line_image, = ax.plot(fil0.ra, fil0.dec, transform=ax.get_transform('fk5'), ls='-', lw=2)
line_vel, = ax3.plot(dsky0, velo0)

# We create the sliders
axcolor = 'paleturquoise'
axtheta0 = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
stheta0 = Slider(axtheta0, r'$\theta_0$', 0, 180., valinit=theta0.value, valstep=delta_theta0)
axphi0 = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
sphi0 = Slider(axphi0, r'$\phi_0$', -90, 270, valinit=phi0.value, valstep=delta_phi0)
axr0 = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor=axcolor)
sr0 = Slider(axr0, r'$r_0$', 1000, 5000, valinit=r0.value, valstep=delta_r0)
axomega0 = plt.axes([0.2, 0.25, 0.6, 0.03], facecolor=axcolor)
somega0 = Slider(axomega0, r'$\Omega_0$', 1.e-14, 15e-13, valinit=omega0.value, valstep=delta_omega0)
axv0 = plt.axes([0.2, 0.3, 0.6, 0.03], facecolor=axcolor)
sv0 = Slider(axv0, r'$v_{r,0}$', 0, 5, valinit=v_r0.value, valstep=delta_v_r0)

# In the update function, add the parameters per slider
def update(val):
    theta = stheta0.val * u.deg
    phi = sphi0.val * u.deg
    # r_c = src0.val * u.au
    omega = somega0.val / u.s
    rnew = sr0.val * u.au
    v_r = sv0.val * u.km / u.s
    fil, dsky, velo = get_streamer(Mtot, rnew, theta, phi, omega, v_r, inc, PA_ang)
    line_image.set_xdata(fil.ra)
    line_image.set_ydata(fil.dec)
    line_vel.set_xdata(dsky)
    line_vel.set_ydata(velo)
    r_cnew = SL.r_cent(Mtot, omega, rnew)
    annotation.set_text(r'$r_c = {}$'.format(np.round(r_cnew,0)))
    fig.canvas.draw_idle()

updateax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(updateax, 'Update', color=axcolor, hovercolor='0.975')
button.on_clicked(update)

plt.show()