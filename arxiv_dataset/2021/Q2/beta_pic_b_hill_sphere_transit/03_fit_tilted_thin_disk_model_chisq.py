import numpy as np
import matplotlib.pyplot as plt
import betapic as bp
from kepler3 import *
from exorings3 import *
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

import os, sys
import datetime
runtime = os.path.abspath((sys.argv[0])) + " run at " + datetime.datetime.now().strftime("%c")
tyb = dict(color='black', fontsize=8)

# for a given tip and tilt, you can calculate the times where you cross that chord across the disk
# if you do hit that radius, you then split the photometry into 'yes we hit it' and 'no we don't'.

# then fit for the flux change between in eclipse and out of eclipse, come up with upper limit on tau.
# the graph is then a grid of points of tip and tilt, with upper limits on the mass of the disk in that configuration for a given radius of the disk.

t_mid = 58009. * u.day # time when star has closest approach to planet

r_hill = rhill(bp.M_star, bp.M_b, bp.a_b).to(u.au) # radius of hill sphere

print('Hill sphere radius {}'.format(r_hill))

f_hill = 0.40 # radius of disk in fraction of hill sphere

r_disk = r_hill * f_hill # radius of disk
print('Disk radius {}'.format(r_disk))

angle_i   = 20   # inclination of the disk in degrees
angle_phi = 50   # tilt of the disk in degrees

impact_b = 0.2 * r_hill # impact parameter of the star
print('Impact parameter distance {}'.format(impact_b))

def ringedge(radius, i_deg, phi_deg, npoints=201):
    'returns evenly spaced points around a ring in the sky coordinates'
    th = np.linspace(0, 2*np.pi, npoints, endpoint=False)
    Xr = radius*np.cos(th)
    Yr = radius*np.sin(th)
    (xs,ys,rs) = ring_to_sky(Xr, Yr, i_deg, phi_deg)
    return (xs,ys)

vplanet = vcirc(bp.M_star, bp.M_b, bp.a_b)
print('circular velocity at planet is {}'.format(vplanet))

# make a time series dataset
t = np.linspace(t_mid-200*u.day, t_mid+200*u.day, 200)

# okay, we need a simple function to convert time to x position w.r.t. the planet
# this should be the full kepler orbit, but we'll go with:

def xp(t, t_mid=58009. * u.day, vtrans = 12.58 * u.km/u.s):
    return ((t-t_mid)*vtrans).to(u.au)

x_coord = xp(t)

f_sig = 0.01
f = np.random.normal(1.0, f_sig, t.size)
ferr = np.full(f.size, f_sig)

# now select the times where the star is within the disk radius
(xring, yring, radring) = sky_to_ring(x_coord, impact_b, angle_i, angle_phi)

indisk = (radring < r_disk)

# put in an absorption trough
taudisk = 0.1
f[indisk] = f[indisk] - taudisk

fig1, (ax1, ax2) = plt.subplots(2,1,figsize=(8,9), constrained_layout=False, sharex=True,gridspec_kw={'hspace': 0.01, 'height_ratios': [3,1]})

ax1.set_aspect('equal')
# move label from bottom of ax1 to top
# tie x axes together
# add second scale of time on ax2

# zoom in centered on beta pic b
zoom = 1.4 * r_hill

# Hill sphere
(xh,yh) = ringedge(r_hill, 90., 0.)
ax1.fill(xh, yh, color='black',alpha=0.1)

# CPD
(xr,yr) = ringedge(r_disk, angle_i, angle_phi)
ax1.fill(xr, yr, color='red',alpha=0.5)

# planet
ax1.axhline(0.0,  color='black')
ax1.axvline(0.0,  color='black')

# path of star
ax1.axhline(impact_b.value, color='green')

ax1.set_xlabel('distance [au]')
ax1.set_ylabel('distance [au]')

# plot the photometry
ax2.set_ylabel('normalised flux')

ax2.scatter(x_coord, f, color='k')
ax1.text(0.98, 0.95, runtime, ha='right', va='bottom', transform=ax1.transAxes, **tyb)

([x0a0,y0a0],[x1a0,y1a0]) = ax1.get_position().get_points()
([x0a1,y0a1],[x1a1,y1a1]) = ax2.get_position().get_points()
from matplotlib.transforms import Bbox

ax2.set_position(Bbox.from_extents([x0a0,y0a1],[x1a0,y1a1]))

plt.draw()
###plt.savefig('simdisk_a.pdf')

def disk_mass(r_disk, tau, mean_a=0.5*u.micron, mean_rho=2.5*u.g/(u.cm*u.cm*u.cm)):
    'simple mass for a face-on circular optically thin disk'

    # cadged from Mellon's derivation in thesis - p.44, eq. 4.9
    Mdisk = (4*np.pi*mean_a*mean_rho*tau*r_disk*r_disk)/3.
    return Mdisk.to(u.g)

print('estimated thin disk mass: {}'.format(disk_mass(1*u.au, 1e-3)))

# mass estimate from ALMA Perez et al. 2019a.
print('Mass estimate from Perez+ 2019: {}'.format((bp.M_b * 5e-8).to(u.g)))

def disk_lc_model(x, xlower, xupper, deltadisk=0.0, foutside=1.0):
    'simple photometric model for a disk'
    indisk = (x>xlower) * (x<xupper)
    f = np.full(x.size, foutside)
    if np.sum(indisk):
        f[indisk] = f[indisk] - deltadisk
    return f

from lmfit import Model

gmodel = Model(disk_lc_model, independent_vars=['x','xlower','xupper'])
print('parameter names: {}'.format(gmodel.param_names))
print('independent variables: {}'.format(gmodel.independent_vars))

params = gmodel.make_params(deltadisk=0.0, foutside=1.0, xlower=1.8, xupper=3.4)

n_i =   55
n_phi = 129

###n_i= 5
###n_phi = 11

i_range = np.linspace(10, 90, n_i)
phi_range = np.linspace(0, 180, n_phi)

deltaf = np.zeros((i_range.size,phi_range.size))
redchisq = np.zeros_like(deltaf)
deltaf_err = np.zeros_like(deltaf)
fout = np.zeros_like(deltaf)
fout_err = np.zeros_like(deltaf)
sanity = np.zeros_like(deltaf)

for i, curr_i in enumerate(i_range):
    for j, curr_phi in enumerate(phi_range):

        (xring, yring, radring) = sky_to_ring(x_coord, impact_b, curr_i, curr_phi)
        indisk2 = (radring < r_disk)
        nindisk = np.sum(indisk2)

        if (nindisk > 10):

            # find the edges of the disk region
            xp_lower = np.min(x_coord[indisk2])
            xp_upper = np.max(x_coord[indisk2])

            result = gmodel.fit(f, params, x=x_coord, xlower=xp_lower, xupper=xp_upper, weights=1./(ferr))
            deltaf[i][j] = result.params['deltadisk'].value
            redchisq[i][j] = result.redchi
            deltaf_err[i][j] = result.params['deltadisk'].stderr
            fout[i][j] = result.params['foutside'].value
            fout_err[i][j] = result.params['foutside'].stderr

            sanity[i][j] = curr_i

# okay, propagate the errors for fout and
# I = I_0 exo(-tau/sin(theta))
# taylor it...
# I = I_0 * (1 - tau.sin(theta))
# I = I_0 - I_0 tau sin theta)
# tau sin theta = (I_0 - I)/I0
# tau sin theta  = 1 - I/I0
# tau sin theta = 1 - (fout-deltaf)/fout
# tau sin theta = deltaf/fout

# tau = deltaf/(fout * sin(theta))

tau = deltaf/fout

tau_err = np.power(np.power((deltaf_err/deltaf),2.)+np.power((fout_err/fout),2),0.5)*tau



### simdisk_b

fig2, f2_axes = plt.subplots(3,1,
                             figsize=(6,10),
                             constrained_layout=False
                             )
(ax1,ax2,ax3) = f2_axes.flatten()

# sets the extent=() limits in imshow() so that the centre of each pixel corresponds to i and phi value
i_min = i_range.min()
i_max = i_range.max()
di = 0.5*((i_max-i_min)/(i_range.size-1))

phi_min = phi_range.min()
phi_max = phi_range.max()
dphi = 0.5*((phi_max-phi_min)/(phi_range.size-1))

im1 = ax1.imshow(deltaf,
                 extent=(phi_min-dphi,phi_max+dphi,i_min-di,i_max+di),
                 cmap=plt.cm.get_cmap('Blues', 20),
                 vmin=0,
                 vmax=taudisk*1.2,
                 origin='lower')

im2 = ax2.imshow(redchisq,
                 extent=(phi_min-dphi,phi_max+dphi,i_min-di,i_max+di),
                 cmap=plt.cm.get_cmap('Blues', 20),
                 origin='lower')

im3 = ax3.imshow(deltaf/deltaf_err,
                 extent=(phi_min-dphi,phi_max+dphi,i_min-di,i_max+di),
                 cmap=plt.cm.get_cmap('Blues', 20),
                 origin='lower')

fig2.colorbar(im1, ax=ax1)
fig2.colorbar(im2, ax=ax2)
fig2.colorbar(im3, ax=ax3)

f = 16
ax1.set_title(r'$\tau\ \sin\ \theta$',
              fontsize=f)
ax2.set_title(r'$\chi^2_r$',
              fontsize=f)
ax3.set_title(r'Signal to noise of $\tau\ \sin\ \theta$',
              fontsize=f)

for a in f2_axes.flatten():
    a.xaxis.set_major_locator(MultipleLocator(30))
    a.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    a.yaxis.set_major_locator(MultipleLocator(30))
    a.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    a.xaxis.set_minor_locator(MultipleLocator(10))
    a.yaxis.set_minor_locator(MultipleLocator(10))

    a.set_xlim(-10,190)
    a.set_ylim(-10,100)
    a.scatter(angle_phi, angle_i, color='red')

aa = fig2.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Tilt $\phi$ [deg]", fontsize=16)

aa.xaxis.set_label_coords(0.4, -0.04)

plt.ylabel(r'Inclination $ \theta $ [deg]', fontsize=16)
aa.yaxis.set_label_coords(-0.08, 0.5)
    
plt.savefig('paper/figs/simdisk_b.pdf', bbox_inches='tight')

plt.show()
