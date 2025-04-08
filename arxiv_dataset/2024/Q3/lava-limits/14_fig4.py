import numpy as np
import matplotlib.pyplot as plt
import h5py
import cartopy.crs as ccrs
import matplotlib.colors as colors
import os
"""
this script plots figure 4
written by Yangcheng Luo
"""

def compute_visible_brightness(lat, lon_ext, tau):
    B = 0.5*tau/(1 + 0.5*tau)

    # No reflection on the nightside
    B[:, :90] = 0
    B[:, 271:] = 0

    B_ext = np.zeros((len(lat), len(lon_ext)))
    B_ext[:, :90] = B[:, 270:]
    B_ext[:, 90:450] = B
    B_ext[:, 450:] = B[:, :91]

    return B, B_ext

def compute_IR_brightness(lat, lon_ext, tau, Bsurf):
    Bcloud = 0.2
    tau = tau*0.1 # Ratio of tau_LW to tau_SW

    B = Bsurf*np.exp(-tau) + Bcloud*(1 - np.exp(-tau))
    B_ext = np.zeros((len(lat), len(lon_ext)))
    B_ext[:, :90] = B[:, 270:]
    B_ext[:, 90:450] = B
    B_ext[:, 450:] = B[:, :91]

    return B, B_ext

def compute_phase_curve(lat, lon_ext, B_ext, phase_angle, filename, outdir):
    phase_curve = np.zeros_like(phase_angle)

    for k in range(len(phase_angle)):
        print('Computing '+filename+' phase angle = '+str(phase_angle[k])+' deg')
        for i in range(len(lon_ext)):
                if abs(np.radians(lon_ext[i]) - np.radians(phase_angle[k])) < np.pi/2:
                    for j in range(len(lat)):
                        phase_curve[k] += B_ext[j, i]*np.cos(np.radians(lat[j]))*np.radians(1)**2*np.cos(abs(np.radians(lon_ext[i]) - np.radians(phase_angle[k])))*np.cos(np.radians(lat[j]))

    # Transit
    phase_curve[:17] = -999999
    phase_curve[344:] = -999999

    # Secondary eclipse
    phase_curve[164:197] = 0

    with h5py.File(outdir+filename+'.h5', 'w') as f:
        f.create_dataset('phase_angle', data=phase_angle)
        f.create_dataset('phase_curve', data=phase_curve[::-1])

    return phase_curve

def plot_hemisphere(lat, lon, B, central_lon, figname, spectral_band, flag_colorbar, figdir):
    proj = ccrs.Orthographic(central_longitude=central_lon, central_latitude=0.0)

    norm = colors.Normalize(vmin=0, vmax=1)
    plt.figure(figsize=(2.5, 2.5))
    ax = plt.axes(projection=proj)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    if spectral_band == 'vis':
        pcolormesh = ax.pcolormesh(lon_grid, lat_grid, B, norm=norm, cmap='binary_r', transform=ccrs.PlateCarree(), rasterized=True)
        if flag_colorbar == True:
            plt.colorbar(pcolormesh, ax=ax, orientation='vertical', label='relative visible brightness')
    elif spectral_band == 'IR':
        pcolormesh = ax.pcolormesh(lon_grid, lat_grid, B, norm=norm, cmap='plasma', transform=ccrs.PlateCarree(), rasterized=True)
        if flag_colorbar == True:
            plt.colorbar(pcolormesh, ax=ax, orientation='vertical', label='relative mid-IR brightness')
    plt.savefig(figdir+figname+'.pdf', bbox_inches='tight', dpi=1000)
    plt.close()

def plot_phase_curve(filename,figdir,outdir):
    f = h5py.File(outdir+filename+'.h5', 'r')

    plt.figure(figsize=(4, 2))
    plt.plot(f['phase_angle'][:], f['phase_curve'][:],c="k")
    plt.xlabel('phase angle ($\degree$)')
    plt.ylabel('relative brightness')
    plt.xlim([-180, 180])
    plt.xticks([-180,-90,0,90,180])
    plt.ylim([-1, 3])
    plt.savefig(figdir+filename+'_curve.pdf', bbox_inches='tight')
    plt.close()

lat = np.linspace(-90, 90, 181)
lon = np.linspace(-180, 179, 360)
lon_ext = np.linspace(-270, 270, 541)
phase_angle = np.linspace(-180, 180, 361)

# set up figure directory 
figdir = "./figs/fig4/"
os.makedirs(figdir, exist_ok=True)
# set up output directory 
outdir = "./out/fig4/"
os.makedirs(outdir, exist_ok=True)

# Stage 1

tau1 = np.zeros((len(lat), len(lon)))

B1vis, B1vis_ext = compute_visible_brightness(lat, lon_ext, tau1)
plot_hemisphere(lat, lon, B1vis, 0, 'B1vis_dayside', 'vis', False,figdir)
plot_hemisphere(lat, lon, B1vis, 180.0, 'B1vis_nightside', 'vis', False,figdir)
phase_curve = compute_phase_curve(lat, lon_ext, B1vis_ext, phase_angle, 'B1vis',outdir)
plot_phase_curve('B1vis',figdir,outdir)

B1surf = np.zeros((len(lat), len(lon)))
B1surf[:, 90:271] = np.cos(np.radians(lat[:, None]))*np.cos(np.radians(lon[None, 90:271]))

B1IR, B1IR_ext = compute_IR_brightness(lat, lon_ext, tau1, B1surf)
plot_hemisphere(lat, lon, B1IR, 0, 'B1IR_dayside', 'IR', False,figdir)
plot_hemisphere(lat, lon, B1IR, 180, 'B1IR_nightside', 'IR', False,figdir)
phase_curve = compute_phase_curve(lat, lon_ext, B1IR_ext, phase_angle, 'B1IR',outdir)
plot_phase_curve('B1IR',figdir,outdir)

# Stage 2

tau2 = np.zeros((len(lat), len(lon)))
tau2[:, :135] = 20
tau2[:, 135:226] = -20/90*(lon[135:226] - 45)
tau2[:, 226:270] = 0
tau2[:, 270:] = 20

B2vis, B2vis_ext = compute_visible_brightness(lat, lon_ext, tau2)
plot_hemisphere(lat, lon, B2vis, 0, 'B2vis_dayside', 'vis', False,figdir)
plot_hemisphere(lat, lon, B2vis, 180, 'B2vis_nightside', 'vis', False,figdir)
phase_curve = compute_phase_curve(lat, lon_ext, B2vis_ext, phase_angle, 'B2vis',outdir)
plot_phase_curve('B2vis',figdir,outdir)

B2surf = np.zeros((len(lat), len(lon)))
B2surf[:, 90:271] = np.cos(np.radians(lat[:, None]))*np.cos(np.radians(lon[None, 90:271]))

B2IR, B2IR_ext = compute_IR_brightness(lat, lon_ext, tau2, B2surf)
plot_hemisphere(lat, lon, B2IR, 0, 'B2IR_dayside', 'IR', False,figdir)
plot_hemisphere(lat, lon, B2IR, 180, 'B2IR_nightside', 'IR', False,figdir)
phase_curve = compute_phase_curve(lat, lon_ext, B2IR_ext, phase_angle, 'B2IR',outdir)
plot_phase_curve('B2IR',figdir,outdir)

# Stage 3

tau3 = np.zeros((len(lat), len(lon)))
tau3[:] = 20

B3vis, B3vis_ext = compute_visible_brightness(lat, lon_ext, tau3)
plot_hemisphere(lat, lon, B3vis, 0, 'B3vis_dayside', 'vis', False,figdir)
plot_hemisphere(lat, lon, B3vis, 180, 'B3vis_nightside', 'vis', False,figdir)
phase_curve = compute_phase_curve(lat, lon_ext, B3vis_ext, phase_angle, 'B3vis',outdir)
plot_phase_curve('B3vis',figdir,outdir)

B3surf = np.zeros((len(lat), len(lon)))
B3surf[:, 90:271] = np.cos(np.radians(lat[:, None]))*np.cos(np.radians(lon[None, 90:271]))/3

B3IR, B3IR_ext = compute_IR_brightness(lat, lon_ext, tau3, B3surf)
plot_hemisphere(lat, lon, B3IR, 0, 'B3IR_dayside', 'IR', False,figdir)
plot_hemisphere(lat, lon, B3IR, 180, 'B3IR_nightside', 'IR', False,figdir)
phase_curve = compute_phase_curve(lat, lon_ext, B3IR_ext, phase_angle, 'B3IR',outdir)
plot_phase_curve('B3IR',figdir,outdir)

# Stage 4

tau4 = np.zeros((len(lat), len(lon)))
tau4[:] = 1

B4vis, B4vis_ext = compute_visible_brightness(lat, lon_ext, tau4)
plot_hemisphere(lat, lon, B4vis, 0, 'B4vis_dayside', 'vis', False,figdir)
plot_hemisphere(lat, lon, B4vis, 180, 'B4vis_nightside', 'vis', True,figdir)
phase_curve = compute_phase_curve(lat, lon_ext, B4vis_ext, phase_angle, 'B4vis',outdir)
plot_phase_curve('B4vis',figdir,outdir)

B4surf = np.zeros((len(lat), len(lon)))
B4surf[:, 90:271] = np.cos(np.radians(lat[:, None]))*np.cos(np.radians(lon[None, 90:271]))/3

B4IR, B4IR_ext = compute_IR_brightness(lat, lon_ext, tau4, B4surf)
plot_hemisphere(lat, lon, B4IR, 0, 'B4IR_dayside', 'IR', False,figdir)
plot_hemisphere(lat, lon, B4IR, 180, 'B4IR_nightside', 'IR', True,figdir)
phase_curve = compute_phase_curve(lat, lon_ext, B4IR_ext, phase_angle, 'B4IR',outdir)
plot_phase_curve('B4IR',figdir,outdir)
