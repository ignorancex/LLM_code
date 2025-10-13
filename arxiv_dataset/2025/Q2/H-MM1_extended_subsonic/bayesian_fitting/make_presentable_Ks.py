""" Generating figures for the upcoming letter """

import numpy as np
import matplotlib.pylab as plt
import astropy.units as u
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from skimage import morphology
from astropy.io import fits
import aplpy
from spectra_xy_list import xlist, ylist, labels

# sane defaults for aplpy-generated figures
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rc('text', usetex=True)
plt.rc('figure', figsize=(4, 6))
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')

mle_x1_file = 'nested-sampling/H-MM1-mle-x1.fits'
gascontourfile = 'data/sample_sig.fits'
Kfile = 'nested-sampling/H-MM1-Ks.fits'
Kcut = 5 # the heuristical ln(Z1/Z2) cut for model selection
distance = 138.4 # in pc
region_lab = 'H-MM1'
beam_text = 'GBT Beam'
fig_size = (5.5, 3.5)
bar_size = 0.1
v_min = 3
v_max = 4.5
v_ticks = [3, 3.3, 3.6, 3.9, 4.2, 4.5]

def header_flatten(head):
    """ Flattens 3D header into 2D. Surely it exists somewhere already... """
    flathead = head.copy()
    for key in flathead.keys():
        if key.endswith('3'):
            flathead.pop(key)
    flathead['NAXIS'] = 2
    flathead['WCSAXES'] = 2

    return flathead


def show_filtered_contours(fig, Karr, Kcut=Kcut, skimage_kwargs={},
                           header=header_flatten(fits.getheader(Kfile)),
                           **kwargs):
    """ Overplots contours with small features removed """
    min_size = skimage_kwargs.pop('min_size', 3)
    levels = kwargs.pop('level', [0.5])
    Karr_clean = morphology.remove_small_objects(Karr > Kcut,
                                                 min_size=min_size).astype(int)
    hdu_clean = fits.PrimaryHDU(Karr_clean,
                                header=header_flatten(fits.getheader(Kfile)))
    fig.show_contour(data=hdu_clean, levels=levels, **kwargs)


Ks = fits.getdata(Kfile)

gas_contour_kwargs = dict(levels=[0], linewidths=1,
                          colors='#ef6548', linestyles='--')
gas_contour_kwargs_on_v = dict(levels=[0], linewidths=0.5,
                          colors='0.45', linestyles='--')
kcut_contour_kwargs = dict(linewidths=0.5, colors='white')

hdu_K01 = fits.PrimaryHDU(Ks[0], header=header_flatten(fits.getheader(Kfile)))
fig01 = aplpy.FITSFigure(hdu_K01, figsize=fig_size)
fig01.show_colorscale(cmap='viridis', vmin=-2, vmax=7000,
                      stretch='power', exponent=0.3)
fig01.set_nan_color('#f0f0f0')
fig01.ticks.set_color('black')
fig01.add_colorbar()
fig01.show_contour(gascontourfile, **gas_contour_kwargs)
show_filtered_contours(fig01, Ks[0], **kcut_contour_kwargs)
fig01.colorbar.set_axis_label_text(r'$\ln K^{\mathrm{1}}_{\mathrm{0}}$')
fig01.colorbar.set_ticks([0, 1000, 2000, 4000, 7000])
fig01.add_beam()
fig01.beam.set_color('black')
fig01.beam.set_corner('top right')
# 138.4 pc distance --> scale bar of 1 pc
fig01.add_scalebar((bar_size*u.pc.to(u.au) / distance)*u.arcsec)
fig01.scalebar.set_corner('bottom left')
fig01.scalebar.set_label('{0} pc'.format(bar_size))
fig01.add_label( 0.15, 0.125, region_lab, relative=True)
fig01.add_label( 0.95, 0.925, beam_text, relative=True, horizontalalignment='right')
fig01.savefig("figs/Ks_01_map_v1.pdf") #, dpi=140, adjust_bbox='tight')
plt.close()

hdu_K21 = fits.PrimaryHDU(Ks[1],
                          header=header_flatten(fits.getheader(Kfile)))
fig21 = aplpy.FITSFigure(hdu_K21, figsize=fig_size)
fig21.show_colorscale(cmap='viridis', vmin=-2, vmax=140, 
                      stretch='power', exponent=0.5)
fig21.set_nan_color('#f0f0f0')
fig21.ticks.set_color('black')
fig21.add_colorbar()
fig21.show_contour(gascontourfile, **gas_contour_kwargs)
show_filtered_contours(fig21, Ks[1], **kcut_contour_kwargs)
fig21.colorbar.set_axis_label_text(r'$\ln K^{\mathrm{2}}_{\mathrm{1}}$')
fig21.colorbar.set_ticks([0, 20, 40, 80, 120])
fig21.add_beam()
fig21.beam.set_color('black')
fig21.beam.set_corner('top right')
# 300 pc distance --> scale bar of 1 pc
fig21.add_scalebar((bar_size*u.pc.to(u.au) / distance)*u.arcsec)
fig21.scalebar.set_corner('bottom left')
fig21.scalebar.set_label('{0} pc'.format(bar_size))
fig21.add_label( 0.15, 0.125, region_lab, relative=True)
fig21.add_label( 0.95, 0.925, beam_text, relative=True, horizontalalignment='right')
fig21.savefig("figs/Ks_21_map_v1.pdf") #, dpi=140, adjust_bbox='tight')
plt.close()


hdu_K32 = fits.PrimaryHDU(Ks[2],
                          header=header_flatten(fits.getheader(Kfile)))
fig32 = aplpy.FITSFigure(hdu_K32, figsize=fig_size)
fig32.show_colorscale(cmap='viridis', vmin=-2, vmax=140, 
                      stretch='power', exponent=0.5)
fig32.set_nan_color('#f0f0f0')
fig32.ticks.set_color('black')
fig32.add_colorbar()
fig32.show_contour(gascontourfile, **gas_contour_kwargs)
show_filtered_contours(fig32, Ks[2], **kcut_contour_kwargs)
fig32.colorbar.set_axis_label_text(r'$\ln K^{\mathrm{3}}_{\mathrm{2}}$')
fig32.colorbar.set_ticks([0, 20, 40, 80, 120])
fig32.add_beam()
fig32.beam.set_color('black')
fig32.beam.set_corner('top right')
# 300 pc distance --> scale bar of 1 pc
fig32.add_scalebar((bar_size*u.pc.to(u.au) / distance)*u.arcsec)
fig32.scalebar.set_corner('bottom left')
fig32.scalebar.set_label('{0} pc'.format(bar_size))
fig32.add_label( 0.15, 0.125, region_lab, relative=True)
fig32.add_label( 0.95, 0.925, beam_text, relative=True, horizontalalignment='right')
fig32.savefig("figs/Ks_32_map_v1.pdf") #, dpi=140, adjust_bbox='tight')
plt.close()

'''

hdu_K43 = fits.PrimaryHDU(Ks[3],
                          header=header_flatten(fits.getheader(Kfile)))
fig43 = aplpy.FITSFigure(hdu_K43, figsize=fig_size)
fig43.show_colorscale(cmap='viridis', vmin=-2, vmax=140, 
                      stretch='power', exponent=0.5)
fig43.set_nan_color('#f0f0f0')
fig43.ticks.set_color('black')
fig43.add_colorbar()
fig43.show_contour(gascontourfile, **gas_contour_kwargs)
show_filtered_contours(fig43, Ks[3], **kcut_contour_kwargs)
fig43.colorbar.set_axis_label_text(r'$\ln K^{\mathrm{3}}_{\mathrm{2}}$')
fig43.colorbar.set_ticks([0, 20, 40, 80, 120])
fig43.add_beam()
fig43.beam.set_color('black')
fig43.beam.set_corner('top right')
# 300 pc distance --> scale bar of 1 pc
fig43.add_scalebar((bar_size*u.pc.to(u.au) / distance)*u.arcsec)
fig43.scalebar.set_corner('bottom left')
fig43.scalebar.set_label('{0} pc'.format(bar_size))
fig43.add_label( 0.15, 0.125, region_lab, relative=True)
fig43.add_label( 0.95, 0.925, beam_text, relative=True, horizontalalignment='right')
fig43.savefig("figs/Ks_43_map_v1.pdf", dpi=140, adjust_bbox='tight')
plt.close()
'''

# not sure which one is better... so why not both?
for flag_noise, suffix in zip([True, False], ['_clean', '_all']):
    mle1 = fits.getdata(mle_x1_file)
    min_size = 9
    if flag_noise:
        Karr_clean = morphology.remove_small_objects(Ks[0] > 0,
                                                     min_size=min_size)
        mle1[4][(Ks[0] < 0) | ~Karr_clean] = np.nan

    hdu_xoff = fits.PrimaryHDU(mle1[4],
                               header=header_flatten(fits.getheader(Kfile)))
    figxoff = aplpy.FITSFigure(hdu_xoff, figsize=fig_size)
    figxoff.show_colorscale(cmap='RdYlBu_r', vmin=v_min, vmax=v_max)
    figxoff.set_nan_color('#f0f0f0')
    figxoff.ticks.set_color('black')
    figxoff.add_colorbar()
    gas_contour_kwargs.update({'linewidths': 0.7})  # hey that's not cool!
    figxoff.show_contour(gascontourfile, **gas_contour_kwargs_on_v)
    show_filtered_contours(figxoff, Ks[0], Kcut=0,
                           skimage_kwargs=dict(min_size=min_size),
                           linewidths=0.7, colors='black')
    figxoff.colorbar.set_axis_label_text(
        r'$V_{\mathrm{lsr}}~\mathrm{(km~s^{-1})}$')
    figxoff.colorbar.set_ticks(v_ticks)
    figxoff.add_beam()
    figxoff.beam.set_color('black')
    figxoff.beam.set_corner('top right')
    # 300 pc distance --> scale bar of 1 pc
    figxoff.add_scalebar((bar_size*u.pc.to(u.au) / distance)*u.arcsec)
    figxoff.scalebar.set_corner('bottom left')
    figxoff.scalebar.set_label('{0} pc'.format(bar_size))
    figxoff.add_label( 0.15, 0.125, region_lab, relative=True)
    figxoff.add_label( 0.95, 0.925, beam_text, relative=True, 
        horizontalalignment='right')
    figxoff.savefig("figs/mle_xoff{}_v1.pdf".format(suffix)) #, dpi=140, 
       # adjust_bbox='tight')



# make the ln(K)>Kcut based map of LoS component numbers
npeaks_map = np.full_like(Ks[0], np.nan)
npeaks_map[Ks[0] <= Kcut] = 0
npeaks_map[Ks[0] > Kcut] = 1
Karr_clean = morphology.remove_small_objects(Ks[0] > Kcut,
                                             min_size=3).astype(int)
                                             
npeaks_map[(Karr_clean == 0) & np.isfinite(npeaks_map)] = 0

npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut)] = 2
npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut) & (Ks[2] > Kcut)] = 3
#npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut) & (Ks[2] > Kcut) & (Ks[3] > Kcut)] = 4
hdu_npeaks = fits.PrimaryHDU(npeaks_map,
                             header=header_flatten(fits.getheader(mle_x1_file)))
hdu_npeaks.writeto('nested-sampling/npeaks_cut5.fits', overwrite=True)


from astropy.wcs import WCS
wcs = WCS(hdu_npeaks.header)

dxlist = np.array([1, 2, -6, 3, -4])
dylist = np.array([1, -2, -4, 3, 3])
ra_list, dec_list = wcs.all_pix2world(xlist, ylist, 0)
ra_list_text, dec_list_text = wcs.all_pix2world(xlist + dxlist, ylist + dylist, 0)
dra_list = ra_list - ra_list_text
ddec_list = dec_list - dec_list_text

fig = plt.figure(figsize=fig_size)
ax = plt.subplot(projection=wcs)
cmaplst = ['#7fc97f', '#fdc086', '#beaed4', '#ffff99']   #  # stolen from colorbrewer2
lcmap = ListedColormap(cmaplst)
lcmap.set_bad('#f0f0f0')
plt.imshow(hdu_npeaks.data, vmin=-0.5, vmax=3.5, origin='lower', cmap=lcmap)

'''
for (x, y) in zip(xlist, ylist):
    r = Circle((x, y), radius=.75, zorder=1, fc="0.8", ec="k", lw=0.2)
    ax.add_patch(r)
for i in range(len(labels)):
    plt.annotate(labels[i], xy=(xlist[i], ylist[i]), xycoords='data',
            xytext=(xlist[i]+dxlist[i], ylist[i]+dylist[i]), #textcoords='data',
            bbox=dict(boxstyle="circle", fc="0.8", alpha=0.8),
            arrowprops=dict(arrowstyle="-", shrinkA=0, shrinkB=0,
                            connectionstyle="arc3, rad=0.2",
                            alpha=0.8))
'''
plt.xlabel('RA (J2000)')
plt.ylabel('Dec (J2000)')
cb = plt.colorbar(ticks=[0, 1, 2, 3], aspect=20, pad=0.03, shrink=0.975)
plt.contour( fits.getdata(gascontourfile), **gas_contour_kwargs)
fig.savefig("figs/npeaks_map_v2.pdf") #, dpi=140, bbox_inches='tight')

plt.close()
