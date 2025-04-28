from spectral_cube import SpectralCube as SC
import astropy.units as u
import pyspeckit
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

do_plot = False

n_cores = 40
rest_f =  93171.617*u.MHz # Frequency used in the observations
file_fit_thick = 'fit_N2Hp/NGC1333-N2Hp_match_fit_near_thick.fits'
file_fit_thin = 'fit_N2Hp/NGC1333-N2Hp_match_fit_near_thin.fits'
file_TdV = 'fit_N2Hp/NGC1333-N2Hp_match_TdV.fits'
file_mom1 = 'fit_N2Hp/NGC1333-N2Hp_match_Mom1.fits'
file_tp = 'fit_N2Hp/NGC1333-N2Hp_match_Tpeak.fits'
file_rms = 'fit_N2Hp/NGC1333-N2Hp_match_rms.fits'
file_in = '../data/NGC1333-N2Hp_match_kms.fits'

cube = SC.read(file_in)
cube.allow_huge_operations = True

subcube = (cube.spectral_slab(-10*u.km/u.s, 10*u.km/u.s)).with_spectral_unit(u.km/u.s)
mom0 = subcube.moment0()
subcube_mom1 = (cube.spectral_slab(4*u.km/u.s, 9.2*u.km/u.s)).with_spectral_unit(u.km/u.s)
mom1 = subcube_mom1.moment1()

spectral_axis = cube.with_spectral_unit(u.km/u.s).spectral_axis  
good_channels = (spectral_axis < -10*u.km/u.s) | (spectral_axis > 12*u.km/u.s)  
masked_cube = cube.with_mask(good_channels[:, np.newaxis, np.newaxis])  
rms = masked_cube.std(axis=0)
tp = subcube.max(axis=0)

mom0.write(file_TdV, format='fits', overwrite=True)
mom1.write(file_mom1, format='fits', overwrite=True)
tp.write(file_tp, format='fits', overwrite=True)
rms.write(file_rms, format='fits', overwrite=True)

spc = pyspeckit.Cube(subcube.hdu)
spc.Registry.add_fitter('n2hp_vtau', pyspeckit.spectrum.models.n2hp.n2hp_vtau_fitter, 4)

spc.xarr.convert_to_unit('km/s')
spc.xarr.velocity_convention = 'radio'
spc.xarr.xtype = 'velocity'

xmax = 303
ymax = 306

# range of parameters
vc_min = 6.0; vc_max = 9.0; vc_mean = 8.0
tex_min = 2.8; tex_max = 50.
tau_min = 0.01; tau_max = 30.
dv_min = 0.02; dv_max = 2.0
# offset between Mom1 and Vlsr of ~1 km/s
vc = mom1.to(u.km/u.s).value - 1.0
bad_vc = np.where((vc > vc_max) | (vc < vc_min))
vc[bad_vc] = vc_mean
# First, assemble the guesses: Tex, tau, vc, sigma_v
my_guess = np.array([7.0*np.ones(mom0.shape),
                     2.0*np.ones(mom0.shape),
                     vc,
                     0.6*np.ones(mom0.shape)])

spc.fiteach(fittype='n2hp_vtau',
            guesses=my_guess,  #spc.momentcube,
            errmap=rms.value,
            signal_cut=3,  # ignore pixels with SNR<3
            blank_value=np.nan,
            start_from_point=(xmax, ymax),
            limitedmax=[True, True, True, True],
            limitedmin=[True, True, True, True],
            maxpars=[tex_max, tau_max, vc_max, dv_max],
            minpars=[tex_min, tau_min, vc_min, dv_min],
            use_neighbor_as_guess=True,
            multicore=n_cores)
spc.write_fit(file_fit_thick, overwrite=True)

if do_plot:
    spc.mapplot()
    # show the fitted Tex
    spc.show_fit_param(0, cmap='viridis', vmin=2.7, vmax=30)
    plt.show()
    input("Press any key to continue")
    # show the fitted Tau
    spc.show_fit_param(1, cmap='viridis', vmin=1., vmax=10.)
    plt.show()
    input("Press any key to continue")
    # show the fitted centroid velocity
    spc.show_fit_param(2, cmap='RdYlBu_r', vmin=vc_min, vmax=vc_max)
    plt.show()
    input("Press any key to continue")
    # show the fitted velocity dispersion
    spc.show_fit_param(3, cmap='viridis', vmin=0, vmax=0.8)
    plt.show()
    input("Press any key to continue")

# First, assemble the guesses:
my_guess2 = np.array([15.0*np.ones(mom0.shape),
                     0.1*np.ones(mom0.shape),
                     vc,
                     0.6*np.ones(mom0.shape)])

spc.fiteach(fittype='n2hp_vtau',
            guesses=my_guess2,  #spc.momentcube,
            errmap=rms.value,
            signal_cut=3,  # ignore pixels with SNR<3
            blank_value=np.nan,
            start_from_point=(xmax, ymax),
            fixed=[False, True, False, False],
            limitedmax=[True, True, True, True],
            limitedmin=[True, True, True, True],
            maxpars=[tex_max*3, tau_max, vc_max, dv_max],
            minpars=[tex_min, tau_min, vc_min, dv_min],
            use_neighbor_as_guess=True,
            multicore=n_cores)
spc.write_fit(file_fit_thin, overwrite=True)

