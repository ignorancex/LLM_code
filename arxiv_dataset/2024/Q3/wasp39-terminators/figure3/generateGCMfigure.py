import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
import astropy.units as u

# ============================================= Setting up grid for plotting ===========================================
fig = plt.figure(figsize=(8, 10))

subfigs = fig.subfigures(3, 1, wspace=0.2, height_ratios=[1., 1., 0.2])
axs0 = subfigs[0].subplots(2, 2, sharex=True)
subfigs[0].suptitle('Aerosols')
axs0[0, 0].set_title('Condensate clouds')
axs0[1, 0].set_title('Photochemical hazes')

axs1 = subfigs[1].subplots(2, 2, sharex=True)
subfigs[1].suptitle('Gas-phase Chemistry')
axs1[0, 0].set_title('Equilibrium chemistry')
axs1[1, 0].set_title('Transport-induced diseq. chem.')

# add labels & set axis limits for all of them
for axs in [axs0, axs1]:
    axs[1, 0].set_xlabel('$\lambda$ ($\mu$m)')
    axs[1, 1].set_xlabel('$\lambda$ ($\mu$m)')
    for i in range(2):
        axs[i, 0].set_ylabel('Limbs depth (ppm)')
        axs[i, 1].set_ylabel('Difference (ppm)')
        axs[i, 1].yaxis.set_label_position("right")
        axs[i, 1].yaxis.tick_right()

    axs[0, 0].set_xlim(2, 5.50)
    for i in range(2):
        axs[i, 1].set_ylim(-1000, 3000)

        axs[i, 1].axhline(0, lw=0.5, color='k')

modelcolor_e = np.array((255, 85, 40)) / 255.
modelcolor_m = np.array((110, 147, 230)) / 255.

datacolor_e = np.array((246, 195, 183)) / 255.
datacolor_m = np.array((196, 212, 246)) / 255.

# =================================== Read in & plot observational data ================================================
# # Load in observational data
fname = 'catwoman_res100_priorlds.txt'
data = np.loadtxt(fname, skiprows=17)
wl_obs = data[:, 0]
Td_obs_evening = data[:, 1]
err_obs_evening = data[:, 2]
Td_obs_morning = data[:, 3]
err_obs_morning = data[:, 4]
cov_obs = data[:, 5]
diff_obs = data[:, 6]
err_diff_obs = data[:, 7]

# Plot observational data
for axs in [axs0, axs1]:
    for i in range(2):
        axs[i, 0].scatter(wl_obs, Td_obs_morning, color=datacolor_m, marker='o', s=10.0, label='catwoman morning',
                          zorder=1)
        axs[i, 0].errorbar(wl_obs, Td_obs_morning, yerr=err_obs_morning, fmt='none', color=datacolor_m, zorder=1)

        axs[i, 0].scatter(wl_obs, Td_obs_evening, color=datacolor_e, marker='o', s=10.0, label='catwoman evening',
                          zorder=1)
        axs[i, 0].errorbar(wl_obs, Td_obs_evening, yerr=err_obs_evening, fmt='none', color=datacolor_e, zorder=1)

        axs[i, 1].scatter(wl_obs, diff_obs, c='0.5', marker='o', s=10.0,
                          zorder=1)
        axs[i, 1].errorbar(wl_obs, diff_obs, yerr=err_diff_obs, fmt='none', c='0.5', zorder=1)

# ==================================== Plot different forward models ===================================================
# .................................... Clouds ..........................................................................
offset = 0
aerosol_dataset = nc.Dataset('WASP_39b_IWFGraz_cloudmodel.nc')
wl_aerosols = aerosol_dataset['wavelength'][:].data
Rstar = (0.932 * u.Rsun).to(u.m).value
data_evening = (aerosol_dataset['evening_spectrum'][:].data / Rstar) ** 2
data_morning = (aerosol_dataset['morning_spectrum'][:].data / Rstar) ** 2

axs0[0, 0].plot(wl_aerosols, data_evening * 100 / 2 + offset, color=modelcolor_e)
axs0[0, 0].plot(wl_aerosols, data_morning * 100 / 2 + offset, color=modelcolor_m)

# Factor 1/2 in two previous lines is because radius from model calculation is for full circle;
# catwoman data shows transit depth for half-circle

axs0[0, 1].plot(wl_aerosols, (data_evening - data_morning) / 2 * 100, color='k')

# .................................... Photochemical hazes and disequilibrium chemistry ................................
filenames = ['haze_steinrueck_30nm',
             'clear_zamyatina_equilibrium_hlev',
             'clear_zamyatina_disequilibrium_hlev']
offsets = np.zeros(3)
offsets = np.array((-600, -600, -600))
# offsets=np.array((0,0,0))
xker = np.arange(91) - 20
sigma = 30 / (
            2. * np.sqrt(2.0 * np.log(2.0)))  # 5.5 is FWHM of resolution element in model wavelength grid "coordiantes"
yker = np.exp(-0.5 * (xker / sigma) ** 2.0)  # making a gaussian
yker /= yker.sum()  # normaliize

for i in range(len(filenames)):
    if i == 0:
        ax = axs0[1, :]
    elif i == 1:
        ax = axs1[0, :]
    else:
        ax = axs1[1, :]

    data_morning = np.loadtxt(
        'kempton_group_' +
        filenames[i] + '_morning.txt', skiprows=3)
    data_evening = np.loadtxt(
        'kempton_group_' +
        filenames[i] + '_evening.txt', skiprows=3)
    #::::: binning spectra ::::::
    # evening terminator
    wl_micron = data_evening[:, 0]
    wl_binned = np.linspace(2, 6, 600)

    conv = np.convolve(data_evening[:, 1], yker, mode='same')
    Td_binned_e = np.interp(wl_binned, data_evening[:, 0], conv)
    Td_binned_e = 0.5 * Td_binned_e  # Factor 0.5 is because radius from model calculation is for full circle;
    # catwoman data shows transit depth for half-circle
    ax[0].plot(wl_binned, Td_binned_e * 1e4 + offsets[i], color=modelcolor_e)

    # morning terminator
    wl_micron = data_morning[:, 0]
    wl_binned = np.linspace(2, 6, 600)

    conv = np.convolve(data_morning[:, 1], yker, mode='same')
    Td_binned_m = np.interp(wl_binned, data_morning[:, 0], conv)
    Td_binned_m = 0.5 * Td_binned_m  # Factor 0.5 is because radius from model calculation is for full circle;
    # catwoman data shows transit depth for half-circle
    ax[0].plot(wl_binned, Td_binned_m * 1e4 + offsets[i], color=modelcolor_m)
    ax[1].plot(wl_binned, (Td_binned_e - Td_binned_m) * 1e4, color='k')
    print(np.mean(Td_binned_e - Td_binned_m))

custom_lines = [Line2D([0], [0], color=modelcolor_e, lw=1, label='evening terminator', ),
                Line2D([0], [0], color=modelcolor_m, lw=1, label='morning terminator'),
                Line2D([0], [0], color='k', lw=1, label='difference (evening-morning)'),
                Line2D([0], [0], marker='o', color='w', label='catwoman evening',
                       markerfacecolor=datacolor_e),
                Line2D([0], [0], marker='o', color='w', label='catwoman morning',
                       markerfacecolor=datacolor_m),
                Line2D([0], [0], marker='o', color='w', label='catwoman difference',
                       markerfacecolor='0.5'),
                ]

subfigs[2].legend(handles=custom_lines, loc='center', ncol=2)

labellist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
for i in range(2):
    for j in range(2):
        at0 = AnchoredText(labellist[i + j * 2], loc=2, pad=0.2, borderpad=0.2, frameon=False)
        at0.txt._text.set_size(16)
        axs0[j, i].add_artist(at0)

        at1 = AnchoredText(labellist[4 + i + j * 2], loc=2, pad=0.2, borderpad=0.2, frameon=False)
        at1.txt._text.set_size(20)
        axs1[j, i].add_artist(at1)

plt.savefig('GCMfigure.pdf')

plt.show()
