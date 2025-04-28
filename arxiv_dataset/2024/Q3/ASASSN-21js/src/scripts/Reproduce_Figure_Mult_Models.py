from Main_Functions import *
import paths

mjd1, flux1, resi1, _ = onedisk_model_mov(250, 84.2, 5, 0.745, 27.5, 0.27, 60145, t_end = 63000, t_step = 9,
                                          r_star = 15, plot_title = False, save_plot = False)
mjd2, flux2, resi2, _ = twodisk_model_mov(250, 83.21, 5.9, 0.75, 65, 1.0, 31.5, 0.17, 60675, t_end = 63000, t_step = 9,
                                          r_star = 15, plot_title=False, save_plot = False)
mjd3, flux3, resi3, _ = fourdisk_model_mov(385, 83.21, 5.9, 0.99, 115, 250, 0.75, 65, 31.5, 0.17, 60675, t_end = 63000, 
                                           t_step = 9, r_star = 15, plot = True, plot_title=False, save_plot = False)
#set t
t = real_lc_MJD
t_step = 9
t_end = 63000
t_pred = np.arange(max(t)+t_step, t_end+t_step, t_step)
t_tot = np.concatenate((t, t_pred))

#build figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 10), gridspec_kw = {'height_ratios': [1.5, 0.5]}, sharex = True)

#plot original data
real_data = ax1.errorbar(x = real_lc_MJD, y = real_lc_flux, yerr = real_lc_fluxerr, 
            c = 'k', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5, label = 'binned data')

#plot one disk
model_onedisk = ax1.plot(mjd1[:len(t)], flux1[:len(t)], color = 'blue', linewidth = 3, alpha = 0.5, label = 'one disk model')
model_onedisk_pred = ax1.plot(t_pred, flux1[(len(t)):], color = 'blue', linestyle = '--', linewidth = 3, label = 'one disk model predicted data')

#plot residuals
y0 = ax2.hlines(0, xmin = (min(mjd)-50), xmax = (max(mjd)+50), colors = 'grey', linestyles = '--')
res_model_onedisk = ax2.errorbar(x = real_lc_MJD, y = resi1, yerr = real_lc_fluxerr, c = 'blue', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5)

#plot one ring
model_onering = ax1.plot(mjd2[:len(t)], flux2[:len(t)], color = 'orange', linewidth = 3, alpha = 0.65, label = 'one ring model')
model_onering_pred = ax1.plot(t_pred, flux2[(len(t)):], color = 'orange', linestyle = '--', linewidth = 3, label = 'one ring model predicted data')

#plot residuals
res_model_onering = ax2.errorbar(x = real_lc_MJD, y = resi2, yerr = real_lc_fluxerr, c = 'orange', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5)

#plot two rings
model_tworing = ax1.plot(mjd3[:len(t)], flux3[:len(t)], color = 'cyan', linewidth = 3, alpha = 0.65, label = 'two rings model')
model_tworing_pred = ax1.plot(t_pred, flux3[(len(t)):], color = 'cyan', linestyle = '--', linewidth = 3, label = 'two rings model predicted data')

#plot residuals
res_model_tworing = ax2.errorbar(x = real_lc_MJD, y = resi3, yerr = real_lc_fluxerr, c = 'cyan', markersize = '6', elinewidth = 0.5, fmt = '.', alpha = 0.5)

#set limits
ax1.set_xlim((min(mjd)-10), (max(mjd)))
ax1.set_ylim((min(flux) - 0.05), 1.05)
ax2.set_xlim((min(mjd)-10), (max(mjd)))

#set legend and grid
ax1.legend(fontsize = 16)
ax1.tick_params(direction = 'in', labelsize = 16)
ax2.tick_params(direction = 'in', labelsize = 16)

#set axes labels
ax1.set_ylabel('Normalized flux', fontsize = 16)
ax2.set_xlabel('MJD', fontsize = 16)
ax2.set_ylabel('Residuals', fontsize = 16)

plt.subplots_adjust(hspace = 0)

plt.savefig(paths.figures / 'asassn21js_combined_model.pdf', dpi = 300, facecolor = 'white')
