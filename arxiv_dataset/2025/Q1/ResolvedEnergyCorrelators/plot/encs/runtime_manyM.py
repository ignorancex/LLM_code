import numpy as np
from scipy.optimize import curve_fit

from encs.plots import plot_runtime
from plotter import enc_data_dir, enc_figure_dir, combine_plotters

from utils.plot_utils import adjust_lightness


# =====================================
# Plot parameters
# =====================================
# 1D plots
runtime_plot_params = {
    'axes.labelsize': 20,
    'ylim': (1e-4, 5e3),
    'xlim': (5, 120),
    'y_scale': 'log',
    'x_scale': 'log'
}
fill_between = False


# For fill_between
color_1d = 'cornflowerblue'
t_ind_1d = 5
color_3d = 'firebrick'
t_ind_3d = 35
color_5d = 'mediumorchid'
t_ind_5d = 40
fill_alpha = 0.2

# For estimating scaling
def poly_fit(x, a, b, c):
    return a*pow(x, b) + c

fit_func = 'integer_degree'


# =====================================
# Main
# =====================================
if __name__ == "__main__":
    # =====================================
    # Opendata Plots
    # =====================================
    # ---------------------------------
    # 2 particles:
    # ---------------------------------
    hist_1d, od_1d = plot_runtime(
            file_name=enc_data_dir/
                    '2particle_od_100k_200bins_nu1-00.py',
            color=color_1d,
            label=r'PENC (Any $N$)',
            save=None,
            **runtime_plot_params,
    )
    # Preparing for fill_between
    means_1d = np.array(hist_1d.metadata['runtime_means'])/1000
    stds_1d  = np.array(hist_1d.metadata['runtime_stds'])/1000
    nans     = np.isnan(means_1d) * np.isnan(stds_1d)
    # Removing nans
    nums_1d  = np.arange(0, len(means_1d))[~nans]
    means_1d = means_1d[~nans]
    stds_1d  = stds_1d[~nans]
    # Getting slope
    if fit_func == 'integer_degree':
        fit_1d, _ = curve_fit(lambda x, a: a*x**2, nums_1d, means_1d)
    else:
        fit_1d, _ = curve_fit(fit_func, nums_1d, means_1d)

    # ---------------------------------
    # 3 particles:
    # ---------------------------------
    hist_3d, od_3d = plot_runtime(
            file_name=enc_data_dir/
                    '3particle_od_100k_150bins_nus_1-00_1-00.py',
            color=color_3d,
            label='RE3C',
            save=None,
            **runtime_plot_params,
    )
    # Preparing for fill_between
    means_3d = np.array(hist_3d.metadata['runtime_means'])/1000
    stds_3d  = np.array(hist_3d.metadata['runtime_stds'])/1000
    nans     = np.isnan(means_3d) * np.isnan(stds_3d)
    # Removing nans
    nums_3d  =np.arange(0, len(means_3d))[~nans]
    means_3d = means_3d[~nans]
    stds_3d  = stds_3d[~nans]
    # Getting slope
    if fit_func == 'integer_degree':
        fit_3d, _ = curve_fit(lambda x, a: a*x**2, nums_3d, means_3d)
    else:
        fit_3d, _ = curve_fit(fit_func, nums_3d, means_3d)

    # ---------------------------------
    # 4 particles:
    # ---------------------------------
    hist_5d, od_5d = plot_runtime(
            file_name=enc_data_dir/
                    '4particle_od_10k_10bins_nus_1-00_1-00_1-00.py',
            color=color_5d,
            alpha=0.2,
            label='RE4C',
            save=None,
            **runtime_plot_params,
    )
    # Preparing for fill_between
    means_5d = np.array(hist_5d.metadata['runtime_means'])/1000
    stds_5d  = np.array(hist_5d.metadata['runtime_stds'])/1000
    nans     = np.isnan(means_5d) * np.isnan(stds_5d)
    # Removing nans
    nums_5d  =np.arange(0, len(means_5d))[~nans]
    means_5d = means_5d[~nans]
    stds_5d  = stds_5d[~nans]
    # Getting slope
    if fit_func == 'integer_degree':
        fit_5d, _ = curve_fit(lambda x, a: a*x**4, nums_5d, means_5d)
    else:
        fit_5d, _ = curve_fit(fit_func, nums_5d, means_5d)

    # ---------------------------------
    # Plotting:
    # ---------------------------------
    # Combining plots
    od_plotters = [od_1d, od_3d, od_5d]

    cplotter = combine_plotters(od_plotters)
    ax = cplotter.axes[0]

    # Filling standard deviations
    if fill_between:
        ax.fill_between(nums_1d,
                means_1d-stds_1d, means_1d+stds_1d,
                color=color_1d, alpha=fill_alpha)
        ax.fill_between(nums_3d,
                means_3d-stds_3d, means_3d+stds_3d,
                color=color_3d, alpha=fill_alpha)
        ax.fill_between(nums_5d,
                means_5d-stds_5d, means_5d+stds_5d,
                color=color_5d, alpha=fill_alpha)

    # Adding slope information
    if fit_func == 'integer_degree':
        fit_xs = np.logspace(0, 3, 100)

        fit_line_1d, = ax.plot(fit_xs, fit_1d[0]*fit_xs**2.,
                               ls='dashed',
                               color=adjust_lightness(color_1d, 1.2),)
        fit_line_3d, = ax.plot(fit_xs, fit_3d[0]*fit_xs**3.,
                               ls='dashed',
                               color=adjust_lightness(color_1d, 1.5),)
        fit_line_5d, = ax.plot(fit_xs, fit_5d[0]*fit_xs**4.,
                               ls='dashed',
                               color=adjust_lightness(color_1d, 1.7),)
        fit_labels = [r'$M^2$', r'$M^3$', r'$M^4$']
        legend2 = ax.legend(fit_lines, fit_labels,
                            loc=(0.42, 0.74), handletextpad=0.2)
    else:
        poly_1d = fit_1d[1]
        poly_3d = fit_3d[1]
        poly_5d = fit_5d[1]
        ax.text(nums_1d[t_ind_1d], means_1d[t_ind_1d]/3,
                              rf"$t \sim M^{{{str(poly_1d)[:4]}}}$",
                              color=adjust_lightness(color_1d,0.5),
                              size=18)
        ax.text(nums_3d[t_ind_3d], means_3d[t_ind_3d]/3,
                              rf"$t \sim M^{{{str(poly_3d)[:4]}}}$",
                              color=adjust_lightness(color_3d,0.5),
                              size=18)
        ax.text(nums_5d[t_ind_5d], means_5d[t_ind_5d]*8,
                              rf"$t \sim M^{{{str(poly_5d)[:4]}}}$",
                              color=adjust_lightness(color_5d,0.5),
                              size=18)

    # Formatting and saving
    ax.set_xlabel(r'Number of Particles ($M$)')
    ax.legend(loc=(0.38, 0.02))
    cplotter.fig.tight_layout()
    cplotter.savefig('opendata_runtimes.pdf',
                     enc_figure_dir/'supplemental/')
