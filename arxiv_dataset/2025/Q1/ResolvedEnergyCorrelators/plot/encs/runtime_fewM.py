import numpy as np
from scipy.optimize import curve_fit
from re import search

from plotter import Plotter, enc_figure_dir

from utils.plot_utils import adjust_lightness


# =====================================
# Plot parameters
# =====================================
# 1D plots
runtime_plot_params = {
    'axes.labelsize': 20,
    'ylim': (5e-4, 8e3),
    'xlim': (5, 130),
    'y_scale': 'log',
    'x_scale': 'log'
}
fill_between = True


colors = {
    '1d': 'cornflowerblue',
    '3d': 'firebrick',
    '5d': 'mediumorchid'
}
fit_colors = {
    '1d': adjust_lightness('cornflowerblue', 1.2),
    '3d': adjust_lightness('firebrick', 1.6),
    '5d': adjust_lightness('mediumorchid', 1.4),
}
labels = {
    '1d': r'PENC (Any $N$)',
    '3d': 'RE3C',
    '5d': 'RE4C'
}



# For estimating scaling
def poly_fit(x, a, b, c):
    return a*pow(x, b)

def log_fit(x, a, b, c, d):
    return pow(x, b)*(a + c*np.log10(x/d))

fit_func = 'integer_degree'


# =====================================
# Main
# =====================================
if __name__ == "__main__":
    nums = [5, 10, 25, 50, 100, 130]

    # 2 particles:
    runtimes_ms = {
        '1d': np.array([3.205, 8.400, 44.157, 327.334,
                        825.644, 1469.395])/1000,
        '3d': np.array([6.927, 43.154, 658.897, 5823.861,
                        42507.488, 92590.226])/1000,
        '5d': np.array([7.645, 100.549, 5352.817, 150986.699,
                        2155941.691, 5570511.067])/1000
    }
    # Getting slope
    runtime_fit = {
        key: curve_fit(fit_func, nums, runtime)[0]
                if fit_func == poly_fit else(
                    curve_fit(fit_func, nums, runtime,
                              p0=[1, i+2, 1, 1], maxfev=100000
                             )[0] if fit_func == log_fit else (
                       curve_fit(lambda x, a: a*x**(i+2), nums,
                                 runtime)[0]
                             ))
        for i, (key, runtime) in enumerate(runtimes_ms.items())
    }

    plotter = Plotter(**runtime_plot_params)
    ax = plotter.axes[0]
    for key in runtimes_ms.keys():
        ax.plot(nums, runtimes_ms[key],
                color=colors[key], label=labels[key])

    fit_lines = []
    fit_labels = []
    fit_xs = np.logspace(0, 3, 100)
    for i, key in enumerate(runtimes_ms.keys()):
        # scipy's fit
        fit_params = runtime_fit[key]

        # My manual fit
        # fit_params = manual_fit[key]

        # Line
        if fit_func == 'integer_degree':
            fit_ys = fit_xs**(i+2)*fit_params[0]
            fit_line, = ax.plot(fit_xs, fit_ys, ls='dashed',
                                color=fit_colors[key],)
            label = rf'$M^{{{int(i+2)}}}$'
        else:
            fit_line, = ax.plot(fit_xs, fit_func(fit_xs, *fit_params),
                                ls='dashed', color=fit_colors[key],)
            degree = round(runtime_fit[key][1], 1)
            if str(degree)[-1] == '0':
                degree = int(degree)
            label = rf'$M^{{{degree}}}\log M$' \
                        if fit_func == poly_fit else(
                            rf'$M^{{{degree}}}\log M$')

        fit_lines.append(fit_line)
        fit_labels.append(label)

    legend2 = ax.legend(fit_lines, fit_labels,
                        loc=(0.42, 0.74), handletextpad=0.2)
    legend = ax.legend(loc=(0.02, 0.75), handletextpad=0.5)
    ax.add_artist(legend2)



    # ---------------------------------
    # Plotting:
    # ---------------------------------
    # Filling standard deviations

    # Adding slope information
    # ax.text(nums[t_ind_1d], means_1d[t_ind_1d]/3,
    #                       rf"$t \sim M^{{{str(poly_1d)[:4]}}}$",
    #                       color=adjust_lightness(color_1d,0.5),
    #                       size=18)
    # ax.text(nums[t_ind_3d], means_3d[t_ind_3d]/3,
    #                       rf"$t \sim M^{{{str(poly_3d)[:4]}}}$",
    #                       color=adjust_lightness(color_3d,0.5),
    #                       size=18)
    # ax.text(nums[t_ind_5d], means_5d[t_ind_5d]*8,
    #                       rf"$t \sim M^{{{str(poly_5d)[:4]}}}$",
    #                       color=adjust_lightness(color_5d,0.5),
    #                       size=18)

    # Stamping
    ax.text(0.96, 0.11,
            r'$\textbf{CMS Open Data}$: '\
            '2011A Jet Primary Dataset',
            fontsize=13, ha='right',
            transform=ax.transAxes)
    ax.text(0.96, 0.05, r'AK5 Jets,  '\
            r'$\!|\eta^\text{jet}| <$ 1.9, '\
            r'$p_T^\text{jet} \in\,$[500, 550] GeV',
            fontsize=11, ha='right',
            transform=ax.transAxes)

    # Formatting and saving
    ax.set_ylabel('Runtime per Jet (ms)')
    ax.set_xlabel(r'Particles per Jet ($M$)')
    plotter.fig.tight_layout()

    plotter.savefig('opendata_runtimes_fewM.pdf',
                     enc_figure_dir/'supplemental/')
