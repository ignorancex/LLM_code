import numpy as np
import matplotlib.pyplot as plt

from re import search

from histogram import HistogramData, HistogramData2D

from plotter import enc_data_dir, enc_figure_dir
from encs.plots import plot_2d_density
from encs.plots import density_colormap

from qcd.qcd_basics import alpha_s

from encs.plots import bl_prop


# =====================================
# Flags
# =====================================
# Whether to plot OD
opendata = True
draw_np = True

# Which pythia data to plot
pythia = []
# pythia.append('qcd')
# pythia.append('w')
# pythia.append('top')
npyth  = '1M_150bins'


# =====================================
# Plot parameters
# ====================================
# 2D density plots
new_density = {
    key: {
        'axes.labelsize': 20,
        'ylim': (0, 1.0),
        # 'xlim': (5e-3, 5e-1),
        'xlim': (5e-3, 12e-1),
        'xlabel': r'$R_1$',
        'ylabel': r'$R_2/R_1$',
        'x_scale': 'log',
        'cbar_cmap': density_colormap[key]
    }
    for key in ['opendata', 'qcd', 'w', 'top']
}
old_density = {
    key: {
        'axes.labelsize': 20,
        'ylim': (0, 1.0),
        # 'xlim': (5e-3, 5e-1),
        'xlim': (5e-3, 12e-1),
        'xlabel': r'$R_L$',
        'ylabel': r'$R_S/R_L$',
        'x_scale': 'log',
        'cbar_cmap': density_colormap[key]
    }
    for key in ['opendata', 'qcd', 'w', 'top']
}

new_density['opendata']['xlim'] = (5e-3, 5e-1)
old_density['opendata']['xlim'] = (5e-3, 5e-1)


def stamp_density(ax, olddef, **metadata):
    # Dataset information
    if metadata['level'] == 'data':
        ax.text(-0.00, 1.10,
                r'$\textbf{CMS Open Data}$: '\
                '2011A Jet Primary Dataset',
                fontsize=13, transform=ax.transAxes)
    else:
        process = metadata['outstate_str']
        if process is None or not process in ['qcd', 'w', 'top']:
            raise ValueError(f"Must specify a pythia process in "
                             "[qcd, w, top] when opendata=False.")
        process_str  = r'$\texttt{Pythia 8.310},\,\,pp\to\,\,$'
        process_str += 'hadrons, ' if process == 'qcd' else \
                       (r'$W^+\,W^-\!,$ ' if process == 'w'
                        else r'$t\overline{t},$ ')
        process_str += r'$\sqrt{s}=\,$14 TeV'
        ax.text(0.0, 1.10, process_str,
                fontsize=13, transform=ax.transAxes)

    # Getting jet radius (more complicated than probably needs to be)
    jet_rad = 10*float(metadata['jet_rad'])
    rnd_digit = len(search("\.(0*)", str(jet_rad)).group(1))
    jet_rad = round(jet_rad, rnd_digit)
    jet_rad_str = metadata['jet_alg'].upper()
    if jet_rad_str in ['ANTI-KT', 'ANTIKT', 'AKT']:
        jet_rad_str = 'AK'

    jet_rad_str += str(jet_rad)[:-2] if str(jet_rad)[-2:] == '.0' \
                   else str(jet_rad)
    # Jet information
    ax.text(0.0, 1.04, jet_rad_str+' Jets,  '\
            r'$\!|\eta^\text{jet}| <$ 1.9, '\
            r'$p_T^\text{jet} \in\,$[500, 550] GeV',
            fontsize=11, transform=ax.transAxes)

    # Axes labels
    if olddef:
        ax.text(1.00, 1.130, r'$R_M$-integrated', fontsize=14,
                transform=ax.transAxes)
        rcParams = plt.rcParams.copy()
        plt.rcParams.update(plt.rcParamsDefault)
        ax.text(1.09, 1.020, 'E3C',
                font_properties=bl_prop,
                fontsize=28, transform=ax.transAxes)
        plt.rcParams.update(rcParams)
    else:
        ax.text(1.02, 1.105, r'$\phi_2$-integrated', fontsize=14,
                transform=ax.transAxes)
        ax.text(1.08, 1.025, r'$\textbf{RE3C}$', fontsize=20,
                transform=ax.transAxes)



if __name__ == "__main__":
    # ==========================================
    # Open Data
    # ==========================================
    if opendata:
        # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
        # Plots with new variables
        # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
        new_hist3d = HistogramData(
            file_name=enc_data_dir/
                  '3particle_od_100k_150bins_nus_1-00_1-00.py'
        )
        new_hist2d = plot_2d_density(
            hist_data=new_hist3d,
            # vmax=50, log_normalize=False,
            vmax=1, log_normalize=True,
            log_colorbar=False,
            save=None, **new_density['opendata'])
        ax = new_hist2d.density.axes[0]
        # Stamp
        stamp_density(ax, olddef=False, **new_hist2d.metadata)

        new_hist2d.density.fig.tight_layout()
        new_hist2d.density.savefig(
            'od_newdef_density.pdf', enc_figure_dir)

        # Fudged version of naive physics because of some phase
        # space constraints that I coded up in Mathematica after
        # pulling them out of my booty.
        # Full solution I derived is the solution to a cubic which
        # is complicated and not obviously real, but is well
        # approximated by the simpler functional form
        # below.
        # Basically, the place where i2 approaches within LambdaQCD of
        # i1 is not exactly where the NP effects kick in -- you
        # need to take into account when they become comparable
        # to the other physics (when R_2-R_1 is equal to Lambda,
        # the NP physics starts but the associated region of phase
        # space is measure zero. As R_2 increases even further,
        # _then_ the NP effects begin to become more important)
        pt = 500
        Lambda_mid  = 5/pt
        theta1s = np.logspace(np.log10(1.5e-2), np.log10(2/5),
                              100)

        ax.plot(theta1s, Lambda_mid/theta1s,
                lw=5, ls='dashed',
                color='darkgoldenrod')
        ax.plot(theta1s, 1-Lambda_mid/2/theta1s,
                lw=5, ls='dashed',
                color='mediumseagreen')

        new_hist2d.density.fig.tight_layout()
        new_hist2d.density.savefig(
            'od_nonpert_density.pdf', str(enc_figure_dir))


        # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
        # Plots with old variables
        # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
        # Basic
        old_hist3d = HistogramData(
            file_name=enc_data_dir/
                      'old_3particle_od_100k_150bins.py'
        )
        old_hist2d = plot_2d_density(
            hist_data=old_hist3d,
            vmax=2, log_colorbar=False,
            save=None, **old_density['opendata'])
        ax = old_hist2d.density.axes[0]
        stamp_density(ax, olddef=True, **old_hist2d.metadata)
        old_hist2d.density.fig.tight_layout()
        old_hist2d.density.savefig(
            str(enc_figure_dir/'od_olddef_density.pdf'))


    # ==========================================
    # Pythia
    # ==========================================
    for process in np.atleast_1d(pythia):
        if process is None:
            continue
        # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
        # Plots with new variables
        # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
        new_hist3d = HistogramData(
            file_name=enc_data_dir/
                  f'3particle_{process}_{npyth}_nus_1-00_1-00.py'
        )
        new_hist2d = plot_2d_density(
            hist_data=new_hist3d,
            vmax=1, log_colorbar=False,
            save=None,
            **new_density[process])
        ax = new_hist2d.density.axes[0]
        stamp_density(ax, olddef=False, **new_hist2d.metadata)
        new_hist2d.density.fig.tight_layout()
        new_hist2d.density.savefig(
            str(enc_figure_dir/f'supplemental/density/{process}_newdef_density.pdf'))

        # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
        # Plots with old variables
        # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
        # Basic
        old_pythiahist3d = HistogramData(
            file_name=enc_data_dir/
                  f'old_3particle_{process}_{npyth}.py'
        )

        old_pythiahist2d = plot_2d_density(
            hist_data=old_pythiahist3d,
            vmax=2, log_colorbar=False,
            save=None,
            **old_density[process])
        ax = old_pythiahist2d.density.axes[0]
        stamp_density(ax, olddef=True, **old_pythiahist2d.metadata)
        old_pythiahist2d.density.fig.tight_layout()
        old_pythiahist2d.density.savefig(
            str(enc_figure_dir/f'supplemental/density/{process}_olddef_density.pdf'))
