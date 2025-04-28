import numpy as np
from itertools import product

import matplotlib.pyplot as plt

from histogram import HistogramData
from histogram import HistogramData2D

from plotter import enc_data_dir, enc_figure_dir
from utils.plot_utils import adjust_lightness
from utils.postprocess import collision_stamp

from encs.plots import bl_prop
from encs.plots import plot_2d_bullseye, bullseye_inset
from encs.plots import density_colormap

# For decimal point truncation
from re import search


# =====================================
# Flags
# =====================================
# Whether to plot OD
opendata = True

pythia = []
# Which pythia data to plot
# pythia.append('qcd')
# pythia.append('w')
# pythia.append('top')

npyth3 = '1M_150bins'
npyth4 = '1M_25bins'


# =====================================
# Flags
# =====================================
# Choice of value
theta1s_od = [0.03, 0.1, 0.3]
theta1s_py = [0.6]
t2_t1_val  = 0.8
phi2_vals  = [np.pi/4, np.pi/2, np.pi-0.001]


# =====================================
# Plot parameters
# =====================================
new_3part_params = {
    key: {
        'ylim': (0, 1.0),
        'cbar_cmap': density_colormap[key],
        # 'vmin': 1e0,
        # 'vmax': 7e1,
    } for key in ['opendata', 'qcd', 'w', 'top']
}

new_4part_params = {
    key: {
        'ylim': (0, 1.0),
        'cbar_cmap': density_colormap[key],
        # 'vmin': 1e0,
        # 'vmax': 4e3
    } for key in ['opendata', 'qcd', 'w', 'top']
}


old_3part_params = {
    key: {
        'ylim': (0, 1.0),
        'cbar_cmap': density_colormap[key],
        # 'vmin': 1e0,
        # 'vmax': 4e3,
    } for key in ['opendata', 'qcd', 'w', 'top']
}


def stamp_bullseye(ax, bins, Npoint=3, olddef=False, **metadata):
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
    ax.text(-0.252, 0.065, jet_rad_str+' Jets, '\
            r'$\!|\eta^\text{jet}| <$ 1.9',
            fontsize=12, transform=ax.transAxes)
    ax.text(-0.243, 0.000, r'$p_T^\text{jet} \in\,$[500, 550] GeV',
            fontsize=12, transform=ax.transAxes)

    # Dataset information
    if metadata['level'] == 'data':
        ax.text(-0.25, -0.072,
                r'$\textbf{CMS Open Data}:$ '\
                '2011A Jet Primary Dataset',
                fontsize=14, transform=ax.transAxes)
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
        ax.text(-0.25, -0.072, process_str,
                fontsize=14, transform=ax.transAxes)

    # Variable info
    inum   = -1
    xshift = 0.36 * (1 + (len(bins)-1)/2)
    for bname, bval in bins.items():
        var = ''
        num = ''
        if bname.startswith('theta') or bname.startswith('R'):
            var = 'R'
            inum += 1
        elif bname.startswith('phi'):
            var = 'phi'
        num = bname[-1]
        rnd_digit = len(search("\.(0*)", str(bval[1]-bval[0])
                               ).group(1))
        lowbnd = str(round(bval[0], rnd_digit+1))
        highbnd = str(round(bval[1], rnd_digit+1))

        if var == 'R':
            yshift = 0
            xshift -= 0.35
            bounds = fr'${var}_{num}$' if inum == 0 else\
                fr'${var}_{num}/{var}_{inum}$'
        elif var == 'phi':
            xshift += 0.003
            yshift += 0.065
            bounds  = fr'$\{var}_{num}\,$'
        else:
            raise ValueError(f"Invalid {var=} from {bname=}")
        bounds += fr"$\,\in [{lowbnd}, {highbnd}]$"
        ax.text(0.67-xshift + 0.075*(0 if var == 'R' else 1),
                1.045-yshift, bounds, fontsize=12,
                transform=ax.transAxes)
        if var == 'phi':
            xshift -= 0.005
            yshift -= 0.065

    # Axes labels
    if Npoint == 3:
        label1 = 'L' if olddef else '1'
        label2 = 'S' if olddef else '2'
        labelphi = r'L\text{-}S' if olddef else '2'
    elif Npoint == 4:
        if olddef:
            raise ValueError(f"Invalid {Npoint=} for {olddef=}")
        label1 = '2'
        label2 = '3'
        labelphi = '3'
    else:
        raise ValueError(f"Invalid {Npoint=}")

    ax.text(1.0, 0.8, rf'$\phi_{{{labelphi}}}$', fontsize=18,
            color=adjust_lightness('dimgrey', 0.4),
            transform=ax.transAxes)
    ax.text(0.92, 0.05, rf'$R_{label2}/R_{label1}$', fontsize=16,
            color=adjust_lightness('dimgrey', 0.4),
            transform=ax.transAxes)

    if olddef:
        assert Npoint == 3, \
            "{Npoint=} not supported, must be 3."
        rcParams = plt.rcParams.copy()
        plt.rcParams.update(plt.rcParamsDefault)
        ax.text(1.11, 1.01, 'E3C',
                font_properties=bl_prop,
                fontsize=28, transform=ax.transAxes)
        plt.rcParams.update(rcParams)
    else:
        ax.text(1.06, 1.035,
                rf'$\textbf{{RE{Npoint}C}}$',
                fontsize=24, transform=ax.transAxes)


def bullseye_arrows(ax):
    # Drawing curves and arrows
    rad_1  = 0.55
    phi_1  = 0.8
    phi_2  = -0.85
    rad_2i = 0.52
    rad_2f = 0.75
    # Angular arrow
    phis = np.arange(0, phi_1, .01)
    rads = np.ones_like(phis)*rad_1
    ax.plot(.5+rads*np.cos(phis), .5+rads*np.sin(phis),
            lw=2, color='dimgrey', clip_on=False,
            transform=ax.transAxes)
    ax.arrow(.5+rad_1*np.cos(phis[-1]),
             .5+rad_1*np.sin(phis[-1]),
             -1e-10*np.sin(phis[-1]), 1e-10*np.cos(phis[-1]),
             clip_on=False, width=0.0,
             facecolor='dimgrey', edgecolor='dimgrey',
             head_width=3e-2,
             head_length=4e-2,
             transform=ax.transAxes)
    # Radial arrow
    ax.plot([.5+rad_2i*np.cos(phi_2),.5+rad_2f*np.cos(phi_2)],
            [.5+rad_2i*np.sin(phi_2),.5+rad_2f*np.sin(phi_2)],
            lw=2, color='dimgrey', clip_on=False,
            transform=ax.transAxes)
    ax.arrow(.5+rad_2f*np.cos(phi_2),
             .5+rad_2f*np.sin(phi_2),
             1e-10*np.cos(phi_2), 1e-10*np.sin(phi_2),
             clip_on=False, width=0.0,
             facecolor='dimgrey', edgecolor='dimgrey',
             head_width=3e-2,
             head_length=4e-2,
             transform=ax.transAxes)



if __name__ == "__main__":
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
    # Open Data:
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
    if opendata:
        for t1_val in theta1s_od:
            # ==========================================
            # Open Data: 3-Particles, new variables
            # ==========================================
            # Black-and-white (main text)
            new_hist3d = HistogramData(
                file_name=enc_data_dir/
                      '3particle_od_100k_150bins_nus_1-00_1-00.py'
            )

            t1_ind = np.digitize([t1_val],
                                 new_hist3d.edges['theta1'])[0] - 1
            t1 = new_hist3d.centers['theta1'][t1_ind]
            t1_bin = (new_hist3d.edges['theta1'][t1_ind],
                      new_hist3d.edges['theta1'][t1_ind+1])
            # Setting up bullseye
            new_hist3d = plot_2d_bullseye(
                hist_data=new_hist3d,
                save=None, theta1=t1_val,
                **new_3part_params['opendata'],
                vmin=5e-4/t1**4.2, vmax=1.5e2/t1**1.8,
            )
            # Creating inset
            bullseye_inset(new_hist3d, inner_rad=t1,
                           inner_var='theta1')
            # Stamping
            bullseye_arrows(new_hist3d.bullseye.axes[0])
            stamp_bullseye(new_hist3d.bullseye.axes[0],
                           {'theta_1': t1_bin},
                           **new_hist3d.metadata)

            # Saving
            new_hist3d.bullseye.fig.tight_layout()
            # new_hist3d.bullseye.savefig(
            #     f'/od/od_3particle_bullseye_{t1}.pdf', enc_figure_dir)

            # ==========================================
            # Open Data: 3-Particle Plots, old variables
            # ==========================================
            # Setup
            old_hist3d = HistogramData(
                file_name=enc_data_dir/
                          'old_3particle_od_100k_150bins.py'
            )
            tL_ind = np.digitize([t1_val],
                                 old_hist3d.edges['thetaL'])[0] - 1
            tL = old_hist3d.centers['thetaL'][tL_ind]
            tL_bin = (old_hist3d.edges['thetaL'][tL_ind],
                      old_hist3d.edges['thetaL'][tL_ind+1])

            # Making bullseye
            old_hist3d = plot_2d_bullseye(
                hist_data=old_hist3d,
                save=None, theta1=tL,
                **old_3part_params['opendata'],
                vmin=3e-3/tL**4.2, vmax=8e1/tL**2.2,
            )

            # Decorating plot
            ax = old_hist3d.bullseye.axes[0]
            bullseye_arrows(ax)
            stamp_bullseye(ax, {'theta_L': tL_bin},
                           olddef=True,
                           **old_hist3d.metadata)

            # Saving
            old_hist3d.bullseye.fig.tight_layout()
            old_hist3d.bullseye.savefig(
                f'/od/od_3particle_old_bullseye_{t1}.pdf',
                enc_figure_dir)

            # ==========================================
            # Open Data: 4-Particle
            # ==========================================
            # Setup
            t2_t1_key = 'theta2_over_theta1'
            t3_t2_key = 'theta3_over_theta2'

            # Histogram processing
            new_hist5d = HistogramData(
                file_name=enc_data_dir/
                   f"4particle_od_100k_25bins_nus_1-00_1-00_1-00.py",
                variable_order=['theta1', t2_t1_key, 'phi2',
                                t3_t2_key, 'phi3']
            )

            # Getting actual histogram centers
            # t1
            t1_ind = np.digitize([t1_val],
                         new_hist5d.edges['theta1'])[0]-1
            t1 = new_hist5d.centers['theta1'][t1_ind]
            t1_bin = (new_hist5d.edges['theta1'][t1_ind],
                      new_hist5d.edges['theta1'][t1_ind+1])
            # t2/t1
            t2_ind = np.digitize([t2_t1_val],
                         new_hist5d.edges['theta2_over_theta1'])[0]-1
            t2_t1 = new_hist5d.centers['theta2_over_theta1'][t2_ind]
            t2_bin = (new_hist5d.edges['theta2_over_theta1'][t2_ind],
                      new_hist5d.edges['theta2_over_theta1'][t2_ind+1])

            pre_bullseye_4part = new_hist5d.\
                                    get_sub_histogram('theta1', t1).\
                                    get_sub_histogram(t2_t1_key,
                                                      t2_t1)

            for phi2_val in phi2_vals:
                # phi2
                phi2_ind = np.digitize([phi2_val],
                             new_hist5d.edges['phi2'])[0]-1
                phi2 = new_hist5d.centers['phi2'][phi2_ind]
                phi2_bin = (new_hist5d.edges['phi2'][phi2_ind],
                            new_hist5d.edges['phi2'][phi2_ind+1])

                # Get data
                bullseye_4part = HistogramData2D(
                    hist_data = pre_bullseye_4part.\
                                    get_sub_histogram('phi2', phi2)
                )

                # bin edges and centers
                bin3_edges = bullseye_4part.edges[t3_t2_key]
                t3_edges   = t1*t2_t1*bin3_edges
                bin3_centers = bullseye_4part.centers[t3_t2_key]
                t3_centers   = t1*t2_t1*bin3_centers

                bullseye_4part.hist = np.array([
                    bullseye_4part.hist[i3] / t3
                    for i3, t3 in enumerate(t3_centers)
                ])

                # Readjusting for correct defn of angles (not ratios)
                bullseye_4part.edges[t3_t2_key] = t3_edges
                bullseye_4part.centers[t3_t2_key] = t3_centers

                # Making plot
                bullseye_4part.make_plot('bullseye',
                         log_norm=True, save=None,
                         radial_coordinate='theta3_over_theta2',
                         **new_4part_params)
                ax = bullseye_4part.bullseye.axes[0]
                ax.set_ylim(0, t1*t2_t1)

                # Inset and stamping
                bullseye_inset(bullseye_4part,
                               inner_rad=t1*t2_t1,
                               inset_pts=[(t1, -phi2)])
                bullseye_arrows(ax)
                stamp_bullseye(bullseye_4part.bullseye.axes[0],
                               {'theta_1': t1_bin,
                                'theta_2': t2_bin,
                                'phi_2':   phi2_bin},
                               Npoint=4, **new_hist5d.metadata)

                # Saving
                bullseye_4part.bullseye.fig.tight_layout()
                bullseye_4part.bullseye.savefig(
                    f'/od/od_4particle_bullseye_{t1}_{t2_t1}_{phi2}.pdf',
                    enc_figure_dir)

    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
    # Pythia:
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
    for process in np.atleast_1d(pythia):
        # Getting style/colormap
        for t1_val in theta1s_py:
            # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
            # 3-Particle Plots with new variables
            # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
            new_hist3d = HistogramData(
                file_name=enc_data_dir/
                      f'3particle_{process}_{npyth3}_nus_1-00_1-00.py',
                variable_order=['theta1', 'theta2_over_theta1', 'phi']
            )
            t1_ind = np.digitize([t1_val],
                     new_hist3d.edges['theta1'])[0] - 1
            t1 = new_hist3d.centers['theta1'][t1_ind]
            t1_bin = (new_hist3d.edges['theta1'][t1_ind],
                      new_hist3d.edges['theta1'][t1_ind + 1])

            new_hist3d = plot_2d_bullseye(
                hist_data=new_hist3d,
                save=None, theta1=t1_val,
                **new_3part_params[process],
                vmin=8e-3/t1**3.0, vmax=2.0e0/t1**2.6,
            )
            # Inset
            bullseye_inset(new_hist3d, inner_rad=t1,
                           inner_var='theta1')
            # Stamping
            bullseye_arrows(new_hist3d.bullseye.axes[0])
            stamp_bullseye(new_hist3d.bullseye.axes[0],
                           {'theta_1': t1_bin},
                           opendata=False, process=pythia,
                           **new_hist3d.metadata)
            # Saving
            new_hist3d.bullseye.fig.tight_layout()
            new_hist3d.bullseye.savefig(
                f'/supplemental/3particle/{process}{t1}_3particle_bullseye.pdf',
                enc_figure_dir)

            # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
            # 3-Particle Plots with old variables
            # =:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
            # Setup
            old_hist3d = HistogramData(
                file_name=enc_data_dir/
                          f'old_3particle_{process}_{npyth3}.py'
            )
            tL_ind = np.digitize([t1_val],
                                 old_hist3d.edges['thetaL'])[0] - 1
            tL = old_hist3d.centers['thetaL'][tL_ind]
            tL_bin = (old_hist3d.edges['thetaL'][tL_ind],
                      old_hist3d.edges['thetaL'][tL_ind+1])

            # Making bullseye
            old_hist3d = plot_2d_bullseye(
                hist_data=old_hist3d,
                save=None, theta1=tL,
                **old_3part_params[process],
                vmin=5e-2/tL**2.8, vmax=8.0e0/tL**2.7,
            )

            # Decorating plot
            ax = old_hist3d.bullseye.axes[0]
            bullseye_arrows(ax)
            stamp_bullseye(ax, {'theta_L': tL_bin},
                           olddef=True,
                           **old_hist3d.metadata)

            # Saving
            old_hist3d.bullseye.fig.tight_layout()
            old_hist3d.bullseye.savefig(
                f'/supplemental/wedge/{process}{tL}_3particle_bullseye.pdf',
                enc_figure_dir)

            # ==========================================
            # Pythia: 4-Particle
            # ==========================================
            # Setup
            t2_t1_key = 'theta2_over_theta1'
            t3_t2_key = 'theta3_over_theta2'

            # Histogram processing
            try:
                new_hist5d = HistogramData(
                    file_name=enc_data_dir/
                       f"4particle_{process}"
                       f"_{npyth4}_nus_1-00_1-00_1-00.py",
                    variable_order=['theta1', t2_t1_key, 'phi2',
                                    t3_t2_key, 'phi3']
                )
            except Exception as exc:
                print("Invalid variable order in: ",
                       enc_data_dir/
                       f"4particle_{process}"
                       f"_{npyth4}_nus_1-00_1-00_1-00.py")
                continue

            # Getting actual histogram centers
            t1_ind = np.digitize([t1_val],
                         new_hist5d.edges['theta1'])[0]-1
            t1 = new_hist5d.centers['theta1'][t1_ind]
            t1_bin = (new_hist5d.edges['theta1'][t1_ind],
                      new_hist5d.edges['theta1'][t1_ind + 1])

            t2_ind = np.digitize([t2_t1_val],
                     new_hist5d.edges['theta2_over_theta1'])[0]-1
            t2_t1 = new_hist5d.centers['theta2_over_theta1'][t2_ind]

            for phi2_val in phi2_vals:
                phi2_ind = np.digitize([phi2_val],
                             new_hist5d.edges['phi2'])[0]-1
                phi2 = new_hist5d.centers['phi2'][phi2_ind]

                t2_bin = (new_hist5d.edges['theta2_over_theta1'][t2_ind],
                          new_hist5d.edges['theta2_over_theta1'][t2_ind+1])
                phi2_bin = (new_hist5d.edges['phi2'][phi2_ind],
                          new_hist5d.edges['phi2'][phi2_ind+1])

                bullseye_4part = HistogramData2D(
                    hist_data = new_hist5d.\
                        get_sub_histogram('theta1', t1).\
                        get_sub_histogram(t2_t1_key, t2_t1).\
                        get_sub_histogram('phi2', phi2)
                )

                bin3_edges = bullseye_4part.edges[t3_t2_key]
                t3_edges   = t1*t2_t1*bin3_edges
                bin3_centers = bullseye_4part.centers[t3_t2_key]
                t3_centers   = t1*t2_t1*bin3_centers
                bullseye_4part.hist = np.array([
                    bullseye_4part.hist[i3] / t3
                    for i3, t3 in enumerate(t3_centers)
                ])
                bullseye_4part.edges[t3_t2_key] = t3_edges
                bullseye_4part.centers[t3_t2_key] = t3_centers

                # Making plot
                bullseye_4part.make_plot('bullseye',
                         log_norm=True, save=None,
                         radial_coordinate='theta3_over_theta2',
                         **new_4part_params[process])
                ax = bullseye_4part.bullseye.axes[0]
                ax.set_ylim(0, t1*t2_t1)
                # bullseye_4part.bullseye.savefig(
                #     str(enc_figure_dir)+'/4particle_test.pdf')
                bullseye_inset(bullseye_4part,
                               inner_rad=t1*t2_t1,
                               inset_pts=[(t1, -phi2)])

                # Drawing curves and arrows
                theta2 = t1*t2_t1
                rad_end = 1.45
                phi_end = 0.8
                # Angular arrow
                phis = np.arange(0, phi_end, .01)
                rads = np.ones_like(phis)*theta2*1.1
                ax.plot(phis, rads, lw=2, color='dimgrey', clip_on=False)
                ax.arrow(phis[-1], theta2*1.1, 1e-10*theta2, 0,
                         clip_on=False, width=0.0,
                         facecolor='dimgrey', edgecolor='dimgrey',
                         head_width=0.05*theta2,
                         head_length=0.25*theta2)
                # Radial arrow
                ax.plot([-0.85, -0.85], [theta2*1.05, theta2*rad_end],
                        lw=2, color='dimgrey', clip_on=False)
                ax.arrow(-0.85, theta2*rad_end, 0, 1e-10*theta2,
                         clip_on=False, width=0.0,
                         facecolor='dimgrey', edgecolor='dimgrey',
                         head_width=0.21*theta2,
                         head_length=0.07*theta2)

                # Stamping
                stamp_bullseye(bullseye_4part.bullseye.axes[0],
                               {'theta_1': t1_bin,
                                'theta_2': t2_bin,
                                'phi_2':   phi2_bin},
                               Npoint=4, **new_hist5d.metadata)

                bullseye_4part.bullseye.fig.tight_layout()
                bullseye_4part.bullseye.savefig(
                    '/supplemental/4particle/'
                    f'{process}{t1}_{t2_t1}_{phi2}_4particle_bullseye.pdf',
                    enc_figure_dir)
