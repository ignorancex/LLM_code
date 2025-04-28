import numpy as np
import matplotlib.pyplot as plt

import matplotlib.path as mpath
import matplotlib.patches as mpatches

from re import search

from utils.postprocess import read_xy_columns

from encs.plots import bl_prop
from utils.plot_utils import adjust_lightness
from plotter import enc_data_dir, enc_figure_dir, combine_plotters
from encs.plots import plot_1d_enc

# =====================================
# Plot parameters
# =====================================
# 1D plots
plot_1d_params = {
    'axes.labelsize': 20,
    # 'ylim': (1e-5, 8e0),
    'ylim': (0, 3.7e0),
    'xlim': (5e-3, 1e0),
    'xlabel': r'$R_1$',
    'ylabel': r'PENC($R_1$)',
    # 'y_scale': 'log',
    'y_scale': 'lin',
    'x_scale': 'log',
    # 'x_scale': 'lin',
}


colors = {
    '1-00' : 'mediumseagreen',
    '4-00' : 'cornflowerblue',
    '9-00' : 'mediumorchid',
    '49-00': 'indianred',
    '99-00': 'goldenrod'
}
runtimes = {
    '1-00' : '18 s',  # '17.8 s',
    '4-00' : '18 s',  # '17.7 s',
    '9-00' : '18 s',  # '17.6 s',
    '49-00': '18 s',  # '17.8 s',
    '99-00': '18 s',  # '17.8 s'
}


def stamp_1d(ax, **metadata):
    # Getting jet radius (more complicated than probably needs to be)
    jet_rad = 10*float(metadata['jet_rad'])
    rnd_digit = len(search("\.(0*)", str(jet_rad)).group(1))
    jet_rad = round(jet_rad, rnd_digit)
    jet_info = metadata['jet_alg'].upper()
    if jet_info in ['ANTI-KT', 'ANTIKT', 'AKT']:
        jet_info = 'AK'

    jet_info += str(jet_rad)[:-2] if str(jet_rad)[-2:] == '.0' \
                   else str(jet_rad)
    jet_info += r' Jets, $\!|\eta^\text{jet}| <$ 1.9'
    jet_info += r', $p_T^\text{jet} \in\,$'
    jet_info += '[500, 550] GeV'

    # Jet information
    ax.text(0.018, 0.87, jet_info, fontsize=12,
            transform=ax.transAxes)

    # Dataset information
    if metadata['level'] == 'data':
        ax.text(0.018, 0.93,
                r'$\textbf{CMS Open Data}$: '\
                r'2011A Jet Primary Dataset, $10^5$ events',
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
        process_str += r'$\sqrt{s}=\,$'+str(int(metadata['energy']))+' TeV'
        ax.text(0.018, 0.93, process_str,
                fontsize=14, transform=ax.transAxes)


# =====================================
# Traditional PENCs
# =====================================
trad_penc = {
    2: read_xy_columns('encs/data/old/PE2C.dat'),
    5: read_xy_columns('encs/data/old/PE5C.dat'),
}

trad_time = {
    2: r'2.5 s',
    5: r'1$\frac{3}{4}$ h',
}

trad_cols = {
    2: adjust_lightness(colors['1-00'], 1.6),
    5: adjust_lightness(colors['4-00'], 1.25),
}



# =====================================
# Main
# =====================================
if __name__ == "__main__":
    plotters = []
    metadata = []

    trad_lines = []
    trad_labels = []

    for nu, color in colors.items():
        N = int(eval(nu.replace('-', '.')) + 1)
        hist = plot_1d_enc(
                file_name=enc_data_dir/
                        f'2particle_od_100k_200bins_nu{nu}.py',
                variable_order=['theta1'],
                color=color,
                label=rf'$N$={N} ({runtimes[nu]})',
                save=None, **plot_1d_params,
        )

        if N in [2, 5, 6]:
            line, = hist.plot.axes[0].plot(
                           *trad_penc[N],
                           ls='dashed',
                           color=trad_cols[N],
                           label=None)
            trad_lines.append(line)
            trad_labels.append(f'Old ('+trad_time[N]+')')

        plotters.append(hist.plot)
        metadata.append(hist.metadata)

    # Combining plots
    cplotter = combine_plotters(plotters)
    ax = cplotter.axes[0]
    stamp_1d(ax, **metadata[0])

    # Legends
    legend2 = ax.legend(trad_lines, trad_labels,
                        loc=(0.36, 0.655),
                        handletextpad=.3,)
    legend = ax.legend(loc=(00.00, 0.43), handletextpad=0.5)
    ax.add_artist(legend2)

    # Adding archaic text for traditional PENC
    # rcParams = plt.rcParams.copy()
    # plt.rcParams.update(plt.rcParamsDefault)
    # ax.text(0.53, 0.76, 'Old PE2C',
    #         font_properties=bl_prop,
    #         fontsize=18, transform=ax.transAxes)
    # ax.text(0.53, 0.678, 'Old PE5C',
    #         font_properties=bl_prop,
    #         fontsize=18, transform=ax.transAxes)
    # Resetting font
    # plt.rcParams.update(rcParams)

    # Indicating N > 10 computationally inaccessible with traditional
    # top = 0.64
    # bottom = 0.47
    # tip_pos = 0.45
    # spine_pos = 0.47
    # bracket = mpatches.PathPatch(
    #     mpath.Path(
    #         [
    #             [tip_pos, top],
    #             [spine_pos, top],
    #             [spine_pos, bottom],
    #             [tip_pos, bottom],
    #         ]
    #     ),
    #     transform=ax.transAxes,
    #     facecolor="none",
    #     edgecolor="darkgrey",
    #     linewidth=2,
    # )
    # ax.add_artist(bracket)
    # Text: "Inaccessible to traditional PENC" for N >= 10
    # text_x = spine_pos + 0.02
    # text_y = top*0.79 + bottom*0.21
    # ax.text(text_x, text_y,
    #         "Inaccessible to",
    #         transform=ax.transAxes,)
    # ax.text(text_x+0.025, text_y-0.055,
    #         "traditional",
    #         transform=ax.transAxes,)
    # Archaic PENC
    # plt.rcParams.update(plt.rcParamsDefault)
    # ax.text(text_x+0.043, text_y-0.13,
    #         "PENC",
    #         font_properties=bl_prop, fontsize=18,
    #         transform=ax.transAxes,)
    # plt.rcParams.update(rcParams)

    # Formatting and saving
    cplotter.fig.tight_layout()
    cplotter.savefig(
        f'od_projected_runtime.pdf',
        enc_figure_dir/'supplemental/2particle/')
