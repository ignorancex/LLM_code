import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotter import Plotter
from plotter import ewoc_figure_dir

from utils.plot_utils import adjust_lightness

from qcd.fragmentation import QCDJetFragmenter
from qcd.splitting_functions import P_qq, P_qg, P_gq, P_gg


LOGARITHMIC = True

def main():
    # Create the DGLAP evolver
    quark_evolver = QCDJetFragmenter(initial_flavor=0,
                                     Q=1000, R_jet=.5,
                                     epsilon=1e-10)

    # Plot the evolved quark and gluon distributions
    plot_kwargs = {
        'xlabel': r'Momentum Fraction $x$',
        'ylabel': r'$x\,F_{j\leftarrow q}(x)$',
        'axes.labelsize': 15
    }
    plotter = Plotter(**plot_kwargs)
    ax = plotter.axes[0]

    colors = {0.01: ('royalblue', 'darkturquoise'),
              0.03: ('rebeccapurple', 'plum'),
              0.1: ('deeppink', 'lightpink'),
              0.3: ('firebrick', 'indianred')}

    for r_sub in [0.01, 0.03, 0.1, 0.3]:
        F_evolved = quark_evolver.evolve(r_sub, n_steps=100)
        xs = quark_evolver.x_centers

        # Plot the evolved distributions
        for flavor in [0, 1]:
            label = rf'$r_\text{{sub}}={r_sub}$' if flavor == 0 \
                        else None
            linestyle = 'solid' if flavor == 0 else 'dashed'

            # Plot the evolved gluon distribution
            if LOGARITHMIC:
                ys = xs*np.log(10)*F_evolved[flavor]*(1-xs)
            else:
                ys = F_evolved[flavor]

            ax.plot(xs, ys, color=colors[r_sub][flavor],
                    linestyle=linestyle, label=label)

    q_line = Line2D([0], [0],
                    label=r"$F_{q\leftarrow q}(x\,|\,\,r_\text{sub}\!\leftarrow\! R_\text{jet})$",
                    color='k')
    g_line = Line2D([0], [0],
                    label=r"$F_{g\leftarrow q}(x\,|\,\,r_\text{sub}\!\leftarrow\! R_\text{jet})$",
                    color=adjust_lightness('darkgrey', 0.9),
                    linestyle='dashed')

    if LOGARITHMIC:
        flavor_legend = ax.legend(loc=(0.5, 0.8), fontsize=14,
                                  handles=[q_line, g_line])
        ax.legend(loc=(0.05, 0.67))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-5, 1e0)
        ax.set_ylim(5e-5, 2e1)
    else:
        flavor_legend = ax.legend(loc=(0.01, 0.835), fontsize=14,
                                  handles=[q_line, g_line])
        ax.legend(loc=(0.1, 0.53))
        ax.set_xlim(0, 1e0)
        ax.set_ylim(0, 2e1)


    ax.add_artist(flavor_legend)
    ax.text(0.73, 0.74, r"$R_\text{jet} = 0.5$",
            fontsize=14, ha='center',
            transform=ax.transAxes)
    plotter.savefig(ewoc_figure_dir/'tests/fragmentation_test.pdf')

if __name__ == '__main__':
    main()
