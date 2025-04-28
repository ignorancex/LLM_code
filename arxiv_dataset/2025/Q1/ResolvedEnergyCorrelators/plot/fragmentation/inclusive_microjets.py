import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotter import Plotter
from plotter import ewoc_figure_dir

from utils.plot_utils import adjust_lightness

from qcd.qcd_basics import alpha_s
from qcd.fragmentation import QCDJetFragmenter
from qcd.splitting_functions import P_qq, P_qg, P_gq, P_gg




def main():
    Q = 1000
    R_jet = 1.0

    # Plot the evolved quark and gluon distributions
    plot_kwargs = {
        'xlabel': r'Momentum Fraction $x$',
        'ylabel': r'$\sum_k F_{k\leftarrow i}(x)$',
        'axes.labelsize': 15
    }
    plotter = Plotter(**plot_kwargs)
    ax = plotter.axes[0]

    colors = {0.04: ('royalblue', 'darkturquoise'),
              0.1: ('mediumorchid', 'plum'),
              0.2: ('deeppink', 'lightpink'),
              0.3: ('firebrick', 'indianred')}

    # Plot the evolved distributions
    for flavor in [0, 1]:
        # Create the DGLAP evolver
        quark_evolver = QCDJetFragmenter(initial_flavor=flavor,
                                         Q=Q, R_jet=R_jet)

        for t in [0.04, 0.1, 0.2, 0.3]:
            r_sub = R_jet * np.exp(-(np.pi*t) / alpha_s(Q*R_jet))

            F_evolved = quark_evolver.evolve(r_sub,
                                             fixed_coupling=True)
            xs = quark_evolver.x_centers

            # ===================================================
            # DEBUG: Checking that sum rules are satisfied
            sum_rule = np.sum(quark_evolver.mellin_moment(2))
            print(f'{sum_rule=}')
            # ===================================================

            # Plot the evolved distributions
            ys = np.zeros_like(xs)
            for final_flavor in [0, 1]:
                # Plot the evolved gluon distribution
                ys += F_evolved[final_flavor]/sum_rule

            linestyle = 'solid' if flavor == 0 else 'dashed'
            label = None if flavor == 1 else rf'$t={t}$'
            color = colors[t][flavor]
            if flavor == 1:
                color = adjust_lightness(color, 1.1)

            ax.plot(xs, ys, color=color, label=label,
                    linestyle=linestyle)


    q_line = Line2D([0], [0],
                    label=r"$F_{\text{jet} \leftarrow q}(x)$",
                    color='k')
    g_line = Line2D([0], [0],
                    label=r"$F_{\text{jet} \leftarrow g}(x)$",
                    color=adjust_lightness('darkgrey', 0.9),
                    linestyle='dashed')


    flavor_legend = ax.legend(loc=(0.7, 0.05), fontsize=14,
                              handles=[q_line, g_line])
    ax.legend(loc=(0.05, 0.05))
    ax.set_xlim(0, 1e0)
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e1)

    ax.annotate(
        r'Recreation of \texttt{Inclusive Microjet Spectrum}',
        xy=(0.27, 0.17), xytext=(0.27, 0.17),
        xycoords='figure fraction',
        verticalalignment='top',
        url='https://arxiv.org/pdf/1411.5182#page=11')

    ax.add_artist(flavor_legend)

    plotter.savefig(ewoc_figure_dir/'tests/microjet_test.pdf')

if __name__ == '__main__':
    main()
