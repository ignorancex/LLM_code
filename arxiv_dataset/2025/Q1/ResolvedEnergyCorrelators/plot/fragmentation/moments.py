import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotter import Plotter
from plotter import ewoc_figure_dir

from utils.plot_utils import adjust_lightness

from qcd.fragmentation import QCDJetFragmenter
from qcd.fragmentation import p2p_fragmentation_moment


def main():
    Q = 1000
    R_jet = 1.0

    # Plot the evolved quark and gluon distributions
    plot_kwargs = {
        'xlabel': r'Mellin Moment $N$',
        'ylabel': r'$\hat{F}_{j\leftarrow i}(N)$',
        'axes.labelsize': 15
    }

    # colors = {0.04: ('royalblue', 'darkturquoise'),
    #           0.1: ('mediumorchid', 'plum'),
    #           0.2: ('deeppink', 'lightpink'),
    #           0.3: ('firebrick', 'indianred')}
    colors = {'00': 'sandybrown',
              '01': 'mediumorchid',
              '10': 'mediumseagreen',
              '11': 'cornflowerblue'}

    # DEBUG
    # r_subs = np.logspace(1e-5, 1, 20)
    # mellin_vals = range(2, 20, 1)
    r_subs = [0.1]
    mellin_vals = np.arange(2, 20, 1)

    moment_list = {inds: {N: [] for N in mellin_vals}
                   for inds in ['00', '01', '10', '11']}

    for r_sub in r_subs:
        plotter = Plotter(**plot_kwargs)
        ax = plotter.axes[0]

        # LL mellin_vals
        jet_calc = p2p_fragmentation_moment(mellin_vals,
                                            Q*r_sub, Q*R_jet)

        # Plot the evolved distributions
        for initial_flavor in [0, 1]:
            # Create the DGLAP evolver
            evolver = QCDJetFragmenter(initial_flavor=initial_flavor,
                                       Q=Q, R_jet=R_jet)

            F_evolved = evolver.evolve(r_sub, fixed_coupling=True)

            # Check that sum rules are satisfied
            sum_rule = np.sum(evolver.mellin_moment(2))
            print(f'{sum_rule=}')
            # DEBUG

            for final_flavor in [0, 1]:
                inds = f'{final_flavor}{initial_flavor}'
                for N in mellin_vals:
                    moment_list[inds][N].append(
                            evolver.mellin_moment(N)[final_flavor]\
                            / sum_rule)


        for initial_flavor in [0, 1]:
            for final_flavor in [0, 1]:
                # Plot style
                linestyle = (0,
                             (3.0+
                              1.1*(1-initial_flavor)+
                              1.3*(1-final_flavor),
                              0.5+.2*initial_flavor+.8*final_flavor))
                inds = f'{final_flavor}{initial_flavor}'
                color = colors[inds]

                # Numerically obtained mellin_vals
                moment_vals = [moment_list[inds][N][-1]
                               for N in mellin_vals]
                ax.plot(mellin_vals, moment_vals, color=color,
                        label=inds, linestyle=linestyle,
                        zorder=2)

                # LL mellin_vals
                ax.plot(mellin_vals,
                        jet_calc[:,final_flavor,initial_flavor],
                        label=None, color=adjust_lightness(color),
                        linestyle='solid', zorder=1)

    ax.legend(loc=(0.05, 0.05))
    ax.set_xlim(0, 2e1)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 2e0)

    plotter.savefig(ewoc_figure_dir/'tests/moment_test.pdf')

if __name__ == '__main__':
    main()
