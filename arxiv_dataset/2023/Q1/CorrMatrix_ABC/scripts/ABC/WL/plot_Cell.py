import sys
import matplotlib.pylab as plt

import CorrMatrix_ABC.nicaea_ABC as nicaea


def plot_C_ell(ell, Cell, output_name=None):

    plt.plot(ell, Cell, '-')

    if output_name:
        plt.savefig(output_name)    

    pass

def main(argv=None):

    lmin = 10
    lmax = 5000
    nell = 50

    err, C_ell_name = nicaea.run_nicaea(lmin, lmax, nell, verbose=True)

    print(C_ell_name)

    ell, C_ell = nicaea.read_Cl('.', C_ell_name)

    plot_C_ell(ell, C_ell, output_name='{}.pdf'.format(C_ell_name))


if __name__ == "__main__":
    sys.exit(main(sys.argv))

