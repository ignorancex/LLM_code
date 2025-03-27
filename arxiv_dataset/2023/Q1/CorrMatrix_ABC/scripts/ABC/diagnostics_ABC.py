#!/usr/bin/env python

# diagnostics_ABC.py
# Martin Kilbinger (2017)

# Compability with python2.x for x>6
from __future__ import print_function

import numpy as np
import glob
import sys
import re


def diagnostics_draws():
    """Return diagnostics related to the number of draws.
    """

    Mtot = 0
    nd_tot = 0
    nit_tot = 0

    nruns = 0

    #dist_dirs = glob.glob('*_dist_*')
    dist_dirs = ["."]

    for dist_dir in dist_dirs:

        nsim_dirs = glob.glob(f'{dist_dir}/nsim_*')

        for nsim_dir in nsim_dirs:

            nr_dirs = glob.glob(f'{nsim_dir}/nr_*')

            for nr_dir in nr_dirs:

                it_max = -1

                files = glob.glob(f'{nr_dir}/*')

                for f in files:

                    m = re.findall('linear_PS(\d+).dat', f)
                    #m = re.findall('quad_PS(\d+).dat', f)
                    if len(m) == 0:
                        continue

                    if int(m[0]) > it_max:
                        it_max = int(m[0])

                    dat = np.genfromtxt(f, names=True, deletechars=['[]'])
                    nd_tot += int(dat['NDraws'].sum())
                    Mtot += len(dat)
                    nit_tot += 1

                if it_max != -1:
                    nruns += 1

                print(nr_dir, it_max)

    return Mtot, nd_tot, nit_tot, nruns



# Main program
def main(argv=None):
    """Main program.
    """

    Mtot, nd_tot, nit_tot, nruns = diagnostics_draws()

    nit_mean = nit_tot / nruns
    ndr_mean = nd_tot / nruns

    print(f'Number of runs = {nruns}')
    print(f'Mean number of draws per run = {ndr_mean}')
    print(f'Mean number of iterations per run = {nit_mean}')


    print('# Mtot nd_tot ratio nit_tot')
    if nd_tot > 0:
        print('{} {} {:.4f} {}'.format(Mtot, nd_tot, float(Mtot)/float(nd_tot), nit_tot))
    else:
        print('- - - -')


if __name__ == "__main__":
    sys.exit(main(sys.argv))
