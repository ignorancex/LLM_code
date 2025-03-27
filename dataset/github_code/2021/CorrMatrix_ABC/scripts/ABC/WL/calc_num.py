#!/usr/bin/env python3

import sys

import numpy as np

from statsmodels.stats.weightstats import DescrStatsW

from CorrMatrix_ABC.covest import weighted_std

def main(argv=None):

    base_name = argv[1]

    op1 = open(f'{base_name}.dat', 'r')
    lin1 = op1.readlines()
    op1.close()

    data1 = [elem.split() for elem in lin1]

    a_samples = np.array([float(line[0]) for line in data1[1:]])
    b_samples = np.array([float(line[1]) for line in data1[1:]])

    weights = np.loadtxt(f'{base_name}weights.dat')

    a_results = DescrStatsW(a_samples, weights=weights, ddof=0)
    b_results = DescrStatsW(b_samples, weights=weights, ddof=0)

    # For some reason a_results.std_mean is NaN, and a_results_std is
    # slightly different from the result of the following computation
    a_std = weighted_std(a_samples, weights)
    b_std = weighted_std(b_samples, weights)

    # store numerical results
    op2 = open('num_res.dat', 'w')
    op2.write('Omegam_mean    ' + str(a_results.mean) + '\n')
    op2.write('Omegam_std     ' + str(a_std) + '\n\n\n')
    op2.write('sigma8_mean    ' + str(b_results.mean) + '\n')
    op2.write('sigma8_std     ' + str(b_std))
    op2.close()

    print('Numerical results:')
    print('Omegam:    ' + str(a_results.mean) + ' +- ' + str(a_std))
    print('b:    ' + str(b_results.mean) + ' +- ' + str(b_std))
    print()


    

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
