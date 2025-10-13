"""An example top-level script used to run QPU experiments for IBM and D-Wave.

Invokes :py:mod:`run_ibm` (or :py:mod:`run_ibm_reserved`) and :py:mod:`run_dwave`.
"""

import argparse
import os, sys
import glob, json
from bench_qubo import DataBaseQUBO, QUBOSolverIBM


def run_example(python_file, *args):
    """A wrapper function that actually executes the device-specific script."""
    assert os.system(f'{sys.executable} {python_file} {" ".join([str(arg) for arg in args])}') == 0


def main():
    """Main script code."""
    # ['MWC4', 'MWC33', 'TSP27', 'TSP189']
    # ['UDMWIS25', 'UDMWIS57', 'UDMWIS165'] + ['MWC21', 'MWC23', 'MWC25'] + ['TSP1', 'TSP4', 'TSP7']
    #python_file = 'run_ibm.py'
    #python_file = 'run_ibm_reserved.py'
    #python_file = 'run_ibm_sim.py'

    python_file = 'run_dwave.py'

    name = 'dwave-1000shots'

    #id_segments = [['UDMWIS25', 'UDMWIS57', 'UDMWIS165']]
    id_segments = [[]]
    for path in glob.glob(r'..\instances\QUBO\UDMIS*EXT*.json'):
        with open(path, 'r') as fh:
            id_segments[0].append(json.load(fh)['description']['instance_id'])

    for ids in id_segments:
        run_example(python_file, f'--name {name}', f'--ids {" ".join(ids)}')


if __name__ == '__main__':
    main()
