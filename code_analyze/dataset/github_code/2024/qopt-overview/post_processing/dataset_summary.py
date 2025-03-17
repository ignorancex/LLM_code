#!/usr/bin/env ipython

"""Summarizes the data in instances/ and run_logs/ folders.

USAGE:
    $ python -m post_processing.dataset_summary summarize_src_QUBOs | tee run_logs/instances.list
"""
import sys
import numpy as np
from glob import glob
from qubo_utils import load_QUBO

def summarize_src_QUBOs(qubodir="./instances/QUBO"):
    """Prints the QUBO statistics to stdout."""
    print("id, type, qubo_vars, Qn_nonzeros")

    for filename in glob(qubodir + "/*.qubo.json"):
        Q, P, C, js = load_QUBO(filename)
        Qn = (1/2)*Q + np.diag(P)
        desc = js['description']
        print(f"{desc['instance_id']},{desc['instance_type']},{len(Q)}, {np.count_nonzero(Qn)}", flush=True)

if __name__ == '__main__':
    args = sys.argv
    globals()[args[1]](*args[2:])
