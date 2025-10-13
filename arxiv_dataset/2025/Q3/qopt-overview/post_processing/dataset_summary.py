#!/usr/bin/env ipython

"""Summarizes the data in instances/ and run_logs/ folders.

USAGE:
    $ python -m post_processing.dataset_summary summarize_src_QUBOs | tee run_logs/instances.list
"""
import sys
import json
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

def summarize_src_MWCs(MWCdir="./instances/orig"):
    """Simmarizes some original MWC instsances stats.

    Notes:

        Output values: ``id`` (instance ID), ``instname`` (full text instance
        name), ``model`` (graph generation model, ERG = Erdös-Rényi graphs)
        ``N`` (number of nodes), ``E`` (number of edges), ``p`` instance
        generation parameter (ERG model parameter).

    """
    print("id,instname,model,N,E,p")
    for filename in glob(MWCdir + "/MWC*.orig.json"):
        with open(filename, 'r') as ofile:
            js = json.load(ofile)

        id = js["description"]["instance_id"]
        N = len(js["nodes"])
        E = len(js["edges"])
        instname = js["description"]["original_instance_name"]
        params = instname.split("_")
        p = float(params[-1][1:])
        model = params[1]
        print(f"{id},{instname},{model},{N},{E},{p}")


if __name__ == '__main__':
    args = sys.argv
    globals()[args[1]](*args[2:])
