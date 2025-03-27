"""Solves MaxCut instances with Gurobi, recording the objectives and solution times.

USAGE:
    python -m classic_solve_MWC_QUBO_only | tee ./run_logs/classic_solutions/MWC_QUBO.log

Relies on :py:func:`qubo_tools.solve_QUBO_soft_timeout` to actually solve the
problems, as they are specified in QUBO formulations.

Filename refers to the fact that for MaxCut we solve QUBO formulations only, as
it turned out to be faster than alternative MIP formulations. (See
:py:mod:`classic_solve_MWCs` for an alternative implementation, which allows to
compare QUBO and LBOP results.)
"""
from MWC_inst import create_MWC_LBOP, extract_G_from_json
from gurobipy import GRB
from time import time
from glob import glob
import pandas as pd
import json
from random import shuffle
from qubo_utils import solve_QUBO_soft_timeout, load_QUBO

def solve_MWC_classically(qubo_file, quiet=True) -> None:
    """Wraps a call to QUBO-specific function, measures time and records the data."""

    Q, P, C, qubojs = load_QUBO(qubo_file)

    instance_id = qubojs["description"]["instance_id"]

    with open(qubojs["description"]["original_instance_file"], 'r') as openfile:
        js = json.load(openfile)

    # First: solve the "original" formulation (with timeout)
    G = extract_G_from_json(js)

    if not quiet:
        print("QUBO >>> ", flush=True, end="\n")
    start = time()
    qubo_model, qubo_x = solve_QUBO_soft_timeout(Q, P, C,
                                                 soft_timeout = 5*60,
                                                 overtime = 15*60,
                                                 gap = 0.05,
                                                 quiet=quiet)
    t_QUBO = time() - start

    if not quiet:
        print(f" QUBO status {qubo_model.status}.", flush=True)

    return pd.DataFrame([{
        "instance_id": instance_id,
        "sol_time_QUBO": t_QUBO,
        "status_QUBO": qubo_model.status,
        "solution_QUBO": "".join([str(int(qubo_x[i].X))
                                  for i in range(len(Q))]),
        "objective_QUBO": qubo_model.objVal,
        "gap_QUBO": qubo_model.MIPGap}])


def main():
    """Main script code (specifies timeout and filenames).

    Solves all instances given by ``./instances/QUBO/MWC*.json`` and saves the
    results into ``./run_logs/classic_solutions/MWC_QUBO.csv``.
    """
    df = pd.DataFrame(columns=[
        "instance_id",
        "sol_time_QUBO",
        "status_QUBO",
        "solution_QUBO",
        "objective_QUBO",
        "gap_QUBO"])

    filelist = [filename for filename in glob("./instances/QUBO/MWC*.json")]
    shuffle(filelist)
    for filename in filelist:
        print(f"Solving {filename}...", flush=True, end="")
        df = pd.concat([df, solve_MWC_classically(filename, quiet=False)])
        df.to_csv("./run_logs/classic_solutions/MWC_QUBO.csv", index=False)
        print("✅", flush=True)


if __name__ == '__main__':
    main()
