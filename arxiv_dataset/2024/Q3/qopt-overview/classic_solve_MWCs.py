"""Illustration: compares QUBO and LBOP formulations for MaxCut. (Not critical for results reproduction.)

Uses Gurobi to obtain objective values for MaxCuts, recording optimality gaps
and solution times.
"""
from MWC_inst import create_MWC_LBOP, extract_G_from_json
from gurobipy import GRB
from time import time
from glob import glob
import pandas as pd
import json
from random import shuffle
from qubo_utils import solve_QUBO, load_QUBO

def solve_MWC_classically(qubo_file, quiet=True, timeout=None) -> None:
    """Solves the MWC instance using Gurobi, in QUBO and LBOP formulations.

    Relies on :py:func:`MWC_inst.create_MWC_LBOP` and :py:func:`qubo_utils.solve_QUBO`.
    """

    Q, P, C, qubojs = load_QUBO(qubo_file)

    instance_id = qubojs["description"]["instance_id"]

    with open(qubojs["description"]["original_instance_file"], 'r') as openfile:
        js = json.load(openfile)

    # First: solve the "original" formulation (with timeout)
    G = extract_G_from_json(js)

    if not quiet:
        print("LBOP >>> ", flush=True, end="")
    start = time()
    model, _, x = create_MWC_LBOP(G, quiet=quiet, timeout=timeout)
    model.update()
    model.optimize()
    t_LBOP = time() - start

    if not quiet:
        print(f"LBOP status {model.status}; QUBO >>> ", end="", flush=True)
    start = time()
    qubo_model, qubo_x = solve_QUBO(Q, P, C,quiet=quiet, timeout=timeout)
    t_QUBO = time() - start

    if not quiet:
        print(f" QUBO status {model.status}.", flush=True)

    return pd.DataFrame([{
        "instance_id": instance_id,
        "sol_time_LBOP": t_LBOP,
        "sol_time_QUBO": t_QUBO,
        "status_LBOP": model.status,
        "status_QUBO": qubo_model.status,
        "solution_LBOP": "".join([str(int(x[i].X)) for i in range(len(x))]),
        "solution_QUBO": "".join([str(int(qubo_x[i].X))
                                  for i in range(len(Q))]),
        "objective_LBOP": model.objVal,
        "objective_QUBO": qubo_model.objVal,
        "gap_LBOP": model.MIPGap,
        "gap_QUBO": qubo_model.MIPGap}])


def main():
    """Main script code (specifies timeout and filenames).

    Solves all instances given by ``./instances/QUBO/MWC*.json`` and saves the
    results into ``./run_logs/classic_solutions/MWC.csv``.
    """
    df = pd.DataFrame(columns=[
        "instance_id",
        "sol_time_LBOP",
        "sol_time_QUBO",
        "status_LBOP",
        "status_QUBO",
        "solution_LBOP",
        "solution_QUBO",
        "objective_LBOP",
        "objective_QUBO",
        "gap_LBOP",
        "gap_QUBO"])

    filelist = [filename for filename in glob("./instances/QUBO/MWC*.json")]
    shuffle(filelist)
    for filename in filelist:
        print(f"Solving {filename}...", flush=True, end="")
        df = pd.concat([df, solve_MWC_classically(filename, quiet=False,
                                                  timeout=5 * 60)])
        df.to_csv("./run_logs/classic_solutions/MWC.csv", index=False)
        print("✅", flush=True)

if __name__ == '__main__':
    main()
