"""Solves TSP in MTZ formulation with Gurobi, saving the objectives and solution times.

USAGE:
    python -m classic_solve_TSPs_MTZ | tee ./run_logs/classic_solutions/TSP_MTZ.log

Relies on :py:func:`TSP_inst.solve_TSP_MTZ` to actually solve each instance.
"""
from TSP_inst import solve_TSP_MTZ, unpack_tour_y, obj_val
from gurobipy import GRB
from time import time
from glob import glob
import pandas as pd
import json


def solve_TSP_classically(instance_file, quiet=True, timeout=None) -> None:
    """Wraps a call to TSP-specific routine, records time and other data."""

    with open(instance_file, 'r') as openfile:
        instance_id = json.load(openfile)["description"]["instance_id"]

    start = time()
    model, y, D = solve_TSP_MTZ(instance_file, quiet, timeout)
    end = time()

    if timeout is None:
        assert model.status == GRB.OPTIMAL
    else:
        assert (model.status == GRB.OPTIMAL) or (model.status
                                                 == GRB.TIME_LIMIT)
    y_tour = unpack_tour_y(y, len(D))

    obj = obj_val(D, y_tour)

    return pd.DataFrame([{
        "instance_id": instance_id,
        "sol_time": end - start,
        "status": model.status,
        "tour": "-".join([str(y) for y in y_tour]),
        "objective": obj,
        "gap": model.MIPGap
    }])


def main():
    """Main script code (specifies timeout and filenames).

    Solves all instances given by ``./instances/orig/TSP*.json`` and saves the
    results into ``./run_logs/classic_solutions/TSP_MTZ.csv``.
    """
    df = pd.DataFrame(columns=[
        "instance_id",
        "sol_time",
        "status",
        "tour",
        "objective",
        "gap"])

    for filename in glob("./instances/orig/TSP*.json"):
        print(f"Solving MTZ: {filename}...", flush=True, end="")
        df = pd.concat([df, solve_TSP_classically(filename, timeout = 30*60)])
        df.to_csv("./run_logs/classic_solutions/TSP_MTZ.csv", index=False)
        print("✅", flush=True)


if __name__ == '__main__':
    main()
