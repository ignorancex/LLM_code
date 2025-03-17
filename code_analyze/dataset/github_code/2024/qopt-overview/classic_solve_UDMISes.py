"""Solves UDMIS instances with Gurobi, recording the objectives and solution times.

USAGE:
    python -m classic_solve_UDMISes | tee ./run_logs/classic_solutions/UDMIS.log


Relies on :py:func:`MIS_inst.create_orig_MIS_IP` to create the MIP model. The
solution itself is implemented in :py:func:`solve_UDMIS_classically`.
"""
from gurobipy import GRB
from time import time
from glob import glob
import pandas as pd
import json
from MIS_inst import load_orig_MIS, create_orig_MIS_IP


def solve_UDMIS_classically(instance_file, quiet=True, timeout=None) -> None:
    """Solves the MIS instance using Gurobi.

    Uses :py:func:`MIS_inst.create_orig_MIS_IP` to create the model.
    """

    G, origjs = load_orig_MIS(instance_file)

    start = time()
    model, x = create_orig_MIS_IP(G)

    if quiet:
        model.setParam("OutputFlag", 0)

    if timeout is not None:
        model.setParam("TimeLimit", timeout)

    model.update()
    model.optimize()
    end = time()

    if timeout is None:
        assert model.status == GRB.OPTIMAL
    else:
        assert (model.status == GRB.OPTIMAL) or (model.status
                                                 == GRB.TIME_LIMIT)
    obj = model.objVal

    return pd.DataFrame([{
        "instance_id": origjs["description"]["instance_id"],
        "sol_time": end - start,
        "status": model.status,
        "solution": "".join([str(int(x[list(G.nodes)[i]].X)) for i in range(len(G.nodes))]),
        "objective": obj,
        "gap": model.MIPGap
    }])


def main():
    """Main script code (specifies timeout and filenames).

    Solves all instances given by ``./instances/orig/UDMIS*.json`` and saves the
    results into ``./run_logs/classic_solutions/UDMIS.csv``.
    """
    df = pd.DataFrame(columns=[
        "instance_id",
        "sol_time",
        "status",
        "solution",
        "objective", "gap"])

    for filename in glob("./instances/orig/UDMIS*.json"):
        print(f"Solving {filename}...", flush=True, end="")
        df = pd.concat([df, solve_UDMIS_classically(filename, timeout = 30*60)])
        df.to_csv("./run_logs/classic_solutions/UDMIS.csv", index=False)
        print(f"({df['sol_time'].iloc[-1]:.1f} sec.) ✅", flush=True)


if __name__ == '__main__':
    main()
