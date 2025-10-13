"""Solves MaxCut instances with a few alternative classical heuristics.

USAGE:
    python -m classic_bm_MWC | tee ./run_logs/classic_solutions/MWC_heuristics.log

Relevant files:

    - instance IDs to solve are taken from ``./run_logs/solved_MWC.list``.

    - D-Wave runtime (for the heuristic 1) is read from ``./run_logs/summaries/dwave_summary.csv``.

The main baseline heuristic is just Gurobi solver with the time limit set to the
   actual runtime of the D-Wave's experiment (for the respective instance).

See also: :py:mod:`classic_solve_MWC_QUBO_only` and :py:mod:`classic_solve_MWCs`.

"""
from time import time
import pandas as pd
from argparse import ArgumentParser

import mlrose_ky as mlrose
import numpy as np

from qubo_utils import load_QUBO, get_QUBO_by_ID, solve_QUBO


def gurobi_QUBO_timeout(inst_id, inst_dir, timeout, quiet=False):
    """Solves a QUBO using Gurobi with a timeout."""

    Q, P, C, _ = load_QUBO(get_QUBO_by_ID(inst_id, inst_dir + "/QUBO"))

    runtime = time()
    m, x = solve_QUBO(Q, P, C, quiet=quiet, timeout=timeout)
    runtime = time() - runtime

    return pd.DataFrame([{
        "instance_id": inst_id,
        "qubo_vars": len(Q),
        "timeout": timeout,
        "method": "GurobiTimeout",
        "runtime": runtime,
        "solution": "".join([str(int(x[i].X))
                             for i in range(len(Q))]),
        "objective": -m.objVal,
        "comment": f"Gurobi status: {m.status}, gap: {m.MIPGap}"}])

def mlrose_SA(inst_id, inst_dir, timeout, quiet=False):
    """Solves a QUBO using mlrose's Simulated Annealing algo."""
    Q, P, C, _ = load_QUBO(get_QUBO_by_ID(inst_id, inst_dir + "/QUBO"))

    def QUBO_fitness(x):
        return 0.5 * x @ Q @ x + P @ x + C

    problem = mlrose.DiscreteOpt(length=len(Q),
                                 fitness_fn=mlrose.CustomFitness(QUBO_fitness),
                                 maximize=False,
                                 max_val=2)
    runtime = time()

    best_x, best_obj, _ = mlrose.simulated_annealing(problem,
                                                     schedule=mlrose.ExpDecay(),
                                                     max_attempts=len(Q)**2)
    runtime = time() - runtime

    return pd.DataFrame([{
        "instance_id": inst_id,
        "qubo_vars": len(Q),
        "timeout": "n/a",
        "method": "SimulatedAnnealing",
        "runtime": runtime,
        "solution": "".join([str(int(xj))
                             for xj in best_x]),
        "objective": -best_obj,
        "comment": ""}])


def mlrose_GA(inst_id, inst_dir, timeout, quiet=False):
    """Solves a QUBO using mlrose's Genetic Algo."""
    Q, P, C, _ = load_QUBO(get_QUBO_by_ID(inst_id, inst_dir + "/QUBO"))

    def QUBO_fitness(x):
        return 0.5 * x @ Q @ x + P @ x + C

    problem = mlrose.DiscreteOpt(length=len(Q),
                                 fitness_fn=mlrose.CustomFitness(QUBO_fitness),
                                 maximize = False,
                                 max_val = 2)
    runtime = time()

    best_x, best_obj, _ = mlrose.genetic_alg(problem,
                                             pop_size=len(Q)**2)
    runtime = time() - runtime

    return pd.DataFrame([{
        "instance_id": inst_id,
        "qubo_vars": len(Q),
        "timeout": "n/a",
        "method": "GeneticAlg",
        "runtime": runtime,
        "solution": "".join([str(int(xj))
                             for xj in best_x]),
        "objective": -best_obj,
        "comment": ""}])

def main():
    """Main script code.

    Solves all instances given by ``./instances/QUBO/MWC*.json`` and saves the
    results into ``./run_logs/classic_solutions/MWC_QUBO.csv``.
    """
    parser = ArgumentParser()
    parser.add_argument('--instlist', type=str, default='./run_logs/solved_MWC.list',
                        help="list of instances to solve")
    parser.add_argument('--instdir', type=str, default='./instances',
                        help="dir with instances (both QUBO and orig)")
    parser.add_argument('--method', type=str, default='gurobi-timeout',
                        help="heuristic 1: Gurobi (QUBO) with a timeout")
    parser.add_argument('--output', type=str, default='./run_logs/classic_bm.csv',
                        help="resulting summary table")

    args = parser.parse_args()

    # output dataframe format
    out = pd.DataFrame(columns=[
        "instance_id",
        "qubo_vars",
        "timeout",
        "method",
        "runtime",
        "solution",
        "objective",
        "comment"])

    try:
        instances = pd.read_csv(args.instlist)
    except:
        print(f"Error reading the instances list: {args.instlist}")
        exit(1)

    # prepare the timeouts
    # Load the main summary file (from D-Wave logs)
    df = pd.read_csv("./run_logs/summaries/dwave_summary.csv")
    df_e = pd.read_csv("./run_logs/summaries/embeddings.csv")
    df = pd.merge(df.drop("embedding_time", axis=1),
                  df_e[["instance_id", "emb_time", "emb_success"]],
                  on="instance_id", how="left")

    timeouts = dict(zip(df['instance_id'], df['end_timestamp'] - df['start_timestamp'] + df['emb_time']))

    methods = {"gurobi-timeout": gurobi_QUBO_timeout,
               "SA": mlrose_SA,
               "GA": mlrose_GA}
    try:
        heuristic = methods[args.method]
    except:
        print(f"Wrong method specified: {args.method}")
        print(f"Currently implemented methods are:")
        print("• " + "\n- ".join([k for k in methods.keys()]))
        exit(1)

    for i, inst in instances.iterrows():
        print(f"[{i+1} / {len(instances)}; {inst['instance_id']} ]: using {args.method} (QPU time was {timeouts[inst['instance_id']]:.1f} sec)...",
              flush=True)

        out = pd.concat([out, heuristic(inst['instance_id'],
                                                 args.instdir,
                                                 timeouts[inst['instance_id']],
                                                 quiet = False)])
        out.to_csv(args.output, index=False)
        rt = list(out['runtime'])[-1]
        print(f"[{i+1} / {len(instances)}; {inst['instance_id']} ] ✓ done ({rt:.1f} sec vs {timeouts[inst['instance_id']]:.1f} of QPU).", flush=True)

if __name__ == '__main__':
    main()
