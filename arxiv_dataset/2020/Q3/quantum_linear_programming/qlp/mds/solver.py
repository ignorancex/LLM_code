"""QUBO solver routines to obtain global optimal solution.
"""
from typing import List, Tuple, Union, Optional, Set, Dict, Any
from itertools import product

import numpy as np

from tqdm import tqdm

from qlp.mds.qubo import get_mds_qubo


def classical_search(
    qubo: "SparseMatrix", n_nodes: Optional[int] = None
) -> Union[int, List[Tuple[int]]]:
    """Classicaly searches minimal energy for all integer solutions.

    Assumes that the first n_nodes entries correspond to sorted vertex numbers.

    Arguments:
        qubo: The QUBO to optimize.
        n_nodes: If present, optimizes the spaces for the full qubo but only returns
            solutions accounting for entries from zero to n_nodes (exclusive).
            This means that the energy and slack contributions are excluced from the
            presentation of the solution.

    Returns:
        e_min: The minimal energy.
        solutions: A list of vertices (qubo entries) which are non-zero.
    """
    energies = []
    vecs = []

    n_iterations = 2 ** qubo.shape[1]
    if n_iterations > 10 ** 5:
        go_on = input(
            f"This search will execute {n_iterations} iterations. Continue (y/n)?"
        )
        if go_on.lower() != "y":
            raise KeyboardInterrupt("Too many iterations abort.")

    # Brute force iterate space
    for vv in tqdm(list(product(*[(0, 1)] * qubo.shape[1]))):
        ee = (vv @ qubo @ vv).flatten()[0]
        energies.append(ee)
        vecs.append(vv)

    # Find minimal energy
    e_min = np.min(energies)
    # And corresponding vectors
    solutions = [tuple(vv) for vv, ee in zip(vecs, energies) if ee == e_min]

    if n_nodes is not None:
        # Reduce solutions to n_nodes entries
        solutions = [sol[:n_nodes] for sol in solutions]

    # Convert bit list to vertex numbers
    solutions = [
        tuple([n for n, vv in enumerate(sol[:n_nodes]) if vv == 1]) for sol in solutions
    ]

    return e_min, solutions


def mds_schedule_submit(
    graph: Set[Tuple[int, int]],
    p_schedules: List[Dict[str, Any]],
    embbedding: "Embedding",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Submits MDS with different schedules for the penalty term.

    The final state of each schedule is used as the initial_state for the next schedule.
    The first schedule does not use initial_state.
    Each schedule uses exactly one read (`num_reads=1`)

    Arguments:
        graph: The graph for the MDS defined by a set of edges.
        p_schedules: The penalty term schedules. A list of dicts with two keys:
            * the penalty (int)
            * and the schedule (list of tuples where the first entry is the start time
                and the second is the value for annealing parameters s).
        embbedding: The embedding used to `sample_qubo`.
        kwargs: Override `sample_qubo` parameters (besides `num_reads`).

    Returns:
        A list of dictinoaries containing input and output of each schedule.

    Example:
        Suppose you want to submit two schedules for p: a small penalty to start and a
        large penalty to fix.

        (1)  |   .                      (2)   | .       .
           s | .      where p = 1 and       s |   . . .     where p = 9
             |_____                           |___________

        This corresponds to

        ```
        p_schedules = [
            {"penalty": 1, "schedule": [(0, 0.0), (200, 1.0)]},
            {"penalty": 9, "schedule": [(0, 1.0), (200, 0.5), (400, 0.5), (600, 1.0)]},
        ]
        ```

    """
    assert kwargs.get("num_reads", 1) == 1

    nodes = set([v for v, _ in graph])
    nodes = nodes.union(set([v for _, v in graph]))
    n_nodes = len(nodes)

    base_params = {
        "answer_mode": "raw",
        "auto_scale": True,
        "num_reads": 1,  # raw will dump out all results
        "reinitialize_state": False,  # I guess this is not important if num_reads=1
    }
    base_params.update(kwargs)

    results = []
    initial_state = None
    for inputs in p_schedules:
        dwave_config = base_params.copy()
        dwave_config["initial_state"] = initial_state
        dwave_config["anneal_schedule"] = inputs["schedule"]

        if initial_state is None:
            dwave_config.pop("initial_state")
            dwave_config.pop("reinitialize_state")

        qubo = get_mds_qubo(graph, triangularize=True, penalty=inputs["penalty"])
        quobo_dict = {key: val for key, val in zip(qubo.keys(), qubo.values())}
        result = embbedding.sample_qubo(quobo_dict, **dwave_config)
        sample = result.first

        data = dwave_config.copy()
        data["penalty"] = inputs["penalty"]
        data["initial_state"] = initial_state
        data["final_state"] = sample.sample.copy()
        data["energy_unshifted"] = sample.energy
        data["energy"] = sample.energy + inputs["penalty"] * n_nodes
        data["chain_break_fraction"] = sample.chain_break_fraction

        results.append(data)

        initial_state = sample.sample.copy()

    return results
