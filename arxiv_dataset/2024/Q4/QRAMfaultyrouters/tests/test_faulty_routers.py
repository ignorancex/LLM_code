import os

import numpy as np
import pytest

from qram_repair import MonteCarloRouterInstances, QRAMRouter


@pytest.mark.parametrize("n", [5, 6, 7])
@pytest.mark.parametrize("eps", [0.02, 0.08])
@pytest.mark.parametrize("top_three_functioning", [True, False])
@pytest.mark.parametrize("num_cpus", [1, 6])
def test_monte_carlo(n, eps, top_three_functioning, num_cpus):
    num_instances = 1000
    rng_seed = 2545685
    filepath = "tmp.h5py"
    mc = MonteCarloRouterInstances(
        n, eps, num_instances, rng_seed, top_three_functioning, filepath
    )
    result = mc.run(num_cpus)
    os.remove(filepath)


@pytest.mark.skip(reason="with streams, different rands for cpus=1 or >1")
def test_parallel_mc():
    n = 7
    eps = 0.06
    num_instances = 20
    rng_seed = 2545685
    top_three_functioning = True
    filepath = "tmp.h5py"
    mc = MonteCarloRouterInstances(
        n, eps, num_instances, rng_seed, top_three_functioning, filepath
    )
    result_lin = mc.run(num_cpus=1)
    result_par = mc.run(num_cpus=4)
    for val_lin, val_par in zip(result_lin.values(), result_par.values()):
        assert np.allclose(val_lin, val_par)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
@pytest.mark.parametrize("eps", [0.01, 0.04, 0.08])
@pytest.mark.parametrize(
    "method,start",
    [("last_layer", None), ("iterative", "two"), ("iterative", "simple_success")],
)
def test_repairs(n, eps, method, start):
    num_instances = 1000
    for instance in range(num_instances):
        tree = QRAMRouter()
        full_tree = tree.create_tree(n)
        rng_seed = 2545685 * n + 2667485973 * instance + 2674
        rng = np.random.default_rng(rng_seed)
        full_tree.fabrication_instance(eps, rng, top_three_functioning=True)
        repair, flag_qubits, _, _ = full_tree.router_repair(method=method, start=start)
        assert flag_qubits < n or flag_qubits == np.inf
        if n <= 4:
            repair_brute, flag_brute, _, _ = full_tree.router_repair(
                method="brute_force"
            )
            assert flag_brute <= flag_qubits


@pytest.mark.parametrize("n", [5, 7, 10])
@pytest.mark.parametrize("eps", [0.02, 0.08])
@pytest.mark.parametrize("top_three_functioning", [True, False])
def test_fabrication_instance(n, eps, top_three_functioning):
    num_instances = 20000
    rng_seed = 2545685 * n
    rng = np.random.default_rng(rng_seed)
    rando_faulty_counter = 0
    for instance in range(num_instances):
        tree = QRAMRouter()
        full_tree = tree.create_tree(n)
        full_tree.fabrication_instance(
            eps, rng, top_three_functioning=top_three_functioning
        )
        assert full_tree.tree_depth() == n
        if top_three_functioning:
            assert full_tree.functioning
            assert (
                full_tree.right_child.functioning and full_tree.left_child.functioning
            )
        if not full_tree.right_child.left_child.left_child.functioning:
            rando_faulty_counter += 1
    if not top_three_functioning:
        ideal_eps = eps + (1 - eps) * eps + (1 - eps) ** 2 * eps + (1 - eps) ** 3 * eps
    else:
        ideal_eps = eps + (1 - eps) * eps
    assert ideal_eps - 0.01 < rando_faulty_counter / num_instances < ideal_eps + 0.01


def test_specific_faulty_tree():
    n = 6
    eps = 0.05
    rng_seed = 2392900
    rng = np.random.default_rng(rng_seed)
    faulty_tree = QRAMRouter()
    faulty_tree = faulty_tree.create_tree(n)
    faulty_tree.fabrication_instance(eps, rng, top_three_functioning=True)
    assignment_last_layer, flag_qubits_last_layer, _, _ = faulty_tree.router_repair(
        method="last_layer"
    )
    assignment_iterative, flag_qubits_iterative, _, _ = faulty_tree.router_repair(
        method="iterative"
    )
    assert flag_qubits_last_layer == flag_qubits_iterative == 1
