import cupy as cp
import pytest
import qutip as qt

from qcheff.magnus.magnus_gpu import MagnusTimeEvol
from qcheff.magnus.pulses import ControlPulse, FourierPulse
from qcheff.magnus.system import QuTiPSystem

nintervals_list = [10**n for n in range(1, 7)]
global_gate_time = 50
global_num_points = 10**6

cp.cuda.Device(1).use()


def create_single_spin_system(drive_freq=1.0):
    drift_ham = -cp.pi * qt.sigmaz()  # qubit frequency = 1
    control_sigs = [
        ControlPulse(
            FourierPulse([1], gate_time=global_gate_time),
            frequency=drive_freq,
            amplitude=1,
        )
    ]
    control_hams = [qt.sigmax()]
    system = QuTiPSystem(drift_ham, control_sigs, control_hams)
    return system


@pytest.mark.parametrize("num_intervals", nintervals_list)
def test_magnus_ham_gpu(benchmark, num_intervals):
    system = create_single_spin_system()
    tlist = cp.linspace(0, global_gate_time, global_num_points)
    points_per_interval = len(tlist) // num_intervals
    magnus = MagnusTimeEvol(system, tlist)

    benchmark.pedantic(
        magnus._magnus1ham,
        kwargs={
            "num_intervals": int(num_intervals),
            "points_per_interval": points_per_interval,
        },
        rounds=20,
    )


@pytest.mark.parametrize("num_intervals", nintervals_list)
def test_magnus_prop_gpu(benchmark, num_intervals):
    system = create_single_spin_system()
    tlist = cp.linspace(0, global_gate_time, global_num_points)
    points_per_interval = len(tlist) // num_intervals
    magnus = MagnusTimeEvol(system, tlist)

    benchmark.pedantic(
        magnus._magnus1prop,
        kwargs={
            "num_intervals": int(num_intervals),
            "points_per_interval": points_per_interval,
        },
        rounds=20,
    )
