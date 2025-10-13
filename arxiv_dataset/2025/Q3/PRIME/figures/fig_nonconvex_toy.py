import matplotlib.pyplot as plt
from tqdm import tqdm

from feedback_opt.optimizers import (
    OptimizerDualHProximal,
    OptimizerDualYProximal,
    OptimizerPrimal,
)
from feedback_opt.simulation import Simulation
from feedback_opt.systems import SystemNonLinear
from feedback_opt.utils import plot_cost_and_violation
from scenarios.scenario_nonconvex_toy import NonConvexToy


def fig_nonconvex_toy():
    # make data
    # fetch parameters for scenario
    params = NonConvexToy()

    # override
    plt.rcParams.update({"figure.figsize": (6, 4)})

    # fetch parameters for scenario
    params = NonConvexToy()

    # instatiate objects
    system = SystemNonLinear(params.sys)
    optimizer = [
        OptimizerPrimal(params.opt_prim, system),
        OptimizerDualYProximal(params.opt_dualyprox_dist, system),
        OptimizerDualHProximal(params.opt_dualhprox_dist, system),
    ]
    simulation = [Simulation(params.sim, system, opt) for opt in optimizer]
    results = [(sim.optimizer.name, sim.run()) for sim in tqdm(simulation)]

    # plot
    plot_cost_and_violation(results, x_tick_spacing=10)


if __name__ == "__main__":
    fig_nonconvex_toy()
    plt.show()
