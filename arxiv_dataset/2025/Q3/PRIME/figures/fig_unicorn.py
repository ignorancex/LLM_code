import matplotlib.pyplot as plt
from tqdm import tqdm

from feedback_opt.optimizers import (
    OptimizerDualHProximal,
    OptimizerDualY,
    OptimizerDualYProximal,
    OptimizerPrimal,
)
from feedback_opt.simulation import Simulation
from feedback_opt.systems import SystemElectrical
from feedback_opt.utils import plot_cost_and_violation
from scenarios.scenario_unicorn import Unicorn


def fig_unicorn():
    # make data
    # fetch parameters for scenario
    params = Unicorn()

    # override
    plt.rcParams.update({"figure.figsize": (6, 4)})

    # instatiate objects
    system = SystemElectrical(params.sys)
    optimizer = [
        OptimizerPrimal(params.opt_prim, system),
        OptimizerDualY(params.opt_dualy, system),
        OptimizerDualYProximal(params.opt_dualyprox_dist, system),
        OptimizerDualHProximal(params.opt_dualhprox_dist, system),
    ]
    simulation = [Simulation(params.sim, system, opt) for opt in optimizer]
    results = [(sim.optimizer.name, sim.run()) for sim in tqdm(simulation)]

    # plot
    plot_cost_and_violation(results, transition=15, max_violation=0.06)


if __name__ == "__main__":
    fig_unicorn()
    plt.show()
