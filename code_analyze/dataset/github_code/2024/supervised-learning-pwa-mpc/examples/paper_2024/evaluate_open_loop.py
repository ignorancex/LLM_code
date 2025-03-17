import pickle

import matplotlib.pyplot as plt
import numpy as np
from model import Model
from mpc_mld import ThisMpcMld

from slpwampc.agents.parc_agent import ParcAgent

np_random = np.random.default_rng(0)
np.random.seed(2)

GENERATE = False  # if false we just plot from already saved data

N = 12  # prediction horizon

nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()

mpc = ThisMpcMld(system_dict, N, nx, nu, X_f=Model.X_f, verbose=False)
agent = ParcAgent(
    system,
    mpc,
    N,
    learn_infeasible_regions=True,
)
agent.load(f"examples/paper_2024/results/parc_agent_N_{N}")

if GENERATE:
    initial_state_samples = Model.sample_state_space(
        d=0.1, np_random=np_random, sample_strategy="grid", num_points=2000
    )
    states = []
    costs_opt = []
    costs_subopt = []
    for i, x0 in enumerate(initial_state_samples):
        print(f"{i}/{len(initial_state_samples)}")
        x0 = x0.reshape(-1, 1)
        _, info = agent.mpc.solve_mpc(state=x0, raises=False)
        if info["cost"] < np.inf:
            states.append(x0)
            costs_opt.append(info["cost"])
            delta = agent.get_switching_sequence(x0)
            if delta is None:
                costs_subopt.append(-1)
            else:
                _, info = agent.mpc.solve_mpc_with_switching_sequence(
                    x0, delta, raises=True
                )
                costs_subopt.append(info["cost"])
    with open(f"examples/paper_2024/results/evaluate_open_loop_N_{N}.pkl", "wb") as f:
        pickle.dump(
            {"states": states, "costs_opt": costs_opt, "costs_subopt": costs_subopt}, f
        )

else:
    with open(f"examples/paper_2024/results/evaluate_open_loop_N_{N}.pkl", "rb") as f:
        data = pickle.load(f)
        states = data["states"]
        costs_opt = data["costs_opt"]
        costs_subopt = data["costs_subopt"]

    f = np.array(
        [
            100 * (costs_subopt[i] - costs_opt[i]) / costs_opt[i]
            for i in range(len(costs_opt))
            if costs_subopt[i] != -1 and costs_opt[i] != 0
        ]
    )
    x = np.array(
        [
            states[i][0]
            for i in range(len(states))
            if costs_subopt[i] != -1 and costs_opt[i] != 0
        ]
    ).reshape(-1)
    y = np.array(
        [
            states[i][1]
            for i in range(len(states))
            if costs_subopt[i] != -1 and costs_opt[i] != 0
        ]
    ).reshape(-1)
    x_o = np.array([states[i][0] for i in range(len(states)) if costs_subopt[i] == -1])
    y_o = np.array([states[i][1] for i in range(len(states)) if costs_subopt[i] == -1])
    contour = plt.tripcolor(x, y, f, cmap="coolwarm")
    plt.colorbar(contour)
    contour.set_rasterized(True)
    plt.axis("off")
    plt.show()
