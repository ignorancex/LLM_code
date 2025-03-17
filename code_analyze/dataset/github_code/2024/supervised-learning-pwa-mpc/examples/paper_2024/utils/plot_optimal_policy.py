import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from examples.paper_2024.model import Model
from slpwampc.misc.regions import Polytope

N = 5
system = Model.get_system()

with open(
    f"examples/paper_2024/results/optimal_policy_N_{N}_samples_Tset.pkl",
    "rb",
) as file:
    data = pickle.load(file)
    states_opt = data["x"]
    actions_opt = data["a"]

fig, ax = plt.subplots()
p = Polytope(system.D, system.E)
p.plot(ax)
for o in np.unique(actions_opt):
    idx = np.where(np.array(actions_opt) == o)
    ax.scatter(
        np.array(states_opt)[idx, 0],
        np.array(states_opt)[idx, 1],
        label=f"Region {o}",
        s=1.5,
        linewidth=1.5,
        marker="s",
        color=f"C{o}",
    )
plt.xlim(-6.2, 8.2)
plt.ylim(-10.5, 10.5)
ax.axis("off")
# ax.legend()
plt.show()
