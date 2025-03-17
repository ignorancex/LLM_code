import os
import pickle
import sys

from sklearn_oblique_tree.oblique import ObliqueTree

sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
from examples.Mallick_2024.mpc_mld import ThisMpcMld
from examples.Mayne_2003.simple_2D.model import Model
from rollout.agents.oblique_decision_tree_agent import ObliqueDecisionTreeAgent
from rollout.misc.regions import Polytope

with open(
    "examples/Mayne_2003/simple_2D/results/odt_policy_N_6_mayne_2d.pkl", "rb"
) as file:
    mayne_data = pickle.load(file)
    x = mayne_data["x"]
    y = mayne_data["y"]
    random_state = mayne_data["random_state"]

N = 3
nx = 2
nu = 1

system = Model.get_system()
system_dict = Model.get_system_dict()
tree = ObliqueTree(
    splitter="oc1",
    number_of_restarts=100,
    max_perturbations=3,
    random_state=random_state,
)
mpc = ThisMpcMld(system_dict, N, nx, nu, verbose=False)
agent = ObliqueDecisionTreeAgent(
    system,
    mpc,
    3,
    num_restarts=100,
    max_perturbations=3,
    num_trees=1,
    first_region_from_policy=False,
    tightened_mpc=None,
    learn_infeasible_regions=True,
)

_, odt_y = agent.get_oblique_label_map(y)

tree.fit(x, odt_y)

partition = tree.get_partition()
regions = [
    Polytope(np.vstack((r[:, :-1], system.D)), np.vstack((r[:, [-1]], system.E)))
    for r in partition
]

# label each region using the trained tree # TODO get labels directly from c code
fig, ax = plt.subplots()
for region in regions:
    if not region.is_empty:
        region.set_label(lambda x: tree.predict(x.T))
        region.plot(ax=ax)

plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.show()
