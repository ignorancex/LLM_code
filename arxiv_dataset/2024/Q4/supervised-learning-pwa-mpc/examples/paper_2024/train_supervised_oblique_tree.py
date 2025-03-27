import pickle

import numpy as np
from model import Model
from mpc_mld import ThisMpcMld, ThisTightenedMpcMld

from slpwampc.agents.oblique_decision_tree_agent import ObliqueDecisionTreeAgent

np_random = np.random.default_rng(0)

SAVE = True
first_region_from_policy = False

N = 6  # prediction horizon
d = 2  # spacing for initial grid sampling

nx, nu = Model.nx, Model.nu
system = Model.get_system()
system_dict = Model.get_system_dict()
# system.B = [np.array([[1], [1]]), np.array([[1], [1]])]
# system_dict["B"] = system.B

mpc = ThisMpcMld(system_dict, N, nx, nu, verbose=False)
tighened_mpc = ThisTightenedMpcMld(system_dict, N, nx, nu, 0.1, verbose=False)
initial_state_samples = Model.sample_state_space(
    d=d, np_random=np_random, sample_strategy="grid"
)

agent = ObliqueDecisionTreeAgent(
    system,
    mpc,
    N,
    num_restarts=100,
    max_perturbations=3,
    num_trees=5,
    first_region_from_policy=first_region_from_policy,
    tightened_mpc=tighened_mpc,
    learn_infeasible_regions=True,
)
x, y, random_state = agent.train(initial_state_samples, plot=True, interactive=True)

if SAVE:
    with open(f"odt_policy_N_{N}_mayne_2d.pkl", "wb") as file:
        pickle.dump({"x": x, "y": y, "random_state": random_state}, file)

# validate
# action_mapper = PwaActionMapper(len(system.A), N if first_region_from_policy else N - 1)
# validation_points = 10000
# validation_samples = Model.sample_state_space(
#     num_points=validation_points, np_random=np_random, sample_strategy="random"
# )

# wrong_feas_count = 0
# wrong_infeas_count = 0
# for idx, state in enumerate(validation_samples):
#     print(f"validating {idx}'th state")
#     optimal_sequence = agent.get_switching_sequence_from_state(state)
#     if optimal_sequence is None:
#         optimal_sequence = None
#     odt_action = tree.predict(np.ascontiguousarray(state.T))
#     label = label_map[odt_action.item()]
#     if label == -1:
#         tree_sequence = None
#     else:
#         tree_sequence = action_mapper.get_action_from_label(np.array([[label]]))

#     if optimal_sequence is not None and tree_sequence is None:
#         wrong_infeas_count += 1
#     if optimal_sequence is None and tree_sequence is not None:
#         raise ValueError("Infeasible state classified as feasible")
#     if optimal_sequence is not None and tree_sequence is not None:
#         if not np.array_equal(optimal_sequence, tree_sequence[0]):
#             wrong_feas_count += 1

#     if tree_sequence is not None:
#         mpc.solve_mpc_with_switching_sequence(state, tree_sequence[0], raises=True)

# print(f"Wrong feas count: {wrong_feas_count}")
# print(f"Wrong infeas count: {wrong_infeas_count}")
