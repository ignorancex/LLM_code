import numpy as np
import torch

device = torch.device("cpu")

# Inputs population models
inputs = np.array(["input_insulin", "input_meal_carbs"])

# Inputs individual models
input_ind = [
    "heart_rate_WRTbaseline",
    "feat_is_weekend",
    "feat_hour_of_day_cos",
    "feat_hour_of_day_sin",
    "sleep_efficiency",
]
idx_robust = [0]

# System's states
states_name = np.array(["Q1", "Q2", "S1", "S2", "I", "X1", "X2", "X3", "C2", "C1"])
states = [
    "output_cgm",
    "state_Q2",
    "state_S1",
    "state_S2",
    "state_I",
    "state_X1",
    "state_X2",
    "state_X3",
    "state_Ug",
    "state_Ug2",
]
states_nobs = [
    "state_Q2",
    "state_S1",
    "state_S2",
    "state_I",
    "state_X1",
    "state_X2",
    "state_X3",
    "state_Ug",
    "state_Ug2",
]

dict_states = {
    "Q1": ["X1", "X3", "Q1", "Q2", "C2"],
    "Q2": ["X1", "X2", "Q1", "Q2"],
    "S1": ["S1", "input_insulin"],
    "S2": ["S1", "S2"],
    "I": ["S2", "I"],
    "X1": ["I", "X1"],
    "X2": ["I", "X2"],
    "X3": ["I", "X3"],
    "C2": ["C1", "C2"],
    "C1": ["C1", "input_meal_carbs"],
}

# Architecture population models
best_params = {
    "batch_size_factor": 126.26958227466093,
    "lr_factor": 3.9027719714406333,
    "overlap": 0.75,
    "seq_len_h": 5.0,
    "c_one": 149.940773,
    "c_two": 140.914069,
    "ins": 197.753064,
    "q_one": 248.656691,
    "q_two": 80.851555,
    "s_one": 174.178028,
    "s_two": 60.289599,
    "x_one": 191.578477,
    "x_three": 127.210117,
    "x_two": 101.790715,
}

n_neurons_pop = {
    "Q1": int(best_params["q_one"]),
    "Q2": int(best_params["q_two"]),
    "S1": int(best_params["s_one"]),
    "S2": int(best_params["s_two"]),
    "I": int(best_params["ins"]),
    "X1": int(best_params["x_one"]),
    "X2": int(best_params["x_two"]),
    "X3": int(best_params["x_three"]),
    "C2": int(best_params["c_two"]),
    "C1": int(best_params["c_one"]),
}

# Architecture individual models
n_neurons = 128
hidden_compartments = {
    "models": [5 + len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
}

