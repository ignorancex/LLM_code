import numpy as np
# config = {
#     "d": 32,
#     "k": 1,
#     "sigma": 1,
#     "N": 10000,
#     "n_ls": np.array([50, 100, 200]),
#     "m0": 50,
#     "mul_ls": np.array([1, 2, 4, 8, 16, 32]),
#     "M": 1000,
#     "num_trials": 1,
# }

config = {
    "d": 10,
    "l1": 0,
    "l2": 1.0,
    "N": 5000,
    #"n_ls": np.array([50, 100, 250, 500, 1000]),
    "n_ls": np.array([2000, 4000, 8000]),
    "m0": 800,
    "mul_ls": np.array([1, 2, 4, 8, 16, 32, 64, 128]),
    "M": 1000,
    "Ltype": 'const',
    "Vtype": 'lognormal',
}


train_config = {
    "learning_rate": 0.01,
    "decay_rate": 0.95,
    "decay_epoch": 400,
    "epochs": 2000,
}