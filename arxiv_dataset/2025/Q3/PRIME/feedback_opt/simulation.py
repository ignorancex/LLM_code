import numpy as np
import pandas as pd

from feedback_opt.optimizers.optimizer_base import OptimizerBase
from feedback_opt.systems.system_base import SystemBase


class Simulation:
    def __init__(
        self,
        params,
        system: SystemBase,
        optimizer: OptimizerBase,
    ):
        # system
        assert isinstance(system, SystemBase)
        self.system = system

        # optimizer
        assert isinstance(optimizer, OptimizerBase)
        self.optimizer = optimizer

        # initial u guess
        if hasattr(params, "u_0"):
            assert np.shape(params.u_0) == (self.system.m, 1)
            self.u_0 = params.u_0
        else:
            self.u_0 = np.zeros((self.system.m, 1))

        # optimal u (known in advance, used for calculating suboptimality)
        if hasattr(params, "u_opt"):
            assert np.shape(params.u_opt) == (self.system.m, 1)
            self.u_opt = params.u_opt
        else:
            self.u_opt = np.zeros((self.system.m, 1))

        # simulation length
        assert isinstance(params.n_steps, int)
        self.n_steps = params.n_steps

        # measurement noise
        if hasattr(params, "noise_seed"):
            self.noise_seed = params.noise_seed
        else:
            self.noise_seed = 0
        if hasattr(params, "noise_y_std"):
            self.noise_y_std = params.noise_y_std
        else:
            self.noise_y_std = 0

    def log(self, history: dict, i: int, row_i: dict):
        assert history.keys() == row_i.keys(), "Dictionaries do not have the same keys"
        for key, value in history.items():
            value[i, :] = row_i[key].flatten()

    def run(self):
        # reset optimizer
        data_k = self.optimizer.data_initial(self.u_0)

        # initialize suboptimality tracker
        data_k["d"] = np.linalg.norm(data_k["u"] - self.u_opt, keepdims=True)

        # initialize history dict
        history = {
            key: np.zeros((self.n_steps + 1, value.shape[0])) for key, value in data_k.items()
        }

        # log initial timestep k=0
        self.log(history, 0, data_k)

        # sample measurement noise
        np.random.seed(self.noise_seed)
        y_noise = np.random.normal(scale=self.noise_y_std, size=(self.system.p, self.n_steps))

        ## simulation loop
        for i in range(self.n_steps):
            # add measurement noise
            if y_noise is not None:
                data_k["y"] = data_k["y"] + y_noise[:, i].reshape(-1, 1)

            # feedback optimization
            data_kp1 = self.optimizer.data_step(data_k)

            # apply new u at system
            data_kp1["y"] = self.system.h(data_kp1["u"])

            # evaluate cost function
            data_kp1 = self.optimizer.data_cost(data_kp1)

            # evaluate y violation
            data_kp1 = self.optimizer.data_y_violation(data_kp1)

            # evaluate dist to opt
            data_kp1["d"] = np.linalg.norm(data_k["u"] - self.u_opt)

            # log noise free performance
            self.log(history, i + 1, data_kp1)

            data_k = data_kp1

        # convert history dict to dataframe
        colnames = []
        data = []
        for name, matrix in history.items():
            num_cols = matrix.shape[1]
            if num_cols == 1:
                colnames.extend([name])
            else:
                colnames.extend([f"{name}_{i}" for i in range(num_cols)])
            data.append(matrix)
        data = np.hstack(data)
        result = pd.DataFrame(data, columns=colnames)
        result.index.name = "t"

        return result
