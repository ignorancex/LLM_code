import matplotlib.pyplot as plt

from t1dsim_ai.utils.preprocess import scaler as scaler_pop
from t1dsim_ai.utils.preprocess import (
    scaler_inverse,
    scale_single_state,
    scale_inverse_Q1,
)
from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import (
    n_neurons_pop,
    hidden_compartments,
    states,
    states_nobs,
    inputs,
    input_ind,
    idx_robust,
)

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import numpy as np
from pickle import load, dump
from librosa.util import frame
import pandas as pd


class WeightClipper(object):
    def __init__(self, min=-1, max=1):

        self.min = min
        self.max = max

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(self.min, self.max)


class ForwardEulerSimulator(nn.Module):

    """This class implements prediction/simulation methods for the SS models structure

     Attributes
     ----------
     ss_pop_model: nn.Module
                   The population-level neural state space models to be fitted
    ss_ind_model: nn.Module
                   The individual-level neural state space models to be fitted
     ts: float
         models sampling time

    """

    def __init__(self, ss_pop_model, ss_ind_model, path_scaler, ts=1.0):
        super(ForwardEulerSimulator, self).__init__()
        self.ss_pop_model = ss_pop_model
        self.ss_ind_model = ss_ind_model

        self.ts = ts
        self.cgm_min = scale_single_state(40, "Q1", path_scaler)
        self.cgm_max = scale_single_state(400, "Q1", path_scaler)

    def adjust_cgm(self, x):
        x[x > self.cgm_max] = self.cgm_max
        x[x < self.cgm_min] = self.cgm_min

        return x

    def forward(
        self, x0_batch: torch.Tensor, u_batch: torch.Tensor, u_batch_ind, is_pers=True
    ) -> torch.Tensor:
        """Multi-step simulation over (mini)batches

        Parameters
        ----------
        x0_batch: Tensor. Size: (q, n_x)
             Initial state for each subsequence in the minibatch

        u_batch: Tensor. Size: (m, q, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (m, q, n_x)
            Simulated state for all subsequences in the minibatch

        """

        # X_sim_list: List[torch.Tensor] = []
        X_sim_list: [torch.Tensor] = []

        x_step = x0_batch

        for step in range(u_batch.shape[0]):
            u_step = u_batch[step]

            if is_pers:
                u_ind_step = u_batch_ind[step]

            X_sim_list += [x_step]

            dx_pop = self.ss_pop_model(x_step, u_step)

            dx = dx_pop
            if is_pers:
                dx_ind = self.ss_ind_model(x_step, u_step, u_ind_step)
                dx[:, 0] += dx_ind[:, 0]

            x_step = x_step + self.ts * dx

            if (len(x_step.shape)) == 1:
                x_step[0] = self.adjust_cgm(x_step[0])

            else:
                x_step[:, 0] = self.adjust_cgm(x_step[:, 0])

        X_sim = torch.stack(X_sim_list, 0)


        return X_sim


class CGMIndividual(nn.Module):
    def __init__(self, hidden_compartments, init_small=True):
        super(CGMIndividual, self).__init__()

        # NN - Model
        layers_model = []
        for i in range(len(hidden_compartments["models"]) - 2):
            layers_model.append(
                nn.Linear(
                    hidden_compartments["models"][i],
                    hidden_compartments["models"][i + 1],
                )
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["models"][-2], 1))
        self.net_model = nn.Sequential(*layers_model)

        clipper = WeightClipper()

        if init_small:
            networks = {"Model": self.net_model}

            for key in networks.keys():
                net = networks[key]
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-4)
                        nn.init.constant_(m.bias, val=0)

                net.apply(clipper)

    def forward(self, in_x, u_pop, u_ind):
        # q1, q2, s1, s2, I, x1, x2, x3, c2, c1
        q1, q2, _, _, _, x1, _, x3, c2, _ = (
            in_x[..., [0]],
            in_x[..., [1]],
            in_x[..., [2]],
            in_x[..., [3]],
            in_x[..., [4]],
            in_x[..., [5]],
            in_x[..., [6]],
            in_x[..., [7]],
            in_x[..., [8]],
            in_x[..., [9]],
        )

        inp = torch.cat((q1, q2, x1, x3, c2, u_ind), -1)
        dQ1_Ind = self.net_model(inp)

        return dQ1_Ind


class IndividualModel:
    def __init__(self, subjectID, df_subj, pathModel, device="cpu"):

        self.subjectID = subjectID
        self.df_subj = df_subj
        self.device = device

        self.popModel = (
            Path(__file__).parent
            / "models/PopulationModel/population_model_05022024_epoch_15.pt"
        )
        self.popModelFolder = str(Path(__file__).parent) + "/models/PopulationModel/"

        self.pathModelFolder = pathModel
        self.pathModel = pathModel + subjectID

        self.split_train_test(states, inputs, input_ind)

        self.LIM_INFERIOR = scale_single_state(70, "Q1", self.popModelFolder)
        self.LIM_SUPERIOR = scale_single_state(250, "Q1", self.popModelFolder)

    def split_train_test(self, states, inputs_pop, input_ind):

        self.df_subj[states_nobs] = 0
        df_subj_train = self.df_subj.loc[self.df_subj.is_train]
        df_subj_test = self.df_subj.loc[~self.df_subj.is_train]

        sim_time_train = len(df_subj_train)
        sim_time_test = len(df_subj_test)

        x_est_train = np.array(df_subj_train[states].values).astype(np.float32)
        u_pop_train = np.array(df_subj_train[inputs_pop].values).astype(np.float32)
        u_ind_train = np.array(df_subj_train[input_ind].values).astype(np.float32)

        x_est_test = np.array(df_subj_test[states].values).astype(np.float32)
        u_pop_test = np.array(df_subj_test[inputs_pop].values).astype(np.float32)
        u_ind_test = np.array(df_subj_test[input_ind].values).astype(np.float32)

        # Scale states and inputs from the population model
        x_est_train, u_pop_train = scaler_pop(
            x_est_train, u_pop_train, self.popModelFolder, False
        )
        x_est_test, u_pop_test = scaler_pop(
            x_est_test, u_pop_test, self.popModelFolder, False
        )

        # Scale new inputs
        self.scaler_featsRobust = load(
            open(Path(__file__).parent / "models/scaler_robust.pkl", "rb")
        )
        u_ind_train[:, idx_robust] = self.scaler_featsRobust.fit_transform(
            u_ind_train[:, idx_robust]
        )
        u_ind_test[:, idx_robust] = self.scaler_featsRobust.transform(
            u_ind_test[:, idx_robust]
        )

        self.cgm_real_train = scale_inverse_Q1(np.copy(x_est_train[:, 0]),self.popModelFolder).reshape(-1, sim_time_train, 1)
        self.y_id_train = np.copy(x_est_train[:, 0]).reshape(-1, sim_time_train, 1)
        self.x_est_train = x_est_train.reshape(-1, sim_time_train, len(states))
        self.u_pop_train = u_pop_train.reshape(-1, sim_time_train, len(inputs_pop))
        self.u_ind_train = u_ind_train.reshape(-1, sim_time_train, len(input_ind))

        self.cgm_real_test = scale_inverse_Q1(np.copy(x_est_test[:, 0]),self.popModelFolder).reshape(-1, sim_time_test, 1)
        self.y_id_test = np.copy(x_est_test[:, 0]).reshape(-1, sim_time_test, 1)
        self.x_est_test = x_est_test.reshape(-1, sim_time_test, len(states))
        self.u_pop_test = u_pop_test.reshape(-1, sim_time_test, len(inputs_pop))
        self.u_ind_test = u_ind_test.reshape(-1, sim_time_test, len(input_ind))

        print("Number of training points:", self.x_est_train.shape[1])
        print("Number of testing points:", self.x_est_test.shape[1])

    def setup_nn(
        self,
        hidden_compartments,
        lr,
        batch_size,
        n_epochs,
        overlap=0.9,
        seq_len = 61,
        ts=5,
        weight_decay=1e-5,
    ):

        # Batch extraction class
        self.seq_len = seq_len
        self.batch = Batch(
            batch_size,
            self.seq_len,
            overlap,
            self.device,
            [self.x_est_train, self.u_pop_train, self.y_id_train, self.u_ind_train,self.cgm_real_train],
        )

        # Setup neural model structure
        self.individual_model = CGMIndividual(hidden_compartments=hidden_compartments)
        self.individual_model.to(self.device)

        # Load population model
        self.ss_pop_model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        self.ss_pop_model.to(self.device)
        self.ss_pop_model.load_state_dict(torch.load(self.popModel))

        for name, param in self.ss_pop_model.named_parameters():
            param.requires_grad = False

        # Simulator
        self.nn_solution = ForwardEulerSimulator(
            self.ss_pop_model, self.individual_model, self.popModelFolder, ts=ts
        )

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.individual_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # lambda1 = lambda epoch: np.exp(-0.1) ** epoch
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda1)

        self.n_epochs = n_epochs + 1
        self.n_iter_max = self.n_epochs * self.batch.n_iter_per_epoch
        self.curr_epoch = 1

        self.best_loss = float("inf")
        self.best_model = None
        self.epochs_without_improvement = 0
        self.max_epochs_without_improvement = 150

    def fit(self, save_model):

        # Training loop
        LOSS = []
        LOSS_TRAIN = []

        print("---Epoch {}: lr {}---".format(1, self.optimizer.param_groups[0]["lr"]))
        loss_temp = []

        while True:  # for itr in range(0, self.n_iter_max):
            self.optimizer.zero_grad()
            # Simulate
            (
                batch_x0_hidden,
                batch_u_pop,
                batch_u_ind,
                batch_y,
                batch_x_original,
            ) = self.batch.get_batch(True)
            batch_x_sim = self.nn_solution(batch_x0_hidden, batch_u_pop, batch_u_ind)

            if torch.isnan(batch_x_sim).any() or torch.isinf(batch_x_sim).any():
                print("INFO: Training had stopped because an inf in batch simulation")
                return np.nan

            # Compute fit loss
            loss = self.loss(batch_x_sim[:, :, [0]], batch_y).to(self.device)
            loss_temp.append(loss.item())

            if self.curr_epoch < self.batch.epoch:
                LOSS.append(np.mean(loss_temp))
                loss_temp = []

                self.curr_epoch = self.batch.epoch

                # if self.curr_epoch%50==0 and self.curr_epoch>10:
                #    self.scheduler.step()

                with torch.no_grad():
                    (
                        batch_x0_hidden,
                        batch_u_pop,
                        batch_u_ind,
                        batch_y,
                        batch_x_original,
                    ) = self.batch.get_all("All")
                    batch_x_sim = self.nn_solution(
                        batch_x0_hidden, batch_u_pop, batch_u_ind
                    )
                    LOSS_TRAIN.append(
                        torch.sqrt(
                            torch.mean(
                                (
                                    scale_inverse_Q1(
                                        batch_x_sim[:, :, [0]], self.popModelFolder
                                    )
                                    - scale_inverse_Q1(
                                        batch_x_original[:, :, [0]], self.popModelFolder
                                    )
                                )
                                ** 2
                            )
                        ).item()
                    )

                # Early stopping condition

                # if LOSS[-1] < self.best_loss:
                #    self.best_loss = LOSS[-1]
                #    self.best_model = self.nn_solution.ss_ind_model.state_dict()
                #    self.epochs_without_improvement = 0

                # else:
                #    self.epochs_without_improvement += 1

                # if self.epochs_without_improvement >= self.max_epochs_without_improvement:
                #    print("Early stopping after {} epochs without improvement.".format(self.epochs_without_improvement))
                #    break

                if self.curr_epoch == self.n_epochs:
                    break
                else:
                    print(
                        f"Epoch {self.curr_epoch-1} | Loss {LOSS[-1]:.6f}  Simulation Loss {LOSS_TRAIN[-1]:.6f} "
                    )
                    # print('---Epoch {} - lr {}---'.format(self.curr_epoch, self.optimizer.param_groups[0]['lr']))
            # Optimize
            loss.backward()
            self.optimizer.step()

        if self.best_model is None:
            self.best_model = self.nn_solution.ss_ind_model.state_dict()

        if save_model:
            if not os.path.exists(self.pathModel):
                os.mkdir(self.pathModel)
            dump(
                self.scaler_featsRobust,
                open(self.pathModel + "/scaler_robust.pkl", "wb"),
            )
            torch.save(
                self.best_model, os.path.join(self.pathModel + "/individual_model.pt")
            )

        return LOSS_TRAIN[-1]
    #"""
    def loss(self, y_pred, y_true):
        err_fit = y_pred[1:, :] - y_true[1:, :]
        err_df = torch.diff(y_pred, axis=0) - torch.diff(y_true, axis=0)

        penalty = torch.ones_like(y_true[1:, :])

        penalty[
            torch.logical_and(
                y_true[1:, :] <= self.LIM_INFERIOR, y_pred[1:, :] > y_true[1:, :]
            )
        ] = 6
        penalty[
            torch.logical_and(
                y_true[1:, :] >= self.LIM_SUPERIOR, y_pred[1:, :] < y_true[1:, :]
            )
        ] = 6

        MSE_cgm = torch.mean(((err_fit) ** 2 * penalty))
        MSE_Dcgm = torch.mean((err_df) ** 2)

        return MSE_cgm + 10 * MSE_Dcgm
    """
    def loss(self, y_pred, y_true):
        err_fit = y_pred[1:, :] - y_true[1:, :]
        MSE_cgm = torch.mean(((err_fit) ** 2))
        return MSE_cgm
    """

class Batch:
    def __init__(self, batch_size, seq_len, overlap, device, data):

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.overlap = int((1 - overlap) * self.seq_len)
        self.device = device

        # Reshape
        x_est, u_fit, y_fit, u_fit_ind, cgm_real = data
        self.x_est = np.array(
            [
                frame(
                    x_est[i], frame_length=self.seq_len, hop_length=self.overlap, axis=0
                )
                for i in range(len(x_est))
            ]
        ).reshape(-1, seq_len, x_est.shape[2])
        self.u_fit = np.array(
            [
                frame(
                    u_fit[i], frame_length=self.seq_len, hop_length=self.overlap, axis=0
                )
                for i in range(len(u_fit))
            ]
        ).reshape(-1, seq_len, u_fit.shape[2])
        self.y_fit = np.array(
            [
                frame(
                    y_fit[i], frame_length=self.seq_len, hop_length=self.overlap, axis=0
                )
                for i in range(len(y_fit))
            ]
        ).reshape(-1, seq_len, y_fit.shape[2])
        self.u_fit_ind = np.array(
            [
                frame(
                    u_fit_ind[i],
                    frame_length=self.seq_len,
                    hop_length=self.overlap,
                    axis=0,
                )
                for i in range(len(u_fit_ind))
            ]
        ).reshape(-1, seq_len, u_fit_ind.shape[2])
        self.cgm_real = np.array(
            [
                frame(
                    cgm_real[i], frame_length=self.seq_len, hop_length=self.overlap, axis=0
                )
                for i in range(len(cgm_real))
            ]
        ).reshape(-1, seq_len, cgm_real.shape[2])

        idx_scenarios = self.filter_seq()  # Filter out sequences

        # Define initial states
        for scenario in idx_scenarios:
            cgm_target = self.cgm_real[scenario, 0, 0].item()
            self.x_est[scenario, 0, :] = getInitSSFromFile(cgm_target)


        self.idx_scenarios_temp = idx_scenarios
        self.idx_scenarios = idx_scenarios

        self.num_scenarios = len(self.idx_scenarios)

        self.n_iter_per_epoch = int(
            self.num_scenarios / self.batch_size
        )  # Number of iterations per epoch
        self.update_batch_idx()
        self.epoch = 1

        print("Number of iteration per epoch:", self.n_iter_per_epoch)
        print("Number of scenarios:", self.num_scenarios)

    def get_all(self, group):

        idx = self.idx_scenarios

        batch_start = np.zeros(idx.shape[0], dtype=np.int8)
        batch_idx = batch_start[:, np.newaxis] + np.arange(
            self.seq_len
        )  # batch samples indices
        batch_idx = (
            batch_idx.T
        )  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data
        batch_x0_hidden = torch.tensor(
            self.x_est[idx, batch_start, :], dtype=torch.float32
        ).to(self.device)
        batch_u_pop = torch.tensor(self.u_fit[idx, batch_idx], dtype=torch.float32).to(
            self.device
        )
        batch_u_ind = torch.tensor(
            self.u_fit_ind[idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_y = torch.tensor(self.y_fit[idx, batch_idx], dtype=torch.float32).to(
            self.device
        )
        batch_x_original = torch.tensor(
            self.x_est[idx, batch_idx], dtype=torch.float32
        ).to(self.device)

        return batch_x0_hidden, batch_u_pop, batch_u_ind, batch_y, batch_x_original

    def get_batch(self, count=True):

        self.batch_scenarios_idx = self.batch_scenarios_idx.astype(int)
        batch_start = np.zeros(self.batch_scenarios_idx.shape[0], dtype=np.int8)
        batch_idx = batch_start[:, np.newaxis] + np.arange(
            self.seq_len
        )  # batch samples indices
        batch_idx = (
            batch_idx.T
        )  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data

        batch_x0_hidden = torch.tensor(
            self.x_est[self.batch_scenarios_idx, batch_start, :], dtype=torch.float32
        ).to(self.device)
        batch_u_pop = torch.tensor(
            self.u_fit[self.batch_scenarios_idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_u_ind = torch.tensor(
            self.u_fit_ind[self.batch_scenarios_idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_y = torch.tensor(
            self.y_fit[self.batch_scenarios_idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_x_original = torch.tensor(
            self.x_est[self.batch_scenarios_idx, batch_idx], dtype=torch.float32
        ).to(self.device)

        if count:
            self.update_batch_idx()

        return batch_x0_hidden, batch_u_pop, batch_u_ind, batch_y, batch_x_original

    def update_batch_idx(self):
        if True:
            if len(self.idx_scenarios_temp) < self.batch_size:
                batch_scenarios_idx1 = self.idx_scenarios_temp
                self.idx_scenarios_temp = self.idx_scenarios

                batch_scenarios_idx2 = np.random.choice(
                    self.idx_scenarios_temp,
                    self.batch_size - len(batch_scenarios_idx1),
                    replace=False,
                )
                self.batch_scenarios_idx = np.concatenate(
                    [batch_scenarios_idx1, batch_scenarios_idx2]
                )
                self.idx_scenarios_temp = np.array(
                    list(set(self.idx_scenarios_temp).difference(batch_scenarios_idx2))
                )

                self.epoch += 1

            else:
                self.batch_scenarios_idx = np.random.choice(
                    self.idx_scenarios_temp, self.batch_size, replace=False
                )
                self.idx_scenarios_temp = np.array(
                    list(
                        set(self.idx_scenarios_temp).difference(
                            self.batch_scenarios_idx
                        )
                    )
                )

    def filter_seq(self):

        dfScenario = pd.DataFrame(
            columns=["isCompleteCGM", "isValid"], index=np.arange(self.x_est.shape[0])
        )
        for scenario in range(self.x_est.shape[0]):
            dfScenario.loc[scenario, "isCompleteCGM"] = np.isnan(
                self.y_fit[scenario, :, 0]
            ).any()

        dfScenario["isValid"] = np.logical_not(dfScenario["isCompleteCGM"])

        return dfScenario[dfScenario.isValid].index.astype(int)


class SequenceSelection:
    def __init__(self, seq_len, device, data):

        self.seq_len = seq_len
        self.device = device

        idx_scenarios = self.get_sequences(data)  # Filter out sequences

        for scenario in idx_scenarios:
            cgm_target = self.cgm_real[scenario, 0, 0].item()
            self.x_est[scenario, 0, :] = getInitSSFromFile(cgm_target)

        # Define initial states
        self.idx_scenarios = idx_scenarios

    def get_sequences(self, data):
        x_est, u_fit, y_fit, u_fit_ind, cgm_real = data

        idx_list = []
        idx = 0
        while idx <= y_fit.shape[1] - self.seq_len:  # Befor < instead of <=
            array = y_fit[0, idx : idx + self.seq_len, 0]

            if ~np.isnan(array[0]):

                if len(np.where(np.isnan(array))[0]) == 0:
                    idx_list.append(idx)
                else:
                    bool = []
                    for jj in np.arange(6, self.seq_len, 6):
                        if np.sum(~np.isnan(array[1 : jj + 1])) / jj < 0.7:
                            bool.append(False)
                        else:
                            bool.append(True)
                    if np.sum(bool) == 10:
                        idx_list.append(idx)
                idx = idx + self.seq_len
            else:
                idx += 1

        batch_start = np.array(idx_list, dtype=np.int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(self.seq_len)

        self.x_est = x_est[0, batch_idx, :]
        self.u_fit = u_fit[0, batch_idx, :]
        self.y_fit = y_fit[0, batch_idx, :]
        self.u_fit_ind = u_fit_ind[0, batch_idx, :]
        self.cgm_real = cgm_real[0, batch_idx, :]

        return np.arange(len(idx_list))

    def get_all(self, group="All"):

        idx = self.idx_scenarios

        batch_start = np.zeros(idx.shape[0], dtype=np.int8)
        batch_idx = batch_start[:, np.newaxis] + np.arange(
            self.seq_len
        )  # batch samples indices
        batch_idx = (
            batch_idx.T
        )  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data
        batch_x0_hidden = torch.tensor(
            self.x_est[idx, batch_start, :], dtype=torch.float32
        ).to(self.device)
        batch_u_pop = torch.tensor(self.u_fit[idx, batch_idx], dtype=torch.float32).to(
            self.device
        )
        batch_u_ind = torch.tensor(
            self.u_fit_ind[idx, batch_idx], dtype=torch.float32
        ).to(self.device)
        batch_y = torch.tensor(self.y_fit[idx, batch_idx], dtype=torch.float32).to(
            self.device
        )
        batch_x_original = torch.tensor(
            self.x_est[idx, batch_idx], dtype=torch.float32
        ).to(self.device)

        return batch_x0_hidden, batch_u_pop, batch_u_ind, batch_y, batch_x_original


class DigitalTwin:
    def __init__(self, n_digitalTwin=0, custom_DT = None,device=torch.device("cpu"), ts=5):
        self.ts = ts
        self.device = device

        if custom_DT is None:
            self.n_digitalTwin = n_digitalTwin

            digitalTwin_list = [
                f.path
                for f in os.scandir(Path(__file__).parent / "models/IndividualModel/")
                if f.is_dir()
            ]
            digitalTwin_list.sort()
            self.digital_twin_folder = digitalTwin_list[self.n_digitalTwin]
        else:
            self.n_digitalTwin = 99
            self.digital_twin_folder  = custom_DT

        self.setup_simulator()

    def setup_simulator(self):
        # Population Model
        ss_pop_model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        ss_pop_model.to(self.device)
        ss_pop_model.load_state_dict(
            torch.load(
                Path(__file__).parent
                / "models/PopulationModel/population_model_05022024_epoch_15.pt"
            )
        )

        # Individual Model
        ss_individual_model = CGMIndividual(hidden_compartments=hidden_compartments)
        ss_individual_model.to(self.device)
        ss_individual_model.load_state_dict(
            torch.load(self.digital_twin_folder + "/individual_model.pt")
        )

        for name, param in ss_pop_model.named_parameters():
            param.requires_grad = False
        for name, param in ss_individual_model.named_parameters():
            param.requires_grad = False

        # Simulator
        self.nn_solution = ForwardEulerSimulator(
            ss_pop_model,
            ss_individual_model,
            str(Path(__file__).parent) + "/models/PopulationModel/",
            ts=self.ts,
        )

        self.scaler_featsRobust = load(
            open(self.digital_twin_folder + "/scaler_robust.pkl", "rb")
        )

    def prepare_data(self, df_scenario):
        dfInitStates = pd.read_csv(
            Path(__file__).parent / "models/initSteadyStates.csv"
        ).set_index("initCGM")

        df_scenario[states] = df_scenario[states].astype(float)


        df_scenario.loc[0, states] = dfInitStates.loc[
            df_scenario.loc[0, states[0]].astype("int64"), states
        ]

        sim_time = len(df_scenario)
        batch_start = np.array([0], dtype=np.int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(sim_time)

        x_est = np.array(df_scenario[states].values).astype(np.float32)
        u_pop = np.array(df_scenario[inputs].values).astype(np.float32)
        u_ind = np.array(df_scenario[input_ind].values).astype(np.float32)

        # Scale states and inputs from the population models
        x_est, u_pop = scaler_pop(
            x_est, u_pop, str(Path(__file__).parent) + "/models/PopulationModel/", False
        )

        # Scale new inputs
        u_ind[:, idx_robust] = self.scaler_featsRobust.transform(u_ind[:, idx_robust])

        u_pop = u_pop.reshape(-1, sim_time, len(inputs))[0, batch_idx, :]
        u_ind = u_ind.reshape(-1, sim_time, len(input_ind))[0, batch_idx, :]

        x0_est = torch.tensor(
            df_scenario.loc[0, states].astype(float).values.reshape(1, -1),
            dtype=torch.float32,
        ).to(self.device)

        u_pop = torch.tensor(u_pop[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )
        u_ind = torch.tensor(u_ind[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )

        return x0_est, u_pop[:, [0], :], u_ind[:, [0], :]

    def simulate(self, df_scenario_original):
        # Prepare data

        df_scenario = df_scenario_original.copy()
        df_scenario = df_scenario.reset_index()
        try:
            df_scenario[states]
        except KeyError:
            df_scenario[states[1:]] = 0

        df_scenario["cgm_Actual"] = df_scenario["output_cgm"]

        x0_est, u_pop, u_ind = self.prepare_data(df_scenario)

        with torch.no_grad():
            x_sim_pop = self.nn_solution(x0_est, u_pop, None, is_pers=False)
            df_scenario[states] = scaler_inverse(
                x_sim_pop[:, 0, :].to("cpu").detach().numpy(),
                str(Path(__file__).parent) + "/models/PopulationModel/",
            )

            x_sim_DT = self.nn_solution(x0_est, u_pop, u_ind, is_pers=True)

            df_scenario[[s + "_DT" for s in states]] = scaler_inverse(
                x_sim_DT[:, 0, :].to("cpu").detach().numpy(),
                str(Path(__file__).parent) + "/models/PopulationModel/",
            )

        df_scenario["cgm_NNPop"] = df_scenario["output_cgm"]
        df_scenario["cgm_NNDT"] = df_scenario["output_cgm_DT"]

        return df_scenario


def getInitSSFromFile(cgm_target):

    if cgm_target < 40:
        cgm_target = 40
    if cgm_target > 400:
        cgm_target = 400

    dfInitStates = pd.read_csv(
        Path(__file__).parent / "models/initSteadyStates.csv"
    ).set_index("initCGM")

    return dfInitStates.loc[int(cgm_target), states]
