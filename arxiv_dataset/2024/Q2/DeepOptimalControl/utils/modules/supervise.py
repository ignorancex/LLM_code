import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from problems.problem import ControlProblem
from modules.networks import DNN_U


class GenDataset(Dataset):
    def __init__(self, data, scaling, problem):
        # Load data
        t = data['t'].T
        X = data['X'].T
        V = data['V'].T
        A = data['A'].T
        U = data['U'].T
        lb = scaling['lb'].T
        ub = scaling['ub'].T
        V_lb = scaling['V_min'].T
        V_ub = scaling['V_max'].T
        A_lb = scaling['A_lb'].T
        A_ub = scaling['A_ub'].T
        U_lb = scaling['U_lb'].T
        U_ub = scaling['U_ub'].T
        # Normalize
        t_scaled = 2. * t / problem.tT - 1.
        X_scaled = 2. * (X - lb) / (ub - lb) - 1.
        V_scaled = 2. * (V - V_lb) / (V_ub - V_lb) - 1.
        A_scaled = 2. * (A - A_lb) / (A_ub - A_lb) - 1.
        U_scaled = 2. * (U - U_lb) / (U_ub - U_lb) - 1.
        self.V_scaled = V_scaled.astype(np.float32)
        self.A_scaled = A_scaled.astype(np.float32)
        self.U_scaled = U_scaled.astype(np.float32)
        self.X_scaled = np.concatenate(
            [t_scaled, X_scaled], axis=1).astype(np.float32)

    def __len__(self):
        return self.X_scaled.shape[0]

    def __getitem__(self, idx):
        return self.X_scaled[idx], self.V_scaled[idx], self.A_scaled[idx], self.U_scaled[idx]


class ModelControl():
    def __init__(self, problem: ControlProblem):
        # Problem
        self.problem = problem
        self.N_s = problem.N_states
        self.N_c = problem.N_controls
        # Model
        self.device = problem.device
        self.u_layers = [self.N_s+1] + [problem.v_units] * problem.v_layers + [self.N_c]
        self.model_u = DNN_U(self.u_layers, device=self.device).to(self.device)
        self.lr = problem.learning_rate
        self.batch_size = problem.batch_size

    def data_parse(self, data):
        # Load data
        t = torch.tensor(data['t'].T.astype(np.float32)).to(self.device)
        X = torch.tensor(data['X'].T.astype(np.float32)).to(self.device)
        V = torch.tensor(data['V'].T.astype(np.float32)).to(self.device)
        A = torch.tensor(data['A'].T.astype(np.float32)).to(self.device)
        U = torch.tensor(data['U'].T.astype(np.float32)).to(self.device)
        # Normalize
        t_scaled = 2. * t / self.problem.tT - 1.
        X_scaled = 2. * (X - self.lb) / (self.ub - self.lb) - 1.
        V_scaled = 2. * (V - self.V_min) / (self.V_max - self.V_min) - 1.
        A_scaled = 2. * (A - self.A_lb) / (self.A_ub - self.A_lb) - 1.
        U_scaled = 2. * (U - self.U_lb) / (self.U_ub - self.U_lb) - 1.

        X_scaled = torch.cat([t_scaled, X_scaled], dim=1)
        return X_scaled, V_scaled, A_scaled, U_scaled

    def Scaling(self, scaling):
        self.scaling = scaling
        self.lb = torch.tensor(
            scaling['lb'].T.astype(np.float32)).to(self.device)
        self.ub = torch.tensor(
            scaling['ub'].T.astype(np.float32)).to(self.device)
        self.A_lb = torch.tensor(
            scaling['A_lb'].T.astype(np.float32)).to(self.device)
        self.A_ub = torch.tensor(
            scaling['A_ub'].T.astype(np.float32)).to(self.device)
        self.U_lb = torch.tensor(
            scaling['U_lb'].T.astype(np.float32)).to(self.device)
        self.U_ub = torch.tensor(
            scaling['U_ub'].T.astype(np.float32)).to(self.device)
        self.V_min = torch.tensor(
            scaling['V_min'].T.astype(np.float32)).to(self.device)
        self.V_max = torch.tensor(
            scaling['V_max'].T.astype(np.float32)).to(self.device)

    def eval_loss(self, X, U_scaled):
        U = self.problem.make_U_NN(self.model_u, X[:, 0:1], X[:, 1:])
        U_pred_scaled = 2.0 * (U - self.U_lb) / (self.U_ub - self.U_lb) - 1.0
        loss_U = ((U_pred_scaled - U_scaled)**2).mean()
        return loss_U

    def train_model(self, data_train, data_val, epochs, experiment_name):
        save_model_path = f"./output/{experiment_name}/model/"
        if self.problem.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model_u.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adagrad(
                self.model_u.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=500, gamma=0.5)  # NOTE: Change the lr scheduler here.

        # Load training data
        train_loader = DataLoader(GenDataset(data_train, self.scaling, self.problem),
                                  batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        # Load validation data
        X_scaled_valid, _, _, U_scaled_valid = self.data_parse(data_val)

        # Training
        for epoch in tqdm(range(epochs)):
            loss_list = []
            for batch_idx, (X_scaled, _, _, U_scaled) in enumerate(train_loader):
                X_scaled = X_scaled.to(self.device)
                U_scaled = U_scaled.to(self.device)

                loss = self.eval_loss(X_scaled, U_scaled)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.detach().cpu().numpy())

            scheduler.step()
            ave_loss = np.mean(loss_list)

            if epoch % 10 == 0:
                print('***** epoch: %d ********' % epoch)
                self.loss_U_valid = self.eval_loss(
                    X_scaled_valid, U_scaled_valid)
                print(f"Train | Loss_U: {ave_loss}")
                print(
                    f"Valid | Loss_U: {self.loss_U_valid.detach().cpu().numpy()}")

            if (epoch+1) % 100 == 0:
                print("Saving model at", save_model_path)
                self.save_model(save_model_path, epoch)
        return 

    def save_model(self, save_path_tf, epoch):
        torch.save(self.model_u.state_dict(), save_path_tf +
                   f"supervised_{self.model_u.__class__.__name__}_{epoch+1}.pkl")

    def load_model(self, load_path_tf):
        self.model_u.load_state_dict(torch.load(load_path_tf))

    def eval_U(self, t, X):
        X = Variable(torch.tensor(X).float().T)
        t = Variable(torch.tensor(t).float().T)
        t_scaled = 2. * t / self.problem.tT - 1.
        X_scaled = 2. * (X - self.lb) / (self.ub - self.lb) - 1.
        U = self.problem.make_U_NN(self.model_u, t_scaled, X_scaled)
        return U.T.detach().cpu().numpy()
