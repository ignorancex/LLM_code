import os
import time
import scipy.io
import numpy as np
import torch

from tensorboardX import SummaryWriter

from modules.networks import DNN_U
from utils.base import parse_args, set_seed_everywhere
from problems.node_dyn import Dynamics_Satellite, NNCDynamics

torch.pi = torch.acos(torch.zeros(1)).item() * 2

class Solver():
    def __init__(self, layers, args):
        # Problem
        self.x_dim = args.x_dim
        self.batch_size = args.batch_size
        self.w1 = 0.5
        self.w2 = 5
        self.w3 = 0.25
        self.w4 = 0.5
        self.device = args.device
        self.num_train_iter = args.num_iters

        # Init range
        self.X0_ub = torch.pi / \
            torch.tensor([[3.], [3.], [3.], [4.], [4.], [4.]]
                         ).T.to(self.device)
        self.X0_lb = -self.X0_ub

        # NOTE: the scale of space and time.
        self.T = args.T
        self.X0_scale = args.X0_scale
        self.X0_ub *= self.X0_scale
        self.X0_lb *= self.X0_scale

        self.load_valid_data()

        self.load_model = args.load_model
        print(self.device)
        self.dyn = Dynamics_Satellite(
            self.w1, self.w2, self.w3, device=self.device)
        self.dnn = DNN_U(layers, device=self.device).to(self.device)

        dir_list = ["", "model", "tensorboard"]
        for dir in dir_list:
            file_path = f"output/{args.X0_scale}_{args.T}_satellite_fix/{dir}"
            if not os.path.exists(file_path):
                os.mkdir(file_path)

        # DNN
        if self.load_model:
            self.pre_epochs = args.pre_epochs
            load_model_path = f"output/{args.X0_scale}_{args.T}_satellite_fix/model/supervised_{self.dnn.__class__.__name__}_{self.pre_epochs}.pkl"
            self.dnn.load_state_dict(torch.load(load_model_path))
            print(
                f"Loading Supervised model from {load_model_path}.\nProblem {args.X0_scale}_{args.T}, Pretrained {self.pre_epochs}.")
            scaling_path = f"data/{args.X0_scale}_{args.T}_satellite_fix/data_fix_scaling.mat"
            scaling = scipy.io.loadmat(scaling_path)
            scaling["T"] = self.T

            self.loss_dyn = NNCDynamics(
                self.dyn, self.dnn, scaling=scaling, device=self.device)
        else:
            print(f"Training from Scratch.\nProblem {args.X0_scale}_{args.T}.")
            self.loss_dyn = NNCDynamics(self.dyn, self.dnn, device=self.device)

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=args.lr)

        if self.load_model:
            self.writer = SummaryWriter(
                f'output/{args.X0_scale}_{args.T}_satellite_fix/tensorboard/aca_time_finetune{self.X0_scale}_{self.T}_{self.pre_epochs}')
        else:
            self.writer = SummaryWriter(
                f'output/{args.X0_scale}_{args.T}_satellite_fix/tensorboard/aca_nonorm_time_direct_{self.X0_scale}_{self.T}')

    def loss_pullback(self, x0):
        def extract_loss(x, w1=0.5):
            interval_loss = x[:, 6]
            terminal_loss = (x[:, 0:6]**2).sum(dim=1)
            ave_loss = (interval_loss + w1 * terminal_loss).mean()
            return ave_loss

        options = {}
        options.update({'t0': 0.})
        options.update({'t1': self.T})
        options.update({'print_neval': True})
        options.update({'method': "Dopri5"})
        options.update({'rtol': 1e-5})
        options.update({'atol': 1e-5})

        fwd_sol = odeint(self.loss_dyn, x0, options)

        ave_loss = extract_loss(fwd_sol)
        return ave_loss

    def load_valid_data(self):
        valid_data = scipy.io.loadmat(
            f"data/{args.X0_scale}_{args.T}_satellite_fix/data_fix_valid.mat")
        idx0 = np.nonzero(np.equal(valid_data.pop('t'), 0.))[1]
        valid_data_x = torch.tensor(
            valid_data['X'][:, idx0]).float().T.to(self.device)
        valid_data_c = torch.zeros((valid_data_x.shape[0], 1)).to(self.device)
        self.valid_data_init = torch.cat([valid_data_x, valid_data_c], dim=1)
        self.valid_data_cost = valid_data['V'][:, idx0]
        print(self.valid_data_init[0, :], self.valid_data_cost[0, 0])

    def gen_batch_data(self):
        cost_init = torch.zeros([self.batch_size, 1], device=self.device)
        x_init = torch.rand([self.batch_size, 6], device=self.device) * \
            (self.X0_ub - self.X0_lb) + self.X0_lb
        aug_x_init = torch.cat([x_init, cost_init], dim=1)
        return aug_x_init

    def train(self):
        for iter in range(self.num_train_iter):
            t1 = time.time()
            batch_x = self.gen_batch_data()
            self.optimizer.zero_grad()
            loss = self.loss_pullback(batch_x)
            loss.backward()
            self.optimizer.step()
            t2 = time.time()

            self.writer.add_scalar(
                'train batch ave loss', loss.item(), global_step=iter)
            print(iter, loss.item(), t2 - t1)

            if (iter+1) % 100 == 0:
                with torch.no_grad():
                    valid_loss = self.loss_pullback(self.valid_data_init)
                    self.writer.add_scalar(
                        'valid loss', valid_loss.item(), global_step=iter)
                    print(
                        f"Valid: {valid_loss.item()}, BVP: {self.valid_data_cost.mean()}")

                if (iter+1) % 100 == 0:
                    if self.load_model:
                        torch.save(self.dnn.state_dict(
                        ), f"output/{args.X0_scale}_{args.T}_satellite_fix/model/finetune_{self.dnn.__class__.__name__}_{self.pre_epochs}_{iter+1}.pkl")
                    else:
                        torch.save(self.dnn.state_dict(
                        ), f"output/{args.X0_scale}_{args.T}_satellite_fix/model/direct_{self.dnn.__class__.__name__}_{iter+1}.pkl")


def main(args):
    # Set seed
    set_seed_everywhere(args.seed)
    # Set NN architecture
    layers = [args.x_dim+1, args.hidden_size,
              args.hidden_size, args.hidden_size, args.u_dim]
    model = Solver(layers, args)

    model.train()


if __name__ == "__main__":
    args = parse_args()
    if args.adjoint:
        from torch_ACA import odesolve_adjoint as odeint
    else:
        from torch_ACA import odesolve as odeint
    main(args)
