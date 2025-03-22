import torch
import torch.nn as nn
import lib.utils as utils
from .gnn_models import maskLinear
from torch.nn.modules.rnn import LSTM, GRU
from lib.utils import get_device
import numpy as np
import torch.nn.functional as F
class maskDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim,thresholding='topk'):
        super().__init__()
        # decode data from latent space where we are solving an ODE back to the data space

        decoder = nn.Sequential(
           maskLinear(latent_dim, input_dim,thresholding=thresholding))

        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)



class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space

        decoder = nn.Sequential(
           nn.Linear(latent_dim, input_dim))

        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)


class Encoder(nn.Module):
    def __init__(self, output_dim, input_dim,hidden_dim,layer_num):
        super(Encoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space

        encoder = utils.create_net(input_dim,output_dim*2,layer_num,hidden_dim)

        utils.init_network_weights(encoder)
        self.encoder = encoder

    def forward(self, data):
        h = self.encoder(data)
        mu,std = self.split_mean_mu(h)
        return mu,std

    def split_mean_mu(self, h):
        last_dim = h.size()[-1] // 2
        mu,std = h[:, :last_dim], h[:, last_dim:]
        return mu,std


class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=100,
                 device=torch.device("cpu")):
        super(GRU_unit, self).__init__()

        if update_gate is None:
            self.update_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            utils.init_network_weights(self.update_gate)
        else:
            self.update_gate = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            utils.init_network_weights(self.reset_gate)
        else:
            self.reset_gate = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim * 2))
            utils.init_network_weights(self.new_state_net)
        else:
            self.new_state_net = new_state_net

    def forward(self, y_mean, y_std, x, masked_update=True):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        assert (not torch.isnan(new_y).any())

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            n_data_dims = x.size(-1) // 2
            mask = x[:, :, n_data_dims:]
            utils.check_mask(x[:, :, :n_data_dims], mask)

            mask = (torch.sum(mask, -1, keepdim=True) > 0).float()

            assert (not torch.isnan(mask).any())

            new_y = mask * new_y + (1 - mask) * y_mean
            new_y_std = mask * new_y_std + (1 - mask) * y_std

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                #print(prev_new_y)
                exit()

        new_y_std = new_y_std.abs()
        return new_y, new_y_std


class Encoder_z0_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, lstm_output_size=20,
                 use_delta_t=True, device=torch.device("cpu")):

        super(Encoder_z0_RNN, self).__init__()

        self.gru_rnn_output_size = lstm_output_size
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.use_delta_t = use_delta_t

        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.gru_rnn_output_size, 50),
            nn.Tanh(),
            nn.Linear(50, latent_dim * 2), )

        utils.init_network_weights(self.hiddens_to_z0)

        input_dim = self.input_dim

        if use_delta_t:
            self.input_dim += 1
        self.gru_rnn = GRU(self.input_dim, self.gru_rnn_output_size).to(device)

    def forward(self, data, time_steps, run_backwards=True):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        # data shape: [n_traj, n_tp, n_dims]
        # shape required for rnn: (seq_len, batch, input_size)
        # t0: not used here
        n_traj = data.size(0)

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        data = data.permute(1, 0, 2)

        if run_backwards:
            # Look at data in the reverse order: from later points to the first
            data = utils.reverse(data)

        if self.use_delta_t:
            delta_t = time_steps[1:] - time_steps[:-1]
            if run_backwards:
                # we are going backwards in time with
                delta_t = utils.reverse(delta_t)
            # append zero delta t in the end
            delta_t = torch.cat((delta_t, torch.zeros(1).to(self.device)))
            delta_t = delta_t.unsqueeze(1).repeat((1, n_traj)).unsqueeze(-1)
            data = torch.cat((delta_t, data), -1)

        outputs, _ = self.gru_rnn(data)

        # LSTM output shape: (seq_len, batch, num_directions * hidden_size)
        last_output = outputs[-1]

        self.extra_info = {"rnn_outputs": outputs, "time_points": time_steps}

        mean, std = utils.split_last_dim(self.hiddens_to_z0(last_output))
        std = std.abs()

        assert (not torch.isnan(mean).any())
        assert (not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0)


class Encoder_z0_ODE_RNN(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver=None,
                 z0_dim=None, GRU_update=None,
                 n_gru_units=100,
                 device=torch.device("cpu")):

        super(Encoder_z0_ODE_RNN, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim,
                                       n_units=n_gru_units,
                                       device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim * 2, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2), )
        utils.init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards=True, save_info=False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        print('size is',data.size(0))
        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:, 0, :].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:

            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards=run_backwards,
                save_info=save_info)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = utils.split_last_dim(self.transform_z0(torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps,
                   run_backwards=True, save_info=False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        device = get_device(data)

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

        prev_t, t_i = time_steps[-1] + 0.01, time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        # print("minimum step: {}".format(minimum_step))

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if (prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert (not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

                assert (not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :] - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_y))
                exit()
            # assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:, i, :].unsqueeze(0)

            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)

            prev_y, prev_std = yi, yi_std
            prev_t, t_i = time_steps[i], time_steps[i - 1]

            latent_ys.append(yi)

            if save_info:
                d = {"yi_ode": yi_ode.detach(),  # "yi_from_data": yi_from_data,
                     "yi": yi.detach(), "yi_std": yi_std.detach(),
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

        latent_ys = torch.stack(latent_ys, 1)

        assert (not torch.isnan(yi).any())
        assert (not torch.isnan(yi_std).any())

        return yi, yi_std, latent_ys, extra_info


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space

        decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim), )

        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)

class Decoder_cdgoe(nn.Module):
    def __init__(self, latent_dim, output_dim,decoder_network = None):
        super(Decoder_cdgoe, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space
        if decoder_network == None:
            decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim//2),
                nn.ReLU(),
                nn.Linear(latent_dim//2,output_dim),
            )
            utils.init_network_weights(decoder)
        else:
            decoder = decoder_network

        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)

class Edge_NRI(nn.Module):

    def __init__(self, in_channels, w_node2edge, num_atoms,device,dropout=0.):
        super(Edge_NRI, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.w_node2edge = w_node2edge  #[2*in_channel, in_channel]

        self.w_edge2value = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, 1))   # No negative weight!
        self.edge_self_evolve = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, in_channels))

        self.num_atoms = num_atoms
        self.device = device
        self.layer_norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        utils.init_network_weights(self.w_edge2value)
        utils.init_network_weights(self.edge_self_evolve)

        self.rel_send,self.rel_rec = self.rel_rec_compute()


    def rel_rec_compute(self):
        fully_connected = np.ones([self.num_atoms, self.num_atoms])
        rel_send = np.array(utils.encode_onehot(np.where(fully_connected)[0]),
                            dtype=np.float32)  # every node as one-hot[10000], (N*N,N)
        rel_rec = np.array(utils.encode_onehot(np.where(fully_connected)[1]),
                           dtype=np.float32)  # every node as one-hot[10000], (N*N,N)
        rel_send = torch.FloatTensor(rel_send).to(self.device)
        rel_rec = torch.FloatTensor(rel_rec).to(self.device)

        return rel_send,rel_rec


    def forward(self, node_inputs, edges_input, num_atoms):
        # NOTE: Assumes that we have the same graph across all samples.
        '''

        :param node_inputs: [K*N,D]
        :param edges: [K*N*N,D], after normalize
        :return:
        '''
        node_feature_num = node_inputs.shape[1]
        edge_feature_num = edges_input.shape[-1]

        node_inputs = node_inputs.view(-1, num_atoms, node_feature_num)  # [K,N,D]

        senders = torch.matmul(self.rel_send, node_inputs)  # [K,N*N,D]
        receivers = torch.matmul(self.rel_rec, node_inputs)  # [K,N*N,D]
        edges = torch.cat([senders, receivers], dim=-1)  # [K,N*N,2D]

        # Compute z for edges
        edges_from_node = F.gelu(self.w_node2edge(edges))
        edges_input = self.layer_norm(edges_input)
        edges_self = self.edge_self_evolve(edges_input) #[K*N*N,D]
        edges_self = edges_self.view(-1,num_atoms*num_atoms,edge_feature_num) #[K,N*N,D]
        edges_z = self.dropout(edges_from_node + edges_self) #[K,N*N,D]

        # edge2value
        edge_2_value = torch.squeeze(F.relu(self.w_edge2value(edges_z)),dim=-1) #[K,N*N]
        edges_z = edges_z.view(-1,node_feature_num) #[K*N*N,D]

        return edges_z,  edge_2_value

