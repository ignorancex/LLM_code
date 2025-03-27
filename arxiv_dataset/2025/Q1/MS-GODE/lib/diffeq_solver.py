import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import lib.utils as utils
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch_scatter import scatter_add
class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method,args,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()
        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.args = args
        self.num_atoms = args.n_balls
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        # graph related
        self.rel_rec, self.rel_send = self.compute_rec_send()

    def compute_rec_send(self):
        off_diag = np.ones([self.num_atoms, self.num_atoms]) - np.eye(self.num_atoms)
        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0]),
                           dtype=np.float32)  # receiver nodes. every node as one-hot[10000], (20,5)
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # every node as one-hot,(20,5)
        rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        rel_send = torch.FloatTensor(rel_send).to(self.device)
        return rel_rec, rel_send

    def forward(self, first_point, time_steps_to_predict, graph,backwards = False):
        '''
        :param first_point: 【n_sample,b*n_ball,d]
        :param time_steps_to_predict: [t]
        :param graph: [2, num_edge]
        :param backwards:
        :return:
        '''
        # code below is to reset the num_balls when new tasks come
        self.num_atoms = int(first_point.shape[1]/graph.shape[0])
        self.rel_rec, self.rel_send = self.compute_rec_send()
        # whether to padding 0 to the time series
        ispadding = False
        if time_steps_to_predict[0] != 0:
            ispadding = True
            time_steps_to_predict = torch.cat((torch.zeros(1,device=time_steps_to_predict.device),time_steps_to_predict))

        n_traj_samples, n_traj,feature = first_point.size()[0], first_point.size()[1],first_point.size()[2]
        first_point_augumented = first_point.view(-1,self.num_atoms,feature) #[n_sample*b, n_ball,d]
        if self.args.augment_dim > 0:
            aug = torch.zeros(first_point_augumented.shape[0],first_point_augumented.shape[1], self.args.augment_dim).to(self.device)
            first_point_augumented = torch.cat([first_point_augumented, aug], 2)
            feature += self.args.augment_dim

        # duplicate graph w.r.t num_sample_traj
        graph_augmented = torch.cat([graph for _ in range(n_traj_samples)], dim=0)

        rel_type_onehot = torch.FloatTensor(first_point_augumented.size(0), self.rel_rec.size(0),
                                                self.args.edge_types).to(self.device)  # [b,20,2]
        rel_type_onehot.zero_()
        rel_type_onehot.scatter_(2, graph_augmented.view(first_point_augumented.size(0), -1, 1), 1)  # [b,20,2]
        # rel_type_onehot[b,20,1]: edge value, [b,20,0] :1-edge value.

        self.ode_func.set_graph(rel_type_onehot,self.rel_rec,self.rel_send,self.args.edge_types)

        pred_y = odeint(self.ode_func, first_point_augumented, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method) #[time_length, n_sample*b,n_ball, d]

        '''
        pred_y = self.ode_func(time_steps_to_predict, first_point_augumented)
        pred_y = pred_y.repeat(time_steps_to_predict.shape[0], 1, 1,1)
        '''

        if ispadding:
            pred_y = pred_y[1:,:,:,:]
            time_steps_to_predict = time_steps_to_predict[1:]

        pred_y = pred_y.view(time_steps_to_predict.size(0), -1, pred_y.size(3)) #[t,n_sample*b*n_ball, d]

        pred_y = pred_y.permute(1,0,2) #[n_sample*b*n_ball, time_length, d]
        pred_y = pred_y.view(n_traj_samples,n_traj,-1,feature) #[n_sample, b*n_ball, time_length, d]

        #assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        if self.args.augment_dim > 0:
            pred_y = pred_y[:, :, :, :-self.args.augment_dim]

        return pred_y

    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot


class GraphODEFunc(nn.Module):
    def __init__(self, ode_func_net,  device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(GraphODEFunc, self).__init__()

        self.device = device
        self.ode_func_net = ode_func_net  #input: x, edge_index
        self.nfe = 0


    def forward(self, t_local, z, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        self.nfe += 1
        #print(self.nfe)
        grad = self.ode_func_net(z)

        if backwards:
            grad = -grad
        return grad

    def set_graph(self, rec_type,rel_rec,rel_send,edge_types):
        #print(self.nfe)
        for layer in self.ode_func_net.gcs:
            layer.base_conv.rel_type = rec_type
            layer.base_conv.rel_rec = rel_rec # one-hot representation of receiver nodes
            layer.base_conv.rel_send = rel_send
            #layer.base_conv.edge_types = 100000# edge_types
        self.nfe = 0


class ODEFunc(nn.Module):
	def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc, self).__init__()

		self.input_dim = input_dim
		self.device = device

		utils.init_network_weights(ode_func_net)
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)




class DiffeqSolver_lode(nn.Module):
	def __init__(self, input_dim, ode_func, method, latents,
			odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
		super(DiffeqSolver_lode, self).__init__()

		self.ode_method = method
		self.latents = latents
		self.device = device
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict, backwards = False):
		"""
		# Decode the trajectory through ODE Solver
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
		n_dims = first_point.size()[-1]

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict,
		n_traj_samples = 1):
		"""
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		# shape: [n_traj_samples, n_traj, n_tp, n_dim]
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y


def compute_edge_initials(first_point_enc, num_atoms,w_node_to_edge_initial):
    '''

    :param first_point_enc: [K*N,D]
    :return: [K*N*N,D']
    '''
    node_feature_num = first_point_enc.shape[1]
    fully_connected = np.ones([num_atoms, num_atoms])
    rel_send = np.array(utils.encode_onehot(np.where(fully_connected)[0]),
                        dtype=np.float32)  # every node as one-hot[10000], (N*N,N)
    rel_rec = np.array(utils.encode_onehot(np.where(fully_connected)[1]),
                       dtype=np.float32)  # every node as one-hot[10000], (N*N,N)

    rel_send = torch.FloatTensor(rel_send).to(first_point_enc.device)
    rel_rec = torch.FloatTensor(rel_rec).to(first_point_enc.device)

    first_point_enc = first_point_enc.view(-1, num_atoms, node_feature_num)  # [K,N,D]

    senders = torch.matmul(rel_send, first_point_enc)  # [K,N*N,D]
    receivers = torch.matmul(rel_rec, first_point_enc)  # [K,N*N,D]

    edge_initials = torch.cat([senders, receivers], dim=-1)  # [K,N*N,2D]
    edge_initials = F.gelu(w_node_to_edge_initial(edge_initials))  # [K,N*N,D_edge]
    edge_initials = edge_initials.view(-1, edge_initials.shape[2])  # [K*N*N,D_edge]

    return edge_initials
class DiffeqSolver_cgode(nn.Module):
    def __init__(self, ode_func, method,args,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = torch.device("cpu")):
        super(DiffeqSolver_cgode, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.args = args
        self.num_atoms = args.n_balls

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol



    def forward(self, first_point,time_steps_to_predict,w_node_to_edge_initial):
        '''

        :param first_point:  [K*N,D]
        :param edge_initials: [K*N*N,D]
        :param time_steps_to_predict: [t]
        :return:
        '''

        # Node ODE Function
        n_traj,feature_node = first_point.size()[0], first_point.size()[1]  #[K*N,d]
        if self.args.augment_dim > 0:
            aug_node = torch.zeros(first_point.shape[0], self.args.augment_dim).to(self.device) #[K*N,D_aug]
            first_point = torch.cat([first_point, aug_node], 1) #[K*N,d+D_aug]
            feature_node += self.args.augment_dim

        # Edge initialization: h_ij = f([u_i,u_j])
        edge_initials = compute_edge_initials(first_point, self.num_atoms, w_node_to_edge_initial)  # [K*N*N,D_edge]
        assert (not torch.isnan(edge_initials).any())

        node_edge_initial = torch.cat([first_point,edge_initials],0)  #[K*N + K*N*N,D+D_aug]
        # Set index
        K_N = int(node_edge_initial.shape[0]/(self.num_atoms+1))
        K = K_N/self.num_atoms
        self.ode_func.set_index_and_graph(K_N,K)

        node_initial = node_edge_initial[:K_N,:]
        self.ode_func.set_initial_z0(node_initial)


        # Results
        pred_y = odeint(self.ode_func, node_edge_initial, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method) #[time_length, K*N + K*N*N, D]

        pred_y = pred_y.permute(1,0,2) #[ K*N + K*N*N, time_length, d]

        assert(pred_y.size()[0] == K_N*(self.num_atoms+1))

        if self.args.augment_dim > 0:
            pred_y = pred_y[:, :, :-self.args.augment_dim]

        return pred_y,K_N



class CoupledODEFunc(nn.Module):
    def __init__(self, node_ode_func_net,edge_ode_func_net,num_atom, dropout,device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(CoupledODEFunc, self).__init__()

        self.device = device
        self.node_ode_func_net = node_ode_func_net  #input: x, edge_index
        self.edge_ode_func_net = edge_ode_func_net
        self.num_atom = num_atom
        self.nfe = 0
        self.dropout = nn.Dropout(dropout)


    def forward(self, t_local, z, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        z:  [H,E] concat by axis0. H is [K*N,D], E is[K*N*N,D], z is [K*N + K*N*N, D]
        """
        self.nfe += 1


        node_attributes = z[:self.K_N,:]
        edge_attributes = z[self.K_N:,:]
        assert (not torch.isnan(node_attributes).any())
        assert (not torch.isnan(edge_attributes).any())

        #grad_edge, edge_value = self.edge_ode_func_net(node_attributes,self.num_atom) # [K*N*N,D],[K,N*N], edge value are non-negative by using relu.
        grad_edge, edge_value = self.edge_ode_func_net(node_attributes,edge_attributes,self.num_atom)  # [K*N*N,D],[K,N*N], edge value are non-negative by using relu.todo:with self-evolution
        edge_value = self.normalize_graph(edge_value,self.K_N)
        assert (not torch.isnan(edge_value).any())
        grad_node = self.node_ode_func_net(node_attributes,edge_value,self.node_z0) # [K*N,D]
        assert (not torch.isnan(grad_node).any())
        assert (not torch.isinf(grad_edge).any())

        assert (not torch.isnan(grad_node).any())
        assert (not torch.isinf(grad_edge).any())

        # Concat two grad
        grad = self.dropout(torch.cat([grad_node,grad_edge],0)) # [K*N + K*N*N, D]


        return grad


    def set_index_and_graph(self,K_N,K):
        '''

        :param K_N: index for separating node and edge matrixs.
        :return:
        '''
        self.K_N = K_N
        self.K = int(K)
        self.K_N_N = self.K_N*self.num_atom
        self.nfe = 0

        # Step1: Concat into big graph: which is set by default.diagonal matrix
        edge_each = np.ones((self.num_atom, self.num_atom))
        edge_whole= block_diag(*([edge_each] * self.K))
        edge_index,_ = utils.convert_sparse(edge_whole)
        self.edge_index = torch.LongTensor(edge_index).to(self.device)

    def set_initial_z0(self,node_z0):
        self.node_z0 = node_z0

    def normalize_graph(self,edge_weight,num_nodes):
      '''
      For asymmetric graph
      :param edge_index: [num_edges,2], torch.LongTensor
      :param edge_weight: [K,N*N] ->[num_edges]
      :param num_nodes:
      :return:
      '''
      assert (not torch.isnan(edge_weight).any())
      assert (torch.sum(edge_weight<0)==0)
      edge_weight_flatten = edge_weight.view(-1)  #[K*N*N]

      row, col = self.edge_index[0], self.edge_index[1]
      deg = scatter_add(edge_weight_flatten, row, dim=0, dim_size=num_nodes) #[K*N]
      deg_inv_sqrt = deg.pow_(-1)
      deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
      if torch.isnan(deg_inv_sqrt).any():
          assert (torch.sum(deg == 0) == 0)
          assert (torch.sum(deg < 0) == 0)

      #assert (not torch.isnan(deg_inv_sqrt).any())


      edge_weight_normalized = deg_inv_sqrt[row] * edge_weight_flatten   #[k*N*N]
      assert (torch.sum(edge_weight_normalized < 0) == 0) and (torch.sum(edge_weight_normalized > 1) == 0)

      # Reshape back

      edge_weight_normalized = torch.reshape(edge_weight_normalized,(self.K,-1)) #[K,N*N]
      assert (not torch.isnan(edge_weight_normalized).any())

      return edge_weight_normalized



