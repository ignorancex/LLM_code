from lib.base_models import VAE_Baseline,VAE_Baseline_lode,VAE_Baseline_cgode
from lib.encoder_decoder import *
import lib.utils as utils
import torch
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn as nn

class LatentGraphODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
				 z0_prior, device, obsrv_std=None):

		super(LatentGraphODE, self).__init__(
			input_dim=input_dim, latent_dim=latent_dim,
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.latent_dim =latent_dim

	def get_reconstruction(self, batch_en,batch_de, batch_g,n_traj_samples=1, run_backwards=True):


        #Encoder:
		first_point_mu, first_point_std = self.encoder_z0(batch_en.x, batch_en.edge_attr,
														  batch_en.edge_index, batch_en.pos, batch_en.edge_same,
														  batch_en.batch, batch_en.y)  # [num_ball,10]

		means_z0 = first_point_mu.repeat(n_traj_samples,1,1) #[n_traj_samples, batchsize * num_ball,10]
		sigmas_z0 = first_point_std.repeat(n_traj_samples,1,1) #[3,num_ball,10]
		first_point_enc = utils.sample_standard_gaussian(means_z0, sigmas_z0) #[3,num_ball,10]
		first_point_std = first_point_std.abs().clamp(1e-3) #self added clamp to avoid 0 in output

		time_steps_to_predict = batch_de["time_steps"]

		assert (torch.sum(first_point_std < 0) == 0.)
		assert (not torch.isnan(time_steps_to_predict).any())

		if torch.isnan(first_point_enc).any():
			print('here')

		assert (not torch.isnan(first_point_enc).any())

		# ODE:Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict, batch_g)

        # Decoder:
		pred_x = self.decoder(sol_y)

		all_extra_info = {
			"first_point": (torch.unsqueeze(first_point_mu,0), torch.unsqueeze(first_point_std,0), first_point_enc),
			"latent_traj": sol_y.detach()
		}

		if torch.isnan(pred_x).any():
			print('hhhherehhhhhh')

		return pred_x, all_extra_info, None, first_point_mu, first_point_std


	def get_mask(self, batch_dict_encoder,batch_dict_decoder,batch_dict_graph ,n_traj_samples = 1, kl_coef = 1.):
		# Condition on subsampled points
		# Make predictions for all the points
		pred_y, info,temporal_weights= self.get_reconstruction(batch_dict_encoder,batch_dict_decoder,batch_dict_graph,n_traj_samples = n_traj_samples)
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		#print("get_reconstruction done -- computing likelihood")
		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs()
		fp_distr = Normal(fp_mu, fp_std)

		assert(torch.sum(fp_std < 0) == 0.)
		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

		if torch.isnan(kldiv_z0).any():
			print(fp_mu)
			print(fp_std)
			raise Exception("kldiv_z0 is Nan!")

		# Mean over number of latent dimensions
		# kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
		# kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
		# shape after: [n_traj_samples]
		kldiv_z0 = torch.mean(kldiv_z0,(1,2))

		# Compute likelihood of all the points
		rec_likelihood = self.get_gaussian_likelihood(
			batch_dict_decoder["data"], pred_y,temporal_weights,
			mask=batch_dict_decoder["mask"])   #negative value

		# set alphas
		self.zero_grad()
		self.apply(lambda m: setattr(m, "task", -1))

		alphas = (
				torch.ones(
					[self.num_tasks, 1, 1], device=self.get_device(), requires_grad=True
				)
				/ self.num_tasks
		)

		#self.apply(lambda m: setattr(m, "num_tasks_learned", max(model.task_total, maxn)))
		self.apply(lambda m: setattr(m, "alphas", alphas))

		# loss

		loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)

		grad = torch.autograd.grad(loss, alphas)[0]
		'''
		grad = -grad.flatten()[:model.num_tasks_learned]

		# p = (grad * 2 * model.num_tasks_learned).softmax(dim=0)
		# p = (10 * np.log(model.num_tasks_learned) * grad).softmax(dim=0)
		p = grad.softmax(dim=0).max()

		# grad_entropy = #p - 1./model.num_tasks_learned #-( p * p.log() ).sum()
		grad_entropy = p
		_, ind = grad.max(dim=0)
		'''
		inferred_task = (-grad).squeeze().argmax()
		# value = 1.2/model.num_tasks_learned
		# seems like optimal value here is somewhere between 1.1 and 1.2
		#value = 1.125 / model.num_tasks_learned

		inferred_task = inferred_task.item()

		'''
		if grad_entropy < value:
			model.task_total += 1
			ind = model.task_total - 1
			print('NEW TASK', ind)
		'''

		return inferred_task


class LatentODE(VAE_Baseline_lode):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
				 z0_prior, device, obsrv_std=None,
				 use_binary_classif=False, use_poisson_proc=False,
				 linear_classifier=False,
				 classif_per_tp=False,
				 n_labels=1,
				 train_classif_w_reconstr=False):

		super(LatentODE, self).__init__(
			input_dim=input_dim, latent_dim=latent_dim,
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std,
			use_binary_classif=use_binary_classif,
			classif_per_tp=classif_per_tp,
			linear_classifier=linear_classifier,
			use_poisson_proc=use_poisson_proc,
			n_labels=n_labels,
			train_classif_w_reconstr=train_classif_w_reconstr)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.use_poisson_proc = use_poisson_proc

	def get_reconstruction(self, batch_en, batch_de, batch_g, n_traj_samples=1, run_backwards=True):
		print('batch is', batch_en)
		print('batch x is', batch_en.x)
		print('batch x shape is', batch_en.x.shape)
		print('batch x[0] shape is', batch_en.x[0].shape)
		print('batch x[0][0] shape is', batch_en.x[0][0].shape)
		# Encoder:
		first_point_mu, first_point_std = self.encoder_z0(batch_en.x, batch_en.pos)  # [num_ball,10]

		means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)  # [n_traj_samples, batchsize * num_ball,10]
		sigmas_z0 = first_point_std.repeat(n_traj_samples, 1, 1)  # [3,num_ball,10]
		first_point_enc = utils.sample_standard_gaussian(means_z0, sigmas_z0)  # [3,num_ball,10]
		first_point_std = first_point_std.abs().clamp(1e-3)  # self added clamp to avoid 0 in output

		time_steps_to_predict = batch_de["time_steps"]

		assert (torch.sum(first_point_std < 0) == 0.)
		assert (not torch.isnan(time_steps_to_predict).any())

		if torch.isnan(first_point_enc).any():
			print('here')

		assert (not torch.isnan(first_point_enc).any())

		# ODE:Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)

		# Decoder:
		pred_x = self.decoder(sol_y)

		all_extra_info = {
			"first_point": (
				torch.unsqueeze(first_point_mu, 0), torch.unsqueeze(first_point_std, 0), first_point_enc),
			"latent_traj": sol_y.detach()
		}

		if torch.isnan(pred_x).any():
			print('hhhherehhhhhh')

		return pred_x, all_extra_info, None, first_point_mu, first_point_std

	def get_mask(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, n_traj_samples=1, kl_coef=1.):
		# Condition on subsampled points
		# Make predictions for all the points
		pred_y, info, temporal_weights = self.get_reconstruction(batch_dict_encoder, batch_dict_decoder,
																 batch_dict_graph, n_traj_samples=n_traj_samples)
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# print("get_reconstruction done -- computing likelihood")
		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs()
		fp_distr = Normal(fp_mu, fp_std)

		assert (torch.sum(fp_std < 0) == 0.)
		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

		if torch.isnan(kldiv_z0).any():
			print(fp_mu)
			print(fp_std)
			raise Exception("kldiv_z0 is Nan!")

		# Mean over number of latent dimensions
		# kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
		# kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
		# shape after: [n_traj_samples]
		kldiv_z0 = torch.mean(kldiv_z0, (1, 2))

		# Compute likelihood of all the points
		rec_likelihood = self.get_gaussian_likelihood(
			batch_dict_decoder["data"], pred_y, temporal_weights,
			mask=batch_dict_decoder["mask"])  # negative value

		# set alphas
		self.zero_grad()
		self.apply(lambda m: setattr(m, "task", -1))

		alphas = (
				torch.ones(
					[self.num_tasks, 1, 1], device=self.get_device(), requires_grad=True
				)
				/ self.num_tasks
		)

		# self.apply(lambda m: setattr(m, "num_tasks_learned", max(model.task_total, maxn)))
		self.apply(lambda m: setattr(m, "alphas", alphas))

		# loss

		loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0, 0)

		grad = torch.autograd.grad(loss, alphas)[0]
		'''
        grad = -grad.flatten()[:model.num_tasks_learned]

        # p = (grad * 2 * model.num_tasks_learned).softmax(dim=0)
        # p = (10 * np.log(model.num_tasks_learned) * grad).softmax(dim=0)
        p = grad.softmax(dim=0).max()

        # grad_entropy = #p - 1./model.num_tasks_learned #-( p * p.log() ).sum()
        grad_entropy = p
        _, ind = grad.max(dim=0)
        '''
		inferred_task = (-grad).squeeze().argmax()
		# value = 1.2/model.num_tasks_learned
		# seems like optimal value here is somewhere between 1.1 and 1.2
		# value = 1.125 / model.num_tasks_learned

		inferred_task = inferred_task.item()

		'''
        if grad_entropy < value:
            model.task_total += 1
            ind = model.task_total - 1
            print('NEW TASK', ind)
        '''

		return inferred_task

	def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples=1):
		# input_dim = starting_point.size()[-1]
		# starting_point = starting_point.view(1,1,input_dim)

		# Sample z0 from prior
		starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

		starting_point_enc_aug = starting_point_enc
		if self.use_poisson_proc:
			n_traj_samples, n_traj, n_dims = starting_point_enc.size()
			# append a vector of zeros to compute the integral of lambda
			zeros = torch.zeros(n_traj_samples, n_traj, self.input_dim).to(self.device)
			starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

		sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict,
														  n_traj_samples=3)

		if self.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

		return self.decoder(sol_y)


class CoupledODE(VAE_Baseline_cgode):
	def __init__(self, w_node_to_edge_initial,ode_hidden_dim, encoder_z0, decoder_node,decoder_edge, diffeq_solver,
				 z0_prior, device, obsrv_std=None, n_balls=0):

		super(CoupledODE, self).__init__(
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std
		)



		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder_node = decoder_node
		self.decoder_edge = decoder_edge
		self.ode_hidden_dim =ode_hidden_dim
		self.n_balls =n_balls


		# Shared with edge ODE
		self.w_node_to_edge_initial = w_node_to_edge_initial #h_ij = W([h_i||h_j])



	def get_reconstruction(self, batch_en,batch_de,num_atoms):

        #Encoder:
		first_point_mu, first_point_std = self.encoder_z0(batch_en.x, batch_en.edge_weight,
														  batch_en.edge_index, batch_en.pos, batch_en.edge_same,
														  batch_en.batch, batch_en.y)  # [K*N,D]

		first_point_enc = utils.sample_standard_gaussian(first_point_mu, first_point_std) #[K*N,D]



		first_point_std = first_point_std.abs()

		time_steps_to_predict = batch_de["time_steps"]


		assert (torch.sum(first_point_std < 0) == 0.)
		assert (not torch.isnan(time_steps_to_predict).any())
		assert (not torch.isnan(first_point_enc).any())

		assert (not torch.isnan(first_point_std).any())
		assert (not torch.isnan(first_point_mu).any())



		# ODE:Shape of sol_y #[ K*N + K*N*N, time_length, d], concat of node and edge.
		# K_N is the index for node.
		sol_y,K_N = self.diffeq_solver(first_point_enc,time_steps_to_predict,self.w_node_to_edge_initial)

		assert(not torch.isnan(sol_y).any())

        # Decoder:
		pred_node = self.decoder_node(sol_y[:K_N,:,:])
		pred_edge = self.decoder_edge(sol_y[K_N:, :, :])


		all_extra_info = {
			"first_point": (first_point_mu, first_point_std, first_point_enc),
			"latent_traj": sol_y.detach()
		}

		return pred_node,pred_edge, all_extra_info, None


	def compute_edge_initials(self,first_point_enc,num_atoms):
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

		first_point_enc = first_point_enc.view(-1,num_atoms,node_feature_num) #[K,N,D]

		senders = torch.matmul(rel_send,first_point_enc) #[K,N*N,D]
		receivers = torch.matmul(rel_rec,first_point_enc) #[K,N*N,D]

		edge_initials = torch.cat([senders,receivers],dim=-1)  #[K,N*N,2D]
		edge_initials = F.relu(self.w_node_to_edge_initial(edge_initials)) #[K,N*N,D_edge]
		edge_initials = edge_initials.view(-1,edge_initials.shape[2]) #[K*N*N,D_edge]

		return edge_initials



