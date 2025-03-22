import pickle
import numpy as np
from lib.likelihood_eval import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn as nn
import torch
from .utils import makedirs, get_device
from lib.gnn_models import set_model_task
import time
import lib.utils as utils

def create_classifier(z0_dim, n_labels):
	return nn.Sequential(
			nn.Linear(z0_dim, 300),
			nn.ReLU(),
			nn.Linear(300, 300),
			nn.ReLU(),
			nn.Linear(300, n_labels),)
class VAE_Baseline(nn.Module):
	def __init__(self, input_dim, latent_dim, 
		z0_prior, device,
		obsrv_std = 0.01, 
		):

		super(VAE_Baseline, self).__init__()
		
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.device = device
		self.mu_proto = {}
		self.std_proto = {}
		self.data_counter = {}

		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

		self.z0_prior = z0_prior

	def get_gaussian_likelihood(self, truth, pred_y,temporal_weights, mask ):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		mask = mask.repeat(pred_y.size(0), 1, 1, 1)
		log_density_data = masked_gaussian_log_density(pred_y, truth_repeated,
			obsrv_std = self.obsrv_std, mask = mask,temporal_weights= temporal_weights) #【num_traj,num_sample_traj] [250,3]
		log_density_data = log_density_data.permute(1,0)
		log_density = torch.mean(log_density_data, 1)

		# shape: [n_traj_samples]
		return log_density


	def get_mse(self, truth, pred_y, mask = None):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]
		n_traj, n_tp, n_dim = truth.size()

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		mask = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute likelihood of the data under the predictions
		log_density_data = compute_mse(pred_y, truth_repeated, mask = mask)
		# shape: [1]
		return torch.mean(log_density_data)

	def update_class_proto(self, task_id, first_point_mu, first_point_std):
		if task_id in self.data_counter:
			# if task_id has been observed
			alpha = first_point_mu.shape[0] / (self.data_counter[task_id] + first_point_mu.shape[0]) # ratio of new data
			self.mu_proto[task_id] = (1-alpha)*self.mu_proto[task_id] + alpha*first_point_mu.mean(0)
			self.std_proto[task_id] = (1-alpha)*self.std_proto[task_id] + alpha*first_point_std.mean(0)
			self.data_counter[task_id] += first_point_mu.shape[0]
		else:
			self.mu_proto[task_id] = first_point_mu.mean(0)
			self.std_proto[task_id] = first_point_std.mean(0)
			self.data_counter[task_id] = first_point_mu.shape[0]


	def proto_dist_triplet_loss(self, fp_mu_current, fp_std_current, batch_dict_encoder,batch_dict_decoder,batch_dict_graph,n_traj_samples, margin=1.0):
		triplet_losses = [] # a triplet loss for each old task
		for old_id in range(self.num_tasks_learned):
			set_model_task(self, old_id, verbose=False)
			with torch.no_grad():
				# current task data representation generated with old masks
				_, _, _, fp_mu_old, fp_std_old = self.get_reconstruction(batch_dict_encoder, batch_dict_decoder,
																			  batch_dict_graph,
																			  n_traj_samples=n_traj_samples)
			dist_old = kl_divergence(Normal(fp_mu_old.mean(0), fp_std_old.mean(0)), Normal(self.mu_proto[old_id], self.std_proto[old_id])) # distance to old protos
			dist_old = torch.mean(dist_old)
			current_distribution = Normal(fp_mu_current, fp_std_current)
			existing_prototype = Normal(fp_mu_old, fp_std_old)
			d = kl_divergence(current_distribution, existing_prototype) # distance to new protos
			triplet_loss = margin + d - dist_old
			triplet_losses.append(triplet_loss)
		loss = torch.mean(sum(triplet_losses)/len(triplet_losses),(0,1))
		return loss

	def compute_all_losses(self, batch_dict_encoder,batch_dict_decoder,batch_dict_graph, task_id=None, n_traj_samples = 1, kl_coef = 1.,save_pred_traj=False, learn_task_id=None, update_proto=False):
		# Condition on subsampled points
		# Make predictions for all the points
		t0=time.time()
		pred_y, info,temporal_weights, first_point_mu, first_point_std= self.get_reconstruction(batch_dict_encoder,batch_dict_decoder,batch_dict_graph,n_traj_samples = n_traj_samples)
		# pred_y: [bs*n_balls, hidden_dim]


		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		#print("get_reconstruction done -- computing likelihood")
		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs()
		fp_distr = Normal(fp_mu, fp_std)

		assert(torch.sum(fp_std < 0) == 0.)
		self.z0_prior = Normal(torch.Tensor([task_id]).to(fp_mu.get_device()), torch.Tensor([1.]).to(fp_mu.get_device())) if task_id else self.z0_prior # task_id is None when testing
		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)
		proto_dist_loss = 0

		if learn_task_id is not None:
			# model learns a prototype for each task
			if learn_task_id == 'mask_prototype':
				if task_id>0:
					# enlarge the distance between other prototypes
					proto_dist_loss = 0 #self.proto_dist_triplet_loss(first_point_mu, first_point_std, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, n_traj_samples)
				if update_proto:
					# only update proto when the model has been trained with the last epoch
					self.update_class_proto(task_id, first_point_mu,
											first_point_std)  # update the prototype of the current task during training

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

		mse = self.get_mse(
			batch_dict_decoder["data"], pred_y,
			mask=batch_dict_decoder["mask"])  # [1]

		# save predicted trajectories if specified
		if save_pred_traj:
			save_path = './predicted_n_gt'
			makedirs(save_path)
			with open(save_path+'/001.pkl','wb') as f:
				pickle.dump([pred_y,batch_dict_decoder["data"]],f)


		# loss

		loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0) #- proto_dist_loss
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0) #- proto_dist_loss

		results = {}
		results["loss"] = torch.mean(loss)
		results["all_loss"] = loss
		results["likelihood"] = torch.mean(rec_likelihood).data.item()
		results["mse"] = torch.mean(mse).data.item()
		results["kl_first_p"] =  torch.mean(kldiv_z0).detach().data.item()
		results["std_first_p"] = torch.mean(fp_std).detach().data.item()

		return results

	def compute_all_losses_bcsr(self, batch_dict_encoder,batch_dict_decoder,batch_dict_graph, task_id=None, n_traj_samples = 1, kl_coef = 1.,save_pred_traj=False, learn_task_id=None, update_proto=False):
		# modified from compute_all_losses, generate loss of each example for BCSR coreset selection
		t0=time.time()
		pred_y, info,temporal_weights, first_point_mu, first_point_std= self.get_reconstruction(batch_dict_encoder,batch_dict_decoder,batch_dict_graph,n_traj_samples = n_traj_samples)
		batch_size = batch_dict_graph.shape[0]
		n_vars = int(pred_y.shape[1]/batch_size)


		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		#print("get_reconstruction done -- computing likelihood")
		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs()
		fp_distr = Normal(fp_mu, fp_std)

		assert(torch.sum(fp_std < 0) == 0.)
		self.z0_prior = Normal(torch.Tensor([task_id]).to(fp_mu.get_device()), torch.Tensor([1.]).to(fp_mu.get_device())) if task_id else self.z0_prior # task_id is None when testing
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
		rec_likelihood_list = []
		for b in range(batch_size):
			start_id, end_id = n_vars*b, n_vars*(b+1)
			rec_likelihood = self.get_gaussian_likelihood(
				batch_dict_decoder["data"][start_id:end_id, ...], pred_y[:, start_id:end_id, ...], temporal_weights,
				mask=batch_dict_decoder["mask"][start_id:end_id, ...])  # negative value
			rec_likelihood_list.append(rec_likelihood)

		# loss
		losses = []
		for rec_likelihood in rec_likelihood_list:
			loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)  # - proto_dist_loss
			if torch.isnan(loss):
				loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0, 0)  # - proto_dist_loss
			losses.append(loss)

		return losses


class VAE_Baseline_lode(nn.Module):
	def __init__(self, input_dim, latent_dim,
				 z0_prior, device,
				 obsrv_std=0.01,
				 use_binary_classif=False,
				 classif_per_tp=False,
				 use_poisson_proc=False,
				 linear_classifier=False,
				 n_labels=1,
				 train_classif_w_reconstr=False):

		super(VAE_Baseline_lode, self).__init__()

		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.device = device
		self.n_labels = n_labels

		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

		self.z0_prior = z0_prior
		self.use_binary_classif = use_binary_classif
		self.classif_per_tp = classif_per_tp
		self.use_poisson_proc = use_poisson_proc
		self.linear_classifier = linear_classifier
		self.train_classif_w_reconstr = train_classif_w_reconstr

		z0_dim = latent_dim
		if use_poisson_proc:
			z0_dim += latent_dim

		if use_binary_classif:
			if linear_classifier:
				self.classifier = nn.Sequential(
					nn.Linear(z0_dim, n_labels))
			else:
				self.classifier = create_classifier(z0_dim, n_labels)
			utils.init_network_weights(self.classifier)

	def get_gaussian_likelihood(self, truth, pred_y, mask=None):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]
		n_traj, n_tp, n_dim = truth.size()

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)
		log_density_data = masked_gaussian_log_density_lode(pred_y, truth_repeated,
													   obsrv_std=self.obsrv_std, mask=mask)
		log_density_data = log_density_data.permute(1, 0)
		log_density = torch.mean(log_density_data, 1)

		# shape: [n_traj_samples]
		return log_density

	def get_mse(self, truth, pred_y, mask=None):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]
		n_traj, n_tp, n_dim = truth.size()

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute likelihood of the data under the predictions
		log_density_data = compute_mse_lode(pred_y, truth_repeated, mask=mask)
		# shape: [1]
		return torch.mean(log_density_data)

	def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, task_id=None,
						   n_traj_samples=1, kl_coef=1., save_pred_traj=False, learn_task_id=None, update_proto=False):
		# Condition on subsampled points
		# Make predictions for all the points
		pred_y, info, temporal_weights, first_point_mu, first_point_std = self.get_reconstruction(batch_dict_encoder,
																								  batch_dict_decoder,
																								  batch_dict_graph,
																								  n_traj_samples=n_traj_samples)

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
			batch_dict_decoder, pred_y)

		mse = self.get_mse(
			batch_dict_decoder, pred_y)

		pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict_decoder))
		if self.use_poisson_proc:
			pois_log_likelihood = compute_poisson_proc_likelihood(
				batch_dict_decoder, pred_y,
				info)
			# Take mean over n_traj
			pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

		################################
		# Compute CE loss for binary classification on Physionet
		device = get_device(batch_dict_decoder)
		ce_loss = torch.Tensor([0.]).to(device)


		# IWAE loss
		loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0, 0)

		if self.use_poisson_proc:
			loss = loss - 0.1 * pois_log_likelihood

		if self.use_binary_classif:
			if self.train_classif_w_reconstr:
				loss = loss + ce_loss * 100
			else:
				loss = ce_loss

		results = {}
		results["loss"] = torch.mean(loss)
		results["likelihood"] = torch.mean(rec_likelihood).detach()
		results["mse"] = torch.mean(mse).detach()
		results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
		results["ce_loss"] = torch.mean(ce_loss).detach()
		results["kl_first_p"] = torch.mean(kldiv_z0).detach()
		results["std_first_p"] = torch.mean(fp_std).detach()

		return results




class VAE_Baseline_cgode(nn.Module):
	def __init__(self,
				 z0_prior, device,
				 obsrv_std=0.01,
				 ):

		super(VAE_Baseline_cgode, self).__init__()

		self.device = device

		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

		self.z0_prior = z0_prior

	def get_gaussian_likelihood(self, truth, pred_y, temporal_weights=None, mask=None):
		# pred_y shape [K*N, n_tp, n_dim]
		# truth shape  [K*N, n_tp, n_dim]

		# Compute likelihood of the data under the predictions

		log_density_data = masked_gaussian_log_density_cgode(pred_y, truth,
													   obsrv_std=self.obsrv_std, mask=mask,
													   temporal_weights=temporal_weights)  # 【num_traj = K*N] [250,3]
		log_density = torch.mean(log_density_data)

		# shape: [n_traj_samples]
		return log_density

	def get_loss(self, truth, pred_y, truth_gt=None, mask=None, method='MSE', istest=False):
		# pred_y shape [n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]

		# Transfer from inc to cum

		truth = utils.inc_to_cum(truth)
		pred_y = utils.inc_to_cum(pred_y)
		num_times = truth.shape[1]
		time_index = [num_times - 1]  # last timestamp

		if istest:
			truth = truth[:, time_index, :]
			pred_y = pred_y[:, time_index, :]  # [N,1,D]
			if truth_gt != None:
				truth_gt = truth_gt[:, time_index, :]

		# Compute likelihood of the data under the predictions
		log_density_data = compute_loss_cgode(pred_y, truth, truth_gt, mask=mask, method=method)
		# shape: [1]
		return torch.mean(log_density_data)

	def print_out_pred(self, pred_node, pred_edge):

		pred_node = pred_node  # [N,T,D]
		pred_node = utils.inc_to_cum(pred_node)

		num_times = pred_node.shape[1]
		time_index = [num_times - 1]  # last timestamp

		pred_node = pred_node[:, time_index, :]

		pred_node = torch.squeeze(pred_node)

		pred_node = pred_node.cpu().detach().tolist()  # [N,1,D]
		return utils.print_MAPE(pred_node)

	def print_out_pred_sum(self, pred_node):
		pred_node = pred_node  # [N,T,D]
		pred_node = utils.inc_to_cum(pred_node)

		num_times = pred_node.shape[1]
		time_index = [num_times - 1]  # last timestamp

		pred_node = pred_node[:, time_index, :]

		pred_node = torch.squeeze(pred_node)

		pred_node = pred_node.cpu().detach().numpy()  # [N,1,D]

		print(np.sum(pred_node))

	def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, task_id=None, n_traj_samples = 1, kl_coef = 1.,save_pred_traj=False, learn_task_id=None, update_proto=False, istest=False,
		edge_lamda = 0.5):
		'''

		:param batch_dict_encoder:
		:param batch_dict_decoder: dict: 1. time 2. data: (K*N, T2, D)
		:param batch_dict_graph: #[K,T2,N,N], ground_truth graph with log normalization
		:param num_atoms:
		:param kl_coef:
		:return:
		'''
		pred_node, pred_edge, info, temporal_weights = self.get_reconstruction(batch_dict_encoder, batch_dict_decoder,
																			   num_atoms=self.n_balls)
		# pred_node [ K*N , time_length, d]
		# pred_edge [ K*N*N, time_length, d]

		if istest:
			mask_index = batch_dict_decoder["masks"]
			pred_node = pred_node[:, mask_index, :]
			pred_edge = pred_edge[:, mask_index, :]

		# Reshape batch_dict_graph
		k = batch_dict_graph.shape[0]
		T2 = batch_dict_graph.shape[1]
		truth_graph = torch.reshape(batch_dict_graph, (k, T2, -1))  # [K,T,N*N]
		truth_graph = torch.unsqueeze(truth_graph.permute(0, 2, 1), dim=3)  # [K,N*N,T,1]
		truth_graph = torch.reshape(truth_graph, (-1, T2, 1))  # [K*N*N,T,1]

		# print("get_reconstruction done -- computing likelihood")

		# KL divergence only contains node-level (only z_node are sampled, z_edge are computed from z_node)
		fp_mu, fp_std, fp_enc = info["first_point"]  # [K*N,D]
		fp_std = fp_std.abs()

		fp_distr = Normal(fp_mu, fp_std)

		assert (torch.sum(fp_std < 0) == 0.)
		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)  # [K*N,D_ode_latent]

		if torch.isnan(kldiv_z0).any():
			print(fp_mu)
			print(fp_std)
			raise Exception("kldiv_z0 is Nan!")

		if torch.isinf(kldiv_z0).any():
			locations = torch.where(kldiv_z0 == float("inf"), torch.Tensor([1]).to(fp_mu.device),
									torch.Tensor([0]).to(fp_mu.device))
			locations = locations.to("cpu").detach().numpy()
			mu_locations = fp_mu.to("cpu").detach().numpy() * locations
			std_locations = fp_std.to("cpu").detach().numpy() * locations
			_, mu_values = utils.convert_sparse(mu_locations)
			_, std_values = utils.convert_sparse(std_locations)
			print(mu_values)
			print(std_values)

		# Mean over number of latent dimensions
		# kldiv_z0 shape: [n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
		# kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
		# shape after: [1]
		kldiv_z0 = torch.mean(kldiv_z0)  # Contains infinity.

		# Compute likelihood of all the points
		rec_likelihood_node = self.get_gaussian_likelihood(
			batch_dict_decoder["data"], pred_node, temporal_weights,
			mask=None)  # negative value

		rec_likelihood_edge = self.get_gaussian_likelihood(
			truth_graph, pred_edge, temporal_weights,
			mask=None)  # negative value

		rec_likelihood = (1 - edge_lamda) * rec_likelihood_node + edge_lamda * rec_likelihood_edge

		mape_node = self.get_loss(
			batch_dict_decoder["data"], pred_node, truth_gt=batch_dict_decoder["data_gt"],
			mask=None, method='MAPE', istest=istest)  # [1]

		mse_node = self.get_loss(
			batch_dict_decoder["data"], pred_node,
			mask=None, method='MSE', istest=istest)  # [1]

		# loss

		loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0, 0)

		results = {}
		results["loss"] = torch.mean(loss)
		results["likelihood"] = torch.mean(rec_likelihood).data.item()
		results["MAPE"] = torch.mean(mape_node).data.item()
		results["MSE"] = torch.mean(mse_node).data.item()
		results["kl_first_p"] = kldiv_z0.detach().data.item()
		results["std_first_p"] = torch.mean(fp_std).detach().data.item()

		# if istest:
		# 	print("Predicted Inc Deaths are:")
		# 	print(self.print_out_pred(pred_node,pred_edge))
		# 	print(self.print_out_pred_sum(pred_node))

		return results


