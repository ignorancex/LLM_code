from lib.likelihood_eval import *
import torch

def gaussian_log_likelihood(mu, data, obsrv_std):
	log_p = ((mu - data) ** 2) / (2 * obsrv_std * obsrv_std)
	neg_log_p = -1*log_p
	return neg_log_p

def generate_time_weight(n_timepoints,n_dims):
	value_min = 1
	value_max = 2
	interval = (value_max - value_min)/(n_timepoints-1)

	value_list = [value_min + i*interval for i in range(n_timepoints)]
	value_list= torch.FloatTensor(value_list).view(-1,1)

	value_matrix= torch.cat([value_list for _ in range(n_dims)],dim = 1)

	return value_matrix

def compute_masked_likelihood_cgode(mu, data ,mask ,mu_gt = None, likelihood_func = None,temporal_weights=None):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj, n_timepoints, n_dims = mu.size()
	if mu_gt != None:
		log_prob = likelihood_func(mu, data,mu_gt)  # [n_traj, n_timepoints, n_dims]
	else:
		log_prob = likelihood_func(mu, data)  # MSE
	if mask != None:
		log_prob_masked = torch.sum(log_prob * mask, dim=1)  # [n_traj, n_dims]
		timelength_per_nodes = torch.sum(mask.permute(0, 2, 1), dim=2)  # [n_traj, n_dims]
		assert (not torch.isnan(timelength_per_nodes).any())
		log_prob_masked_normalized = torch.div(log_prob_masked,
											   timelength_per_nodes)  # 【n_traj, feature], average each feature by dividing time length
		# Take mean over the number of dimensions
		res = torch.mean(log_prob_masked_normalized, -1)  # 【n_traj], average among features.
	else:
		res = torch.sum(log_prob , dim=1)  # [n_traj,n_dims]
		time_length = log_prob.shape[1]
		res = torch.div(res,time_length)
		res = torch.mean(res,-1)


	return res

def compute_masked_likelihood(mu, data, mask, likelihood_func,temporal_weights=None):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	log_prob = likelihood_func(mu, data)  # [n_traj, n_traj_samples, n_timepoints, n_dims]
	if temporal_weights!= None:
		weight_for_times = torch.cat([temporal_weights for _ in range(n_dims)],dim = 1)
		weight_for_times = weight_for_times.to(mu.device)
		weight_for_times = weight_for_times.repeat(n_traj_samples, n_traj, 1, 1)
		log_prob_masked = torch.sum(log_prob * mask * weight_for_times, dim=2)  # [n_traj, n_traj_samples, n_dims]
	else:
		log_prob_masked = torch.sum(log_prob * mask, dim=2)  # [n_traj, n_traj_samples, n_dims]


	timelength_per_nodes = torch.sum(mask.permute(0,1,3,2),dim=3)
	assert (not torch.isnan(timelength_per_nodes).any())
	log_prob_masked_normalized = torch.div(log_prob_masked , timelength_per_nodes) #【n_traj_sample, n_traj, feature], average each feature by dividing time length
	# Take mean over the number of dimensions
	res = torch.mean(log_prob_masked_normalized, -1) # 【n_traj_sample, n_traj], average among features.
	res = res.transpose(0,1)
	return res


def compute_masked_likelihood_old(mu, data, mask, likelihood_func):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()


	log_prob = likelihood_func(mu, data)  # [n_traj, n_traj_samples, n_timepoints, n_dims]
	log_prob_masked = torch.sum(log_prob * mask, dim=2)  # [n_traj, n_traj_samples, n_dims]

	timelength_per_nodes = torch.sum(mask.permute(0, 1, 3, 2), dim=3)
	assert (not torch.isnan(timelength_per_nodes).any())
	log_prob_masked_normalized = torch.div(log_prob_masked,
										   timelength_per_nodes)  # 【n_traj_sample, n_traj, feature], average each feature by dividing time length
	# Take mean over the number of dimensions
	res = torch.mean(log_prob_masked_normalized, -1)  # 【n_traj_sample, n_traj], average among features.
	res = res.transpose(0, 1)
	return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask,temporal_weights=None):

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	func = lambda mu, data: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std)
	res = compute_masked_likelihood(mu, data,mask, func,temporal_weights)
	return res

def masked_gaussian_log_density_cgode(mu, data, obsrv_std, mask=None,temporal_weights=None):

	n_traj, n_timepoints, n_dims = mu.size()  #n_traj = K*N

	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	print('size of mu, data',mu.shape,data.shape)
	func = lambda mu, data: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std)
	res = compute_masked_likelihood_cgode(mu, data,mask, likelihood_func=func)  #[n_traj = K*N]
	return res

def masked_gaussian_log_density_lode(mu, data, obsrv_std, mask=None):
	# these cases are for plotting through plot_estim_density
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	assert (data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)

		res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
		res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		func = lambda mu, data, indices: gaussian_log_likelihood(mu, data, obsrv_std=obsrv_std, indices=indices)
		res = compute_masked_likelihood(mu, data, mask, func)
	return res

def mse(mu,data):
	return  (mu - data) ** 2


def compute_mse(mu, data, mask):

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	res = compute_masked_likelihood(mu, data, mask, mse)
	return res

	
def compute_mse_lode(mu, data, mask = None):
	# these cases are for plotting through plot_estim_density
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		res = mse(mu_flat, data_flat)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		res = compute_masked_likelihood(mu, data, mask, mse)
	return res


def compute_poisson_proc_likelihood(truth, pred_y, info, mask=None):
	# Compute Poisson likelihood
	# https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process
	# Sum log lambdas across all time points
	if mask is None:
		poisson_log_l = torch.sum(info["log_lambda_y"], 2) - info["int_lambda"]
		# Sum over data dims
		poisson_log_l = torch.mean(poisson_log_l, -1)
	else:
		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		mask_repeated = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
		int_lambda = info["int_lambda"]
		f = lambda log_lam, data, indices: poisson_log_likelihood(log_lam, data, indices, int_lambda)
		poisson_log_l = compute_masked_likelihood(info["log_lambda_y"], truth_repeated, mask_repeated, f)
		poisson_log_l = poisson_log_l.permute(1, 0)
	# Take mean over n_traj
	# poisson_log_l = torch.mean(poisson_log_l, 1)

	# poisson_log_l shape: [n_traj_samples, n_traj]
	return poisson_log_l


def compute_loss_cgode(mu, data,mu_gt=None, mask=None,method=None):
	# mu is prediction; data is groud truth

	n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	if method in ["MSE","RMSE"]:
		res = compute_masked_likelihood(mu, data, mask,likelihood_func = mse)
	elif method == "MAPE":
		res = compute_masked_likelihood(mu, data,mask, mu_gt = mu_gt,likelihood_func=mape)


	return res
