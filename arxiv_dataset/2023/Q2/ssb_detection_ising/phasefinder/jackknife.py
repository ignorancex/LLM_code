import numpy as np

def calculate_samples(data):
	M = data.shape[0]
	sums = data.sum(0)
	estimates = sums/M
	samples = (sums[None,:] - data)/(M-1)
	samples = np.concatenate((estimates[None,:], samples), 0)
	return samples

def calculate_mean_std(samples, reduce_bias=True, return_bias=False):
	D = samples.ndim
	if D == 1:
		samples = samples[:,None]
	estimates, samples = samples[0], samples[1:]
	N = samples.shape[0]
	mean = samples.mean(0)
	var = (N-1)*samples.var(0)
	mean_bias = N*(estimates-mean)
	var_bias = (N-1)*(estimates-mean)**2
	if reduce_bias:
		mean = mean + mean_bias
		var = var + var_bias
	if D == 1:
		mean, var, mean_bias, var_bias = mean.item(), var.item(), mean_bias.item(), var_bias.item()
	if return_bias:
		return mean, np.sqrt(var), mean_bias, np.sqrt(var_bias)
	return mean, np.sqrt(var)
