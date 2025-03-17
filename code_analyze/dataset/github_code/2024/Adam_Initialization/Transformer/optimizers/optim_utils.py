import numpy as np
import torch
import contextlib


def cal_grad_stats(optimizer,criterion,data_set, model, device="cpu", data_size=-1):
	# Initialize accumulators for gradient statistics
	grad_sum = {p: torch.zeros_like(p.data, device=device) for group in optimizer.param_groups for p in group['params']
	            if  p.requires_grad}
	grad_sq_sum = {p: torch.zeros_like(p.data, device=device) for group in optimizer.param_groups for p in
	               group['params'] if  p.requires_grad}
	count = 0

	# Set model to evaluation mode for gradient estimation
	model.eval()
	# Iterate over the data set to accumulate gradient statistics
	bsz = 0
	for batch_idx, (inputs, targets) in enumerate(data_set):
		inputs, targets = inputs.to(device), targets.to(device)
		if bsz ==0:
			bsz = len(inputs)

		# Forward pass to compute the output and loss
		outputs = model(inputs)
		loss = criterion(outputs, targets)

		# Backward pass to compute the gradients
		optimizer.zero_grad()
		loss.backward()

		# Accumulate gradients and gradient squares for each parameter
		for group in optimizer.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad_sum[p] += p.grad.detach()
				grad_sq_sum[p] += p.grad.detach().pow(2)

		count += 1
		if data_size > 0 and count * len(inputs) > data_size:
			break

	# Finalize statistics: compute averages
	grad_mean = {p: grad_sum[p] / count for p in grad_sum}
	grad_var = {p: (grad_sq_sum[p] / count) - grad_mean[p].pow(2) for p in grad_sum}  # Variance: E[g^2] - (E[g])^2
	grad_sq = {p: grad_sq_sum[p] / count for p in grad_sum}

	return grad_mean, grad_var, grad_sq, bsz


def cal_grad_stats_fairseq(trainer, progress, device='cpu',data_size=-1):
	optimizer = trainer.optimizer.optimizer
	grad_sum = {p: torch.zeros_like(p.data, device=device) for group in optimizer.param_groups for p in group['params']
	            if  p.requires_grad}
	grad_sq_sum = {p: torch.zeros_like(p.data, device=device) for group in optimizer.param_groups for p in
	               group['params'] if  p.requires_grad}
	count = 0
	bsz = 0
	trainer.model.eval()
	trainer.criterion.eval()
	#trainer.model.train()
	#trainer.criterion.train()
	trainer.zero_grad()
	for i, samples in enumerate(progress):
		for j, sample in enumerate(samples):
			sample = trainer._prepare_sample(sample)
			for key, value in sample.items():
				break
			#with contextlib.ExitStack():
			loss, sample_size, logging_output, gradsH = trainer.task.train_step(
				sample, trainer.model, trainer.criterion, trainer.optimizer,
				ignore_grad=False,ignore_optimizer=True
			)
			#loss, sample_size, logging_output = trainer.criterion(trainer.model, sample)
			trainer.zero_grad()
			loss.backward()
			if bsz ==0:
				bsz = sample_size
			# Accumulate gradients and gradient squares for each parameter
			for group in optimizer.param_groups:
				for p in group['params']:
					if p.grad is None:
						continue
					grad_sum[p] += p.grad.detach()
					grad_sq_sum[p] += p.grad.detach().pow(2)

			count += sample_size
			if data_size > 0 and count  > data_size:
				break
			print('count done', count, sample_size)
		if data_size > 0 and count > data_size:
			break
	# Finalize statistics: compute averages
	trainer.zero_grad()
	grad_mean = {p: grad_sum[p] / count for p in grad_sum}
	grad_var = {p: (grad_sq_sum[p] / count) - grad_mean[p].pow(2) for p in grad_sum}  # Variance: E[g^2] - (E[g])^2
	grad_sq = {p: grad_sq_sum[p] / count for p in grad_sum}

	return grad_mean, grad_var, grad_sq, bsz


def init_optim(optimizer,criterion=None,data_set=None, model=None, init_method='random', init_state='mv',
               scaling_factor=1.0, scaling_factor_m0=1.0, device="cpu",data_size=-1,
               algo='adam',trainer=None, progress=None):
	scale_logger={'m_0':[],'v_0':[],'shape':[]}
	if init_method in ['random','random-kaiming']:
		for group in optimizer.param_groups:
			for p in group['params']:
				if not p.requires_grad:
					continue
				# Access the optimizer state for the parameter
				state = optimizer.state[p]
				if len(state) == 0:
					# if algo=='adam':
					# 	state['step'] = torch.zeros(1,device=device)
					# else:
					state['step'] = 0
					if algo=='sgd':
						state['momentum_buffer'] = torch.zeros_like(p.data)
					else:
						state['exp_avg'] = torch.zeros_like(p.data)
						state['exp_avg_sq'] = torch.zeros_like(p.data)

				exp_avg_sq = torch.zeros_like(p).detach()
				exp_avg = torch.zeros_like(p).detach()
				m_0 = torch.zeros_like(p).detach()
				v_0 = torch.zeros_like(p).detach()

				if init_method == "random":
					if len(p.data.size()) > 1:
						if 'v' in init_state:
							torch.nn.init.xavier_normal_(exp_avg_sq)
							v_0 = exp_avg_sq ** 2
						if 'm' in init_state:
							torch.nn.init.xavier_normal_(exp_avg)
							m_0 =  exp_avg
				elif init_method == "random-kaiming":
					if len(p.data.size()) > 1:
						if 'v' in init_state:
							torch.nn.init.kaiming_normal_(exp_avg_sq)
							v_0 = exp_avg_sq ** 2
						if 'm' in init_state:
							torch.nn.init.kaiming_normal_(exp_avg)
							m_0 =  exp_avg

				v_0 = scaling_factor * v_0
				m_0 = scaling_factor_m0 * m_0
				if 'exp_avg_sq' in state:
					state['exp_avg'].copy_(m_0)
					state['exp_avg_sq'].copy_(v_0)
				else:
					state['momentum_buffer'].copy_(m_0)

				scale_logger['m_0'].append(m_0.mean().item())
				scale_logger['v_0'].append(v_0.mean().item())
				scale_logger['shape'].append(tuple(p.data.size()))

	elif init_method in ['grad-mean','grad-mean-var',"grad-mean-random"]:
		model.eval()
		#grad_mean, grad_var, grad_sq, bsz = cal_grad_stats(optimizer, criterion, data_set, model, device=device, data_size=data_size)
		grad_mean, grad_var, grad_sq, bsz = cal_grad_stats_fairseq(trainer, progress, device=device,data_size=data_size)
		for group in optimizer.param_groups:
			for p in group['params']:
				if not p.requires_grad:
					continue
				# Access the optimizer state for the parameter
				state = optimizer.state[p]
				if len(state) == 0:
					# if algo=='adam':
					# 	state['step'] = torch.zeros(1,device=device)
					# else:
					state['step'] = 0
					if algo=='sgd':
						state['momentum_buffer'] = torch.zeros_like(p.data)
					else:
						state['exp_avg'] = torch.zeros_like(p.data)
						state['exp_avg_sq'] = torch.zeros_like(p.data)

				m_0 = torch.zeros_like(p).detach()
				v_0 = torch.zeros_like(p).detach()
				exp_avg_sq = torch.zeros_like(p).detach()

				if init_method == "grad-mean":
					# Initialize v_0 with variance of the gradient
					if 'v' in init_state:
						v_0 =  grad_mean[p] ** 2
					if 'm' in init_state:
						m_0 = grad_mean[p]
				elif init_method == "grad-mean-random":
					# Initialize v_0 with variance of the gradient
					if 'v' in init_state:
						v_0 =  grad_mean[p] ** 2
						if len(p.data.size()) > 1:
							torch.nn.init.xavier_normal_(exp_avg_sq)
							v_0 += exp_avg_sq ** 2
					if 'm' in init_state:
						m_0 = grad_mean[p]
				elif init_method == "grad-mean-var":
					# Initialize v_0 with variance of the gradient
					if 'v' in init_state:
						v_0 =  grad_mean[p] ** 2 + bsz*grad_var[p]
					if 'm' in init_state:
						m_0 = grad_mean[p]

				v_0 = scaling_factor * v_0
				m_0 = scaling_factor_m0 * m_0
				if 'exp_avg_sq' in state:
					state['exp_avg'].copy_(m_0)
					state['exp_avg_sq'].copy_(v_0)
				else:
					state['momentum_buffer'].copy_(m_0)

				scale_logger['m_0'].append(m_0.mean().item())
				scale_logger['v_0'].append(v_0.mean().item())
				scale_logger['shape'].append(tuple(p.data.size()))

		optimizer.zero_grad()
		model.train()

	else:
		raise ValueError(f"Unknown initialization strategy: {init_method}")

	# Reset gradients and set model back to train mode

	print(f"Conduct initialization strategy: {init_method}-{init_state}-{scaling_factor}, with scale m_0:({np.mean(scale_logger['m_0'])},{np.std(scale_logger['m_0'])}), v_0:({np.mean(scale_logger['v_0'])},{np.std(scale_logger['v_0'])})")

	return scale_logger
