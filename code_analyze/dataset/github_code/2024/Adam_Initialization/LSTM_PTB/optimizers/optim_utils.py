import numpy as np
import torch

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def cal_grad_stats(optimizer,criterion,data_set, model, device="cpu", data_size=-1,args=None):
	# Initialize accumulators for gradient statistics
	grad_sum = {p: torch.zeros_like(p.data, device=device) for group in optimizer.param_groups for p in group['params']
	            if  p.requires_grad}
	grad_sq_sum = {p: torch.zeros_like(p.data, device=device) for group in optimizer.param_groups for p in
	               group['params'] if  p.requires_grad}
	count = 0
	model.train()

	hidden = model.init_hidden(args.batch_size)
	batch, batch_idx = 0, 0

	# Set model to evaluation mode for gradient estimation

	# Iterate over the data set to accumulate gradient statistics
	bsz = 0
	while batch_idx < data_set.size(0) - 1 - 1:
		#inputs, targets = inputs.to(device), targets.to(device)
		bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
		# Prevent excessively small or negative sequence lengths
		seq_len = max(5, int(np.random.normal(bptt, 5)))
		# There's a very small chance that it could select a very long sequence length resulting in OOM
		# seq_len = min(seq_len, args.bptt + 10)

		data, targets = get_batch(data_set, batch_idx, args, seq_len=seq_len)

		# Starting each batch, we detach the hidden state from how it was previously produced.
		# If we didn't, the model would try backpropagating all the way to start of the dataset.
		hidden = repackage_hidden(hidden)

		optimizer.zero_grad()
		# Forward pass to compute the output and loss
		output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
		loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
		# Backward pass to compute the gradients
		loss.backward()

		if bsz ==0:
			bsz = data.size(1)

		# Accumulate gradients and gradient squares for each parameter
		for group in optimizer.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad_sum[p] += p.grad.detach()
				grad_sq_sum[p] += p.grad.detach().pow(2)

		count += 1
		if data_size > 0 and count * data.size(1) > data_size:
			break

	# Finalize statistics: compute averages
	grad_mean = {p: grad_sum[p] / count for p in grad_sum}
	grad_var = {p: (grad_sq_sum[p] / count) - grad_mean[p].pow(2) for p in grad_sum}  # Variance: E[g^2] - (E[g])^2
	grad_sq = {p: grad_sq_sum[p] / count for p in grad_sum}

	return grad_mean, grad_var, grad_sq, bsz


def init_optim(optimizer,criterion=None,data_set=None, model=None, init_method='random', init_state='mv',
               scaling_factor=1.0, scaling_factor_m0=1.0, device="cpu",data_size=-1,
               algo='adam',args=None):
	scale_logger={'m_0':[],'v_0':[],'shape':[]}
	if init_method in ['random','random-kaiming']:
		for group in optimizer.param_groups:
			for p in group['params']:
				if not p.requires_grad:
					continue
				# Access the optimizer state for the parameter
				state = optimizer.state[p]
				if len(state) == 0:
					if algo=='adam':
						state['step'] = torch.zeros(1,device=device)
					else:
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
		grad_mean, grad_var, grad_sq, bsz = cal_grad_stats(optimizer, criterion, data_set, model, device=device, data_size=data_size,
		                                                   args=args)
		for group in optimizer.param_groups:
			for p in group['params']:
				if not p.requires_grad:
					continue
				# Access the optimizer state for the parameter
				state = optimizer.state[p]
				if len(state) == 0:
					if algo=='adam':
						state['step'] = torch.zeros(1,device=device)
					else:
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
