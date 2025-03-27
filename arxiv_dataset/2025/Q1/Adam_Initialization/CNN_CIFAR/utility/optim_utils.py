import torch

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
	for batch_idx, (inputs, targets) in enumerate(data_set):
		inputs, targets = inputs.to(device), targets.to(device)

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

	return grad_mean, grad_var, grad_sq


def init_optim(optimizer,criterion=None,data_set=None, model=None, init_method='constant-layer',
               scaling_factor=1.0, constant_value=1e-2, device="cpu",data_size=-1,
               algo='adam'):
	scale_logger=[]
	if init_method in ['constant', 'constant-layer','constant-w','random','random-kaiming','randommv','constant-layermv']:
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
					state['exp_avg'] = torch.zeros_like(p.data)
					state['exp_avg_sq'] = torch.zeros_like(p.data)

				if init_method == "constant":
					# Use constant initialization for v_0
					v_0 = constant_value
					state['exp_avg_sq'].fill_(v_0)
					scale_logger.append(v_0)

				elif init_method == "constant-layer":
					# Use layer-wise scaling for v_0 initialization
					n_in, n_out = p.data.size(0), p.data.size(1) if p.data.ndim > 1 else 1
					v_0 = scaling_factor / (n_in + n_out)
					state['exp_avg_sq'].fill_(v_0)
					scale_logger.append(v_0)

				elif init_method == "constant-w":
					# Use layer-wise scaling for v_0 initialization
					if len(p.data.size()) > 1:
						n_in, n_out = p.data.size(0), p.data.size(1) if p.data.ndim > 1 else 1
						v_0 = scaling_factor / (n_in + n_out)
						state['exp_avg_sq'].fill_(v_0)
						scale_logger.append(v_0)

				elif init_method == "random":
					exp_avg_sq = torch.zeros_like(p).detach()
					if len(p.data.size()) > 1:
						torch.nn.init.xavier_normal_(exp_avg_sq)
						v_0 = scaling_factor * (exp_avg_sq ** 2)
						state['exp_avg_sq'].copy_(v_0)
						scale_logger.append(v_0.mean().item())

				elif init_method == "random-kaiming":
					exp_avg_sq = torch.zeros_like(p).detach()
					if len(p.data.size()) > 1:
						torch.nn.init.kaiming_normal_(exp_avg_sq)
						v_0 = scaling_factor * (exp_avg_sq ** 2)
						state['exp_avg_sq'].copy_(v_0)
						scale_logger.append(v_0.mean().item())

				elif init_method == "randommv":
					exp_avg_sq = torch.zeros_like(p).detach()
					exp_avg = torch.zeros_like(p).detach()
					if len(p.data.size()) > 1:
						torch.nn.init.xavier_normal_(exp_avg_sq)
						v_0 = scaling_factor * (exp_avg_sq ** 2)
						state['exp_avg_sq'].copy_(v_0)
						scale_logger.append(v_0.mean().item())

						torch.nn.init.xavier_normal_(exp_avg)
						state['exp_avg'].copy_(exp_avg)

				elif init_method == "constant-layermv":
					# Use layer-wise scaling for v_0 initialization
					n_in, n_out = p.data.size(0), p.data.size(1) if p.data.ndim > 1 else 1
					v_0 = scaling_factor / (n_in + n_out)
					state['exp_avg_sq'].fill_(v_0)
					scale_logger.append(v_0)

					state['exp_avg'].fill_(1/ (n_in + n_out))


	elif init_method in ['grad-var', 'grad-sq','hessian']:
		grad_mean, grad_var, grad_sq = cal_grad_stats(optimizer, criterion, data_set, model, device=device, data_size=data_size)
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
					state['exp_avg'] = torch.zeros_like(p.data)
					state['exp_avg_sq'] = torch.zeros_like(p.data)

				if init_method == "grad-var":
					# Initialize v_0 with variance of the gradient
					v_0 = scaling_factor * grad_var[p]
					state['exp_avg_sq'].copy_(v_0)
					scale_logger.append(v_0.mean().item())

				elif init_method == "grad-sq":
					# Initialize v_0 with variance of the gradient
					v_0 = scaling_factor * grad_sq[p]
					state['exp_avg_sq'].copy_(v_0)
					scale_logger.append(v_0.mean().item())

				elif init_method == "hessian":
					# Initialize v_0 with variance of the gradient
					grad = grad_mean[p]  # Use average gradient for approximation
					hessian_diag = \
					torch.autograd.grad(outputs=grad, inputs=p, grad_outputs=torch.ones_like(grad), retain_graph=True,
					                    create_graph=True)[0]
					v_0 =  hessian_diag.detach().pow(2)
					state['exp_avg_sq'].copy_(v_0)
					scale_logger.append(v_0.mean().item())

	else:
		raise ValueError(f"Unknown initialization strategy: {init_method}")

	# Reset gradients and set model back to train mode
	optimizer.zero_grad()
	model.train()
	scale_logger = torch.tensor(scale_logger)
	print(f"Conduct initialization strategy: {init_method} with scale:{scale_logger.mean()}")
