import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_normalized_directions(model):
	"""Generate two random normalized directions (δ and η) for each parameter θ."""
	d1, d2 = [], []
	for param in model.parameters():
		# Generate random Gaussian directions
		delta = torch.randn_like(param)
		eta = torch.randn_like(param)
		# Normalize by the norm of the parameter θ
		delta = delta / torch.norm(delta) * torch.norm(param)
		eta = eta / torch.norm(eta) * torch.norm(param)
		d1.append(delta)
		d2.append(eta)
	return d1, d2


def perturb_model(model, original_params, d1, d2, alpha, beta):
	"""Perturb model parameters using directions δ and η."""
	for i, param in enumerate(model.parameters()):
		param.data = original_params[i] + alpha * d1[i] + beta * d2[i]


def evaluate_loss(model, dataloader, criterion,device='cpu'):
	"""Compute the loss over a dataset."""
	model.eval()
	total_loss = 0.0
	with torch.no_grad():
		for inputs, targets in dataloader:
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			total_loss += loss.item()
	return total_loss / len(dataloader)


def plot_loss_landscape(model, dataloader, criterion, alpha_range, beta_range, resolution,device='cpu'):
	"""Visualize the loss landscape."""
	original_params = [param.data.clone() for param in model.parameters()]
	d1, d2 = generate_normalized_directions(model)

	alpha_values = np.linspace(*alpha_range, resolution)
	beta_values = np.linspace(*beta_range, resolution)
	loss_values = np.zeros((resolution, resolution))

	for i, alpha in enumerate(alpha_values):
		for j, beta in enumerate(beta_values):
			perturb_model(model, original_params, d1, d2, alpha, beta)
			loss_values[i, j] = evaluate_loss(model, dataloader, criterion,device=device)
			print(alpha, beta, loss_values[i, j])

	# Reset original parameters
	for i, param in enumerate(model.parameters()):
		param.data = original_params[i]

	# Plot the loss landscape
	plt.figure(figsize=(8, 6))
	plt.contourf(alpha_values, beta_values, loss_values, levels=50, cmap='viridis')
	plt.colorbar(label='Loss')
	plt.xlabel('Alpha (δ direction)')
	plt.ylabel('Beta (η direction)')
	plt.title('Loss Landscape Visualization')
	plt.show()
