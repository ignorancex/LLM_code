import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from . import models


class Autoencoder(object):

	def __init__(self, encoder, decoder, epochs=1, lr=1e-3, rescale_lr=False, equivariance_reg=0, equivariance_pre=0, device="cpu"):
		self.encoder = encoder.to(device)
		self.decoder = decoder.to(device)
		self.epochs = epochs
		self.lr = lr
		self.rescale_lr = rescale_lr
		self.equivariance_reg = equivariance_reg
		self.equivariance_pre = equivariance_pre
		self.device = device

		self.latent_dim = self.encoder.output_dim
		if self.equivariance_reg > 0 and self.latent_dim == 2:
			transforms = { \
				"rho": lambda x: x[:,[1,3,0,2]], \
				"tau": lambda x: x[:,[1,0,3,2]], \
				"sigma": lambda x: -x \
			}
			self.regularizer_2D = models.SymmetryReg(self.encoder, transforms).to(self.device)
		else:
			self.regularizer_2D = None

		if self.rescale_lr:
			param_group1 = [encoder.linear1.weight]
			param_group2 = [decoder.linear2.weight, decoder.linear2.bias]
			param_group3 = [encoder.linear1.bias, encoder.linear2.weight, encoder.linear2.bias, decoder.linear1.weight, decoder.linear1.bias]
			self.optimizer = optim.Adam([{"params": param_group1, "lr": 2*self.lr/encoder.linear1.weight.shape[1]}, {"params": param_group2, "lr": self.lr/2}, {"params": param_group3, "lr": self.lr}])
		else:
			parameters = list(self.encoder.parameters()) + list(decoder.parameters())
			if self.regularizer_2D is not None:
				parameters = parameters + list(self.regularizer_2D.parameters())
			self.optimizer = optim.Adam(parameters, lr=self.lr)

	def fit(self, train_loader, callbacks):
		# callbacks before training
		callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
		for cb in callbacks:
			cb.start_of_training()
		print("Training model ...")
		print("---")
		for epoch in range(self.epochs):
			# callbacks at the start of the epoch
			for cb in callbacks:
				cb.start_of_epoch(epoch)
			batch_logs = {"loss": []}
			for (X_batch, T_batch) in train_loader:
				X_batch = X_batch.to(self.device)
				T_batch = T_batch.to(self.device)
				Z = self.encoder(X_batch)
				logits = self.decoder(torch.cat([Z, T_batch], 1))
				loss = F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2)
				if self.equivariance_reg > 0 and epoch >= self.equivariance_pre:
					if self.latent_dim == 1:
						transforms = [ \
							lambda x: torch.flip(x, [1]), \
							lambda x: -x \
						]
						cos_similarities = []
						for transform in transforms:
							Z_transformed = self.encoder(transform(X_batch))
							Z_square_norm = Z.pow(2).sum()
							Z_transformed_square_norm = Z_transformed.pow(2).sum()
							loss += self.equivariance_reg*( \
								(1 - Z_transformed_square_norm/Z_square_norm)**2 + \
								1 - (Z*Z_transformed).sum()**2/(Z_square_norm*Z_transformed_square_norm) )
							cos_similarities.append( (Z*Z_transformed).sum().unsqueeze(0)/torch.sqrt(Z_square_norm*Z_transformed_square_norm) )
						loss += 1 + torch.cat(cos_similarities, 0).min()
					else:
						loss += self.equivariance_reg * self.regularizer_2D(X_batch)
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()
				batch_logs["loss"].append(loss.item())
			# callbacks at the end of the epoch
			for cb in callbacks:
				cb.end_of_epoch(epoch, batch_logs)

		# callbacks at the end of training
		for cb in callbacks:
			cb.end_of_training()
		print("---")


	def evaluate(self, val_loader):
		losses = []
		with torch.no_grad():
			for (X_batch, T_batch) in val_loader:
				X_batch = X_batch.to(self.device)
				T_batch = T_batch.to(self.device)
				logits = self.decoder(torch.cat([self.encoder(X_batch), T_batch], 1))
				loss = F.binary_cross_entropy_with_logits(logits, (X_batch+1)/2)
				losses.append(loss.item())
		loss = np.mean(np.array(losses))

		return loss

	def encode(self, val_loader, transform=None):
		encodings = []
		with torch.no_grad():
			for (X_batch, _) in val_loader:
				X_batch = X_batch.to(self.device)
				if transform is not None:
					X_batch = transform(X_batch)
				encodings_batch = self.encoder(X_batch)
				encodings.append( encodings_batch.cpu().numpy() )
		encodings = np.concatenate(encodings, 0)

		return encodings

	def generator_reps(self, val_loader):
		assert self.latent_dim == 1, "generator_reps can only be used for latent dimension 1. For latent dimension 2, use generator_reps_2D."
		Z = self.encode(val_loader)
		Z_flip = self.encode(val_loader, transform=lambda X: torch.flip(X, [1]))
		Z_neg = self.encode(val_loader, transform=lambda X: -X)

		flip_rep = np.sum(Z*Z_flip)/np.sum(Z**2)
		neg_rep = np.sum(Z*Z_neg)/np.sum(Z**2)
		return flip_rep, neg_rep

	def generator_reps_2D(self, val_loader):
		assert self.latent_dim == 2, "generator_reps_2D can only be used for latent dimension 2. For latent dimension 1, use generator_reps."
		Z = self.encode(val_loader)
		P = np.eye(Z.shape[1], dtype=Z.dtype) - np.linalg.lstsq(Z, Z)[0]
		latent_param = self.regularizer_2D.latent_param.detach().cpu().numpy()
		offset = P@latent_param

		reps = {gen: np.linalg.lstsq(Z, self.encode(val_loader, transform=transform))[0] + offset for (gen, transform) in self.regularizer_2D.transforms.items()}
		return reps

	def save_encoder(self, filename):
		torch.save(self.encoder.state_dict(), filename)

	def save_decoder(self, filename):
		torch.save(self.decoder.state_dict(), filename)

	def load_encoder(self, filename):
		self.encoder.load_state_dict(torch.load(filename))

	def load_decoder(self, filename):
		self.decoder.load_state_dict(torch.load(filename))
