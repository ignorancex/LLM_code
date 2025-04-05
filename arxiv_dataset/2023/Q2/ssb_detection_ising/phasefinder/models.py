import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import utils


class MLP(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MLP, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.activation = nn.LeakyReLU()

		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, X):
		out = self.linear2(self.activation(self.linear1(X)))
		if out.ndim == 1:
			out = torch.reshape(out, [-1, 1])
		return out


class SymmetryReg(nn.Module):

	def __init__(self, encoder, transforms):
		super(SymmetryReg, self).__init__()
		self.encoder = encoder
		self.transforms = transforms

		self.dim = self.encoder.output_dim
		self.latent_param = nn.Parameter(torch.randn(self.dim, self.dim))

	def forward(self, X):
		Z = self.encoder(X)
		P = utils.nullproj(Z)
		I = torch.eye(Z.shape[1], dtype=Z.dtype)

		loss = 0.0
		for (_, transform) in self.transforms.items():
			Z_transformed = self.encoder(transform(X))
			sol = torch.linalg.lstsq(Z, Z_transformed).solution
			orth_sol = sol + P@self.latent_param
			loss += 1.0 - (Z_transformed*(Z@sol)).sum()/Z_transformed.pow(2).sum() \
				+ (I - orth_sol.T@orth_sol).pow(2).sum()

		return loss
