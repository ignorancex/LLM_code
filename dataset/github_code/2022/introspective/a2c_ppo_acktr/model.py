import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
import pdb


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)


class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, is_lstm, dim, base=None, base_kwargs=None):
		super(Policy, self).__init__()
		if base_kwargs is None:
			base_kwargs = {}
		# if base is None:
		# 	if len(obs_shape) == 3:
		# 		base = CNNBase
		# 	elif len(obs_shape) == 1:
		# 		base = MLPBase
		# 	else:
		# 		raise NotImplementedError
		base = MLPBase
		self.base = base(obs_shape[0], dim, is_lstm, **base_kwargs)


		num_outputs = 4
		self.dist = Categorical(self.base.output_size, num_outputs)

		# if action_space.__class__.__name__ == "Discrete":
		#     num_outputs = action_space.n
		#     self.dist = Categorical(self.base.output_size, num_outputs)
		# elif action_space.__class__.__name__ == "Box":
		#     num_outputs = action_space.shape[0]
		#     self.dist = DiagGaussian(self.base.output_size, num_outputs)
		# elif action_space.__class__.__name__ == "MultiBinary":
		#     num_outputs = action_space.shape[0]
		#     self.dist = Bernoulli(self.base.output_size, num_outputs)
		# else:
		#     raise NotImplementedError

	@property
	def is_recurrent(self):
		return self.base.is_recurrent

	@property
	def recurrent_hidden_state_size(self):
		"""Size of rnn_hx."""
		return self.base.recurrent_hidden_state_size

	def forward(self, inputs, rnn_hxs, masks):
		raise NotImplementedError

	def act(self, inputs, rnn_hxs, rnn_hxs_1, masks, deterministic=False):
		value, actor_features, rnn_hxs, rnn_hxs_1, distance, distance_ahead = self.base(inputs, rnn_hxs, rnn_hxs_1, masks)
		dist = self.dist(actor_features)

		if deterministic:
			#pdb.set_trace()
			#dist.probs[0][3]=0
			action = dist.mode()
		else:
			action = dist.sample()

		action_log_probs = dist.log_probs(action)
		dist_entropy = dist.entropy().mean()

		return value, action, action_log_probs, rnn_hxs, rnn_hxs_1, distance, distance_ahead

	def get_value(self, inputs, rnn_hxs, rnn_hxs_1, masks):
		value, _, _, _, dist, dist_ahead = self.base(inputs, rnn_hxs, rnn_hxs_1, masks)
		return value

	def evaluate_actions(self, inputs, rnn_hxs, rnn_hxs_1, masks, action):
		value, actor_features, rnn_hxs, rnn_hxs_1, distance, distance_ahead = self.base(inputs, rnn_hxs, rnn_hxs_1, masks)
		dist = self.dist(actor_features)

		action_log_probs = dist.log_probs(action)
		dist_entropy = dist.entropy().mean()

		return value, action_log_probs, dist_entropy, rnn_hxs, rnn_hxs_1, distance, distance_ahead


class NNBase(nn.Module):
	def __init__(self, recurrent, dim, recurrent_input_size, hidden_size, lstm):
		super(NNBase, self).__init__()

		self._hidden_size = dim
		self._recurrent = recurrent
		self.use_lstm = lstm

		if recurrent:
			# if self.use_lstm:
			# 	self.gru = nn.LSTM(recurrent_input_size - 2 + 8, dim)
			# else:
			self.gru = nn.GRU(recurrent_input_size - 2 + 8, dim)
			for name, param in self.gru.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

	@property
	def is_recurrent(self):
		return self._recurrent

	@property
	def recurrent_hidden_state_size(self):
		if self._recurrent:
			return self._hidden_size
		return 1

	@property
	def output_size(self):
		return self._hidden_size

	def _forward_gru(self, x, hxs, hxs_1, masks):
		if x.size(0) == hxs.size(0):
			#if self.use_lstm:
			#x, (hxs, hxs_1) = self.gru(x.unsqueeze(0), ((hxs * masks).unsqueeze(0), (hxs_1 * masks).unsqueeze(0)))
			# else:
			x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
			x = x.squeeze(0)
			hxs = hxs.squeeze(0)
			hxs_1 = hxs_1.squeeze(0)
		else:

			#pdb.set_trace()
			# x, (hxs, hxs_1) = self.gru(x.unsqueeze(1), ((hxs * masks), (hxs_1 * masks)))
			# x = x.squeeze(0)
			# hxs = hxs.squeeze(0)
			# hxs_1 = hxs_1.squeeze(0)

			#print("CHECK")
			# x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
			N = hxs.size(0)
			T = int(x.size(0) / N)

			# unflatten
			x = x.view(T, N, x.size(1))
			#pdb.set_trace()

			# Same deal with masks
			masks = masks.view(T, N)

			# Let's figure out which steps in the sequence have a zero for any agent
			# We will always assume t=0 has a zero in it as that makes the logic cleaner
			has_zeros = ((masks[1:] == 0.0) \
							.any(dim=-1)
							.nonzero()
							.squeeze()
							.cpu())

			# +1 to correct the masks[1:]
			if has_zeros.dim() == 0:
				# Deal with scalar
				has_zeros = [has_zeros.item() + 1]
			else:
				has_zeros = (has_zeros + 1).numpy().tolist()

			# add t=0 and t=T to the list
			has_zeros = [0] + has_zeros + [T]

			hxs = hxs.unsqueeze(0)
			hxs_1 = hxs_1.unsqueeze(0)
			outputs = []
			for i in range(len(has_zeros) - 1):
				# We can now process steps that don't have any zeros in masks together!
				# This is much faster
				start_idx = has_zeros[i]
				end_idx = has_zeros[i + 1]

				#pdb.set_trace()

				# if self.use_lstm:
				# 	rnn_scores, (hxs, hxs_1) = self.gru(
				# 		x[start_idx:end_idx],
				# 		(hxs * masks[start_idx].view(1, -1, 1), hxs_1 * masks[start_idx].view(1, -1, 1)))
				# else:
				rnn_scores, hxs = self.gru(
					x[start_idx:end_idx],
					hxs * masks[start_idx].view(1, -1, 1))

				outputs.append(rnn_scores)

			# assert len(outputs) == T
			# x is a (T, N, -1) tensor
			x = torch.cat(outputs, dim=0)
			# flatten
			x = x.view(T * N, -1)
			hxs = hxs.squeeze(0)
			hxs_1 = hxs_1.squeeze(0)

		return x, hxs, hxs_1


class CNNBase(NNBase):
	def __init__(self, num_inputs, recurrent=False, hidden_size=512):
		super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

		init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
							   constant_(x, 0), nn.init.calculate_gain('relu'))

		self.main = nn.Sequential(
			init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
			init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
			init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
			init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

		init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
							   constant_(x, 0))

		self.critic_linear = init_(nn.Linear(hidden_size, 1))

		self.train()

	def forward(self, inputs, rnn_hxs, masks):
		x = self.main(inputs / 255.0)

		if self.is_recurrent:
			x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

		return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
	def __init__(self, num_inputs, dim, is_lstm, recurrent=False, hidden_size=256):
		super(MLPBase, self).__init__(recurrent, dim, num_inputs, hidden_size, is_lstm)

		self.embed = nn.Embedding(4, 8)
		self.recurrent = recurrent

		if recurrent:
			num_inputs = dim
		else:
			print(num_inputs)
			num_inputs = num_inputs - 2 + 8

		init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
							   constant_(x, 0), np.sqrt(2))

		# self.linear_1 = nn.Sequential(
		# 	init_(nn.Linear(num_inputs, dim)), nn.Tanh())

		self.actor = nn.Sequential(
			init_(nn.Linear(num_inputs, dim)), nn.Tanh(),
			init_(nn.Linear(dim, dim)), nn.Tanh())

		self.critic = nn.Sequential(
			init_(nn.Linear(num_inputs, dim)), nn.Tanh(),
			init_(nn.Linear(dim, dim)), nn.Tanh())

		self.distance = nn.Sequential(
			init_(nn.Linear(num_inputs, dim)), nn.Tanh(),
			init_(nn.Linear(dim, 145)))

		self.critic_linear = init_(nn.Linear(dim, 1))

		self.train()

	def forward(self, inputs, rnn_hxs, rnn_hxs_1, masks):
		#x = inputs
		x = inputs[:, :-3]
		y = inputs[:, -1]
		embed_action = self.embed(y.int())
		x = torch.cat((x, embed_action), dim = 1)
		#x = self.linear_1(x)


		# if self.is_attention:
		# 	x = x.unsqueeze(2)
		# 	x, y = self.attention(x, x, x)
		# 	x = x.squeeze()
		#print(x.shape)

		#x = x.squeeze()

		if self.is_recurrent:
			x, rnn_hxs, rnn_hxs_1 = self._forward_gru(x, rnn_hxs, rnn_hxs_1, masks)
			#print(rnn_hxs.shape)
		else:
			#x = self.linear_1(x)
			rnn_hxs = torch.zeros(32, 256)
			rnn_hxs_1 = torch.zeros(32, 256)

		hidden_critic = self.critic(x)
		hidden_actor = self.actor(x)
		dist = self.distance(x)
		#ang = self.ang(x)
		#dist_ahead = dist

		return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, rnn_hxs_1, dist, dist
