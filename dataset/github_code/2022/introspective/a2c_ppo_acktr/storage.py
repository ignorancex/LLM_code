import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pdb

def _flatten_helper(T, N, _tensor):
	return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
	def __init__(self, num_steps, num_processes, obs_shape, action_space,
				 recurrent_hidden_state_size):
		self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
		self.location = torch.zeros(num_steps + 1, num_processes, 1)
		self.recurrent_hidden_states = torch.zeros(
			num_steps + 1, num_processes, recurrent_hidden_state_size)
		self.recurrent_hidden_states_1 = torch.zeros(
			num_steps + 1, num_processes, recurrent_hidden_state_size)
		self.rewards = torch.zeros(num_steps, num_processes, 1)
		self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
		self.returns = torch.zeros(num_steps + 1, num_processes, 1)
		self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
		action_shape = 1
		self.actions = torch.zeros(num_steps, num_processes, action_shape)
		self.actions = self.actions.long()
		self.masks = torch.ones(num_steps + 1, num_processes, 1)

		# Masks that indicate whether it's a true terminal state
		# or time limit end state
		self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

		self.num_steps = num_steps
		self.step = 0

	def to(self, device):
		self.obs = self.obs.to(device)
		self.location = self.location.to(device)
		self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
		self.recurrent_hidden_states_1 = self.recurrent_hidden_states_1.to(device)
		self.rewards = self.rewards.to(device)
		self.value_preds = self.value_preds.to(device)
		self.returns = self.returns.to(device)
		self.action_log_probs = self.action_log_probs.to(device)
		self.actions = self.actions.to(device)
		self.masks = self.masks.to(device)
		self.bad_masks = self.bad_masks.to(device)

	def insert(self, obs, recurrent_hidden_states, recurrent_hidden_states_1, actions, action_log_probs,
			   value_preds, rewards, masks, bad_masks, location):
		self.obs[self.step + 1].copy_(obs)
		self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
		self.recurrent_hidden_states_1[self.step + 1].copy_(recurrent_hidden_states_1)
		self.actions[self.step].copy_(actions)
		self.action_log_probs[self.step].copy_(action_log_probs)
		self.value_preds[self.step].copy_(value_preds)
		self.rewards[self.step].copy_(rewards)
		self.masks[self.step + 1].copy_(masks)
		self.bad_masks[self.step + 1].copy_(bad_masks)
		self.location[self.step + 1].copy_(location)
		self.step = (self.step + 1) % self.num_steps

	def after_update(self):
		self.obs[0].copy_(self.obs[-1])
		self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
		self.recurrent_hidden_states_1[0].copy_(self.recurrent_hidden_states_1[-1])
		self.masks[0].copy_(self.masks[-1])
		self.bad_masks[0].copy_(self.bad_masks[-1])
		self.location[0].copy_(self.location[-1])

	def compute_returns(self,
						next_value,
						use_gae,
						gamma,
						gae_lambda,
						use_proper_time_limits=True):
		if use_proper_time_limits:
			if use_gae:
				self.value_preds[-1] = next_value
				gae = 0
				for step in reversed(range(self.rewards.size(0))):
					delta = self.rewards[step] + gamma * self.value_preds[
						step + 1] * self.masks[step +
											   1] - self.value_preds[step]
					gae = delta + gamma * gae_lambda * self.masks[step +
																  1] * gae
					gae = gae * self.bad_masks[step + 1]
					self.returns[step] = gae + self.value_preds[step]
			else:
				self.returns[-1] = next_value
				for step in reversed(range(self.rewards.size(0))):
					self.returns[step] = (self.returns[step + 1] * \
						gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
						+ (1 - self.bad_masks[step + 1]) * self.value_preds[step]
		else:
			if use_gae:
				self.value_preds[-1] = next_value
				gae = 0
				for step in reversed(range(self.rewards.size(0))):
					delta = self.rewards[step] + gamma * self.value_preds[
						step + 1] * self.masks[step +
											   1] - self.value_preds[step]
					gae = delta + gamma * gae_lambda * self.masks[step +
																  1] * gae
					self.returns[step] = gae + self.value_preds[step]
			else:
				#pdb.set_trace()
				self.returns[-1] = next_value
				#pdb.set_trace()
				for st in reversed(range(self.rewards.size(0))):


					#self.returns[step] = sum(self.rewards[step:step+5])

					self.returns[st] = self.returns[st + 1] * gamma * self.masks[st + 1] + self.rewards[st]
					#self.returns[st] = self.returns[st + 1] * self.masks[st + 1] + self.rewards[st]

	def feed_forward_generator(self,
							   advantages,
							   num_mini_batch=None,
							   mini_batch_size=None):
		num_steps, num_processes = self.rewards.size()[0:2]
		batch_size = num_processes * num_steps

		if mini_batch_size is None:
			assert batch_size >= num_mini_batch, (
				"PPO requires the number of processes ({}) "
				"* number of steps ({}) = {} "
				"to be greater than or equal to the number of PPO mini batches ({})."
				"".format(num_processes, num_steps, num_processes * num_steps,
						  num_mini_batch))
			mini_batch_size = batch_size // num_mini_batch
		sampler = BatchSampler(
			SubsetRandomSampler(range(batch_size)),
			mini_batch_size,
			drop_last=True)
		for indices in sampler:
			obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
			recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1, self.recurrent_hidden_states.size(-1))[indices]
			recurrent_hidden_states_1_batch = self.recurrent_hidden_states_1[:-1].view(-1, self.recurrent_hidden_states_1.size(-1))[indices]
			actions_batch = self.actions.view(-1,
											  self.actions.size(-1))[indices]
			value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
			return_batch = self.returns[:-1].view(-1, 1)[indices]
			masks_batch = self.masks[:-1].view(-1, 1)[indices]
			locations = self.masks[:-1].view(-1, 1)[indices]
			old_action_log_probs_batch = self.action_log_probs.view(-1,
																	1)[indices]
			if advantages is None:
				adv_targ = None
			else:
				adv_targ = advantages.view(-1, 1)[indices]

			yield obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_1_batch, actions_batch, \
				value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, 0

	def recurrent_generator(self, advantages, num_mini_batch):
		num_processes = self.rewards.size(1)
		assert num_processes >= num_mini_batch, (
			"PPO requires the number of processes ({}) "
			"to be greater than or equal to the number of "
			"PPO mini batches ({}).".format(num_processes, num_mini_batch))
		num_envs_per_batch = num_processes // num_mini_batch
		perm = torch.randperm(num_processes)
		for start_ind in range(0, num_processes, num_envs_per_batch):
			obs_batch = []
			recurrent_hidden_states_batch = []
			recurrent_hidden_states_1_batch = []
			actions_batch = []
			value_preds_batch = []
			return_batch = []
			masks_batch = []
			old_action_log_probs_batch = []
			adv_targ = []
			locations_batch = []

			for offset in range(num_envs_per_batch):
				#pdb.set_trace()
				ind = perm[start_ind + offset]
				#print(ind)
				obs_batch.append(self.obs[:-1, ind])
				recurrent_hidden_states_batch.append(
					self.recurrent_hidden_states[0:1, ind])
				recurrent_hidden_states_1_batch.append(
					self.recurrent_hidden_states_1[0:1, ind])
				actions_batch.append(self.actions[:, ind])
				value_preds_batch.append(self.value_preds[:-1, ind])
				return_batch.append(self.returns[:-1, ind])
				masks_batch.append(self.masks[:-1, ind])
				old_action_log_probs_batch.append(
					self.action_log_probs[:, ind])
				adv_targ.append(advantages[:, ind])
				locations_batch.append(self.location[:-1, ind])

			T, N = self.num_steps, num_envs_per_batch
			# These are all tensors of size (T, N, -1)
			obs_batch = torch.stack(obs_batch, 1)
			actions_batch = torch.stack(actions_batch, 1)
			value_preds_batch = torch.stack(value_preds_batch, 1)
			return_batch = torch.stack(return_batch, 1)
			masks_batch = torch.stack(masks_batch, 1)
			locations_batch = torch.stack(locations_batch, 1)
			old_action_log_probs_batch = torch.stack(
				old_action_log_probs_batch, 1)
			adv_targ = torch.stack(adv_targ, 1)

			# States is just a (N, -1) tensor
			recurrent_hidden_states_batch = torch.stack(
				recurrent_hidden_states_batch, 1).view(N, -1)
			recurrent_hidden_states_1_batch = torch.stack(
				recurrent_hidden_states_1_batch, 1).view(N, -1)

			# Flatten the (T, N, ...) tensors to (T * N, ...)
			obs_batch = _flatten_helper(T, N, obs_batch)
			actions_batch = _flatten_helper(T, N, actions_batch)
			value_preds_batch = _flatten_helper(T, N, value_preds_batch)
			return_batch = _flatten_helper(T, N, return_batch)
			masks_batch = _flatten_helper(T, N, masks_batch)
			#locations_batch = _flatten_helper(T, N, locations_batch)
			old_action_log_probs_batch = _flatten_helper(T, N, \
					old_action_log_probs_batch)
			adv_targ = _flatten_helper(T, N, adv_targ)

			yield obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_1_batch, actions_batch, \
				value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, locations_batch
