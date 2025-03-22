import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
from vars import pargs


class PPO():
	def __init__(self,
				 actor_critic,
				 clip_param,
				 ppo_epoch,
				 num_mini_batch,
				 value_loss_coef,
				 entropy_coef,
				 lr=None,
				 eps=None,
				 max_grad_norm=None,
				 use_clipped_value_loss=True):

		self.actor_critic = actor_critic

		self.clip_param = clip_param
		self.ppo_epoch = ppo_epoch
		self.num_mini_batch = num_mini_batch

		self.value_loss_coef = value_loss_coef
		self.entropy_coef = entropy_coef

		self.max_grad_norm = max_grad_norm
		self.use_clipped_value_loss = False

		self.ce_loss_dist = nn.CrossEntropyLoss()
		self.ce_loss_dist_ahead = nn.CrossEntropyLoss()

		self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
		self.lr = lr

		self.good_catch = False
		self.first = True

	def update(self, rollouts, j):

		lr = self.lr - (self.lr * (j / 10000.0))
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

		advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
		#advantages = rollouts.returns[:-1]
		# pdb.set_trace()
		adv_mean = rollouts.returns[:-1].mean()
		adv_true_mean = (rollouts.returns[:-1] - rollouts.value_preds[:-1]).mean()
		advantages = (advantages - advantages.mean()) / (
			advantages.std() + 1e-5)

		value_loss_epoch = 0
		action_loss_epoch = 0
		dist_entropy_epoch = 0
		total_supervised_loss = 0
		total_supervised_ahead_loss = 0
		total_loss = 0

		for e in range(1):
			if self.actor_critic.is_recurrent:
				data_generator = rollouts.recurrent_generator(
					advantages, self.num_mini_batch)
			else:
				data_generator = rollouts.feed_forward_generator(
					advantages, self.num_mini_batch)



			for sample in data_generator:
				obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_1_batch, actions_batch, \
				   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
						adv_targ, locations = sample

				#pdb.set_trace()
				#print(obs_batch.shape)

				# Reshape to do in a single forward pass for all steps
				#print("eval")
				values, action_log_probs, dist_entropy, _, _, distance, dist_ahead = self.actor_critic.evaluate_actions(
					obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_1_batch, masks_batch,
					actions_batch)


				supervised_loss_ahead = torch.tensor([0])
				distance_gt = obs_batch[:, -3].long()
				supervised_loss = self.ce_loss_dist(distance ,distance_gt)
				#pdb.set_trace()


				ratio = torch.exp(action_log_probs -
								  old_action_log_probs_batch)
				surr1 = ratio * adv_targ
				surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
									1.0 + self.clip_param) * adv_targ
				action_loss = -torch.min(surr1, surr2).mean()

				if self.use_clipped_value_loss:
					value_pred_clipped = value_preds_batch + \
						(values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
					value_losses = (values - return_batch).pow(2)
					value_losses_clipped = (
						value_pred_clipped - return_batch).pow(2)
					value_loss = 0.5 * torch.max(value_losses,
												 value_losses_clipped).mean()
				else:
					value_loss = 0.5 * (return_batch - values).pow(2).mean()

				self.optimizer.zero_grad()
				(value_loss * self.value_loss_coef + action_loss -
				 dist_entropy * self.entropy_coef + supervised_loss * 1.0).backward()
				nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
										 self.max_grad_norm)
				self.optimizer.step()

				value_loss_epoch += value_loss.item() * self.value_loss_coef
				action_loss_epoch += action_loss.item()
				dist_entropy_epoch += dist_entropy.item() * self.entropy_coef
				total_supervised_loss += supervised_loss.item() * 1.0
				total_supervised_ahead_loss += supervised_loss_ahead.item()
				total_loss += (value_loss * self.value_loss_coef + action_loss -
				 dist_entropy * self.entropy_coef + supervised_loss).item()


		num_updates = self.ppo_epoch * self.num_mini_batch

		value_loss_epoch /= num_updates
		action_loss_epoch /= num_updates
		dist_entropy_epoch /= num_updates
		total_supervised_loss /= num_updates
		total_loss /= num_updates

		return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, total_supervised_loss, total_supervised_ahead_loss, total_loss, adv_mean, adv_true_mean, ratio
