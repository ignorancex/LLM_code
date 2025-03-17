import copy
import glob
import os
import time
from collections import deque

#import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from tensorboardX import SummaryWriter
import pdb
import warnings
warnings.filterwarnings("ignore")

def main(save_dir, writer, writer_step, args, actor_critic_predator, actor_critic_prey, gamma):

	envs = make_vec_envs('simple_tag', args.seed, args.num_processes,
						 gamma, save_dir, device, False)


	agent_predator = algo.PPO(
		actor_critic_predator,
		args.clip_param,
		args.ppo_epoch,
		args.num_mini_batch,
		args.value_loss_coef,
		args.entropy_coef,
		lr=args.lr,
		eps=args.eps,
		max_grad_norm=args.max_grad_norm)

	agent_prey = algo.PPO(
		actor_critic_prey,
		args.clip_param,
		args.ppo_epoch,
		args.num_mini_batch,
		args.value_loss_coef,
		args.entropy_coef,
		lr=args.lr,
		eps=args.eps,
		max_grad_norm=args.max_grad_norm)


	obs = envs.reset()
	obs_predator = torch.cat((obs[:,0,:].to(device), torch.zeros(args.num_processes, 1).to(device)), dim = 1)
	obs_prey = torch.cat((obs[:,1,:].to(device), torch.zeros(args.num_processes, 1).to(device)), dim = 1)
	rollouts_predator = RolloutStorage(args.num_steps, args.num_processes,
							  np.array(obs_predator[0].cpu()).shape, envs.action_space,
							  512)

	rollouts_prey = RolloutStorage(args.num_steps, args.num_processes,
							  np.array(obs_prey[0].cpu()).shape, envs.action_space,
							  512)

	rollouts_predator.obs[0].copy_(obs_predator)
	rollouts_predator.to(device)

	rollouts_prey.obs[0].copy_(obs_prey)
	rollouts_prey.to(device)

	episode_rewards = deque(maxlen=10)

	start = time.time()

	total_runs = 0.0
	good_runs = 0.0
	total_pred_pos = 0
	correct_pred_pos = 0
	total_prey_pos = 0
	correct_prey_pos = 0

	curr_step = torch.zeros(args.num_processes, 1)

	done_number = []

	num_updates = 10000

	value_loss_predator_all = 0.0
	action_loss_predator_all = 0.0
	dist_entropy_predator_all = 0.0
	supervised_loss_predator_all = 0.0
	total_supervised_ahead_loss_predator_all = 0.0
	total_loss_predator_all = 0.0
	adv_mean_all = 0.0
	adv_mean_true_all = 0.0

	total_over_80 = 0

	for j in range(num_updates):
		writer_step += 1

		for step in range(args.num_steps):

			# Sample actions
			with torch.no_grad():
				value_predator, action_predator, action_log_prob_predator, recurrent_hidden_states_predator, recurrent_hidden_states_1_predator, distance_bucket_predator, distance_ahead_pred = actor_critic_predator.act(
					rollouts_predator.obs[step].to(device), rollouts_predator.recurrent_hidden_states[step].to(device), rollouts_predator.recurrent_hidden_states_1[step].to(device),
					rollouts_predator.masks[step].to(device))


				value_prey, action_prey, action_log_prob_prey, recurrent_hidden_states_prey, recurrent_hidden_states_1_prey, distance_bucket_prey, distance_ahead_prey = actor_critic_prey.act(
					rollouts_prey.obs[step].to(device), rollouts_prey.recurrent_hidden_states[step].to(device), rollouts_prey.recurrent_hidden_states_1[step].to(device),
					rollouts_prey.masks[step].to(device))

			distance_bucket_predator_num = torch.max(distance_bucket_predator,dim=1)[1].float()
			gt_distance_bucket_predator = obs[:, 0, -2].float()
			gt_distance_bucket_ahead_predator = obs[:, 0, -1].float()
			not_visable = obs[:, 0, 0] == -1

			correct_pred_pos = sum(distance_bucket_predator_num == gt_distance_bucket_predator)
			total_pred_pos = len(distance_bucket_predator_num)
			correct_pred_pos_unseen = sum(distance_bucket_predator_num[not_visable] == gt_distance_bucket_predator[not_visable])
			total_pred_pos_unseen = sum(not_visable)

			distance_bucket_prey_num = torch.max(distance_bucket_prey,dim=1)[1].float()
			gt_distance_bucket_prey = obs[:, 1, -2].float()
			gt_distance_bucket_ahead_prey = obs[:, 1, -1].float()
			correct_prey_pos = sum(distance_bucket_prey_num == gt_distance_bucket_prey)
			total_prey_pos = len(distance_bucket_prey_num)


			current_num = torch.zeros_like(action_predator)
			current_num[:] = j

			actions_all = torch.stack([action_predator, action_predator], dim = 1)
			obs, reward, done, infos = envs.step(actions_all)


			reward_predator = reward[:, :, 0]
			reward_prey = reward[:, :, 1]

			curr_step += torch.ones(args.num_processes, 1)
			if sum(done) != 0:
				done_number = done_number + curr_step[done][0].tolist()
			curr_step[done] = 0

			obs_predator = torch.cat((obs[:,0,:], action_predator), dim=1)
			obs_prey = torch.cat((obs[:,1,:], action_prey), dim=1)

			catch = torch.zeros(reward_predator.shape)
			catch[reward_predator == 5.0] = 1
			good_runs += sum(catch)
			catch[reward_predator == -1.0] = 1
			total_runs += sum(catch)


			for info in infos:
				if 'episode' in info.keys():
					episode_rewards.append(info['episode']['r'])

			# If done then clean the history of observations.
			masks = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = torch.FloatTensor(
				[[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])
			rollouts_predator.insert(obs_predator, recurrent_hidden_states_predator, recurrent_hidden_states_1_predator, action_predator,
							action_log_prob_predator, value_predator, reward_predator, masks, bad_masks, gt_distance_bucket_ahead_predator.view(-1, 1))

			rollouts_prey.insert(obs_prey, recurrent_hidden_states_prey, recurrent_hidden_states_1_prey, action_prey,
							action_log_prob_prey, value_prey, reward_prey, masks, bad_masks, gt_distance_bucket_ahead_prey.view(-1, 1))

		with torch.no_grad():
			next_value_predator = actor_critic_predator.get_value(
				rollouts_predator.obs[-1], rollouts_predator.recurrent_hidden_states[-1], rollouts_predator.recurrent_hidden_states_1[-1],
				rollouts_predator.masks[-1]).detach()

			next_value_prey = actor_critic_prey.get_value(
				rollouts_prey.obs[-1], rollouts_prey.recurrent_hidden_states[-1], rollouts_prey.recurrent_hidden_states_1[-1],
				rollouts_prey.masks[-1]).detach()


		rollouts_predator.compute_returns(next_value_predator, args.use_gae, gamma, args.gae_lambda, False)

		rollouts_prey.compute_returns(next_value_prey, args.use_gae, .9, args.gae_lambda, False)


		value_loss_predator, action_loss_predator, dist_entropy_predator, supervised_loss_predator, total_supervised_ahead_loss_predator, total_loss_predator, adv_mean, adv_true_mean, ratio = agent_predator.update(rollouts_predator, j)
		value_loss_prey, action_loss_prey, dist_entropy_prey, supervised_loss_prey, total_supervised_ahead_loss_prey, total_loss_prey, adv_mean_pred, adv_true_mean_pred, ratio_pred = agent_prey.update(rollouts_prey, j)

		value_loss_predator_all += value_loss_predator
		action_loss_predator_all += action_loss_predator
		dist_entropy_predator_all += dist_entropy_predator
		supervised_loss_predator_all += supervised_loss_predator
		total_supervised_ahead_loss_predator_all += total_supervised_ahead_loss_predator
		total_loss_predator_all += total_loss_predator
		adv_mean_all += adv_mean
		adv_mean_true_all += adv_true_mean

		#pdb.set_trace()
		rollouts_predator.after_update()
		rollouts_prey.after_update()

		# save for every interval-th episode or for the last epoch
		if (j % args.save_interval == 0 or j == num_updates - 1) and save_dir != "":
			torch.save(actor_critic_predator.state_dict(), os.path.join(save_dir, "predator_" + str(writer_step) + ".pt"))
			torch.save(actor_critic_prey.state_dict(), os.path.join(save_dir, "prey_" + str(writer_step) + ".pt"))

		if j % args.log_interval == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * args.num_processes * args.num_steps
			end = time.time()
			writer.add_scalar('reward', float(np.mean(episode_rewards)), writer_step)
			writer.add_scalar('catch_ratio', (good_runs)/total_runs, writer_step)
			writer.add_scalar('avg_ep_length', sum(done_number)/max(len(done_number), 1), writer_step)
			writer.add_scalar('predator_location_acc/all', correct_pred_pos/total_pred_pos, writer_step)
			writer.add_scalar('predator_location_acc/unseens', correct_pred_pos_unseen/total_pred_pos_unseen, writer_step)
			writer.add_scalar('prey_location_acc', correct_prey_pos/total_prey_pos, writer_step)
			print(total_runs)


			writer.add_scalar('predator_loss/value_loss', (value_loss_predator_all)/args.log_interval, writer_step)
			writer.add_scalar('predator_loss/action_loss', (action_loss_predator_all)/args.log_interval, writer_step)
			writer.add_scalar('predator_loss/dist_entropy', (dist_entropy_predator_all)/args.log_interval, writer_step)
			writer.add_scalar('predator_loss/supervised_loss', (supervised_loss_predator_all)/args.log_interval, writer_step)
			writer.add_scalar('predator_loss/supervised_ahead_loss', (total_supervised_ahead_loss_predator_all)/args.log_interval, writer_step)
			writer.add_scalar('predator_loss/total_loss', (total_loss_predator_all)/args.log_interval, writer_step)
			writer.add_scalar('predator_loss/advantages', adv_mean_all/args.log_interval, writer_step)
			writer.add_scalar('predator_loss/advantages_true', adv_mean_true_all/args.log_interval, writer_step)


			value_loss_predator_all = 0.0
			action_loss_predator_all = 0.0
			dist_entropy_predator_all = 0.0
			supervised_loss_predator_all = 0.0
			total_supervised_ahead_loss_predator_all = 0.0
			total_loss_predator_all = 0.0
			adv_mean_all = 0.0
			adv_mean_true_all = 0.0

			done_number = []

			total_pred_pos = 0
			correct_pred_pos = 0
			total_prey_pos = 0
			correct_prey_pos = 0
			total_runs = 0.0
			good_runs = 0.0
			print(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
				.format(j, total_num_steps,
						int(total_num_steps / (end - start)),
						len(episode_rewards), np.mean(episode_rewards),
						np.median(episode_rewards), np.min(episode_rewards),
						np.max(episode_rewards), dist_entropy_predator, value_loss_predator,
						action_loss_predator))

	envs.close()


if __name__ == "__main__":

	args = get_args()
	save_dir = './log/planning_' + str(args.planning) + '_vision_' + str(args.vision) + "_speed_" + str(args.speed) + "_0"
	writer = SummaryWriter(save_dir)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = False

	torch.set_num_threads(1)
	device = torch.device("cuda:0" if args.cuda else "cpu")


	if args.planning == 'low':
		gamma = 0.9
	elif args.planning == 'mid':
		gamma = 0.93
	elif args.planning == 'high':
		gamma = 0.99

	envs = make_vec_envs('simple_tag', args.seed, args.num_processes,
						 gamma, save_dir, device, False)

	actor_critic_predator = Policy(
		envs.observation_space.shape,
		envs.action_space, False, 512,
		base_kwargs={'recurrent': True})
	actor_critic_predator.to(device)


	print(envs.observation_space.shape)
	print(envs.action_space)

	actor_critic_prey = Policy(
		envs.observation_space.shape,
		envs.action_space, False, 512,
		base_kwargs={'recurrent': True})
	actor_critic_prey.to(device)

	writer_step = 0
	actor_critic_predator, writer_step = main(save_dir, writer, writer_step, args, actor_critic_predator, actor_critic_prey, gamma)
