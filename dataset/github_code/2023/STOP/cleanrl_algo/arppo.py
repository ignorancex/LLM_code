# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
import pdb

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from utils import plot_heatmap

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env, use_action_mask = False):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )
        self.env = env
        self.use_action_mask = use_action_mask

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if x.ndim == 1:
            x = x.view(1, x.shape[0])
        logits = self.actor(x)
        if self.use_action_mask:
            act_mask = torch.Tensor(self.env.mask_extractor(x.numpy()))
            # apply mask
            logits = logits.masked_fill(act_mask == 1, value = -1e8)

        probs = Categorical(logits=logits)
        prob_dist = probs.probs
        if action is None:
            action = probs.sample()
            if action.ndim == 1:
                action = action[0]
                prob_dist = prob_dist[0]
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), prob_dist

class ARPPO:
    def __init__(self, env,
                num_envs = 1,
                num_minibatches = 4,
                num_steps = 256,
                learning_rate = 3e-4,
                anneal_lr = False,
                update_epochs = 10,
                clip_coef = 0.2,
                clip_range_vf = None,
                clip_vloss = False,
                ent_coef = 0.0, 
                vf_coef = 0.5,
                max_grad_norm = 0.5,
                target_kl = None,
                variant = 'zhang',
                gamma = 0.99,
                gae_lambda = 0.95,
                norm_adv = True,
                use_action_mask = False,
                adam_beta = 0.9):

        self.env = env
        self.agent = Agent(env, use_action_mask = use_action_mask)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.agent.parameters(),\
            lr=learning_rate, eps = 1e-8,\
            betas = (adam_beta, adam_beta))
        self.anneal_lr = anneal_lr
        self.num_steps = num_steps
        self.batch_size = num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_envs = 1

        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv

        self.variant = variant
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def train(self, total_timesteps = 100_000):

        print_freq = self.round_to_multiple(10_000, self.batch_size)
        backlog = []
        time = []

        # setup
        # ALGO Logic: Storage setup
        obs = torch.zeros((self.num_steps, self.num_envs) + self.env.observation_space.shape)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.action_space.shape)
        logprobs = torch.zeros((self.num_steps, self.num_envs))
        rewards = torch.zeros((self.num_steps, self.num_envs))
        dones = torch.zeros((self.num_steps, self.num_envs))
        values = torch.zeros((self.num_steps, self.num_envs))

        self.num_iterations = total_timesteps // self.batch_size

        # TRY NOT TO MODIFY: start the game
        next_obs, _ = self.env.reset()
        next_obs = torch.Tensor(next_obs)
        next_done = torch.zeros(self.num_envs)
        rew_running_mean = torch.zeros(self.num_envs)
        rew_tau = 1.
        #rew_tau = nn.Parameter(torch.tensor(0.), requires_grad = True)
        #rew_tau_optimizer = torch.optim.Adam([rew_tau], lr = 1e-4)

        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, _ = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = reward#torch.tensor(reward).view(-1)
                next_obs, next_done = torch.Tensor(next_obs), torch.Tensor([next_done])
                backlog.append(infos['backlog'])
                time.append(infos['time'])

            rew_running_mean = (1 - rew_tau) * rew_running_mean + rew_tau * torch.mean(rewards, axis = 0)
            mean_rew = rew_running_mean
            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(-1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]

                    if self.variant == 'zhang':
                        sub_diff = rewards[t] - mean_rew
                        # average reward
                        target = sub_diff + nextvalues * nextnonterminal
                        advantages[t] = target - values[t]
                    elif self.variant == 'discounted':
                        delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        #mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        mb_advantages = (mb_advantages) / (mb_advantages.std() + 1e-8)


                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_range_vf,
                            self.clip_range_vf,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            if time[-1] % print_freq == 0:
                denom = np.arange(1, len(backlog) + 1)
                avg_backlog = np.divide(np.cumsum(backlog), denom)
                print (avg_backlog)
        
        self.time = time
        self.backlog = backlog

    def round_to_multiple(self, number, multiple):
        quotient = number / multiple
        rounded_quotient = round(quotient)
        rounded_number = rounded_quotient * multiple
        return rounded_number
    
    def get_stats(self):
        stats = {
            'backlog': self.backlog,
            'time': self.time
        }
        return stats
