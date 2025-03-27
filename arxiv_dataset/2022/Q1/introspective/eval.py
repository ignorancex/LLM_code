import argparse
import os
# workaround to unpickle olf model files
import sys
from multiagent_envs.make_env import make_env as make_make_multi_env
import pdb
import numpy as np
import torch
from a2c_ppo_acktr.model import Policy
import time
import random
from a2c_ppo_acktr.arguments import get_args

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
sys.path.append('a2c_ppo_acktr')

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.set_num_threads(1)
np.random.seed(seed=1)
random.seed(1)

args = get_args()

actor_critic_predator = Policy([14], 0, False, 512, base_kwargs={'recurrent': True})
actor_critic_predator.to('cpu')
actor_critic_predator.load_state_dict(torch.load(args.predator_weights, map_location='cpu'))
actor_critic_predator.eval()


actor_critic_prey = Policy([14], 0, False, 512, base_kwargs={'recurrent': True})
actor_critic_prey.to('cpu')
actor_critic_prey.load_state_dict(torch.load(args.prey_weights, map_location='cpu'))
actor_critic_prey.eval()


device = torch.device("cpu")
envs = make_make_multi_env('simple_tag')

hidden_size_curr = actor_critic_predator.recurrent_hidden_state_size
hidden_size_curr = 512

recurrent_hidden_states_predator = torch.zeros(1, 512)
recurrent_hidden_states_1_predator = torch.zeros(1, 512)

recurrent_hidden_states_prey = torch.zeros(1, hidden_size_curr)
recurrent_hidden_states_1_prey = torch.zeros(1, hidden_size_curr)

masks = torch.ones(1, 1)

obs = envs.reset()
obs_predator = torch.tensor(np.append(obs[0],0)).unsqueeze(0).float()
obs_prey = torch.tensor(np.append(obs[1],0)).unsqueeze(0).float()

catch = 0.0
length = 0
total_len = 0
total = 0
step = 0


while total < 10000:
	with torch.no_grad():
		value_predator, action_predator, _, recurrent_hidden_states_predator, recurrent_hidden_states_1_predator, distance_predator, dist_ahead = actor_critic_predator.act(
			obs_predator, recurrent_hidden_states_predator, recurrent_hidden_states_1_predator, masks, deterministic=False)

		value_prey, action_prey, _, recurrent_hidden_states_prey, recurrent_hidden_states_1_prey, distance_prey, dist_ahead = actor_critic_prey.act(
			obs_prey, recurrent_hidden_states_prey, recurrent_hidden_states_1_prey, masks, deterministic=False)

		if args.video:
			_, _ = envs.render()
			seeing = []
			time.sleep(.001)

		# Obser reward and next obs
		actions_all = torch.stack([action_predator, action_prey], dim = 1).squeeze()
		obs, reward, done, _ = envs.step(actions_all)

		obs_predator = torch.tensor(np.append(obs[0], action_predator[0][0])).unsqueeze(0).float()
		obs_prey = torch.tensor(np.append(obs[1], action_prey[0][0])).unsqueeze(0).float()

		if reward[0] == 5.0:
			catch += 1

		step += 1
		total += 1

		masks = torch.ones(1, 1)
		if done:
			length += step
			total_len += 1
			step = 0
			masks = torch.zeros(1, 1)
			obs = envs.reset()
			obs_predator = torch.tensor(np.append(obs[0], action_predator[0][0])).unsqueeze(0).float()
			obs_prey = torch.tensor(np.append(obs[1], action_prey[0][0])).unsqueeze(0).float()
			recurrent_hidden_states_prey = torch.zeros(1, actor_critic_prey.recurrent_hidden_state_size)
			recurrent_hidden_states_prey_1 = torch.zeros(1, actor_critic_prey.recurrent_hidden_state_size)
			recurrent_hidden_states_predator = torch.zeros(1, actor_critic_prey.recurrent_hidden_state_size)
			recurrent_hidden_states_predator_1 = torch.zeros(1, actor_critic_prey.recurrent_hidden_state_size)
			done = False

print("Number of prey caught in 10,000 steps is " + str(int(catch)))
