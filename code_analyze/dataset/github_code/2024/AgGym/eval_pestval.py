from modules import env_modules_pestval
from utils import general_utils as gu
from configobj import ConfigObj

import pfrl
import torch
import torch.nn
import gym
import numpy as np
from einops import rearrange
from torch import nn
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--seed', type=int, default=123, help='Description for foo argument')
args = parser.parse_args()

class QFunction(torch.nn.Module):

	def __init__(self, obs_size, n_actions):
		super().__init__()
		self.l1 = torch.nn.Linear(obs_size, 50)
		self.l2 = torch.nn.Linear(50, 50)
		self.l3 = torch.nn.Linear(50, n_actions)

	def forward(self, x):
		h = rearrange(x, "b h w -> b (h w)")
		h = torch.nn.functional.relu(self.l1(h))
		h = torch.nn.functional.relu(self.l2(h))
		h = self.l3(h)
		return pfrl.action_value.DiscreteActionValue(h)

gu.seed_everything(args.seed)
# 11111
config = ConfigObj('eval_config.ini')
env = env_modules_pestval.cartesian_grid()
gu.set_as_attr(env, config)
env.init()
env.rl_init()

obs_size = env.state_space
n_actions = env.action_space.n
env.gpu = int(env.gpu)
q_func = QFunction(obs_size, n_actions)
if env.model == 'ddqn':
    q_func = QFunction(obs_size, n_actions)

    optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
    gamma = 0.99

    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=0.95, end_epsilon=0.1, decay_steps=350000, random_action_func=env.action_space.sample)

    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)
    gpu = env.gpu
	
    agent = pfrl.agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        replay_start_size=20000,
        update_interval=130,
        target_update_interval=1300,
        phi=phi,
        gpu=gpu,
    )
elif env.model == 'ppo':
    def lecun_init(layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        else:
            pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
            pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
        return layer
    
    model = nn.Sequential(
            lecun_init(nn.Linear(obs_size, 128)),
            nn.ReLU(),
            lecun_init(nn.Linear(128, 64)),
            nn.ReLU(),
            lecun_init(nn.Linear(64, 32)),
            nn.ReLU(),
            pfrl.nn.Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(32, n_actions), 1e-2),
                    pfrl.policies.SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(32, 1)),
            ),
        )
    opt = torch.optim.RMSprop(model.parameters(),lr=0.0007477847290354089)
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=env.gpu,
        phi=phi,
        update_interval=4000,
        minibatch_size=512,
        epochs=4,
        clip_eps=0.2837622690867982,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=0.2,
        recurrent=False,
        max_grad_norm=0.5,
    )
elif env.model == 'trpo':
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_actions),
        pfrl.policies.SoftmaxCategoricalHead(),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )
    
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1e-2)
    
    vf_opt = torch.optim.Adam(vf.parameters())
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = pfrl.agents.TRPO(
        policy=policy,
        vf=vf,
        phi=phi,
        vf_optimizer=vf_opt,
        obs_normalizer=None,
        gpu=env.gpu,
        update_interval=5000,
        max_kl=0.01,
        conjugate_gradient_max_iter=20,
        conjugate_gradient_damping=1e-1,
        gamma=0.995,
        lambd=0.97,
        vf_epochs=10,
        entropy_coef=0,
    )

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
gamma = 0.99
explorer = pfrl.explorers.ConstantEpsilonGreedy(
	epsilon=0.3, random_action_func=env.action_space.sample)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
gpu = 0
# agent = pfrl.agents.DoubleDQN(
# 	q_func,
# 	optimizer,
# 	replay_buffer,
# 	gamma,
# 	explorer,
# 	replay_start_size=100000,
# 	update_interval=130,
# 	target_update_interval=1300,
# 	phi=phi,
# 	gpu=gpu,
# )


max_episode = config["eval"].as_int("max_episode")

agent.load(config['general']['result_path'] + '/agent_weights/'+config["agent"]["best_agent"])

with agent.eval_mode():
        for ep in range(max_episode):
            obs = env.reset()
            R = 0
            i=0
            while True: # for i in range(env.max_timestep): 
                #while true:
                obs=np.reshape(obs, (1, 41))
                action = agent.act(obs)
                action=int(action)
                obs, reward, done = env.step(action)
                R += reward
                agent.observe(obs, reward, done, done)
                print(f"Step: {i}, Action: {action} Reward: {reward}, Done: {done}")
                i+=1
                if done == True:
                    print(f"dead_count={env.dead_counts[-1]}")
                    print(f"infect_count={env.infect_counts[-1]}")
                    break
            print(f"Evaluation Episode: {ep}, Reward: {R}")

