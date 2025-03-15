"""Multi-objective Soft Actor-Critic (SAC) algorithm for discrete action spaces.

It implements a multi-objective critic with weighted sum scalarization.
The implementation of this file is largely based on CleanRL's SAC implementation
https://github.com/vwxyzjn/cleanrl/blob/28fd178ca182bd83c75ed0d49d52e235ca6cdc88/cleanrl/sac_atari.py
"""

import os
import time
from copy import deepcopy
from typing import Optional, Tuple, Union, List
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import wandb

from mo_utils.buffer import ReplayBuffer
from mo_utils.weights import equally_spaced_weights
from mo_utils.evaluation import (
    log_all_multi_policy_metrics,
    policy_evaluation_mo,
)
from mo_utils.morl_algorithm import MOAgent
from mo_utils.networks import (
    NatureCNN,
    get_grad_norm,
    layer_init,
    mlp,
    polyak_update,
)
from morl_generalization.generalization_evaluator import MORLGeneralizationEvaluator


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    """Soft Q-network: S, A -> ... -> R (single-objective)."""

    def __init__(self, obs_shape, action_dim, net_arch):
        """"Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            reward_dim: number of objectives
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        if len(obs_shape) == 1:
            self.feature_extractor = mlp(obs_shape[0], -1, net_arch[:1])
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=net_arch[0])
        # S, A -> ... -> |A| * R
        self.net = mlp(net_arch[0], action_dim, net_arch[1:])
        self.apply(layer_init)

    def forward(self, obs):
        """Predict Q values for all actions."""
        input = self.feature_extractor(obs)
        q_values = self.net(input)
        return q_values



class Actor(nn.Module):
    """Actor network: S -> A."""

    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        net_arch=[256, 256],
    ):
        """Initialize SAC actor."""
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.net_arch = net_arch

        if len(obs_shape) == 1:
            self.feature_extractor = mlp(obs_shape[0], -1, net_arch[:1])
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=net_arch[0])

        self.net = mlp(net_arch[0], action_dim, net_arch[1:])
        self.apply(layer_init)

    def forward(self, x):
        """Forward pass of the actor network."""
        input = self.feature_extractor(x)
        logits = self.net(input)

        return logits

    def get_action(self, x):
        """Get action from the actor network."""
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, log_prob, action_probs


class SACDiscrete(MOAgent):
    """Soft Actor-Critic (SAC) algorithm for discrete action spaces."""

    def __init__(
        self,
        env: gym.Env,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 1.0,
        batch_size: int = 128,
        learning_starts: int = int(2e4),
        net_arch=[256, 256],
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        update_frequency: int = 4,
        target_net_freq: int = 2000,
        alpha: float = 0.2,
        autotune: bool = True,
        target_entropy_scale: float = 0.89,
        project_name: str = "MORL-Generalization",
        experiment_name: str = "SAC Discrete Action",
        wandb_entity: Optional[str] = None,
        wandb_group: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        offline_mode: bool = False,
        id: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        log: bool = True,
        seed: int = 42,
        parent_rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the MOSAC algorithm.

        Args:
            env: Env
            weights: weights for the scalarization
            scalarization: scalarization function
            buffer_size: buffer size
            gamma: discount factor
            tau: target smoothing coefficient (polyak update)
            batch_size: batch size
            learning_starts: how many steps to collect before triggering the learning
            net_arch: number of nodes in the hidden layers
            policy_lr: learning rate of the policy
            q_lr: learning rate of the q networks
            policy_freq: the frequency of training policy (delayed)
            target_net_freq: the frequency of updates for the target networks
            alpha: Entropy regularization coefficient
            autotune: automatic tuning of alpha
            target_entropy_scale: coefficient for scaling the autotune entropy target
            wandb_entity: The entity to use for logging.
            wandb_group: The wandb group to use for logging.
            wandb_tags: Extra wandb tags to use for experiment versioning.
            offline_mode: Whether to run wandb in offline mode.
            id: id of the SAC policy, for multi-policy algos
            device: torch device
            torch_deterministic: whether to use deterministic version of pytorch
            log: logging activated or not
            seed: seed for the random generators
            parent_rng: parent random generator, for multi-policy algos
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        self.id = id
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device
        self.global_step = 0
        
        # Seeding
        self.seed = seed
        self.parent_rng = parent_rng
        if parent_rng is not None:
            self.np_random = parent_rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # env setup
        self.env = env
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        self.obs_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        self.reward_dim = self.env.unwrapped.reward_space.shape[0]
        self.batch_size = batch_size

        # SAC Parameters
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.net_arch = net_arch
        self.policy_lr = policy_lr
        self.learning_rate = policy_lr
        self.q_lr = q_lr
        self.update_frequency = update_frequency
        self.target_net_freq = target_net_freq
        assert self.target_net_freq % self.update_frequency == 0, "target_net_freq should be divisible by update_frequency"
        self.target_entropy_scale = target_entropy_scale

        # Networks
        self.actor = Actor(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            net_arch=self.net_arch,
        ).to(self.device)

        self.qf1 = SoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf2 = SoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf1_target = SoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf2_target = SoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf1_target.requires_grad_(False)
        self.qf2_target.requires_grad_(False)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr, eps=1e-4)

        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -self.target_entropy_scale * th.log(1 / th.tensor(self.action_dim))
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr, eps=1e-4)
        else:
            self.alpha = alpha
        self.alpha_tensor = th.scalar_tensor(self.alpha).to(self.device)

        # Buffer
        self.env.observation_space.dtype = np.float32
        self.buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=1, # ouput singular index for action
            rew_dim=1,
            max_size=self.buffer_size,
        )

        # Logging
        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity, wandb_group, wandb_tags, offline_mode)

    def get_config(self) -> dict:
        """Returns the configuration of the policy."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "net_arch": self.net_arch,
            "policy_lr": self.policy_lr,
            "q_lr": self.q_lr,
            "update_freq": self.update_frequency,
            "target_net_freq": self.target_net_freq,
            "alpha": self.alpha,
            "autotune": self.autotune,
            "target_entropy_scale": self.target_entropy_scale,
            "seed": self.seed,
        }

    def __deepcopy__(self, memo):
        """Deep copy of the policy.

        Args:
            memo (dict): memoization dict
        """
        copied = type(self)(
            env=self.env,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            tau=self.tau,
            batch_size=self.batch_size,
            learning_starts=self.learning_starts,
            net_arch=self.net_arch,
            policy_lr=self.policy_lr,
            q_lr=self.q_lr,
            update_frequency=self.update_frequency,
            target_net_freq=self.target_net_freq,
            alpha=self.alpha,
            autotune=self.autotune,
            target_entropy_scale=self.target_entropy_scale,
            id=self.id,
            device=self.device,
            log=self.log,
            seed=self.seed,
            parent_rng=self.parent_rng,
        )

        # Copying networks
        copied.actor = deepcopy(self.actor)
        copied.qf1 = deepcopy(self.qf1)
        copied.qf2 = deepcopy(self.qf2)
        copied.qf1_target = deepcopy(self.qf1_target)
        copied.qf2_target = deepcopy(self.qf2_target)

        copied.global_step = self.global_step
        copied.actor_optimizer = optim.Adam(copied.actor.parameters(), lr=self.policy_lr, eps=1e-4)
        copied.q_optimizer = optim.Adam(list(copied.qf1.parameters()) + list(copied.qf2.parameters()), lr=self.q_lr, eps=1e-4)
        if self.autotune:
            copied.a_optimizer = optim.Adam([copied.log_alpha], lr=self.q_lr, eps=1e-4)
        copied.alpha_tensor = th.scalar_tensor(copied.alpha).to(self.device)
        copied.buffer = deepcopy(self.buffer)
        return copied

    @override
    def get_buffer(self):
        return self.buffer

    @override
    def set_buffer(self, buffer):
        self.buffer = buffer

    @override
    def get_policy_net(self) -> th.nn.Module:
        return self.actor

    def get_save_dict(self, save_replay_buffer: bool = False) -> dict:
        """Returns a dictionary of all components needed for saving the MOSAC instance."""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'qf1_target_state_dict': self.qf1_target.state_dict(),
            'qf2_target_state_dict': self.qf2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'alpha': self.alpha,
        }

        if save_replay_buffer:
            save_dict['buffer'] = self.buffer

        if self.autotune:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['a_optimizer_state_dict'] = self.a_optimizer.state_dict()
            save_dict["target_entropy_scale"] = self.target_entropy_scale

        return save_dict

    def save(self, save_dir="weights/", filename=None, save_replay_buffer=True):
        """Save the agent's weights and replay buffer."""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        save_dict = self.get_save_dict(save_replay_buffer)
        th.save(save_dict, save_path)

    def load(self, save_dict: Optional[dict] = None, path: Optional[str] = None, load_replay_buffer: bool = True):
        """Load the model and the replay buffer if specified.
        """
        if save_dict is None:
            assert path is not None, "Either save_dict or path should be provided."
            save_dict = th.load(path, map_location=self.device)

        self.actor.load_state_dict(save_dict['actor_state_dict'])
        self.qf1.load_state_dict(save_dict['qf1_state_dict'])
        self.qf2.load_state_dict(save_dict['qf2_state_dict'])
        self.qf1_target.load_state_dict(save_dict['qf1_target_state_dict'])
        self.qf2_target.load_state_dict(save_dict['qf2_target_state_dict'])
        self.actor_optimizer.load_state_dict(save_dict['actor_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(save_dict['q_optimizer_state_dict'])

        if 'log_alpha' in save_dict: # previously used autotune
            self.log_alpha = save_dict['log_alpha']
            self.a_optimizer.load_state_dict(save_dict['a_optimizer_state_dict'])
            self.target_entropy_scale = save_dict["target_entropy_scale"]

        if load_replay_buffer:
            self.buffer = save_dict['buffer']

        self.alpha = save_dict['alpha']

    @override
    def eval(
        self, 
        obs: np.ndarray, 
        w: Optional[np.ndarray] = None,
        num_envs: int = 1,
        **kwargs,
    ) -> Union[int, np.ndarray]:
        """Returns the best action to perform for the given obs.

        Args:
            obs: observation as a numpy array
            w: None
        Return:
            action as a numpy array (discrete actions)
        """
        obs = th.as_tensor(obs).float().to(self.device)
        if num_envs == 1:
            obs = np.expand_dims(obs, 0)
        with th.no_grad():
            action, _, _ = self.actor.get_action(obs)

        if num_envs > 1:
            action = action.detach().cpu().numpy()
        else:
            action = action[0].detach().item()

        return action

    @override
    def update(self):
        (mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = self.buffer.sample(
            self.batch_size, to_tensor=True, device=self.device
        )

        with th.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(mb_next_obs)
            qf1_next_target = self.qf1_target(mb_next_obs)
            qf2_next_target = self.qf2_target(mb_next_obs)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                th.min(qf1_next_target, qf2_next_target) - self.alpha_tensor * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = mb_rewards.flatten() + (1 - mb_dones.flatten()) * self.gamma * (min_qf_next_target)

        qf1_values = self.qf1(mb_obs)
        qf2_values = self.qf2(mb_obs)
        qf1_a_values = qf1_values.gather(1, mb_act.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, mb_act.long()).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        _, log_pi, action_probs = self.actor.get_action(mb_obs)
        with th.no_grad():
            qf1_values = self.qf1(mb_obs)
            qf2_values = self.qf2(mb_obs)
            min_qf_values = th.min(qf1_values, qf2_values)
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            # re-use action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha_tensor = self.log_alpha.exp()
            self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if self.global_step % self.target_net_freq == 0:
            polyak_update(params=self.qf1.parameters(), target_params=self.qf1_target.parameters(), tau=self.tau)
            polyak_update(params=self.qf2.parameters(), target_params=self.qf2_target.parameters(), tau=self.tau)
            self.qf1_target.requires_grad_(False)
            self.qf2_target.requires_grad_(False)

        if self.global_step % 100 == 0 and self.log:
            log_str = f"_{self.id}" if self.id is not None else ""
            to_log = {
                f"losses{log_str}/alpha": self.alpha,
                f"losses{log_str}/qf1_values": qf1_a_values.mean().item(),
                f"losses{log_str}/qf2_values": qf2_a_values.mean().item(),
                f"losses{log_str}/qf1_loss": qf1_loss.item(),
                f"losses{log_str}/qf2_loss": qf2_loss.item(),
                f"losses{log_str}/qf_loss": qf_loss.item() / 2.0,
                f"losses{log_str}/actor_loss": actor_loss.item(),
                "global_step": self.global_step,
            }
            if self.autotune:
                to_log[f"losses{log_str}/alpha_loss"] = alpha_loss.item()
            wandb.log(to_log)

    def train(
        self, 
        total_timesteps: int, 
        eval_env: Union[gym.Env, MORLGeneralizationEvaluator],
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        start_time = None,
        eval_mo_freq: int = 10000,
        test_generalization: bool = False,
    ):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps to train the agent for.
            eval_env (gym.Env): Environment to use for evaluation.
            ref_point (np.ndarray): Reference point for hypervolume calculation.
            known_pareto_front (Optional[List[np.ndarray]]): Optimal Pareto front, if known.
            num_eval_weights_for_front (int): Number of weights to evaluate for the Pareto front.
            num_eval_episodes_for_front (int): number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            start_time (Optional[float]): Start time of the training.
            eval_mo_freq (int): Number of timesteps between evaluations during an iteration.
            test_generalization (bool): Whether to test generalizability of the model.
        """
        if start_time is None:
            start_time = time.time()

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)
        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset()
        for _ in range(total_timesteps):
            self.global_step += 1
            # ALGO LOGIC: put action logic here
            if self.global_step < self.learning_starts:
                actions = self.env.action_space.sample()
            else:
                th_obs = th.as_tensor(obs).float().to(self.device)
                th_obs = th_obs.unsqueeze(0)
                actions, _, _ = self.actor.get_action(th_obs)
                actions = actions[0].detach().cpu().numpy()

            # execute the game and log data
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs
            if "final_observation" in infos:
                real_next_obs = infos["final_observation"]
            self.buffer.add(obs=obs, next_obs=real_next_obs, action=actions, reward=rewards, done=terminated)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            if terminated or truncated:
                obs, _ = self.env.reset()

            # ALGO LOGIC: training.
            if self.global_step > self.learning_starts:
                if self.global_step % self.update_frequency == 0:
                    self.update()
                if self.log and self.global_step % 100 == 0:
                    print("SPS:", int(self.global_step / (time.time() - start_time)))
                    wandb.log(
                        {"charts/SPS": int(self.global_step / (time.time() - start_time)), "global_step": self.global_step}
                    )
                
                if self.log and self.global_step % eval_mo_freq == 0:
                    # Evaluation
                    if test_generalization:
                        eval_env.eval(self, ref_point=ref_point, reward_dim=self.reward_dim, global_step=self.global_step)
                    else:
                        returns_test_tasks = [
                            policy_evaluation_mo(self, eval_env, ew, rep=num_eval_episodes_for_front)[3] for ew in eval_weights
                        ]
                        log_all_multi_policy_metrics(
                            current_front=returns_test_tasks,
                            hv_ref_point=ref_point,
                            reward_dim=self.reward_dim,
                            global_step=self.global_step,
                            n_sample_weights=num_eval_weights_for_front,
                            ref_front=known_pareto_front,
                        )

        if self.log:
            self.close_wandb()