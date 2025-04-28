# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = ""
    env: str = ""  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1000)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    sample_ratio: float = 0.005
    reward_scale: float = 1.0 
    
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 64  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 1e-3  # V function learning rate
    qf_lr: float = 1e-3  # Critic learning rate
    actor_lr: float = 1e-3  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "wikiRL"
    group: str = env + '-iql'
    name: str = str(seed)
    drl_algo: str = "a2c"  # Add this line to include the drl_algo in the config
    dataset_path: str = ""
    test_trajectory: str = ""


    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device
        self.rng = np.random.default_rng(seed=0)  # Initialize with a seed

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_custom_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into a non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset loaded with {n_transitions} transitions.")


    def sample(self, batch_size: int) -> TensorBatch:
        indices = self.rng.integers(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.set_seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 8,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        # print("state type", type(state))
        if isinstance(state, tuple) :
            state = np.array(state[0])
        if isinstance(state, list):
            state = np.array(state)

        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 8,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def cloning_loss(self, predicted_actions: torch.Tensor, true_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Squared Error between predicted and true actions.
        
        Args:
            predicted_actions (torch.Tensor): Actions predicted by the policy.
            true_actions (torch.Tensor): Ground truth actions from the dataset.
        
        Returns:
            torch.Tensor: Computed MSE loss.
        """
        return F.mse_loss(predicted_actions, true_actions)

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

         # Compute Cloning Loss
        with torch.no_grad():
            if isinstance(self.actor, DeterministicPolicy):
                predicted_actions = self.actor(observations)
            elif isinstance(self.actor, GaussianPolicy):
                # Use mean actions for GaussianPolicy
                predicted_actions = self.actor(observations).mean
            else:
                raise NotImplementedError("Unsupported actor type for cloning loss computation.")
        
        true_actions = actions  # Ground truth actions from the batch
        loss_cloning = self.cloning_loss(predicted_actions, true_actions)
    
        # Add Cloning Loss to Log Dictionary
        log_dict['cloning_loss'] = loss_cloning.item()
    
        # Optionally, print the cloning loss
        print(f"Cloning Loss: {loss_cloning.item():.6f}")

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]

def get_dataset(env, ratio):
    env_name = env.split('-')
    ratio_str = '' if ratio == 1 else '-'+str(ratio)+'-d1'
    if env_name[0] in ["kitchen"]:
        suffix = 'kitchen'
    elif env_name[0] in ['hopper', 'halfcheetah', 'walker2d', 'reacher2d', 'ant']:
        suffix = 'mujoco'
    if "v" not in env_name[2]:
        dataset_path = "../data/" + suffix +'/' + env_name[0] + '-' + env_name[1] + '-' + env_name[2] + ratio_str + '-' + env_name[3] + ".pkl"
    else:
        dataset_path = "../data/" + suffix +'/' + env_name[0] + '-' + env_name[1] + ratio_str + '-' + env_name[2] + ".pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    dataset = {'actions': [], 'next_observations': [], 'observations': [], 'rewards': [], 'terminals': []}
    for path in trajectories:
        dataset['actions'].append(path['actions'])
        dataset['next_observations'].append(path['next_observations'])
        dataset['observations'].append(path['observations'])
        dataset['rewards'].append(path['rewards'])
        dataset['terminals'].append(path['terminals'])
    observations = np.concatenate(dataset['observations'], axis=0).astype(np.float32)
    actions=np.concatenate(dataset['actions'], axis=0).astype(np.float32)
    next_observations=np.concatenate(dataset['next_observations'], axis=0).astype(np.float32)
    rewards=np.concatenate(dataset['rewards'], axis=0).astype(np.float32)
    dones=np.concatenate(dataset['terminals'], axis=0).astype(np.float32)
    observations=observations.reshape(-1, observations.shape[-1])
    actions=actions.reshape(-1, actions.shape[-1])
    next_observations=next_observations.reshape(-1, next_observations.shape[-1])
    rewards=rewards.reshape(-1)
    dones=dones.reshape(-1)
    
    return dict(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=dones
    )

import pickle
from datetime import datetime
import numpy as np
import pandas as pd

def backtest_iql_agent(env, agent, device, n_episodes=10, variant=None, target_reward_raw=None, train_or_test='test', drl_algo='ppo', random_seed=0, dataset_path=None, test_trajectory=None):
    """
    Backtest the IQL agent and save the total asset values per episode as pickle files.

    Args:
        // ... existing arguments ...
        dataset_path (str): Path to the dataset pickle file
        test_trajectory (str): Path to the test trajectory pickle file
    """
    total_asset_lists = []
    episode_bc_losses = []  # New list to store behavior cloning losses

    # Load the appropriate dataset based on train_or_test
    if train_or_test == 'train':
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        actual_actions = data[0]['actions']
    else:  # test case
        with open(test_trajectory, 'rb') as f:
            data = pickle.load(f)
        actual_actions = data[0]['actions']

    for episode in range(n_episodes):
        reset_output = env.reset()
        print(f"Episode {episode + 1} reset_output:", reset_output)  # Debug statement

        if isinstance(reset_output, tuple):
            state, _ = reset_output  # Unpack state from tuple
        else:
            state = reset_output
        done = False
        episode_reward = 0.0
        actions_taken = []

        # Initialize total_asset_value_list with initial_amount
        initial_amount = env.initial_amount if hasattr(env, 'initial_amount') else 1000000.00
        total_asset_value_list = [initial_amount]
        print(f"Initial asset value: {initial_amount}")

        episode_bc_losses = []  # Store BC losses for this episode

        t = 0
        while not done:
            # Get the predicted action from the agent
            predicted_action = agent.actor.act(state, device)
            
            # Get the actual action from the loaded dataset
            actual_action = actual_actions[t]
            
            # Use the actual action for the environment step
            next_output = env.step(predicted_action)

            if isinstance(next_output, tuple):
                if len(next_output) == 4:
                    next_state, reward, done, info = next_output
                elif len(next_output) == 5:
                    next_state, reward, done, truncated, info = next_output
                else:
                    raise ValueError(f"Unexpected return format from env.step(): {next_output}")
            else:
                raise ValueError(f"Unexpected return type from env.step(): {type(next_output)}")

            # Calculate behavior cloning loss (MSE between predicted and actual action)
            bc_loss = F.mse_loss(
                torch.tensor(predicted_action, device=device),
                torch.tensor(actual_action, device=device)
            ).item()
            print("Behavior cloning loss:", bc_loss)
            episode_bc_losses.append(bc_loss)

            print("Reward:", reward)
            scaled_reward = reward * (1 / env.reward_scaling) if hasattr(env, 'reward_scaling') else reward
            print("Adding scaled reward to total_asset_value_list:", scaled_reward)
            new_total_asset = total_asset_value_list[-1] + scaled_reward
            total_asset_value_list.append(new_total_asset)
            print(f"Total asset at timestep {t}: {new_total_asset}")

            episode_reward += reward
            actions_taken.append(actual_action)
            t += 1
            state = next_state

        total_asset_value_list = total_asset_value_list[:-1]         
        
        # Create directory for storing pickle files
        checkpoint_dir = f"checkpoints/{drl_algo}_iql_{random_seed}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save total asset values
        asset_pkl_filename = f'total_asset_value_change_{train_or_test}.pkl'
        asset_pkl_path = os.path.join(checkpoint_dir, asset_pkl_filename)
        with open(asset_pkl_path, 'wb') as f:
            pickle.dump(total_asset_value_list, f)
        print(f"Saved asset values to {asset_pkl_path}")

        # Save behavior cloning losses
        bc_loss_pkl_filename = f'{train_or_test}_loss_list.pkl'
        bc_loss_pkl_path = os.path.join(checkpoint_dir, bc_loss_pkl_filename)
        with open(bc_loss_pkl_path, 'wb') as f:
            pickle.dump(episode_bc_losses, f)
        print(f"Saved behavior cloning losses to {bc_loss_pkl_path}")

        break  # Remove this if you want to run multiple episodes

    return total_asset_lists, episode_bc_losses


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pyrallis.wrap()
def train(config: TrainConfig, args):
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    hidden_dim = 256
    config.seed = args.seed
    config.sample_ratio = args.sample_ratio
    config.group = f"{config.env}-iql-ratio={config.sample_ratio}-hidden_dim={hidden_dim}"
    config.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    config.qf_lr = args.qf_lr

    # Prepare training environment
    import pandas as pd
    train_data_file = 'train_data.csv'
    train_pd = pd.read_csv(train_data_file)
    train_pd = train_pd.set_index(train_pd.columns[0])
    train_pd.index.names = ['']

    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    # Define environment parameters
    from finrl.config import INDICATORS, TRAINED_MODEL_DIR
    stock_dimension = len(train_pd.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,  # Define your technical indicators
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    env = StockTradingEnv(df=train_pd, **env_kwargs)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State Dimension: {state_dim}, Action Dimension: {action_dim}")

    print("Loading dataset from pickle file")
    with open(args.dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Number of trajectories loaded: {len(data)}")

    data_0 = data[0]

    dataset = dict(
        observations=data_0['observations'],
        actions=data_0['actions'],
        next_observations=data_0['next_observations'],
        rewards=data_0['rewards'],
        terminals=data_0['terminals']
    )

    if config.normalize_reward:
        modify_reward(
            dataset,
            config.env,
            max_episode_steps=env.max_episode_steps,  # Ensure env has this attribute
            reward_scale=config.reward_scale,
            # reward_scale=1e-4,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std, reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_custom_dataset(dataset)
    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Initialize networks
    q_network = TwinQ(state_dim, action_dim, hidden_dim=hidden_dim).to(config.device)
    v_network = ValueFunction(state_dim, hidden_dim=hidden_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout, hidden_dim=hidden_dim
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout, hidden_dim=hidden_dim
        )
    ).to(config.device)

    # Print parameter counts
    print(f"Q-network parameters: {count_parameters(q_network)}")
    print(f"V-network parameters: {count_parameters(v_network)}")
    print(f"Actor parameters: {count_parameters(actor)}")
    print(f"Total parameters: {count_parameters(q_network) + count_parameters(v_network) + count_parameters(actor)}")

    # Initialize optimizers
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")

    # Initialize IQL Trainer
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file, map_location=config.device))
        actor = trainer.actor

    for t in range(int(config.max_timesteps)):
        print("training timestep:", t)
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        
        # Perform a training step
        log_dict = trainer.train(batch)
        print(log_dict)

    # --- Backtesting ---
    print("Starting Backtesting...")

    # Initialize backtesting environment
    test_data_file = 'test_data.csv'
    test_pd = pd.read_csv(test_data_file)
    test_pd = test_pd.set_index(test_pd.columns[0])
    test_pd.index.names = ['']

    # Reinitialize the environment for backtesting
    test_env = StockTradingEnv(df=test_pd, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)

    train_data_file = 'train_data.csv'
    train_pd = pd.read_csv(train_data_file)
    train_pd = train_pd.set_index(train_pd.columns[0])
    train_pd.index.names = ['']
    train_env = StockTradingEnv(df=train_pd, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)


    # Define variant and target_reward_raw if needed
    variant = {
        'exp_name': 'iql_experiment',  # Replace with actual experiment name
        'drl_algo': 'IQL'  # Algorithm name
    }
    target_reward_raw = 1381034  # Replace with the actual target reward if needed

    # Backtest the IQL agent
    backtest_iql_agent(
        env=test_env,
        agent=trainer,
        device=config.device,
        n_episodes=config.n_episodes,
        variant=variant,
        target_reward_raw=target_reward_raw,
        train_or_test='test',
        drl_algo=config.drl_algo,  # Use config.drl_algo here
        random_seed=config.seed,
        dataset_path=args.dataset_path,
        test_trajectory=args.test_trajectory
    )

    backtest_iql_agent(
        env=train_env,
        agent=trainer,
        device=config.device,
        n_episodes=config.n_episodes,
        variant=variant,
        target_reward_raw=target_reward_raw,
        train_or_test='train',
        drl_algo=config.drl_algo,  # Use config.drl_algo here
        random_seed=config.seed,
        dataset_path=args.dataset_path,
        test_trajectory=args.test_trajectory
    )


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl_algo", type=str, required=False, help="Name of the DRL algorithm")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--sample_ratio", type=float, default=1, help="Sample ratio")
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    parser.add_argument("--device", type=int, default=0, help="GPU device number")
    parser.add_argument("--qf_lr", type=float, default=1e-3, help="Q-function learning rate")
    parser.add_argument("--dataset_path", type=str, required=False, 
                        help="Path to the dataset pickle file", 
                        default='data/train_a2c_trajectory_2024-10-13_12-47-12.pkl')
    parser.add_argument("--test_trajectory", type=str, 
                        help="Path to the test trajectory pickle file", 
                        default='data/test_a2c_trajectory_2024-10-13_12-48-25.pkl',
                        required=False)

    args = parser.parse_args()
    
    # Set global random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    train(args=args)