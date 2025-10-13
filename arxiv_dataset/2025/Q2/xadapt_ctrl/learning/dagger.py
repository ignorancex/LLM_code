import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from mpl_toolkits.mplot3d import Axes3D
from ruamel.yaml import YAML
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv


class PhaseTwoAlgorithm(BaseAlgorithm):
    """
    Base class for Phase Two algorithms (e.g., DAGGER).
    
    This class implements the core functionality for Phase Two algorithms including:
    - Experience collection with extrinsic information
    - Environment latent encoder training
    - Policy updates
    - Environment interaction
    - Logging and evaluation

    Parameters
    ----------
    env_latent_encoder : nn.Module
        Neural network for encoding environment latent information
    policy : Union[str, Type[ActorCriticPolicy]]
        The policy model to use
    env : Union[GymEnv, str]
        The environment to learn from
    learning_rate : Union[float, Schedule]
        The learning rate
    n_steps : int
        The number of steps to run for each environment per update
    use_tanh_act : bool
        Whether to use tanh activation for actions
    gamma : float
        Discount factor
    gae_lambda : float
        Factor for trade-off of bias vs variance for GAE
    ent_coef : float
        Entropy coefficient for the loss calculation
    vf_coef : float
        Value function coefficient for the loss calculation
    max_grad_norm : float
        The maximum value for the gradient clipping
    use_sde : bool
        Whether to use State Dependent Exploration
    sde_sample_freq : int
        Sample a new noise matrix every n steps when using gSDE
    history_len : int
        Length of history to use for latent encoding
    state_obs_size : int
        Size of state observations
    env_obs_size : int
        Size of environment observations
    latent_size : int
        Size of latent space
    act_size : int
        Size of action space
    tensorboard_log : Optional[str]
        The log location for tensorboard
    create_eval_env : bool
        Whether to create a second environment for evaluation
    eval_env : Union[GymEnv, str]
        The environment used for evaluation
    monitor_wrapper : bool
        Whether to wrap the environment in a Monitor wrapper
    policy_kwargs : Optional[Dict[str, Any]]
        Additional arguments to be passed to the policy on creation
    verbose : int
        The verbosity level: 0 no output, 1 info, 2 debug
    seed : Optional[int]
        Seed for the pseudo random generators
    device : Union[th.device, str]
        Device (cpu, cuda, ...) on which the code should be run
    _init_setup_model : bool
        Whether or not to build the network at the creation of the instance
    supported_action_spaces : Optional[Tuple[gym.spaces.Space, ...]]
        The action spaces supported by the algorithm
    """

    def __init__(
        self,
        env_latent_encoder: nn.Module,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        use_tanh_act: bool,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        history_len: int,
        state_obs_size: int,
        env_obs_size: int,
        latent_size: int,
        act_size: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        eval_env: Union[GymEnv, str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
        super(PhaseTwoAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        # Store algorithm parameters
        self.env_latent_encoder = env_latent_encoder
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_tanh_act = use_tanh_act
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        # Store adaptation module specific parameters
        self.history_len = history_len
        self.act_size = act_size
        self.state_obs_size = state_obs_size
        self.env_obs_size = env_obs_size
        self.latent_size = latent_size

        # Setup evaluation environment
        self.eval_env = eval_env

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Initialize the model and setup the learning rate schedule."""
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Initialize rollout buffer
        self.rollout_buffer = ExtrinsincsRolloutBuffer(
            self.n_steps,
            self.latent_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
        )

    def normalize_state_obs(self, obs_n_norm: np.ndarray, eval: bool = False) -> np.ndarray:
        """
        Normalize state observations.
        
        Parameters
        ----------
        obs_n_norm : np.ndarray
            Observations to normalize
        eval : bool
            Whether to use evaluation normalization
            
        Returns
        -------
        np.ndarray
            Normalized observations
        """
        obs_mean, obs_var = self.env.get_obs_norm()

        # Normalize state history
        obs_state_history_n_normalized = obs_n_norm[:, -self.history_len*(
            self.act_size+self.state_obs_size):-self.history_len*self.act_size]
        obs_state_mean = np.tile(obs_mean[:self.state_obs_size], [self.n_envs, self.history_len])
        obs_state_var = np.tile(obs_var[:self.state_obs_size], [self.n_envs, self.history_len])
        obs_state_history_normalized = (
            obs_state_history_n_normalized - obs_state_mean) / np.sqrt(obs_state_var + 1e-8)
        
        # Normalize action history
        obs_act_history_n_normalized = obs_n_norm[:, -self.history_len*self.act_size:]
        obs_act_mean = np.tile(obs_mean[self.state_obs_size:self.state_obs_size+self.act_size], [self.n_envs, self.history_len])
        obs_act_var = np.tile(obs_var[self.state_obs_size:self.state_obs_size+self.act_size], [self.n_envs, self.history_len])
        obs_act_history_normalized = (
            obs_act_history_n_normalized - obs_act_mean) / np.sqrt(obs_act_var + 1e-8)

        # Update normalized observations
        obs_n_norm[:, -self.history_len*(self.act_size+self.state_obs_size):-self.history_len*self.act_size] = obs_state_history_normalized
        obs_n_norm[:, -self.history_len*self.act_size:] = obs_act_history_normalized

        return obs_n_norm

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: ExtrinsincsRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a rollout buffer.
        
        Parameters
        ----------
        env : VecEnv
            The training environment
        callback : BaseCallback
            Callback that will be called at each step
        rollout_buffer : ExtrinsincsRolloutBuffer
            Buffer to fill with rollouts
        n_rollout_steps : int
            Number of experiences to collect per environment
            
        Returns
        -------
        bool
            True if function returned with at least n_rollout_steps collected,
            False if callback terminated rollout prematurely
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # Normalize observations
            obs_n_norm = self._last_obs
            obs_norm = self.normalize_state_obs(obs_n_norm)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(obs_norm).to(self.device)

                # Extract features and compute latent representations
                obs_current = obs_tensor[:, :-self.history_len*(self.act_size+self.state_obs_size)]
                obs_history = obs_tensor[:, -self.history_len*(self.act_size+self.state_obs_size):]
                features = self.policy.extract_features(obs_current)

                # Compute shared and environment latent representations
                shared_latent_base = self.policy.mlp_extractor.shared_net(features[:, :-self.env_obs_size])
                extrinsincs = self.policy.mlp_extractor.env_net(features[:, -self.env_obs_size:])
                
                # Store last true extrinsics for first step
                if n_steps == 0:
                    last_true_extrinsics = extrinsincs
                    
                # Compute estimated extrinsics
                est_extrinsincs = self.env_latent_encoder(obs_history.float())

                # Combine latent representations
                shared_latent = th.cat((shared_latent_base, est_extrinsincs), 1)
                latent_pi = self.policy.mlp_extractor.policy_net(shared_latent)
                latent_vf = self.policy.mlp_extractor.value_net(shared_latent)

                # Get actions
                distribution = self.policy._get_action_dist_from_latent(latent_pi)
                actions = distribution.get_actions(deterministic=True)

            actions = actions.cpu().numpy()

            # Clip actions if needed
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            # Perform action in environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # Update callback
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            # Reshape actions if discrete
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Add experience to buffer
            rollout_buffer.add(
                obs_norm, actions, rewards, self._last_episode_starts,
                extrinsincs, est_extrinsincs, last_true_extrinsics
            )
            
            # Update last true extrinsics
            with th.no_grad():
                last_true_extrinsics = extrinsincs
            
            self._last_obs = new_obs
            self._last_episode_starts = dones

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        To be implemented by child classes.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: Tuple = (10, 100),
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PhaseTwoAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        env_cfg: str = None,
    ) -> "PhaseTwoAlgorithm":
        """
        Train the policy for a specified number of timesteps.
        
        Parameters
        ----------
        total_timesteps : int
            Total number of timesteps to train for
        callback : MaybeCallback, optional
            Callback(s) called at every step with state of the algorithm
        log_interval : Tuple, optional
            The interval for logging
        eval_env : Optional[GymEnv], optional
            Environment to use for evaluation
        eval_freq : int, optional
            Evaluate the policy every eval_freq timesteps
        n_eval_episodes : int, optional
            Number of episodes to evaluate on
        tb_log_name : str, optional
            Name of the run for tensorboard log
        eval_log_path : Optional[str], optional
            Path to a folder where the evaluations will be saved
        reset_num_timesteps : bool, optional
            Whether or not to reset the current timestep number
        env_cfg : str, optional
            Environment configuration
            
        Returns
        -------
        PhaseTwoAlgorithm
            The trained algorithm
        """
        iteration = 0

        # Setup training
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        # Save configuration
        new_cfg_dir = self.logger.get_dir() + "/config.yaml"
        with open(new_cfg_dir, "w") as outfile:
            YAML().dump(self.env_cfg, outfile)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # Collect rollouts
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Update policy
            self.train()

            # Save model and evaluate
            if log_interval is not None and iteration % log_interval[1] == 0:
                policy_path = self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                example_input = th.rand(1, (self.state_obs_size + self.act_size) *
                                      self.history_len + self.state_obs_size + self.act_size +
                                      self.env_obs_size).cpu()
                self.save_deterministic_graph(
                    policy_path + f"/iter_{iteration:05d}.pt", example_input
                )
                self.eval(iteration)

        callback.on_training_end()

        return self

    def save_deterministic_graph(
        self,
        fname_env_latent_encoder: str,
        example_input: th.Tensor,
        device: str = 'cpu'
    ) -> None:
        """
        Save the environment latent encoder as a deterministic graph.
        
        Parameters
        ----------
        fname_env_latent_encoder : str
            Filename to save the encoder to
        example_input : th.Tensor
            Example input for tracing
        device : str, optional
            Device to use for tracing
        """
        hlen = (self.state_obs_size + self.act_size) * self.history_len
        prop_encoder_graph = th.jit.trace(
            self.env_latent_encoder.to(device), example_input[:, -hlen:]
        )
        th.jit.save(prop_encoder_graph, fname_env_latent_encoder)
        self.env_latent_encoder.to(self.device)

class DAGGER(PhaseTwoAlgorithm):
    """
    DAGGER (Dataset Aggregation) algorithm for Phase Two.
    
    This implementation is based on the DAGGER algorithm from the paper:
    "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
    by Ross et al. (2011)
    https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf

    Parameters
    ----------
    env_latent_encoder : nn.Module
        Neural network for encoding environment latent information
    policy : Union[str, Type[ActorCriticPolicy]]
        The policy model to use
    env : Union[GymEnv, str]
        The environment to learn from
    learning_rate : Union[float, Schedule]
        The learning rate
    n_steps : int
        The number of steps to run for each environment per update
    use_tanh_act : bool
        Whether to use tanh activation for actions
    gamma : float
        Discount factor
    gae_lambda : float
        Factor for trade-off of bias vs variance for GAE
    ent_coef : float
        Entropy coefficient for the loss calculation
    vf_coef : float
        Value function coefficient for the loss calculation
    max_grad_norm : float
        The maximum value for the gradient clipping
    use_sde : bool
        Whether to use State Dependent Exploration
    sde_sample_freq : int
        Sample a new noise matrix every n steps when using gSDE
    history_len : int
        Length of history to use for latent encoding
    state_obs_size : int
        Size of state observations
    env_obs_size : int
        Size of environment observations
    latent_size : int
        Size of latent space
    act_size : int
        Size of action space
    tensorboard_log : Optional[str]
        The log location for tensorboard
    create_eval_env : bool
        Whether to create a second environment for evaluation
    eval_env : Union[GymEnv, str]
        The environment used for evaluation
    monitor_wrapper : bool
        Whether to wrap the environment in a Monitor wrapper
    policy_kwargs : Optional[Dict[str, Any]]
        Additional arguments to be passed to the policy on creation
    verbose : int
        The verbosity level: 0 no output, 1 info, 2 debug
    seed : Optional[int]
        Seed for the pseudo random generators
    device : Union[th.device, str]
        Device (cpu, cuda, ...) on which the code should be run
    _init_setup_model : bool
        Whether or not to build the network at the creation of the instance
    supported_action_spaces : Optional[Tuple[gym.spaces.Space, ...]]
        The action spaces supported by the algorithm
    """

    def __init__(
        self,
        env_latent_encoder: nn.Module,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        use_tanh_act: bool,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        history_len: int,
        state_obs_size: int,
        env_obs_size: int,
        latent_size: int,
        act_size: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        eval_env: Union[GymEnv, str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
        super(DAGGER, self).__init__(
            env_latent_encoder=env_latent_encoder,
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            use_tanh_act=use_tanh_act,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            history_len=history_len,
            state_obs_size=state_obs_size,
            env_obs_size=env_obs_size,
            latent_size=latent_size,
            act_size=act_size,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            eval_env=eval_env,
            monitor_wrapper=monitor_wrapper,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=supported_action_spaces,
        )

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            [*self.env_latent_encoder.parameters()],
            lr=learning_rate
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.1
        )
        self.loss_fn = nn.MSELoss()
        self.best_loss = float('inf')

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        
        This method implements the DAGGER training loop, including:
        1. Environment latent encoder updates
        2. Policy updates with extrinsic information
        3. Loss computation and optimization
        """
        # Train for n_epochs epochs
        extrinsics_mse = 0
        loss_counter = 0

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            # Get observations and compute features
            obs_tensor = th.as_tensor(rollout_data["observations"]).to(self.device)
            obs_current = obs_tensor[:, :-self.history_len*(self.act_size+self.state_obs_size)]
            obs_history = obs_tensor[:, -self.history_len*(self.act_size+self.state_obs_size):]
            features = self.policy.extract_features(obs_current)

            # Get true and estimated extrinsics
            true_extrinsincs = self.policy.mlp_extractor.env_net(
                features[:, -self.env_obs_size:]
            ).clone()
            est_extrinsincs = self.env_latent_encoder(obs_history.float()).clone()

            # Compute loss
            loss = self.loss_fn(est_extrinsincs, true_extrinsincs)

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            extrinsics_mse += loss.item()
            loss_counter += 1

        # Update learning rate
        self.scheduler.step()

        # Compute average loss
        avg_extrinsics_loss = extrinsics_mse / loss_counter

        # Log metrics
        self._n_updates += 1

        # Save best model
        if avg_extrinsics_loss < self.best_loss:
            self.best_loss = avg_extrinsics_loss
            policy_path = self.logger.get_dir() + "/Policy"
            os.makedirs(policy_path, exist_ok=True)
            example_input = th.rand(1, (self.state_obs_size + self.act_size) *
                                  self.history_len + self.state_obs_size + self.act_size +
                                  self.env_obs_size).cpu()
            self.save_deterministic_graph(
                policy_path + "/iter_888.pt", example_input
            )
            self.eval()

        # Log metrics
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/avg_extrinsics_loss", avg_extrinsics_loss)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

   
