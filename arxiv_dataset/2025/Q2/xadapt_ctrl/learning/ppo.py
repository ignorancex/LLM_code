import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from gym import spaces
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

import wandb
from collections import defaultdict

class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base class for On-Policy algorithms (e.g., A2C/PPO).

    Parameters
    ----------
    policy : Union[str, Type[ActorCriticPolicy]]
        The policy model to use (MlpPolicy, CnnPolicy, ...)
    env : Union[GymEnv, str]
        The environment to learn from
    learning_rate : Union[float, Schedule]
        The learning rate, can be a function of the current progress
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
    tensorboard_log : Optional[str]
        The log location for tensorboard
    best_reward : float
        Best reward achieved so far
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
    _init_setup_policy : bool
        Whether or not to setup the policy at the creation of the instance
    supported_action_spaces : Optional[Tuple[gym.spaces.Space, ...]]
        The action spaces supported by the algorithm
    """

    def __init__(
        self,
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
        tensorboard_log: Optional[str] = None,
        best_reward: float = -float('inf'),
        create_eval_env: bool = False,
        eval_env: Union[GymEnv, str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        _init_setup_policy: bool = True,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = None,
    ):
        super(OnPolicyAlgorithm, self).__init__(
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
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_tanh_act = use_tanh_act
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        # Setup evaluation environment
        self.eval_env = eval_env
        self.policy_to_load = policy
        self.best_reward = best_reward
        self._init_setup_policy = _init_setup_policy

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Initialize the model and setup the learning rate schedule."""
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Initialize rollout buffer -- store rollouts for Phase 1 training
        self.rollout_buffer = Ph1RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Setup policy
        if self._init_setup_policy:
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs
            )

            # Add Tanh activation if specified
            if self.use_tanh_act:
                self.policy.action_net = th.nn.Sequential(
                    self.policy.action_net, th.nn.Tanh()
                )

            self.policy = self.policy.to(self.device)
        else:
            self.policy = self.policy_to_load

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: Ph1RolloutBuffer,
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
        rollout_buffer : Ph1RolloutBuffer
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

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            # Get normalized observations
            obs_norm = self._last_obs

            with th.no_grad():
                # Convert to pytorch tensor and get actions
                obs_tensor = th.as_tensor(obs_norm).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Clip actions if needed
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            # Perform action in environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            privileged_actions = env.getQuadPrivAct()
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
                obs_norm, actions, privileged_actions,
                rewards, self._last_episode_starts, values, log_probs
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Compute value for the last timestep
        with th.no_grad():
            obs_norm = self._last_obs
            obs_tensor = th.as_tensor(obs_norm).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        # Compute returns and advantages
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: Tuple = (10, 100),
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        env_cfg: str = None,
    ) -> "OnPolicyAlgorithm":
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
        OnPolicyAlgorithm
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

            # Update running mean and standard deviation
            if iteration % 10 == 0:
                self.env.update_rms()

            # Save model and evaluate
            if log_interval is not None and iteration % log_interval[1] == 0:
                policy_path = self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                self.policy.save(policy_path + f"/iter_{iteration:05d}.pth")

                self.env.save_rms(
                    save_dir=self.logger.get_dir() + "/RMS", n_iter=iteration
                )
                self.eval(iteration)

                # Save best model
                ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                if ep_rew_mean > self.best_reward:
                    self.best_reward = ep_rew_mean
                    wandb.log({'rollout/save_at_iter': iteration}, step=self.num_timesteps)
                    best_iter = 888
                    self.policy.save(policy_path + f"/iter_{best_iter:05d}.pth")
                    self.env.save_rms(
                        save_dir=self.logger.get_dir() + "/RMS", n_iter=best_iter
                    )
                    self.eval(best_iter)

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """Get the parameters to save in the torch model."""
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    
    This implementation is based on the PPO algorithm from the paper:
    "Proximal Policy Optimization Algorithms" by Schulman et al. (2017)
    https://arxiv.org/abs/1707.06347

    The algorithm uses a clipped objective function to prevent large policy updates
    and includes behavior cloning (BC) loss for imitation learning.

    Parameters
    ----------
    policy : Union[str, Type[ActorCriticPolicy]]
        The policy model to use (MlpPolicy, CnnPolicy, ...)
    env : Union[GymEnv, str]
        The environment to learn from
    learning_rate : Union[float, Callable], optional
        The learning rate, can be a function of the current progress, by default 3e-4
    n_steps : int, optional
        The number of steps to run for each environment per update, by default 2048
    use_tanh_act : bool, optional
        Whether to use tanh activation for actions, by default True
    batch_size : Optional[int], optional
        Minibatch size, by default 64
    n_epochs : int, optional
        Number of epoch when optimizing the surrogate loss, by default 10
    gamma : float, optional
        Discount factor, by default 0.99
    gae_lambda : float, optional
        Factor for trade-off of bias vs variance for GAE, by default 0.95
    clip_range : Union[float, Callable], optional
        Clipping parameter for the policy objective, by default 0.2
    clip_range_vf : Union[None, float, Callable], optional
        Clipping parameter for the value function, by default None
    ent_coef : float, optional
        Entropy coefficient for the loss calculation, by default 0.0
    vf_coef : float, optional
        Value function coefficient for the loss calculation, by default 0.5
    min_BC_coef : float, optional
        Minimum BC coefficient for behavior cloning loss, by default 0.5
    BC_scale : float, optional
        BC coefficient scale factor, by default 1.0
    BC_alpha : float, optional
        BC coefficient decay rate, by default 0.999
    max_grad_norm : float, optional
        The maximum value for the gradient clipping, by default 0.5
    use_sde : bool, optional
        Whether to use State Dependent Exploration, by default False
    sde_sample_freq : int, optional
        Sample a new noise matrix every n steps when using gSDE, by default -1
    target_kl : Optional[float], optional
        Limit the KL divergence between updates, by default None
    tensorboard_log : Optional[str], optional
        The log location for tensorboard, by default None
    create_eval_env : bool, optional
        Whether to create a second environment for evaluation, by default False
    eval_env : Union[GymEnv, str], optional
        The environment used for evaluation, by default None
    policy_kwargs : Optional[Dict[str, Any]], optional
        Additional arguments to be passed to the policy on creation, by default None
    verbose : int, optional
        The verbosity level: 0 no output, 1 info, 2 debug, by default 0
    seed : Optional[int], optional
        Seed for the pseudo random generators, by default None
    device : Union[th.device, str], optional
        Device (cpu, cuda, ...) on which the code should be run, by default "auto"
    env_cfg : str, optional
        Environment configuration, by default None
    _init_setup_model : bool, optional
        Whether or not to build the network at the creation of the instance, by default True
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        use_tanh_act: bool = True,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Callable] = 0.2,
        clip_range_vf: Union[None, float, Callable] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        min_BC_coef: float = 0.5,
        BC_scale: float = 1.0,
        BC_alpha: float = 0.999,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        eval_env: Union[GymEnv, str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        env_cfg: str = None,
        _init_setup_model: bool = True,
    ):
        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            use_tanh_act=use_tanh_act,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            eval_env=eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Validate environment setup
        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1, (
                f"`n_steps * n_envs` must be greater than 1. "
                f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            )
            
            # Check batch size compatibility
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        # Store PPO specific parameters
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        
        # Behavior cloning parameters
        self.min_BC_coef = min_BC_coef
        self.BC_alpha = BC_alpha
        self.BC_scale = BC_scale

        self.env_cfg = env_cfg
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Initialize the model and setup the learning rate schedule."""
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        
        This method implements the PPO training loop, including:
        1. Policy and value function updates
        2. Behavior cloning loss computation
        3. Entropy bonus for exploration
        4. Gradient clipping
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Initialize metrics
        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        BC_losses = []
        clip_fractions = []

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix if using gSDE
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Evaluate actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Calculate policy loss
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Calculate value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                
                # Calculate BC loss
                obs_tensor = th.as_tensor(rollout_data.observations).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
                BC_loss = F.mse_loss(rollout_data.privileged_actions, actions)

                # Calculate entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # Combine losses
                loss = (
                    min(1, 1-self.BC_scale*self.BC_alpha**self._n_updates) * (
                        policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    ) + max(0, self.BC_scale*self.BC_alpha**self._n_updates) * BC_loss
                )

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Record metrics
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                BC_losses.append(BC_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            # Early stopping if KL divergence is too high
            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        # Calculate explained variance
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), 
            self.rollout_buffer.returns.flatten()
        )

        # Log metrics
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        wandb.log({'train/entropy_loss': np.mean(entropy_losses)}, step=self.num_timesteps)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        wandb.log({'train/policy_gradient_loss': np.mean(pg_losses)}, step=self.num_timesteps)
        self.logger.record("train/value_loss", np.mean(value_losses))
        wandb.log({'train/value_loss': np.mean(value_losses)}, step=self.num_timesteps)
        self.logger.record("train/BC_loss", np.mean(BC_losses))
        wandb.log({'train/BC_loss': np.mean(BC_losses)}, step=self.num_timesteps)
        self.logger.record("train/BC_coeff", max(self.min_BC_coef, self.BC_scale*self.BC_alpha**self._n_updates))
        wandb.log({'train/BC_coeff': max(self.min_BC_coef, self.BC_scale*self.BC_alpha**self._n_updates)}, step=self.num_timesteps)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        wandb.log({'train/approx_kl': np.mean(approx_kl_divs)}, step=self.num_timesteps)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        wandb.log({'train/clip_fraction': np.mean(clip_fractions)}, step=self.num_timesteps)
        self.logger.record("train/loss", loss.item())
        wandb.log({'train/loss': loss.item()}, step=self.num_timesteps)
        self.logger.record("train/explained_variance", explained_var)
        
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            wandb.log({'train/std': th.exp(self.policy.log_std).mean().item()}, step=self.num_timesteps)
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        wandb.log({'train/n_updates': self._n_updates}, step=self.num_timesteps)
        self.logger.record("train/clip_range", clip_range)
        wandb.log({'train/clip_range': clip_range}, step=self.num_timesteps)
        
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
            wandb.log({'train/clip_range_vf': clip_range_vf}, step=self.num_timesteps)
        
        self._n_updates += self.n_epochs
