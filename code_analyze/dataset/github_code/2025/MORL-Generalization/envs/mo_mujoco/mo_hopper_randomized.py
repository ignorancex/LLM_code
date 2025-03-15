"""Implementation of the Hopper environment supporting
domain randomization optimization.

For all details: https://www.gymlibrary.ml/environments/mujoco/hopper/
"""
from typing import Dict, Tuple, Union, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium.spaces import Box
from envs.mo_mujoco.utils.random_mujoco_env import RandomMujocoEnv
from morl_generalization.algos.dr import DREnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class MOHopperDR(RandomMujocoEnv, DREnv, EzPickle):
    """
    ## Description
    Multi-objective version of Gymnasium's Mujoco Hopper environment with randomizable environment parameters.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/hopper/) for more information.

    ## Episode End
    ### Termination
    If `terminate_when_unhealthy is True` (the default), the environment terminates when the Hopper is unhealthy.
    The Hopper is unhealthy if any of the following happens:

    1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, otherwise `observation[2:]`) is no longer contained in the closed interval specified by the `healthy_state_range` argument (default is $[-100, 100]$).
    2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[1]`) is no longer contained in the closed interval specified by the `healthy_z_range` argument (default is $[0.7, +\infty]$) (usually meaning that it has fallen).
    3. The angle of the torso (`observation[1]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[2]`) is no longer contained in the closed interval specified by the `healthy_angle_range` argument (default is $[-0.2, 0.2]$).

    ## Reward Space
    The reward is 3-dimensional:
    - 0: Reward for going forward on the x-axis
    - 1: Reward for jumping high on the z-axis
    - 2: Control cost of the action
    If the cost_objective flag is set to False, the reward is 2-dimensional, and the cost is added to other objectives.

    ## Credits:
    - Domain randomization by https://github.com/gabrieletiboni/random-envs
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        cost_objective=True, 
        dr: bool = False, # leave as false for now, will call `reset_random` separately from `reset`
        noisy: bool = False,
        task: Optional[List[float]] = None,
        **kwargs,
    ):
        DREnv.__init__(self)
        EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            task,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        RandomMujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            task=task,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        # ================== Domain Randomization ==================
        self.dr_training = dr
        self.noisy = noisy
        self.noise_level = 1e-4
        self.original_task = np.copy(self.get_task())
        self.nominal_values = np.concatenate([self.original_task])
        self.task_dim = self.nominal_values.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'torsomass', 1: 'thighmass', 2: 'legmass', 3: 'footmass',
                                4: 'damping0', 5: 'damping1', 6: 'damping2', 7: 'friction'}

        self.cost_objective = cost_objective
        self.reward_dim = 3 if cost_objective else 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))
        
        self.set_task_search_bounds() # set the randomization bounds
        
    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               'torsomass': (0.1, 10.0),
               'thighmass': (0.1, 10.0),
               'legmass': (0.1, 10.0),
               'footmass': (0.1, 10.0),
               'damping0': (0.1, 3.),
               'damping1': (0.1, 3.),
               'damping2': (0.1, 3.),
               'friction': (0.1, 3.)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
            'torsomass': 0.001,
            'thighmass': 0.001,
            'legmass': 0.001,
            'footmass': 0.001,
            'damping0': 0.05,
            'damping1': 0.05,
            'damping2': 0.05,
            'friction': 0.01
        }

        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task(self):
        masses = np.array(self.model.body_mass[1:])
        damping = np.array(self.model.dof_damping[3:])
        friction = np.array([self.model.pair_friction[0, 0]])
        return np.concatenate([masses, damping, friction])

    def set_task(self, *task):
        self.model.body_mass[1:] = task[:4]
        self.model.dof_damping[3:] = task[4:7]  # damping on the three actuated joints
        self.model.pair_friction[0, :2] = np.repeat(task[7], 2)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "original_scalar_reward": reward,
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        height = 10 * info["z_distance_from_origin"]
        energy_cost = np.sum(np.square(action))
        if self.cost_objective:
            vec_reward = np.array([x_velocity, height, -energy_cost], dtype=np.float32)
        else:
            vec_reward = np.array([x_velocity, height], dtype=np.float32)
            vec_reward -= self._ctrl_cost_weight * energy_cost

        vec_reward += info["reward_survive"]

        if self.render_mode == "human":
            self.render()

        return observation, vec_reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]
        
        obs = np.concatenate((position, velocity)).ravel()
        if self.noisy:
            obs += np.sqrt(self.noise_level)*np.random.randn(obs.shape[0])

        return obs

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        if self.dr_training:
            self.set_random_task() # Sample new dynamics
            
        return self._get_obs()
    
    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }
    
    def reset_random(self):
        self.set_random_task()


gym.envs.register(
        id="RandomHopper-v0",
        entry_point="%s:RandomHopperEnv" % __name__,
        max_episode_steps=500
)

gym.envs.register(
        id="RandomHopperNoisy-v0",
        entry_point="%s:RandomHopperEnv" % __name__,
        max_episode_steps=500,
        kwargs={"noisy": True}
)