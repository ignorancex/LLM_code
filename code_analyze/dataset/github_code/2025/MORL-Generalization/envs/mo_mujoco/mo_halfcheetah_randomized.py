"""Implementation of the HalfCheetah environment supporting
domain randomization optimization.

Randomizations:
    - 7 mass links
    - 1 friction coefficient (sliding)

For all details: https://www.gymlibrary.ml/environments/mujoco/half_cheetah/
"""
from typing import Dict, Union
import numpy as np
import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium.spaces import Box
from envs.mo_mujoco.utils.random_mujoco_env import RandomMujocoEnv
from morl_generalization.algos.dr import DREnv
from copy import deepcopy

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class MOHalfCheetahDR(RandomMujocoEnv, DREnv, EzPickle):
    """
    ## Description
    Multi-objective version of Gymansium's Mujoco Cheetah environment with randomizable environment parameters.

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for running forward
    - 1: Control cost of the action

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
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        dr: bool = False, # leave as false for now, will call `reset_random` separately from `reset`
        noisy: bool = False,
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
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

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
        self.original_masses = np.copy(self.model.body_mass[1:])
        self.original_friction = np.array([0.4])
        self.nominal_values = np.concatenate([self.original_masses, self.original_friction])
        self.task_dim = self.nominal_values.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'torso', 1: 'bthigh', 2: 'bshin', 3: 'bfoot', 4: 'fthigh', 5: 'fshin', 6: 'ffoot', 7: 'friction'}

        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

        self.set_task_search_bounds() # set the randomization bounds


    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
               'torso': (0.1, 10.0),
               'bthigh': (0.1, 10.0),
               'bshin': (0.1, 10.0),
               'bfoot': (0.1, 10.0),
               'fthigh': (0.1, 10.0),
               'fshin': (0.1, 10.0),
               'ffoot': (0.1, 10.0),
               'friction': (0.1, 2.0),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torso': 0.1,
                    'bthigh': 0.1,
                    'bshin': 0.1,
                    'bfoot': 0.1,
                    'fthigh': 0.1,
                    'fshin': 0.1,
                    'ffoot': 0.1,
                    'friction': 0.1,
        }

        return lowest_value[self.dyn_ind_to_name[index]]

    def get_task(self):
        masses = np.array(self.model.body_mass[1:])
        friction = np.array(self.model.pair_friction[0,0])
        task = np.append(masses, friction)
        return task

    def set_task(self, *task):
        self.model.body_mass[1:] = task[:-1]
        self.model.pair_friction[0:2,0:2] = task[-1]

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    
    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, reward_info

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {
            "original_scalar_reward": reward,
            "x_position": x_position_after, 
            "x_velocity": x_velocity, 
            **reward_info
        }
        terminated = False
        vec_reward = np.array([info["reward_forward"], (1/self._ctrl_cost_weight)*info["reward_ctrl"]], dtype=np.float32)
        
        if self.render_mode == "human":
            self.render()
        return observation, vec_reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

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
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
        }
    
    def reset_random(self):
        self.set_random_task()
        # self.reset()
