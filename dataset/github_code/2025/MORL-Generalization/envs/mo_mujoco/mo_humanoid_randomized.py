"""Implementation of the Humanoid environment supporting
domain randomization optimization.

Randomizations:
    - 13 mass links
    - 17 joint damping

First 45 dims in state space are qpos and qvel

For all details: https://www.gymlibrary.ml/environments/mujoco/humanoid/
"""
from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium.spaces import Box
from envs.mo_mujoco.utils.random_mujoco_env import RandomMujocoEnv
from morl_generalization.algos.dr import DREnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

class MOHumanoidDR(RandomMujocoEnv, DREnv, EzPickle):
    """
    ## Description
    Multi-objective version of Gymnasium's Mujoco Humanoid environment with randomizable environment parameters.

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for running forward
    - 1: Control cost of the action

    ## Important Changes to Note
    - The original Gymnasium Humanoid environment has healthy_reward=5.0, but that dominates the control cost and forward reward. 
      We made the following changes: healthy_reward=2.0 + ctrl_cost_weight=1e-3. This makes sure there is differentiation between a SORL and MORL agent.
      Does not affect convergence of single-objective agent in default environment (tested)
    - The original Gymnasium Humanoid environment has contact_cost_weight = 5e-7, which is negligible. We have set it to 0.0.

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
        xml_file: str = "humanoid.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 1e-3,
        contact_cost_weight: float = 0.0,
        contact_cost_range: Tuple[float, float] = (-np.inf, 10.0),
        healthy_reward: float = 2.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        dr: bool = False, # leave as false for now, will call `reset_random` separately from `reset`
        noisy: bool = False,
        **kwargs,
    ):
        DREnv.__init__(self)
        EzPickle.__init__(self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

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

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6)
            * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
        }

        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

        # ================== Domain Randomization ==================
        self.dr_training = dr
        self.noisy = noisy
        self.noise_level = 1e-3
        self.original_masses = np.copy(self.model.body_mass[1:])
        self.original_damping = np.copy(self.model.dof_damping[6:])

        self.task_dim = self.original_masses.shape[0] + self.original_damping.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'mass0', 1: 'mass1', 2: 'mass2', 3: 'mass3',
                                 4: 'mass4', 5: 'mass5', 6: 'mass6', 7: 'mass7',
                                 8: 'mass8', 9: 'mass9', 10: 'mass10', 11: 'mass11', 12: 'mass12',
                                 13: 'damp1', 14: 'damp2', 15: 'damp3', 16: 'damp4', 17: 'damp5',
                                 18: 'damp6', 19: 'damp7', 20: 'damp8', 21: 'damp9', 22: 'damp10',
                                 23: 'damp11', 24: 'damp12',  25: 'damp13', 26: 'damp14', 27: 'damp15',
                                 28: 'damp16', 29: 'damp17'}

        self.set_task_search_bounds() # set the randomization bounds


    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized,
        the stdev bounds are set accordingly in dropo.
        """
        search_bounds_mean = {
               'mass0': (0.5, 10.0),
               'mass1': (0.5, 10.0),
               'mass2': (0.5, 10.0),
               'mass3': (0.5, 10.0),
               'mass4': (0.5, 10.0),
               'mass5': (0.5, 10.0),
               'mass6': (0.5, 10.0),
               'mass7': (0.5, 10.0),
               'mass8': (0.5, 10.0),
               'mass9': (0.5, 10.0),
               'mass10': (0.5, 10.0),
               'mass11': (0.5, 10.0),
               'mass12': (0.5, 10.0),

               'damp1': (1, 10.0),
               'damp2': (1, 10.0),
               'damp3': (1, 10.0),
               'damp4': (1, 10.0),
               'damp5': (1, 10.0),
               'damp6': (1, 10.0),
               'damp8': (1, 10.0),
               'damp9': (1, 10.0),
               'damp10': (1, 10.0),

               'damp7': (.2, 5.0),
               'damp11': (.2, 5.0),
               'damp12': (.2, 5.0),
               'damp13': (.2, 5.0),
               'damp14': (.2, 5.0),
               'damp15': (.2, 5.0),
               'damp16': (.2, 5.0),
               'damp17': (.2, 5.0),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'mass0': 0.2,
                    'mass1': 0.2,
                    'mass2': 0.2,
                    'mass3': 0.2,
                    'mass4': 0.2,
                    'mass5': 0.2,
                    'mass6': 0.2,
                    'mass7': 0.2,
                    'mass8': 0.2,
                    'mass9': 0.2,
                    'mass10': 0.2,
                    'mass11': 0.2,
                    'mass12': 0.2,

                    'damp1': 0.8,
                    'damp2': 0.8,
                    'damp3': 0.8,
                    'damp4': 0.8,
                    'damp5': 0.8,
                    'damp6': 0.8,
                    'damp8': 0.8,
                    'damp9': 0.8,
                    'damp10': 0.8,

                    'damp7': .15,
                    'damp11': .15,
                    'damp12': .15,
                    'damp13': .15,
                    'damp14': .15,
                    'damp15': .15,
                    'damp16': .15,
                    'damp17': .15,
        }

        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task(self):
        masses = np.array( self.model.body_mass[1:] )
        damping = np.array( self.model.dof_damping[6:]  )
        return np.append(masses, damping)

    def set_task(self, *task):
        self.model.body_mass[1:] = task[:13]
        self.model.dof_damping[6:] = task[13:]

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost
    
    @property
    def contact_cost(self):
        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy
    
    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_ctrl_contact": -costs,
        }

        return reward, reward_info


    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy

        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "original_scalar_reward": reward,
            **reward_info
        }

        vec_reward = np.array([info["x_velocity"], (1/self._ctrl_cost_weight)*info["reward_ctrl"]], dtype=np.float32)
        vec_reward += self.healthy_reward

        if self.render_mode == "human":
            self.render()

        return observation, vec_reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = np.array([])

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self.noisy:
            return np.concatenate(
                (
                    position + np.sqrt(self.noise_level)*np.random.randn(position.shape[0]), # 22
                    velocity + np.sqrt(self.noise_level)*np.random.randn(velocity.shape[0]), # 23
                    com_inertia,
                    com_velocity,
                    actuator_forces,
                    external_contact_forces
                )
        )

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

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
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def reset_random(self):
        self.set_random_task()
        # self.reset()


gym.envs.register(
        id="RandomHumanoid-v0",
        entry_point="%s:RandomHumanoidEnv" % __name__,
        max_episode_steps=500
)

gym.envs.register(
        id="RandomHumanoidNoisy-v0",
        entry_point="%s:RandomHumanoidEnv" % __name__,
        max_episode_steps=500,
        kwargs={"noisy": True}
)