import os
import sys

import numpy as np

import torch

from typing import Tuple

sys.path.insert(0,
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__))
                        if "__file__" in locals() else os.getcwd()))

from utils import constants, common


class RobotEnvironment(object):

    def __init__(self) -> object:
        
        # state and action normalization parameters
        self.state_norms = [constants.MAX_DISTANCE_TO_OBJECT,
                            constants.MAX_DISTANCE_TO_TARGET,
                            constants.MAX_DISTANCE_TO_GROUND]
        self.action_norms = [constants.END_EFFECTOR_POSITION_RANGE_X,
                             constants.END_EFFECTOR_POSITION_RANGE_Y,
                             constants.END_EFFECTOR_POSITION_RANGE_Z,
                             [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
        
        # define initial state means and variances
        self.initial_state_means = constants.INITIAL_STATE_MEANS
        self.initial_state_variances = constants.INITIAL_STATE_VARIANCES

        # define specific ranges for squashing
        self.min_ranges = constants.INITIAL_STATE_MIN_RANGES
        self.max_ranges = constants.INITIAL_STATE_MAX_RANGES

        self.max_steps = constants.TRAJECTORY_SIZE
        self.state_counter = 0

        self.reward_network = None
        self.is_reward_inference = True
    
    def reset(self) -> torch.Tensor:

        self.state_counter = 0

        state = torch.Tensor(self.generate_random_state(count=1)[0])

        return state
    
    def step(self,
             state: torch.Tensor,
             action: torch.Tensor) -> Tuple[torch.Tensor,
                                            torch.Tensor,
                                            bool]:
        
        reward = self.get_reward(state=state,
                                 action=action)

        action_denorm = common.denormalize_action(action_norm=action.unsqueeze(0).detach().numpy(),
                                                  norm_range_list=self.action_norms)
        
        next_state_denorm = self.get_next_state(action_denorm=action_denorm[0],
                                                obstacle_location=np.array(constants.OBSTACLE_LOCATION),
                                                target_location=np.array(constants.TARGET_LOCATION))
        
        next_state = torch.Tensor(common.normalize_state(state=next_state_denorm,
                                                         norm_value_list=self.state_norms))
        
        self.state_counter += 1

        done = self.get_termination(action_denorm=action_denorm[0],
                                    next_state_denorm=next_state_denorm)

        return next_state, reward, done
    
    def get_reward(self,
                   state: torch.Tensor,
                   action: torch.Tensor) -> torch.Tensor:
        
        reward_value = self.reward_network.estimate_reward(state=state,
                                                           action=action,
                                                           is_reward_inference=self.is_reward_inference)
        
        return reward_value

    def get_next_state(self,
                       action_denorm: np.ndarray,
                       obstacle_location: np.ndarray,
                       target_location: np.ndarray) -> np.ndarray:
        
        # because both obstacle location and action prediction locations are computed w.r.t. the robot base,
        # we could direcly calculate euclidean distance between them without any transformation
        object_distance = np.linalg.norm(obstacle_location - action_denorm[:3])
        target_distance = np.linalg.norm(target_location - action_denorm[:3])
        ground_distance = action_denorm[2] + constants.ROBOT_BASE_HEIGHT
        
        next_state_denorm = np.array([object_distance,
                                      target_distance,
                                      ground_distance])
        
        return next_state_denorm

    def get_termination(self,
                        action_denorm: np.ndarray,
                        next_state_denorm: np.ndarray) -> bool:
        
        x_out_of_range = not constants.END_EFFECTOR_POSITION_RANGE_X[0] < action_denorm[0] < constants.END_EFFECTOR_POSITION_RANGE_X[1]
        y_out_of_range = not constants.END_EFFECTOR_POSITION_RANGE_Y[0] < action_denorm[1] < constants.END_EFFECTOR_POSITION_RANGE_Y[1]
        z_out_of_range = not constants.END_EFFECTOR_POSITION_RANGE_Z[0] < action_denorm[2] < constants.END_EFFECTOR_POSITION_RANGE_Z[1]

        out_of_object_reach = next_state_denorm[0] > constants.MAX_DISTANCE_TO_OBJECT
        out_of_target_reach = next_state_denorm[1] > constants.MAX_DISTANCE_TO_TARGET
        out_of_ground_reach = next_state_denorm[2] > constants.MAX_DISTANCE_TO_GROUND

        is_trajectory_limit_reached = self.state_counter >= self.max_steps

        is_done = any((x_out_of_range, y_out_of_range, z_out_of_range,
                       out_of_object_reach, out_of_target_reach, out_of_ground_reach,
                       is_trajectory_limit_reached))
        
        return is_done
    
    def generate_random_state(self,
                              count: int) -> np.ndarray:
        
        def squash_function(x, min_value, max_value):
            return 0.5 * (x + 1) * (max_value - min_value) + min_value
        
        state_size = len(self.initial_state_means)

        # generate random samples for the initial state
        state_init_rand = np.random.normal(loc=self.initial_state_means,
                                           scale=np.sqrt(self.initial_state_variances),
                                           size=(count, state_size))

        # squash the random samples to the specified ranges
        state_squashed = np.zeros_like(state_init_rand)

        for i in range(state_size):
            state_squashed[:, i] = np.clip(squash_function(x=state_init_rand[:, i],
                                                           min_value=self.min_ranges[i],
                                                           max_value=self.max_ranges[i]),
                                           self.min_ranges[i],
                                           self.max_ranges[i])
        
        return state_squashed

    def set_reward_network(self,
                           reward_network: torch.nn.Module) -> None:
        
        self.reward_network = reward_network
    
    def calculate_continuous_reward(self,
                                    state_vector: torch.Tensor) -> torch.Tensor:
        
        # extract state values
        distance_to_obstacle, distance_to_target, distance_to_ground = state_vector

        # define scaling factors based on the problem requirements
        scale_obstacle = 0.1
        scale_target = 0.5
        scale_ground = 0.2

        # calculate individual rewards
        reward_obstacle = torch.exp(-scale_obstacle * distance_to_obstacle)
        
        # use negative distance as the reward term for the target
        reward_target = -distance_to_target * scale_target
        
        reward_ground = torch.exp(-scale_ground * distance_to_ground)

        # combine individual rewards into a total reward
        total_reward = reward_target * reward_ground / (reward_obstacle + 1e-8)

        return total_reward.unsqueeze(0)
