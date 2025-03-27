import numpy as np
import pandas as pd

import torch

from typing import List, Tuple
from torch.utils.data import Dataset
from utils import common, constants


class PolicyDatasetLoader(Dataset):
    
    def __init__(self,
                 demo_data_json_paths: List[str]) -> object:
        
        self.state_columns = constants.STATE_COLUMNS
        self.action_columns = constants.ACTION_COLUMNS
        self.end_effector_pose_columns = constants.ACTION_COLUMNS[:3]

        self.state_norms = [constants.MAX_DISTANCE_TO_OBJECT,
                            constants.MAX_DISTANCE_TO_TARGET,
                            constants.MAX_DISTANCE_TO_GROUND]
        self.action_norms = [constants.END_EFFECTOR_POSITION_RANGE_X,
                             constants.END_EFFECTOR_POSITION_RANGE_Y,
                             constants.END_EFFECTOR_POSITION_RANGE_Z,
                             [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
        
        self.state_number_column = constants.STATE_NUMBER_COLUMN
        self.traj_index_column = constants.NUMBER_TRAJECTORY_COLUMN
        self.trajectory_length = constants.TRAJECTORY_SIZE
        
        self.demo_state_data, self.demo_action_data = self.load_data(json_paths=demo_data_json_paths,
                                                                     column_names=constants.COLUMN_NAMES)
        
        self.dataset_size = len(self.demo_state_data) if not self.demo_state_data.empty else 0

        print("\n================== Policy Dataset Loader ==================\n")
        print("Number of Trajectories: ", len(demo_data_json_paths))
        print("Each Trajectory Length: ", self.trajectory_length)
        print("Full Demo Dataset Size: ", self.dataset_size)
    
    def __len__(self) -> int:

        return self.dataset_size
    
    def __getitem__(self,
                    idx: int) -> Tuple[torch.FloatTensor,
                                       torch.FloatTensor,
                                       int,
                                       int]:
        
        # get normalized state and action vectors as dataframe
        sample_states = self.demo_state_data[self.state_columns].iloc[idx].astype(np.float64)
        sample_actions = self.demo_action_data[self.action_columns].iloc[idx].astype(np.float64)

        # get trajectory index number and state number in that trajectory
        traj_idx_number = self.demo_state_data[self.traj_index_column].iloc[idx]
        state_number = self.demo_state_data[self.state_number_column].iloc[idx]
        
        # convert state and action to torch tensors
        state_tensor = torch.FloatTensor(sample_states.values.flatten())
        action_tensor = torch.FloatTensor(sample_actions.values.flatten())

        return state_tensor, action_tensor, traj_idx_number, state_number
    
    def load_data(self,
                  json_paths: List[str],
                  column_names: List[str]) -> Tuple[pd.DataFrame,
                                                    pd.DataFrame]:
        
        state_dfs, action_dfs = [], []

        # loop through every trajectory data specified in json_paths
        for traj_idx, json_path in enumerate(json_paths):

            df = common.json2dataframe(json_path=json_path,
                                       column_names=column_names)
            
            # discritize the trajectory into equally spaced points
            df = common.discritize_dataframe(df=df,
                                             return_n_rows=self.trajectory_length)
            
            # add a new column to store the trajectory index number
            df[self.traj_index_column] = traj_idx

            # reset the index and store it in a new column
            df[self.state_number_column] = df.reset_index().index
            
            # TODO: remove the following correction functions after bug in data collection is solved
            # currently, this setup works because of the deterministic experimental system
            df = self.make_state_correction(trajectory_df=df)

            # correct action label correspondence, as we want to predict next action given the state
            df = common.shift_action_label(df=df,
                                           action_columns=self.action_columns,
                                           shift_amount=constants.ACTION_LABEL_SHIFT_IDX)
            
            state_dfs.append(
                common.extract_state_vector(df=df,
                                            traj_idx_column=self.traj_index_column,
                                            state_idx_column=self.state_number_column,
                                            state_columns=self.state_columns,
                                            norm_value_list=self.state_norms))
            action_dfs.append(
                common.extract_action_vector(df=df,
                                             traj_idx_column=self.traj_index_column,
                                             state_idx_column=self.state_number_column,
                                             action_columns=self.action_columns,
                                             norm_range_list=self.action_norms))
        
        # all trajectories are concatenated into one dataframe
        concatenated_state_df = pd.concat(state_dfs,
                                          ignore_index=True)
        concatenated_action_df = pd.concat(action_dfs,
                                          ignore_index=True)
        
        return concatenated_state_df, concatenated_action_df
    
    def make_state_correction(self,
                              trajectory_df: pd.DataFrame) -> pd.DataFrame:
        
        # manually update distance to the obstacle location as state vector is not realiable, but action vector is reliable
        trajectory_df = self.correct_distance_to_object(df=trajectory_df) if "distance_to_object" in self.state_columns else trajectory_df
        # manually update distance to the target location as state vector is not realiable, but action vector is reliable
        trajectory_df = self.correct_distance_to_target(df=trajectory_df) if "distance_to_target" in self.state_columns else trajectory_df
        # manually correct distance to the ground as state vector is not realiable, but action vector is reliable
        trajectory_df = self.correct_distance_to_ground(df=trajectory_df) if "distance_to_ground" in self.state_columns else trajectory_df
        
        return trajectory_df
    
    def correct_distance_to_object(self,
                                   df: pd.DataFrame) -> pd.DataFrame:
        
        # correct Euclideandistance to the object given constant location of the object (obstacle)
        distances = np.linalg.norm(np.array(constants.OBSTACLE_LOCATION) - np.array(df[self.end_effector_pose_columns],
                                                                                    dtype=np.float64), axis=1)
        df[self.state_columns[0]] = distances
        
        return df
    
    def correct_distance_to_target(self,
                                   df: pd.DataFrame) -> pd.DataFrame:
        
        # calculate the Euclidean distance for each row
        distances = np.linalg.norm(np.array(constants.TARGET_LOCATION) - np.array(df[self.end_effector_pose_columns],
                                                                                  dtype=np.float64), axis=1)
        # correct distance to the target given constant location on the ground
        df[self.state_columns[1]] = distances

        return df
    
    def correct_distance_to_ground(self,
                                   df: pd.DataFrame) -> pd.DataFrame:
        
        # correct distance to the ground given the base height of the robot
        distances = df[self.end_effector_pose_columns[2]] + constants.ROBOT_BASE_HEIGHT
        df[self.state_columns[2]] = distances

        return df
