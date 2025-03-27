import json
import rosbag

import numpy as np
import pandas as pd

from typing import List, Union


def read_json(json_path: str) -> list:

    if not isinstance(json_path, str):
        raise TypeError("Input 'json_path' in read_json function must be a string.")
    
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    
    return data


def json2dataframe(json_path: str,
                   column_names: List[str]) -> pd.DataFrame:
    
    if not isinstance(json_path, str):
        raise TypeError("Input 'json_path' in json2dataframe function must be a string.")
    if not isinstance(column_names, list):
        raise TypeError("Input 'column_names' in json2dataframe function must be a list of strings.")

    # set the display options to show longer decimals
    pd.set_option("display.float_format",
                  "{:.12f}".format)
    
    # only one trajectory and returns list of lists
    trajectory = read_json(json_path)
    
    df = pd.DataFrame(index=range(len(trajectory)),
                      columns=column_names)
    
    timestamps = []
    traj_idx = 0
    
    # loop through each sample element in the trajectory
    for sample in trajectory:

        # some trajectories could be missing position data
        if len(sample) != 5:
            # remove the last row from df if it exists
            df = df.drop(df.index[-1])
            continue

        entry = {
            column_names[0]: int(sample[4]["timestamp"]),
            column_names[1]: sample[0]["message"],
            column_names[2]: sample[1]["message"],
            column_names[3]: sample[3]["message"],
        }

        arm_action = sample[4]["message"]["position"]
        entry.update({
            column_names[4]: arm_action["x"],
            column_names[5]: arm_action["y"],
            column_names[6]: arm_action["z"]
        })

        arm_rotation = sample[4]["message"]["orientation"]
        entry.update({
            column_names[7]: arm_rotation["x"],
            column_names[8]: arm_rotation["y"],
            column_names[9]: arm_rotation["z"],
            column_names[10]: arm_rotation["w"]
        })

        df.loc[traj_idx] = entry
        traj_idx += 1

        timestamps.append(int(sample[4]["timestamp"]))
    
    # convert timestamps to seconds and store in 'timestamp' column
    df[column_names[0]] = timestamp2second(timestamps)
    
    return df


def timestamp2second(timestamps: list) -> list:

    if not isinstance(timestamps, list):
        raise TypeError("Input 'timestamps' in timestamp2second function must be a list.")

    start_time = min(timestamps)

    # convert nanoseconds to seconds
    time_seconds = [(t - start_time) / 1e9 for t in timestamps]

    return time_seconds


def extract_state_vector(df: pd.DataFrame,
                         traj_idx_column: str,
                         state_idx_column: str,
                         state_columns: List[str],
                         norm_value_list: List[float]) -> pd.DataFrame:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' in extract_state_vector function must be a pandas dataframe.")
    if not isinstance(traj_idx_column, str):
        raise TypeError("Input 'traj_idx_column' in extract_state_vector function must be a string.")
    if not isinstance(state_idx_column, str):
        raise TypeError("Input 'state_idx_column' in extract_state_vector function must be a string.")
    if not isinstance(state_columns, list):
        raise TypeError("Input 'state_columns' in extract_state_vector function must be a list of strings.")
    if not isinstance(norm_value_list, list):
        raise TypeError("Input 'norm_value_list' in extract_state_vector function must be a list of float values.")
    
    traj_idxs = df[traj_idx_column].iloc[0]
    state_numbers = df[state_idx_column]
    state_df = df[state_columns]

    state_array = state_df.to_numpy()

    # normalize state vector to range [0, 1]
    normalized_state_array = normalize_state(state=state_array,
                                             norm_value_list=norm_value_list)
    normalized_state_df = pd.DataFrame(normalized_state_array,
                                       columns=state_columns)
    
    # add trajectory index number column and state number column
    normalized_state_df[traj_idx_column] = traj_idxs
    normalized_state_df[state_idx_column] = state_numbers.values

    return normalized_state_df


def extract_action_vector(df: pd.DataFrame,
                          traj_idx_column: str,
                          state_idx_column: str,
                          action_columns: List[str],
                          norm_range_list: List[List[float]]) -> pd.DataFrame:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' in extract_action_vector function must be a pandas dataframe.")
    if not isinstance(traj_idx_column, str):
        raise TypeError("Input 'traj_idx_column' in extract_action_vector function must be a string.")
    if not isinstance(state_idx_column, str):
        raise TypeError("Input 'state_idx_column' in extract_action_vector function must be a string.")
    if not isinstance(action_columns, list):
        raise TypeError("Input 'action_columns' in extract_action_vector function must be a list of strings.")
    if not isinstance(norm_range_list, list):
        raise TypeError("Input 'norm_range_list' in extract_action_vector function must be a list of list of float values.")
    
    traj_idxs = df[traj_idx_column].iloc[0]
    state_numbers = df[state_idx_column]
    action_df = df[action_columns]

    action_array = action_df.to_numpy()

    # normalize each element of the action vector to range [-1, 1]
    normalized_state_array = normalize_action(action=action_array,
                                              norm_range_list=norm_range_list)
    normalized_action_df = pd.DataFrame(normalized_state_array,
                                        columns=action_columns)
    
    # add trajectory index number column and state number column
    normalized_action_df[traj_idx_column] = traj_idxs
    normalized_action_df[state_idx_column] = state_numbers.values

    return normalized_action_df


def normalize_state(state: np.ndarray,
                    norm_value_list: List[float]) -> np.ndarray:
    
    if not isinstance(state, np.ndarray):
        raise TypeError("Input 'state' must be a numpy array.")
    if not isinstance(norm_value_list, list):
        raise TypeError("Input 'norm_value_list' must be a list.")
    if state.shape[-1] != len(norm_value_list):
        raise ValueError("Length of 'state' and 'norm_value_list' must be the same.")
    
    normed_state = state / np.array(norm_value_list)

    return normed_state


def normalize_action(action: np.ndarray,
                     norm_range_list: List[List[float]]) -> np.ndarray:
    
    if not isinstance(action, np.ndarray):
        raise TypeError("Input 'action' must be a numpy array.")
    if not isinstance(norm_range_list, list):
        raise TypeError("Input 'norm_range_list' must be a list.")
    if len(action.shape) != 2 or action.shape[1] != len(norm_range_list):
        raise ValueError("Shape of 'action' and 'norm_range_list' must be compatible.")
    
    normed_action = np.zeros_like(action,
                                  dtype=np.float64)
    
    for i in range(len(norm_range_list)):
        min_val, max_val = norm_range_list[i]
        normed_action[:, i] = ((2 * (action[:, i] - min_val)) / (max_val - min_val)) - 1
    
    return normed_action


def denormalize_state(state_norm: np.ndarray,
                      norm_value_list: List[float]) -> np.ndarray:
    
    if not isinstance(state_norm, np.ndarray):
        raise TypeError("Input 'state_norm' must be a numpy array.")
    if not isinstance(norm_value_list, list):
        raise TypeError("Input 'norm_value_list' must be a list.")
    if state_norm.shape[-1] != len(norm_value_list):
        raise ValueError("Length of 'state_norm' and 'norm_value_list' must be the same.")
    
    denormed_state = state_norm * np.array(norm_value_list)

    return denormed_state


def denormalize_action(action_norm: np.ndarray,
                       norm_range_list: List[List[float]]) -> np.ndarray:
    
    if not isinstance(action_norm, np.ndarray):
        raise TypeError("Input 'action_norm' must be a numpy array.")
    if not isinstance(norm_range_list, list):
        raise TypeError("Input 'norm_range_list' must be a list.")
    if len(action_norm.shape) != 2 or action_norm.shape[-1] != len(norm_range_list):
        raise ValueError("Shape of 'action_norm' and 'norm_range_list' must be compatible.")
    
    denormed_action = np.zeros_like(action_norm, dtype=np.float64)
    
    for i in range(len(norm_range_list)):
        min_val, max_val = norm_range_list[i]
        denormed_action[:, i] = ((action_norm[:, i] + 1) * (max_val - min_val) / 2) + min_val

    return denormed_action


def rosbag2json(bag_file: str,
                output_json_file: str,
                ros_topics: List[str]) -> None:
    
    if not isinstance(bag_file, str):
        raise TypeError("Input 'bag_file' in rosbag2json function must be a string path.")
    if not isinstance(output_json_file, str):
        raise TypeError("Input 'output_json_file' in rosbag2json function must be a string path.")
    if not isinstance(ros_topics, list):
        raise TypeError("Input 'ros_topics' in rosbag2json function must be a list of strings.")
    
    with rosbag.Bag(bag_file, "r") as bag:
        messages, message_list = [], []

        for topic, msg, t in bag.read_messages():
            
            if topic not in ros_topics:
                continue
            
            if hasattr(msg, "data"):
                message = msg.data
                
                # check if the value is numeric and has at most one decimal point
                if not is_float(message):
                    continue
                
            elif hasattr(msg, "position"):
                message = {
                    "position": {
                        "x": msg.position.x,
                        "y": msg.position.y,
                        "z": msg.position.z
                    },
                    "orientation": {
                        "x": msg.orientation.x,
                        "y": msg.orientation.y,
                        "z": msg.orientation.z,
                        "w": msg.orientation.w
                    }
                }
            
            else:
                messages.append(message_list)
                message_list = []
                
                # ignore arm_state for now
                continue
            
            message_dict = {
                "topic": topic,
                "timestamp": str(t),
                "message": message
            }
            message_list.append(message_dict)
    
    with open(output_json_file, "w") as json_file:
        json.dump(messages, json_file,  indent=2)


def is_float(value) -> bool:

    # check if the value is numeric and has at most one decimal point
    binary = str(value).replace(".", "", 1).isnumeric()

    return binary


def discritize_dataframe(df: pd.DataFrame,
                         return_n_rows: int) -> pd.DataFrame:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' in discritize_dataframe function must be a pandas dataframe.")
    if not isinstance(return_n_rows, int):
        raise TypeError("Input 'return_n_rows' in discritize_dataframe function must be an integer.")
    
    # get the number of rows in the dataframe
    num_rows = df.shape[0]

    # get the number of rows to skip (interval number)
    skip_rows = num_rows // return_n_rows

    # select rows at regular intervals
    df_subset = df.iloc[::skip_rows]

    return df_subset


def shift_action_label(df: pd.DataFrame,
                       action_columns: List[str],
                       shift_amount: int) -> pd.DataFrame:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' in shift_action_label function must be a pandas dataframe.")
    if not isinstance(action_columns, list):
        raise TypeError("Input 'action_columns' in shift_action_label function must be a list of strings.")
    if not isinstance(shift_amount, int):
        raise TypeError("Input 'shift_amount' in shift_action_label function must be an integer.")
    
    # shift the action label by the specified amount
    df[action_columns] = df[action_columns].shift(shift_amount)

    # fill NaN parts with the corresponding values from the last valid row
    for column in action_columns:
        df[column].fillna(df[column],
                          inplace=True)
    
    # remove the last shift amount of rows that are NaN
    df = df.dropna()
    
    return df


def generate_gaussian_random_variables(means: np.ndarray,
                                       variances: np.ndarray,
                                       num_samples: int = 1) -> np.ndarray:
    
    if not isinstance(means, np.ndarray):
        raise TypeError("Input 'means' in generate_gaussian_random_variables function must be a numpy array.")
    if not isinstance(variances, np.ndarray):
        raise TypeError("Input 'variances' in generate_gaussian_random_variables function must be a numpy array.")
    if not isinstance(num_samples, int):
        raise TypeError("Input 'num_samples' in generate_gaussian_random_variables function must be an integer.")
    
    covariance_matrix = np.diag(variances)
    samples = np.random.multivariate_normal(means,
                                            covariance_matrix,
                                            num_samples)
    
    return samples


def compute_annealing_factor(epoch: int,
                             max_epochs: int) -> float:
    
    if not isinstance(epoch, int):
        raise TypeError("Input 'epoch' in compute_annealing_factor function must be an instance of int.")
    if not isinstance(max_epochs, int):
        raise TypeError("Input 'max_epochs' in compute_annealing_factor function must be an instance of int.")
    
    initial_std_weight = 0.05
    final_std_weight = 1.0
    
    fraction = min(1.0,
                   epoch / max_epochs)
    
    annealing_factor = initial_std_weight + fraction * (final_std_weight - initial_std_weight)

    return annealing_factor


def squash_reward_to_performance(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:

    if isinstance(value, float):
        # squash a single float value
        squashed = (2 * value) - 1
    
    elif isinstance(value, np.ndarray):
        # squash each element of the array
        squashed = (2 * value) - 1
    
    else:
        raise TypeError("Input 'value' in squash_reward_to_performance function must be either a float or a NumPy array.")
    
    return squashed
