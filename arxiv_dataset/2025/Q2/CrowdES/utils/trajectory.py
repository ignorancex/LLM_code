import os
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from pykalman import KalmanFilter

from utils.homography import image2world, world2image


def augment_trajectory(obs_traj, pred_traj, flip=True, reverse=True):
    """Flip and reverse the trajectory

    Params:
        obs_traj (torch.Tensor): observed trajectory with shape (num_peds, obs_len, 2)
        pred_traj (torch.Tensor): predicted trajectory with shape (num_peds, pred_len, 2)
        flip (bool): whether to flip the trajectory
        reverse (bool): whether to reverse the trajectory

    Returns:
        obs_traj (torch.Tensor): augmented observed trajectory
        pred_traj (torch.Tensor): augmented predicted trajectory
    """

    if flip:
        obs_traj = torch.cat([obs_traj, obs_traj * torch.FloatTensor([[[1, -1]]])], dim=0)
        pred_traj = torch.cat([pred_traj, pred_traj * torch.FloatTensor([[[1, -1]]])], dim=0)
    elif reverse:
        full_traj = torch.cat([obs_traj, pred_traj], dim=1)  # NTC
        obs_traj = torch.cat([obs_traj, full_traj.flip(1)[:, :obs_traj.size(1)]], dim=0)
        pred_traj = torch.cat([pred_traj, full_traj.flip(1)[:, obs_traj.size(1):]], dim=0)
    return obs_traj, pred_traj


def nearest_nonzero_idx_deprecated(a, x, y):
    """
    Find the nearest non-zero index in a 2D array.

    Params:
        a (np.ndarray): 2D numpy array where non-zero elements are considered.
        x (int): x-coordinate (column index).
        y (int): y-coordinate (row index).

    Returns:
        tuple: (x, y) coordinates of the nearest non-zero element in the array.
    """

    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    
    try:
        if 0 <= x < a.shape[1] and 0 <= y < a.shape[0]:
            if a[y, x] != 0:
                return x, y
    except IndexError:
        pass

    r, c = np.nonzero(a)
    min_idx = ((c - x)**2 + (r - y)**2).argmin()
    return c[min_idx], r[min_idx]


def preprocess_kdtree(a):
    """
    Preprocess a 2D array to create a KD-tree for fast nearest neighbor search.

    Params:
        a (np.ndarray): 2D numpy array where non-zero elements are considered for KD-tree.

    Returns:
        cKDTree: A cKDTree object built on the non-zero elements of the array.
    """

    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise ValueError('Input must be a 2D numpy array.')

    nonzero_y, nonzero_x = np.nonzero(a)  # Find indices of non-zero elements
    kdtree = cKDTree(np.column_stack((nonzero_x, nonzero_y)))  # Build KD-tree on non-zero points
    return kdtree


def batched_nearest_nonzero_idx_kdtree(kdtree, coords):
    """
    Find the nearest non-zero indices in a 2D array using a pre-built KD-tree.

    Params:
        kdtree (cKDTree): cKDTree object built on the non-zero elements of the array.
        coords (np.ndarray): Array of shape (..., 2) where each row is a coordinate (x, y)
                             for which to find the nearest non-zero index.
   
    Returns:
        np.ndarray: Array of shape (..., 2) containing the nearest non-zero indices (x, y).
    """

    if not isinstance(kdtree, cKDTree):
        raise ValueError('kdtree must be a cKDTree object.')
    
    distances, indices = kdtree.query(coords)
    nearest_points = kdtree.data[indices]
    return nearest_points.astype(int)


def image_to_world_trajectory(traj, H):
    """
    Transform trajectory from image to world coordinates.

    Params:
        traj (pd.DataFrame | np.ndarray | list): Trajectory data.
            - If pd.DataFrame, it should have columns=['x', 'y', 'frame', 'agent_id', 'scene_id'].
            - If np.ndarray, it should be of shape (..., 2) where each row is (x, y).
            - If list, it should contain numpy arrays of shape (..., 2).
        H (np.ndarray): Homography matrix of shape (3, 3).

    Returns:
        pd.DataFrame | np.ndarray | list: Trajectory in world coordinates.
    """

    if type(traj) == pd.DataFrame:
        traj_meter = traj.copy()
        traj_meter[['x', 'y']] = image2world(traj[['x', 'y']].values, H)
    elif type(traj) == np.ndarray:
        traj_meter = traj.copy()
        traj_meter = image2world(traj, H)
    elif type(traj) == list:
        traj_meter = [image2world(t, H) for t in traj]
    else:
        raise ValueError(f'Type {type(traj)} not supported')
    return traj_meter


def world_to_image_trajectory(traj, H):
    """
    Transform trajectory from world to image coordinates.

    Params:
        traj (pd.DataFrame | np.ndarray | list): Trajectory data.
            - If pd.DataFrame, it should have columns=['x', 'y', 'frame', 'agent_id', 'scene_id'].
            - If np.ndarray, it should be of shape (..., 2) where each row is (x, y).
            - If list, it should contain numpy arrays of shape (..., 2).
        H (np.ndarray): Homography matrix of shape (3, 3).

    Returns:
        pd.DataFrame | np.ndarray | list: Trajectory in image coordinates.
    """

    if type(traj) == pd.DataFrame:
        traj_pixel = traj.copy()
        traj_pixel[['x', 'y']] = world2image(traj[['x', 'y']].values, H)
    elif type(traj) == np.ndarray:
        traj_pixel = traj.copy()
        traj_pixel = world2image(traj, H)
    elif type(traj) == list:
        traj_pixel = [world2image(t, H) for t in traj]
    else:
        raise ValueError(f'Type {type(traj)} not supported')
    return traj_pixel


def filter_short_trajectories(df, threshold):
    """
    Filter trajectories that are shorter in timesteps than the threshold.

    Params:
        df (pd.DataFrame): DataFrame with columns=['x', 'y', 'frame', 'trackId', 'sceneId', 'metaId'].
        threshold (int): Number of timesteps as threshold, only trajectories over threshold are kept.

    Returns:
        pd.DataFrame: DataFrame with trajectory length over threshold.
    """
    # Codebase borrowed from YNet
    len_per_id = df.groupby(by='agent_id', as_index=False).count()  # sequence-length for each unique pedestrian
    idx_over_thres = len_per_id[len_per_id['frame'] >= threshold]  # rows which are above threshold
    idx_over_thres = idx_over_thres['agent_id'].unique()  # only get metaIdx with sequence-length longer than threshold
    df = df[df['agent_id'].isin(idx_over_thres)]  # filter df to only contain long trajectories
    return df


def groupby_sliding_window(x, window_size, stride):
    """
    Groupby function to chunk trajectories into chunks of length window_size.
    When stride < window_size then chunked trajectories are overlapping.

    Params:
        x (pd.DataFrame): DataFrame with columns=['x', 'y', 'frame', 'agent_id', 'scene_id'].
        window_size (int): Sequence-length of one trajectory, mostly obs_len + pred_len.
        stride (int): Timesteps to move from one trajectory to the next one.

    Returns:
        pd.DataFrame: DataFrame with chunked trajectories.
    """
    # Codebase borrowed from YNet
    x_len = len(x)
    n_chunk = (x_len - window_size) // stride + 1
    idx = []
    metaId = []
    for i in range(n_chunk):
        idx += list(range(i * stride, i * stride + window_size))
        metaId += ['{}_{}'.format(x.agent_id.unique()[0], i)] * window_size
    df = x.iloc()[idx]
    df['temp_id'] = metaId
    return df


def sliding_window(df, window_size, stride):
    """
    Assumes downsampled df, chunks trajectories into chunks of length window_size.
    When stride < window_size then chunked trajectories are overlapping.

    Params:
        df (pd.DataFrame): DataFrame.
        window_size (int): Sequence-length of one trajectory, mostly obs_len + pred_len.
        stride (int): Timesteps to move from one trajectory to the next one.

    Returns:
        pd.DataFrame: DataFrame with chunked trajectories.
    """
    # Codebase borrowed from YNet
    gb = df.groupby(['agent_id'], as_index=False)
    df = gb.apply(groupby_sliding_window, window_size=window_size, stride=stride)
    df['meta_id'] = pd.factorize(df['temp_id'], sort=False)[0]
    df = df.drop(columns='temp_id')
    df = df.reset_index(drop=True)
    return df


def export_generated_scenario_to_socialgan_data(export_path, generated_scenario, scene_H):
    """
    Save generated scenario to text file for trajectory prediction model training.

    Params:
        export_path (str): Path to save the generated scenario, e.g. './generated/scenario.txt'.
        generated_scenario (pd.DataFrame): Generated scenario data.
        scene_H (np.ndarray): Homography matrix for the scene.
    """
    
    # Convert coordinates from image to world
    traj_meter = image2world(generated_scenario[['x', 'y']].values, scene_H)
    frames = generated_scenario['frame'].values
    agent_ids = generated_scenario['agent_id'].values

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    # Save to text file
    with open(export_path, 'w') as f:
        for frame, agent_id, (x, y) in zip(frames, agent_ids, traj_meter):
            if int(frame) % 10 == 0:
                f.write(f'{frame:d}\t{agent_id:d}\t{x:.8f}\t{y:.8f}\n')

    print(f'Generated scenario saved at {export_path}')


def make_neighbor_dict(traj, hist, nth, num_threshold, range_threshold):
    """
    Create a dictionary of neighboring agents based on their trajectories.

    Params:
        traj (pd.DataFrame): DataFrame with columns=['x', 'y', 'frame', 'meta_id', ...].
        hist (pd.DataFrame): DataFrame with columns=['x', 'y', 'frame', 'meta_id', ...].
        nth (int): The nth frame to consider from the trajectory.
        num_threshold (int): Maximum number of neighboring agents to consider for each trajectory.
        range_threshold (float): Maximum distance to consider an agent as a neighbor.
        
    Returns:
        dict: Keys are meta_id of traj, values are lists of meta_id of neighboring agents.
    """
    
    # Group by meta_id, and extract the first frame from each group
    traj_frame = traj.groupby('meta_id').nth(nth).reset_index().drop(columns=['scene', 'agent_type'])
    hist_frame = hist.groupby('meta_id').first().reset_index().drop(columns=['scene', 'agent_type'])

    # Get the neighbor meta_id group of traj_frame if it shares the same start frame with hist_frame.
    group = pd.merge(traj_frame, hist_frame, how='left', on='frame', suffixes=('_traj', '_hist'))
    group = group[group['agent_id_traj'] != group['agent_id_hist']]

    # Calculate the distance between xy coordinates, and drop the rows where distance is larger than range_threshold
    group['distance'] = np.linalg.norm(group[['x_traj', 'y_traj']].values - group[['x_hist', 'y_hist']].values, axis=1)
    group = group[group['distance'] < range_threshold]

    # Truncate if number of meta_id_hist is larger than num_threshold
    group = group.sort_values(by=['meta_id_traj', 'distance']).groupby('meta_id_traj').head(num_threshold)

    # group by 'meta_id_traj', and extract the meta_id_traj list as dict (meta_id_traj: [meta_id_hist])
    group = group.groupby('meta_id_traj')['meta_id_hist'].apply(list).reset_index()
    group_dict = dict(zip(group['meta_id_traj'], group['meta_id_hist']))

    # fill the empty list to avoid error
    group_dict = {i: group_dict.get(i, []) for i in traj_frame['meta_id'].unique()}
    return group_dict


class KalmanModel:
    """
    Kalman filter model for trajectory prediction.
    This model uses a constant acceleration model for filtering and smoothing trajectories.
    """
    
    def __init__(self, dt, n_dim=2, n_iter=4):
        self.n_iter = n_iter
        self.n_dim = n_dim

        self.A = np.array([[1, dt, dt ** 2],
                           [0, 1, dt],
                           [0, 0, 1]])
        self.C = np.array([[1, 0, 0]])
        self.Q = np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                           [dt ** 4 / 8., dt ** 3 / 3, dt ** 2 / 2],
                           [dt ** 3 / 6., dt ** 2 / 2, dt / 1]]) * 0.5
        self.R = np.array([[1]])
        self.kf = [KalmanFilter(transition_matrices=self.A, observation_matrices=self.C,
                                transition_covariance=self.Q, observation_covariance=self.R) for _ in range(n_dim)]

    def filter(self, measurement):
        filtered_means = []
        for dim in range(self.n_dim):
            f = self.kf[dim].em(measurement[:, dim], n_iter=self.n_iter)
            (filtered_state_means, filtered_state_covariances) = f.filter(measurement[:, dim])
            filtered_means.append(filtered_state_means)
        filtered_means = np.stack(filtered_means)
        return filtered_means[:, :, 0].T, filtered_means[:, :, 1].T

    def smooth(self, measurement):
        smoothed_means = []
        if measurement.shape[0] == 1:
            return measurement, np.zeros((1, 2))
        for dim in range(self.n_dim):
            f = self.kf[dim].em(measurement[:, dim], n_iter=self.n_iter)
            (smoothed_state_means, smoothed_state_covariances) = f.smooth(measurement[:, dim])
            smoothed_means.append(smoothed_state_means)
        smoothed_means = np.stack(smoothed_means)
        return smoothed_means[:, :, 0].T, smoothed_means[:, :, 1].T
