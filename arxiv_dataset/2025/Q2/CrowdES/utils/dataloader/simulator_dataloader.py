import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import pickle

from utils.homography import image2world, world2image
from utils.trajectory import filter_short_trajectories, sliding_window, make_neighbor_dict
from utils.navmesh import get_control_point, filter_collinear_polygons
from utils.dataloader.base_dataloader import BaseDataset


class SimulatorDataset_singlethread(BaseDataset):
    r"""Dataloader for state-switching crowd simulator model"""

    def __init__(self, config, phase):
        super().__init__(config, phase)
        
        # Load from cache if exists
        cache_path = os.path.join(config.dataset.dataset_preprocessed_path, 'cache', f'{phase}_simulator.pickle')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                for key, value in cache.items():
                    setattr(self, key, value)
            return

        pbar = tqdm(range(len(self.scene_list) * 2 + 2))
        pbar.set_description(f'Simulator Dataset {phase}: Load data')
        self.load_scene_image()
        self.load_scene_segmentation()
        self.load_scene_homography()
        self.load_appearance_map()
        self.load_population_map()
        self.load_flow_map()
        self.load_scene_size()
        self.load_origin_goal()
        self.load_trajectory()
        self.load_navigation_mesh()
        self.load_scene_walkable()
        pbar.update()

        # Make scene image guidance
        pbar.set_description(f'Simulator Dataset {phase}: Prepare input data')
        environment_size = tuple(config.crowd_simulator.simulator.environment_size)
        environment_types = config.crowd_simulator.simulator.environment_types
        environment_pixel_meter = config.crowd_simulator.simulator.environment_pixel_meter

        self.env_data = {}
        for scene in self.scene_list:
            env = [getattr(self, it)[scene] for it in environment_types]
            env = [(i[:, :, None] if i.ndim == 2 else i) for i in env]
            env = np.concatenate(env, axis=2)
            env = torch.FloatTensor(env).permute(2, 0, 1)  # shape=[C, H, W]
            env_dim = env.shape[0]
            self.env_data[scene] = env
            pbar.update()
        self.env_size = environment_size
        self.env_dim = env_dim

        # Create pointer matrix for fast access to environment data
        env_base_pointer_h = np.linspace(-environment_size[0] / 2, environment_size[0] / 2, environment_size[0])
        env_base_pointer_w = np.linspace(-environment_size[1] / 2, environment_size[1] / 2, environment_size[1])
        self.env_base_pointer = np.stack(np.meshgrid(env_base_pointer_h, env_base_pointer_w), axis=-1) * environment_pixel_meter  # shape=[64, 64, 2]
        
        # Prepare trajectory data
        history_length = config.crowd_simulator.simulator.history_length
        fututure_length = config.crowd_simulator.simulator.future_length
        window_size = history_length + fututure_length
        window_stride = config.crowd_simulator.simulator.window_stride
        interaction_range = config.crowd_simulator.simulator.interaction_range
        interaction_max_num_agents = config.crowd_simulator.simulator.interaction_max_num_agents
        control_time_offset = config.crowd_simulator.simulator.control_time_offset

        self.num_seq_scene = {}
        self.traj_obs_norm_scene = {}
        self.traj_pred_norm_scene = {}
        self.goal_norm_scene = {}
        self.neighbor_norm_scene = {}
        self.attr_scene = {}
        self.startpoint_for_norm_scene = {}

        self.startpoint_scene = {}
        self.goal_scene = {}
        self.speed_scene = {}
        self.control_point_scene = {}

        for scene in self.scene_list:
            traj_df = self.trajectory[scene].copy()

            # Transform pixel coordinate to meter coordinate
            H = self.scene_H[scene]
            traj_df[['x', 'y']] = image2world(traj_df[['x', 'y']].values, H) 
            traj_df = filter_short_trajectories(traj_df, threshold=window_size)
            traj_df = sliding_window(traj_df, window_size=window_size, stride=window_stride)
            
            # Build neighborhood dictionary
            hist_df = self.trajectory[scene].copy()
            hist_df[['x', 'y']] = image2world(hist_df[['x', 'y']].values, H)
            hist_df = filter_short_trajectories(hist_df, threshold=history_length)
            hist_df = sliding_window(hist_df, window_size=history_length, stride=window_stride)
            
            neighbor_dict = make_neighbor_dict(traj_df, hist_df, history_length-1, interaction_max_num_agents, interaction_range)

            # Make a torch tensor of traj of each meta_id
            origin_goal_df = self.origin_goal[scene]
            origin_goal_df = origin_goal_df[['agent_id', 'agent_type', 'preferred_speed', 'goal_x', 'goal_y']]

            traj_obs_norm_all = []  # For model training
            traj_pred_norm_all = []
            goal_norm_all = []
            neighbor_norm_all = []
            attr_all = []

            startpoint_all = []  # For navmesh
            goal_all = []
            speed_all = []

            for meta_id, group_meta_id in tqdm(neighbor_dict.items()):
                target_df = traj_df[traj_df['meta_id'] == meta_id]
                target_traj = target_df[['x', 'y']].values  # (N, 2)
                target_agent_id = target_df['agent_id'].values[0]

                # Get the agent_type, preferred_speed, goal_x, goal_y from origin_goal_df
                target_metadata = origin_goal_df.loc[origin_goal_df['agent_id'] == target_agent_id].values[0]
                target_attr = target_metadata[[1, 2]]
                target_goal = target_metadata[[3, 4]]
                target_speed = target_metadata[2]
                target_startpoint = target_traj[history_length-1]

                # Get the neighbor's traj
                neighbor_traj_norm = np.zeros((interaction_max_num_agents, history_length, 2))
                neighbor_df = hist_df[hist_df['meta_id'].isin(group_meta_id)]
                neighbor_traj = neighbor_df[['x', 'y']].values.reshape(len(group_meta_id), history_length, 2)
                neighbor_traj_norm[:len(group_meta_id), :, :] = neighbor_traj - target_startpoint

                traj_obs_norm_all.append(target_traj[:history_length] - target_startpoint)
                traj_pred_norm_all.append(target_traj[history_length:] - target_startpoint)
                goal_norm_all.append(image2world(target_goal, H) - target_startpoint)
                neighbor_norm_all.append(neighbor_traj_norm)
                attr_all.append(target_attr)

                startpoint_all.append(target_startpoint)
                goal_all.append(target_goal)
                speed_all.append(target_speed)
            
            # Convert lists to tensors
            self.traj_obs_norm_scene[scene] = torch.FloatTensor(np.array(traj_obs_norm_all))
            self.traj_pred_norm_scene[scene] = torch.FloatTensor(np.array(traj_pred_norm_all))
            self.goal_norm_scene[scene] = torch.FloatTensor(np.array(goal_norm_all))
            self.neighbor_norm_scene[scene] = torch.FloatTensor(np.array(neighbor_norm_all))
            self.attr_scene[scene] = torch.FloatTensor(np.array(attr_all))
            self.startpoint_for_norm_scene[scene] = torch.FloatTensor(np.array(startpoint_all))

            self.startpoint_scene[scene] = world2image(np.array(startpoint_all), H)
            self.goal_scene[scene] = np.array(goal_all)
            self.speed_scene[scene] = np.array(speed_all)

            # Build the guided point using navmesh
            navmesh = self.navigation_mesh[scene]
            vertices = navmesh['vertices']
            polygons = navmesh['polygons']
    
            # Filter out collinear polygons
            polygons = filter_collinear_polygons(vertices, polygons)

            start_points = self.startpoint_scene[scene]
            finish_points = self.goal_scene[scene]
            distances = self.speed_scene[scene] * control_time_offset

            # Calculate control points and shift them by the startpoint for normalization
            control_points = get_control_point(vertices, polygons, start_points, finish_points, H, distances)
            control_points = torch.FloatTensor(np.array(control_points))
            control_points = control_points - self.startpoint_for_norm_scene[scene]
            self.control_point_scene[scene] = control_points

            num_seq_scene = len(traj_obs_norm_all)
            self.num_seq_scene[scene] = num_seq_scene
            pbar.update()

        pbar.set_description(f'Simulator Dataset {phase}: Gather data pointers')
        self.idx2data = []
        for scene in self.scene_list:
            self.idx2data.extend([(scene, idx) for idx in range(self.num_seq_scene[scene])])
        pbar.update()
        pbar.set_description(f'Simulator Dataset {phase}: {len(self.scene_list)} Scenes, {len(self.idx2data)} Items')
        pbar.total = pbar.n = len(self.scene_list)
        pbar.close()

        # Remove unnecessary variables
        del self.scene_img
        del self.scene_seg
        del self.scene_appearance
        del self.scene_population
        del self.scene_flow
        del self.trajectory
        del self.origin_goal

        # Caching the data for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def __len__(self):
        return len(self.idx2data)
    
    def __getitem__(self, index):
        scene, idx = self.idx2data[index]

        # Get history and future traj data
        traj_obs_norm = self.traj_obs_norm_scene[scene][idx]
        traj_pred_norm = self.traj_pred_norm_scene[scene][idx]
        goal_norm = self.goal_norm_scene[scene][idx]
        neighbor_norm = self.neighbor_norm_scene[scene][idx]
        attr = self.attr_scene[scene][idx]
        startpoint_for_norm_scene = self.startpoint_for_norm_scene[scene][idx]
        control = self.control_point_scene[scene][idx]

        # Get environmental map data
        env_data = self.env_data[scene]
        _, h, w = env_data.shape
        H = self.scene_H[scene]
        env_pointer = self.env_base_pointer + startpoint_for_norm_scene[[1, 0]].numpy()  # xy -> hw in meter
        env_pointer_pixel = world2image(env_pointer.transpose(1, 0, 2), H).transpose(1, 0, 2)  # hw -> wh for xy in pixel.
        env_pointer_pixel = env_pointer_pixel.round().astype(int)
        env_pointer_pixel = np.clip(env_pointer_pixel, 0, [h-1, w-1])
        env_data_crop = env_data[:, env_pointer_pixel[:, :, 0], env_pointer_pixel[:, :, 1]]

        out = {'traj_hist': traj_obs_norm.detach(),
               'traj_fut': traj_pred_norm.detach(),
               'goal': goal_norm.detach(),
               'attr': attr.detach(),
               'control': control.detach(),
               'neighbor': neighbor_norm.detach(),
               'environment': env_data_crop.detach(),
               'startpoint': startpoint_for_norm_scene.detach()}
        return out


def process_scene(scene, config, trajectory, origin_goal, navigation_mesh, H):
    """Helper function to process a single scene in parallel"""

    # Prepare trajectory data
    history_length = config.crowd_simulator.simulator.history_length
    fututure_length = config.crowd_simulator.simulator.future_length
    window_size = history_length + fututure_length
    window_stride = config.crowd_simulator.simulator.window_stride
    interaction_range = config.crowd_simulator.simulator.interaction_range
    interaction_max_num_agents = config.crowd_simulator.simulator.interaction_max_num_agents
    control_time_offset = config.crowd_simulator.simulator.control_time_offset

    # Transform pixel coordinate to meter coordinate
    traj_df = trajectory.copy()
    traj_df[['x', 'y']] = image2world(traj_df[['x', 'y']].values, H)
    traj_df = filter_short_trajectories(traj_df, threshold=window_size)
    traj_df = sliding_window(traj_df, window_size=window_size, stride=window_stride)
    
    # Build neighborhood dictionary
    hist_df = trajectory.copy()
    hist_df[['x', 'y']] = image2world(hist_df[['x', 'y']].values, H)
    hist_df = filter_short_trajectories(hist_df, threshold=history_length)
    hist_df = sliding_window(hist_df, window_size=history_length, stride=window_stride)

    neighbor_dict = make_neighbor_dict(traj_df, hist_df, history_length-1, interaction_max_num_agents, interaction_range)

    # Make a torch tensor of traj of each meta_id
    origin_goal_df = origin_goal[['agent_id', 'agent_type', 'preferred_speed', 'goal_x', 'goal_y']]

    traj_obs_norm_all = []  # For model training
    traj_pred_norm_all = []
    goal_norm_all = []
    neighbor_norm_all = []
    attr_all = []

    startpoint_all = []  # For navmesh
    goal_all = []
    speed_all = []

    for meta_id, group_meta_id in tqdm(neighbor_dict.items()):
        target_df = traj_df[traj_df['meta_id'] == meta_id]
        target_traj = target_df[['x', 'y']].values  # (N, 2)
        target_agent_id = target_df['agent_id'].values[0]

        # Get the agent_type, preferred_speed, goal_x, goal_y from origin_goal_df
        target_metadata = origin_goal_df[origin_goal_df['agent_id'] == target_agent_id].values[0]
        target_attr = target_metadata[[1, 2]]
        target_goal = target_metadata[[3, 4]]
        target_speed = target_metadata[2]
        target_startpoint = target_traj[history_length-1]

        # Get the neighbor's traj
        neighbor_traj_norm = np.zeros((interaction_max_num_agents, history_length, 2))
        neighbor_df = hist_df[hist_df['meta_id'].isin(group_meta_id)]
        neighbor_traj = neighbor_df[['x', 'y']].values.reshape(len(group_meta_id), history_length, 2)
        neighbor_traj_norm[:len(group_meta_id), :, :] = neighbor_traj - target_startpoint

        traj_obs_norm_all.append(target_traj[:history_length] - target_startpoint)
        traj_pred_norm_all.append(target_traj[history_length:] - target_startpoint)
        goal_norm_all.append(image2world(target_goal, H) - target_startpoint)
        neighbor_norm_all.append(neighbor_traj_norm)
        attr_all.append(target_attr)

        startpoint_all.append(target_startpoint)
        goal_all.append(target_goal)
        speed_all.append(target_speed)

    # Convert lists to tensors
    traj_obs_norm = torch.FloatTensor(np.array(traj_obs_norm_all))
    traj_pred_norm = torch.FloatTensor(np.array(traj_pred_norm_all))
    goal_norm = torch.FloatTensor(np.array(goal_norm_all))
    neighbor_norm = torch.FloatTensor(np.array(neighbor_norm_all))
    attr = torch.FloatTensor(np.array(attr_all))
    startpoint_for_norm = torch.FloatTensor(np.array(startpoint_all))

    startpoint = world2image(np.array(startpoint_all), H)
    goal = np.array(goal_all)
    speed = np.array(speed_all)

    # Build the guided point using navmesh
    vertices = navigation_mesh['vertices']
    polygons = navigation_mesh['polygons']
    
    # Filter out collinear polygons
    polygons = filter_collinear_polygons(vertices, polygons)

    start_points = startpoint
    finish_points = goal
    distances = speed * control_time_offset

    # Calculate control points and shift them by the startpoint for normalization
    control_points = get_control_point(vertices, polygons, start_points, finish_points, H, distances)
    control_points = torch.FloatTensor(np.array(control_points))
    control_points = control_points - startpoint_for_norm

    return_dict = {'traj_obs_norm': traj_obs_norm,
                   'traj_pred_norm': traj_pred_norm,
                   'goal_norm': goal_norm,
                   'neighbor_norm': neighbor_norm,
                   'attr': attr,
                   'startpoint_for_norm': startpoint_for_norm,
                   'startpoint': startpoint,
                   'goal': goal,
                   'speed': speed,
                   'control_point': control_points,
                   'num_seq_scene': len(traj_obs_norm_all)}

    return scene, return_dict


class SimulatorDataset(BaseDataset):
    r"""Dataloader for crowd dynamics model"""

    def __init__(self, config, phase):
        super().__init__(config, phase)
        
        # Load from cache if exists
        cache_path = os.path.join(config.dataset.dataset_preprocessed_path, 'cache', f'{phase}_simulator.pickle')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                for key, value in cache.items():
                    setattr(self, key, value)
            return

        pbar = tqdm(range(len(self.scene_list) * 2 + 2))
        pbar.set_description(f'Simulator Dataset {phase}: Load data')
        self.load_scene_image()
        self.load_scene_segmentation()
        self.load_scene_homography()
        self.load_appearance_map()
        self.load_population_map()
        self.load_flow_map()
        self.load_scene_size()
        self.load_origin_goal()
        self.load_trajectory()
        self.load_navigation_mesh()
        self.load_scene_walkable()
        pbar.update()

        # Make scene image guidance
        pbar.set_description(f'Simulator Dataset {phase}: Prepare input data')
        environment_size = tuple(config.crowd_simulator.simulator.environment_size)
        environment_types = config.crowd_simulator.simulator.environment_types
        environment_pixel_meter = config.crowd_simulator.simulator.environment_pixel_meter

        self.env_data = {}
        for scene in self.scene_list:
            env = [getattr(self, it)[scene] for it in environment_types]
            env = [(i[:, :, None] if i.ndim == 2 else i) for i in env]
            env = np.concatenate(env, axis=2)
            env = torch.FloatTensor(env).permute(2, 0, 1)  # shape=[C, H, W]
            env_dim = env.shape[0]
            self.env_data[scene] = env
            pbar.update()
        self.env_size = environment_size
        self.env_dim = env_dim

        # Create pointer matrix for fast access to environment data
        env_base_pointer_h = np.linspace(-environment_size[0] / 2, environment_size[0] / 2, environment_size[0])
        env_base_pointer_w = np.linspace(-environment_size[1] / 2, environment_size[1] / 2, environment_size[1])
        self.env_base_pointer = np.stack(np.meshgrid(env_base_pointer_h, env_base_pointer_w), axis=-1) * environment_pixel_meter  # shape=[64, 64, 2}
        
        self.num_seq_scene = {}
        self.traj_obs_norm_scene = {}
        self.traj_pred_norm_scene = {}
        self.goal_norm_scene = {}
        self.neighbor_norm_scene = {}
        self.attr_scene = {}
        self.startpoint_for_norm_scene = {}

        self.startpoint_scene = {}
        self.goal_scene = {}
        self.speed_scene = {}
        self.control_point_scene = {}

        results = Parallel(n_jobs=256)(delayed(process_scene)(scene, 
                                                              config, 
                                                              self.trajectory[scene], 
                                                              self.origin_goal[scene], 
                                                              self.navigation_mesh[scene], 
                                                              self.scene_H[scene]) for scene in self.scene_list)

        for scene, return_dict in results:
            self.traj_obs_norm_scene[scene] = return_dict['traj_obs_norm']
            self.traj_pred_norm_scene[scene] = return_dict['traj_pred_norm']
            self.goal_norm_scene[scene] = return_dict['goal_norm']
            self.neighbor_norm_scene[scene] = return_dict['neighbor_norm']
            self.attr_scene[scene] = return_dict['attr']
            self.startpoint_for_norm_scene[scene] = return_dict['startpoint_for_norm']
            self.startpoint_scene[scene] = return_dict['startpoint']
            self.goal_scene[scene] = return_dict['goal']
            self.speed_scene[scene] = return_dict['speed']
            self.control_point_scene[scene] = return_dict['control_point']
            self.num_seq_scene[scene] = return_dict['num_seq_scene']

            pbar.update()

        pbar.set_description(f'Simulator Dataset {phase}: Gather data pointers')
        self.idx2data = []
        for scene in self.scene_list:
            self.idx2data.extend([(scene, idx) for idx in range(self.num_seq_scene[scene])])
        pbar.update()
        pbar.set_description(f'Simulator Dataset {phase}: {len(self.scene_list)} Scenes, {len(self.idx2data)} Items')
        pbar.total = pbar.n = len(self.scene_list)
        pbar.close()

        # Remove unnecessary variables
        del self.scene_img
        del self.scene_seg
        del self.scene_appearance
        del self.scene_population
        del self.scene_flow
        del self.trajectory
        del self.origin_goal

        # Caching the data for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def __len__(self):
        return len(self.idx2data)
    
    def __getitem__(self, index):
        scene, idx = self.idx2data[index]

        # Get history and future traj data
        traj_obs_norm = self.traj_obs_norm_scene[scene][idx]
        traj_pred_norm = self.traj_pred_norm_scene[scene][idx]
        goal_norm = self.goal_norm_scene[scene][idx]
        neighbor_norm = self.neighbor_norm_scene[scene][idx]
        attr = self.attr_scene[scene][idx]
        startpoint_for_norm_scene = self.startpoint_for_norm_scene[scene][idx]
        control = self.control_point_scene[scene][idx]

        # Get environmental map data
        env_data = self.env_data[scene]
        _, h, w = env_data.shape
        H = self.scene_H[scene]

        env_pointer = self.env_base_pointer + startpoint_for_norm_scene[[1, 0]].numpy()  # xy -> hw in meter
        env_pointer_pixel = world2image(env_pointer.transpose(1, 0, 2), H).transpose(1, 0, 2)  # hw -> wh for xy in pixel.
        env_pointer_pixel = env_pointer_pixel.round().astype(int)
        env_pointer_pixel = np.clip(env_pointer_pixel, 0, [h-1, w-1])
        env_data_crop = env_data[:, env_pointer_pixel[:, :, 0], env_pointer_pixel[:, :, 1]]

        out = {'traj_hist': traj_obs_norm.detach(),
               'traj_fut': traj_pred_norm.detach(),
               'goal': goal_norm.detach(),
               'attr': attr.detach(),
               'control': control.detach(),
               'neighbor': neighbor_norm.detach(),
               'environment': env_data_crop.detach(),
               'startpoint': startpoint_for_norm_scene.detach()}
        
        return out


if __name__ == '__main__':
    from utils.config import get_config

    config = get_config('./configs/model/CrowdES_hotel.yaml')
    dataloader = SimulatorDataset(config, 'emission', 'train')
    dataloader.__len__()
    dataloader.__getitem__(0)
