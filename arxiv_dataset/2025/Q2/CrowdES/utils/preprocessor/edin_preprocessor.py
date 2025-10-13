import os
import ast
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed

from utils.config import DotDict
from utils.preprocessor.base_preprocessor import BaseDataPreprocessor


class EDINPreprocessor(BaseDataPreprocessor):
    r"""Preprocessor for Edinburgh dataset"""

    def __init__(self, config, phase):
        super().__init__(config, phase)
        
        # Load scene image & segmentation
        scene_img_path = os.path.join(config.dataset.dataset_path, 'reference', 'edinburgh_image.jpg')
        scene_img = Image.open(scene_img_path)
        scene_bg_path = os.path.join(config.dataset.dataset_path, 'reference', 'edinburgh_bg.png')
        scene_bg_path = scene_bg_path if os.path.exists(scene_bg_path) else scene_img_path
        scene_bg = Image.open(scene_bg_path)
        scene_seg_path = os.path.join(config.dataset.dataset_path, 'segmentation', 'edinburgh_seg.png')
        scene_seg = Image.open(scene_seg_path).convert('L')
        scene_width, scene_height = scene_img.size
        scene_size = DotDict({'width': scene_width, 'height': scene_height})

        # Load homography matrix
        homography_file = os.path.join(config.dataset.dataset_path, 'homography', 'edinburgh_H.txt')
        scene_H = np.loadtxt(homography_file)

        # Scene list
        scene_list = config.dataset['dataset_' + phase]

        # Process scenes in parallel
        results = Parallel(n_jobs=-1)(delayed(self.process_scene)(scene, config, scene_size, scene_H) for scene in scene_list)
        
        # Scene_seg to one-hot
        scene_seg = self.reduce_labels_transform(np.array(scene_seg))
        scene_seg_enc = np.eye(self.num_seg_classes)[np.array(scene_seg)]

        # Unpack results
        for scene, scene_data in results:
            (trajectory_dense, trajectory, origin_goal, 
            appearance_map, population_map, flow_map) = scene_data

            self.scene_img[scene] = scene_img
            self.scene_bg[scene] = scene_bg
            self.scene_seg[scene] = scene_seg_enc
            self.scene_size[scene] = scene_size
            self.scene_H[scene] = scene_H
            self.trajectory_dense[scene] = trajectory_dense
            self.trajectory[scene] = trajectory
            self.origin_goal[scene] = origin_goal
            self.scene_appearance[scene] = appearance_map
            self.scene_population[scene] = population_map
            self.scene_flow[scene] = flow_map

        # Augmentation
        if phase == 'train':
            augmentations = config.dataset.dataset_augmentation
            frame_skip = config.dataset.dataset_fps // config.crowd_simulator.simulator.simulator_fps
            self.scene_augmentation(augmentations, frame_skip)

        # Scene statistics
        self.scene_statistics()

        # Generate navigation mesh
        walkable_class = config.crowd_simulator.locomotion.walkable_class
        walkable_id = [self.label2id[wc] for wc in walkable_class]
        self.generate_navmesh(walkable_id)

        # Save preprocessed data
        self.save_data(config, phase)

    def process_scene(self, scene, config, scene_size, scene_H):
        # Load trajectory
        raw_data_path = os.path.join(config.dataset.dataset_path, 'annotation', scene + '.txt')

        # Load data from all files
        data = pd.read_csv(raw_data_path, sep='\n|=', header=None, index_col=0, skiprows=[0, 1], engine='python')
        data.reset_index(inplace=True)
        data.columns = ['index', 'value']
        data = data[data['index'].str.startswith('TRACK')]

        # Reconstruct the data in arrays 
        track_data = []
        for row in range(len(data)):
            # Sometimetimes the track length in properties is wrong, discard it
            one_track = data.iloc[row, 1].split(';')
            one_track.pop()
            one_track[0] = one_track[0].replace('[[','[')
            one_track[-1] = one_track[-1].replace(']]',']')
            one_track = np.array([ast.literal_eval(i.replace(' [','[').replace(' ',',')) for i in one_track])
            one_track = np.c_[one_track, np.ones(one_track.shape[0], dtype=int) * row]
            track_data.extend(one_track)
        track_data_pd = pd.DataFrame(data=np.array(track_data), columns=['x','y','frame','agent_id'])

        # Clear repeated trajectories
        clean_track = []
        for i in track_data_pd.groupby('agent_id'):
            i[1].drop_duplicates(subset='frame', keep='first', inplace=True)
            # Clean repeated trajectory for the same agent 
            for j in i[1].groupby(['frame','x','y']):
                j[1].drop_duplicates(subset='frame', keep='first', inplace=True)
                clean_track.append(j[1])
        clean_track = pd.concat(clean_track, ignore_index=True)

        # Filter out the trajectory with descending frames or missing frames more than 12 (tracker error)
        clean_track = clean_track.groupby('agent_id').filter(lambda x: np.all(np.diff(x['frame']) > 0))
        clean_track = clean_track.groupby('agent_id').filter(lambda x: np.all(np.diff(x['frame']) <= 12))
        clean_track['x'] = clean_track['x'].astype(float)
        clean_track['y'] = clean_track['y'].astype(float)
        clean_track = clean_track[['agent_id', 'frame', 'x', 'y']]
        clean_track.sort_values(by=['agent_id','frame'], inplace=True)

        # Frame interpolation
        dense_trajectory = []
        for agent_id, traj in clean_track.groupby('agent_id'):
            data_t, data_x, data_y = traj['frame'], traj['x'], traj['y']
            data_t_intrp = np.arange(data_t.min(), data_t.max() + 1, 1)
            data_x_intrp = np.interp(data_t_intrp, data_t, data_x)
            data_y_intrp = np.interp(data_t_intrp, data_t, data_y)
            traj_interp = pd.DataFrame({'frame': data_t_intrp, 'x': data_x_intrp, 'y': data_y_intrp})
            traj_interp['frame'] = traj_interp['frame'].astype(int)
            traj_interp['scene'] = scene
            traj_interp['agent_id'] = agent_id
            traj_interp['agent_type'] = 0  # Pedestrian
            traj_interp = traj_interp[['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']]
            dense_trajectory.append(traj_interp)
        dense_trajectory = pd.concat(dense_trajectory, axis=0)
        dense_trajectory.sort_values(by=['agent_id', 'frame'], inplace=True)
            
        # Trajectory filtering and downsampling
        frame_skip = config.dataset.dataset_fps // config.crowd_simulator.simulator.simulator_fps
        dense_trajectory, downsampled_trajectory = self.trim_and_downsample_trajectory(dense_trajectory, scene_size, frame_skip)

        # Make origin goal pair
        origin_goal = self.origin_and_goal_extraction(dense_trajectory, scene_H, config.dataset.dataset_fps)

        # GT map generation
        blur_sigma = config.crowd_emitter.emitter_pre.blur_sigma
        appearance_map, population_map, flow_map = self.generate_gt_map(scene_size, origin_goal, dense_trajectory, blur_sigma=blur_sigma)

        return (scene, (dense_trajectory, downsampled_trajectory, origin_goal, 
        appearance_map, population_map, flow_map))
