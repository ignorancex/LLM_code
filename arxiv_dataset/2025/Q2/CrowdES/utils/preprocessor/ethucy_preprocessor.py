import os
import re
import ast
import glob
import json
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from scipy.interpolate import PchipInterpolator

from utils.config import DotDict
from utils.preprocessor.base_preprocessor import BaseDataPreprocessor



class ETHUCYPreprocessor(BaseDataPreprocessor):
    r"""Preprocessor for ETH-UCY dataset"""

    def __init__(self, config, phase):
        super().__init__(config, phase)
        
        scene_list = config.dataset['dataset_' + phase]

        # Process scenes in parallel
        results = Parallel(n_jobs=-1)(delayed(self.process_scene)(scene, config) for scene in scene_list)

        # Unpack results
        for scene, scene_data in results:
            (scene_img, scene_bg, scene_seg, scene_size, scene_H, 
             trajectory_dense, trajectory, origin_goal, 
             appearance_map, population_map, flow_map) = scene_data

            self.scene_img[scene] = scene_img
            self.scene_bg[scene] = scene_bg
            self.scene_seg[scene] = scene_seg
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
        

    def process_scene(self, scene, config):
        # Load scene image & segmentation
        scene_img_path = os.path.join(config.dataset.dataset_path, 'reference', scene + '_image.png')
        scene_img = Image.open(scene_img_path)
        scene_bg_path = os.path.join(config.dataset.dataset_path, 'reference', scene + '_bg.png')
        scene_bg_path = scene_bg_path if os.path.exists(scene_bg_path) else scene_img_path
        scene_bg = Image.open(scene_bg_path)
        scene_seg_path = os.path.join(config.dataset.dataset_path, 'segmentation', scene + '_seg.png')
        scene_seg = Image.open(scene_seg_path).convert('L')
        scene_width, scene_height = scene_img.size
        scene_size = DotDict({'width': scene_width, 'height': scene_height})

        # Load homography matrix
        homography_file = os.path.join(config.dataset.dataset_path, 'homography', scene + '_H.txt')
        scene_H = np.loadtxt(homography_file)

        # Load trajectory
        raw_data_path = os.path.join(config.dataset.dataset_path, 'annotation', scene + '.vsp')
        with open(raw_data_path, 'r') as f:
            raw_data = f.readlines()
        
        # Line parsing
        raw_spline_list = []
        is_spline = False
        for line in raw_data:
            if 'the number of splines' in line:
                is_spline = False
            elif 'number of line obstacles' in line:
                is_spline = False
            elif 'number of cylinder obstacles' in line:
                is_spline = False
            if 'Num of control points' in line:
                is_spline = True
                raw_spline_list.append([])
            else:
                if is_spline:
                    raw_spline_list[-1].append(line)
            
        # Filter out empty or single point spline
        raw_spline_list = [spline for spline in raw_spline_list if len(spline) > 1]
            
        # Make it to float
        # {x_coord} {y_coord} {frame_no} {heading_angle} - (2D point, m_id)
        spline_list = []
        for spline in raw_spline_list:
            new_spline = []
            for line in spline:
                data = line.split(' ')
                new_spline.append([float(data[2]), float(data[0]), float(data[1])])
            new_spline = np.array(new_spline)

            # Shift the XY coordinate
            new_spline[:, 1] = new_spline[:, 1] + scene_width // 2
            new_spline[:, 2] = -new_spline[:, 2] + scene_height // 2

            # Fix framerate issue for eth scene
            if scene == 'seq_eth':
                new_spline[:, 0] = new_spline[:, 0] / 0.6
            spline_list.append(new_spline)
        
        # Frame interpolation
        dense_trajectory = []
        for ped_id, spline in enumerate(spline_list):
            pci_x = PchipInterpolator(spline[:, 0], spline[:, 1])  # Non-overshooting
            pci_y = PchipInterpolator(spline[:, 0], spline[:, 2])  # Non-overshooting
            time_intrp = np.arange(np.ceil(spline[0, 0]), np.floor(spline[-1, 0]) + 1, 1)
            traj_intrp = pd.DataFrame({'frame': time_intrp, 'x': pci_x(time_intrp), 'y': pci_y(time_intrp)})
            traj_intrp['frame'] = traj_intrp['frame'].astype(int)
            traj_intrp['scene'] = scene
            traj_intrp['agent_id'] = ped_id
            traj_intrp['agent_type'] = 0  # Pedestrian
            traj_intrp = traj_intrp[['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']]
            dense_trajectory.append(traj_intrp)
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

        # Scene_seg to one-hot
        scene_seg = self.reduce_labels_transform(np.array(scene_seg))
        scene_seg_enc = np.eye(self.num_seg_classes)[np.array(scene_seg)]

        return (scene, (scene_img, scene_bg, scene_seg_enc, scene_size, scene_H, 
                        dense_trajectory, downsampled_trajectory, origin_goal, 
                        appearance_map, population_map, flow_map))
