import os
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed

from utils.config import DotDict
from homography import generate_homography
from utils.preprocessor.base_preprocessor import BaseDataPreprocessor


class SDDPreprocessor(BaseDataPreprocessor):
    r"""Preprocessor for Stanford Drone Dataset"""

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
        place, video = scene.split('_')
        
        # Load scene image & segmentation
        scene_img_path = os.path.join(config.dataset.dataset_path, 'annotation', place, 'video' + video, 'reference.jpg')
        scene_img = Image.open(scene_img_path)
        scene_bg = Image.open(scene_img_path)
        scene_seg_path = os.path.join(config.dataset.dataset_path, 'segmentation', scene + '_seg.png')
        scene_seg = Image.open(scene_seg_path).convert('L')
        scene_width, scene_height = scene_img.size
        scene_size = DotDict({'width': scene_width, 'height': scene_height})

        # Load homography matrix
        scale_file = os.path.join(config.dataset.dataset_path, 'homography', 'estimated_scales.yaml')
        with open(scale_file, 'r') as f:
            scale = yaml.load(f, Loader=yaml.SafeLoader)[place]['video' + video]['scale']
        scene_H = generate_homography(scale=1/scale)

        # Load trajectory
        raw_data_path = os.path.join(config.dataset.dataset_path, 'annotation', place, 'video' + video, 'annotations.txt')
        column_name = ['agent_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
        label2type = {'Pedestrian': 0, 'Biker': 1, 'Skater': 1, 'Cart': 2, 'Car': 2, 'Bus': 2}
        dense_trajectory = pd.read_csv(raw_data_path, header=0, names=column_name, delimiter=' ')
        dense_trajectory['x'] = (dense_trajectory['xmax'] + dense_trajectory['xmin']) / 2
        dense_trajectory['y'] = (dense_trajectory['ymax'] + dense_trajectory['ymin']) / 2
        dense_trajectory['agent_type'] = dense_trajectory['label'].map(label2type)
        dense_trajectory = dense_trajectory[dense_trajectory['lost'] == 0]
        dense_trajectory = dense_trajectory.drop(columns=['xmin', 'ymin', 'xmax', 'ymax', 'lost', 'occluded', 'generated', 'label'])
        dense_trajectory['scene'] = scene
        dense_trajectory = dense_trajectory[['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']]
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
