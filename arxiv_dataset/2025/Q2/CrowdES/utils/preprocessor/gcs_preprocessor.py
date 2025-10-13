import os
import re
import glob
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed

from utils.config import DotDict
from utils.preprocessor.base_preprocessor import BaseDataPreprocessor


class GCSPreprocessor(BaseDataPreprocessor):
    r"""Preprocessor for GCS dataset"""

    def __init__(self, config, phase):
        super().__init__(config, phase)

        # GCS contains one long video, split the scene by the ratio
        scene = 'terminal'
        trainsplit = float(re.sub('[^\d\.]', '', str(config.dataset['dataset_train'])))
        testsplit = float(re.sub('[^\d\.]', '', str(config.dataset['dataset_test'])))
        train_set_ratio = trainsplit / (trainsplit + testsplit)

        # Load scene image & segmentation
        scene_img_path = os.path.join(config.dataset.dataset_path, 'reference', scene + '_image.jpg')
        self.scene_img[scene] = Image.open(scene_img_path)
        scene_bg_path = os.path.join(config.dataset.dataset_path, 'reference', scene + '_bg.png')
        scene_bg_path = scene_bg_path if os.path.exists(scene_bg_path) else scene_img_path
        self.scene_bg[scene] = Image.open(scene_bg_path)
        scene_seg_path = os.path.join(config.dataset.dataset_path, 'segmentation', scene + '_seg.png')
        self.scene_seg[scene] = Image.open(scene_seg_path).convert('L')
        scene_width, scene_height = self.scene_img[scene].size
        scene_size = {'width': scene_width, 'height': scene_height}
        self.scene_size[scene] = scene_size

        # Load homography matrix
        homography_file = os.path.join(config.dataset.dataset_path, 'homography', scene + '_H.txt')
        self.scene_H[scene] = np.loadtxt(homography_file)

        # Load trajectory
        annotation_list = glob.glob(os.path.join(config.dataset.dataset_path, 'annotation', '*.txt'))
        annotation_list = sorted(annotation_list, key=lambda x: int(re.sub(r'\D', '', os.path.basename(x))))
        assert len(annotation_list) == 12684, 'There are {} missing files in the dataset'.format(12684 - len(annotation_list))

        if False:
            # Sequential processing
            dense_trajectory = self.process_interpolate_trajectory_sequential(annotation_list, scene)
        else:
            # Parallel processing
            results = Parallel(n_jobs=-1)(delayed(self.process_trajectory)(file) for file in annotation_list)
            raw_data_list = []
            curr_idx = 0
            for raw_data, new_idx in results:
                raw_data[:, 1] += curr_idx
                raw_data_list.append(raw_data)
                curr_idx += new_idx
            raw_data_list = np.concatenate(raw_data_list, axis=0)

            ped_ids = np.unique(raw_data_list[:, 1])
            results = Parallel(n_jobs=-1)(delayed(self.interpolate_trajectory)(ped_id, raw_data_list, scene) for ped_id in ped_ids)
            dense_trajectory = pd.concat(results, axis=0)

        # Officially, only up to 100,000 frames are labeled, so frames beyond this are removed
        dense_trajectory = dense_trajectory[dense_trajectory['frame'] <= 100000]

        # Train-Test split by frame
        min_frame, max_frame = dense_trajectory['frame'].min(), dense_trajectory['frame'].max()
        split_frame = min_frame + (max_frame - min_frame) * train_set_ratio
        if phase == 'train':
            dense_trajectory = dense_trajectory[dense_trajectory['frame'] <= split_frame]
        else:
            dense_trajectory = dense_trajectory[dense_trajectory['frame'] > split_frame]
            dense_trajectory['frame'] = dense_trajectory['frame'] - split_frame  # Adjust the frame number to start from 0  # TODO: CHECK THIS LINE FIX BUG

        # Trajectory filtering and downsampling
        frame_skip = config.dataset.dataset_fps // config.crowd_simulator.simulator.simulator_fps
        dense_trajectory, downsampled_trajectory = self.trim_and_downsample_trajectory(dense_trajectory, scene_size, frame_skip)
        self.trajectory_dense[scene] = dense_trajectory
        self.trajectory[scene] = downsampled_trajectory

        # Make origin goal pair
        origin_goal = self.origin_and_goal_extraction(dense_trajectory, self.scene_H[scene], config.dataset.dataset_fps)
        self.origin_goal[scene] = origin_goal

        # GT map generation
        blur_sigma = config.crowd_emitter.emitter_pre.blur_sigma
        appearance_map, population_map, flow_map = self.generate_gt_map(scene_size, origin_goal, dense_trajectory, blur_sigma=blur_sigma)
        self.scene_appearance[scene] = appearance_map
        self.scene_population[scene] = population_map
        self.scene_flow[scene] = flow_map
        
        # Scene_seg to one-hot
        scene_seg = self.scene_seg[scene]
        scene_seg = self.reduce_labels_transform(np.array(scene_seg))
        scene_seg_enc = np.eye(self.num_seg_classes)[np.array(scene_seg)]
        self.scene_seg[scene] = scene_seg_enc
        
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

    @staticmethod
    def process_trajectory(annotation_file):
        curr_idx = 0
        raw_data = np.loadtxt(annotation_file, dtype=float)
        raw_data = raw_data.reshape((-1, 3))  # x, y, frame
        raw_data_list = []
        
        for idx, frame_data in enumerate(raw_data):
            # There are several trajectory files with non-continuous timestamps
            # so they need to be counted as different agents to prevent the agent ID from being reused
            if idx > 0 and raw_data[idx, 2] - raw_data[idx - 1, 2] > 20:
                # Check if the non-continuous data is just a static object
                threshold = (raw_data[idx, 2] - raw_data[idx - 1, 2]) #/ 20 * 20
                if np.linalg.norm(raw_data[idx, :2] - raw_data[idx - 1, :2], axis=0) > threshold:
                    curr_idx += 1  # Non-continuous frame detected. Update the object ID
            elif idx > 0 and np.linalg.norm(raw_data[idx, :2] - raw_data[idx - 1, :2], axis=0) > 200:
                curr_idx += 1  # Abnormal coordinate change detected. Update the object ID
                
            raw_data_list.append([frame_data[2], curr_idx, frame_data[0], frame_data[1]])
        raw_data_list = np.array(raw_data_list)
        num_agents = curr_idx + 1
        return raw_data_list, num_agents

    @staticmethod
    def interpolate_trajectory(ped_id, raw_data_list, scene):
        raw_data = raw_data_list[raw_data_list[:, 1] == ped_id]
        data_t, data_x, data_y = raw_data[:, 0], raw_data[:, 2], raw_data[:, 3]
        data_t_intrp = np.arange(data_t.min(), data_t.max() + 1, 1)
        data_x_intrp = np.interp(data_t_intrp, data_t, data_x)
        data_y_intrp = np.interp(data_t_intrp, data_t, data_y)
        
        traj_interp = pd.DataFrame({'frame': data_t_intrp, 'x': data_x_intrp, 'y': data_y_intrp})
        traj_interp['frame'] = traj_interp['frame'].astype(int)
        traj_interp['scene'] = scene
        traj_interp['agent_id'] = int(ped_id)
        traj_interp['agent_type'] = 0  # Pedestrian
        return traj_interp[['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']]
    
    def process_interpolate_trajectory_sequential(self, annotation_list, scene):
        raw_data_list = []
        curr_idx = 0
        for annotation_file in annotation_list:
            raw_data = np.loadtxt(annotation_file, dtype=float)
            raw_data = raw_data.reshape((-1, 3))  # x, y, frame

            for idx, frame_data in enumerate(raw_data):
                # There are several trajectory files with non-continuous timestamps
                # so they need to be counted as different agents to prevent the agent ID from being reused
                if idx > 0 and raw_data[idx, 2] - raw_data[idx - 1, 2] > 20:
                    # Check if the non-continuous data is just a static object
                    threshold = (raw_data[idx, 2] - raw_data[idx - 1, 2]) / 20 * 20
                    if np.linalg.norm(raw_data[idx, :2] - raw_data[idx - 1, :2], axis=0) > threshold:
                        curr_idx += 1  # Non-continuous frame detected. Update the object ID
                # Check if the tracked object is changed
                elif idx > 0 and np.linalg.norm(raw_data[idx, :2] - raw_data[idx - 1, :2], axis=0) > 200:
                    curr_idx += 1  # Abnormal coordinate change detected. Update the object ID
                
                raw_data_list.append([frame_data[2], curr_idx, frame_data[0], frame_data[1]])
            curr_idx += 1
        
        raw_data_list = np.array(raw_data_list)

        # Frame interpolation
        dense_trajectory = []
        for ped_id in np.unique(raw_data_list[:, 1]):
            raw_data = raw_data_list[raw_data_list[:, 1] == ped_id]
            data_t, data_x, data_y,  = raw_data[:, 0], raw_data[:, 2], raw_data[:, 3]
            data_t_intrp = np.arange(data_t.min(), data_t.max() + 1, 1)
            data_x_intrp = np.interp(data_t_intrp, data_t, data_x)
            data_y_intrp = np.interp(data_t_intrp, data_t, data_y)
            traj_interp = pd.DataFrame({'frame': data_t_intrp, 'x': data_x_intrp, 'y': data_y_intrp})
            traj_interp['frame'] = traj_interp['frame'].astype(int)
            traj_interp['scene'] = scene
            traj_interp['agent_id'] = int(ped_id)
            traj_interp['agent_type'] = 0  # Pedestrian
            traj_interp = traj_interp[['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']]
            dense_trajectory.append(traj_interp)
        dense_trajectory = pd.concat(dense_trajectory, axis=0)
        dense_trajectory.sort_values(by=['agent_id', 'frame'], inplace=True)

        return dense_trajectory
