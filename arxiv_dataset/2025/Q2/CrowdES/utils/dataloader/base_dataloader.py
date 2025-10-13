import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import cv2
from PIL import Image


class BaseDataset(Dataset):
    r"""Base class for dataloader"""

    def __init__(self, config, phase):
        assert phase in ['train', 'test']
        self.config = config
        self.phase = phase
        self._check_phase()

        # Load scene list
        self.dataset_path = os.path.join(config.dataset.dataset_preprocessed_path, phase)
        scene_list = sorted(os.listdir(os.path.join(self.dataset_path, 'image')))
        self.scene_list = [scene.replace('_image.png', '') for scene in scene_list]

        self.trajectory_dense = {}
        self.trajectory = {}
        self.origin_goal = {}

        self.scene_img = {}
        self.scene_bg = {}
        self.scene_seg = {}
        self.scene_size = {}
        self.scene_H = {}

        self.scene_appearance = {}
        self.scene_population = {}
        self.scene_flow = {}

        self.scene_walkable = {}
        self.navigation_mesh = {}

        self.id2agent = {0: 'Pedestrian', 1: 'Rider', 2: 'Vehicle'}
        self.agent2id = {v: k for k, v in self.id2agent.items()}
        self.num_agent_types = len(self.id2agent)

        self.label2id = json.load(open(os.path.join(config.dataset.dataset_path, '..', 'segmentation_classes.json'), 'r'))
        self.label2id = {k: int(v) - 1 for k, v in self.label2id.items()}  # Reduced transformation
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_seg_classes = len(self.label2id)
        self.seg_ignore_label = 255
        
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def _check_phase(self):
        dataloader_name = self.__class__.__name__
        phase = self.phase

        if phase == 'test':
            if dataloader_name not in ['EvaluationDataset', 'SyntheticDataset']:
                raise ValueError(f'The test set cannot be used in {dataloader_name}. '
                                 'Test set must remain blind during model training at any case, and '
                                 'should only be used for evaluation after training has completed.')

    def load_scene_image(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                scene_img_path = os.path.join(self.dataset_path, 'image', scene + '_image.png')
                scene_img = Image.open(scene_img_path)
                self.scene_img[scene] = scene_img
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                scene_img_path = os.path.join(dataset_path, 'image', scene + '_image.png')
                scene_img = Image.open(scene_img_path)
                return scene, scene_img
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_img = {scene: img for scene, img in results}
    
    def load_scene_background(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                scene_bg_path = os.path.join(self.dataset_path, 'image_terrain', scene + '_bg.png')
                scene_bg = Image.open(scene_bg_path)
                self.scene_bg[scene] = scene_bg
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                scene_bg_path = os.path.join(dataset_path, 'image_terrain', scene + '_bg.png')
                scene_bg = Image.open(scene_bg_path)
                return scene, scene_bg
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_bg = {scene: bg for scene, bg in results}

    def load_scene_segmentation(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                scene_seg_path = os.path.join(self.dataset_path, 'segmentation', scene + '_seg.png')
                scene_seg = Image.open(scene_seg_path).convert('L')  # Already reduced transformation
                scene_seg_enc = np.eye(self.num_seg_classes)[np.array(scene_seg)]
                self.scene_seg[scene] = scene_seg_enc
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                scene_seg_path = os.path.join(dataset_path, 'segmentation', scene + '_seg.png')
                scene_seg = Image.open(scene_seg_path).convert('L')  # Already reduced transformation
                scene_seg_enc = np.eye(self.num_seg_classes)[np.array(scene_seg)]
                return scene, scene_seg_enc
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_seg = {scene: seg for scene, seg in results}
        
    def load_scene_size(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                scene_size_path = os.path.join(self.dataset_path, 'information', scene + '_info.json')
                with open(scene_size_path, 'r') as f:
                    scene_size = json.load(f)
                self.scene_size[scene] = scene_size
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                scene_size_path = os.path.join(dataset_path, 'information', scene + '_info.json')
                with open(scene_size_path, 'r') as f:
                    scene_size = json.load(f)
                return scene, scene_size
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_size = {scene: size for scene, size in results}
    
    def load_scene_homography(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                homography_path = os.path.join(self.dataset_path, 'homography', scene + '_H.txt')
                scene_H = np.loadtxt(homography_path)
                self.scene_H[scene] = scene_H
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                homography_path = os.path.join(dataset_path, 'homography', scene + '_H.txt')
                scene_H = np.loadtxt(homography_path)
                return scene, scene_H
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_H = {scene: H for scene, H in results}

    def load_dense_trajectory(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                trajectory_dense_path = os.path.join(self.dataset_path, 'trajectory_dense', scene + '_trajectory_dense.csv')
                trajectory_dense = pd.read_csv(trajectory_dense_path)
                self.trajectory_dense[scene] = trajectory_dense
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                trajectory_dense_path = os.path.join(dataset_path, 'trajectory_dense', scene + '_trajectory_dense.csv')
                trajectory_dense = pd.read_csv(trajectory_dense_path)
                return scene, trajectory_dense
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.trajectory_dense = {scene: traj_dense for scene, traj_dense in results}
    
    def load_trajectory(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                trajectory_path = os.path.join(self.dataset_path, 'trajectory', scene + '_trajectory.csv')
                trajectory = pd.read_csv(trajectory_path)
                self.trajectory[scene] = trajectory
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                trajectory_path = os.path.join(dataset_path, 'trajectory', scene + '_trajectory.csv')
                trajectory = pd.read_csv(trajectory_path)
                return scene, trajectory
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.trajectory = {scene: traj for scene, traj in results}

    def load_origin_goal(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                origin_goal_path = os.path.join(self.dataset_path, 'origin_goal', scene + '_origin_goal.csv')
                origin_goal = pd.read_csv(origin_goal_path)
                self.origin_goal[scene] = origin_goal
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                origin_goal_path = os.path.join(dataset_path, 'origin_goal', scene + '_origin_goal.csv')
                origin_goal = pd.read_csv(origin_goal_path)
                return scene, origin_goal
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.origin_goal = {scene: origin_goal for scene, origin_goal in results}

    def load_appearance_map(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                appearance_density_path = os.path.join(self.dataset_path, 'gt', scene + '_appearance_density.png')
                appearance_map = cv2.imread(appearance_density_path, cv2.IMREAD_GRAYSCALE)
                appearance_map = appearance_map.astype(np.float32) / 255
                self.scene_appearance[scene] = appearance_map
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                appearance_density_path = os.path.join(dataset_path, 'gt', scene + '_appearance_density.png')
                appearance_map = cv2.imread(appearance_density_path, cv2.IMREAD_GRAYSCALE)
                appearance_map = appearance_map.astype(np.float32) / 255
                return scene, appearance_map
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_appearance = {scene: appearance for scene, appearance in results}

    def load_population_map(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                population_density_path = os.path.join(self.dataset_path, 'gt', scene + '_population_density.png')
                population_map = cv2.imread(population_density_path, cv2.IMREAD_GRAYSCALE)
                population_map = population_map.astype(np.float32) / 255
                self.scene_population[scene] = population_map
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                population_density_path = os.path.join(dataset_path, 'gt', scene + '_population_density.png')
                population_map = cv2.imread(population_density_path, cv2.IMREAD_GRAYSCALE)
                population_map = population_map.astype(np.float32) / 255
                return scene, population_map
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_population = {scene: population for scene, population in results}

    def load_flow_map(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                flow_map_path = os.path.join(self.dataset_path, 'gt', scene + '_flow_map.png')
                flow_map = cv2.imread(flow_map_path, cv2.IMREAD_COLOR)
                flow_map = flow_map[..., [0, 1]].astype(np.float32) / 255
                self.scene_flow[scene] = flow_map
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                flow_map_path = os.path.join(dataset_path, 'gt', scene + '_flow_map.png')
                flow_map = cv2.imread(flow_map_path, cv2.IMREAD_COLOR)
                flow_map = flow_map[..., [0, 1]].astype(np.float32) / 255
                return scene, flow_map
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_flow = {scene: flow for scene, flow in results}

    def load_navigation_mesh(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                navmesh_path = os.path.join(self.dataset_path, 'navmesh', scene + '_navmesh.json')
                with open(navmesh_path, 'r') as f:
                    navigation_mesh = json.load(f)
                self.navigation_mesh[scene] = navigation_mesh
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                navmesh_path = os.path.join(dataset_path, 'navmesh', scene + '_navmesh.json')
                with open(navmesh_path, 'r') as f:
                    navigation_mesh = json.load(f)
                return scene, navigation_mesh
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.navigation_mesh = {scene: nav_mesh for scene, nav_mesh in results}

    def load_scene_walkable(self, parallel=False):
        if not parallel:
            for scene in self.scene_list:
                walkable_area_path = os.path.join(self.dataset_path, 'navmesh', scene + '_walkable_area.png')
                walkable_area = cv2.imread(walkable_area_path, cv2.IMREAD_GRAYSCALE)
                self.scene_walkable[scene] = walkable_area
        else:
            @staticmethod
            def load_function(scene, dataset_path):
                walkable_area_path = os.path.join(dataset_path, 'navmesh', scene + '_walkable_area.png')
                walkable_area = cv2.imread(walkable_area_path, cv2.IMREAD_GRAYSCALE)
                return scene, walkable_area
            
            results = Parallel(n_jobs=-1)(delayed(load_function)(scene, self.dataset_path) for scene in self.scene_list)
            self.scene_walkable = {scene: walkable for scene, walkable in results}

    def load_data_all(self):
        self.load_scene_image()
        self.load_scene_background()
        self.load_scene_segmentation()
        self.load_scene_size()
        self.load_scene_homography()
        self.load_dense_trajectory()
        self.load_trajectory()
        self.load_origin_goal()
        self.load_appearance_map()
        self.load_population_map()
        self.load_flow_map()
        self.load_navigation_mesh()
        self.load_scene_walkable()
