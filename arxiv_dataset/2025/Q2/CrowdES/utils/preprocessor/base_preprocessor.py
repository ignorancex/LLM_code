import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from itertools import chain, combinations

import cv2
from PIL import Image
from scipy.spatial import Delaunay
import pathfinder.navmesh_baker as nmb

from utils.config import DotDict
from homography import generate_homography, image2world, world2image
from utils.image import generate_density_map, generate_flow_map


class BaseDataPreprocessor(Dataset):
    r"""Base class for dataset preprocessor"""

    def __init__(self, config, phase):
        assert phase in ['train', 'test']
        self.config = config
        self.phase = phase

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
        self.label2id = {k: int(v) for k, v in self.label2id.items()}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_seg_classes = len(self.label2id)
        self.seg_ignore_label = 255
        
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
    
    @staticmethod
    def reduce_labels_transform(labels: np.ndarray, **kwargs) -> np.ndarray:
        """Set `0` label as with value 255 and then reduce all other labels by 1.

        Example:
            Initial class labels:         0 - background; 1 - road; 2 - car;
            Transformed class labels:   255 - background; 0 - road; 1 - car;

        **kwargs are required to use this function with albumentations.
        """
        labels[labels == 0] = 255
        labels = labels - 1
        labels[labels == 254] = 255
        return labels
    
    @staticmethod
    def trim_and_downsample_trajectory(dense_trajectory, scene_size, frame_skip):
        # Trajectory filtering and downsampling
        scene_width, scene_height = scene_size['width'], scene_size['height']

        dense_trajectory_filtered = []
        downsampled_traj = []
        for agent_id, traj in dense_trajectory.groupby('agent_id'):
            mask = (traj['x'] >= 0) & (traj['x'] <= scene_width - 1) & (traj['y'] >= 0) & (traj['y'] <= scene_height - 1)
            start_idx, end_idx = np.argmax(mask), len(mask) - np.argmax(mask[::-1]) - 1
            if start_idx < end_idx:
                traj_temp = traj.iloc[start_idx:end_idx+1]
                traj_temp.loc[:, 'x'] = traj_temp['x'].clip(0, scene_width - 1)
                traj_temp.loc[:, 'y'] = traj_temp['y'].clip(0, scene_height - 1)
                frame_mask = traj_temp['frame'] % frame_skip == 0
                if frame_mask.sum() >= 2:
                    traj_temp.loc[:, 'agent_id'] = len(dense_trajectory_filtered)  # Reset agent_id
                    dense_trajectory_filtered.append(traj_temp)
                    downsampled_traj.append(traj_temp[frame_mask])

        dense_trajectory = pd.concat(dense_trajectory_filtered, axis=0, ignore_index=True)
        downsampled_trajectory = pd.concat(downsampled_traj, axis=0, ignore_index=True)
        return dense_trajectory, downsampled_trajectory

    @staticmethod
    def origin_and_goal_extraction(dense_trajectory, homography, fps, standing_threshold=0.1):
        # Make origin and goal pair for each agent
        origin_goal = []
        origin_goal_column_name = ['scene', 'agent_id', 'agent_type', 'frame_origin', 'origin_x', 'origin_y', 'frame_goal', 'goal_x', 'goal_y']
        for agent_id, traj in dense_trajectory.groupby('agent_id'):
            origin_goal.append([traj['scene'].iloc[0], agent_id, traj['agent_type'].iloc[0], 
                                traj['frame'].iloc[0], traj['x'].iloc[0], traj['y'].iloc[0], 
                                traj['frame'].iloc[-1], traj['x'].iloc[-1], traj['y'].iloc[-1]])
        origin_goal = pd.DataFrame(origin_goal, columns=origin_goal_column_name)

        # Calculate average speed using dense trajectory
        origin_goal['preferred_speed'] = 0.
        for agent_id, traj in dense_trajectory.groupby('agent_id'):
            traj_meter = image2world(traj[['x', 'y']].values, homography)
            speed_frame = np.linalg.norm(np.diff(traj_meter, axis=0), axis=1) * fps
            speed_initial = np.mean(speed_frame)

            # filter out the speed with standing still threshold (0.1 m/s)
            speed_frame = speed_frame[speed_frame > standing_threshold]

            speed = np.mean(speed_frame) if len(speed_frame) > 0 else speed_initial
            origin_goal.loc[origin_goal['agent_id'] == agent_id, 'preferred_speed'] = speed

        return origin_goal

    @staticmethod
    def generate_gt_map(scene_size, origin_goal, dense_trajectory, output_map=['appearance', 'population', 'flow'], **kwargs):
        # Extract the arguments
        scene_size = (scene_size['width'], scene_size['height'])
        mode = kwargs.get('mode', 'replace')
        clip_value = kwargs.get('clip_value', 3)
        blur_sigma = kwargs.get('blur_sigma', 15)
        
        # Draw an appearance density map
        if 'appearance' in output_map:
            origin = np.round(origin_goal[['frame_origin', 'origin_x', 'origin_y']].values).astype(np.int32)
            goal = np.round(origin_goal[['frame_goal', 'goal_x', 'goal_y']].values).astype(np.int32)
            
            # Filter out initial and final frames because it is not a real start and finish point
            origin = origin[(origin[:, 0] != origin[:, 0].min())][:, 1:]
            goal = goal[(goal[:, 0] != goal[:, 0].max())][:, 1:]
            points = np.concatenate([origin, goal], axis=0)
            appearance_density_map = generate_density_map(scene_size, points, mode=mode, clip_value=clip_value, blur_sigma=blur_sigma, normalize=True)  # False
        else:
            appearance_density_map = None

        # Draw a population density map
        if 'population' in output_map:
            points = np.round(dense_trajectory[['x', 'y']].values).astype(np.int32)
            population_density_map = generate_density_map(scene_size, points, mode=mode, clip_value=clip_value, blur_sigma=blur_sigma, normalize=True)
        else:
            population_density_map = None

        # Draw a crowd flow map
        if 'flow' in output_map:
            dense_traj_list = []
            for agent_id, traj in dense_trajectory.groupby('agent_id'):
                dense_traj_list.append(traj[['x', 'y']].values)
            crowd_flow_map = generate_flow_map(scene_size, dense_traj_list, blur_sigma=blur_sigma)
        else:
            crowd_flow_map = None

        return appearance_density_map, population_density_map, crowd_flow_map
    
    def scene_augmentation(self, augmentations, frame_skip=1):
        scene_list = list(self.scene_img.keys())
        augmentation_combinations  = chain.from_iterable(combinations(augmentations, r) for r in range(1, len(augmentations) + 1))
        for combination in augmentation_combinations:
            for scene in scene_list:
                self.apply_augmentations(scene, combination, frame_skip)

    def apply_augmentations(self, scene, augmentation_combination, frame_skip):
        """Apply a series of augmentations in a given combination."""
        
        # Initialize the augmented data with the original values
        scene_name_aug = scene
        scene_img_aug = self.scene_img[scene].copy()
        scene_bg_aug = self.scene_bg[scene].copy()
        scene_seg_aug = self.scene_seg[scene].copy()
        scene_size_aug = self.scene_size[scene].copy()
        scene_H_aug = self.scene_H[scene].copy()
        trajectory_dense_aug = self.trajectory_dense[scene].copy(deep=True)
        trajectory_aug = self.trajectory[scene].copy(deep=True)
        origin_goal_aug = self.origin_goal[scene].copy(deep=True)
        scene_appearance_aug = self.scene_appearance[scene].copy()
        scene_population_aug = self.scene_population[scene].copy()
        scene_flow_aug = self.scene_flow[scene].copy()

        # Apply each augmentation in sequence
        for augmentation in augmentation_combination:
            if augmentation == 'hflip':
                # Horizontal flip
                scene_img_aug = scene_img_aug.transpose(Image.FLIP_LEFT_RIGHT)
                scene_bg_aug = scene_bg_aug.transpose(Image.FLIP_LEFT_RIGHT)
                scene_seg_aug = np.flip(scene_seg_aug, axis=1)
                scene_appearance_aug = np.flip(scene_appearance_aug, axis=1)
                scene_population_aug = np.flip(scene_population_aug, axis=1)
                scene_flow_aug = np.flip(scene_flow_aug, axis=1)
                
                # Flip trajectory and trajectory_dense in x direction
                trajectory_aug['x'] = scene_size_aug['width'] - trajectory_aug['x']
                trajectory_dense_aug['x'] = scene_size_aug['width'] - trajectory_dense_aug['x']
                
                # Adjust origin_goal
                origin_goal_aug['origin_x'] = scene_size_aug['width'] - origin_goal_aug['origin_x']
                origin_goal_aug['goal_x'] = scene_size_aug['width'] - origin_goal_aug['goal_x']
                
                # Adjust homography for horizontal flip
                scene_H_aug_inv = np.linalg.inv(scene_H_aug)
                scene_H_aug_inv[0, 2] = scene_size_aug['width'] - scene_H_aug_inv[0, 2]
                scene_H_aug_inv[0, 0] *= -1
                scene_H_aug = np.linalg.inv(scene_H_aug_inv)
                
                scene_name_aug += '_hflip'

            elif augmentation == 'vflip':
                # Vertical flip
                scene_img_aug = scene_img_aug.transpose(Image.FLIP_TOP_BOTTOM)
                scene_bg_aug = scene_bg_aug.transpose(Image.FLIP_TOP_BOTTOM)
                scene_seg_aug = np.flip(scene_seg_aug, axis=0)
                scene_appearance_aug = np.flip(scene_appearance_aug, axis=0)
                scene_population_aug = np.flip(scene_population_aug, axis=0)
                scene_flow_aug = np.flip(scene_flow_aug, axis=0)
                
                # Flip trajectory and trajectory_dense in y direction
                trajectory_aug['y'] = scene_size_aug['height'] - trajectory_aug['y']
                trajectory_dense_aug['y'] = scene_size_aug['height'] - trajectory_dense_aug['y']
                
                # Adjust origin_goal
                origin_goal_aug['origin_y'] = scene_size_aug['height'] - origin_goal_aug['origin_y']
                origin_goal_aug['goal_y'] = scene_size_aug['height'] - origin_goal_aug['goal_y']
                
                # Adjust homography for vertical flip
                scene_H_aug_inv = np.linalg.inv(scene_H_aug)
                scene_H_aug_inv[1, 2] = scene_size_aug['height'] - scene_H_aug_inv[1, 2]
                scene_H_aug_inv[1, 1] *= -1
                scene_H_aug = np.linalg.inv(scene_H_aug_inv)
                
                scene_name_aug += '_vflip'

            elif augmentation == 'tp':
                # Transpose (swap x and y)
                scene_img_aug = scene_img_aug.transpose(Image.TRANSPOSE)
                scene_bg_aug = scene_bg_aug.transpose(Image.TRANSPOSE)
                scene_seg_aug = np.transpose(scene_seg_aug, (1, 0, 2))
                scene_appearance_aug = np.transpose(scene_appearance_aug, (1, 0))
                scene_population_aug = np.transpose(scene_population_aug, (1, 0))
                scene_flow_aug = np.transpose(scene_flow_aug, (1, 0, 2))
                
                # Swap x and y in trajectory and trajectory_dense
                trajectory_aug[['x', 'y']] = trajectory_aug[['y', 'x']]
                trajectory_dense_aug[['x', 'y']] = trajectory_dense_aug[['y', 'x']]
                
                # Adjust origin_goal
                origin_goal_aug[['origin_x', 'origin_y']] = origin_goal_aug[['origin_y', 'origin_x']]
                origin_goal_aug[['goal_x', 'goal_y']] = origin_goal_aug[['goal_y', 'goal_x']]
                
                # Adjust homography for transpose (swap x and y)
                scene_H_aug[[0, 1]] = scene_H_aug[[1, 0]]  # Swap rows
                scene_H_aug[:, [0, 1]] = scene_H_aug[:, [1, 0]]  # Swap columns
                
                # Swap scene size (width and height)
                scene_size_aug['width'], scene_size_aug['height'] = scene_size_aug['height'], scene_size_aug['width']
                
                scene_name_aug += '_tp'

            elif augmentation == 'rev':
                # Time reverse
                trajectory_aug = trajectory_aug.iloc[::-1].reset_index(drop=True)
                trajectory_dense_aug = trajectory_dense_aug.iloc[::-1].reset_index(drop=True)
                
                # Reset frame values using max frame
                max_frame = trajectory_dense_aug['frame'].max()
                max_frame = int(np.ceil(max_frame / frame_skip) * frame_skip)  # Ensure the frame is divisible by frame_skip
                trajectory_aug['frame'] = max_frame - trajectory_aug['frame']
                trajectory_dense_aug['frame'] = max_frame - trajectory_dense_aug['frame']
                origin_goal_aug['frame_origin'] = max_frame - origin_goal_aug['frame_origin']
                origin_goal_aug['frame_goal'] = max_frame - origin_goal_aug['frame_goal']

                # swap origin_goal
                origin_goal_aug[['origin_x', 'origin_y', 'frame_origin']], origin_goal_aug[['goal_x', 'goal_y', 'frame_goal']] = \
                    origin_goal_aug[['goal_x', 'goal_y', 'frame_goal']], origin_goal_aug[['origin_x', 'origin_y', 'frame_origin']]
                
                scene_name_aug += '_rev'

        # Apply the scene_name_aug to the trajectories
        trajectory_aug['scene'] = scene_name_aug
        trajectory_dense_aug['scene'] = scene_name_aug

        # Store the augmented data
        self.scene_img[scene_name_aug] = scene_img_aug
        self.scene_bg[scene_name_aug] = scene_bg_aug
        self.scene_seg[scene_name_aug] = scene_seg_aug
        self.scene_size[scene_name_aug] = scene_size_aug
        self.scene_H[scene_name_aug] = scene_H_aug
        self.trajectory_dense[scene_name_aug] = trajectory_dense_aug
        self.trajectory[scene_name_aug] = trajectory_aug
        self.origin_goal[scene_name_aug] = origin_goal_aug
        self.scene_appearance[scene_name_aug] = scene_appearance_aug
        self.scene_population[scene_name_aug] = scene_population_aug
        self.scene_flow[scene_name_aug] = scene_flow_aug

        return scene_name_aug
    
    def generate_navmesh(self, walkable_id):
        scene_list = list(self.scene_seg.keys())
        results = Parallel(n_jobs=-1)(delayed(self.generate_mesh_from_image)(self.scene_seg[scene], walkable_id) for scene in scene_list)
        for scene, (walkable_area, navmesh) in zip(scene_list, results):
            self.scene_walkable[scene] = walkable_area
            self.navigation_mesh[scene] = navmesh

    @staticmethod
    def generate_mesh_from_image(scene_seg, walkable_id=[5, 6, 7], dilation=6, erosion=0, fast_baking=False, refine_mesh=False, baking_divider=10.0, verbose=False):
        H, W, _ = scene_seg.shape
        # Here, scene_seg is one-hot encoded
        # Mark the walkable area as white
        walkable_id = np.array(walkable_id) - 1
        walkable_area = np.any(scene_seg[:, :, walkable_id] > 0.5, axis=-1).astype(np.uint8)
        binary = walkable_area.copy() * 255
        
        # Perform morphological operations to fill in gaps and smooth the white areas
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=dilation) if dilation > 0 else binary
        binary = cv2.erode(binary, kernel, iterations=erosion) if erosion > 0 else binary

        if fast_baking:
            # Find contours of the white areas in the image
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            vertices = np.vstack(list(contours[i] for i in range(len(contours)))).squeeze()  # (X, Y)

            if refine_mesh:
                y, x = np.meshgrid(np.arange(0, H, 32), np.arange(0, W, 32))
                grid_points = np.array([x.flatten(), y.flatten()]).T
                vertices = np.vstack((vertices, grid_points))

            # Perform Delaunay triangulation on the 2D points
            delaunay_tri = Delaunay(vertices)

            # Filter out triangles whose centers fall on black areas
            polygons = []
            for simplex in delaunay_tri.simplices:
                tri_pts = vertices[simplex]
                center_x, center_y = np.mean(tri_pts, axis=0).astype(int)
                if binary[center_y, center_x] == 255:
                    polygons.append(simplex)
            polygons = np.array(polygons)
            
            # Removes unused points and remaps polygon indices
            used_idx = np.unique(polygons)
            idx_map = np.zeros(vertices.shape[0], dtype=int)
            idx_map[used_idx] = np.arange(used_idx.shape[0])
            vertices = vertices[used_idx]
            polygons = idx_map[polygons]

            # Extend the edge of the image to prevent the agent from falling off the edge
            vertices[vertices[:, 0] == 0, 0] = -W//2
            vertices[vertices[:, 0] == W-1, 0] = W-1+W//2
            vertices[vertices[:, 1] == 0, 1] = -H//2
            vertices[vertices[:, 1] == H-1, 1] = H-1+H//2

        else:
            # Add all white pixel squares as vertices
            white_points = np.argwhere(binary == 255)[:, [1, 0]]
            vertices = np.vstack((white_points, white_points + [1, 0], white_points + [1, 1], white_points + [0, 1]))
            polygons = np.arange(4) * len(white_points) + np.arange(len(white_points))[:, None]

            # Removes duplicated points and remaps polygon indices
            vertices, idx_map = np.unique(vertices, axis=0, return_inverse=True)
            polygons = idx_map[polygons]

            # Extend the edge of the image to prevent the agent from falling off the edge
            vertices[vertices[:, 0] == 0, 0] = -W//2
            vertices[vertices[:, 0] == W, 0] = W+W//2
            vertices[vertices[:, 1] == 0, 1] = -H//2
            vertices[vertices[:, 1] == H, 1] = H+H//2

        print(f'Input Source - Vertices: {len(vertices)} Polygons: {len(polygons)}') if verbose else None

        # Convert from xy to Unity coordinate system
        vertices_temp = np.array((vertices[:, 0], np.zeros(vertices.shape[0]), vertices[:, 1])).T
        vertices_temp = vertices_temp / baking_divider 
        polygons_temp = polygons[:, ::-1]

        # Baking the mesh
        navmesh_baker = nmb.NavmeshBaker()
        navmesh_baker.add_geometry(vertices_temp, polygons_temp)
        is_bake: bool = navmesh_baker.bake(agent_radius=0.1)
        if not is_bake:
            raise Exception(f'Failed to bake the mesh on the image')
        
        baked_vertices, baked_polygons = navmesh_baker.get_polygonization()
        baked_vertices = np.around(np.array(baked_vertices)[:, [0, 2]] * baking_divider, decimals=4)
        baked_vertices = baked_vertices.tolist()

        print(f'Baked Source - Vertices: {len(baked_vertices)} Polygons: {len(baked_polygons)}') if verbose else None
        
        navmesh = {'vertices': baked_vertices, 'polygons': baked_polygons}
        return walkable_area, navmesh

    def scene_statistics(self):
        scene_list = list(self.scene_img.keys())
        for scene in scene_list:
            width = self.scene_size[scene]['width']
            height = self.scene_size[scene]['height']
            self.scene_size[scene] = DotDict({'width': width, 'height': height})
            self.scene_size[scene]['length'] = int(self.trajectory_dense[scene]['frame'].max()) + 1
            self.scene_size[scene]['num_agents'] = len(self.trajectory_dense[scene]['agent_id'].unique())
            self.scene_size[scene]['fps'] = self.config.dataset.dataset_fps
            self.scene_size[scene]['sim_fps'] = self.config.crowd_simulator.simulator.simulator_fps

            # generate statistics of number of agent at each frame
            agent_count = self.trajectory_dense[scene].groupby('frame').size()
            agent_count = agent_count.value_counts().sort_index()
            agent_count = agent_count / self.scene_size[scene]['length'] # to probability
            agent_count = agent_count.to_dict()
            agent_count[0] = 1 - sum(agent_count.values())
            agent_count = dict(sorted(agent_count.items()))
            self.scene_size[scene]['population_probability'] = agent_count

    def save_data(self, config, phase):
        scene_list = list(self.scene_img.keys())
        base_path = os.path.join(config.dataset.dataset_preprocessed_path, phase)

        # Create directories
        os.makedirs(os.path.join(base_path, 'image'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'image_terrain'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'segmentation'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'information'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'homography'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'trajectory_dense'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'trajectory'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'origin_goal'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'navmesh'), exist_ok=True)

        for scene in scene_list:
            # Save scene image
            scene_img_path = os.path.join(base_path, 'image', scene + '_image.png')
            self.scene_img[scene].save(scene_img_path)

            # Save scene background
            scene_bg_path = os.path.join(base_path, 'image_terrain', scene + '_bg.png')
            self.scene_bg[scene].save(scene_bg_path)

            # Save scene segmentation
            scene_seg_path = os.path.join(base_path, 'segmentation', scene + '_seg.png')
            scene_seg = np.argmax(self.scene_seg[scene], axis=2)
            cv2.imwrite(scene_seg_path, scene_seg.astype(np.uint8))

            # Save scene_size
            scene_size_path = os.path.join(base_path, 'information', scene + '_info.json')
            with open(scene_size_path, 'w') as f:
                json.dump(self.scene_size[scene], f)

            # Save homography matrix
            homography_path = os.path.join(base_path, 'homography', scene + '_H.txt')
            np.savetxt(homography_path, self.scene_H[scene])

            # Save dense trajectory
            trajectory_dense_path = os.path.join(base_path, 'trajectory_dense', scene + '_trajectory_dense.csv')
            self.trajectory_dense[scene].to_csv(trajectory_dense_path, index=False)

            # Save trajectory
            trajectory_path = os.path.join(base_path, 'trajectory', scene + '_trajectory.csv')
            self.trajectory[scene].to_csv(trajectory_path, index=False)

            # Save origin_goal
            origin_goal_path = os.path.join(base_path, 'origin_goal', scene + '_origin_goal.csv')
            self.origin_goal[scene].to_csv(origin_goal_path, index=False)

            # Save appearance map
            appearance_density_path = os.path.join(base_path, 'gt', scene + '_appearance_density.png')
            cv2.imwrite(appearance_density_path, (self.scene_appearance[scene] * 255).astype(np.uint8))

            # Save population map
            population_density_path = os.path.join(base_path, 'gt', scene + '_population_density.png')
            cv2.imwrite(population_density_path, (self.scene_population[scene] * 255).astype(np.uint8))

            # Save flow map
            flow_map_path = os.path.join(base_path, 'gt', scene + '_flow_map.png')
            rgb_flow_map = np.concatenate([self.scene_flow[scene], np.zeros((*self.scene_flow[scene].shape[:2], 1))], axis=2)#[..., [0, 2, 1]]
            cv2.imwrite(flow_map_path, (rgb_flow_map * 255).astype(np.uint8))

            # Save navigation mesh
            navmesh_path = os.path.join(base_path, 'navmesh', scene + '_navmesh.json')
            with open(navmesh_path, 'w') as f:
                json.dump(self.navigation_mesh[scene], f)

            # Save scene walkable area
            walkable_area_path = os.path.join(base_path, 'navmesh', scene + '_walkable_area.png')
            cv2.imwrite(walkable_area_path, self.scene_walkable[scene])
