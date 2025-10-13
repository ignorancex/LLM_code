import os
import numpy as np
from tqdm import tqdm
from scipy.spatial import Delaunay
import pathfinder.navmesh_baker as nmb
import pickle
import cv2
from PIL import Image

from utils.config import DotDict
from utils.homography import generate_homography
from utils.dataloader.base_dataloader import BaseDataset


class SyntheticDataset(BaseDataset):
    r"""Dataloader for synthetic dataset"""

    def __init__(self, config, scene_list):
        super().__init__(config, phase='test')
        
        pbar = tqdm(range(1))
        pbar.set_description(f'Synthetic Dataset {scene_list}: Load data')

        # Load scene image
        pbar.set_description(f'Synthetic Dataset {scene_list}: Load scene image')
        self.scene_list = scene_list
        self.dataset_path = './datasets/Synthetic/'

        for scene in self.scene_list:
            scene_bg_path = os.path.join(self.dataset_path, 'image_terrain', scene + '_bg.png')
            scene_bg = Image.open(scene_bg_path)
            self.scene_bg[scene] = scene_bg
            self.scene_img[scene] = scene_bg

        # Load appearance and population map
        pbar.set_description(f'Synthetic Dataset {scene_list}: Load appearance and population map')
        for scene in self.scene_list:
            appearance_density_path = os.path.join(self.dataset_path, 'gt', scene + '_appearance_density.png')
            appearance_map = cv2.imread(appearance_density_path, cv2.IMREAD_GRAYSCALE)
            appearance_map = appearance_map.astype(np.float32) / 255
            self.scene_appearance[scene] = appearance_map

            population_density_path = os.path.join(self.dataset_path, 'gt', scene + '_population_density.png')
            population_map = cv2.imread(population_density_path, cv2.IMREAD_GRAYSCALE)
            population_map = population_map.astype(np.float32) / 255
            self.scene_population[scene] = population_map

        # Make scene size
        pbar.set_description(f'Synthetic Dataset {scene_list}: Make scene size')
        for scene in self.scene_list:
            width, height = self.scene_bg[scene].size
            self.scene_size[scene] = DotDict({'width': width, 'height': height})
            self.scene_size[scene]['length'] = 30 * 60  # 30 fps * 60 seconds (temporary)
            self.scene_size[scene]['num_agents'] = 0
            self.scene_size[scene]['fps'] = 30
            self.scene_size[scene]['sim_fps'] = 5
            self.scene_size[scene]['population_probability'] = {0: 1.0}

        # Make scene homography
        pbar.set_description(f'Synthetic Dataset {scene_list}: Make scene homography')
        for scene in self.scene_list:
            # read file in the dataset
            scale_path = os.path.join(self.dataset_path, 'homography', scene + '_scale.txt')
            with open(scale_path, 'r') as f:
                scale = float(f.readline().strip())
            scene_H = generate_homography(scale=1/scale)
            self.scene_H[scene] = scene_H

        # Make Blank ground-truth trajectory 
        pbar.set_description(f'Synthetic Dataset {scene_list}: Make none')
        for scene in self.scene_list:
            self.trajectory_dense[scene] = None
            self.trajectory[scene] = None
            self.origin_goal[scene] = None

        # Build segmentation map & walkable map
        pbar.set_description(f'Synthetic Dataset {scene_list}: Make segmentation map & walkable map')
        for scene in self.scene_list:
            scene_walkable = self.scene_population[scene].copy()
            scene_walkable = (scene_walkable > 0.5).astype(np.uint8)
            self.scene_walkable[scene] = scene_walkable

            # Make 0 to class bush, 1 to class sidewalk
            grass_id = self.label2id['grass']
            sidewalk_id = self.label2id['sidewalk']
            scene_seg = np.zeros_like(scene_walkable)
            scene_seg[scene_seg == 0] = grass_id
            scene_seg[scene_seg == 1] = sidewalk_id
            
            # Convert to one-hot encoding
            scene_seg = np.eye(len(self.id2label))[scene_seg]
            self.scene_seg[scene] = scene_seg

        # Build navigation mesh
        pbar.set_description(f'Synthetic Dataset {scene_list}: Make navigation mesh')
        for scene in self.scene_list:
            # Check there are navmesh cache
            navmesh_path = os.path.join(self.dataset_path, 'navmesh', scene + '_navmesh.pkl')
            if os.path.exists(navmesh_path):
                with open(navmesh_path, 'rb') as f:
                    navmesh = pickle.load(f)
                self.navigation_mesh[scene] = navmesh
                continue

            scene_walkable = self.scene_walkable[scene]
            binary = scene_walkable.copy() * 255

            # Bake navigation mesh
            dilation=6
            erosion=0
            fast_baking=False
            refine_mesh=False
            baking_divider=10.0
            H, W = binary.shape
            
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

            # print(f'Input Source: {filename} Vertices: {len(vertices)} Polygons: {len(polygons)}') if verbose else None

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
            # print(f'Baked Source: {filename} Vertices: {len(baked_vertices)} Polygons: {len(baked_polygons)}') if verbose else None
            
            navmesh = {'vertices': baked_vertices, 'polygons': baked_polygons}
            self.navigation_mesh[scene] = navmesh

            # Save navigation mesh for future use
            navmesh_path = os.path.join(self.dataset_path, 'navmesh', scene + '_navmesh.pkl')
            os.makedirs(os.path.dirname(navmesh_path), exist_ok=True)
            with open(navmesh_path, 'wb') as f:
                pickle.dump(navmesh, f)

        pbar.set_description(f'Synthetic Dataset {self.scene_list}: {len(self.scene_list)} Scenes')
        pbar.total = pbar.n = len(self.scene_list)
        pbar.close()


    def __len__(self):
        return len(self.scene_list)
    
    def __getitem__(self, index):
        scene = self.scene_list[index]
        img = self.scene_img[scene]
        bg = self.scene_bg[scene]
        seg = self.scene_seg[scene]
        size = self.scene_size[scene]
        H = self.scene_H[scene]

        trajectory_dense = self.trajectory_dense[scene]
        trajectory = self.trajectory[scene]
        origin_goal = self.origin_goal[scene]

        appearance = self.scene_appearance[scene]
        population = self.scene_population[scene]
        
        walkable = self.scene_walkable[scene]
        navmesh = self.navigation_mesh[scene]

        out = {'scene': scene,
               'img': np.array(img),
               'bg': bg,
               'seg': seg,
               'size': size,
               'H': H,
               'trajectory_dense': trajectory_dense,
               'trajectory': trajectory,
               'origin_goal': origin_goal,
               'appearance': appearance,
               'population': population,
               'walkable': walkable,
               'navmesh': navmesh}
        
        return out
