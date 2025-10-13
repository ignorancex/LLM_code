import numpy as np
from tqdm import tqdm

from utils.dataloader.base_dataloader import BaseDataset


class EvaluationDataset(BaseDataset):
    r"""Dataloader for evaluation"""

    def __init__(self, config, phase):
        super().__init__(config, phase)
        
        pbar = tqdm(range(len(self.scene_list)))
        pbar.set_description(f'Evaluation Dataset {phase}: Load data')

        # Model inference
        self.load_scene_image()
        self.load_scene_segmentation()
        self.load_scene_homography()
        self.load_scene_walkable()
        self.load_navigation_mesh()

        # Model evaluation
        self.load_scene_size()
        self.load_dense_trajectory()

        # Visualization
        self.load_scene_background()
        
        pbar.set_description(f'Evaluation Dataset {phase}: {len(self.scene_list)} Scenes')
        pbar.total = pbar.n = len(self.scene_list)
        pbar.close()

    def __len__(self):
        return len(self.scene_list)
    
    def __getitem__(self, index):
        scene = self.scene_list[index]
        img = self.scene_img[scene]
        seg = self.scene_seg[scene]
        H = self.scene_H[scene]
        walkable = self.scene_walkable[scene]
        navmesh = self.navigation_mesh[scene]

        size = self.scene_size[scene]
        trajectory_dense = self.trajectory_dense[scene]
        
        bg = self.scene_bg[scene]

        out = {'scene': scene,
               'img': np.array(img),
               'seg': seg,
               'H': H,
               'walkable': walkable,
               'navmesh': navmesh,
               'size': size,
               'trajectory_dense': trajectory_dense,
               'bg': bg}
        
        return out
