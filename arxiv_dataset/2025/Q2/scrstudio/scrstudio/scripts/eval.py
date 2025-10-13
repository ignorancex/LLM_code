# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import poselib
import torch
import tyro
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from scrstudio.configs.method_configs import method_configs
from scrstudio.data.samplers import PQKNN


def pose_estimate(keypoints, scene_coords, camera,ransac_opt ,  gt_pose ):
    pose, info = poselib.estimate_absolute_pose(keypoints, scene_coords, camera, ransac_opt, {})
    pred_pose = np.eye(4)
    R_pred=pose.R.T
    pred_pose[:3,:3]=R_pred
    pred_pose[:3,3]=-R_pred@pose.t

    return {
        "num_inliers": info['num_inliers'],
        "t_err": np.linalg.norm(gt_pose[:3,3] - pred_pose[:3,3]),
        "r_err": np.linalg.norm(cv2.Rodrigues(gt_pose[:3,:3]@pose.R)[0]) * 180 / math.pi,
        "pose_q": pose.q,
        "pose_t": pose.t,
    }


acc_thresh = {
    "outdoor": ((5, 10), (0.5, 5), (0.25, 2)),
    "indoor": ((1, 5), (0.25, 2), (0.1, 1))
}
@dataclass
class ComputeKNNPose:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    config: Optional[str] = None
    load_config: Optional[Path] = None
    load_checkpoint: Optional[Path] = None
    data: Optional[Path] = None
    split: str = "test"
    retrieval: str = "netvlad_feats"
    n_neighbors: int = 10
    num_workers: int = 8
    output_path: Optional[Path] = None
    threshold: float = 10
    max_iter: int = 10000
    suffix: str = ""
    acc_thresh: str = "outdoor"
    # whether to use the base name of the file for the frame name
    base_name: bool = False

    def main(self) -> None:

        if self.config is not None:
            config = method_configs[self.config]
        elif self.load_config is not None:
            config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)
        elif self.load_checkpoint is not None and self.load_checkpoint.parts[-4] in method_configs:
            config = method_configs[self.load_checkpoint.parts[-4]]
        else:
            raise ValueError("Config not found.")
        device = torch.device("cuda")
        ckpt_path=self.load_checkpoint
        if ckpt_path is None:
            assert self.load_config is not None, "If no checkpoint provided, config file must be provided."
            ckpt_path= self.load_config.parent / 'scrstudio_models' / 'head.pt'
        state_dict = torch.load(ckpt_path, map_location=device,weights_only=True)

        if self.data is not None:
            config.data=self.data
        assert config.data is not None, "Data path must be provided."
        train_dataset=config.pipeline.datamanager.train_dataset
        train_dataset.data=config.data
        trainset=train_dataset.setup()
        model=config.pipeline.model.setup(metadata=trainset.metadata)
        model.load_state_dict(state_dict)

        encoder=config.pipeline.datamanager.encoder.setup(data_path=config.data)

        eval_dataset=config.pipeline.datamanager.eval_dataset
        eval_dataset.data=config.data
        eval_dataset.split=self.split

        value_feats=torch.tensor(trainset.global_feats,device=device)
        C_global=value_feats.shape[1]
        query_feats=config.data / self.split/ f"{self.retrieval}.npy"
        query_feats=np.load(query_feats)
        key_pq_path=config.data / "train"/ f"{self.retrieval}_pq.pkl"
        with open(key_pq_path, 'rb') as f:
            pq, codes = pickle.load(f)
        knn=PQKNN(pq,codes,n_neighbors=self.n_neighbors)
        testset=eval_dataset.setup(preprocess=encoder.preprocess)

        encoder = encoder.to(device)
        encoder.eval()
        model = model.to(device)
        model.eval()

        if self.output_path:
            output_dir = self.output_path
        elif self.load_config:
            output_dir = self.load_config.parent / "test_results"
        elif self.load_checkpoint:
            output_dir = self.load_checkpoint.parent.parent / "test_results"
        else:
            raise ValueError("Output path not specified.")
        output_dir.mkdir(parents=True, exist_ok=True)
        testset_loader = DataLoader(testset, shuffle=False, num_workers=self.num_workers)

        log_file = output_dir / f'testlog{self.suffix}.txt'
        _logger = logging.getLogger(__name__)
        _logger.addHandler(logging.FileHandler(log_file))
        _logger.info(f'Test images found: {len(testset)}')
        metric_file = output_dir / f'test{self.suffix}.txt'
        pose_file = output_dir / f'poses{self.suffix}.txt'

        metrics=defaultdict(list)
        poses=defaultdict(list)
        pool = Pool(self.num_workers)
        pool_results = []
        ransac_opt = {'max_reproj_error': self.threshold ,'max_iterations' : self.max_iter}

        # Testing loop.
        with torch.no_grad():
            for batch in tqdm(testset_loader):
                image,  idx = batch['image'].to(device, non_blocking=True), batch['idx']
                neighbor_indices=knn.kneighbors(query_feats[idx[0]])
                global_feat=value_feats[neighbor_indices] 
                assert global_feat.shape[0] == self.n_neighbors
                assert global_feat.ndim == 2

                with autocast("cuda",enabled=True):
                    encoder_output= encoder.keypoint_features({"image": image}, n=0)
                    keypoints = encoder_output["keypoints"]
                    descriptors = encoder_output["descriptors"]
                    N, C_local= descriptors.shape
                    gl_feat=torch.empty((self.n_neighbors, N, C_global + C_local), device=device)
                    gl_feat[:,:,:C_global]=global_feat.unsqueeze(1).expand(-1,N,-1)
                    gl_feat[:,:,C_global:]=descriptors.unsqueeze(0).expand(self.n_neighbors,-1,-1)
                    scene_coords = model({"features": gl_feat.reshape(self.n_neighbors*N,-1)})['sc']
                    keypoints = keypoints.float().cpu()

                scene_coords = scene_coords.float().cpu().numpy().reshape(self.n_neighbors, N, 3)
                gt_pose, intrinsics, frame_name= batch['pose'][0].numpy(), batch['intrinsics'][0].numpy(), batch['filename'][0]
                keypoints_np=keypoints.numpy()
                camera = {
                    'model': 'PINHOLE',
                    'width': image.shape[3],
                    'height': image.shape[2],
                    'params': intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]],
                }

                knn_results = []
                for neighbor_idx in range(self.n_neighbors):
                    knn_results.append(pool.apply_async(pose_estimate, args=(keypoints_np, scene_coords[neighbor_idx], camera, ransac_opt, gt_pose)))

                pool_results.append((frame_name, knn_results))

        for frame_name,knn_results in pool_results:
            knn_results=[res.get() for res in knn_results]
            result = max(knn_results, key=lambda x: x["num_inliers"])
            for key in ('pose_q', 'pose_t'):
                poses[key].append(result[key])
            for key in ('t_err', 'r_err', 'num_inliers'):
                metrics[key].append(result[key])

        pool.close()
        frame_names = [frame_name for frame_name, _ in pool_results]
        if self.base_name:
            frame_names = [os.path.basename(frame_name) for frame_name, _ in pool_results]
        poses_df=pd.DataFrame(np.concatenate([np.stack(poses[k]) for k in ("pose_q", "pose_t")], axis=1), columns=['q_w', 'q_x', 'q_y', 'q_z', 't_x', 't_y', 't_z'])
        poses_df["frame_name"] = frame_names
        poses_df.to_csv(pose_file, index=False, sep=' ', header=False, columns=['frame_name', 'q_w', 'q_x', 'q_y', 'q_z', 't_x', 't_y', 't_z'])
        metrics=pd.DataFrame(metrics)
        metrics["frame_name"] = frame_names
        metrics.to_csv(metric_file, index=False, sep=' ', header=False, columns=['frame_name', 't_err', 'r_err', 'num_inliers'])



        _logger.info("===================================================")
        _logger.info("Test complete.")
        _logger.info('Accuracy:')
        for t, r in acc_thresh[self.acc_thresh]:
            acc = (metrics["t_err"] < t) & (metrics["r_err"] < r)
            _logger.info(f"Accuracy: {t}m/{r}deg: {acc.mean() * 100:.1f}%")
        median_rErr = metrics['r_err'].median()
        median_tErr = metrics['t_err'].median() * 100
        _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")

        

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeKNNPose).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputeKNNPose)  # noqa
