#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import torch
from tqdm import tqdm
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from scene.envlight import EnvLight
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, calculate_mean_and_std
from scene.waymo_loader import readWaymoInfo
from scene.kittimot_loader import readKittiMotInfo
from scene.emer_waymo_loader import readEmerWaymoInfo
import logging
sceneLoadTypeCallbacks = {
    "Waymo": readWaymoInfo,
    "KittiMot": readKittiMotInfo,
    'EmerWaymo': readEmerWaymoInfo,
}

class Scene:

    gaussians : GaussianModel

    def __init__(self, args, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            logging.info("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = sceneLoadTypeCallbacks[args.scene_type](args)
        
        self.time_interval = args.frame_interval
        self.gaussians.time_duration = scene_info.time_duration
        # print("time duration: ", scene_info.time_duration)
        # print("frame interval: ", self.time_interval)


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.resolution_scales = args.resolution_scales
        self.scale_index = len(self.resolution_scales) - 1
        for resolution_scale in self.resolution_scales:
            logging.info("Loading Training Cameras at resolution scale {}".format(resolution_scale))
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            logging.info("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            logging.info("Computing nearest_id")
            image_name_to_id = {"train": {cam.image_name: id for id, cam in enumerate(self.train_cameras[resolution_scale])}, 
                    "test": {cam.image_name: id for id, cam in enumerate(self.test_cameras[resolution_scale])}}
            with open(os.path.join(self.model_path, "multi_view.json"), 'w') as file:
                json_d = []
                for id, cur_cam in enumerate(tqdm(self.train_cameras[resolution_scale], bar_format='{l_bar}{bar:50}{r_bar}')):
                    image_name_to_id_map = image_name_to_id["train"]
                    # cur_image_name = cur_cam.image_name
                    cur_colmap_id = cur_cam.colmap_id
                    nearest_colmap_id_candidate = [cur_colmap_id - 10, cur_colmap_id + 10, cur_colmap_id - 20, cur_colmap_id + 20]
                    
                    for colmap_id in nearest_colmap_id_candidate:
                        near_image_name = "{:03d}_{:1d}".format(colmap_id // 10, colmap_id % 10)
                        if near_image_name in image_name_to_id_map:
                            cur_cam.nearest_id.append(image_name_to_id_map[near_image_name])
                            cur_cam.nearest_names.append(near_image_name)
                    
                    json_d.append({'ref_name' : cur_cam.image_name, 'nearest_name': cur_cam.nearest_names, "id": id, 'nearest_id': cur_cam.nearest_id})
                json.dump(json_d, file)
              
            
            if resolution_scale == 1.0:
                logging.info("Computing mean and std of dataset")
                mean = []
                std = []
                all_cameras = self.train_cameras[resolution_scale] + self.test_cameras[resolution_scale]
                for idx, viewpoint in enumerate(tqdm(all_cameras, bar_format="{l_bar}{bar:50}{r_bar}")):
                    gt_image = viewpoint.original_image # [3, H, W]  
                    mean.append(gt_image.mean(dim=[1, 2]).cpu().numpy())
                    std.append(gt_image.std(dim=[1, 2]).cpu().numpy())
                mean = np.array(mean)
                std = np.array(std)
                # calculate mean and std of dataset
                mean_dataset, std_rgb_dataset = calculate_mean_and_std(mean, std)

                if gaussians.uncertainty_model is not None:
                    gaussians.uncertainty_model.img_norm_mean = torch.from_numpy(mean_dataset)
                    gaussians.uncertainty_model.img_norm_std = torch.from_numpy(std_rgb_dataset)
                    
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, 1)

    def upScale(self):
        self.scale_index = max(0, self.scale_index - 1)

    def getTrainCameras(self):
        return self.train_cameras[self.resolution_scales[self.scale_index]]
    
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

