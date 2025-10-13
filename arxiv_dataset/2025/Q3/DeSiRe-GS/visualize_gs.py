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
import glob
import os
import torch

from scene import GaussianModel, EnvLight
from utils.general_utils import seed_everything
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

EPS = 1e-5



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config_path", type=str, required=True)
    params, _ = parser.parse_known_args()
    
    args = OmegaConf.load(params.config_path)
    args.resolution_scales = args.resolution_scales[:1]
    # print('Configurations:\n {}'.format(pformat(OmegaConf.to_container(args, resolve=True, throw_on_missing=True))))
    # convert to DictConfig
    conf = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**conf)))

    seed_everything(args.seed)

    sep_path = os.path.join(args.model_path, 'point_cloud')
    os.makedirs(sep_path, exist_ok=True)
    
    gaussians = GaussianModel(args)
    
    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    print(f"Loading checkpoint {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args)
    
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
    print("Number of Gaussians: ", gaussians.get_xyz.shape[0])   
    
    
    save_timestamp = 0.0
    gaussians.save_ply_at_t(os.path.join(args.model_path, "point_cloud", "iteration_{}".format(first_iter), "point_cloud.ply"), save_timestamp)
    

