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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
from pathlib import Path
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from utils.loss_utils import l1_loss
from gaussian_renderer import render
import sys, yaml
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

def get_tensor_values(tensor, p,  mode='nearest',
                      scale=True, detach=True, detach_p=True, align_corners=False):
    '''
        MIT License

        Copyright (c) 2022 Wenjing Bian

        Reference: https://github.com/ActiveVisionLab/nope-nerf
    ''' 
    
    _, _, h, w = tensor.shape
    
    if detach_p:
        p = p.detach()
    if scale:
        p[:, :, 0] = 2.*p[:, :, 0]/w - 1
        p[:, :, 1] = 2.*p[:, :, 1]/h - 1
    p = p.unsqueeze(1)
    values = torch.nn.functional.grid_sample(tensor, p, mode=mode, align_corners=align_corners)
    values = values.squeeze(2)

    if detach:
        values = values.detach()
    values = values.permute(0, 2, 1)

    return values


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    dataset.model_path = os.path.join(dataset.output, dataset.g)
    os.makedirs(dataset.model_path, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    outdir = os.path.join(dataset.model_path, "output")
    os.makedirs(outdir, exist_ok=True)
    start_time = time.time()
    print(dataset.model_path)
    count = 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack[0]
        count -= 1
        if count == -1:
            viewpoint_stack.pop(0)
            count = 1
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg["rend_depth"][0]
        uncertain = render_pkg["rend_uncertainty"]
        gt_image = viewpoint_cam.original_image
        uv = viewpoint_cam.uv.clone()
        maxv = uv[:, 1].max()
        getdepth = get_tensor_values(
            depth.unsqueeze(0).unsqueeze(0), 
            uv[None, ...], 
            mode="bilinear", 
            scale=True, 
            detach=False, detach_p=False, align_corners=True).squeeze()
        
        getuncertain = get_tensor_values(
            uncertain.unsqueeze(0).unsqueeze(0), 
            uv[None, ...], 
            mode="bilinear", 
            scale=False, 
            detach=False, detach_p=False, align_corners=True).squeeze()
        
        Ll1 = l1_loss(getdepth, gt_image)

        # ensure unreached rays are empty
        lowhalf = torch.ones(viewpoint_cam.image_height, dtype=torch.bool, device=uv.device)
        lowhalf[:maxv.long()] = False
        op_loss = depth[lowhalf].mean()

        if iteration < 1000:
            loss = Ll1 + op_loss
        else:
            uncertain_l1 = (getdepth - gt_image).abs().detach()
            uncertain_loss = l1_loss(getuncertain, uncertain_l1)
            loss = Ll1 + op_loss + uncertain_loss
        
        surf_normal = render_pkg['surf_normal']
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 100 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 100 else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss  
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{loss.item():.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration % 1000==0:
                    gaussians.prune()
                    gaussians.densify(gt_image, getdepth, visibility_filter, viewpoint_cam)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration > opt.iterations - 5):
                idx = viewpoint_cam.uid
                plt.imsave(f"{outdir}/normal{idx}-3d.png", (surf_normal.permute(1, 2, 0).detach().cpu().numpy()[::-1]+1)/2)
                plt.imsave(f"{outdir}/depth{idx}-3d.png", (depth.clamp_max(50)/50).detach().cpu().numpy()[::-1])
                gt_depth = torch.zeros_like(depth)
                dh, dw = gt_depth.shape[:2]
                gt_depth[(uv[:, 1]*75+75).long(), (uv[:, 0]*300+300).long()] = gt_image.float()
                plt.imsave(f"{outdir}/depth_gt{idx}-3d.png", (gt_depth.clamp_max(50)/50).detach().cpu().numpy()[::-1])
                plt.imsave(f"{outdir}/uncertain{idx}-3d.png", uncertain.clamp_max(1).detach().cpu().numpy()[::-1])
    end_time = time.time()
    with open("times.txt", "a") as f:
        f.write(f"{end_time - start_time}\n")
    torch.save((gaussians.capture(), iteration), os.path.join(dataset.model_path, "2dgs-test.pth"))

def parse(file_name):
    assert os.path.exists(file_name), "option file does not exist"
    opt = yaml.safe_load(Path(file_name).read_text())
    return edict(opt)

if __name__ == "__main__":
    # Set up command line argument parser
    opt = parse("option.yml")
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, opt)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")