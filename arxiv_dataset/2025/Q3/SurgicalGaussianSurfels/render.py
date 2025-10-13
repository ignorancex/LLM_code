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
import torch
import torchvision
import torchvision.utils
import numpy as np
from tqdm import tqdm
from scene import Scene, GaussianModel, DeformModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.image_utils import depth2rgb, normal2rgb
from os import makedirs
import open3d as o3d
from utils.initial_utils import process_depth_sequence_and_save


def render_set(model_path, load2gpu_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_color")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    rgb_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rgb_depth")
    gts_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")

    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
    normal_np_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_np")
    surf_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "surf_normals")
    surf_normal_np_path = os.path.join(model_path, name, "ours_{}".format(iteration), "surf_normal_np")

    depth_np_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_np")
    gts_depth_np_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth_np")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")

    combined = os.path.join(model_path, name, "ours_{}".format(iteration), "combined")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(rgb_depth_path, exist_ok=True)
    makedirs(gts_depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(normal_np_path, exist_ok=True)
    makedirs(surf_normal_path, exist_ok=True)
    makedirs(surf_normal_np_path, exist_ok=True)
    makedirs(depth_np_path, exist_ok=True)
    makedirs(gts_depth_np_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    makedirs(combined, exist_ok=True)

    # Variables for tracking performance
    total_rendering_time = 0
    peak_memory_list = []
    num_frames = len(views)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()

        # Start GPU timer
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)

        # Stop GPU timer
        end_event.record()
        torch.cuda.synchronize()  # Ensure timing is accurate
        frame_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        total_rendering_time += frame_time

        # Get peak memory used during rendering
        peak_memory = torch.cuda.max_memory_allocated()
        peak_memory_list.append(peak_memory)

        rendering = results["render"]
        depth_np = results["surf_depth"]
        depth = depth_np / (depth_np.max() + 1e-5)
        mask = view.mask

        alpha_np = results["rend_alpha"]
        mask_vis = (alpha_np.detach() > 1e-5)
        normals = results["rend_normal"]  # Extract normal maps
        normals_visual = (normals + 1) / 2  # Normalize normals to [0, 1]
        normal_wrt = normal2rgb(normals_visual, mask_vis)

        surf_normals = results['surf_normal']
        surf_normals_visual = (surf_normals + 1) / 2  # Normalize normals to [0, 1]

        rgb_depth = depth2rgb(depth_np, mask_vis)

        gt = view.original_image[0:3, :, :]
        if view.depth is not None:
            gts_depth_np = view.depth.unsqueeze(0).cpu().numpy()
            np.save(os.path.join(gts_depth_np_path, '{0:05d}'.format(idx) + ".npy"), gts_depth_np)

            gts_depth = view.depth.unsqueeze(0)
            gts_depth = gts_depth / (gts_depth.max() + 1e-5)
            torchvision.utils.save_image(gts_depth, os.path.join(gts_depth_path, '{0:05d}'.format(idx) + ".png"))

        img_wrt = torch.cat([gt, rendering, normals_visual * alpha_np, rgb_depth * alpha_np], 2)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rgb_depth, os.path.join(rgb_depth_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(normal_wrt, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mask, os.path.join(masks_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(surf_normals_visual,
                                     os.path.join(surf_normal_path, '{0:05d}'.format(idx) + ".png"))

        np.save(os.path.join(depth_np_path, '{0:05d}'.format(idx) + '.npy'), depth_np.cpu().numpy())
        np.save(os.path.join(normal_np_path, '{0:05d}'.format(idx) + ".npy"), normals.cpu().numpy())
        np.save(os.path.join(surf_normal_np_path, '{0:05d}'.format(idx) + ".npy"), surf_normals.cpu().numpy())

        torchvision.utils.save_image(img_wrt, os.path.join(combined, '{0:05d}'.format(idx) + ".png"))

    # Calculate average FPS
    FPS = num_frames / total_rendering_time
    # print(f"Average FPS: {FPS:.2f}")

    # Calculate average and maximum peak memory usage
    average_peak_memory = sum(peak_memory_list) / num_frames
    max_peak_memory = max(peak_memory_list)
    # print(f"Average Peak Memory Usage: {average_peak_memory / (1024 ** 2):.2f} MB")
    # print(f"Maximum Peak Memory Usage: {max_peak_memory / (1024 ** 2):.2f} MB")


def render_sets(dataset, iteration, pipeline, skip_train, skip_test, mode, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Create deformation model based on MLP flags
        deform = DeformModel(use_cutlass=args.cutlassMLP, use_fullyfused=args.fullyfusedMLP)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        else:
            render_func = render_set

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "train", scene.loaded_iter,
                         scene.getTrainCameras(), gaussians, pipeline,
                         background, deform)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "test", scene.loaded_iter,
                         scene.getTestCameras(), gaussians, pipeline,
                         background, deform)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--cutlassMLP", action="store_true", default=False, 
                       help="Use tiny-cuda-nn CutlassMLP for deformation network (faster training/inference)")
    parser.add_argument("--fullyfusedMLP", action="store_true", default=False, 
                       help="Use tiny-cuda-nn FullyFusedMLP for deformation network (fastest, 128 neurons)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Ensure only one MLP type is selected
    if args.cutlassMLP and args.fullyfusedMLP:
        raise ValueError("Cannot use both --cutlassMLP and --fullyfusedMLP at the same time.")

    # Print which MLP type is being used
    if args.fullyfusedMLP:
        print("[INFO] Using FullyFusedMLP for rendering")
    elif args.cutlassMLP:
        print("[INFO] Using CutlassMLP for rendering")
    else:
        print("[INFO] Using standard MLP for rendering")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args)
