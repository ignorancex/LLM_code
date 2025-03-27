import os
import torch
import numpy as np
import argparse
import glob
from monai.metrics import DiceHelper, compute_surface_dice

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Video Predictor")
    parser.add_argument('--base_video_dir', type=str,  help="Base directory containing all video directories", required=True)
    parser.add_argument('--prefix', type=str, help="Prefix for video directories", default="BraTS20_Training_")
    return parser.parse_args()

# Use bfloat16 for the entire notebook
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    if torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    # Parse arguments
    args = parse_args()
    base_video_dir = args.base_video_dir
    prefix = args.prefix

    # Collect all video directories
    search_pattern = os.path.join(base_video_dir, f"{prefix}*")
    video_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)] 

    # Initialize metrics lists
    dice_scores = []
    nsd_scores = []

    for video_dir in video_dirs:
        ann_obj_id = 1
        # Scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        print(f"Processing {video_dir}, number of frames: {len(frame_names)}")

        if len(frame_names) == 0:
            print(f"No frames found in {video_dir}, skipping...")
            continue
        
        frame_idx = len(frame_names) // 2
        
        # from the beginning to the end
        inference_state = predictor.init_state(video_path=video_dir)

        out_mask_logits = np.load(os.path.join(video_dir, frame_names[frame_idx].replace(".jpg", ".npy")))

        predictor.reset_state(inference_state)
        predictor.add_new_mask(inference_state=inference_state, frame_idx=frame_idx, obj_id=ann_obj_id, mask=out_mask_logits > 0.0)

        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=frame_idx-1, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        video_seg_3d = np.stack([video_segments[k][1] for k in video_segments])
        gt_3d = np.stack([np.load(os.path.join(video_dir, frame_names[k].replace(".jpg", ".npy")))[None] for k in video_segments])
        print(video_seg_3d.shape, gt_3d.shape)
        n_classes, batch_size = 1, 1
        spatial_shape = (video_seg_3d.shape[0], video_seg_3d.shape[2], video_seg_3d.shape[3]) 

        y_pred = torch.tensor(video_seg_3d).float().reshape(batch_size, n_classes, *spatial_shape)  # prediction
        y = torch.tensor(gt_3d).float().reshape(batch_size, n_classes, *spatial_shape)  # ground truth

        score, not_nans = DiceHelper(include_background=False, sigmoid=True, softmax=True)(y_pred, y)
        dice = score.item()
        nsd = compute_surface_dice(y_pred, y, class_thresholds=[1]).item()
        dice_scores.append(dice)
        nsd_scores.append(nsd)
        print(f"Dice: {dice:.4f}, nsd: {nsd:.4f}")

        # Calculate and print current average metrics
        current_average_dice = np.mean(dice_scores)
        current_average_nsd = np.mean(nsd_scores)
        print(f"Current Average Dice: {current_average_dice:.4f}, Current Average nsd: {current_average_nsd:.4f}")

    # Calculate final average metrics
    final_average_dice = np.mean(dice_scores)
    final_average_nsd = np.mean(nsd_scores)

    print(f"Final Average Dice: {final_average_dice:.4f}, Final Average nsd: {final_average_nsd:.4f}")
