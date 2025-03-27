import os
import torch
import numpy as np
import argparse
import glob
from monai.metrics import DiceHelper, compute_surface_dice
from scipy import ndimage


def find_largest_enclosed_point(img, disturb=False):   
    # written by chatgpt4, and modified by yun
    img = np.pad(img, 1, mode='constant')

    # Mark the connected area and divide all the foreground areas in the image into each connected area.
    label, num_labels = ndimage.label(img)
    max_distance = 1.42
    max_coordinate = (0, 0)

    for i in range(1, num_labels+1):
        # Set the current area to 1 and other areas to 0.
        region = (label == i).astype(int)
        distance_transform = ndimage.distance_transform_edt(region)
        
        # Find the maximum distance field value
        dist_max = np.max(distance_transform)
        
        # If the maximum distance field in the current region is greater than the existing maximum, the maximum value and the corresponding coordinates will be updated.
        if dist_max > max_distance:
            max_distance = dist_max
            max_coordinate = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)
    
    if max_distance == 1.42:
        return max_distance, (None, None)
    # Return the coordinates of the maximum distance field value, pay attention to subtract the filled 1, and restore the original size.
    if not disturb:
        return max_distance, (max_coordinate[1] - 1, max_coordinate[0] - 1)
    else:
        retry = 5
        while retry > 0:
            retry -= 1
            disturb_x = np.random.randint(-5, 5)
            disturb_y = np.random.randint(-5, 5)
            if region[max_coordinate[0] + disturb_y, max_coordinate[1] + disturb_x] == 1:
                return max_distance, (max_coordinate[1] + disturb_x - 1, max_coordinate[0] + disturb_y - 1)
        return max_distance, (max_coordinate[1] - 1, max_coordinate[0] - 1)


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Video Predictor")
    parser.add_argument('--base_video_dir', type=str,  help="Base directory containing all video directories", default="/mnt/data/shenchuyun/sam2_som/videos/Brats2020")
    parser.add_argument('--prefix', type=str, help="Prefix for video directories", default="BraTS20_Training_")
    parser.add_argument('--steps', type=int, help="Number of steps to run the algorithm", default=5)
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

        # Initial mask is full 0
        points = np.array([], dtype=np.float32)
        labels = np.array([], np.int32)
        gt = np.load(os.path.join(video_dir, frame_names[frame_idx].replace(".jpg", ".npy")))
        if gt.sum() <= 100:
            print("GT is too small, skipping...")
            continue
        out_mask = np.zeros_like(gt)

        # Define a function for an iteration.
        def perform_iteration():
            global points, labels, out_mask

            FN = np.logical_and(gt, np.logical_not(out_mask))
            FP = np.logical_and(np.logical_not(gt), out_mask)
            assert len(FN.shape) == 2, len(FN.shape) == 2 

            max_dis_1, col_1 = find_largest_enclosed_point(FN, disturb=False)
            max_dis_2, col_2 = find_largest_enclosed_point(FP, disturb=False)
            
            
            if max_dis_1 >= max_dis_2:
                col = col_1
                input_label = 1
            else:
                col = col_2
                input_label = 0
            assert len(col) == 2

            if col[0] is None:
                return
            else:
                if points.size == 0:
                    points = np.array([col], dtype=np.int32)
                else:
                    points = np.append(points, [col], axis=0).astype(np.int32)
                labels = np.append(labels, input_label).astype(np.int32)
                
                _, _, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
            out_mask = (out_mask_logits.cpu() > 0.0)[0][0]

        # loop 5 times
        for _ in range(args.steps):
            perform_iteration()

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
        # print(f"Dice: {dice:.4f}")

        # Calculate and print current average metrics
        current_average_dice = np.mean(dice_scores)
        current_average_nsd = np.mean(nsd_scores)
        print(f"Current Average Dice: {current_average_dice:.4f}, Current Average nsd: {current_average_nsd:.4f}")
        # print(f"Current Average Dice: {current_average_dice:.4f}")

    # Calculate final average metrics
    final_average_dice = np.mean(dice_scores)
    final_average_nsd = np.mean(nsd_scores)

    print(f"Final Average Dice: {final_average_dice:.4f}, Final Average nsd: {final_average_nsd:.4f}")
    # print(f"Final Average Dice: {final_average_dice:.4f}")
