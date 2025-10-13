import cv2
from PIL import Image
import numpy as np
import importlib
import os
import glob
import argparse
from tqdm import tqdm
import torch
import gc
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model
from core.utils import to_tensors
from utils.image_ops import *

parser = argparse.ArgumentParser(description="Benchmark")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-m", "--mask_name", type=str, required=True)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("--model", type=str, default='e2fgvi_hq')
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=6)
parser.add_argument("--neighbor_stride", type=int, default=4)
parser.add_argument("--savefps", type=int, default=24)

# args for e2fgvi_hq (which can handle videos with arbitrary resolution)
parser.add_argument("--set_size", action='store_true', default=False)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)

args = parser.parse_args()

ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size = None
if args.model == "e2fgvi":
    size = (432, 240)
elif args.set_size:
    size = (args.width, args.height)    

net = importlib.import_module('model.' + args.model)
model = net.InpaintGenerator().to(device)
data = torch.load(args.ckpt, map_location=device)
model.load_state_dict(data)
print(f'Loading model from: {args.ckpt}')
model.eval()

i3d_model = init_i3d_model() # for VFID 

def infer_video_single_loop(imgs, frames, masks, size, desc):
    video_length = len(frames)
    h, w = size[1], size[0]
    binary_masks = [
        (np.transpose(np.array(m), (1, 2, 0)) != 0).astype(np.uint8) for m in list(masks[0])
    ]
    comp_frames = [None] * video_length

    for f in tqdm(range(0, video_length, neighbor_stride), desc=desc):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, num_ref, ref_length)
        selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
        
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks) 

            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            pred_imgs, _ = model(masked_imgs.to(device), len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w].cpu()
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.permute(0, 2, 3, 1).numpy() * 255
            
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_imgs[i]).astype(
                    np.uint8) * binary_masks[idx] + frames[idx] * (
                        1 - binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32) * 0.5 + img.astype(np.float32) * 0.5

        del masked_imgs, selected_imgs, selected_masks, pred_imgs
        torch.cuda.empty_cache()
        gc.collect()
    return comp_frames

def infer_video(video_idx, frames, mask_name, total_videos, save_video=True):
    video_length = len(frames)
    frames, size = resize_frames(frames, None)
    h, w = size[1], size[0]
    imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
    frames = [np.array(f).astype(np.uint8) for f in frames]
    comp_frames = None
    masks = read_mask(os.path.join(args.video, mask_name), size, video_length)
    masks = to_tensors()(masks).unsqueeze(0)

    desc = f'V {video_idx}/{total_videos}'
    comp_frames = infer_video_single_loop(imgs, frames, masks, size, desc)

    psnr, ssim = 0.0, 0.0
    frames_PIL, comp_PIL = [], [] # to calculate VFID 
    for i in range(video_length):
        frame_psnr, frame_ssim = calc_psnr_and_ssim(frames[i], comp_frames[i])
        psnr += frame_psnr
        ssim += frame_ssim
        frames_PIL.append(Image.fromarray(frames[i].astype(np.uint8)))
        comp_PIL.append(Image.fromarray(comp_frames[i].astype(np.uint8)))
    psnr, ssim = psnr / video_length, ssim / video_length 
    frames_i3d, comp_i3d = calculate_i3d_activations(frames_PIL, comp_PIL, i3d_model, device=device)

    if save_video:
        save_dir_name = os.path.join(args.video, 'results')
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        save_path = os.path.join(save_dir_name, f'{video_idx}.mp4')
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                    default_fps, size)
        for f in range(video_length):
            comp = comp_frames[f].astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
        writer.release()
    
    return psnr, ssim, frames_i3d, comp_i3d

# prepare datset
if not os.path.exists(args.video):
    raise FileNotFoundError(f"{args.video} doesn't exist")
args.use_mp4 = True 
print(
    f'Loading videos and masks from: {args.video} | INPUT MP4 format: {args.use_mp4}'
)
video_paths = glob.glob(os.path.join(args.video, 'video', '*.mp4'))
videos = [read_frames_from_video(vpath) for vpath in video_paths]
total_videos = len(videos)

all_psnr, all_ssim, real_i3d_activations, output_i3d_activations = 0.0, 0.0, [], []
for idx, video in enumerate(videos):
    psnr, ssim, frames_i3d, comp_i3d = infer_video(idx+1, video, args.mask_name, total_videos, True)
    real_i3d_activations.append(frames_i3d)
    output_i3d_activations.append(comp_i3d)
    all_psnr += psnr
    all_ssim += ssim
    print('Video PSNR/SSIM: 'f'{psnr:.2f}/{ssim:.4f}/')
avg_psnr = all_psnr / len(videos)
avg_ssim = all_ssim / len(videos)   
fid_score = calculate_vfid(real_i3d_activations, output_i3d_activations)
print('Finish evaluation... Average PSNR/SSIM/VFID: '
          f'{avg_psnr:.2f}/{avg_ssim:.4f}/{fid_score:.3f}')