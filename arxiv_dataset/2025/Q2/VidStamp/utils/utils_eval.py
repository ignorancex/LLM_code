import os
import torch
from torchvision import transforms
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, load_image
from utils.utils_finetuning import get_attacks
import utils.utils_img as utils_img
import utils.utils as utils
from utils.utils_Tamper_Localization import *

# Setup for Multi-GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

def get_bit_acc_I2V(frames_w, msg_decoder, keys, attacks):
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    
    # Convert to tensor if necessary
    if isinstance(frames_w, np.ndarray):
        frames_w = torch.tensor(frames_w, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to PyTorch tensor and reorder dimensions
    elif isinstance(frames_w, list):
        frames_w = torch.stack([transforms.ToTensor()(frame) for frame in frames_w], dim=0).to(device)

    log_stats = {}
    for name, attack in attacks.items():
        frames_aug = attack(frames_w.float())
        decoded = msg_decoder(vqgan_to_imnet(frames_aug.to(device))) # b c h w -> b k
        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
    
    return log_stats

def get_bit_acc_T2V(frames_w, msg_decoder, keys, attacks):
    log_stats = {}
    for name, attack in attacks.items():
        frames_aug = attack(frames_w.float())
        decoded = msg_decoder(frames_aug.to(device)) # b c h w -> b k
        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
    
    return log_stats

def load_keys_eval(args):
    with open(args.keys_file, "r") as file:
        data = json.load(file)
    keys_list = data["keys"]
    keys = torch.tensor([[int(bit) for bit in key] for key in keys_list], dtype=torch.uint8, device=device)
    keys_str =["".join([ str(int(ii)) for ii in keys.tolist()[j]]) for j in range(16)]
    print(f'Keys: {keys_str}')
    return keys

def load_pipeline(model_name, checkpoint_path):
    print(f"Loading model: {model_name}")
    
    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pipeline.vae.decoder.load_state_dict(checkpoint)
        print("Fine-tuned decoder loaded successfully.")
    else:
        print("Checkpoint not found. Using the default decoder.")
    
    pipeline.safety_checker = None
    return pipeline

def tamper_localization_evaluation(video_frames_w,msg_decoder, keys, args):
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    
    if args.swap_pairs and args.num_frames >= 2:
        indices = list(range(args.num_frames))
        random.shuffle(indices)
        swap_pairs = [(indices[i], indices[i+1]) for i in range(0, min(2*args.num_swaps, len(indices) - 1), 2)]
    else:
        swap_pairs = []

    drop_indices = random.sample(range(args.num_frames), args.num_drops) if args.drop_indices and args.num_frames >= 1 else []
    insertions = [
    (random.randint(0, args.num_frames - 1), torch.randint(0, 256, (3, args.img_size, args.img_size), dtype=torch.float32))
    for _ in range(min(args.num_inserts, args.num_frames))
    ] if args.insertions and args.num_frames >= 1 else []
    
    if isinstance(video_frames_w, np.ndarray):
        video_frames_w = torch.tensor(video_frames_w, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to PyTorch tensor and reorder dimensions
    elif isinstance(video_frames_w, list):
        video_frames_w = torch.stack([transforms.ToTensor()(frame) for frame in video_frames_w], dim=0).to(device)
    
    tampered_frames, frame_sequence = tamper_video(
        video_frames_w,
        swap_pairs=swap_pairs,
        drop_indices=drop_indices,
        insertions=insertions
    )

    decoded = msg_decoder(vqgan_to_imnet(tampered_frames.to(device)))
    tampered_keys = decoded > 0
    accuracy = temporal_tamper_localization(keys, tampered_keys, frame_sequence, threshold=args.temproal_tamper_threshold)

    log_stats = {}
    log_stats["Tamper Localization Accuracy"] = accuracy
    
    return log_stats

def Generate_and_Evaluate_I2V(pipeline, keys, msg_decoder, args):
    output_vids_dir = os.path.join(args.output_folder, str(args.model_abbreviation) + "_generated_videos_tamper_k" + str(args.length_key_segments))
    log_dir = os.path.join(args.output_folder, 'logs')
    if args.temproal_tamper:
        log_file = 'log_' + args.model_abbreviation + "_tamper_localization_"+'.txt'
    else:
        log_file = 'log_' + args.model_abbreviation +'.txt'
    
    os.makedirs(output_vids_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    metric_logger = utils.MetricLogger(delimiter="  ")

    for idx, img_name in enumerate(os.listdir(args.input_folder)):
            output_video_path = os.path.join(output_vids_dir, f"{os.path.splitext(img_name)[0]}.mp4")
        #if idx > 157:
            if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(args.input_folder, img_name)
                image = load_image(img_path)
                
                with torch.no_grad():
                    video_frames_w = pipeline(image=image, height=args.img_size, width=args.img_size, 
                                            num_frames=args.num_frames, decode_chunk_size=16, output_type='np',
                                            num_inference_steps=25).frames[0]
                
                if not args.temproal_tamper:
                    log_stats = get_bit_acc_I2V(video_frames_w, msg_decoder, keys, get_attacks()) 
                    export_to_video(video_frames_w, output_video_path=output_video_path)
                else:
                    log_stats = tamper_localization_evaluation(video_frames_w, msg_decoder, keys, args)

                for name, loss in log_stats.items():
                    metric_logger.update(**{name:loss})

                print(log_stats)

    print("Averaged {} stats:".format('eval'), metric_logger)
    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_stats = {
            **{f'val_{k}': v for k, v in val_stats.items()},
        }
    log_stats['swap'] = args.swap_pairs
    log_stats['drop'] = args.drop_indices
    log_stats['insert'] = args.insertions
    log_stats['threshold'] = args.temproal_tamper_threshold
    log_stats["num_swaps"] = args.num_swaps, 
    log_stats["num_drops"] = args.num_drops, 
    log_stats["num_inserts"] = args.num_inserts
    
    with (Path(log_dir) / log_file).open("a") as f:
        f.write(json.dumps(log_stats) + "\n")

def Generate_and_Evaluate_T2V(pipeline, keys, msg_decoder, args):
    output_vids_dir = os.path.join(args.output_folder, str(args.model_abbreviation) + "_generated_videos")
    log_dir = os.path.join(args.output_folder, 'logs')
    log_file = 'log_' + args.model_abbreviation +'.txt'
    os.makedirs(output_vids_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    metric_logger = utils.MetricLogger(delimiter="  ")

    with open(args.input_prompts, "r") as f:
        all_prompts = f.readlines()

    for prompt in enumerate(all_prompts):    
        with torch.no_grad():
            latents = pipeline(prompt=prompt[1], num_frames=args.num_frames, 
                       num_inference_steps=50, 
                       height=args.img_size, width=args.img_size, 
                       output_type='latent').frames[0]

            latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4)
            imgs_w = pipeline.vae.decode(latents).sample
            video_frames_w = imgs_w.squeeze(0).permute(1, 0, 2, 3) 
            #video_frames_w = pipeline(prompt=prompt[1], num_frames=args.num_frames, num_inference_steps=50, height=args.img_size, width=args.img_size,
                                     #output_type='pt').frames[0]
                            
        # Get Bit Accuracy
        log_stats = get_bit_acc(video_frames_w, msg_decoder, keys, get_attacks())
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})

        print(log_stats)

        # Save Generated Video
        output_video_path = os.path.join(output_vids_dir, str(prompt[0])+ ".mp4")
        export_to_video(video_frames_w, output_video_path=output_video_path)

    print("Averaged {} stats:".format('eval'), metric_logger)
    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_stats = {
            **{f'val_{k}': v for k, v in val_stats.items()},
        }
    with (Path(log_dir) / log_file).open("a") as f:
        f.write(json.dumps(log_stats) + "\n")

def Generate_and_Evaluate(pipeline, keys, msg_decoder, args):
    if args.model_abbreviation == "SVD":
        Generate_and_Evaluate_I2V(pipeline, keys, msg_decoder, args)
    else:
        Generate_and_Evaluate_T2V(pipeline, keys, msg_decoder, args)

