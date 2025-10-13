import torch
import torch.optim as optim
from torch import nn
import json
import copy
from torchvision.utils import save_image
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from torchvision import transforms
from pathlib import Path
from utils.utils import *
import utils.utils_img as utils_img
import os
import sys
sys.path.append('src')
from loss.loss_provider import LossProvider

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline_finetune(model_name, checkpoint_path):
    print(f"Loading model: {model_name}")
    
    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    pipeline.safety_checker = None
    
    model = copy.deepcopy(pipeline.vae)
    for param in model.decoder.parameters():
        param.requires_grad = True
    
    return pipeline, model

def generate_keys(args, device="cuda"):
    print(f'\n>>> Creating {args.num_frames // args.length_key_segments} keys with {args.num_bits} bits...')
    
    num_unique_keys = args.num_frames // args.length_key_segments
    if args.num_frames % args.length_key_segments != 0:
        num_unique_keys += 1  # Account for remainder frames

    # Generate unique keys
    unique_keys = torch.randint(0, 2, (num_unique_keys, args.num_bits), dtype=torch.float32, device=device)

    # Repeat each key 'k' times and trim to required number of frames
    keys = unique_keys.repeat_interleave(args.length_key_segments, dim=0)[:args.num_frames]

    keys_str = ["".join([str(int(ii)) for ii in keys.tolist()[j]]) for j in range(args.num_frames)]
    print(f'Keys: {keys_str}')
    
    log_stats = {'keys': keys_str}
    with (Path(args.keys_dir) / args.keys_file).open("w") as f:
        f.write(json.dumps(log_stats) + "\n")

    with (Path(args.log_dir) / args.log_file).open("a") as f:
        f.write(json.dumps(log_stats) + "\n")

    return keys

def load_keys(args):
    with open(args.keys, "r") as file:
        data = json.load(file)
    keys_list = data["keys"]
    keys = torch.tensor([[int(bit) for bit in key] for key in keys_list], dtype=torch.float32, device=device)
    keys_str =["".join([ str(int(ii)) for ii in keys.tolist()[j]]) for j in range(16)]
    print(f'Keys: {keys_str}')
    return keys

def get_keys(args):
    if args.finetuning_stage == "first":
        return generate_keys(args)
    elif args.finetuning_stage == "second":
        return load_keys(args)

def get_losses(args):
    print(f'>>> Creating losses...')
    print(f'Losses: {args.loss_w} and {args.loss_i}...')
    if args.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif args.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    
    if args.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    elif args.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif args.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif args.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    else:
        raise NotImplementedError
    
    return loss_i, loss_w

def train_decoder(model, original_vae, msg_decoder, dataloader, keys, args):
    print(f"Starting fine-tuning of {args.model_abbreviation} decoder...")

    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])

    # Set model to train mode
    model.decoder.train()

    original_vae.eval()

    # Define optimizer (AdamW)
    optimizer = optim.AdamW(model.decoder.parameters(), lr=args.lr)

    # Define loss functions
    loss_i, loss_w = get_losses(args)

    metric_logger = MetricLogger(delimiter="  ")
    base_lr = optimizer.param_groups[0]["lr"]
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        for ii, frames in enumerate(metric_logger.log_every(dataloader, args.log_freq, "First Stage Finetuning")):
            frames = frames.to(device)
            
            if args.finetuning_stage == "second" and args.model_abbreviation == "SVD":
                frames = frames.squeeze(0)
            
            if args.model_abbreviation == "CVX":
                frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)   
            
            adjust_learning_rate(optimizer, ii, args.num_epochs * args.train_steps, args.warmup_steps, base_lr)

            # Encode images into latent space
            latents = original_vae.encode(frames).latent_dist.sample()

            # Decode latents with original and fine-tuned decoder
            if args.model_abbreviation == "SVD":
                imgs_d0 = original_vae.decode(latents, num_frames=args.num_frames).sample 
                imgs_w = model.decode(latents, num_frames=args.num_frames).sample
                imgs_w_perm, imgs_d0_perm, frames_perm = imgs_w, imgs_d0, frames
            else:
                imgs_d0 = original_vae.decode(latents).sample
                imgs_w = model.decode(latents).sample
                imgs_w_perm = imgs_w.squeeze(0).permute(1, 0, 2, 3)  # (B, T, C, H, W)
                imgs_d0_perm = imgs_d0.squeeze(0).permute(1, 0, 2, 3)
                frames_perm = frames.permute(0, 2, 1, 3, 4).squeeze(0)
            
            if args.model_abbreviation == "SVD":
                decoded = msg_decoder(vqgan_to_imnet(imgs_w_perm))
            else:
                decoded = msg_decoder(imgs_w_perm)

            # Computer Loss
            lossw = loss_w(decoded, keys)
            lossi = loss_i(imgs_w_perm, imgs_d0_perm)
            loss = args.lambda_w * lossw + args.lambda_i * lossi

            # Optim step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log stats
            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_accs = (bit_accs == 1) # b
            log_stats = {
                "iteration": ii,
                "loss": loss.item(),
                "loss_w": lossw.item(),
                "loss_i": lossi.item(),
                "bit_acc_avg": torch.mean(bit_accs).item(),
                "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            for name, loss in log_stats.items():
                metric_logger.update(**{name:loss})
            if ii % args.log_freq == 0:
                print(json.dumps(log_stats))

            if ii % args.save_img_freq == 0:
                save_image(torch.clamp(utils_img.unnormalize_vqgan(frames_perm),0,1), os.path.join(args.sample_imgs_dir, f'{ii:03}_train_orig.png'), nrow=8)
                save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0_perm),0,1), os.path.join(args.sample_imgs_dir, f'{ii:03}_train_d0.png'), nrow=8)
                save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w_perm),0,1), os.path.join(args.sample_imgs_dir, f'{ii:03}_train_w.png'), nrow=8)
    
    # Save the checkpoint
    checkpoint_name = args.model_abbreviation + '_' + args.finetuning_stage + '_stage_' + str(args.num_bits) + '_bit_' + str(args.length_key_segments) + '_k.pth' 
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    torch.save(model.decoder.state_dict(), checkpoint_path)
    print(f"Fine-tuned decoder saved to {args.checkpoint_dir}")

    # Print and save training info
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
            }
    with (Path(args.log_dir) / args.log_file).open("a") as f:
        f.write(json.dumps(log_stats) + "\n")

def get_attacks():
    attacks = {
        'none': lambda x: x, 
        'resize_03': lambda x: utils_img.resize(x, 0.3),
        'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        'crop_03': lambda x: utils_img.center_crop(x, 0.3),
        'rot_25': lambda x: utils_img.rotate(x, 25),
        'rot_90': lambda x: utils_img.rotate(x, 90),
        'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
        'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
        'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
        'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
        'gaussian_blur': lambda x: utils_img.gaussian_blur(x, 4),
        'gaussian_noise': lambda x: utils_img.add_noise(x, 0.1),
        'MPEG4': lambda x: utils_img.encode_mpeg4(x),
    }
    return attacks

def val_decoder(model, original_vae, msg_decoder, dataloader, keys, args):
    print(f"Evaluation of {args.model_abbreviation} decoder...")

    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])

    # Set model to eval mode
    model.decoder.eval()
    original_vae.eval()

    metric_logger = MetricLogger(delimiter="  ")
    attacks = get_attacks()

    with torch.no_grad():
        for ii, frames in enumerate(metric_logger.log_every(dataloader, args.log_freq, "Evaluation")):
            frames = frames.to(device)
            if args.finetuning_stage == "second":
                frames = frames.squeeze(0)
            
            if args.model_abbreviation == "CVX":
                frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)   
            
            # Encode images into latent space
            latents = original_vae.encode(frames).latent_dist.sample()

            # Decode latents with original and fine-tuned decoder
            if args.model_abbreviation == "SVD":
                imgs_d0 = original_vae.decode(latents, num_frames=args.num_frames).sample 
                imgs_w = model.decode(latents, num_frames=args.num_frames).sample
                imgs_w_perm, imgs_d0_perm, frames_perm = imgs_w, imgs_d0, frames
            else:
                imgs_d0 = original_vae.decode(latents).sample
                imgs_w = model.decode(latents).sample
                imgs_w_perm = imgs_w.squeeze(0).permute(1, 0, 2, 3)  # (B, T, C, H, W)
                imgs_d0_perm = imgs_d0.squeeze(0).permute(1, 0, 2, 3)
                frames_perm = frames.permute(0, 2, 1, 3, 4).squeeze(0)
            
            log_stats = {"iteration": ii}
            for name, attack in attacks.items():
                imgs_aug = attack(vqgan_to_imnet(imgs_w_perm))
                decoded = msg_decoder(imgs_aug) # b c h w -> b k
                diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
                bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                word_accs = (bit_accs == 1) # b
                log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
                log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
            for name, loss in log_stats.items():
                metric_logger.update(**{name:loss})

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_stats = {
                **{f'val_{k}': v for k, v in val_stats.items()},
            }
    with (Path(args.log_dir) / args.log_file).open("a") as f:
        f.write(json.dumps(log_stats) + "\n")