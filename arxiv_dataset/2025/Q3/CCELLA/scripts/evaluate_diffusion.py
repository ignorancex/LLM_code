# Script based on MONAI realism_diversity_metrics.py and MAISI diff_model_infer.py & infer_controlnet.py
# Sources:
# https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/realism_diversity_metrics/realism_diversity_metrics.py
# https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi


from tqdm import tqdm
from monai.metrics.fid import FIDMetric
import argparse
import json
import logging
import os
import sys
from datetime import datetime

from monai.inferers import sliding_window_inference
import numpy as np
import torch
import torch.distributed as dist
from monai.data import decollate_batch, MetaTensor
from monai.networks.utils import copy_model_state
from monai.transforms import SaveImage, MapTransform
from monai.utils import RankFilter
import pickle
import random
import nibabel as nib

from .utils import define_instance, prepare_maisi_controlnet_json_dataloader_path_eval, prepare_maisi_controlnet_json_dataloader_base_eval, setup_ddp, prepare_maisi_controlnet_json_dataloader_text_eval, ReconModelRaw
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from monai.utils import set_determinism

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from typing import Union
    
def set_random_seed(seed: int) -> int:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.

    Returns:
        int: Set random seed.
    """
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple:
    """
    Load the autoencoder and UNet models.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load models on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Loaded autoencoder, UNet model, and scale factor.
    """
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception as e:
        logger.error(e)
        logger.error("The trained_autoencoder_path does not exist!")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    if not args.modelname_override:
        model_filepath = f"{args.model_dir}/{args.model_filename}"
    else:
        model_filepath = f"{args.model_dir}/{args.modelname_override}"
    logger.info(f"Loading from {model_filepath}.")
    checkpoint = torch.load(model_filepath, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
    logger.info(f"checkpoints {model_filepath} loaded.")

    scale_factor = checkpoint["scale_factor"]
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, scale_factor


def run_inference_base(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: float,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        top_region_index_tensor (torch.Tensor): Top region index tensor.
        bottom_region_index_tensor (torch.Tensor): Bottom region index tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    noise = torch.randn(
        (
            spacing_tensor.shape[0],
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}, {noise.shape}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    recon_model = ReconModelRaw(autoencoder=autoencoder).to(device)
    autoencoder.eval()
    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        for t in tqdm(noise_scheduler.timesteps, ncols=110):
            timestep_preshape = [t for _ in range(spacing_tensor.shape[0])]
            model_output = unet(
                x=image,
                timesteps=torch.Tensor(timestep_preshape).to(device),
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                spacing_tensor=spacing_tensor,
            )
            image, _ = noise_scheduler.step(model_output, t, image)

        synthetic_images = recon_model(image, scale_factor=scale_factor)
        
        data = synthetic_images
        b_min=0.0
        b_max=1.0
        data = torch.clip(data, b_min, b_max)
        return data

def run_inference_path(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: Union[float,tuple],
    pirads_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor | None,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        pirads_tensor (torch.Tensor): PIRADS tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    noise = torch.randn(
        (
            spacing_tensor.shape[0],
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}, {noise.shape}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    recon_model = ReconModelRaw(autoencoder=autoencoder).to(device)
    autoencoder.eval()
    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        for t in tqdm(noise_scheduler.timesteps, ncols=110):
            timestep_preshape = [t for _ in range(spacing_tensor.shape[0])]
            model_output = unet(
                x=image,
                timesteps=torch.Tensor(timestep_preshape).to(device),
                pirads=pirads_tensor,
                spacing_tensor=spacing_tensor,
            )
            image, _ = noise_scheduler.step(model_output, t, image)

        synthetic_images = recon_model(image, scale_factor=scale_factor)

        data = synthetic_images
        b_min=0.0
        b_max=1.0
        data = torch.clip(data, b_min, b_max)
        return data

def run_inference_text(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: Union[float,tuple],
    pirads_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor | None,
    text_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        pirads_tensor (torch.Tensor): PIRADS tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        text_tensor (torch.Tensor): Text tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    noise = torch.randn(
        (
            spacing_tensor.shape[0],
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}, {noise.shape}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    recon_model = ReconModelRaw(autoencoder=autoencoder).to(device)
    autoencoder.eval()
    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        for t in tqdm(noise_scheduler.timesteps, ncols=110):
            timestep_preshape = [t for _ in range(spacing_tensor.shape[0])]
            model_output = unet(
                x=image,
                timesteps=torch.Tensor(timestep_preshape).to(device),
                pirads=pirads_tensor,
                spacing_tensor=spacing_tensor,
                text_encoding=text_tensor,
            )
            image, _ = noise_scheduler.step(model_output, t, image)

        synthetic_images = recon_model(image, scale_factor=scale_factor)

        data = synthetic_images
        b_min=0.0
        b_max=1.0
        data = torch.clip(data, b_min, b_max)
        return data

       
def run_inference_textclass(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: Union[float,tuple],
    spacing_tensor: torch.Tensor | None,
    text_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        spacing_tensor (torch.Tensor): Spacing tensor.
        text_tensor (torch.Tensor): Text tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    noise = torch.randn(
        (
            spacing_tensor.shape[0],
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}, {noise.shape}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    recon_model = ReconModelRaw(autoencoder=autoencoder).to(device)
    autoencoder.eval()
    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        for t in tqdm(noise_scheduler.timesteps, ncols=110):
            timestep_preshape = [t for _ in range(spacing_tensor.shape[0])]
            model_output, class_pred = unet(
                x=image,
                timesteps=torch.Tensor(timestep_preshape).to(device),
                spacing_tensor=spacing_tensor,
                text_encoding=text_tensor,
            )
            image, _ = noise_scheduler.step(model_output, t, image)

        synthetic_images = recon_model(image, scale_factor=scale_factor)

        data = synthetic_images
        b_min=0.0
        b_max=1.0
        data = torch.clip(data, b_min, b_max)
        return data, class_pred


def spatial_average_2d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)

def get_features_2d(image, radnet):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Get model outputs
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            feature_image = radnet.forward(image)
            # flattens the image spatially
            feature_image = spatial_average_2d(feature_image, keepdim=False)

    return feature_image

def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)

def get_features(image, radnet):
    # Get model outputs
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            feature_image = radnet.forward(image)
            # flattens the image spatially
            feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image

def main_diffusion(args):
    logger = logging.getLogger("maisi.evaluate_diffusion")
    args.num_gpus=1
    # whether to use distributed data parallel
    use_ddp = args.num_gpus > 1
    if use_ddp:
        raise NotImplementedError("I don't think ddp works on this eval script for now")
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{args.device}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    with open(args.environment_file, "r") as env_file:
        env_dict = json.load(env_file)
    with open(args.config_file, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.training_config, "r") as training_config_file:
        training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in training_config_dict.items():
        setattr(args, k, v)

    # # Step 1: set data loader

    imkey=args.diffusion_unet_inference["imkey"]

    local_rank = 0
    world_size = 1
    device = torch.device("cuda", args.device)
    logger = setup_logging("inference")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank
        if args.diffusion_unet_inference["random_seed"]
        else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])
    output_prefix = args.output_prefix

    if local_rank == 0:
        logger.info(f"[config] random_seed -> {random_seed}.")
        logger.info(f"[config] output_prefix -> {output_prefix}.")
        logger.info(f"[config] output_size -> {output_size}.")
        logger.info(f"[config] out_spacing -> {out_spacing}.")

    synth_features = {imkey:[]}
    real_features = {imkey:[]}
    synth_features_2d_x = {imkey:[]}
    real_features_2d_x = {imkey:[]}
    synth_features_2d_y = {imkey:[]}
    real_features_2d_y = {imkey:[]}
    synth_features_2d_z = {imkey:[]}
    real_features_2d_z = {imkey:[]}

    if 'use_text_conditioning' in args and args.use_text_conditioning:
        val_loader = prepare_maisi_controlnet_json_dataloader_text_eval(
            json_data_list=args.test_json,
            data_base_dir=args.data_base_dir,
            embedding_base_dir=args.embedding_base_dir,
            imkey=imkey,
            lblkey='label',
            is_val=True,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            cache_rate=0.0,
        )
        if 'use_text_class_pred' in args and args.use_text_class_pred:
            class_preds = []
            class_reals = []
    else:
        if args.base_maisi:
            prepare_maisi_diffusion_json_dataloader = prepare_maisi_controlnet_json_dataloader_base_eval
        else:
            prepare_maisi_diffusion_json_dataloader = prepare_maisi_controlnet_json_dataloader_path_eval

        val_loader = prepare_maisi_diffusion_json_dataloader(
            json_data_list=args.test_json,
            data_base_dir=args.data_base_dir,
            embedding_base_dir=args.embedding_base_dir,
            imkey=imkey,
            lblkey='label',
            is_val=True,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            cache_rate=0.0,
        )

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)
    logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

    radnet_2d = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True, trust_repo=True, ) # 2D
    radnet_2d.to(device)
    radnet_2d.eval()

    radnet = torch.hub.load("Warvito/MedicalNet-models", model="medicalnet_resnet50_23datasets", verbose=True, trust_repo=True, )
    radnet.to(device)
    radnet.eval()

    gen_flag = 4* int((not args.no_infer))

    for idx, batch in tqdm(enumerate(val_loader)):
        if args.base_maisi:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
            spacing_tensor = batch["spacing"].to(device)
        elif 'use_text_conditioning' in args and args.use_text_conditioning:
            pirads_tensor = batch["pirads"].to(device)
            spacing_tensor = batch["spacing"].to(device)
            text_tensor = batch["text"].to(device)
        else:
            pirads_tensor = batch["pirads"].to(device)
            spacing_tensor = batch["spacing"].to(device)

        if args.base_maisi:
            synthetic_images = run_inference_base(
                args,
                device,
                autoencoder,
                unet,
                scale_factor,
                top_region_index_tensor,
                bottom_region_index_tensor,
                spacing_tensor,
                output_size,
                divisor,
                logger,
            )
        elif 'use_text_conditioning' in args and args.use_text_conditioning:
            if 'use_text_class_pred' in args and args.use_text_class_pred:
                synthetic_images, class_pred = run_inference_textclass(
                    args,
                    device,
                    autoencoder,
                    unet,
                    scale_factor,
                    spacing_tensor,
                    text_tensor,
                    output_size,
                    divisor,
                    logger,
                )

                # Generate softmaxes now 
                class_pred_torch = torch.nn.functional.softmax(class_pred, dim=1).cpu().detach().numpy()
                class_real_torch = torch.argmax(pirads_tensor, dim=1).cpu().detach().numpy()
                for case_i in range(class_pred.shape[0]):
                    class_preds.append(class_pred_torch[case_i])
                    class_reals.append(class_real_torch[case_i])
            else:
                synthetic_images = run_inference_text(
                    args,
                    device,
                    autoencoder,
                    unet,
                    scale_factor,
                    pirads_tensor,
                    spacing_tensor, # Spacing tensor
                    text_tensor,
                    output_size,
                    divisor,
                    logger,
                )
        else:
            synthetic_images = run_inference_path(
                args,
                device,
                autoencoder,
                unet,
                scale_factor,
                pirads_tensor,
                spacing_tensor, # Spacing tensor
                output_size,
                divisor,
                logger,
            )

            
        real_im = batch[f"{imkey}_orig"].to(device)
        # For FID
        for j in range(synthetic_images.shape[2]):
            r_im_features_2d = get_features_2d(real_im[:,:,j,:,:], radnet_2d)
            s_im_features_2d = get_features_2d(synthetic_images[:,:,j,:,:], radnet_2d)

            for i in range(synthetic_images.shape[0]):
                real_features_2d_x[imkey].append(r_im_features_2d[i,...].unsqueeze(0))
                synth_features_2d_x[imkey].append(s_im_features_2d[i,...].unsqueeze(0))

        for j in range(synthetic_images.shape[3]):
            r_im_features_2d = get_features_2d(real_im[:,:,:,j,:], radnet_2d)
            s_im_features_2d = get_features_2d(synthetic_images[:,:,:,j,:], radnet_2d)

            for i in range(synthetic_images.shape[0]):
                real_features_2d_y[imkey].append(r_im_features_2d[i,...].unsqueeze(0))
                synth_features_2d_y[imkey].append(s_im_features_2d[i,...].unsqueeze(0))

        for j in range(synthetic_images.shape[4]):
            r_im_features_2d = get_features_2d(real_im[:,:,:,:,j], radnet_2d)
            s_im_features_2d = get_features_2d(synthetic_images[:,:,:,:,j], radnet_2d)

            for i in range(synthetic_images.shape[0]):
                real_features_2d_z[imkey].append(r_im_features_2d[i,...].unsqueeze(0))
                synth_features_2d_z[imkey].append(s_im_features_2d[i,...].unsqueeze(0))

        real_im_features = get_features(real_im, radnet)
        synth_im_features = get_features(synthetic_images, radnet)

        for i in range(synth_im_features.shape[0]):
            real_features[imkey].append(real_im_features[i,...].unsqueeze(0))
            synth_features[imkey].append(synth_im_features[i,...].unsqueeze(0))


        if gen_flag > 0: # Save the first generated image
            gen_flag -= 1
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            if imkey=='axt2':
                mult = 962.0
            elif imkey=='b1600':
                mult = 117.0
            elif imkey=='adc':
                mult = 3000.0
            else:
                raise ValueError(f"Unknown imkey {imkey}")
            synth_np = (synthetic_images[0,0,...].cpu().detach().numpy()*mult).astype(np.int16)
            real_np = (real_im[0,0,...].cpu().detach().numpy()*mult).astype(np.int16)

            synth_nii = nib.Nifti1Image(synth_np, np.eye(4))
            synth_filename = os.sep.join([args.output_dir, f"{output_prefix}_{gen_flag}_synth.nii.gz"])
            nib.save(synth_nii, synth_filename)
            real_nii = nib.Nifti1Image(real_np, np.eye(4))
            real_filename = os.sep.join([args.output_dir, f"{output_prefix}_{gen_flag}_real.nii.gz"])
            nib.save(real_nii, real_filename)

            if args.infer_only:
                raise ValueError("Pause, Infer done")
        

    fid_results = {}
    fid_results_2d_x = {}
    fid_results_2d_y = {}
    fid_results_2d_z = {}
    fid = FIDMetric()

    for k in real_features:
        synth_features[k] = torch.vstack(synth_features[k])
        real_features[k] = torch.vstack(real_features[k])
        
        fid_results[k] = fid(synth_features[k], real_features[k]) # y_pred, y

        synth_features_2d_x[k] = torch.vstack(synth_features_2d_x[k])
        real_features_2d_x[k] = torch.vstack(real_features_2d_x[k])
        fid_results_2d_x[k] = fid(synth_features_2d_x[k], real_features_2d_x[k]) # y_pred, y
        
        synth_features_2d_y[k] = torch.vstack(synth_features_2d_y[k])
        real_features_2d_y[k] = torch.vstack(real_features_2d_y[k])
        fid_results_2d_y[k] = fid(synth_features_2d_y[k], real_features_2d_y[k]) # y_pred, y
        
        synth_features_2d_z[k] = torch.vstack(synth_features_2d_z[k])
        real_features_2d_z[k] = torch.vstack(real_features_2d_z[k])
        fid_results_2d_z[k] = fid(synth_features_2d_z[k], real_features_2d_z[k]) # y_pred, y

        print(f"Imkey: {k}, FID: {fid_results[k]:.4f}, FID_2D_X: {fid_results_2d_x[k]:.4f}, FID_2D_Y: {fid_results_2d_y[k]:.4f}, FID_2D_Z: {fid_results_2d_z[k]:.4f}")

    
    if not args.modelname_override:
        pkl_filepath = f"{args.model_filename.removesuffix('.pt')}"
    else:
        pkl_filepath = f"{args.modelname_override.removesuffix('.pt')}"

        
    with open(f'metrics_diffusion_{pkl_filepath}.pkl','wb') as f:
        pickle.dump((fid_results,fid_results_2d_x,fid_results_2d_y,fid_results_2d_z),f)

    if 'use_text_conditioning' in args and args.use_text_conditioning and 'use_text_class_pred' in args and args.use_text_class_pred:
        class_preds_raw = np.array(class_preds)
        class_preds = np.argmax(class_preds_raw,axis=1)
        class_reals = np.array(class_reals)
        class_acc = accuracy_score(class_reals, class_preds)
        tn, fp, fn, tp = confusion_matrix(class_reals, class_preds).ravel()
        class_sens = tp/(tp+fn)
        class_spec = tn/(tn+fp)
        class_ppv = tp/(tp+fp)
        class_npv = tn/(tn+fn)

        auc = roc_auc_score(class_reals, class_preds_raw[:,1])
        ap = average_precision_score(class_reals, class_preds_raw[:,1])

        print(f"AUC: {auc:.4f}, AP: {ap:.4f}")
        print(f"Class Acc: {class_acc:.4f}, Sens: {class_sens:.4f}, Spec: {class_spec:.4f}, PPV: {class_ppv:.4f}, NPV: {class_npv:.4f}")
        with open(f'class_metrics_diffusion_{pkl_filepath}.pkl','wb') as f:
            # pickle.dump((class_acc, tn, fp, fn, tp),f)
            pickle.dump((class_preds_raw, class_reals, class_acc, tn, fp, fn, tp, auc, ap),f)


    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="maisi.evaluate_diffusion")
    parser.add_argument(
        "-e",
        "--environment-file",
        "--env_config",
        default="./configs/environment_maisi_controlnet_train.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        "--model_def",
        default="./configs/config_maisi.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-t",
        "--training-config",
        "--model_config",
        default="./configs/config_maisi_controlnet_train.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument(
        "-r",
        "--modelname_override",
        default="",
        help="Override the model name",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="GPU ID to run on",
    )
    parser.add_argument("-g", "--num_gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="batch size")
    parser.add_argument('-i','--infer_only',action='store_true',help='Only generate images')
    parser.add_argument('-j','--no_infer',action='store_true',help='Do not infer images')
    args = parser.parse_args()
    
    with torch.no_grad():
        main_diffusion(args)