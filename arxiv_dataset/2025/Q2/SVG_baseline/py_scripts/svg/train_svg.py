import logging
import math
import os
import random
from pathlib import Path
import itertools
import scipy
import shutil

from typing import Any, Dict, Optional

import accelerate
import datasets
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from pytorchvideo.transforms import UniformTemporalSubsample
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan, CLIPTextModel, CLIPTokenizer, SpeechT5FeatureExtractor

import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel, MotionAdapter, UNetMotionModel, UNet3DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from args import parse_args

from va_core.dataset import get_dataset_info
from va_core.model.svg.unet_svg import UNetModelSVG
from va_core.model.svg.utils import TrainingModel, logmel_extractor
from va_core.model.svg.pipeline_svg import VideoAudioGenWithSVGPipeline


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# [V2A] need to update this for v2a
def log_validation(test_dataloader,
                    vae_a,
                    text_encoder_a,
                    tokenizer_a,
                    vocoder,
                    scheduler_a,
                    unet_a,
                    vae_v,
                    text_encoder_v,
                    tokenizer_v,
                    scheduler_v,
                    unet_v,
                    unet_svg,
                    args, 
                    accelerator, 
                    epoch,
                    ):
    logger.info("Running validation... ")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = VideoAudioGenWithSVGPipeline(
            vae_a=vae_a,
            text_encoder_a=text_encoder_a,
            tokenizer_a=tokenizer_a,
            vocoder=vocoder,
            scheduler_a=scheduler_a,
            unet_a=unet_a,
            vae_v=vae_v,
            text_encoder_v=text_encoder_v,
            tokenizer_v=tokenizer_v,
            scheduler_v=scheduler_v,
            unet_v=unet_v,
            unet_svg=unet_svg,
    )
    pipeline = pipeline.to(accelerator.device, torch_dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    dirname = os.path.join(args.output_dir, "val", str(epoch))
    os.makedirs(dirname, exist_ok=True)
    negative_prompt = "bad quality, worse quality"
    for step, batch in enumerate(test_dataloader):
        # Assume bs=1 now so we access batch["text"][0] explicitly.
        prompt = batch["text"][0]
        prompt_a = prompt
        prompt_v = prompt

        output = pipeline(prompt_a=prompt_a, prompt_v=prompt_v, negative_prompt_v=negative_prompt, negative_prompt_a=negative_prompt, length_in_s=args.duration_per_sample, frame_rate=args.num_frames_per_sample//args.duration_per_sample, height_v=args.resolution, width_v=args.resolution)
        video_to_save = (output.videos[0] * 255.0).astype(np.uint8)
        audio_to_save = output.audios[0]
        scipy.io.wavfile.write(dirname + f"/{step}_gen_{prompt}.wav", rate=16000, data=audio_to_save)
        torchvision.io.write_video(dirname + f"/{step}_gen_{prompt}.mp4", video_to_save, fps=args.num_frames_per_sample/args.duration_per_sample)

        if step >= args.num_validation_samples - 1:
            break

    del pipeline
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    path_video_model = ""
    if args.video_model == "animatediff":
        # AnimateDiff
        # Load the motion adapter
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
        # Load SD 1.5 based finetuned model
        path_video_model = "SG161222/Realistic_Vision_V5.1_noVAE"
        scheduler_v = DDIMScheduler.from_pretrained(path_video_model, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1, beta_schedule="linear")
        tokenizer_v = CLIPTokenizer.from_pretrained(path_video_model, subfolder="tokenizer")
        unet_i = UNet2DConditionModel.from_pretrained(path_video_model, subfolder="unet")
        # Create unet for video
        unet_v = UNetMotionModel.from_unet2d(unet_i, adapter)
    elif args.video_model == "modelscope":
        # ModelScope
        path_video_model = "damo-vilab/text-to-video-ms-1.7b"
        scheduler_v = DDIMScheduler.from_pretrained(path_video_model, subfolder="scheduler")
        tokenizer_v = CLIPTokenizer.from_pretrained(path_video_model, subfolder="tokenizer")
        text_encoder_v = CLIPTextModel.from_pretrained(path_video_model, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
        vae_v = AutoencoderKL.from_pretrained(path_video_model, subfolder="vae", torch_dtype=torch.float16, variant="fp16")
        unet_v = UNet3DConditionModel.from_pretrained(path_video_model, subfolder="unet", torch_dtype=torch.float16, variant="fp16")
    elif args.video_model == "zeroscope":
        path_video_model = "cerspense/zeroscope_v2_576w"
        scheduler_v = DDIMScheduler.from_pretrained(path_video_model, subfolder="scheduler")
        tokenizer_v = CLIPTokenizer.from_pretrained(path_video_model, subfolder="tokenizer")
        text_encoder_v = CLIPTextModel.from_pretrained(path_video_model, subfolder="text_encoder", torch_dtype=torch.float16)
        vae_v = AutoencoderKL.from_pretrained(path_video_model, subfolder="vae", torch_dtype=torch.float16)
        unet_v = UNet3DConditionModel.from_pretrained(path_video_model, subfolder="unet", torch_dtype=torch.float16)
    else:
        print(f"Unknown video model: {args.video_model}")
        raise
    scheduler_v.set_timesteps(100, device=accelerator.device) # dummy for x0 prediction

    # AudioLDM
    path_audioldm = "cvssp/audioldm-m-full"
    scheduler_a = DDIMScheduler.from_pretrained(path_audioldm, subfolder="scheduler")
    tokenizer_a = RobertaTokenizer.from_pretrained(path_audioldm, subfolder="tokenizer")
    unet_a = UNet2DConditionModel.from_pretrained(path_audioldm, subfolder="unet")
    vocoder = SpeechT5HifiGan.from_pretrained(path_audioldm, subfolder="vocoder")
    unet_a_upsample_factor = 2 ** unet_a.num_upsamplers
    scheduler_a.set_timesteps(100, device=accelerator.device) # dummy for x0 prediction

    # SVG
    connector_config = {}
    if args.connector_in_type is not None:
        connector_config["connector_in_type"] = args.connector_in_type
    if args.use_x0_pred_for_connectors:
        connector_config["connector_out_input"] = "x_pred"
    if any(connector_config):
        unet_svg = UNetModelSVG(unet_a.config, unet_v.config, connector_audio_config=connector_config, connector_video_config=connector_config)
    else:
        unet_svg = UNetModelSVG(unet_a.config, unet_v.config)

    # Wrapper model
    train_model = TrainingModel(unet_svg, unet_a, unet_v)

    # Parameters of log-mel feature extractor are from AudioLDM paper
    sampling_rate = 16000
    # mel_extractor = SpeechT5FeatureExtractor(sampling_rate=sampling_rate,
    #                                          num_mel_bins=64,
    #                                          hop_length=160*1000//sampling_rate, 
    #                                          win_length=1024*1000//sampling_rate, 
    #                                          fmin=0, 
    #                                          fmax=8000)
    mel_extractor = logmel_extractor(sampling_rate)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder_a = ClapTextModelWithProjection.from_pretrained(path_audioldm, subfolder="text_encoder")
        vae_a = AutoencoderKL.from_pretrained(path_audioldm, subfolder="vae")
        text_encoder_v = CLIPTextModel.from_pretrained(path_video_model, subfolder="text_encoder")
        vae_v = AutoencoderKL.from_pretrained(path_video_model, subfolder="vae")

    # Freeze parameters
    text_encoder_a.requires_grad_(False)
    vae_a.requires_grad_(False)
    text_encoder_v.requires_grad_(False)
    vae_v.requires_grad_(False)
    if not args.finetune_unet:
        for k, v in train_model.named_parameters():
            if "connector" not in k:
                phrase_temp = "motion" if isinstance(unet_v, UNetMotionModel) else "temp_"
                if args.fix_temporal or phrase_temp not in k:
                    v.requires_grad_(False)
                    if accelerator.is_main_process:
                        print(f"{k} is fixed.")

    # Create EMA for the unet.
    if args.use_ema:
        if any(connector_config):
            ema_unet = UNetModelSVG(unet_a.config, unet_v.config, connector_audio_config=connector_config, connector_video_config=connector_config)
        else:
            ema_unet = UNetModelSVG(unet_a.config, unet_v.config)
        tmp_config = {k: v for k, v in ema_unet.config.items() if k != "_use_default_values"}
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNetModelSVG, model_config=tmp_config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
            
            assert len(models) == 1

            tm: TrainingModel = accelerator.unwrap_model(models[0])
            u_a = tm.unet_a
            u_v = tm.unet_v
            u_c = tm.unet_svg

            u_a.save_pretrained(os.path.join(output_dir, "unet_a"))
            u_v.save_pretrained(os.path.join(output_dir, "unet_v"), safe_serialization=False)
            u_c.save_pretrained(os.path.join(output_dir, "unet_svg"), safe_serialization=False)

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetModelSVG)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            del models

            # load unet parameters
            targets = [("unet_a", unet_a, unet_a.__class__), ("unet_v", unet_v, unet_v.__class__), ("unet_svg", unet_svg, unet_svg.__class__)]
            for key, model, model_cls in targets:
                loaded = model_cls.from_pretrained(input_dir, subfolder=key)
                model.register_to_config(**loaded.config)
                model.load_state_dict(loaded.state_dict())
                del loaded

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet_a.enable_gradient_checkpointing()
        # unet_v.enable_gradient_checkpointing()
        unet_svg.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * total_batch_size
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        [x for x in train_model.parameters() if x.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # setup tokenizer for captions.
    def tokenize_captions(batched_captions, is_train=True):
        captions = []
        for caption in batched_captions:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `text` should contain either strings or lists of strings."
                )
        return captions

    # Get a dataset
    ds_common_args = {
        "dataset_names": args.dataset_name,
        "duration_per_sample": args.duration_per_sample,
        "sampling_rate": sampling_rate,
        "sample_random_segments": True,
    }
    train_dataset_info = get_dataset_info(dataset_types="train",
                                          max_samples=args.max_train_samples,
                                          tokenizer=tokenize_captions,
                                          **ds_common_args)
    train_dataset = train_dataset_info.dataset
    train_collate_fn = train_dataset_info.collate_fn

    test_dataset_info = get_dataset_info(dataset_types="test",
                                         max_samples=args.num_validation_samples,
                                         tokenizer=lambda x: x,
                                         **ds_common_args)
    test_dataset = test_dataset_info.dataset
    test_collate_fn = test_dataset_info.collate_fn

    dataset_size = train_dataset_info.size
    if args.max_train_samples is not None and dataset_size > args.max_train_samples:
        dataset_size = args.max_train_samples

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True if not isinstance(train_dataset, torch.utils.data.IterableDataset) else False,
        collate_fn=train_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    tf_video = torch.nn.Sequential(
        UniformTemporalSubsample(args.num_frames_per_sample, temporal_dim=1)
    ).to(device=accelerator.device)

    tf_audio = torch.nn.Sequential(
        UniformTemporalSubsample(sampling_rate * args.duration_per_sample, temporal_dim=-1)
    ).to(device=accelerator.device)

    # Data processing for training
    train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: 2.0 * (x / 255.0) - 1.0),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(dataset_size / total_batch_size)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    train_model, optimizer, train_dataloader, lr_scheduler = \
        accelerator.prepare(
            train_model, optimizer, train_dataloader, lr_scheduler
        )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder_a.to(accelerator.device, dtype=weight_dtype)
    text_encoder_v.to(accelerator.device, dtype=weight_dtype)
    vae_a.to(accelerator.device, dtype=weight_dtype)
    vae_v.to(accelerator.device, dtype=weight_dtype)
    train_model.to(accelerator.device, dtype=weight_dtype)
    vocoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(dataset_size / total_batch_size)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  The number of parameters to be trained = {sum(p.numel() for p in train_model.parameters() if p.requires_grad)}")
    logger.info(f"  The number of parameters in unet_a = {sum(p.numel() for p in unet_a.parameters())}")
    logger.info(f"  The number of parameters in unet_v = {sum(p.numel() for p in unet_v.parameters())}")
    logger.info(f"  The number of parameters in unet_svg = {sum(p.numel() for p in unet_svg.parameters())}")
    logger.info(f"  The total number of parameters = {sum(p.numel() for p in train_model.parameters())}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    # first validation
    if accelerator.is_main_process:
        train_model_unwraped: TrainingModel = accelerator.unwrap_model(train_model)
        unet_svg_unwraped = train_model_unwraped.unet_svg
        unet_a_unwraped = train_model_unwraped.unet_a
        unet_v_unwraped = train_model_unwraped.unet_v
        log_validation(test_dataloader, 
                        vae_a=vae_a,
                        text_encoder_a=text_encoder_a,
                        tokenizer_a=tokenizer_a,
                        vocoder=vocoder,
                        scheduler_a=scheduler_a,
                        unet_a=unet_a_unwraped,
                        vae_v=vae_v,
                        text_encoder_v=text_encoder_v,
                        tokenizer_v=tokenizer_v,
                        scheduler_v=scheduler_v,
                        unet_v=unet_v_unwraped,
                        unet_svg=unet_svg_unwraped,
                        args=args, 
                        accelerator=accelerator, 
                        epoch=0)

    xtype = ["audio", "video"]
    scheduler = [scheduler_a, scheduler_v]
    for epoch in range(first_epoch, args.num_train_epochs):
        # train_loss = 0.0
        train_loss = [0.,] * len(xtype)
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue
            
            train_model.train()
            
            # get data from batch
            text_input = batch["text"]
            input_data = [tf_audio(batch["audio"].to(device="cpu")), tf_video(batch["video"].to(device="cpu"))]

            latents = []
            timesteps = []
            timesteps_con = []
            timesteps_shared = torch.randint(0, 1000, (args.train_batch_size,))
            for i, xtype_i in enumerate(xtype):
                if xtype_i == "audio":
                    # Convert raw to latent features
                    audio_target = input_data[i].to(weight_dtype).cpu().numpy()
                    mel_features = mel_extractor(audio_target, args.duration_per_sample)
                    mel_features = torch.from_numpy(mel_features).to(accelerator.device, dtype=weight_dtype).unsqueeze(1)
                    # mel_features_dict = mel_extractor(audio_target=audio_target,
                    #                                 sampling_rate=sampling_rate, 
                    #                                 max_length=sampling_rate//160*args.duration_per_sample,
                    #                                 truncation=True)
                    # mel_features = 10 ** torch.from_numpy(mel_features_dict["input_values"])
                    # mel_features = torch.log(torch.clamp(mel_features.to(accelerator.device, dtype=weight_dtype), min=1e-5)).unsqueeze(1)
                    latents_a = vae_a.encode(mel_features).latent_dist.sample()
                    latents_a = latents_a * vae_a.config.scaling_factor
                    latents.append(latents_a)
                    # Sample a random timestep
                    timesteps_a = timesteps_shared.to(device=latents_a.device) * (scheduler_a.config.num_train_timesteps // 1000)
                    timesteps_a = timesteps_a.long()
                    timesteps.append(timesteps_a)
                    t_con = torch.randint(0, scheduler_a.config.num_train_timesteps, (args.train_batch_size,)).to(device=accelerator.device)
                    timesteps_con.append(t_con)
                elif xtype_i == "video":
                    # Convert raw to latent features
                    B, T = input_data[i].shape[:2]
                    video_target = train_transforms(input_data[i].flatten(end_dim=1).to(device=accelerator.device, dtype=weight_dtype))
                    latents_v = vae_v.encode(video_target).latent_dist.sample()
                    latents_v = latents_v * vae_v.config.scaling_factor
                    latents_v = latents_v.reshape((B, T) + latents_v.shape[1:])
                    latents_v = latents_v.permute((0, 2, 1, 3, 4))
                    latents.append(latents_v)
                    # Sample a random timestep
                    timesteps_v = timesteps_shared.to(device=latents_v.device) * (scheduler_v.config.num_train_timesteps // 1000)
                    timesteps_v = timesteps_v.long()
                    timesteps.append(timesteps_v)
                    t_con = torch.randint(0, scheduler_v.config.num_train_timesteps, (args.train_batch_size,)).to(device=accelerator.device)
                    timesteps_con.append(t_con)

            # Sample noise that we'll add to the latents
            noise = [torch.randn_like(tmp, device=tmp.device) for tmp in latents]

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = []
            noisy_latents_con = []
            for i, xtype_i in enumerate(xtype):
                noisy_latents.append(scheduler[i].add_noise(latents[i], noise[i], timesteps[i]))
                noisy_latents_con.append(scheduler[i].add_noise(latents[i], noise[i], timesteps_con[i]))
                if xtype_i == "audio":
                    # Padding might be required for audio
                    original_length = noisy_latents[i].shape[2]
                    padded_length = int(np.ceil(original_length / unet_a_upsample_factor)) * unet_a_upsample_factor
                    if original_length != padded_length:
                        noisy_latents[i] = F.pad(noisy_latents[i], (0, 0, 0, padded_length - original_length))
                        noisy_latents_con[i] = F.pad(noisy_latents_con[i], (0, 0, 0, padded_length - original_length))

            # Get the text embedding for conditioning
            device = latents[0].device
            context = []
            class_labels = []
            text_input_processed = text_input
            if args.drop_rate_cfg > 0:
                null_tokens = ""
                index_null = np.where(np.random.rand(args.train_batch_size) < args.drop_rate_cfg)[0]
                if np.any(index_null):
                    text_input_processed = [null_tokens if i in list(index_null) else text_input[i] for i in range(args.train_batch_size)]

            for i, xtype_i in enumerate(xtype):
                if xtype_i == "audio":
                    text_inputs_a = tokenizer_a(text_input_processed, padding="max_length", max_length=tokenizer_a.model_max_length, truncation=True, return_tensors="pt")
                    prompt_embeds_a = text_encoder_a(text_inputs_a.input_ids.to(device), text_inputs_a.attention_mask.to(device))
                    prompt_embeds_a_pooled = F.normalize(prompt_embeds_a.text_embeds, dim=-1)
                    context.append(None)
                    class_labels.append(prompt_embeds_a_pooled)
                elif xtype_i == "video":
                    text_inputs_v = tokenizer_v(text_input_processed, padding="max_length", max_length=tokenizer_v.model_max_length, truncation=True, return_tensors="pt")
                    prompt_embeds_v = text_encoder_v(text_inputs_v.input_ids.to(device), text_inputs_v.attention_mask.to(device))
                    context.append(prompt_embeds_v[0])                    
                    class_labels.append(None)

            if args.use_x0_pred_for_connectors:
                block_con = None
                with torch.no_grad():
                    x_con_dummy = [torch.cat([tmp, torch.zeros_like(tmp, device=tmp.device)], dim=1) for tmp in noisy_latents_con]
                    noise_pred = train_model(noisy_latents, 
                                            timesteps,
                                            context,
                                            class_labels,
                                            xtype,
                                            block_connection=block_con,
                                            x_con=x_con_dummy)
                    for i, xtype_i in enumerate(xtype):
                        x_pred = []
                        for j in range(noise_pred[i].shape[0]):
                            if xtype_i == "audio":
                                x_pred.append(scheduler_a.step(noise_pred[i][j], timesteps[i][j], noisy_latents[i][j]).pred_original_sample.unsqueeze(0))
                            elif xtype_i == "video":
                                x_pred.append(scheduler_v.step(noise_pred[i][j], timesteps[i][j], noisy_latents[i][j]).pred_original_sample.unsqueeze(0))
                            else:
                                raise
                        x_pred = torch.cat(x_pred, dim=0).to(device=accelerator.device)
                        x_pred = torch.where((torch.rand(x_pred.shape[0]).to(device=accelerator.device) > 0.5).reshape([-1,] + [1]*(len(x_pred.shape)-1)).expand_as(x_pred), x_pred, torch.zeros_like(x_pred).to(device=accelerator.device))
                        noisy_latents_con[i] = torch.cat([noisy_latents_con[i], x_pred], dim=1)

            with accelerator.accumulate(train_model):
                model_pred = train_model(noisy_latents, 
                                         timesteps,
                                         context,
                                         class_labels,
                                         xtype,
                                         noisy_latents_con,
                                         timesteps_con)
               
                for i, xtype_i in enumerate(xtype):
                    if xtype_i == "audio" and original_length != padded_length:
                        model_pred[i] = model_pred[i][:, :, :original_length, :]

                # Get the target for loss depending on the prediction type
                target = []
                for i in range(len(xtype)):
                    if scheduler[i].config.prediction_type == "epsilon":
                        target.append(noise[i])
                    elif scheduler[i].config.prediction_type == "v_prediction":
                        target.append(scheduler[i].get_velocity(latents[i], noise[i], timesteps[i]))
                    else:
                        raise ValueError(f"Unknown prediction type {scheduler[i].config.prediction_type}")

                loss = [F.mse_loss(model_pred[i].float(), target[i].float(), reduction="none") for i in range(len(xtype))]

                # Gather the losses across all processes for logging (if we use distributed training).
                all_loss = [accelerator.gather(loss[i]) for i in range(len(xtype))]
                train_loss = [train_loss[i] + all_loss[i].mean().item() / args.gradient_accumulation_steps for i in range(len(xtype))]

                # Backpropagate
                accelerator.backward(sum([l.mean() for l in loss]))
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients and (step + 1) % args.gradient_accumulation_steps == 0:
                if args.use_ema:
                    train_model_unwraped: TrainingModel = accelerator.unwrap_model(train_model)
                    ema_unet.step(train_model_unwraped.unet_svg.parameters())
                progress_bar.update(1)
                global_step += 1
                for i in range(len(xtype)):
                    accelerator.log({f"train_loss_{i}": train_loss[i]}, step=global_step)
                train_loss = [0.0,] * len(xtype)

                if global_step % args.checkpointing_steps == 0:
                    # save state
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    # check the number of checkpoints
                    if args.checkpoints_total_limit > 0 and accelerator.is_main_process:
                        dirs = os.listdir(args.output_dir)
                        dirs = [d for d in dirs if d.startswith("checkpoint")]
                        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                        if len(dirs) > args.checkpoints_total_limit:
                            path = dirs[0]
                            shutil.rmtree(os.path.join(args.output_dir, path))

            logs = {"lr": lr_scheduler.get_last_lr()[0]}
            for i in range(len(xtype)):
                logs[f"step_loss_{i}"] = loss[i].mean().detach().item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_epochs > 0 and (epoch + 1) % args.validation_epochs == 0:
                train_model_unwraped: TrainingModel = accelerator.unwrap_model(train_model)
                unet_svg_unwraped = train_model_unwraped.unet_svg
                unet_a_unwraped = train_model_unwraped.unet_a
                unet_v_unwraped = train_model_unwraped.unet_v

                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet_svg_unwraped.parameters())
                    ema_unet.copy_to(unet_svg_unwraped.parameters())

                log_validation(test_dataloader, 
                                vae_a=vae_a,
                                text_encoder_a=text_encoder_a,
                                tokenizer_a=tokenizer_a,
                                vocoder=vocoder,
                                scheduler_a=scheduler_a,
                                unet_a=unet_a_unwraped,
                                vae_v=vae_v,
                                text_encoder_v=text_encoder_v,
                                tokenizer_v=tokenizer_v,
                                scheduler_v=scheduler_v,
                                unet_v=unet_v_unwraped,
                                unet_svg=unet_svg_unwraped,
                                args=args, 
                                accelerator=accelerator, 
                                epoch=epoch+1)
                
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet_svg_unwraped.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        train_model: TrainingModel = accelerator.unwrap_model(train_model)
        unet_svg_unwraped = train_model_unwraped.unet_svg
        unet_a_unwraped = train_model_unwraped.unet_a
        unet_v_unwraped = train_model_unwraped.unet_v
        if args.use_ema:
            ema_unet.copy_to(unet_svg_unwraped.parameters())

        # need to define a new pipeline 
        pipe = VideoAudioGenWithSVGPipeline(
                vae_a=vae_a,
                text_encoder_a=text_encoder_a,
                tokenizer_a=tokenizer_a,
                vocoder=vocoder,
                scheduler_a=scheduler_a,
                unet_a=unet_a_unwraped,
                vae_v=vae_v,
                text_encoder_v=text_encoder_v,
                tokenizer_v=tokenizer_v,
                scheduler_v=scheduler_v,
                unet_v=unet_v_unwraped,
                unet_svg=unet_svg_unwraped,
        )
        # Do not use safetensors to avoid an error related to shared tensors.
        pipe.save_pretrained(os.path.join(args.output_dir, "pipe"), safe_serialization=False)

    accelerator.end_training()


if __name__ == "__main__":
    main()
