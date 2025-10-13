# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageFilter, ImageDraw

import cv2
from einops import rearrange
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from realgeneral.model import CogVideoXTransformer3DModel
from realgeneral.pipeline import CogVideoXPipeline_incontext


if is_wandb_available():
    import wandb

os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--instance_column",
        type=str,
        default='',
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=81, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
    )
    return parser.parse_args()


class VideoDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        instance_column: str = "instance",
        height: int = 512,
        width: int = 512,
        fps: int = 8,
        max_num_frames: int = 81,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""
        self.instance_column = instance_column
        self.instance = self.load_instance()

        self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            "instance_video": self.process_data(self.instance_video_paths[index]),
            "instance": self.instance[index]
        }

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )
        return instance_prompts, instance_videos
    
    def load_instance(self):
        instance_path = self.instance_data_root.joinpath(self.instance_column)
        with open(instance_path, "r", encoding="utf-8") as file:
            instance = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        return instance
        
    
    def process_data(self, path):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        decord.bridge.set_bridge("torch")

        video_reader = decord.VideoReader(uri=path.as_posix(), width=self.width, height=self.height)
        video_num_frames = len(video_reader) # 2 frame

        start_frame = min(self.skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - self.skip_frames_end)
        if end_frame <= start_frame:
            frames = video_reader.get_batch([start_frame])
        elif end_frame - start_frame <= self.max_num_frames:
            frames = video_reader.get_batch(list(range(start_frame, end_frame)))
        else:
            indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
            frames = video_reader.get_batch(indices)

        frames = frames[: self.max_num_frames]
        # Training transforms
        frames = frames.float()
        frames = frames.permute(0, 3, 1, 2).contiguous() # [F, C, H, W]
        frames = torch.stack([self.train_transforms(frame) for frame in frames], dim=0)
        return frames
    
    def _get_canny_edge(self, img):
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        return Image.fromarray(edges).convert("RGB")
    
def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"video_{i}.mp4"}}
            )

    model_description = f"""
# CogVideoX LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA weights for {base_model}.

The weights were trained using the [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

Was LoRA for the text encoder enabled? No.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import CogVideoXPipeline_incontext
import torch

pipe = CogVideoXPipeline_incontext.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name=["cogvideox-lora"])

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-lora"], [32 / 64])

video = pipe("{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) and [here](https://huggingface.co/THUDM/CogVideoX-2b/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "cogvideox",
        "cogvideox-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    free_memory()

    return videos


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 224,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
    instance=None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("`text_input_ids` must be provided when tokenizer is not specified.")

    target_embeds = None
    matching_indices = []
    for j, item in enumerate(instance):
        target_subtokens = tokenizer.tokenize(item)
        target_ids = tokenizer.convert_tokens_to_ids(target_subtokens)
        target_ids_tensor = torch.tensor(target_ids)
            
        # Compare the tokenized IDs with the text_input_ids
        if target_ids_tensor.size(0) > 1:
            for i in range(text_input_ids.size(1) - len(target_ids_tensor) + 1):
                if torch.equal(text_input_ids[0, i:i+len(target_ids_tensor)], target_ids_tensor):
                    match_pos = torch.tensor([i])
                    break
        else:
            matches = (text_input_ids[j] == target_ids_tensor)
            match_pos = torch.nonzero(matches, as_tuple=False)  # Get non-zero positions for each batch   
            
        first_match = match_pos[0:1].item()  # Get the first matching position (column index)
        matching_indices.append(first_match)         
        
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
    target_embeds = []
    for i, indices in enumerate(matching_indices):
        target_embeds.append(prompt_embeds[i, indices:indices+len(target_ids_tensor), :].mean(dim=0))
    target_embeds = torch.stack(target_embeds, dim=0)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, target_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 224,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
    instance=None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds, target_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
        instance=instance
    )
    return prompt_embeds, target_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, instance=None, requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds, target_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            instance=instance
        )
    else:
        with torch.no_grad():
            prompt_embeds, target_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                instance=instance
            )
    return prompt_embeds, target_embeds

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        base_num_frames = num_frames
    else:
        base_num_frames = (
            num_frames + patch_size_t - 1
        ) // patch_size_t
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(grid_height, grid_width),
        device=device,
    )

    return freqs_cos, freqs_sin

def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    expected_midxed_precision = "bf16" if "5b" in args.pretrained_model_name_or_path.lower() else "fp16"
    if args.mixed_precision != expected_midxed_precision:
        raise ValueError(f"Mixed precision {args.mixed_precision} does not match the model precision, should be {expected_midxed_precision}")

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if accelerator.state.deepspeed_plugin:
        # Set deepspeed config according to args
        config = {
            'optimizer': {
                'type': args.optimizer,
                'params': {
                    'lr': args.learning_rate,
                    'betas': [args.adam_beta1, args.adam_beta2]
                },
                'torch_adam': True
            },
            'bf16': {
                'enabled': True if args.mixed_precision == "bf16" else False
            },
            'fp16': {
                'enabled': True if args.mixed_precision == "fp16" else False
            },
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'train_batch_size': args.train_batch_size
        }
        accelerator.state.deepspeed_plugin.deepspeed_config.update(config)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer_uce",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter LoRA layers
    transformer.change_norm()
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    
    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "linear1", "linear2", "linear3"],
        modules_to_save=["image_index_embedding", "cond_param"]
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            CogVideoXPipeline_incontext.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        lora_state_dict = CogVideoXPipeline_incontext.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    for name, param in transformer.named_parameters():
        if "modules_to_save" in name:
            param.requires_grad = True
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    print("trainable parameters:", sum(p.numel() for p in transformer.parameters() if p.requires_grad))
    
    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    train_dataset = VideoDataset(
        instance_data_root=args.instance_data_root,
        caption_column=args.caption_column,
        video_column=args.video_column,
        instance_column=args.instance_column,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )

    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist            
        return latent_dist

    def collate_fn(examples):
        videos = []
        for example in examples:
            video = example["instance_video"]
            video = rearrange(video, "F C H W -> F C 1 H W")
            latents = encode_video(video) 
            latents = latents.sample() * vae.config.scaling_factor
            latents = rearrange(latents, "F C 1 H W -> 1 F C H W")
            videos.append(latents)

        prompts = [example["instance_prompt"] for example in examples]
        videos = torch.cat(videos)
        videos = videos.to(memory_format=torch.contiguous_format).float()

        return {
            "videos": videos,
            "prompts": prompts,
            "instance": [example["instance"] for example in examples],
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            warmup_num_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        num = 0
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                model_input = batch["videos"].to(dtype=weight_dtype)  # [B, F, C, H, W]
                prompts = batch["prompts"]
                instance = batch["instance"]
                # encode prompts
                prompt_embeds, target_embeds = compute_prompt_embeddings(
                    tokenizer,
                    text_encoder,
                    prompts,
                    model_config.max_text_seq_length,
                    accelerator.device,
                    weight_dtype,
                    requires_grad=False,
                    instance=instance,
                )

                # Sample noise that will be added to the latents
                noise = torch.randn_like(model_input[:, -1:])
                batch_size, num_frames, num_channels, height, width = model_input.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height,
                        width=args.width,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t if model_config.patch_size_t is not None else 1,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )
                image_rotary_emb = [torch.cat([image_rotary_emb_sub, image_rotary_emb_sub], dim=0) for image_rotary_emb_sub in image_rotary_emb]
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = scheduler.add_noise(model_input[:, -1:], noise, timesteps)
                noisy_model_input = noisy_model_input.repeat(1, 2, 1, 1, 1)
                cond_input = model_input[:, :-1].repeat(1, 2, 1, 1, 1)
                noisy_model_input = torch.cat((cond_input, noisy_model_input), dim=1)
                
                # Predict the noise residual
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    image_index_embedding = target_embeds,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input[:, -1:].repeat(1, 2, 1, 1, 1)
                model_pred = model_pred[:, -2:]            
                
                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        CogVideoXPipeline_incontext.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                videos=validation_outputs,
                base_model=args.pretrained_model_name_or_path,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                fps=args.fps,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
