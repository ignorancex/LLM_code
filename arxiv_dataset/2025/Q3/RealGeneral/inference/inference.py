import logging
import argparse
from typing import Literal, Optional
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
import torch
from diffusers import (
    CogVideoXDPMScheduler,
    AutoencoderKLCogVideoX
)
from transformers import AutoTokenizer, T5EncoderModel
from diffusers.utils.torch_utils import is_compiled_module
import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from realgeneral.model import CogVideoXTransformer3DModel
from realgeneral.pipeline import CogVideoXPipeline_incontext


logging.basicConfig(level=logging.INFO)

def ref_video_encoder(image_path, vae, device, width=512, height=512):
    try:
        import decord
    except ImportError:
        raise ImportError(
            "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
        )
    decord.bridge.set_bridge("torch")

    train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
        ]
    )
    
    image = Image.open(image_path)
    image = image.resize((height, width)).convert("RGB")
    frames = torch.from_numpy(np.array(image)).unsqueeze(0)
    
    # Training transforms
    frames = frames.float()
    frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
    frames = frames.permute(0, 3, 1, 2).contiguous() # [F, C, H, W]

    # encode
    video = frames
    video = video.to(device, dtype=vae.dtype).unsqueeze(0)
    video = video.permute(0, 2, 1, 3, 4)  # [1, C, 1, H, W]
    latent_dist = vae.encode(video).latent_dist

    latent = latent_dist.sample() * vae.config.scaling_factor # [1, C, 1, H, W]
    latent = latent.permute(0, 2, 1, 3, 4)
    return latent

def unwrap_model(model):
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    reference_img: str = "",
    lora_alpha: int = 256,
    instance=None
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer_uce",
        torch_dtype=dtype,
    )
    transformer.change_norm()
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    
    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder").to("cuda",dtype=dtype)

    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae").to("cuda", dtype=dtype)

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    pipe = CogVideoXPipeline_incontext(
        tokenizer=tokenizer,  
        transformer=unwrap_model(transformer).to("cuda",dtype=dtype),
        text_encoder=unwrap_model(text_encoder),
        vae=unwrap_model(vae),
        scheduler=scheduler,
    )


    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")
        lora_scaling=lora_alpha/lora_rank
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    latent = ref_video_encoder(reference_img, pipe.vae, "cuda", width, height)
    latent = latent.repeat(1, 2, 1, 1, 1)

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    seed = torch.randint(0, 10000, (1,)).item()
    video_generate = pipe(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        reference_latent=latent,
        instance=instance,
    ).frames[0]
    video_generate[0].save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX1.5-5B", help="Path of the pre-trained model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--reference_img", type=str, default='')
    parser.add_argument("--lora_alpha", type=int, default=256, help="The alpha value for LoRA")
    parser.add_argument("--instance", type=str, default=None, help="The path of the LoRA weights to be used")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
        reference_img=args.reference_img,
        lora_alpha=args.lora_alpha,
        instance=args.instance
    )
