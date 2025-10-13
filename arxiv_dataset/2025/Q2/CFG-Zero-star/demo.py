import gradio as gr
from models.sd.sd3_pipeline import StableDiffusion3Pipeline
import torch
import random
import numpy as np
import os
import gc
import tempfile
import imageio
from diffusers import AutoencoderKLWan
from models.wan.wan_pipeline import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from PIL import Image
from diffusers.utils import export_to_video
import spaces

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3000"

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Model paths
model_paths = {
    "sd3.5": "stabilityai/stable-diffusion-3.5-large",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "wan-t2v": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
}

# Global variable for current model
current_model = None

# Folder to save video outputs
OUTPUT_DIR = "generated_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_name):
    global current_model
    if current_model is not None:
        del current_model  # Delete the old model
        torch.cuda.empty_cache()  # Free GPU memory
        gc.collect()  # Force garbage collection
    
    if "wan-t2v" in model_name:
        vae = AutoencoderKLWan.from_pretrained(model_paths[model_name], subfolder="vae", torch_dtype=torch.float32)
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=8.0)
        current_model = WanPipeline.from_pretrained(model_paths[model_name], vae=vae, torch_dtype=torch.bfloat16).to("cuda")
        current_model.scheduler = scheduler
    else:
        current_model = StableDiffusion3Pipeline.from_pretrained(model_paths[model_name], torch_dtype=torch.bfloat16).to("cuda")
    
    return current_model

@spaces.GPU(duration=2000)
def generate_content(prompt, model_name, guidance_scale=7.5, num_inference_steps=50, use_cfg_zero_star=True, use_zero_init=True, zero_steps=0, seed=None, compare_mode=False):
    model = load_model(model_name)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)

    is_video_model = "wan-t2v" in model_name

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    if is_video_model:
        if True:
            set_seed(seed)
            video1_frames = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=480,
                width=832,
                num_frames=81,
                guidance_scale=guidance_scale,
                use_cfg_zero_star=True,
                use_zero_init=True,
                zero_steps=zero_steps
            ).frames[0]
            video1_path = os.path.join(OUTPUT_DIR, f"{seed}_CFG-Zero-Star.mp4")
            export_to_video(video1_frames, video1_path, fps=16)

            set_seed(seed)
            video2_frames = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=480,
                width=832,
                num_frames=81,
                guidance_scale=guidance_scale,
                use_cfg_zero_star=False,
                use_zero_init=False,
                zero_steps=0
            ).frames[0]
            video2_path = os.path.join(OUTPUT_DIR,  f"{seed}_CFG.mp4")
            export_to_video(video2_frames, video2_path, fps=16)

            return None, None, video1_path, video2_path, seed

    if compare_mode:
        set_seed(seed)
        image1 = model(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            use_cfg_zero_star=True,
            use_zero_init=use_zero_init,
            zero_steps=zero_steps
        ).images[0]

        set_seed(seed)
        image2 = model(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            use_cfg_zero_star=False,
            use_zero_init=use_zero_init,
            zero_steps=zero_steps
        ).images[0]

        return image1, image2, None, None, seed
    else:
        image = model(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            use_cfg_zero_star=use_cfg_zero_star,
            use_zero_init=use_zero_init,
            zero_steps=zero_steps
        ).images[0]
        if use_cfg_zero_star:
            return image, None, None, None, seed
        else:
            return None, image, None, None, seed

# Gradio UI
demo = gr.Interface(
    fn=generate_content,
    inputs=[
        gr.Textbox(value="A spooky haunted mansion on a hill silhouetted by a full moon.", label="Enter your prompt"),
        gr.Dropdown(choices=list(model_paths.keys()), label="Choose Model"),
        gr.Slider(1, 20, value=4.0, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 100, value=28, step=5, label="Inference Steps"),
        gr.Checkbox(value=True, label="Use CFG Zero Star"),
        gr.Checkbox(value=True, label="Use Zero Init"),
        gr.Slider(0, 20, value=0, step=1, label="Zero out steps"),
        gr.Number(value=42, label="Seed (Leave blank for random)"),
        gr.Checkbox(value=True, label="Compare Mode")
    ],
    outputs=[
        gr.Image(type="pil", label="CFG-Zero* Image"),
        gr.Image(type="pil", label="CFG Image"),
        gr.Video(label="CFG-Zero* Video"),
        gr.Video(label="CFG Video"),
        gr.Textbox(label="Used Seed")
    ],
    title="CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models",
)

demo.launch(server_name="127.0.0.1", server_port=7860)

