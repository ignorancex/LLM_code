import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan#, WanPipeline
from wan_pipeline import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

import random
import os
import numpy as np


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")

prompt = "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

os.makedirs("output",exist_ok=True)

seed = random.randint(0, 2**32 - 1)
print('seed: ',seed)
set_seed(seed)
output = pipe(
     prompt=prompt,
     negative_prompt=negative_prompt,
     height=480,
     width=832,
     num_frames=81,
     guidance_scale=5.0,
     use_cfg_zero_star=False,
     use_zero_init=False,
     zero_steps=0
    ).frames[0]
export_to_video(output, f"output/{seed}_base.mp4", fps=15)

set_seed(seed)
output = pipe(
     prompt=prompt,
     negative_prompt=negative_prompt,
     height=480,
     width=832,
     num_frames=81,
     guidance_scale=5.0,
     use_cfg_zero_star=True,
     use_zero_init=True,
     zero_steps=1
    ).frames[0]
export_to_video(output, f"output/{seed}_ours.mp4", fps=15)




