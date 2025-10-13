import torch
from diffusers import HunyuanVideoTransformer3DModel
from pipeline import HunyuanVideoPipeline
from diffusers.utils import export_to_video
import random
import os
import numpy as np

os.makedirs("output",exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.to("cuda")

seed = random.randint(0, 2**32 - 1)
print('seed: ',seed)

set_seed(seed)
output = pipe(
    prompt="In an ornate, historical hall, a massive tidal wave peaks and begins to crash. A man is surfing, cinematic film shot in 35mm. High quality, high defination.",
    height=720,
    width=1280,
    num_frames=61,
    num_inference_steps=50,
    use_zero_init=False,
    zero_steps=0,
).frames[0]
export_to_video(output, f"output/{seed}_output_cfg.mp4", fps=15)

set_seed(seed)
output = pipe(
    prompt="In an ornate, historical hall, a massive tidal wave peaks and begins to crash. A man is surfing, cinematic film shot in 35mm. High quality, high defination.",
    height=720,
    width=1280,
    num_frames=61,
    num_inference_steps=50,
    use_zero_init=True,
    zero_steps=1,
).frames[0]
export_to_video(output, f"output/{seed}_output_ours.mp4", fps=15)
