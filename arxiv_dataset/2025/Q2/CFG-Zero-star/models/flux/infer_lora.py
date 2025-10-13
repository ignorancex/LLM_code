import torch
from pipeline import FluxPipeline
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

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

# Black Myth: Wukong
pipe.load_lora_weights('Shakker-Labs/FLUX.1-dev-LoRA-collections', weight_name='FLUX-dev-lora-Black_Myth_Wukong_hyperrealism_v1.safetensors')
pipe.fuse_lora(lora_scale=1.2)
pipe.to("cuda")

prompt = "aiyouxiketang, a man in armor with a beard and a beard"

seed = random.randint(0, 2**32 - 1)
print('seed: ',seed)

set_seed(seed)

image = pipe(
    prompt, 
    num_inference_steps=25, 
    guidance_scale=5.0,
    use_zero_init=False,
    zero_steps=0,
).images[0]
image.save("output/image_cfg.png")

set_seed(seed)

image = pipe(
    prompt, 
    num_inference_steps=25, 
    guidance_scale=5.0,
    use_zero_init=True,
    zero_steps=0,
).images[0]
image.save("output/image_ours.png")
