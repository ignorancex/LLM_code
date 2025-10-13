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
pipe.enable_model_cpu_offload()

prompt = "a tiny astronaut hatching from an egg on the moon"

seed = random.randint(0, 2**32 - 1)
print('seed: ',seed)

set_seed(seed)
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
    use_zero_init=True,
    zero_steps=0,
).images[0]
out.save("output/image_ours.png")

set_seed(seed)
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
    use_zero_init=False,
    zero_steps=0,
).images[0]
out.save("output/image_cfg.png")

