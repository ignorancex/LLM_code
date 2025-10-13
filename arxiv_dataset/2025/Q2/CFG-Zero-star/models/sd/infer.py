import torch
from sd3_pipeline import StableDiffusion3Pipeline


import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)

pipe = pipe.to("cuda")

seed = random.randint(0, 2**32 - 1)
print('seed: ',seed)

set_seed(seed)

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
    use_cfg_zero_star=True,
    use_zero_init=True,
    zero_steps=0
).images[0]
image.save("output/output_ours.png")

set_seed(seed)

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
    use_cfg_zero_star=False,
    use_zero_init=False,
    zero_steps=0
).images[0]
image.save("output/output_cfg.png")



