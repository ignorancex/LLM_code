import os

import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

from .constants import BG_DIR

IMG_HEIGHT = 1024
IMG_WIDTH = 1024


class WuerstchenBackgroundGenerator:

    """
    @inproceedings{
      pernias2024wrstchen,
      title={W\"urstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models},
      author={Pablo Pernias and Dominic Rampas and Mats Leon Richter and Christopher Pal and Marc Aubreville},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=gU58d5QeGv}
    }
    """

    def __init__(self, model_name="warp-diffusion/wuerstchen", device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)

    def generate_backgrounds(self, prompts, background_types, n=50, output_dir=BG_DIR):

        os.makedirs(output_dir, exist_ok=True)

        for prompt, background_type in zip(prompts, background_types):
            prompt_dir = os.path.join(output_dir, background_type.replace(" ", "_"))
            os.makedirs(prompt_dir, exist_ok=True)

            for i in range(n):
                file_name = os.path.join(prompt_dir, f"{background_type.replace(' ', '_')}_{i + 1}.png")
                if os.path.exists(file_name):
                    print(f"File already exists, skipping: {file_name}")
                    continue

                with torch.no_grad():
                    output = self.pipeline(
                        prompt=prompt,
                        height=IMG_HEIGHT,
                        width=IMG_WIDTH,
                        prior_guidance_scale=4.0,  # https://huggingface.co/warp-ai/wuerstchen
                        decoder_guidance_scale=0.0,  # https://huggingface.co/warp-ai/wuerstchen
                    ).images[0]

                output.save(file_name)
                print(f"Saved: {file_name}")


# Usage example
"""
background_types = ["cloudscape", "space", "jungle", "desert", "arctic", "volcanic", "ocean", "abstract_patterns"]

prompts = [
    f"A vast sky filled with dramatic clouds in varying shapes and layers, emphasizing realistic lighting and textures, devoid of objects or creatures."
    if background == "cloudscape" else
    f"A vast space skybox with stars, galaxies, and nebulae, creating a serene cosmic atmosphere, devoid of objects or creatures."
    if background == "space" else
    f"A tranquil underwater ocean scene with soft light beams filtering through the water, focusing on the textures of sand, water ripples, and vibrant coral formations."
    if background == "ocean" else
    f"A volcanic scene with a dark sky, an erupting volcano, glowing red lava flows, and ash clouds in the atmosphere."
    if background == "volcanic" else
    f"A {background} background scene with a realistic and immersive environment, devoid of any objects, creatures, or vehicles."
    if "abstract" not in background else
    f"An {background.replace('_', ' ')} background with smooth, surreal elements and vibrant colors."
    for background in background_types
]

generator = WuerstchenBackgroundGenerator()
generator.generate_backgrounds(prompts, background_types, n=50)
"""



class UniformNoiseBackgroundGenerator:
    def __init__(self, output_dir=BG_DIR):
        self.output_dir = os.path.join(output_dir, "uniform_noise")
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_noise_images(self, n=1):
        for i in range(n):
            file_name = f"uniform_noise_{i+1}.png"
            if os.path.exists(file_name):
                    print(f"File already exists, skipping: {file_name}")
                    continue
            self.generate_noise_image(file_name)

    def generate_noise_image(self, filename):
        # Generate uniform random noise in the range [0, 255] for an RGB image
        noise = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        # Create an image from the noise array
        image = Image.fromarray(noise, 'RGB')

        # Save the image as a PNG file
        output_path = os.path.join(self.output_dir, filename)
        image.save(output_path)
        print(f"Image saved to {output_path}")
