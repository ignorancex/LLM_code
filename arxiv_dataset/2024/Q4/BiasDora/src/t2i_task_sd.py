import os
import json
import pandas as pd
import requests
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from torch import autocast
import argparse
import torch
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Generate images using Stable Diffusion model."
)
parser.add_argument(
    "--setting", type=str, default="1", help="Setting of images to generate"
)
args = parser.parse_args()

setting = str(args.setting)

df = pd.read_csv("../data/t2i_setting" + setting + ".csv")

if setting == "1":
    df["article"] = df["target_singular"].apply(
        lambda x: "an" if x[0].lower() in ["a", "e", "i", "o", "u"] else "a"
    )
    df["prompt"] = (
        df["template_singular"].str.replace(" a/an", "")
        + " "
        + df["article"]
        + " "
        + df["target_singular"].str.lower()
    )

elif setting == "2":
    df["prompt"] = df["template_plural"] + " " + df["target_plural"].str.lower()

print(df.head())

# Load secrets
with open("secrets.json", "r") as f:
    secrets = json.load(f)

# Initialize the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
).to("cuda")
pipe.vae = vae
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.to("cuda")


def generate_and_save_images(prompt, idx, target, number_of_images=10):
    directory = f"../outputs/stable_diffusion/setting{setting}/"
    os.makedirs(directory, exist_ok=True)

    for i in range(1, number_of_images + 1):
        try:
            with autocast("cuda"):
                generator = torch.Generator("cuda").manual_seed(
                    random.randint(0, 1000000)
                )
                image = pipe(
                    prompt, generator=generator, num_inference_steps=20
                ).images[0]
            # Save the image
            target = target.replace(" ", "_")
            image_path = os.path.join(directory, f"id_{idx}_target_{target}_{i}.png")
            image.save(image_path)
            print(f"Saved: {image_path}")
        except Exception as e:
            print(f"Error generating image for row {idx}, prompt {prompt}: {str(e)}")


for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["prompt"]
    if setting == "1":
        target = row["target_singular"].lower()
    elif setting == "2":
        target = row["target_plural"].lower()
    prompt += "50mm photography, hard rim lighting photography --beta --ar 2:3  --beta --upbeta 0.1 --upnoise 0.1 --upalpha 0.1 --upgamma 0.1 --upsteps 20"
    generate_and_save_images(prompt, idx, target)
