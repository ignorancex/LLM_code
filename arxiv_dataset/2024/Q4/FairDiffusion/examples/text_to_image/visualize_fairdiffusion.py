import argparse
import os
import shutil
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from data_handler import FairGenMed
from datasets import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch_fidelity
from train_text_to_image import normalize_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FairDiffusion")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--initial_model", type=str, default=None)
    parser.add_argument("--race_prompt", action="store_true")
    parser.add_argument("--gender_prompt", action="store_true")
    parser.add_argument("--ethnicity_prompt", action="store_true")
    parser.add_argument("--datasets", type=str, default=None, nargs="+")
    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--vis_dir", type=str, default=None, required=True)
    parser.add_argument("--prompts", type=str, default=None, required=True)
    parser.add_argument("--repeat_prompt", type=int, default=None, required=True)
    args = parser.parse_args()
    return args

def visualize_fairdiffusion(args):
    out_dir = args.vis_dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.mkdir(out_dir)
    os.mkdir(f'{out_dir}/actual')
    os.mkdir(f'{out_dir}/generated')
    os.mkdir(f'{out_dir}/prompts')

    unet = UNet2DConditionModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(args.initial_model, unet=unet, torch_dtype=torch.float16)
    pipe.to("cuda")
    print(pipe.text_encoder.text_model.encoder.layers[0].mlp.fc2.weight[:3,:3].tolist())
    print(unet.mid_block.resnets[0].conv1.weight[:3,:3,0,0].tolist())

    actual_imgs = []
    generated_imgs = []
    with open(args.prompts, 'r') as file:
        all_prompts = file.readlines()

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation((-90, -90)),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ]
    )

    # for idx, item in enumerate(dataset):
    for idx, prompt in enumerate(all_prompts):
        for repeat_idx in range(args.repeat_prompt):
            gen_img = pipe(prompt=prompt.strip(), num_inference_steps=75, height=512, width=512, guidance_scale=7.5).images[0]
            gen_img = gen_img.rotate(90)
            gen_img.save(os.path.join(f"{out_dir}/generated", f"{idx+1}_{repeat_idx+1}.jpg"))

    # save all_prompts
    with open(os.path.join(f"{out_dir}/prompts", "prompts.txt"), "w") as f:
        for prompt in all_prompts:
            f.write(prompt)

if __name__ == "__main__":
    args = parse_args()
    visualize_fairdiffusion(args)