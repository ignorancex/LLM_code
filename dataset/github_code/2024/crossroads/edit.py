# first install groundingdino and torchmetrics in your virtual environment like this
# git clone https://github.com/IDEA-Research/GroundingDINO.git
# cd GroundingDINO/
# pip install -e .
# pip install --upgrade diffusers[torch] (for stable diffusion)
# pip install torchmetrics --upgrade (for CLIP based metrics)
# input parameters are in edit_config.json
# run this script with python edit.py

import argparse
from functools import partial
import cv2
import requests
import json
import os

from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path


import warnings
warnings.filterwarnings("ignore")


import torch
from torchvision.ops import box_convert

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download

import os
import supervision as sv

from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler, AutoencoderKL

import torch
_ = torch.manual_seed(42)
from torchmetrics.functional.multimodal import clip_score

import random

def generate_masks_with_grounding(image_source, boxes):
    h, w, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image_source)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box
        mask[int(y0):int(y1), int(x0):int(x1), :] = 255
    return mask

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"


model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.vae = vae
pipe = pipe.to("cuda")

with open("edit_config.json", "r") as f:
    config = json.load(f)
DINO_PROMPT = config["DINO_PROMPT"]
BOX_TRESHOLD = config["BOX_TRESHOLD"]
TEXT_TRESHOLD = config["TEXT_TRESHOLD"]
local_image_path = config["LOCAL_IMAGE_PATH"]
CATEGORY = config["CATEGORY"]
STABLE_DIFF_PROMPT = config["STABLE_DIFF_PROMPT"]
country1 = config["COUNTRY_1"]
#assert country1 is a substring of local_image_path string
assert country1 in local_image_path, f"country1 {country1} is not in local_image_path {local_image_path}"
country2 = config["COUNTRY_2"]

image_source, image = load_image(local_image_path)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=DINO_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
annotated_frame = annotated_frame[...,::-1]

image_mask = generate_masks_with_grounding(image_source, boxes)

annotated_image = Image.fromarray(annotated_frame)

OUTPUT_PATH = f"results/edits/dino/{country1}/{CATEGORY}/annotated_image.jpg"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
annotated_image.save(OUTPUT_PATH)

image_source = Image.fromarray(image_source)
annotated_frame = Image.fromarray(annotated_frame)
image_mask = Image.fromarray(image_mask)
image_source_for_inpaint = image_source.resize((512, 512))
image_mask_for_inpaint = image_mask.resize((512, 512))
prompt = STABLE_DIFF_PROMPT
prompt += "intricate details. 4k. high resolution. high quality."
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
generator = torch.Generator("cuda").manual_seed(random.randrange(0, 100000))
image_inpainting = pipe(prompt=prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint, generator=generator).images[0]
image_inpainting = image_inpainting.resize((image_source.size[0], image_source.size[1]))

OUTPUT_PATH_2 = f"results/edits/sd/{country1}/{CATEGORY}/inpainting_image_{country2}.jpg"
os.makedirs(os.path.dirname(OUTPUT_PATH_2), exist_ok=True)
image_inpainting.save(OUTPUT_PATH_2)


image_source_tensor = np.array(image_source)
image_inpainting_tensor = np.array(image_inpainting)
image_source_tensor = torch.from_numpy(image_source_tensor).permute(2, 0, 1).float()
image_inpainting_tensor = torch.from_numpy(image_inpainting_tensor).permute(2, 0, 1).float()
i1_country1 = clip_score(image_source_tensor, country1, "openai/clip-vit-base-patch16")
i1_country2 = clip_score(image_source_tensor, country2, "openai/clip-vit-base-patch16")
i2_country1 = clip_score(image_inpainting_tensor, country1, "openai/clip-vit-base-patch16")
i2_country2 = clip_score(image_inpainting_tensor, country2, "openai/clip-vit-base-patch16")
delta1 = i2_country1 - i1_country1
delta2 = i2_country2 - i1_country2
# ideal case:
# if delta1 is negative, then i2 is less of country1 than i1
# if delta2 is positive, then i2 is more of country2 than i1
print("-"*80)
print("I1_country1: ", i1_country1)
print("I1_country2: ", i1_country2)
print("I2_country1: ", i2_country1)
print("I2_country2: ", i2_country2)
print("Delta1: ", delta1)
print("Delta2: ", delta2)
print("-"*80)
