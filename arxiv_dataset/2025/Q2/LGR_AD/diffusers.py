import openai
from diffusers import StableDiffusionPipeline
from ldm.models.diffusion.ddpm import LatentDiffusion
from transformers import CLIPTextModel, CLIPTokenizer
import torch


def dalle_generator(prompt, openai_api_key):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

def stable_diffusion_generator(prompt):
    model_id = "stabilityai/stable-diffusion-v2"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe(prompt)["images"][0]
    return image

def ldm_generator(prompt):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_inputs = tokenizer(prompt, return_tensors="pt")
    text_features = text_model(**text_inputs)
    ldm = LatentDiffusion("path/to/ldm/checkpoint")
    return ldm.generate(text_features)
