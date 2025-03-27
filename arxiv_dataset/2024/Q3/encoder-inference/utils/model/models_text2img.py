from typing import Iterable
import torch

from diffusers import DiffusionPipeline


def _build_diff(name: str, device: str, num_img):
    pipe = DiffusionPipeline.from_pretrained(name, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    def generator(prompt: Iterable):
        return pipe(prompt, num_images_per_prompt=num_img).images

    return generator


def sd_v1_2(device: str, num_img: int):
    return _build_diff("CompVis/stable-diffusion-v1-2", device, num_img)


def sd_v1_4(device: str, num_img: int):
    return _build_diff("CompVis/stable-diffusion-v1-4", device, num_img)


def sd_v2_1_base(device: str, num_img: int):
    return _build_diff("stabilityai/stable-diffusion-2-1-base", device, num_img)


def sd_v2_1(device: str, num_img: int):
    return _build_diff("stabilityai/stable-diffusion-2-1", device, num_img)


__model_zoo__ = {
    "sd-v1-2": sd_v1_2,
    "sd-v1-4": sd_v1_4,
    "sd-v2-1-base": sd_v2_1_base,
    "sd-v2-1": sd_v2_1,
}


def get_text2img_model(name: str, device: str = "cuda:0", num_img: int = 4):
    return __model_zoo__[name](device, num_img)
