
import argparse
import os

import numpy as np
import torch
import torch.utils.checkpoint
import torchvision.transforms as transforms
from PIL import Image
from diffusers import AutoencoderKL
from diffusers import (
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from mgface.pipelines_mgface.pipeline_mgface import MgPipeline as MgPipelineInference
from mgface.pipelines_mgface.unet_ID_2d_condition import UNetID2DConditionModel
from mgface.pipelines_mgface.unet_deno_2d_condition import UNetDeno2DConditionModel

# AU mapping
ind_dict = {'AU1':0, 'AU2':1, 'AU4':2, 'AU5':3, 'AU6':4, 'AU9':5,
            'AU12':6, 'AU15':7, 'AU17':8, 'AU20':9, 'AU25':10, 'AU26':11}

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a MagicFace test script.")
    # /home/mengting/Desktop/diffusion_models/stable-diffusion-v1-5
    # sd-legacy/stable-diffusion-v1-5
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='sd-legacy/stable-diffusion-v1-5',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument("--seed", type=int, default=424,
                        help="A seed for reproducible training.")

    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--denoising_unet_path",
        type=str,
        default='mengtingwei/magicface',
    )
    
    parser.add_argument(
        "--ID_unet_path",
        type=str,
        default='mengtingwei/magicface',
    )

    parser.add_argument(
        "--au_test",
        type=str,
        default='',
    )

    parser.add_argument(
        "--AU_variation",
        type=str,
        default='',
    )

    parser.add_argument(
        "--img_path",
        type=str,
        default='',
    )

    parser.add_argument(
        "--bg_path",
        type=str,
        default='',
    )

    parser.add_argument(
        "--saved_path",
        type=str,
        default='edited_images',
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def make_data(args):

    transform = transforms.ToTensor()

    img_name = args.img_path
    bg_name = args.bg_path

    source = Image.open(img_name)
    source = transform(source)

    bg = Image.open(bg_name)
    bg = transform(bg)

    return source, bg


def tokenize_captions(tokenizer, captions, max_length):

    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs.input_ids


def main(args):

    device = 'cuda'
    denoising_unet_path = args.denoising_unet_path
    ID_unet_path = args.ID_unet_path

    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            cache_dir='./'
        ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
        cache_dir='./'
        ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
        cache_dir='./'
        )

    unet_ID = UNetID2DConditionModel.from_pretrained(
            ID_unet_path,
            subfolder='ID_enc',
            # torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            cache_dir='./',
        )

    # 
    unet_deno = UNetDeno2DConditionModel.from_pretrained(
            denoising_unet_path,
            subfolder='denoising_unet',
            # torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        cache_dir='./',
        )

    unet_deno.requires_grad_(False)
    unet_ID.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    

    weight_dtype = torch.float16


    pipeline = MgPipelineInference.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet_ID=unet_ID,
        unet_deno=unet_deno,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(device)
    
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    source, bg = make_data(args)
    prompt = 'A close up of a person.'
    source = source.unsqueeze(0)
    bg = bg.unsqueeze(0)
    
    prompt_embeds = text_encoder(tokenize_captions(tokenizer, [prompt], 2).to(device))[0]
    au_prompt = np.zeros((12,))
    au_test_file = args.au_test
    AU_variation = args.AU_variation

    if '+' not in au_test_file:
        print('you are testing editing with a single AU')
        tgt_au_ind = au_test_file
        au_change = int(AU_variation)
        au_prompt[ind_dict[tgt_au_ind]] = au_change
    else:
        print('you are testing editing with AU combinations')
        au_test_file = au_test_file.split('+')
        AU_variation = AU_variation.split('+')

        for item1, item2 in zip(au_test_file, AU_variation):
            tgt_au_ind = item1
            au_prompt[ind_dict[tgt_au_ind]] = item2
    
    print(au_prompt)

    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    img_name = args.img_path.split('/')[-1]

    tor_exp = torch.from_numpy(au_prompt).unsqueeze(0)
    samples = pipeline(
        prompt_embeds=prompt_embeds,
        source=source,
        bg = bg,
        au=tor_exp,
        num_inference_steps=args.inference_steps,
        generator=generator,
    ).images[0]
    samples.save(os.path.join(saved_path, img_name))
    print('done')
    # exps = np.load(os.path.join('./test_aus/test_relative_aus', au_test_file))
    # saved_path = os.path.join('./test_out/test_out_only_wild_cartoon2', au_test_file.replace('.npy', ''))
    # os.makedirs(saved_path, exist_ok=True)
    # for i, exp in enumerate(exps):
    #
    #     tor_exp = torch.from_numpy(exp)
    #     tor_exp = tor_exp.unsqueeze(0)
    #     samples = pipeline(
    #         prompt_embeds=prompt_embeds,
    #         source=source,
    #         bg = bg,
    #         au=tor_exp,
    #         num_inference_steps=args.inference_steps,
    #         generator=generator,
    #     ).images[0]
    #     samples.save(os.path.join(saved_path, f'{i:03}.jpg'))
    # print('done')

if __name__ == "__main__":
    args = parse_args()

    main(args)
