"""
    This is the script of scaled self-reference customization inference with Calligrapher.
"""

import os
import json
import random
from PIL import Image
import numpy as np
from datetime import datetime

import torch
from diffusers.utils import load_image

from pipeline_calligrapher import CalligrapherPipeline
from models.calligrapher import Calligrapher
from models.transformer_flux_inpainting import FluxTransformer2DModel

from utils import resize_img_and_pad, generate_context_reference_image


def infer_calligrapher(test_image_dir, result_save_dir,
                       target_h=512, target_w=512,
                       gen_num_per_case=2, load_prompt_from_txt=True):
    # Set job dir.
    job_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_save_path = os.path.join(result_save_dir, job_name)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path, exist_ok=True)

    # Load models.
    base_model_path = path_dict['base_model_path']
    image_encoder_path = path_dict['image_encoder_path']
    calligrapher_path = path_dict['calligrapher_path']
    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = CalligrapherPipeline.from_pretrained(base_model_path,
                                                transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
    model = Calligrapher(pipe, image_encoder_path, calligrapher_path,
                         device="cuda", num_tokens=128)

    source_image_names = [i for i in os.listdir(test_image_dir) if 'source.png' in i]
    # Loading prompts from the bench txt and printing them.
    if load_prompt_from_txt:
        prompt_dict = {}
        with open(os.path.join(test_image_dir, 'self_bench.txt'), 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split('-', 1)
                    prompt_dict[int(key)] = value

        i = 0
        print('Printing given prompts...')
        for img_id in sorted(prompt_dict.keys()):
            i += 1
            text = prompt_dict[img_id]
            print(f'Sample #{i}: {img_id} - {text}')

    count = 0
    for source_image_name in sorted(source_image_names):
        count += 1
        source_image_path = os.path.join(test_image_dir, source_image_name)
        reference_image_name = source_image_name.replace('source', 'ref')
        reference_image_path = os.path.join(test_image_dir, reference_image_name)
        mask_image_name = source_image_name.replace('source', 'mask')
        mask_image_path = os.path.join(test_image_dir, mask_image_name)

        print('source_image_path: ', source_image_path)
        print('mask_image_path: ', mask_image_path)
        print('reference_image_path: ', reference_image_path)

        if load_prompt_from_txt:
            img_id = int(source_image_name.split("test")[1].split("_")[0])
            text = prompt_dict[img_id]
            prompt = f"The text is '{text}'."
        else:
            prompt = "The text is 'Generation'."
        print(f'prompt: {prompt}')

        source_image = load_image(source_image_path)
        mask_image = load_image(mask_image_path)
        # Resize source and mask.
        source_image = source_image.resize((target_w, target_h))
        mask_image = mask_image.resize((target_w, target_h), Image.NEAREST)
        mask_np = np.array(mask_image)
        mask_np[mask_np > 0] = 255
        mask_image = Image.fromarray(mask_np.astype(np.uint8))
        source_img_w, source_img_h = source_image.size

        # resize reference to fit CLIP.
        reference_image = Image.open(reference_image_path).convert("RGB")
        reference_image_to_encoder = resize_img_and_pad(reference_image, target_size=[512, 512])

        reference_context = generate_context_reference_image(reference_image, source_img_w)
        # Concat the context image on the top.
        source_with_context = Image.new(source_image.mode, (source_img_w, reference_context.size[1] + source_img_h))
        source_with_context.paste(reference_context, (0, 0))
        source_with_context.paste(source_image, (0, reference_context.size[1]))
        # Concat the 0 mask on the top of the mask image.
        mask_with_context = Image.new(mask_image.mode,
                                      (mask_image.size[0], reference_context.size[1] + mask_image.size[0]), color=0)
        mask_with_context.paste(mask_image, (0, reference_context.size[1]))

        # Identifiers in filename.
        ref_id = reference_image_name.split('_')[0]
        safe_prompt = prompt.replace(" ", "_").replace("'", "").replace(",", "").replace('"', '').replace('?', '')[:50]
        for i in range(gen_num_per_case):
            seed = random.randint(0, 2 ** 32 - 1)
            images = model.generate(
                image=source_with_context,
                mask_image=mask_with_context,
                ref_image=reference_image_to_encoder,
                prompt=prompt,
                scale=1.0,
                num_inference_steps=50,
                width=source_with_context.size[0],
                height=source_with_context.size[1],
                seed=seed,
            )

            index = len(os.listdir(result_save_path))
            output_filename = f"result_{index}_{ref_id}_{safe_prompt}_{seed}.png"

            result_img = images[0]
            result_img_vis = result_img.crop((0, reference_context.size[1], result_img.width, result_img.height))
            result_img_vis.save(os.path.join(result_save_path, output_filename))

            target_size = (source_image.size[0], source_image.size[1])
            vis_img = Image.new('RGB', (source_image.size[0] * 3, source_image.size[1]))
            vis_img.paste(source_image.resize(target_size), (0, 0))
            vis_img.paste(reference_context.resize(target_size), (source_image.size[0], 0))
            vis_img.paste(result_img_vis.resize(target_size), (source_image.size[0] * 2, 0))
            vis_img_save_path = os.path.join(result_save_path, f'vis_{output_filename}'.replace('.png', '.jpg'))
            vis_img.save(vis_img_save_path)
            print(f"Generated images saved to {vis_img_save_path}.")


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'path_dict.json'), 'r') as f:
        path_dict = json.load(f)
    # Set directory paths.
    test_image_dir = path_dict['data_dir']
    result_save_dir = path_dict['cli_save_dir']
    infer_calligrapher(test_image_dir, result_save_dir,
                       target_h=512, target_w=512,
                       gen_num_per_case=2, load_prompt_from_txt=True)
    print('Finished!')
