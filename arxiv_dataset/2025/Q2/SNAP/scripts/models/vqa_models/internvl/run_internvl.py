import math
import numpy as np
import os
import sys
import json
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import pickle
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternVL:
# If you set `load_in_8bit=True`, you will need one 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least two 80GB GPUs.
    def __init__(self, model_name, chunk_id=None):
        self.model_name = model_name
        self.path = f'OpenGVLab/{model_name}'

        self.data_path = os.environ['SNAP_DATA_PATH']
        self.input_path = '../../../../annotations/SNAP_VQA.json'
        self.results_path = f'../../../../raw_results/vqa/{model_name}.xlsx'

        os.makedirs('../../../../raw_results/vqa/', exist_ok=True)
        os.makedirs('cache', exist_ok=True)


    def split_model(self):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
            'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[self.model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def setup_model(self):

        device_map = self.split_model()
        self.model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
            cache_dir='weights').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=False)


    def write_pickle(self, data, filename):
        with open(filename, 'wb') as fid:
            pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)

    def run_model(self):

        with open(self.input_path, 'r') as fid:
            data_dict = json.load(fid)

        cache_file = f"cache/{os.path.basename(self.results_path).replace('xlsx', 'pkl')}"

        results = []
        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as fid:
                results = pickle.load(fid)

        processed_images = {x['Path']: 0 for x in results}
        generation_config = dict(max_new_tokens=128, do_sample=True)

        num_images = len(data_dict)
        with torch.no_grad():
            num_processed = 0
            for path, question_dict in tqdm(data_dict.items(), total=num_images):
                if path not in processed_images:
                    img_path = os.path.join(self.data_path, path)
                    # set the max number of tiles in `max_num`
                    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()

                    response_dict = {'Path': path}
                    for i in range(1,8):
                        question = question_dict[f'Q{i}']

                        # single-image single-round conversation
                        question = f'<image>\n{question}'
                        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
                        #print(f'User: {img_path} {question}\nAssistant: {response}')
                        response_dict[f'A{i}'] = response
                    results.append(response_dict)
                    num_processed += 1

                    # save results periodically if the script is interrupted
                if (num_processed > 0) and (num_processed % 100 == 0):
                    self.write_pickle(results, cache_file)
                # print(f'Saved results to {cache_file}')

        self.write_pickle(results, cache_file)
        print(f'Saved results to {cache_file}')

        pd.DataFrame.from_dict(results).to_excel(self.results_path)


# InternVL2-1B
# InternVL2-2B
# InternVL2-4B
# InternVL2-8B
# InternVL2-26B
# InternVL2-40B
# InternVL2-Llama3-76B

if __name__ == '__main__':
    model_name = sys.argv[1]
    chunk_id = None
    if len(sys.argv) > 2:
        # to speed up processing we split data into 4 chunks
        # wuthout passing the argument, the program will run on the entire dataset
        chunk_id = sys.argv[2]
    internVL = InternVL(model_name, chunk_id=chunk_id)
    internVL.setup_model()
    internVL.run_model()