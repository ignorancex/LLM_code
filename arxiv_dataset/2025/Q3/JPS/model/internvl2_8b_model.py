import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .base_model import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from PIL import Image
import json
import logging

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

class Internvl28BModel(BaseModel):
    def __init__(self, config):
        self.config = config
        path = config['model']['model_path']
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        self.pipeline={
            'model': self.model,
            'tokenizer': self.tokenizer
        }

        self.image_type = config['data_loader']['image_type']
        self.inference_save_dir = config['inference_config']['inference_save_dir']

    def test(self, data, adv_img_path, steer_prompt = '', save = True):
        image_height = self.config['data_loader']['image_height']
        image_width = self.config['data_loader']['image_width']

        if self.config['attack_config']['attack_type'] == 'patch':
            patch_height = self.config['attack_config']['patch_size']['height']
            patch_width = self.config['attack_config']['patch_size']['width']

        type_list = [data_dict['type'] for data_dict in data]

        if self.image_type == 'blank':
            question_list = [steer_prompt + data_dict['origin_question'] for data_dict in data]
            ori_image_list = [torch.zeros((1, 3, image_height, image_width),device = self.model.device, dtype=self.model.dtype)] * len(question_list)
        elif self.image_type == 'related_image':
            question_list = [steer_prompt + data_dict['rephrased_question'] for data_dict in data]
            ori_image_list = [data_dict['image'] for data_dict in data]

        if self.config['attack_config']['attack_type'] in ['full', 'grid']:
            adv_image = self.load_image(adv_img_path, input_size=image_width,resize_shape=(image_width,image_height))
        elif self.config['attack_config']['attack_type'] == 'patch':
            adv_image = self.load_image(adv_img_path, input_size=patch_width,resize_shape=(patch_width,patch_height))
        else:
            adv_image = self.load_image(Image.new('RGB', (self.config['data_loader']['image_width'], self.config['data_loader']['image_height']), (255, 255, 255)))
       
        image_list = []
        for id in range(len(ori_image_list)):
            if self.config['attack_config']['attack_type'] == 'full':    
                normalized_min = [(0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
                normalized_max = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

                min_values = torch.tensor(normalized_min, device = adv_image.device, dtype = adv_image.dtype)
                max_values = torch.tensor(normalized_max, device = adv_image.device, dtype = adv_image.dtype)

                min_values = min_values.view(1, 3, 1, 1)
                max_values = max_values.view(1, 3, 1, 1)

                image = torch.clamp(ori_image_list[id] + adv_image, min_values, max_values) 

            elif self.config['attack_config']['attack_type'] == 'patch':
                patch_height = self.config['attack_config']['patch_size']['height']
                patch_width = self.config['attack_config']['patch_size']['width']
                image = ori_image_list[id]
                image[:, :, :patch_height, :patch_width] = adv_image

            elif self.config['attack_config']['attack_type'] == 'grid':
                image=adv_image
            
            elif self.config['attack_config']['attack_type'] == 'text_only':
                image=None

            elif self.config['attack_config']['attack_type'] == 'white':
                image=adv_image
            
            elif self.config['attack_config']['attack_type'] == 'related_image':
                image=ori_image_list[id]

            image_list.append(image)

        if self.config['attack_config']['attack_type'] == 'text_only':
            image_list = None
            
        outputs = self.inference(question_list, image_list, max_new_tokens=self.config['inference_config']['max_new_tokens'], batch_size=self.config['inference_config']['batch_size'])

        assert len(question_list) == len(outputs) == len(type_list), "Length of questions, outputs, and type_list must be the same"

        if save is False:
            return outputs, type_list
        
        relative_path = os.path.relpath(adv_img_path, self.config['attack_config']['adv_save_dir'])
        filename = os.path.splitext(relative_path)[0]
        result_file = os.path.join(self.inference_save_dir, filename+ '.json')
        parts = result_file.split('/')
        parts[-3] = self.config['data_loader']['dataset_name']
        result_file = '/'.join(parts)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        results = [{"question": q, "answer": a, "type": t} for q, a, t in zip(question_list, outputs, type_list)]

        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)        

        return result_file

    def inference(self, texts, images, max_new_tokens=1024, batch_size=5):
        assert isinstance(texts, list), "prompts should be a list"

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

        if images is None:
            responses = []
            for i in tqdm(range(0, len(texts), batch_size),file=self.config['tqdm_out']):  # Add tqdm here
                batch_texts = texts[i:i+batch_size]
                batch_responses = self.model.batch_chat(self.tokenizer, None,
                    num_patches_list=[1] * len(batch_texts),
                    questions=batch_texts,
                    generation_config=generation_config)
                responses.extend(batch_responses)

            return responses
        
        assert isinstance(images, list), "prompts should be a list"
        assert (len(texts) == len(images))

        responses = []
        for i in tqdm(range(0, len(texts), batch_size),file=self.config['tqdm_out']):  # Add tqdm here
            batch_texts = texts[i:i+batch_size]
            batch_images = images[i:i+batch_size]
            pixel_values = torch.cat(batch_images, dim=0)
            num_patches_list = [x.size(0) for x in batch_images]
            batch_responses = self.model.batch_chat(self.tokenizer, pixel_values,
                    num_patches_list=num_patches_list,
                    questions=batch_texts,
                    generation_config=generation_config)
            responses.extend(batch_responses)

        return responses

    def load_image(self, image_file, input_size=448, max_num=12, resize_shape = None):
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = image_file.convert('RGB')

        if resize_shape is not None:  
            image = image.resize(resize_shape)

        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
        return pixel_values