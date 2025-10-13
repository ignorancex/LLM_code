import argparse
import torch
import os
import json
from tqdm import tqdm
import importlib
import pandas as pd
import numpy as np
import time
import requests
from  typing import List
import shutil
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, LlavaForConditionalGeneration

from collections import OrderedDict
import base64
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, AutoModel
from qwen_vl_utils import process_vision_info
from models.base import Mllm
import torch
from tqdm import tqdm
# from utils import RESPONSE_DICT
import random
import jsonlines
import open_clip
import cv2 as cv
import numpy as np
import traceback

from models.llava.conversation import conv_templates
from models.llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token
)
from models.llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info


import base64
from openai import OpenAI
# from utils import evaluate_on_mmvet, encode_image



api = "xxxxxxxxxxxxxxxxx"
client = OpenAI(
    base_url = "xxxxxxxxxxxxxxxxxxxxxxxx,
    api_key= api 
)





def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



class GPT4o:
    def __init__(self, engine="gpt-4o", temperature=0, sleep_time=10) -> None:
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time
        self.model = engine

    def evaluate(self,filepath, prompt="What's in this image?"):


        image_path = filepath

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature # 自行修改温度等参数
        )
        # print(response)
        output = response.choices[0].message.content

        return output

class o4mini:
    def __init__(self, api_key, engine="o4-mini", temperature=0, sleep_time=10,image_detail="auto",
                 system_text="You are a helpful assistant. Generate a short and concise response to the following image text pair."):
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time
        self.image_detail = image_detail
        self.system_text = system_text


    def get_response(self, image_path, prompt="What's in this image?"):

        base64_image = encode_image(image_path)
        # image_format = "data:image/png;base64" if 'png' in image_path else "data:image/jpeg;base64"
        # messages = []

        # content = [
        #     {
        #         "text": prompt,
        #     },
        #     {
        #         "image": f"file://{image_path}"
        #     }
        # ]


        # image_path = filepath

        # base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature # 自行修改温度等参数
        )
        # print(response)
        output = response.choices[0].message.content

        return output


class claude4:
    def __init__(self, api_key, engine="gpt-4o", temperature=0, sleep_time=10) -> None:
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time


    def evaluate(self, prompt, filepath):


        image_path = filepath

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature # 自行修改温度等参数
        )
        # print(response)
        output = response.choices[0].message.content

        return output


class grok:
    def __init__(self, api_key, engine="gpt-4o", temperature=0, sleep_time=10) -> None:
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time


    def evaluate(self, prompt, filepath):


        image_path = filepath

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="grok-3",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature # 自行修改温度等参数
        )
        # print(response)
        output = response.choices[0].message.content

        return output



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







default_generation_config = {
    "do_sample": False, 
    "max_new_tokens": 128, 
    "top_p": 0, 
    "temperature": 0.0
}

from eval_utils  import evaluate_vlsbench_function

arch_to_module = {
    "openai": "load_openai",
    "llava": "load_llava",
    "llava_hf": "load_llava_hf",
    "llava_next": "load_llava_next",
    "mllama": "load_mllama",
    'qwen2vl': 'load_qwen2vl',
    'qwen2.5_vl': 'load_qwen2.5vl',
}

# print(f"[INFO] Load qwenvl2.5forconditionalgeneration")
#########QWEN

# model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16,trust_remote_code=True, device_map="auto")
# processor = AutoProcessor.from_pretrained(model_path)

####InterVL

# model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()  
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)



####YIVL
import traceback

from models.llava.conversation import conv_templates
from models.llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token
)
from models.llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info

from models.base import Mllm

class YIVL(Mllm):
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        try:
            model_path = os.path.expanduser(model_name_or_path)
            key_info['model_path'] = model_path
            get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path
            )
            self.model = self.model.to(dtype = torch.bfloat16)
        except Exception as e:
            traceback.print_exc()
        
    def _disable_torch_init(self):
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
    def evaluate(self, prompt, filepath):
        self._disable_torch_init()
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv = conv_templates['mm_default'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = (
            tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0)
            .cuda()
        )
        
        image = Image.open(filepath)
        if getattr(self.model.config, 'image_aspect_ratio', None) == 'pad':
            image = expand2square(
                image,
                tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
        image_tensor = self.image_processor.preprocess(
            image,
            return_tensors = 'pt'
        )['pixel_values'][0]
        
        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images = image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample = False,
                temperature = 0.1,
                num_beams = 1,
                stopping_criteria = [stopping_criteria],
                max_new_tokens = 128,
                use_cache = True
            )
        
        input_token_len = input_ids.shape[1]
        
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]
        ).sum().item()
        if n_diff_input_output != 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input ids!')
        
        response = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens = True
        )[0]
        response = response.strip()
        
        if response.endswith(stop_str):
            response = response[: -len(stop_str)]
            
        response = response.strip()
        
        return response

###########################llava
# class Llava(Mllm):
    
#     def __init__(self, model_name_or_path, **kwargs):
#         self.model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
#         self.processor = AutoProcessor.from_pretrained(model_name_or_path)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
        
#     def evaluate(self, prompt, filepath):
#         image = Image.open(filepath)
#         prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
#         inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.device)
#         generate_ids = self.model.generate(**inputs, max_length=1024)
#         output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         # post process
#         replace_text = 'ASSISTANT: '
#         output = output[output.find(replace_text) + len(replace_text):]
#         return output

#################deepseekvl##########################
# import torch
# from models.deepseek_vl.models.modeling_vlm import MultiModalityCausalLM

# from models.deepseek_vl.models import VLChatProcessor
# from models.deepseek_vl.utils.io import load_pil_images

# from models.base import Mllm


# class DeepSeek(Mllm):
#     def __init__(self, model_name_or_path, *args, **kwargs) -> None:
#         super().__init__(model_name_or_path, *args, **kwargs)
#         self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(
#             model_name_or_path
#         )
#         self.tokenizer = self.processor.tokenizer
#         from transformers import AutoModelForCausalLM
#         self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
#             model_name_or_path,
#             trust_remote_code = True
#         ).to(torch.bfloat16).cuda().eval()
        
#     def evaluate(self, prompt, filepath):
#         conversation = [
#             {
#                 'role': 'User',
#                 'content': f'<image_placeholder>{prompt}',
#                 'images': [filepath]
#             },
#             {
#                 'role': 'Assistant',
#                 'content': ''
#             }
#         ]
#         pil_images = load_pil_images(conversation)
#         prepare_inputs = self.processor(
#             conversations = conversation,
#             images = pil_images,
#             force_batchify = True
#         ).to(self.model.device)
#         inputs_embeds = self.model.prepare_inputs_embeds(
#             **prepare_inputs
#         )
#         outputs = self.model.language_model.generate(
#             inputs_embeds = inputs_embeds,
#             attention_mask = prepare_inputs.attention_mask,
#             pad_token_id = self.tokenizer.eos_token_id,
#             bos_token_id = self.tokenizer.bos_token_id,
#             eos_token_id = self.tokenizer.eos_token_id,
#             max_new_tokens = 128,
#             do_sample = False,
#             use_cache = True 
#         )
#         response = self.tokenizer.decode(
#             outputs[0].cpu().tolist(),
#             skip_special_tokens = True
#         )
#         return response



# model = DeepSeek(model_path)
# model inference
def model_inference(question, image_path) -> str:
    # load model
   
##########QWEN
    # conversations = [    # qwen default a system message You are a helpful assistant.
    #     { 
    #         "role": "user",
    #         "content": 
    #         [
    #             {"type": "image", "image": image_path},
    #             {"type": "text", "text": question},
    #         ],
    #     },
    # ]
    # texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info(conversations)
    # inputs = processor(
    #     text=texts,
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")
    # output_ids = model.generate(**inputs, **default_generation_config)
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    # response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
####InternVL
    # pixel_values = load_image(image_path, max_num=12).to(torch.float16).cuda()
    # generation_config = dict(max_new_tokens=1024, do_sample=True)

    # response = model.chat(tokenizer, pixel_values, question, generation_config)

############YIVL
    # response =   model.evaluate(question,image_path)

# #############llava
#     response =   model.evaluate(question,image_path)
    return response



# JSON 文件路径，根据实际路径修改
json_file_path = "/Datasets/mmbench/original_data/new.json"  


    # 打开 JSON 文件并加载数据
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
        
# print(len(data))
# exit(0)
# model_path = 

# model_path = 
########QWEN
# model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16,trust_remote_code=True, device_map="auto")
# processor = AutoProcessor.from_pretrained(model_path)


# model = YIVL(model_path)
model = GPT4o(engine="gpt-4o")



default_generation_config = {
    "do_sample": False, 
    "max_new_tokens": 128, 
    "top_p": 1.0, 
    "temperature": 0.0
}

result_json_path = "/Datasets/mmbench/original_data/Qwen2.5-VL-7B-Instruct.json"  
result_list = []


for item in tqdm(data, desc="处理数据", unit="条"):
    
    # print(item)
     # 访问问题、答案和选项
    question = item['question']
    answer = item['answer']
    hint = item['hint']
    options = item['options']
    options_str = "\n".join(options)
    image = item['image_path']
    # print(image)

    full_prompt = f'''Hint: {hint}
Question: {question}
Options: {options_str }
Please select the correct answer from the options above.'''
    
# ##########QWEN
    # conversations = [    # qwen default a system message You are a helpful assistant.
    #     { 
    #         "role": "user",
    #         "content": 
    #         [
    #             {"type": "image", "image": image},
    #             {"type": "text", "text": question},
    #         ],
    #     },
    # ]
    # texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info(conversations)
    # inputs = processor(
    #     text=texts,
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")
    # output_ids = model.generate(**inputs, **default_generation_config)
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    # response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

###########################YIVL
    response = model.evaluate(image, full_prompt)
    # evaluate
    print(response)

    result_dict = {
            "question": question,
            "image_path": image,
            "answer": answer,
            "response": response
        }
    result_list.append(result_dict)

    print(result_dict)

if result_list:
    with open(result_json_path, 'w', encoding='utf-8') as result_file:
        json.dump(result_list, result_file, ensure_ascii=False, indent=4)
    print(f"结果已保存到 {result_json_path}")
else:
    print("未获取到有效结果，未生成结果 JSON 文件。")

