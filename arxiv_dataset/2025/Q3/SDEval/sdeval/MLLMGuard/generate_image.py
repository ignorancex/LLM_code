"""
This script is used to evaluate MLLMs.
"""
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from models.base import Mllm
import torch
from tqdm import tqdm
from utils import CAPTION_DICT
import random
import jsonlines
import open_clip
import cv2 as cv
import numpy as np
import os
import argparse
import logging
import random
import time
import wandb
import torch


from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline

from utils import (
    process_data, process_data_json
)

import numpy as np
import cv2 as cv

def seed_all(seed = 8888):
    torch.manual_seed(seed)
    random.seed(seed)
    
def evaluate(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

@evaluate
def evaluate_model(model, args, data):
    model.batch_evaluate(args, data)    

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='gpt4v', help="specifies the model to be evaluated.")
    parser.add_argument('--data_path', type=str, default='/Datasets/caption', help="specifies the path to the data")
    parser.add_argument('--save_path', type=str, default='results/toxicity_en.jsonl', help='specifies the path to save the results.')

    args = parser.parse_args()

    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    return args
  

"""
使用 GPT-4 Vision API 处理多张图片和文本
"""
# image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

# images_base64 = [image_to_base64(path) for path in image_paths]
# text_prompt = 'You are now an image identification expert. You will compare the two images I input to see if they are semantically consistent,\
#      and whether there are factors that will fundamentally affect the output results. If so, return Yes, otherwise return No.\
#          Just return one of these two answers, without any other information.'

def process_images_with_gpt4(images_base64, text_prompt):

    # 构建消息列表
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
            ]
        }
    ]

    # 添加图片内容
    for image_base64 in images_base64:
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            }
        )

    # 调用 GPT-4 Vision API
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages
    )

    return response.choices[0].message.content



def generate_images(data_path):
    model_id = "/model/stable_diffsuion_3.5"

  

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    )

    pipeline = pipeline.to("cuda")

   
    save_path = data_path
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith('.jsonl'):
                # 构建完整的文件路径
                file_path = os.path.join(root, filename)
                # print(save_path)
                # print(file_path)
                category = os.path.basename(file_path).split('.')[0]
                
                data = process_data_json(file_path)
                save_path = os.path.join(save_path,category,'images')
                print(save_path)

                exit(0)    
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    print(f"Image folder created at: {save_path}")

                else:
                    continue

                    for sample in tqdm(data):
                        caption = sample['caption']
                        image = sample['img_url']

                        # print(caption)
                        image_new = pipeline(
                                        prompt=caption,
                                        num_inference_steps=28,
                                        guidance_scale=4.5,
                                        max_sequence_length=512,
                                    ).images[0]
                        file_id = os.path.basename(image).split('.')[0] +'.jpg'
                        image_save_path = os.path.join(save_path,file_id)
                        # print(image_save_path)
                        image_new.save(image_save_path)

        

def main(args):
    

    generate_images(args.data_path)
       
if __name__ == "__main__":
    args = get_args()
    main(args)
    seed_all(5555)

