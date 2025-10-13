# Use the modified diffusers & peft library
import sys
import os
# workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icedit"))

# if workspace_dir not in sys.path:
#     sys.path.insert(0, workspace_dir)
    
from diffusers import FluxFillPipeline

# Below is the original library
import torch
from PIL import Image
import numpy as np
import argparse
import random
import tqdm  
def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='gpt4v', help="specifies the model to be evaluated.")
    parser.add_argument('--data_path', type=str, default='/Datasets/vlb/new.jsonl', help="specifies the path to the data")
    parser.add_argument('--save_path', type=str, default='/Datasets/vlb', help='specifies the path to save the results.')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--flux-path", type=str, default='/model/icedit/flux', help="Path to the model")
    parser.add_argument("--lora-path", type=str, default='/model/icedit/lora', help="Path to the LoRA weights")
    parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")

    args = parser.parse_args()

    # for arg in vars(args):
    #     logging.info(f"{arg}: {getattr(args, arg)}")
    return args
import jsonlines


import json

def load_jsonl_file(file_path):
    """
    读取JSONL文件并返回解析后的字典列表
    
    参数:
        file_path: JSONL文件路径
        
    返回:
        包含所有JSON对象的列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                # 去除行首尾的空白字符
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    # 逐行解析JSON
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"解析第{line_num}行时出错: {e}")
        return data
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None



def load_data(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in tqdm(reader, desc="Loading data..."):
            data.append(line)
    # import json
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     data = json.load(file)

        return data
def process_data_json(data_path):
    # category = os.path.basename(data_path)+'.jsonl'
    # json_path = os.path.join(data_path,category)
    # print(json_path)

    data = load_data(data_path)
    # n = data.shape[0]
    # cnt = 0
    # print(data[0])
    # exit(0)
    return data


def main(args):
    
    data = load_jsonl_file(args.data_path)

    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.lora_path)
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload() 
    else:
        pipe = pipe.to("cuda")

    for index, sample in enumerate(data ):
        caption = sample["caption"]
        image_url = sample["img_url"]
        image = Image.open(image_url )
        image = image.convert("RGB")
        # print(caption)
        # exit(0)
        if image.size[0] != 512:
            print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
            new_width = 512
            scale = new_width / image.size[0]
            new_height = int(image.size[1] * scale)
            new_height = (new_height // 8) * 8  
            image = image.resize((new_width, new_height))
            print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")
            instruction = caption
            print(f"Instruction: {instruction}")
            
            width, height = image.size
            combined_image = Image.new("RGB", (width * 2, height))
            combined_image.paste(image, (0, 0))
            combined_image.paste(image, (width, 0))
            mask_array = np.zeros((height, width * 2), dtype=np.uint8)
            mask_array[:, width:] = 255 
            mask = Image.fromarray(mask_array)

            result_image = pipe(
                prompt=instruction,
                image=combined_image,
                mask_image=mask,
                height=height,
                width=width * 2,
                guidance_scale=50,
                num_inference_steps=28,
                generator=torch.Generator("cpu").manual_seed(args.seed) if args.seed is not None else None,
            ).images[0]

            result_image = result_image.crop((width,0,width*2,height))

            os.makedirs(args.save_path, exist_ok=True)

            image_name = image_url.split("/")[-1]
            result_image.save(os.path.join(args.save_path, f"{image_name}"))
            print(f"\033[92mResult saved as {os.path.abspath(os.path.join(args.save_path, image_name))}\033[0m")
            # exit(0)
       
if __name__ == "__main__":
    args = get_args()
    main(args)
    seed_all(5555)


