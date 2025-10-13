# import os
# from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
# from lmdeploy.vl import load_image
# import torch

######################################################################################
# inference with lmdeploy
# class Skywork_R1V:
#     def __init__(self, args):
#         engine_config = TurbomindEngineConfig(cache_max_entry_count=0.75) 
#         chat_template_config = ChatTemplateConfig(model_name=args.model_path)
#         pipe = pipeline(args.model_path, 
#                         backend_config=engine_config, 
#                         chat_template_config=chat_template_config,
#                     )
#         self.model = pipe
#         self.args = args

#     def inference(self, prompt_batch, image_path_batch):

#         image_batch = [load_image(image_path) for image_path in image_path_batch]
#         for prompt in prompt_batch:
#             prompt = prompt.split("<multiple choice question end>")[0]
#             prompt += "\n\nReason step by step and put your final answer in \\boxed{}."
#         model_input_prompts = [(prompt, image) for prompt, image in zip(prompt_batch, image_batch)]
#         with torch.no_grad():
#             response_batch = self.model(
#                 model_input_prompts,
#                 gen_config=GenerationConfig(
#                 max_new_tokens=64000,
#                 do_sample=True,
#                 temperature=0.6,
#                 top_p=0.95,
#                 repetition_penalty=1.05,
#                 )
#                 )

#         response_batch = [response.text.strip() for response in response_batch]
        
#         return response_batch


######################################################################################
# inference with transformers

import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig

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
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
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

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
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
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map


import torch
from transformers import AutoModel, AutoTokenizer

class Skywork_R1V:
    def __init__(self, args):
        # Split the model across devices
        device_map = split_model(args.model_path)
        
        # Load the model
        self.model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            trust_remote_code=True, 
            use_fast=False,
        )
        self.args = args

    def inference(self, prompt_batch, image_path_batch):
        # Load and preprocess images
        pixel_values_list = [load_image(image_path, max_num=12).to(torch.bfloat16).cuda() for image_path in image_path_batch]
        image_counts = [pixel_values.size(0) for pixel_values in pixel_values_list]
        pixel_values = torch.cat(pixel_values_list, dim=0)

        # Prepare prompts
        questions = prompt_batch

        # Generation configuration
        generation_config = dict(
            max_new_tokens=64000,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Batch chat inference
        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=image_counts,
            questions=questions,
            generation_config=generation_config
        )

        # free memory
        del pixel_values
        torch.cuda.empty_cache()

        return [response.strip() for response in responses]






















##############################################################################
# internvl interface
# import torch
# from transformers import AutoModel, AutoTokenizer


# class Skywork_R1V:
#     def __init__(self, args):
#         # Split the model across devices
#         device_map = split_model(args.model_path)
        
#         # Load the model
#         self.model = AutoModel.from_pretrained(
#             args.model_path,
#             torch_dtype=torch.bfloat16,
#             load_in_8bit=False,
#             low_cpu_mem_usage=True,
#             use_flash_attn=True,
#             trust_remote_code=True,
#             device_map=device_map
#         ).eval()
        
#         # Ensure the model is on the correct device
#         # self.model = self.model.cuda()

#         # Load the tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             args.model_path, 
#             trust_remote_code=True, 
#             use_fast=False,
#         )
#         self.args = args

#     def inference(self, prompt_batch, image_path_batch):

#         # Prepare prompts
#         response_batch = []
#         for prompt, image_path in zip(prompt_batch, image_path_batch):
#             pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
#             formatted_prompt = prompt
#             generation_config = dict(
#                 max_new_tokens=64000,
#                 do_sample=True,
#                 temperature=0.6,
#                 top_p=0.95,
#                 repetition_penalty=1.05,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
            
#             # Generate response
#             response = self.model.chat(
#                 self.tokenizer,
#                 pixel_values,
#                 formatted_prompt,
#                 generation_config
#             )
#             response_batch.append(response.strip())
        
#         return response_batch


#########################################################
import json
def parse_reasoning_response(res_path, parsed_path, error_path):
    with open(res_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    parsed = []
    error = []

    for item in res:
        response = item["response"] # \\boxed{A
        try: 
            Answer = response.split("boxed{")[1].split("}")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A. ") or Answer.startswith("B. ") or Answer.startswith("C. ") or Answer.startswith("D. "):
                    Answer = Answer[0]
                else:
                    error.append(item)
                    print(Answer)
            item["response"] = Answer
            item["reasoning"] = response
            parsed.append(item)
        except:
            error.append(item)
            print(response)
            item["response"] = "ERROR"
            item["reasoning"] = response
            parsed.append(item)

    with open(error_path, 'w') as f:
        json.dump(error, f, indent=2)
    with open(parsed_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    with open(parsed_path+'l', 'w') as f:
        for item in parsed:
            f.write(json.dumps(item) + "\n")
            f.flush()
    print(f"{len(error)} errors.")