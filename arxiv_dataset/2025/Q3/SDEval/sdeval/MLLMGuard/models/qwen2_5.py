from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from models.base import Mllm
import torch
from tqdm import tqdm
from utils import RESPONSE_DICT
import random
import jsonlines
import open_clip
import cv2 as cv
import numpy as np


class Qwen2_5(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16,trust_remote_code=True, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        # self.model.to(self.device)
    def evaluate(self, prompt, filepath):
        messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": filepath,
            },
            {"type": "text", "text": prompt},
        ],
    }
]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

class Qwen2(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16,trust_remote_code=True, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        # self.model.to(self.device)
    def evaluate(self, prompt, filepath):
        messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": filepath,
            },
            {"type": "text", "text": prompt},
        ],
    }
]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]




class GPT4o(Mllm):
    def __init__(self, api_key, engine="gpt-4o", temperature=0, sleep_time=10) -> None:
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time


    def evaluate(self, prompt, filepath):


        image_path = filepath

        base64_image = encode_image(image_path)

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