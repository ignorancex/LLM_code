import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .base_model import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from PIL import Image
import json
import logging
from qwen_vl_utils import process_vision_info

class Qwen2VL7BModel(BaseModel):
    def __init__(self, config):
        self.config = config
        path = config['model']['model_path']
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        # attn_implementation="eager",
                        device_map="auto",
                    ).eval().cuda()
        
        self.processor = AutoProcessor.from_pretrained(path)

        self.pipeline={
            'model': self.model,
            'processor': self.processor
        }

        self.image_type = config['data_loader']['image_type']
        self.inference_save_dir = config['inference_config']['inference_save_dir']

    def test(self, data, adv_img_path, steer_prompt = '', save = True):
        image_height = self.config['data_loader']['image_height']
        image_width = self.config['data_loader']['image_width']

        if self.config['attack_config']['attack_type'] == 'patch':
            patch_height = self.config['attack_config']['patch_size']['height']
            patch_width = self.config['attack_config']['patch_size']['width']

        if self.image_type == 'related_image' and self.config['attack_config']['attack_type'] == 'full':
            raise ValueError("Related image attack is not supported for full attack")

        if self.config['attack_config']['attack_type'] == 'patch':
            raise ValueError("Patch attack is not supported for Qwen2VL7B model")

        type_list = [data_dict['type'] for data_dict in data]

        if self.image_type == 'blank':
            question_list = [steer_prompt + data_dict['origin_question'] for data_dict in data]
        elif self.image_type == 'related_image':
            question_list = [steer_prompt + data_dict['rephrased_question'] for data_dict in data]
            ori_image_list = [data_dict['image'] for data_dict in data]

        if self.config['attack_config']['attack_type'] in ['full', 'grid']:
            adv_image = Image.open(adv_img_path).convert('RGB').resize((image_width, image_height))
        else:
            adv_image = Image.new('RGB', (self.config['data_loader']['image_width'], self.config['data_loader']['image_height']), (255, 255, 255))
       
        image_list = []
        for id in range(len(question_list)):
            if self.config['attack_config']['attack_type'] in ['full', 'grid']:    
                image=adv_image

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
            messages = [self.get_prompt(text) for text in texts]
        else:
            assert isinstance(images, list), "prompts should be a list"
            assert (len(texts) == len(images))
            messages = [self.get_prompt(text, image) for text,image in zip(texts, images)]

        responses = []
        for i in tqdm(range(0, len(messages), batch_size), file=self.config['tqdm_out']):
            batch_messages = messages[i:i + batch_size]
            
            batch_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
    
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = self.processor(
                text=batch_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            generated_ids = self.model.generate(**inputs, **generation_config)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            responses.extend(batch_output_texts)

        return responses

    def get_prompt(self,input_text, image=None):
        if image == None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ],
                }
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": input_text},
                    ],
                }
            ]
        
        return conversation