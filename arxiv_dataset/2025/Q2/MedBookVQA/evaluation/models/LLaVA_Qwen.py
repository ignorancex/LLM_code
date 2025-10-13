from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import requests
import copy
import torch
import warnings


class LLaVA_Qwen:
    def __init__(self, args):
        self.args = args
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.model_path, 
            None, 
            "llava_qwen", 
            device_map="auto"
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.max_length = max_length
    def inference(self, prompt_batch, image_path_batch):
        output_batch = []
        for image_path, prompt in zip(image_path_batch, prompt_batch):
            image = Image.open(image_path)
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]

            conv_template = "qwen_1_5"
            question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
            image_sizes = [image.size]
            cont = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=16384,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            output_batch.append(text_outputs[0])
        return output_batch

