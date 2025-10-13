## move to HealthGPT/llava/DOABbench_test
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from llava.model import *
from llava import conversation as conversation_lib
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token
from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square, com_vision_args
import transformers
from PIL import Image
import os

class HealthGPT:
    def __init__(self, args):
        # find baseModel_path, hlora_path | fusion_layer_path(don't need in eval)
        if "HealthGPT-M3" in args.model_path:
            self.BaseModel_path = os.path.join(args.model_path, "Phi-3-mini-4k-instruct")
            self.vit_path = os.path.join(args.model_path, "clip-vit-large-patch14-336") if not args.HealthGPT_VIS_path else args.HealthGPT_VIS_path
            self.hlora_path = os.path.join(args.model_path, "HealthGPT-M3", "com_hlora_weights.bin")
            self.hlora_r = 64
            self.hlora_alpha = 128
            self.instruct_template = 'phi3_instruct'
        elif "HealthGPT-L4" in args.model_path:
            self.BaseModel_path = os.path.join(args.model_path, "Phi-4")
            self.vit_path = os.path.join(args.model_path, "clip-vit-large-patch14-336") if not args.HealthGPT_VIS_path else args.HealthGPT_VIS_path
            self.hlora_path = os.path.join(args.model_path, "HealthGPT-L4", "???")
            self.hlora_r = 32
            self.hlora_alpha = 64
            self.instruct_template = 'phi4_instruct'
        # XL32 use different inference template
        # elif "HealthGPT-XL32" in args.model_path:
        #     self.BaseModel_path = os.path.join(args.model_path, "Qwen2.5-32B-Instruct")
        #     self.vit_path = os.path.join(args.model_path, "clip-vit-large-patch14-336") if not args.HealthGPT_VIS_path else args.HealthGPT_VIS_path
        #     self.hlora_path = os.path.join(args.model_path, "HealthGPT-XL32", "com_hlora_weights_QWEN_32B.bin")

        model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=self.BaseModel_path,
        attn_implementation=None,
        torch_dtype=torch.float16
        )

        from llava.peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=self.hlora_r,
            lora_alpha=self.hlora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.0,
            bias='none',
            task_type="CAUSAL_LM",
            lora_nums=4,
        )
        model = get_peft_model(model, lora_config)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.BaseModel_path,
            padding_side="right",
            use_fast=False,
        )
        num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, 8192)
        print(f"Number of new tokens added for unified task: {num_new_tokens}")

        com_vision_args.model_name_or_path = self.BaseModel_path
        com_vision_args.vision_tower = self.vit_path
        com_vision_args.version = self.instruct_template

        model.get_model().initialize_vision_modules(model_args=com_vision_args)
        model.get_vision_tower().to(dtype=torch.float16)

        model = load_weights(model, self.hlora_path)
        model.eval()
        model.to(torch.float16).cuda()

        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def inference(self, prompt_batch, image_path_batch):
        output_batch = []
        for question, img_path in zip(prompt_batch, image_path_batch):
            question = question.replace('<image>\n', '') # must, otherwise wrong size
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv = conversation_lib.conv_templates[self.instruct_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
            
            image = Image.open(img_path).convert('RGB')
            image = expand2square(image, tuple(int(x*255) for x in self.model.get_vision_tower().image_processor.image_mean))
            image_tensor = self.model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
            
            with torch.inference_mode():
                output_ids = self.model.base_model.model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image.size,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=32768,
                use_cache=True)
            
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
            output_batch.append(response)

        return output_batch