import os
import torch
import random
import numpy as np
import transformers
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import types
from typing import List, Optional
from packaging import version


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_name(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    

def load_pretrained_llava(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
    return processor, model


def inference_llava_once(prompt, image, model, processor):
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=512)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    
    return generated_text


def inference_llava_batch(prompts, images, model, processor):
    inputs = processor(prompts, images, return_tensors="pt", padding=True).to(model.device)
    output = model.generate(**inputs, max_new_tokens=512)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    
    return generated_text


def load_pretrained_MobileVLM(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    from MobileVLM.mobilevlm.model.mobilellama import MobileLlamaForCausalLM
    from MobileVLM.mobilevlm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = MobileLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if 'v2' in getattr(model.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()
    elif not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, context_len


def inference_MobileVLM_once(prompt, images, model, tokenizer, image_processor, conv_mode="v1", generation_config=None):
    from MobileVLM.mobilevlm.conversation import conv_templates, SeparatorStyle
    from MobileVLM.mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
    from MobileVLM.mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    
    disable_torch_init()
    if not isinstance(images, list):
        images = [images]
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # Input
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device))
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    # Generation config
    if generation_config is None:
            temperature = 0
            top_p = None
            num_beams = 1
            max_new_tokens = 512
    else:
        temperature = generation_config.temperature if hasattr(generation_config, "temperature") else 0
        top_p = generation_config.top_p if hasattr(generation_config, "top_p") else None
        num_beams = generation_config.num_beams if hasattr(generation_config, "num_beams") else 1
        max_new_tokens = generation_config.max_new_tokens if hasattr(generation_config, "max_new_tokens") else 512
    # Inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    # Result-Decode
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    generated_text = outputs.strip()
    
    return generated_text


def load_pretrained_MGM(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    try:
        from MGM.mgm.model import MGMLlamaForCausalLM, MGMMixtralForCausalLM, MGMGemmaForCausalLM
    except:
        print("New model not imported. Try to update Transformers to 4.38.0 or later.")
    from MGM.mgm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    if 'mgm' in model_name.lower():        
        # Load MGM model
        if model_base is not None:
            # this may be mm projector only
            print('Loading MGM from base model...')
            
            if "8x7b" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                model = MGMMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            elif "2b" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                model = MGMGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = MGMLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if "8x7b" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = MGMMixtralForCausalLM.from_pretrained(model_path, **kwargs)
            elif "2b" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = MGMGemmaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = MGMLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    if 'mgm' in model_name.lower():
        vision_tower_aux = model.get_vision_tower_aux()
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()
        vision_tower_aux.to(device=device, dtype=torch.float16)
        
        # initialize attention modules
        model.config.model_path = model_path
        model.get_model().initialize_uni_modules(model.config, for_eval=True)
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    # workaround for static cache in new transformers version
    if version.parse(transformers.__version__) >= version.parse("4.38.0"):
        def new_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_aux: Optional[torch.FloatTensor] = None,
            cache_position = None,
            return_dict: Optional[bool] = None,
        ):
            return self.__class__.forward(
                self,
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                images,
                images_aux,
                return_dict,
            )
        
        model.forward = types.MethodType(new_forward, model)
    
    return tokenizer, model, image_processor, context_len


def inference_MGM_once(prompt, images, model, tokenizer, image_processor, conv_mode=None, ocr=False, generation_config=None):
    from MGM.mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from MGM.mgm.conversation import conv_templates
    from MGM.mgm.utils import disable_torch_init
    from MGM.mgm.mm_utils import process_images, tokenizer_image_token
    try:
        if ocr:
            from paddleocr import PaddleOCR
    except:
        raise ImportError('please install paddleocr following https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md')
        
    disable_torch_init()
    
    if images is not None and not isinstance(images, list):
        images = [images]
        
    if ocr and images is not None:
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="ch")
        str_in_image = ''
        for image in images:
            result = ocr.ocr(np.array(image))   
            if result[0] is not None:
                result = [res[1][0] for res in result[0] if res[1][1] > 0.1]
                if len(result) > 0:
                    str_in_image += ', '.join(result)
    
    conv = conv_templates[conv_mode].copy()
    
    if images is not None:
        if hasattr(model.config, 'image_size_aux'):
            if not hasattr(image_processor, 'image_size_raw'):
                image_processor.image_size_raw = image_processor.crop_size.copy()
            image_processor.crop_size['height'] = model.config.image_size_aux
            image_processor.crop_size['width'] = model.config.image_size_aux
            image_processor.size['shortest_edge'] = model.config.image_size_aux
        
        image_tensor = process_images(images, image_processor, model.config)
        
        image_grid = getattr(model.config, 'image_grid', 1)
        if hasattr(model.config, 'image_size_aux'):
            raw_shape = [image_processor.image_size_raw['height'] * image_grid, image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor 
            image_tensor = torch.nn.functional.interpolate(image_tensor, size=raw_shape, mode='bilinear', align_corners=False)
        else:
            image_tensor_aux = []
    
        if image_grid >= 2:            
            raw_image = image_tensor.reshape(3, image_grid, image_processor.image_size_raw['height'], image_grid, image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(-1, 3, image_processor.image_size_raw['height'], image_processor.image_size_raw['width'])
                    
            if getattr(model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(global_image, size=[image_processor.image_size_raw['height'], image_processor.image_size_raw['width']], 
                                                               mode='bilinear', align_corners=False)
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)
    
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(model.device, dtype=torch.float16) for image in image_tensor_aux]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(model.device, dtype=torch.float16)
    else:
        images = None
        image_tensor = None
        image_tensor_aux = []
        
    if ocr and len(str_in_image) > 0:
        prompt = prompt + '\nReference OCR Token: ' + str_in_image + '\n'
    
    if images is not None:
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = (DEFAULT_IMAGE_TOKEN + '\n')*len(images) + prompt
        conv.append_message(conv.roles[0], prompt)
        # images = None
    else:
        conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if prompt.count(DEFAULT_IMAGE_TOKEN) >= 2:
        final_str = ''
        sent_split = prompt.split(DEFAULT_IMAGE_TOKEN)
        for _idx, _sub_sent in enumerate(sent_split):
            if _idx == len(sent_split) - 1:
                final_str = final_str + _sub_sent
            else:
                final_str = final_str + _sub_sent + f'Image {_idx+1}:' + DEFAULT_IMAGE_TOKEN
        prompt = final_str
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    if generation_config is None:
        temperature = 0.2
        max_new_tokens = 512
    else:
        temperature = generation_config.temperature if hasattr(generation_config, "temperature") else 0.2
        max_new_tokens = generation_config.max_new_tokens if hasattr(generation_config, "max_new_tokens") else 512
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
            eos_token_id=tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=tokenizer.pad_token_id,  # Pad token
            use_cache=True,
            )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    generated_text = outputs
    
    return generated_text


def load_pretrained_MiniCPM(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    from transformers import AutoModel, AutoTokenizer
    
    kwargs = {'device_map': device_map}

    if load_8bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=['out_proj', 'kv_proj', 'lm_head'],
        )
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            llm_int8_skip_modules=['out_proj', 'kv_proj', 'lm_head'],
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    model.eval()
    
    return tokenizer, model


def inference_MiniCPM_once(prompt, image, model, processor, generation_config=None):
    msgs = [{'role': 'user', 'content': prompt}]
    generated_text = model.chat(image=image, msgs=msgs, tokenizer=processor, sampling=True, temperature=0.7)
    
    return generated_text


def load_pretrained_phi3V(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    from transformers import AutoModelForCausalLM, AutoProcessor
    
    kwargs = {'device_map': device_map}

    if load_8bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    
    return processor, model


def inference_phi3V_once(prompt, image, model, processor, generation_config=None):
    if generation_config is None:
        generation_args = { 
            "max_new_tokens": 512, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
    else:
        generation_args = {
            "max_new_tokens": generation_config.max_new_tokens if hasattr(generation_config, "max_new_tokens") else 512,
            "temperature": generation_config.temperature if hasattr(generation_config, "temperature") else 0.0,
            "do_sample": generation_config.do_sample if hasattr(generation_config, "do_sample") else False,
        }

    messages = [{'role': 'user', 'content': prompt}]
    messages = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(messages, [image], return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    
    return generated_text
    

load_pretrained_func = {
    "llava": load_pretrained_llava,
    "MobileVLM": load_pretrained_MobileVLM,
    "MGM": load_pretrained_MGM,
    "MiniCPM": load_pretrained_MiniCPM,
    "Phi-3-V": load_pretrained_phi3V,
}

inference_func = {
    "llava": {"once": inference_llava_once, "batch": inference_llava_batch},
    "MobileVLM": {"once": inference_MobileVLM_once, "batch": None},
    "MGM": {"once": inference_MGM_once, "batch": None},
    "MiniCPM": {"once": inference_MiniCPM_once, "batch": None},
    "Phi-3-V": {"once": inference_phi3V_once, "batch": None},
}