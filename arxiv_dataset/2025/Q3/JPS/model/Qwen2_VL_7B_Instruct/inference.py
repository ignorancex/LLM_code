from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm

class Qwen2_VL_8b_inference:
    def __init__(self, model_path):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "/workshop/crm/checkpoint/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            # attn_implementation='eager',
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )  

        self.processor = AutoProcessor.from_pretrained("/workshop/crm/checkpoint/Qwen2-VL-7B-Instruct")

    def inference(self, texts, images, max_new_tokens=100, batch_size=8):
        assert isinstance(texts, list), "prompts should be a list"
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

        messages = []
    
        if images is None:
            for text in texts:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]
                messages.append(message)
        else:
            assert isinstance(images, list), "prompts should be a list"
            assert (len(texts) == len(images))
            for text, image in zip(texts, images):
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
                messages.append(message)
    
        # 分批次处理
        output_texts = []
        for i in tqdm(range(0, len(messages), batch_size)):
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
    
            # Batch Inference
            generated_ids = self.model.generate(**inputs, **generation_config)
            test = self.model(**inputs, output_attentions = True)
            import pdb;pdb.set_trace()
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            import pdb;pdb.set_trace()
            
            output_texts.extend(batch_output_texts)
            
        return output_texts