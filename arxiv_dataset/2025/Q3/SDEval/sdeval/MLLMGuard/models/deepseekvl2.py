import torch
from transformers import AutoModelForCausalLM

from models.deepseek_vl2.deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from models.deepseek_vl2.deepseek_vl2.utils.io import load_pil_images


from models.base import Mllm


class DeepSeek_v2(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_name_or_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


        
    def evaluate(self, prompt, filepath):
        
        conversation = [
            {
                "role": "<|User|>",
                "content": f'<image>\n'+prompt,
                "images": [filepath],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True
        )[0]
        print(outputs)
        # exit(0)
        
        answer = self.tokenizer.decode(outputs.cpu().tolist(), skip_special_tokens=False)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer