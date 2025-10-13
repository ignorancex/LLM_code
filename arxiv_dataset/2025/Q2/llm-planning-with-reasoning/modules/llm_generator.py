from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig

import torch

class LLMGenerator():

    def __init__(self, base_model_name, device):
        # load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=False
            )
        except:
            # load tokenizer without use_fast=False
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name
            )

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        # set other params
        self.base_model_name = base_model_name
        self.device = device
    

    def generate(self, prompt, generation_config,max_new_tokens=300):
        batch = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            completion = self.model.generate(
                inputs=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_config=GenerationConfig(**generation_config)
            )

        return [self.tokenizer.decode(c, skip_special_tokens=True) for c in completion]


    def ppl(self, prompt, completion):
        batch = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        input_length = len(batch["input_ids"][0])

        batch = self.tokenizer(
            completion, 
            return_tensors="pt", 
            add_special_tokens=False
        )

        input_ids = batch["input_ids"].to(self.device)
        labels = torch.zeros_like(input_ids) - 100
        labels[:, input_length:] = input_ids[:, input_length:]

        with torch.no_grad():
            loss = self.model(
                input_ids=input_ids,
                labels=labels
            ).loss.item()

        return loss * (len(input_ids[0]) - input_length)


class LLMLoraGenerator(LLMGenerator):

    def __init__(self, base_model_name, peft_model_dir, device):
        # load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=False
            )
        except:
            # load tokenizer without use_fast=False
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name
            )
        # load model
        # first, set up the quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  
            llm_int8_threshold=6.0, 
            llm_int8_enable_fp32_cpu_offload=True  
        )
        # then, load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )#.to(device)

        # load the PEFT model
        self.model = PeftModel.from_pretrained(
            self.model,
            peft_model_dir
        )
        self.model.eval()

        # set other params
        self.device = device
