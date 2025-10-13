from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset
import pandas as pd
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class InferenceModel:
    def __init__(self, model_name):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", offload_buffers=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        self.num_layers = self.get_num_layers()
        self.intermediate_size = self.get_inter_size()
        self.model.eval()

    def get_templated_prompt(self, prompt, apply_template=True):
        messages = [] # supposed to be text appended by the chat template.
        if self.model_name.startswith('Qwen/Qwen'): # This has to be model instruct 
            messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
            messages = messages if apply_template else prompt
            # text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
        elif self.model_name.startswith('google/gemma'): # This has to be model instruct
            messages = [
                {"role": "user", "content": prompt}
            ]
            messages = messages if apply_template else prompt
            # text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
        elif self.model_name.startswith('SeaLLMs'): # This has to be model instruct
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            messages = messages if apply_template else prompt
            # text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
        else:
            print("Model is not available!")
            raise ValueError("Model is not available!")


        def apply_template_single(single_prompt):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        text = apply_template_single(messages) if apply_template else messages
        # print(text)
        return text
    def get_num_layers(self):
        num_layers = 0
        if self.model_name.startswith('Qwen/Qwen'): # This has to be model instruct 
            num_layers = len(self.model.model.layers)
        elif self.model_name.startswith('google/gemma'): # This has to be model instruct
            num_layers = len(self.model.model.layers)
        elif self.model_name.startswith('SeaLLMs'): # This has to be model instruct
            num_layers = len(self.model.model.layers)
        else:
            print("Model is not available!")
            raise ValueError("Model is not available!")
        assert num_layers != 0 
        return num_layers
    def get_inter_size(self):
        inter_size = 0
        if self.model_name.startswith('Qwen/Qwen'): # This has to be model instruct 
            inter_size = self.model.model.layers[0].mlp.down_proj.in_features
        elif self.model_name.startswith('google/gemma'): # This has to be model instruct
            inter_size = self.model.model.layers[0].mlp.down_proj.in_features
        elif self.model_name.startswith('SeaLLMs'): # This has to be model instruct
            inter_size = self.model.model.layers[0].mlp.down_proj.in_features
        else:
            print("Model is not available!")
            raise ValueError("Model is not available!")
        assert inter_size != 0 
        return inter_size

    def inference(self, text, max_new_tokens=1, debug=False):
        """
        Performs inference on a given prompt.
        Returns the decodede output
        """
        generated_text = ''
        len_sentence = 0
        model_inputs = self.tokenizer([text], return_tensors="pt", add_special_tokens=False).to(self.model.device)
        # print(f"(model_inputs['input_ids'] {(model_inputs['input_ids'])}")
        
        # print(f"len(model_inputs['input_ids'] {len(model_inputs['input_ids'])}")
        sentence_token_texts = self.tokenizer.convert_ids_to_tokens((model_inputs['input_ids'])[0])
        # print(f"sentence_token_texts {sentence_token_texts}")
        
        
        generated_ids = self.model.generate(
                **model_inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        len_sentence = len(self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["input_ids"])
        # print()
        if debug:
            print(f"generated_text: {generated_text}")
            print(f"model_inputs: {model_inputs}. size: {model_inputs['input_ids'].shape }")
            
        
        return generated_text, len_sentence

    def batch_tokenize(self, texts, slice_ranges):
        slice_ranges_pad_left = slice_ranges.copy() 
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True,  padding_side="left", add_special_tokens=False).to(self.model.device)
        sentence_token_texts = self.tokenizer.convert_ids_to_tokens((model_inputs['input_ids'])[0])
        # print(f"sentence_token_texts {sentence_token_texts}")
        
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = model_inputs['input_ids']  # (batch, seq_len)
        # print(f"input_ids: {input_ids}")
        pad_counts = (input_ids == pad_token_id).sum(dim=1).tolist()  # [pad_count_0, pad_count_1, ...]
        # print(f"pad_counts: {pad_counts}")

        slice_ranges_pad_left = [(start + pad, end + pad) for (start, end), pad in zip(slice_ranges_pad_left, pad_counts)]

        input_lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]

        # slice_ranges_pad_left = []
        # seq_len = input_ids[1]
        # for i, text in enumerate(texts):
        #     unpadded_len = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        #     pad = seq_len - unpadded_len
        #     start_raw, end_raw = slice_ranges[i]
        #     slice_ranges_pad_left.append((start_raw + pad, end_raw + pad))
        return model_inputs, input_lengths, slice_ranges_pad_left
    
    def batch_inference(self, texts, model_inputs, slice_ranges_pad_left, max_new_tokens=1, debug=False):
        """
        Performs inference on a given prompt.
        Returns the decodede output
        """
        if debug:
            input_ids = model_inputs['input_ids']  # shape: [batch_size, max_seq_length]
            attention_mask = model_inputs['attention_mask']  
            sentence_token_texts = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
            print(f"sentence_token_texts : {sentence_token_texts}\n SHape : {len(sentence_token_texts)}")
            for i, text in enumerate(texts):
                print(f"\n--- Debug info for input {i} ---")
                print(f"text: {text}")
                
                input_ids = model_inputs['input_ids'][i]
                print(f"input_ids: {input_ids}")
        
                sentence_token_text = sentence_token_texts[i]
                print(f"with prompt: {sentence_token_text}")
                start, end = slice_ranges_pad_left[i]
                print(f"start: {start}, end: {end}")
                print(f"without prompt: {sentence_token_text[start:end]}\nshape: {len(sentence_token_text[start:end])}")
        
        
        generated_ids = self.model.generate(
                **model_inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # len_sentence = len(self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["input_ids"])
        # print()
        
            
        
        return generated_texts
