CACHE_DIR = '/scratch/'

import os
# os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
import random
import csv
import tqdm
import argparse
import torch
import itertools
import wandb
from transformers import GenerationConfig, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import MllamaForConditionalGeneration, MllamaForCausalLM, AutoProcessor, Llama4ForConditionalGeneration, FbgemmFp8Config, AutoModelForImageTextToText

import openai
import time
from typing import *
import json
import re
import copy
import math
import datetime

from torch.nn import functional as F

from cappr.huggingface.classify import cache, predict_proba
from cappr.openai.classify import predict_proba as openai_predict_proba
import numpy as np

from openai import RateLimitError, Timeout, APIError, APIConnectionError, OpenAIError, AzureOpenAI, OpenAI
import cohere

import base64

import logging

from huggingface_hub import login


# Please provide the api key in api_key.txt!
with open("api_key.txt", "r") as f:
    API_KEY = f.readline().strip()

# fill in your api keys
config_data = json.load(open("../config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
COHERE_API = config_data["cohere_api"]
login(token = HF_TOKEN)
PROMPT_DIR = 'prompt_instructions'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

class LLM:
    
    """LLM wrapper class.
        Meant for use with local Llama-based HuggingFace models (all instruct models). Not using pipleline here
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        
        
        if not load_in_8bit and not load_in_4bit:
            # self.model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
            # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR)
            # device_kwargs = {"device_map":"auto"}
        elif load_in_8bit:
            # self.model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        else:
            # self.model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
            # device_kwargs = {"device_map":"auto", "load_in_8bit": True}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loaded base llama: ", model_name)

    

    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]

        m_1 = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        },
        ###
        {
            "role": "assistant", 
            "content": "Sure! In this context, I will write a story:"
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        # eosToken = "[/INST]"
        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        ## old method, also worked
        # inputs = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        # # inputs = self.tokenizer(text=text, return_tensors="pt").to(self.model.device)
        # output = self.model.generate(inputs, generation_config=generation_config)

        
        if "Llama-2" in self.model_name:
            return_output = self.tokenizer.apply_chat_template(m_1, return_tensors="pt", return_dict=True, continue_final_message=True).to(self.model.device) ###
        else:
            return_output = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device) ### ori

        output = self.model.generate(**return_output, generation_config=generation_config, pad_token_id=self.tokenizer.eos_token_id)

        # response = self.tokenizer.decode(output[0]).split("<|eot_id|>")[2].split("<|end_header_id|>")[1].strip()

        if "Llama-2" in self.model_name: # llamma 2 chat model, chat
            # response = self.generator(prompt, generation_config=generation_config)[0]['generated_text'].split("[/INST]")[1]
            # print(self.tokenizer.decode(output[0]))
            response = self.tokenizer.decode(output[0]).split("[/INST]")[1].split("</s>")[0].strip()
        elif "Llama-3" in self.model_name: # 3.1-Instruct prompt format or above
            # print(self.tokenizer.decode(output[0]).split("<|end_header_id|>")[3].strip()) # same
            response = self.tokenizer.decode(output[0]).split("<|eot_id|>")[2].split("<|end_header_id|>")[1].strip()
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response

    
    def encode_phrases(self, phrases, batch_size=5):
        self.model.eval()

        ### batch version
        embeddings_list = []
        for i in range(0, len(phrases), batch_size):
            batch_phrases = phrases[i:i+batch_size]
            inputs = self.tokenizer(batch_phrases, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

                # hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                hidden_states = outputs.hidden_states[-1]
                attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch_size, seq_len, 1)

                # Zero out padded positions
                masked_hidden = hidden_states * attention_mask  # Mask padding
                sum_hidden = masked_hidden.sum(dim=1)  # Sum real tokens
                lengths = attention_mask.sum(dim=1)  # How many real tokens for each phrase

                batch_embeddings = (sum_hidden / lengths).detach().cpu()
            
                embeddings_list.append(batch_embeddings)  # Move back to CPU to save GPU memory

        embeddings = torch.cat(embeddings_list, dim=0)

        return embeddings
    

class LLM3_2_V:
    
    """LLM wrapper class.
        Meant for use with local Llama3.2-based HuggingFace models. Vison-Included
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            # self.model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR)
            # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="flash_attention_2")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR)
            # device_kwargs = {"device_map":"auto"}
        elif load_in_8bit:
            # self.model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR)
        else:
            # self.model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loaded llama 3 vision: ", model_name)


    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": "You are a useful assistant."
            }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        # eosToken = "[/INST]"
        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        inputs = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)
        output = self.model.generate(**inputs, generation_config=generation_config)

        response = self.tokenizer.decode(output[0]).split("<|eot_id|>")[2].split("<|end_header_id|>")[1].strip()
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response

    
    def getActivations(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states

        return hidden_states



class LLM4:
    
    """LLM wrapper class.
        Meant for use with local Llama4-based HuggingFace models.
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        # load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        # load_in_4bit=False,
        quanti=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not quanti:
            
            self.model = Llama4ForConditionalGeneration.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="sdpa", torch_dtype=torch.bfloat16)

            
        elif quanti:
            self.model = Llama4ForConditionalGeneration.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, attn_implementation="flex_attention", quantization_config=FbgemmFp8Config())

        ###
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id ###
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.processor = AutoProcessor.from_pretrained(model_name)

        print("Loaded llama 4: ", model_name)

    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": "You are a useful assistant."
            }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)


        inputs = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        outputs = self.model.generate(**inputs.to(self.model.device), max_new_tokens=self.max_new_tokens, temperature=self.temperature, repetition_penalty=self.repetition_penalty, do_sample=self.do_sample, top_p=self.top_p, top_k=self.top_k,)

        response = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response



class MISTRAL:
    
    """LLM wrapper class.
        MISTRAL.
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Mistral loaded")

    

    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        return_output = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)


        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        output = self.model.generate(input_ids=return_output["input_ids"], generation_config=generation_config, attention_mask=return_output["attention_mask"], pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(output[0]).split("[/INST]")[1].split("</s>")[0].strip()
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response
    


class QWEN:
    
    """LLM wrapper class.
        qwen hf.
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("QWEN loaded")


    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        return_output = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)


        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        output = self.model.generate(**return_output, generation_config=generation_config, pad_token_id=self.tokenizer.eos_token_id)
        
        response = self.tokenizer.decode(output[0]).split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(response)
        
        return response


class QWEN3:
    
    """LLM wrapper class.
        qwen hf.
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("QWEN3 loaded")

    

    def getOutput(self, prompt):
        
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)


        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )


        text = self.tokenizer.apply_chat_template(
            m,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switch between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print("thinking content:", thinking_content)
        # print("content:", content)
        
        if self.verbose:
            print("### RESPONSE ###")
            print("thinking content:", thinking_content)
            print(content)
            logging.info("### RESPONSE ###")
            logging.info(content)
        
        return content
    
class GPT_2:
    
    """LLM wrapper class.
        gpt-2 hf.
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, torch_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("gpt-2 loaded")

    

    def getOutput(self, prompt):


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        # prompt = "Replace me by any text you'd like."

        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # conduct text completion
        output = self.model.generate(
            **model_inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            generation_config=generation_config
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        content = response[len(prompt):].strip()

        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(content)
        
        return content
    
class FALCON:
    
    """LLM wrapper class.
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=2048,                    # Maximum number of new tokens to generate
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        

        if not load_in_8bit and not load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        print("falcon loaded")

    

    def getOutput(self, prompt):


        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        # prompt = "Replace me by any text you'd like."

        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # conduct text completion
        output = self.model.generate(
            **model_inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            generation_config=generation_config
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        content = response[len(prompt):].strip()

        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
            logging.info("### RESPONSE ###")
            logging.info(content)
        
        return content


class ChatGPT:
    """ChatGPT wrapper.
    """
    def __init__(self, model_name, api_key=None, temperature=0.3, verbose=False):
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        ### ori AuzreOpenAI here
        self.client = AzureOpenAI(
            api_key=API_KEY,  
            api_version="",
            azure_endpoint = ''
        )
        ###

        # self.client = OpenAI(
        #     organization='',
        #     project='',
        #     api_key=API_KEY
        # )

    def get_prob(self, prompt, completions, max_retries=30):
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        def prompt_func(prompt) -> str:
            return f'''
        You are a useful assistant.

        {prompt}'''

        formatted_prompt = prompt_func(prompt)

        for i in range(max_retries):
            try:
                pred_probs = openai_predict_proba(prompts=formatted_prompt, completions=completions, model=self.model_name, client=self.client, api_key=API_KEY)
                
                if self.verbose:
                    print("### RESPONSE ###")
                    print(pred_probs)
                
                return pred_probs
            
            except OpenAIError as e:
                # Exponential backoff
                if i == max_retries - 1:  # If this was the last attempt
                    raise  # re-throw the last exception
                else:
                    # Wait for a bit before retrying and increase the delay each time
                    sleep_time = (2 ** i) + random.random()  # Exponential backoff with full jitter
                    time.sleep(sleep_time)

    def getOutput(self, prompt:str, max_retries=30) -> str:
        """Gets output from OpenAI ChatGPT API.

        Args:
            prompt (str): Prompt
            max_retries (int, optional): Max number of retries for when API call fails. Defaults to 30.

        Returns:
            str: ChatGPT response.
        """
        
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        m = [
        {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": "You are a useful assistant."
            }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        ]

        for i in range(max_retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=m,
                    temperature = self.temperature,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0.6,
                    presence_penalty=0
                )
                output = res.choices[0].message.content
                
                # print("### RAW RESPONSE ###")
                # print(output)
                
                if output == None or ("sorry" in output.lower() and "can't" in output.lower()):
                    output = "response filtered"
                
                if self.verbose:
                    print("### RESPONSE ###")
                    print(output)
                    logging.info("### RESPONSE ###")
                    logging.info(output)
                    
                return output
            except OpenAIError as e:
                print(f"Error: {e}")
                if 'content_filter' in str(e):
                    return "response filtered"
                # Exponential backoff
                if i == max_retries - 1:  # If this was the last attempt
                    raise  # re-throw the last exception
                else:
                    # Wait for a bit before retrying and increase the delay each time
                    sleep_time = (2 ** i) + random.random()  # Exponential backoff with full jitter
                    time.sleep(sleep_time)

    
    def get_output_batch(self, file_name):
        # model_name: gpt-4o-20240513-global-batch
        # Upload a file with a purpose of "batch"
        file = self.client.files.create(
            file=open(file_name, "rb"),
            # file=open("../data/example.jsonl", "rb"), ## for testing only
            purpose="batch"
        )
        print(file.model_dump_json(indent=2))
        file_id = file.id

        # Awaiting system file processing
        time.sleep(10) ### may need to take longer time

        # Submit a batch job with the file
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions",
            completion_window="24h",
        )

        # Save batch ID for later use
        batch_id = batch_response.id
        print(batch_response.model_dump_json(indent=2))

        """
        Track batch job progress (advised to write into a separate file)
        """
        # Tracking batch status
        status = "validating"
        while status not in ("completed", "failed", "canceled"):
            time.sleep(5)
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status
            print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

        if batch_response.status == "failed":
            for error in batch_response.errors.data:  
                print(f"Error code {error.code} Message {error.message}")

        """
        Retrieve batch job output file (advised to write into a separate file)
        """
        output_file_id = batch_response.output_file_id

        if not output_file_id:
            output_file_id = batch_response.error_file_id

        if output_file_id:
            file_response = self.client.files.content(output_file_id)

            return file_response.text
        else:
            return "ERROR OUTPUT!!!"


class COHERE:
    """cohere wrapper.
    """
    # model name: "command-r-plus-08-2024"
    def __init__(self, model_name, api_key=None, temperature=0.3, verbose=False):
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        self.client = cohere.ClientV2(api_key=COHERE_API)

    def getOutput(self, prompt:str, max_retries=30) -> str:
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]

        for i in range(max_retries):
            try:
                res = self.client.chat(
                    model=self.model_name,
                    messages=m,
                    temperature = self.temperature,
                    max_tokens=2048,
                    p=1,
                    k=50,
                    frequency_penalty=0.6,
                    presence_penalty=0
                )
                output = res.message.content[0].text
                
                # print("### RAW RESPONSE ###")
                # print(output)
                
                if output == None or ("sorry" in output.lower() and "can't" in output.lower()):
                    output = "response filtered"
                
                if self.verbose:
                    print("### RESPONSE ###")
                    print(output)
                    logging.info("### RESPONSE ###")
                    logging.info(output)
                    
                return output
            
            except Exception as e:
                print(f"Error: {e}")
                # Exponential backoff
                if i == max_retries - 1:  # If this was the last attempt
                    raise  # re-throw the last exception
                else:
                    # Wait for a bit before retrying and increase the delay each time
                    sleep_time = (2 ** i) + random.random()  # Exponential backoff with full jitter
                    time.sleep(sleep_time)