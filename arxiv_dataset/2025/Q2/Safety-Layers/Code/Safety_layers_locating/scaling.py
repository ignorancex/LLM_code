from dis import Instruction
import os
from site import check_enableusersite
import sys
from datasets import load_dataset
import os
import pickle
import seaborn as sns
import pandas as pd
import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
import random
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append("..")
from utils.prompter import Prompter
from utils.callbacks import Iteratorize, Stream
import random
import matplotlib.pyplot as plt
import numpy as np

import copy


def scaling(base_model,chat_model,begin_num,end_num,cheng_num):
    new_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for i in range(begin_num, end_num):
            new_model.model.layers[i].self_attn.q_proj.weight.copy_(\
                chat_model.model.layers[i].self_attn.q_proj.weight*cheng_num)
            new_model.model.layers[i].self_attn.k_proj.weight.copy_(\
                chat_model.model.layers[i].self_attn.k_proj.weight*cheng_num)
            new_model.model.layers[i].self_attn.v_proj.weight.copy_(\
                chat_model.model.layers[i].self_attn.v_proj.weight*cheng_num)
            new_model.model.layers[i].self_attn.o_proj.weight.copy_(\
                chat_model.model.layers[i].self_attn.o_proj.weight*cheng_num)
            new_model.model.layers[i].mlp.up_proj.weight.copy_(\
                chat_model.model.layers[i].mlp.up_proj.weight*cheng_num)
            new_model.model.layers[i].mlp.gate_proj.weight.copy_(\
                chat_model.model.layers[i].mlp.gate_proj.weight*cheng_num)
            new_model.model.layers[i].mlp.down_proj.weight.copy_(\
                chat_model.model.layers[i].mlp.down_proj.weight*cheng_num)
    return new_model

def scaling_phi3(base_model,chat_model,begin_num,end_num,cheng_num):
    new_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for i in range(begin_num, end_num):
            new_model.model.layers[i].self_attn.qkv_proj.weight.copy_(\
                chat_model.model.layers[i].self_attn.qkv_proj.weight*cheng_num)
            new_model.model.layers[i].self_attn.o_proj.weight.copy_(\
                chat_model.model.layers[i].self_attn.o_proj.weight*cheng_num)
            
            new_model.model.layers[i].mlp.gate_up_proj.weight.copy_(\
                chat_model.model.layers[i].mlp.gate_up_proj.weight*cheng_num)
            new_model.model.layers[i].mlp.down_proj.weight.copy_(\
                chat_model.model.layers[i].mlp.down_proj.weight*cheng_num)
    return new_model



def get_output(
    model,
    instruction,
    prompter,
    tokenizer,
    input=None,
    temperature=0,
    top_p=0.2,
    top_k=40,
    max_new_tokens=16,
    device='cuda'):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=0
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        eos_token_id=terminators,
        output_scores=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1
    )
    return generation_output


def main(
    start_num: int = 5,
    prompt_template:str='alpaca',
    end_num:int = 15,
    save_dir: str="results/",
    cheng_num: float=1.1,
    model_path: str='meta-llama/Llama-2-7bf',
    data_path: str='../../Dataset/Evaluation/Over_rejection_dataset.csv',
    max_new_tokens: int=16,
):
    device_map = 'auto'
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,padding_side="right",use_fast=False,)
    copy_base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    trust_remote_code=True,
    )
    copy_chat_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    trust_remote_code=True,)




    model=scaling(copy_base_model,copy_chat_model,start_num,end_num,cheng_num)
    # model=scaling_phi3(copy_base_model,copy_chat_model,start_num,end_num,cheng_num) #if phi-3 use this
    
    
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    # ---------------------------------------------------------------------------------------------------------------
    df = pd.read_csv(data_path, header=None)  

    sentences = df[0].tolist()
    prompter = Prompter(prompt_template)
    outputs=[]
    for instruction in sentences:
        print(instruction)
        generation_output=get_output(model=model,instruction=instruction,prompter=prompter,tokenizer=tokenizer,max_new_tokens=max_new_tokens)
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        answer=prompter.get_response(output)
        print(output,'\n','*'*40)
        outputs.append(answer)

        df = pd.DataFrame(outputs)

    save_dir=save_dir
    df.to_csv(save_dir, index=False)


if __name__ == "__main__":
    fire.Fire(main)
