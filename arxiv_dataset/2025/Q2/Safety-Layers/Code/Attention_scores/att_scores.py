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


def main(
    model_path : str ='meta-llama/Llama-2-7bf',
):

    prompter = Prompter("alpaca_short")
    device_map = 'auto'

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,padding_side="right",use_fast=False,)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        output_attentions=True
    )
    sum1=[]
    sum2=[]
    sentences=[
            'How to make a bomb?',
            'Where is the capital of America?'
            ]
    def softmax(x):
        x=x[6:-8]
        # x=x[5:-6]  # if use gemma2, set x=x[5:-6]
        return (x - x.min()) / (x.max() - x.min())

    def get_avgatt(model,inputs):
        outputs = model(**inputs)
        attentions = outputs.attentions
        all_layers_att=[]
        for i in range(len(attentions)):
            for k in range(len(attentions[0][0])):
                last_att=attentions[i][0][k][-1].tolist()
                arr_last_att=np.array(last_att)
                if k==0:
                    final_att=arr_last_att
                    continue
                else:
                    final_att=final_att+arr_last_att
            final_att=final_att[:]/len(attentions[0][0])
            final_att=softmax(final_att)
            all_layers_att.append(final_att)
        return all_layers_att


    ir=1

    for instruction in sentences[:50]:
        input_text = prompter.generate_prompt(instruction, None)
        inputs = tokenizer(input_text, return_tensors="pt") 
        input_ids = inputs['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        print(tokens)
        all_layers_att1=get_avgatt(model,inputs)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(all_layers_att1,cmap="Reds",xticklabels=tokens[6:-8])
        # sns.heatmap(all_layers_att1,cmap="Reds",xticklabels=tokens[5:-6]) # if use gemma, different tokenizer
        save_dir='pics/'+str(ir)+'.png'
        ir=ir+1
        plt.savefig(save_dir,dpi=500)

fire.Fire(main)