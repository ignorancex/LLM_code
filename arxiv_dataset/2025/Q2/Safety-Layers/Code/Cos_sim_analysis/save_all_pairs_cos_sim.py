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
import random

warnings.filterwarnings('ignore')

def get_output(
    model,
    instruction,
    prompter,
    tokenizer,
    input=None,
    temperature=0.5,
    top_p=0.2,
    top_k=40,
    num_beams=4,
    max_new_tokens=1,
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
    generation_output = model.generate(
        input_ids=input_ids,
        output_hidden_states= True,
        generation_config=generation_config,
        return_dict_in_generate=True,
        # output_scores=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1
    )
    return generation_output


def get_r_lists_cossim(model,prompter,tokenizer,datapath1,datapath2,seed,r=500):
    df = pd.read_csv(datapath1, header=None)
    sentences_1 = df[0].tolist()
    
    df = pd.read_csv(datapath2, header=None)
    sentences_2 = df[0].tolist()

    allcos=[]
    # plt.figure()
    for sss in range(r):
        random.seed(seed)
        seed=seed+1
        intermediate_outputs=[]
        
        if datapath1 == datapath2:
            random_items = random.sample(sentences_1, 2)
            instruction1=random_items[0]
            instruction2=random_items[1]
        else:
            instruction1=random.sample(sentences_1, 1)[0]
            instruction2=random.sample(sentences_2, 1)[0]
        all_vectors=[]
        generation_output1=get_output(model=model,instruction=instruction1,prompter=prompter,tokenizer=tokenizer)
        hs1 = generation_output1['hidden_states']
        for i in range(len(hs1[0])):
            if i==0:
                continue
            all_vectors.append(hs1[0][i][0][-1])
        all_vectors2=[]
        generation_output2=get_output(model=model,instruction=instruction2,prompter=prompter,tokenizer=tokenizer)
        hs2 = generation_output2['hidden_states']
        for i in range(len(hs2[0])):
            if i==0:
                continue
            all_vectors2.append(hs2[0][i][0][-1])
        cso=[]
        for k in range(len(all_vectors2)):
            a=all_vectors[k].cpu().detach().numpy()
            b=all_vectors2[k].cpu().detach().numpy()
            cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cso.append(cosine_similarity)
            
        allcos.append(cso)
    # plt.show()
    
    print('end')
    
    return allcos
    

def main(
    normal_path: str='normal.csv',
    malicious_path: str='malicious.csv',
    model_path: str='meta-llama/Llama-2-7bf',
    save_dir: str='cos_sims/',
    r: int=500
    ):
    
    prompter = Prompter("alpaca")
    device_map = 'auto'

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,padding_side="right",use_fast=False,)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
    )
            
    allcos_Normal_Norma_pairs=get_r_lists_cossim(model,prompter,tokenizer,normal_path,normal_path,10,r)
    allcos_Mali_Mali_pairs=get_r_lists_cossim(model,prompter,tokenizer,malicious_path,malicious_path,100,r)
    allcos_Normal_Mali_pairs=get_r_lists_cossim(model,prompter,tokenizer,normal_path,malicious_path,1000,r)
    

    import pickle
    kkk=[allcos_Normal_Norma_pairs,allcos_Mali_Mali_pairs,allcos_Normal_Mali_pairs]
    # 使用Pickle写入
    sa=save_dir+'/all_cos.pkl'
    with open(sa, 'wb') as f:
        pickle.dump(kkk, f)
        
fire.Fire(main)