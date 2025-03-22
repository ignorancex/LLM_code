import numpy as np
from icecream import ic
from tqdm import trange
import torch
import pandas as pd
from warping_dataset_like_GPT2 import GPT3TokenizerWarper, GPT3Warper, generate_prompts
import argparse
from datasets import load_from_disk

if __name__=='__main__':
    setup = 3
    df = pd.read_csv('finalized_short_prompt_new.csv')
    perfix_list, postfix_list = generate_prompts(df, setup)

    tokenizer_list = []
    for i in range(len(perfix_list)):
        tokenizer_list.append(GPT3TokenizerWarper([perfix_list[i], postfix_list[i]]))

    model = GPT3Warper("/path2OpenAIKey", f"gpt3_result_setup_{setup}_short_small_train.jsonl")

    dataset = load_from_disk('./yelp_small_train_new')
    #ic(dataset[:]['text'])

    wraped_list = []
    for i in range(len(perfix_list)):
        wraped_list.append(tokenizer_list[i](dataset[:]['text']))
    #print(wraped_list[0]['input_prompts'][0])
    ic(len(wraped_list[0]))
    result_arr = []
    for i in wraped_list:
        result_arr.append(model(**i))
    a = np.array(result_arr)
    #ic(a[0])
    np.save(f"gpt3_result_setup_{setup}_short_small_train.npy",a)
