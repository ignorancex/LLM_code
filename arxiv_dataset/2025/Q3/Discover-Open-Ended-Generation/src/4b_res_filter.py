import os
import random
import csv
import tqdm
import argparse
import itertools
import wandb
import logging
from time import strftime
import sys
from llm_utils import *
from prettytable import PrettyTable
import pandas as pd
from datasets import load_dataset
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import ast
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact, chi2_contingency
from natsort import natsorted
from sklearn.cluster import HDBSCAN
import ast
from glob import glob


CACHE_DIR = '/scratch/jpan23'

DATA_DIR = '../data'
PROMPT_DIR = 'prompt_instructions'
RES = '../FINAL_RAW_RES'
T = strftime('%Y%m%d-%H%M')

def helper():
    if "gpt" in args.model_name:
        perspectiveModel = ChatGPT(args.model_name, temperature=args.temperature, verbose=args.verbose)
    elif "Qwen3" in args.model_name:
        perspectiveModel = QWEN3(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Qwen" in args.model_name:
        perspectiveModel = QWEN(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Mistral" in args.model_name:
        perspectiveModel = MISTRAL(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Llama-4" in args.model_name:
        perspectiveModel = LLM4(args.model_name, temperature=args.temperature, quanti=args.eight_bit, verbose=args.verbose)
    elif "Llama" in args.model_name and 'Vision' not in args.model_name:
        perspectiveModel = LLM(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Llama" in args.model_name and 'Vision' in args.model_name:
        perspectiveModel = LLM3_2_V(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "command" in args.model_name:
        perspectiveModel = COHERE(args.model_name, temperature=args.temperature, verbose=args.verbose)

    out_folder = '../FINAL_RES_ALL'

    concept_dir = os.path.join(out_folder)
    if not os.path.exists(concept_dir):
        os.mkdir(concept_dir)

    # Find all CSV files in the input folder
    csv_files = glob(os.path.join(RES, '*.csv'))

    for csv_file in csv_files:
        if 'llama2_7b' in csv_file:
            continue

        count_exclu = 0
        with open(csv_file, "r") as f:
            reader = csv.reader(f, delimiter=',')
            all_rows = list(reader)
        
        print('--------------------------------------')
        print('L69: ', csv_file)

        table = PrettyTable()
        table.field_names = ["Location Category", "Demographic Identity", "Phrase", "Score", "P-Value", 'Prompt Response']
        
        for index, row in enumerate(tqdm.tqdm(all_rows[1:])):
            # location cat,identity,phrase,score,pval
            loc_cat = row[0]
            id_ = row[1]
            phrase = row[2]
            score = row[3]
            pval = row[4]
            cat = ''

            if 'race' in csv_file:
                # print('L80: ', csv_file)
                id_ = id_ + ' race'
                cat = 'race'
            elif 'religions' in csv_file:
                # print('L83: ', csv_file)
                if id_ == 'jew':
                    id_ = 'Judaism'
                if id_ == 'christian':
                    id_ = 'Christian'
                if id_ == 'budd':
                    id_ = 'Buddhism'
                if id_ == 'mus':
                    id_ = 'Muslim'
                cat = 'religions'
            else:
                # print('L99: ', csv_file)
                cat = 'gender'
            
            with(open(f'{PROMPT_DIR}/res_filter.txt', 'r')) as f:
                instruction = f.read()
            
            prompt = f"""{instruction}
Phrase: {phrase}
Identity: {id_}
Category: {cat}"""
            
            predicted_answer = perspectiveModel.getOutput(prompt)

            if predicted_answer.lower() == 'no':
                table.add_row([loc_cat, id_, phrase, score, pval, predicted_answer])
            else:
                count_exclu += 1
                print('L117: ', predicted_answer, '::', phrase, '::', id_, '::', cat)

        print('L119 exclusive count total one file: ', count_exclu)

        out_df = pd.DataFrame(table.rows, columns=table.field_names)
        out_filename = os.path.join(out_folder, os.path.basename(csv_file))
        out_df.to_csv(out_filename, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-32B')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--gpu', type=int, default=0) # which gpu to load on
    parser.add_argument('--eight_bit', action='store_true') # load model in 8-bit? and quantized for llama 4?
    parser.add_argument('--T', type=str, default="")
    
    global args
    args = parser.parse_args()

    log_dir = os.path.join('../logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


    # for _ in range(10):
    args.T = strftime('%Y%m%d-%H%M')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    # log_file = '../logs/' + args.T + '_' + args.dataset + '_output_results.log'
    log_file = '../logs/' + args.T + '_output_results.log'
    if log_file is None:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=log_format)

    
    # logging.basicConfig(filename='../logs/' + t + '_output.log', level=logging.INFO)
    logging.info(args)
    print(args)

    helper()

    # helper_t()

    logging.shutdown()

if __name__ == '__main__':
    main()