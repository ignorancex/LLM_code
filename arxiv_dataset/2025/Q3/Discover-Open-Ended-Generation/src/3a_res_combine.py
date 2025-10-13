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

# from transformers import AutoTokenizer, Llama4ForConditionalGeneration, BitsAndBytesConfig
# import torch
# CACHE_DIR = '/scratch/jpan23'

DATA_DIR = '../data'
PROMPT_DIR = 'prompt_instructions'
RES = '../SUMMARY_B'
T = strftime('%Y%m%d-%H%M')



def concepts_combine2():
    # data = args.dataset

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

    if "gpt" not in args.model_name and "command" not in args.model_name:
        device_map = perspectiveModel.model.device
        print("Device Map:", device_map)
        print("full device map (multi-gpus):", perspectiveModel.model.hf_device_map)

        # Check if any part of the model is on a GPU
        if device_map.type == 'cuda':
            print("The model is using a GPU.")
        else:
            print("The model is not using any GPU.")
            return 0

    table = PrettyTable()

    table_2 = PrettyTable()
    
    # table.field_names = ["question", "gold", "prediction", "grade"]
    
    # Print stuff
    print("\n------------------------")
    print("    EVALUATING WITH      ")
    print("------------------------")

    logging.info("\n------------------------")
    logging.info("    EVALUATING WITH      ")
    logging.info("------------------------")
    print(f"MODEL: {args.model_name}")
    # print(f"CAT: {cat}")
    logging.info(f"MODEL: {args.model_name}")
    # logging.info(f"CAT: {cat}")
    print("------------------------\n")

    # logging.info(f"NUM PROBS: {args.num_probs}")
    logging.info("------------------------\n")

    table.field_names = ["prompt", "summ", "location", "identity", "id1", "id2", "bias type", "place_category"]

    if args.info_ == 'two1':

        if args.condition == 'gender':
            # gender
            # cb
            if args.add_in == 'base':
                csv_name = os.path.join(RES, '') # no des, close-box 20 more

            if args.add_in == 'ob':
                csv_name = os.path.join(RES, '') # no des, open-box 20 more sl 2

            if args.add_in == 'simple2':
                csv_name = os.path.join(RES, '') #

            if args.add_in == 'simple3':
                csv_name = os.path.join(RES, '')
            
            if args.add_in == 'llama3_2_11b':
                csv_name = os.path.join(RES, '')
            
            if args.add_in == 'qwen3_8b':
                csv_name = os.path.join(RES, '')

        elif args.condition == 'race':
            # race
            # cb
            if args.add_in == 'base':
                csv_name = os.path.join(RES, '')

            # ob
            if args.add_in == 'ob':
                csv_name = os.path.join(RES, '')

            if args.add_in == 'simple2':
                csv_name = os.path.join(RES, '')
            
            if args.add_in == 'simple3':
                csv_name = os.path.join(RES, '') #

            if args.add_in == 'llama3_2_11b':
                csv_name = os.path.join(RES, '')
            
            if args.add_in == 'qwen3_8b':
                csv_name = os.path.join(RES, '')

        elif args.condition == 'religions':
            # religions
            # cb
            if args.add_in == 'base':
                csv_name = os.path.join(RES, '')

            # ob
            if args.add_in == 'ob':
                csv_name = os.path.join(RES, '')

            if args.add_in == 'simple2':
                csv_name = os.path.join(RES, '') #
            
            if args.add_in == 'simple3':
                csv_name = os.path.join(RES, '')#

            if args.add_in == 'llama3_2_11b':
                csv_name = os.path.join(RES, '')
            
            if args.add_in == 'qwen3_8b':
                csv_name = os.path.join(RES, '')

    # Increase field size limit
    csv.field_size_limit(sys.maxsize)

    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=',')
        all_rows = list(reader)
    
    # args.num_probs = min(args.num_probs, len(all_rows))

    for index, row in enumerate(tqdm.tqdm(all_rows[1:])):
        # prompt,summ,location,identity,id1,id2,bias type # two 1
        # prompt,summ_diff,summ_final,location,identity,id1,id2,bias type # two 2

        if args.info_ == 'two1':
            # prompt,summ,location,identity,id1,id2,bias type, p
            p = row[0]
            response = row[1]
            location = row[2]
            identity = row[3]
            id1 = row[4]
            id2 = row[5]
            bias_type = row[6]
            place = row[7] if len(row) > 7 else 'N/A'

        with(open(f'{PROMPT_DIR}/res_combine2.txt', 'r')) as f:
            instruction = f.read()

        instruction = instruction.replace('[[D1]]', id1).replace('[[D2]]', id2)

        
        # Base prompt. The instruction corresponds to the above instruction prompts for diff. methods.
        prompt = f"""{instruction}

{response}"""
        
        predicted_answer = perspectiveModel.getOutput(prompt)

        table.add_row([prompt, predicted_answer, location, identity, id1, id2, bias_type, place])


    concept_dir = os.path.join('../SUMM_COMBINE')
    if not os.path.exists(concept_dir):
        os.mkdir(concept_dir)

    if "llama" in args.model_name:
        model_name = args.model_name[11:]
    elif "Qwen" in args.model_name:
        model_name = args.model_name[5:]
    elif "mistral" in args.model_name:
        model_name = args.model_name[10:]
    else:
        model_name = args.model_name

    save_concepts = pd.DataFrame(table._rows, columns=table.field_names) ###
        
    # T = strftime('%Y%m%d-%H%M')
    temp = str(args.temperature)

    save_concepts_dir = os.path.join(concept_dir, T + '_' + args.info_ + '_' + model_name + '_' + args.condition + '_' + args.add_in + '_all_summ.csv')
    save_concepts.to_csv(save_concepts_dir, index = False, header=True)

    print("\n------------------------")
    print("         COMPLETE        ")
    print("------------------------")



def concepts_combine1():
    # data = args.dataset

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

    if "gpt" not in args.model_name and "command" not in args.model_name:
        device_map = perspectiveModel.model.device
        print("Device Map:", device_map)
        print("full device map (multi-gpus):", perspectiveModel.model.hf_device_map)

        # Check if any part of the model is on a GPU
        if device_map.type == 'cuda':
            print("The model is using a GPU.")
        else:
            print("The model is not using any GPU.")
            return 0

    table = PrettyTable()

    # table_2 = PrettyTable()
    
    # table.field_names = ["question", "gold", "prediction", "grade"]
    
    # Print stuff
    print("\n------------------------")
    print("    EVALUATING WITH      ")
    print("------------------------")

    logging.info("\n------------------------")
    logging.info("    EVALUATING WITH      ")
    logging.info("------------------------")
    print(f"MODEL: {args.model_name}")
    # print(f"CAT: {cat}")
    logging.info(f"MODEL: {args.model_name}")
    # logging.info(f"CAT: {cat}")
    print("------------------------\n")

    # logging.info(f"NUM PROBS: {args.num_probs}")
    logging.info("------------------------\n")

    table.field_names = ["prompt", "summ", "location", "identity", "id", "bias type", "place_category"]

    if args.condition == 'gender':
        if args.add_in == 'base':
            csv_name = os.path.join(RES, '')  ##
        
        if args.add_in == 'ob':
            csv_name = os.path.join(RES, '') ##

        if args.add_in == 'llama3_2_11b':
            csv_name = os.path.join(RES, '')
        
        if args.add_in == 'qwen3_8b':
            csv_name = os.path.join(RES, '')
    
    if args.condition == 'race':
        if args.add_in == 'base':
            csv_name = os.path.join(RES, '')  ##
        
        if args.add_in == 'ob':
            csv_name = os.path.join(RES, '')

        if args.add_in == 'llama3_2_11b':
            csv_name = os.path.join(RES, '')
        
        if args.add_in == 'qwen3_8b':
            csv_name = os.path.join(RES, '')

    if args.condition == 'religions':
        if args.add_in == 'base':
            csv_name = os.path.join(RES, '')  ##
        
        if args.add_in == 'ob':
            csv_name = os.path.join(RES, '')

        if args.add_in == 'llama3_2_11b':
            csv_name = os.path.join(RES, '')
        
        if args.add_in == 'qwen3_8b':
            csv_name = os.path.join(RES, '')

    # Increase field size limit
    csv.field_size_limit(sys.maxsize)

    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=',')
        all_rows = list(reader)
    
    # args.num_probs = min(args.num_probs, len(all_rows))

    for index, row in enumerate(tqdm.tqdm(all_rows[1:])):
        # prompt,responses,location,identity,bias type
        p = row[0]
        response = row[1]
        location = row[2]
        identity = row[3]
        id = row[4]
        bias_type = row[5]
        place = row[6] if len(row) > 6 else 'N/A'

        with(open(f'{PROMPT_DIR}/res_combine1.txt', 'r')) as f:
            instruction = f.read()

        instruction = instruction.replace('[[D1]]', id)

        
        # Base prompt. The instruction corresponds to the above instruction prompts for diff. methods.
        prompt = f"""{instruction}

{response}"""
        
        predicted_answer = perspectiveModel.getOutput(prompt)


        table.add_row([prompt, predicted_answer, location, identity, id, bias_type, place])


    concept_dir = os.path.join('../SUMM_COMBINE')
    if not os.path.exists(concept_dir):
        os.mkdir(concept_dir)

    if "llama" in args.model_name:
        model_name = args.model_name[11:]
    elif "Qwen" in args.model_name:
        model_name = args.model_name[5:]
    elif "mistral" in args.model_name:
        model_name = args.model_name[10:]
    else:
        model_name = args.model_name

    save_concepts = pd.DataFrame(table._rows, columns=table.field_names) ###
    # save_concepts = pd.DataFrame(table_2._rows, columns=table_2.field_names) ### two steps analysis
    
    # T = strftime('%Y%m%d-%H%M')
    temp = str(args.temperature)

    save_concepts_dir = os.path.join(concept_dir, T + '_' + args.info_ + '_' + model_name + '_' + args.condition + '_' + args.add_in + '_all_summ_.csv')
    save_concepts.to_csv(save_concepts_dir, index = False, header=True)

    print("\n------------------------")
    print("         COMPLETE        ")
    print("------------------------")
    


def main():
    # python evaluate.py --model_name=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset=bbq --num_probs=10
    parser = argparse.ArgumentParser()
    # parser.add_argument('--condition', type=str, default='disambig') # disambig, ambig
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('--temperature', type=float, default=0.3)
    # parser.add_argument('--num_probs', '-n', type=int, default=200)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--gpu', type=int, default=0) # which gpu to load on
    parser.add_argument('--eight_bit', action='store_true') # load model in 8-bit? and quantized for llama 4?
    # parser.add_argument('--dataset', type=str, default='crows') # choose from crows, stereo_intra, stereo_inter, bbq_<category>
    parser.add_argument('--info_', type=str, default='one') # one, two1, two2
    parser.add_argument('--T', type=str, default="")
    parser.add_argument('--bias_cat', type=str, default ="gender_and_sex") # bias type
    parser.add_argument('--condition', type=str, default='gender') # gender, race, religions
    parser.add_argument('--add_in', type=str, default='base')
    
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

    if args.info_ == 'one':
        concepts_combine1()
    elif args.info_ == 'two1':
        concepts_combine2()

    
    logging.shutdown()

if __name__ == '__main__':
    main()