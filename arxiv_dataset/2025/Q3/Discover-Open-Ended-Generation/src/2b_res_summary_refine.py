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

DATA_DIR = '../data'
PROMPT_DIR = 'prompt_instructions'
RES = '../SUMMARY'
RES_O = '../results_responses'
T = strftime('%Y%m%d-%H%M')



def bias_summ():
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

    table_2.field_names = ["prompt","summ_diff", "summ_final", "location", "identity", "id1", "id2", "bias type", "place_category"]

    # fill in the csv files after you run for the previous step
    if args.condition == 'gender':
        # # open box two char
        if args.add_in == 'ob':
            RES_O = '../results_responses_patch2'
            model_n = 'Llama-3.2-3B-Instruct'

            csv_name_story = os.path.join(RES_O, model_n, '20250618-1344_generations_narra_1_3_gender_res_no_des_2char-2.csv') # 2 more 20 ob
            csv_name = os.path.join(RES, '')

        # cb
        if args.add_in == 'base':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250618-1342_0.9_Llama-3.2-3B-Instruct_narra_1_gender_no_des_base_20.csv') # 2 more 20 cb
            csv_name = os.path.join(RES, '') ###

        if args.add_in == 'simple2':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250623-1137_0.9_Llama-3.2-3B-Instruct_simple_2_gender.csv') # 2 more 20 cb
            csv_name = os.path.join(RES, '') ###

        if args.add_in == 'simple3':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250625-0955_0.9_Llama-3.2-3B-Instruct_simple_3_gender.csv')
            csv_name = os.path.join(RES, '')

        if args.add_in == 'llama3_2_11b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '') #
            csv_name_story = os.path.join(RES_O, '20250624-0834_0.9_Llama-3.2-11B-Vision-Instruct_narra_1_gender.csv')
        
        if args.add_in == 'qwen3_8b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, '20250628-1325_0.9_Qwen3-8B_narra_1_gender.csv') #

    elif args.condition == 'race':
        # race
        # cb
        if args.add_in == 'base':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250619-2135_0.9_Llama-3.2-3B-Instruct_narra_1_race_base_10.csv')
            csv_name = os.path.join(RES, '')
        
        if args.add_in == 'simple2':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250623-1138_0.9_Llama-3.2-3B-Instruct_simple_2_race.csv') # 2 more 20 cb
            csv_name = os.path.join(RES, '') ###

        if args.add_in == 'simple3':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250625-0955_0.9_Llama-3.2-3B-Instruct_simple_3_race.csv')
            csv_name = os.path.join(RES, '')

        if args.add_in == 'llama3_2_11b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '') #
            csv_name_story = os.path.join(RES_O, '20250624-0838_0.9_Llama-3.2-11B-Vision-Instruct_narra_1_race.csv') # cb
        
        if args.add_in == 'qwen3_8b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, '20250628-1325_0.9_Qwen3-8B_narra_1_race.csv') #

        # ob
        if args.add_in == 'ob':
            RES_O = '../results_responses_patch2'
            model_n = 'Llama-3.2-3B-Instruct'
            csv_name_story = os.path.join(RES_O, model_n, '20250620-2052_generations_narra_1_3_race_res_no_des_2char-2.csv') # 2 more 20 cb
            csv_name = os.path.join(RES, '') ###

    
    elif args.condition == 'religions':
        # religions
        # cb
        if args.add_in == 'base':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250619-2133_0.9_Llama-3.2-3B-Instruct_narra_1_religions_base_10.csv')
            csv_name = os.path.join(RES, '20250621-1133_two1_Qwen3-32B_religions_all_summ_cb_10.csv')
        
        if args.add_in == 'simple2':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250625-0954_0.9_Llama-3.2-3B-Instruct_simple_2_religions.csv') 
            csv_name = os.path.join(RES, '20250626-1015_two1_Qwen3-32B_religions_simple2_all_summ_cb.csv')

        if args.add_in == 'simple3':
            RES_O = '../results_responses'
            csv_name_story = os.path.join(RES_O, '20250625-0956_0.9_Llama-3.2-3B-Instruct_simple_3_religions.csv')
            csv_name = os.path.join(RES, '20250627-2031_two1_Qwen3-32B_religions_simple3_all_summ_cb.csv')

        if args.add_in == 'llama3_2_11b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '') #
            csv_name_story = os.path.join(RES_O, '20250624-0838_0.9_Llama-3.2-11B-Vision-Instruct_narra_1_religions.csv') # cb
        
        if args.add_in == 'qwen3_8b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, '20250628-1325_0.9_Qwen3-8B_narra_1_religions.csv') #

        # ob
        if args.add_in == 'ob':
            RES_O = '../results_responses_patch2'
            model_n = 'Llama-3.2-3B-Instruct'
            csv_name_story = os.path.join(RES_O, model_n, '20250620-2052_generations_narra_1_3_religions_res_no_des_2char-2.csv') # 2 more 20 cb
            csv_name = os.path.join(RES, '') ###
    

    # Increase field size limit
    csv.field_size_limit(sys.maxsize)
    
    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=',')
        all_rows = list(reader)

    with open(csv_name_story, "r") as f:
        reader = csv.reader(f, delimiter=',')
        all_rows_story = list(reader)
    
    # args.num_probs = min(args.num_probs, len(all_rows))

    # for index, row in enumerate(tqdm.tqdm(all_rows[1:])):
    for index, (row_story, row) in enumerate(tqdm.tqdm(zip(all_rows_story[1:], all_rows[1:]), total=len(all_rows_story) - 1)):
        # prompt,responses,location,identity,bias type
        p = row[0]
        response = row[1]
        location = row[2]
        identity = row[3]
        id1 = row[4]
        id2 = row[5]
        bias_type = row[6]
        place_category = row[7]

        response_story = row_story[1]

        # # test data
        # story = row[0]
        # id1 = row[3]
        # id2 = row[4]

        # prevent hallucination
        with(open(f'{PROMPT_DIR}/summary_2fil.txt', 'r')) as f:
            instruction_hallu = f.read()
        instruction_hallu = instruction_hallu.replace('[[D1]]', id1).replace('[[D2]]', id2)

        
        # Base prompt. The instruction corresponds to the above instruction prompts for diff. methods.
        if args.info_ == 'two1':
            prompt = f"""{instruction_hallu}

Original Story: {response_story}
{response}"""
            
            predicted_answer = perspectiveModel.getOutput(prompt)

            table.add_row([prompt, predicted_answer, location, identity, id1, id2, bias_type, place_category])


    concept_dir = os.path.join('../SUMMARY_B')
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

    if args.info_ == 'two1' or args.info_ == 'two2':
        save_concepts = pd.DataFrame(table._rows, columns=table.field_names) ###
    temp = str(args.temperature)

    save_concepts_dir = os.path.join(concept_dir, T + '_' + args.info_ + '_' + model_name + '_' + args.condition + '_' + args.add_in + '_all_summ.csv')
    save_concepts.to_csv(save_concepts_dir, index = False, header=True)

    print("\n------------------------")
    print("         COMPLETE        ")
    print("------------------------")



def bias_summ_single():
    # data = args.dataset

    # RES_O = '../results_responses_patch2'
    RES_O = '../results_responses'

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
        if args.part == 'base':
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, 'gender_one')
        
        if args.add_in == 'ob':
            RES_O = '../results_responses_patch2'
            model_n = 'Llama-3.2-3B-Instruct'

            csv_name_story = os.path.join(RES_O, model_n, '20250711-0006_generations_narra_1_single_simple_3_gender_res_no_des_2char-2.csv') 
            csv_name = os.path.join(RES, '')
        
        if args.add_in == 'llama3_2_11b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '') #
            csv_name_story = os.path.join(RES_O, '20250710-1157_0.9_Llama-3.2-11B-Vision-Instruct_narra_1_single_simple_gender_1_.csv') # cb
        
        if args.add_in == 'qwen3_8b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, '20250710-1158_0.9_Qwen3-8B_narra_1_single_simple_gender_1_.csv') #
    
    if args.condition == 'race':
        if args.part == 'base':
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, 'race_one')
        
        if args.add_in == 'ob':
            RES_O = '../results_responses_patch2'
            model_n = 'Llama-3.2-3B-Instruct'

            csv_name_story = os.path.join(RES_O, model_n, '20250711-0006_generations_narra_1_single_simple_3_race_res_no_des_2char-2.csv') 
            csv_name = os.path.join(RES, '')
        
        if args.add_in == 'llama3_2_11b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '') #
            csv_name_story = os.path.join(RES_O, '20250710-1157_0.9_Llama-3.2-11B-Vision-Instruct_narra_1_single_simple_race_1_.csv') # cb
        
        if args.add_in == 'qwen3_8b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, '20250710-1245_0.9_Qwen3-8B_narra_1_single_simple_race_1_.csv') #

    if args.condition == 'religions':
        if args.part == 'base':
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, 'religions_one')
        
        if args.add_in == 'ob':
            RES_O = '../results_responses_patch2'
            model_n = 'Llama-3.2-3B-Instruct'

            csv_name_story = os.path.join(RES_O, model_n, '20250711-0006_generations_narra_1_single_simple_3_religions_res_no_des_2char-2.csv') 
            csv_name = os.path.join(RES, '')
        
        if args.add_in == 'llama3_2_11b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '') #
            csv_name_story = os.path.join(RES_O, '20250710-1157_0.9_Llama-3.2-11B-Vision-Instruct_narra_1_single_simple_religions_1_.csv') # cb
        
        if args.add_in == 'qwen3_8b':
            RES_O = '../results_responses'
            csv_name = os.path.join(RES, '')
            csv_name_story = os.path.join(RES_O, '20250710-1614_0.9_Qwen3-8B_narra_1_single_simple_religions_1_.csv') #

    # Increase field size limit
    csv.field_size_limit(sys.maxsize)

    with open(csv_name_story, "r") as f:
        reader = csv.reader(f, delimiter=',')
        all_rows_story = list(reader)

    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=',')
        all_rows = list(reader)
    
    # args.num_probs = min(args.num_probs, len(all_rows))

    for index, (row_story, row) in enumerate(tqdm.tqdm(zip(all_rows_story[1:], all_rows[1:]), total=len(all_rows_story) - 1)):
        # prompt,responses,location,identity,bias type
        p = row[0]
        response = row[1]
        location = row[2]
        identity = row[3]
        id = row[4]
        bias_type = row[5]
        place_category = row[6]

        response_story = row_story[1]

        match = re.search(r"Sure! In this context, I will write a story:(.*?)(?:<\|eot_id\|>|$)", response_story, re.DOTALL)
        if match:
            story = match.group(1).strip()
            response_story = story

        # prevent hallucination
        with(open(f'{PROMPT_DIR}/summary_1fill.txt', 'r')) as f:
            instruction_hallu = f.read()
        instruction_hallu = instruction_hallu.replace('[[D1]]', id)
        prompt = f"""{instruction_hallu}

Original Story: {response_story}
{response}"""
            
        predicted_answer = perspectiveModel.getOutput(prompt)

        table.add_row([prompt, predicted_answer, location, identity, id, bias_type, place_category])


    concept_dir = os.path.join('../SUMMARY_B')
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

    save_concepts_dir = os.path.join(concept_dir, T + '_' + args.info_ + '_' + model_name + '_' + args.condition + '_' + args.add_in + '_all_summ.csv')
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
        bias_summ_single()
    elif args.info_ == 'two1':
        bias_summ()

    logging.shutdown()

if __name__ == '__main__':
    main()