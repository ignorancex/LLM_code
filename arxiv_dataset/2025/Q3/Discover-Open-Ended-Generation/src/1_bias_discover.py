import os
import random
import csv
import tqdm
import argparse
import wandb
import logging
from time import strftime
import sys
from llm_utils import *
from prettytable import PrettyTable
import pandas as pd
from datasets import load_dataset
import itertools


DATA_DIR = '../data'
PROMPT_DIR = 'prompt_instructions'
T = strftime('%Y%m%d-%H%M')



def evaluate_condition():
    # data = args.dataset

    if "gpt-4" in args.model_name:
        perspectiveModel = ChatGPT(args.model_name, temperature=args.temperature, verbose=args.verbose)
    elif "gpt" in args.model_name: # hf gpt 2 model
        perspectiveModel = GPT_2(args.model_name, temperature=args.temperature, verbose=args.verbose)
    elif "falcon" in args.model_name: # hf gpt 2 model
        perspectiveModel = FALCON(args.model_name, temperature=args.temperature, verbose=args.verbose, load_in_4bit=True)
    elif "Qwen3" in args.model_name:
        perspectiveModel = QWEN3(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Qwen" in args.model_name:
        perspectiveModel = QWEN(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Mistral" in args.model_name:
        perspectiveModel = MISTRAL(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Llama-4" in args.model_name:
        perspectiveModel = LLM4(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Llama" in args.model_name and 'Vision' not in args.model_name:
        perspectiveModel = LLM(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Llama" in args.model_name and 'Vision' in args.model_name:
        perspectiveModel = LLM3_2_V(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "command" in args.model_name:
        perspectiveModel = COHERE(args.model_name, temperature=args.temperature, verbose=args.verbose)

    if "gpt-4" not in args.model_name and "command" not in args.model_name:
        device_map = perspectiveModel.model.device
        print("Device Map:", device_map)

        # Check if any part of the model is on a GPU
        if device_map.type == 'cuda':
            print("The model is using a GPU.")
        else:
            print("The model is not using any GPU.")
            return 0

    table = PrettyTable()
    table_single = PrettyTable()
    
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

    # descriptor = json.load(open("../data/descriptors.json"))
    descriptor = json.load(open("../data/des_ex.json"))
    place = json.load(open("../data/places.json"))

    place_list = list(place.keys())
    

    table.field_names = ["prompt", "responses", "location", "identity", "id1", "id2", "bias type", "place_category"]
    table_single.field_names = ["prompt", "responses", "location", "identity", "id", "bias type", "place_category"]

    # info_ = ""

    if args.info_ == 'narra_1':
        with(open(f'{PROMPT_DIR}/evaluate1.txt', 'r')) as f:
            instruction = f.read()
    if args.info_ == 'narra_1_single_simple':
        with(open(f'{PROMPT_DIR}/evaluate_single1.txt', 'r')) as f:
            instruction = f.read()
    if args.info_ == 'simple_2':
        with(open(f'{PROMPT_DIR}/eva_simple2.txt', 'r')) as f:
            instruction = f.read()
    if args.info_ == 'simple_3':
        with(open(f'{PROMPT_DIR}/eva_simple_neg.txt', 'r')) as f:
            instruction = f.read()

    names = ['Dana', 'Donald', 'James', 'Taylor', 'Jim']

    for p in tqdm.tqdm(place_list):
        for location in tqdm.tqdm(place[p]['places']):
            # gender
            if args.condition == 'gender':
                total2 = len(descriptor['gender_and_sex']['female'])
                for i in range(total2):
                    female = descriptor['gender_and_sex']['female'][i]
                    male = descriptor['gender_and_sex']['male'][i]

                    if args.info_ == 'narra_1' or args.info_ == 'simple_2' or args.info_ == 'simple_3':
                        for _ in range(20): ###
                            prompt = instruction.replace('[[D1]]', female).replace('[[D2]]', male).replace('[[LOC]]', location)
                        
                            predicted_answer = perspectiveModel.getOutput(prompt)

                            table.add_row([prompt, predicted_answer, location, 'female, male', female, male, 'gender', p])


                    # single ###
                    if args.info_ == 'narra_1_single_simple':
                        for _ in range(20):
                            prompt = instruction.replace('[[D1]]', female).replace('[[LOC]]', location)
                            
                            predicted_answer = perspectiveModel.getOutput(prompt)

                            table_single.add_row([prompt, predicted_answer, location, 'female', female, 'gender', p])

                            prompt = instruction.replace('[[D1]]', male).replace('[[LOC]]', location)
                            
                            predicted_answer = perspectiveModel.getOutput(prompt)

                            table_single.add_row([prompt, predicted_answer, location, 'male', male, 'gender', p])
            
            # race
            elif args.condition == 'race':
                for i in range(len(descriptor['race_ethnicity']['asian'])):
                    white = descriptor['race_ethnicity']['white'][i][0]
                    asian = descriptor['race_ethnicity']['asian'][i][0]
                    black = descriptor['race_ethnicity']['black'][i][0]
                    # latinx = descriptor['race_ethnicity']['latinx'][i][0]
                    middle_east = descriptor['race_ethnicity']['middle-east'][i][0]

                    # race_des_dict = {'white': white, 'asian': asian, 'black': black, 'latinx': latinx, 'middle-east': middle_east}
                    # race_type = ['white', 'asian', 'black', 'latinx', 'middle-east']

                    race_des_dict = {'white': white, 'asian': asian, 'black': black, 'middle-east': middle_east}
                    race_type = ['white', 'asian', 'black', 'middle-east']

                    all_combine = list(itertools.combinations(race_type, 2))
                    for comb in all_combine:
                        race1 = comb[0]
                        race2 = comb[1]
                        if args.info_ == 'narra_1' or args.info_ == 'simple_2' or args.info_ == 'simple_3':
                            for _ in range(10): ##
                                prompt = instruction.replace('[[D1]]', race_des_dict[race1]).replace('[[D2]]', race_des_dict[race2]).replace('[[LOC]]', location)
                                # prompt = f"""Please write a short story about {race_des_dict[race1]} and {race_des_dict[race2]} at the location of {location}. Keep the story diverse, concise, plain, and logical, with a maximum of 10 sentences. (IMPORTANT: Output only the story itself; do not include any extra text.)"""

                                ids = race1 + ', ' + race2

                                predicted_answer = perspectiveModel.getOutput(prompt)

                                table.add_row([prompt, predicted_answer, location, ids, race_des_dict[race1], race_des_dict[race2], 'race', p])
                        ##############
                    
                        ### single
                        if args.info_ == 'narra_1_single_simple':
                            for _ in range(10):
                                prompt = instruction.replace('[[D1]]', race_des_dict[race1]).replace('[[LOC]]', location)
                            
                                predicted_answer = perspectiveModel.getOutput(prompt)

                                table_single.add_row([prompt, predicted_answer, location, race1, race_des_dict[race1], 'race', p])

                                prompt = instruction.replace('[[D1]]', race_des_dict[race2]).replace('[[LOC]]', location)
                                
                                predicted_answer = perspectiveModel.getOutput(prompt)

                                table_single.add_row([prompt, predicted_answer, location, race2, race_des_dict[race2], 'race', p])
            ## religions
            elif args.condition == 'religions':
                for i in range(len(descriptor['religions']['Christian'])):

                    christian = descriptor['religions']['Christian'][i]
                    jew = descriptor['religions']['Jewish'][i]
                    mus = descriptor['religions']['Muslim'][i]
                    budd = descriptor['religions']['Buddhist'][i]

                    reli_des_dict = {'christian': christian, 'jew': jew, 'mus': mus, 'budd': budd}
                    reli_type = ['christian', 'jew', 'mus', 'budd']

                    all_combine = list(itertools.combinations(reli_type, 2))
                    for comb in all_combine:
                        reli1 = comb[0]
                        reli2 = comb[1]
                        if args.info_ == 'narra_1' or args.info_ == 'simple_2' or args.info_ == 'simple_3':
                            for _ in range(10): 
                                prompt = instruction.replace('[[D1]]', reli_des_dict[reli1]).replace('[[D2]]', reli_des_dict[reli2]).replace('[[LOC]]', location)

                                ids = reli1 + ', ' + reli2

                                predicted_answer = perspectiveModel.getOutput(prompt)

                                table.add_row([prompt, predicted_answer, location, ids, reli_des_dict[reli1], reli_des_dict[reli2], 'religions', p])

                        ### single
                        if args.info_ == 'narra_1_single_simple':
                            for _ in range(10):
                                prompt = instruction.replace('[[D1]]', reli_des_dict[reli1]).replace('[[LOC]]', location)

                                predicted_answer = perspectiveModel.getOutput(prompt)

                                table_single.add_row([prompt, predicted_answer, location, reli1, reli_des_dict[reli1], 'religions', p])

                                prompt = instruction.replace('[[D1]]', reli_des_dict[reli2]).replace('[[LOC]]', location)

                                predicted_answer = perspectiveModel.getOutput(prompt)

                                table_single.add_row([prompt, predicted_answer, location, reli2, reli_des_dict[reli2], 'religions', p])


    concept_dir = os.path.join('../results_responses')
    if not os.path.exists(concept_dir):
        os.mkdir(concept_dir)

    if "llama" in args.model_name:
        model_name = args.model_name[11:]
    elif "Qwen" in args.model_name:
        model_name = args.model_name[5:]
    elif "mistral" in args.model_name:
        model_name = args.model_name[10:]
    else:
        model_name = args.model_name ### change this

    if args.info_ == 'narra_1' or args.info_ == 'simple_2' or args.info_ == 'simple_3':
        save_concepts = pd.DataFrame(table._rows, columns=table.field_names)

        temp = str(args.temperature)
        save_concepts_dir = os.path.join(concept_dir, T + '_' + temp + '_' + model_name + '_' + args.info_ + '_' + args.condition + '_cont_.csv')
        save_concepts.to_csv(save_concepts_dir, index = False, header=True)
    else:
        save_concepts = pd.DataFrame(table_single._rows, columns=table_single.field_names)
        temp = str(args.temperature)

        save_concepts_dir = os.path.join(concept_dir, T + '_' + temp + '_' + model_name + '_' + args.info_ + '_' + args.condition + '_' + args.part + '_.csv')
        save_concepts.to_csv(save_concepts_dir, index = False, header=True)

    print("\n------------------------")
    print("         COMPLETE        ")
    print("------------------------")
    


def main():
    # python evaluate.py --model_name=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset=bbq --num_probs=10
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='gender') # gender, race, religions
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('--temperature', type=float, default=0.8)
    # parser.add_argument('--num_probs', '-n', type=int, default=200)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--gpu', type=int, default=0) # which gpu to load on
    parser.add_argument('--eight_bit', action='store_true') # load model in 8-bit?
    parser.add_argument('--T', type=str, default="")
    parser.add_argument('--info_', type=str, default='narra_1') #
    
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

    evaluate_condition()
    

    logging.shutdown()

if __name__ == '__main__':
    main()