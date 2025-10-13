#%% 
from accelerate.utils import set_seed
from tqdm import tqdm

import json
import numpy as np
import random
import torch
from utils.functions import set_seed_all, set_result_filename, run_end2end, check_nltk_resource, swap_memory
import os
from datetime import datetime

from modules.llm_evaluator import LLMEvaluator, LLMLoraEvaluator, LLMReasoningEvaluator
from modules.llm_generator import LLMGenerator, LLMLoraGenerator
import subprocess
import gc

#%%
llm_names =[
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # generation
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "stabilityai/stable-code-3b", # generation
    "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-ai/deepseek-coder-1.3b-instruct", # generation
    "codellama/CodeLlama-7b-Instruct-hf"
]


# generate lora model names
lora_model_names = []
for m in llm_names:
   lora_model_names.append( m.split("/")[1]+"_spider")

#%% batch
model_indices = [4]
all_settings = ['base', 'check']

for m in model_indices:
    for ss in all_settings:
        #%% params

        seed = 42
        device_swap = True # swap between cuda and cpu to save vram
        evaluator_type = 'base' # 'base' / 'FT' / 'reason'
        evaluator_setting = ss # 'base', 'check', 'exec', 'pro'
        ft_type = 'NoExecResult' # 'withExecResult'

        # evaluator
        model_indx = m # choose the model to evaluate
        
        evaluator_faliValue = -0.5
        evaluator_max_new_tokens = 512
        useSchema = False # Use schema as context for evaluator
        evaluator_name = llm_names[model_indx] #base model name
        model_savename = lora_model_names[model_indx] + "_" + ft_type #lora model save name
        evaluation_config = f"configs/{evaluator_setting}.json"


        print(f"evaluator_name: {evaluator_name}")
        print(f"evaluator_lora: {model_savename}")
        print(f"evaluator FT training type: {ft_type}")
        print(f"evaluator_type: {evaluator_type}")
        print(f"evaluator setup: {evaluator_setting}")
        print(f"FT training type: {ft_type}")
        print(f"Use Schema: {useSchema}")
        print(f"evaluator_max_new_tokens: {evaluator_max_new_tokens}")
        print(f"evaluation_config = {json.load(open(evaluation_config))}\n")

        # generator
        model_indx = 2 # choose the model to generate
        generator_name = llm_names[model_indx] #base model name
        generator_lora_savename = lora_model_names[model_indx] #lora model save name
        print(f"generator_name: {generator_name}")
        #print(f"generator_lora: {generator_lora_savename}")

        prompt_method = 0 # 0 for tinyllama

        # populate other parameters
        current_directory = os.getcwd() #parameters
        model_savedatapath = os.path.join(current_directory,f"checkpts/{model_savename}/model")
        evaluator_peft_dir = model_savedatapath

        curr_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_fname = "data/spider_dev_400.json"
        dataset_name = "spider"
        db_path ="data/spider/database"
        method_name = "rerank" # planning method: greedy, rerank, iterative, treesearch, prune
        # result_fname: where the results will be saved for evaluation
        result_fname = f"results/{set_result_filename(evaluator_name, generator_name, dataset_name, method_name, evaluator_type)}_{evaluator_setting}_e2e_t2_{curr_timestamp}" + ".sql" # where the results will be saved for evaluation
        log_name = f"log/{set_result_filename(evaluator_name, generator_name, dataset_name, method_name, evaluator_type)}_{evaluator_setting}_e2e_t2_{curr_timestamp}" + ".json" # log for result generation
        result_eval_fname = f"results/{set_result_filename(evaluator_name, generator_name, dataset_name, method_name, evaluator_type)}_{evaluator_setting}_e2e_t2_{curr_timestamp}" + ".txt" # evaluation report

        retriever_gen = None # retriever generator
        retriever_eval = None # retriever evaluator

        """
        yes_token_indx: 
            the index of the token in the vocabulary that corresponds to the "Yes" text.
            CodeLlama-Instruct: "No" 1939 "Yes" 3869
            TinyLlama: "Yes" 3869
        """
        yes_token_indx=None #3869

        # print parameters
        print(f"prompt_method: {prompt_method}")
        print(f"method_name: {method_name}")
        print(f"dataset_name: {dataset_name}")
        print(f"\nResult output file name: {result_fname}\nLog file name: {log_name}\nEvaluation result file name: {result_eval_fname}")


        #%% set seed
        set_seed_all(seed)

        #%% Load evaluator LM
        if method_name != "greedy":
            if evaluator_type == 'base':
                evaluator = LLMEvaluator(evaluator_name, db_path, device="cuda",yes_token_indx=yes_token_indx)
                print(f"Loaded base model")
            elif evaluator_type == 'FT':
                evaluator = LLMLoraEvaluator(evaluator_name, evaluator_peft_dir, db_path, device="cuda",yes_token_indx=yes_token_indx)
                print(f"Loaded LoRA FT model")
            else:
                evaluator = LLMReasoningEvaluator(evaluator_name, db_path, device="cuda",\
                                            failvalue=evaluator_faliValue, max_new_tokens=evaluator_max_new_tokens)
                print(f"Loaded for reasoning")

            # move model to cpu for now
            if device_swap:
                swap_memory(evaluator.model, device="cpu",verbose=True)

            #yindx=evaluator.get_yes_token()
            #print(f"Yes token index: {yindx}")
        else:
            evaluator = None
            print('Generated empty evaluator')    

        #%% Load generator

        generator = LLMGenerator(generator_name, device="cuda")
        #generator = LLMLoraGenerator(generator_name, generator_peft_dir, device="cuda")

        # move model to cpu for now
        if device_swap:
            swap_memory(generator.model, device="cpu",verbose=True)

        # param check
        if method_name == "greedy" and device_swap:
            print("Caution: you enabled device_swap during Greedy method. This will be slow..")

        #%% planning method

        if method_name == "rerank":
            from modules.llm_planner import rerank as planner
            generation_config = "configs/temp_sampling.json" # there are two configs for generation: temp_sampling.json (5 candidates) and greedy.json (1 candidate)

        elif method_name == "greedy":
            from modules.llm_planner import greedy as planner
            generation_config = "configs/greedy.json"

        elif method_name == "iterative":
            from modules.llm_planner import iter_correction as planner
            generation_config = "configs/temp_sampling.json"

        elif method_name == "prune":
            from modules.llm_planner import prune as planner
            generation_config = "configs/temp_sampling.json"

        elif method_name == "treesearch":
            from modules.llm_planner import tree_search_mc as planner
            generation_config = "configs/temp_sampling.json"

        else:
            raise ValueError(f"Unknown planning method: {method_name}")
        print(f"LLM planner: {method_name}")


        #%% Summary

        print(f"evaluator_name: {evaluator_name}")
        print(f"evaluator_lora: {model_savename}")
        print(f"evaluator_type: {evaluator_type}")
        print(f"evaluator setup: {evaluator_setting}")
        print(f"evaluation_config = {json.load(open(evaluation_config))}\n")
        print(f"generator_name: {generator_name}")
        print(f"prompt_method: {prompt_method}")
        print(f"method_name: {method_name}")
        print(f"dataset_name: {dataset_name}")
        print(f"LLM planner: {method_name}")
        print(f"generation_config = {json.load(open(generation_config))}")
        print(f"\nResult output file name: {result_fname}\nLog file name: {log_name}\nEvaluation result file name: {result_eval_fname}")


        params_log={
        "evaluator_name": evaluator_name,
        "evaluator_lora": model_savename,
        "evaluator_peft_dir": evaluator_peft_dir,
        "evaluator_type": evaluator_type,
        "evaluator_setting": evaluator_setting,
        "evaluation_config": json.load(open(evaluation_config)),
        "FT training type:": ft_type,
        "Use Schema in prompt": useSchema,
        "seed": seed,
        "test_fname": test_fname,
        "evaluator_faliValue": evaluator_faliValue,
        "evaluator_max_new_tokens": evaluator_max_new_tokens,
        "generator_name": generator_name,
        "generation_config": json.load(open(generation_config)),
        "prompt_method": prompt_method,
        "method_name": method_name,
        "dataset_name": dataset_name,
        "Result output file name": result_fname,
        "Log file name": log_name,
        "Evaluation result file name": result_eval_fname
        }

        print(json.dumps(params_log,indent=2))

        #%% Prepare test

        gold_sql = "data/spider_dev_400_gold.sql" # Gold truth for test dataset
        temp_gold_sql = "data/temp_spider.sql" # Where selected gold truth for our problems will be

        # load main test dataset
        print("Main test dataset:")
        test_data = json.load(open(test_fname))
        print(f"Total: {len(test_data)}\nKeys: {test_data[0].keys()}")
        print(f"Diffulty values: {set(t['difficulty'] for t in test_data)}")

        # Full test:
        #"""
        temp_gold_sql = gold_sql
        test_data_subset = test_data
        #"""

        # Partial test:
        """
        # make a sub-set mask
        mask = [False] * len(test_data)
        for i in range(5):
            mask[i] = True
            #mask = [ t['difficulty']=='medium' for t in test_data ]

        # apply mask on test dataset
        test_data_subset = [t for t, m in zip(test_data, mask) if m]
        print(f"Total: {len(test_data_subset)}") 

        # Step 1: Open the gold .sql file and read its content
        with open(gold_sql, 'r') as file:
            glist = file.readlines()

        # Step 2: apply same mask
        glist_new = [g for g, m in zip(glist, mask) if m ]

        # Step 3: check
        for i in range(len(glist_new)):
            if glist_new[i] != test_data_subset[i]:
                Warning('There is a problem. Your testing will be wrong.')

        # Step 4: save temp sql
        with open(temp_gold_sql, 'w') as file:
            file.writelines(glist_new)
        """


        print("\nDataset Ready for evaludation")
        print(f"Total: {len(test_data_subset)}\nKeys: {test_data_subset[0].keys()}")
        print(f"Diffulty values: {set(t['difficulty'] for t in test_data_subset)}")

        #%% run

        run_end2end(generator, evaluator,generation_config, \
                    evaluation_config, planner, retriever_gen, retriever_eval, \
                        test_data_subset,dataset_name,result_fname,log_name,device_swap,prompt_method,\
                            useSchema=useSchema,params_log=params_log)


        #%% eval results

        db = db_path # the directory that contains all the databases and test suites
        table = "data/spider/tables.json" # the tables.json schema file
        pred = result_fname # the path to the predicted queries
        gold = temp_gold_sql #"data/spider_dev_400_gold.sql" # the path to the gold queries
        etype = "all" # evaluation type, exec for test suite accuracy, match for the original exact set match accuracy
        pscript = "test-suite-sql-eval/evaluation.py" # the evaluation script

        cmd = [
            "python", "-u", pscript,
            "--gold", gold,
            "--pred", pred,
            "--db", db,
            "--table", table,
            "--etype", etype
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)  # Check for errors

        #%% save
        # Save output to a text file
        with open(result_eval_fname, "w") as f:
            f.write(result.stdout)

        # Optional: Print confirmation
        print(f"Output saved to: {result_eval_fname}")

        #%% clear
        del evaluator
        del generator
        # Clear memory for PyTorch (if using GPU)
        torch.cuda.empty_cache()

        # Garbage collection to free up memory
        gc.collect()
        # %%
