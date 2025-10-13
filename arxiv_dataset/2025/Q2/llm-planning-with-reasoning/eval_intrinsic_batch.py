#%% imports
import os
from modules.llm_evaluator import LLMEvaluator, LLMLoraEvaluator, LLMReasoningEvaluator
from utils.functions import set_seed_all
from utils.functions import eval_intrinsic
import json
from datetime import datetime
import torch
import gc

#%% model names

evaluator_names =[
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # generation
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "stabilityai/stable-code-3b", # generation
    "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-ai/deepseek-coder-1.3b-instruct", # generation
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf"
]

# generate lora model names
lora_model_names = []
for m in evaluator_names:
   lora_model_names.append( m.split("/")[1]+"_spider")

#%% batch run
max_new_token_list = [512, 768]
for m in max_new_token_list:

# eval_settings = ['base','check']
# for es in eval_settings:

    #%% params

    model_indx = 1 # choose the model to evaluate
    evaluator_type = 'reason' # 'base' / 'FT' / 'reason'
    evaluator_setting = 'check' # 'base', 'check', 'exec', 'pro'
    ft_type = 'NoExecResult' # 'NoExecResult' / 'withExecResult'
    useSchema = False # Use schema as context for evaluator
    evaluator_faliValue = -0.5
    evaluator_max_new_tokens = m

    evaluator_name = evaluator_names[model_indx]
    model_savename = lora_model_names[model_indx] + "_" + ft_type # + '_b256_e1'

    print(f"evaluator_name: {evaluator_name}")
    print(f"evaluator_lora: {model_savename}")
    print(f"evaluator_type: {evaluator_type}")
    print(f"evaluator setup: {evaluator_setting}")
    print(f"FT training type: {ft_type}")
    print(f"Use Schema: {useSchema}")
    print(f"evaluator_max_new_tokens: {evaluator_max_new_tokens}")

    current_directory = os.getcwd() #parameters
    model_savedatapath = os.path.join(current_directory,f"checkpts/{model_savename}/model")
    evaluator_peft_dir = model_savedatapath

    seed = 42
    curr_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_fname = "data/spider_intrin_eval.json"
    # test_fname = "data/spider_evaluator_train.json"
    log_name = f"{model_savename}_{curr_timestamp}.json"
    dataset_name = "spider"
    db_path ="data/spider/database"
    evaluation_config = f"configs/{evaluator_setting}.json"
    print(f"evaluation_config = {json.load(open(evaluation_config))}")
    """
    yes_token_indx: 
        the index of the token in the vocabulary that corresponds to the "Yes" text.
        CodeLlama-Instruct: "No" 1939 "Yes" 3869
        TinyLlama: "Yes" 3869
    """
    yes_token_indx=None#3869

    params_log={
    "evaluator_name": evaluator_name,
    "evaluator_lora": model_savename,
    "evaluator_type": evaluator_type,
    "evaluator_setting": evaluator_setting,
    "evaluation_config": json.load(open(evaluation_config)),
    "FT training type:": ft_type,
    "Use Schema in prompt": useSchema,
    "seed": seed,
    "test_fname": test_fname,
    "evaluator_faliValue": evaluator_faliValue,
    "evaluator_max_new_tokens": evaluator_max_new_tokens
    }

    print(params_log)

    #%% set seed
    set_seed_all(seed)

    #%% load evaluator

    if evaluator_type == 'base':
        evaluator = LLMEvaluator(evaluator_name, db_path, device="cuda",yes_token_indx=yes_token_indx)
        print(f"Loaded base model")
    elif evaluator_type == 'FT':
        evaluator = LLMLoraEvaluator(evaluator_name, evaluator_peft_dir, db_path, device="cuda",yes_token_indx=yes_token_indx)
        print(f"Loaded LoRA FT model")
    else:
        evaluator = LLMReasoningEvaluator(base_model_name=evaluator_name, db_path=db_path, device="cuda",\
                                        failvalue=evaluator_faliValue, max_new_tokens=evaluator_max_new_tokens)
        print(f"Loaded for reasoning")
    # yindx=evaluator.get_yes_token()
    # print(f"Yes token index: {yindx}")    

    #%% run intrinsic eval

    eval_intrinsic(evaluator, test_fname, evaluation_config, log_fname=log_name, useSchema=useSchema,params_log=params_log)

    #%% clear
    del evaluator

    # Clear memory for PyTorch (if using GPU)
    torch.cuda.empty_cache()

    # Garbage collection to free up memory
    gc.collect()