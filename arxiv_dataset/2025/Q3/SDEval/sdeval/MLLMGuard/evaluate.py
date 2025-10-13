"""
This script is used to evaluate MLLMs.
"""
import argparse
import logging
import random
import time
import wandb

import torch

from utils import (
    process_data, process_data_json
)

def seed_all(seed = 8888):
    torch.manual_seed(seed)
    random.seed(seed)
    
def evaluate(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

@evaluate
def evaluate_model(model, args, data):
    model.batch_evaluate(args, data)    

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='gpt4v', help="specifies the model to be evaluated.")
    parser.add_argument('--openai', type=str, default=None, help="specifies the api_key")
    parser.add_argument('--tokenizer', type=str, default=None, help='specifies the tokenizer to be used.')
    
    parser.add_argument('--data_path', type=str, default='data/toxicity', help="specifies the path to the data")
    parser.add_argument('--log_file', type=str, default='logs/default.log', help='specifies the name of the log file')
    parser.add_argument('--save_path', type=str, default='results/toxicity_en.jsonl', help='specifies the path to save the results.')

    parser.add_argument('--verbose', type=bool, default=True, help='specifies whether to display verbose outputs.')
    
    parser.add_argument('--project_name', type=str, default='mllmguard', help='specifies the project name in wandb.')
    parser.add_argument('--entity_name', type=str, default='entity_name', help='specifies the entity name in wandb.')
    
    args = parser.parse_args()
    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    return args

def main(args):
    
    data = process_data_json(args.data_path)
    model_name = args.model.lower()
    print(model_name)

    if 'o3-2025-04-16' in model_name:
        from apis.gpt4o import GPT4o
        api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        gpt = GPT4o(api_key)
        evaluate_model(gpt, args, data)



    elif 'yi_plus' in model_name:
        from apis.yi_vl_plus import Yi_VL
        if args.model:
            yi_plus = Yi_VL(args.model)
            evaluate(yi_plus, args, data)
        else:
            raise ValueError('API Key is not specified.')
        
    elif 'gemini' in model_name:
        from apis.geminipro import Gemini
        if args.model:
            gemini = Gemini(args.model)
            evaluate(gemini, args, data)
            
    else:
        if 'llava' in model_name:
            from models.llava_inference import Llava
            llava = Llava(args.model)
            evaluate_model(llava, args, data)
        elif 'qwen-vl' in model_name:
            from models.qwen import QwenVL
            qwenvl = QwenVL(args.model)
            evaluate_model(qwenvl, args, data)
        elif 'qwen2-vl' in model_name:
            from models.qwen2_5 import Qwen2
            qwen = Qwen2(args.model)
            evaluate_model(qwen, args, data)

        # elif 'qwen2.5' in model_name:
        #     from models.qwen2_5 import Qwen2_5agent
        #     agent = Qwen2_5agent(args.model)
        #     agent.generate_text(args, data)
        #     evaluate_model(qwen, args, data)

        elif 'qwen' in model_name:
            from models.qwen2_5 import Qwen2_5
            qwen = Qwen2_5(args.model)
            evaluate_model(qwen, args, data)
        elif 'internvl' in model_name:
            from models.intervl import Intervlm
            intervl = Intervlm(args.model)
            evaluate_model(intervl, args, data)
        
        elif 'cogvlm' in model_name:
            from models.cogvlm import CogVLM
            cogvlm = CogVLM(args.model, args.tokenizer)
            evaluate_model(cogvlm, args, data)
        elif 'yi' in model_name:
            from models.yi import YIVL
            yi_vl = YIVL(args.model)
            evaluate_model(yi_vl, args, data)
        elif 'deepseek-vl2' in model_name:
            from models.deepseekvl2 import DeepSeek_v2
            deepseek = DeepSeek_v2(args.model)
            evaluate_model(deepseek, args, data)
            
        elif 'deepseek' in model_name:
            from models.deepseek import DeepSeek
            deepseek = DeepSeek(args.model)
            evaluate_model(deepseek, args, data)
        elif 'mplug-owl2' in model_name:
            from models.mplug import mPLUG_Owl2
            mplug2 = mPLUG_Owl2(args.model)
            evaluate_model(mplug2, args, data)
        elif 'mplug-owl' in model_name:
            from models.mplug import mPLUG_Owl
            mplug1 = mPLUG_Owl(args.model)
            evaluate_model(mplug1, args, data)
        elif 'seed-llama-14b' in model_name:
            from models.seed import SeedLLaMA14B
            seed = SeedLLaMA14B(args.model)
            evaluate_model(seed, args, data)
        elif 'seed-llama-8b' in model_name:
            from models.seed import SeedLLaMA8B
            seed = SeedLLaMA8B(args.model)
            evaluate_model(seed, args, data)
        elif 'minigpt-v2' in model_name:
            from models.minigptv2 import MiniGPTV2
            minigptv2 = MiniGPTV2(args.model, data)
            evaluate_model(minigptv2, args, data)
        elif 'sharegpt' in model_name:
            from models.sharegpt4v import ShareGPT
            sharegpt = ShareGPT(args.model, data)
            evaluate_model(sharegpt, args, data)
        elif 'xcomposer' in model_name:
            from models.xcomposer import Xcomposer
            xcomposer = Xcomposer(args.model, data)
            evaluate_model(xcomposer, args, data)
        else:
            raise NotImplementedError(
                f'Model {model_name} has not been implemented.'
            )
            
if __name__ == "__main__":
    args = get_args()
    # wandb.init(project=args.project_name, entity=args.entity_name, config=args)
    main(args)
    seed_all(5555)