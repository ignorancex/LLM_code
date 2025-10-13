import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='QwQ')
args = parser.parse_args()

chain_type_list = ['parallel', 'forward', 'backward', 'chaotic']
k_list = [5, 10, 20, 50, 100, 200]
question_type_list = ['single', 'total']


for chain_type in chain_type_list:
    for k in k_list:
        for question_type in question_type_list:
            output_name = f"{args.model}__{chain_type}__k{str(k)}__{question_type}"
            print(output_name)
            os.system(f"""
python inference_call.py \
--model_name {args.model} \
--output_name {output_name} \
--chain_type {chain_type} \
--question_type {question_type} \
--k {k} \
--results_dir ./results
wait
""")

#     --tool \

# nohup python inference_all.py --model QwQ > logs/inference &  # 11
# nohup python inference_all.py --model qwen2.5-32B > logs/inference &  # 12
# nohup python inference_all.py --model qwen2.5-32B > logs/inference &  # 12

# nohup python inference_all.py --model qwen_long > logs/inference &  # 12

# nohup python inference_all.py --model llama3.3-70B > logs/inference &  # 13

# nohup python inference_all.py --model qwen3-32B > logs/inference &  # 12
# nohup python inference_all.py --model gemma3 > logs/inference &  # 16

# nohup python inference_all.py --model gpt-4o > logs/gpt &
# nohup python inference_all.py --model o3 > logs/inference &  # 16
# nohup python inference_all.py --model qwen2.5-DS > logs/inference &  # 16
# export VLLM_ATTENTION_BACKEND="XFORMERS"
