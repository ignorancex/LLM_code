import os
import json
import argparse
from multi import multi
from binary import cot, base

def run(args):
    os.makedirs(args.output, exist_ok = True)

    system_prompt = json.load(open("prompt.json"))
    if args.task == "multi":
        multi(args, system_prompt["multi_system_prompt"])
    elif args.task == "binary_base":
        base(args, system_prompt["binary_base_system_prompt"])
    else:
        cot(args, system_prompt["binary_cot_system_prompt"])



def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'], default='gpt-4')
    args.add_argument('--temperature', type=float, default=0)
    args.add_argument('--task', type=str, required=True, choices=['multi', 'binary_base', 'binary_cot'])
    args.add_argument('--openai_key', type=str, required=True)
    args.add_argument("--data", type=str, default="../data")
    args.add_argument("--output", type=str, default="output")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)