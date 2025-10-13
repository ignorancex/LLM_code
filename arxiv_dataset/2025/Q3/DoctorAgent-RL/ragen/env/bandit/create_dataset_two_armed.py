"""
Preprocess dataset for bandit task
"""

import os
import json
from datasets import Dataset
import argparse
from ragen.env import TwoArmedBanditEnv
import random

templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}

intro = """You are playing a two-armed bandit game. Goal: Maximize your total reward by choosing which arm to pull.
x
Game Rules:
1. There are 2 arms, named {name_a} and {name_b}
2. Each arm has its own reward distribution, related to their names. 
3. Analyze the symbolic meaning of each arm's name to guess how their reward distribution might behave.
4. Based on the symbolic meaning of their names, which arm do you think is more likely to give higher rewards on average? Choose between {name_a} and {name_b}, and output like <answer> [{name_a} or {name_b}] </answer>.
Current State:
{observation}
Think and choose which arm to pull:\
"""

def main():
    parser = argparse.ArgumentParser(description="Generate trajectories for two-armed bandit environment.")
    parser.add_argument("--env", type=str, default="two_armed_bandit", help="Environment name (default: 'two_armed_bandit').")
    parser.add_argument("--seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/two_armed_bandit", help="Output directory (default: 'data/two_armed_bandit').")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training instances (default: 10000).")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of test instances (default: 1000).")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])

    args = parser.parse_args()
    
    assert args.env == "two_armed_bandit", f"Unsupported environment: {args.env}"
    os.makedirs(args.output, exist_ok=True)
    data_source = args.env

    low_risk_name = os.environ.get("LOW_RISK_NAME")
    high_risk_name = os.environ.get("HIGH_RISK_NAME")
    
    low_risk_val_name = os.environ.get("LOW_RISK_VAL_NAME", low_risk_name)
    high_risk_val_name = os.environ.get("HIGH_RISK_VAL_NAME", high_risk_name)

    if low_risk_val_name != low_risk_name:
        print("[INFO] YOU ARE USING DIFFERENT TRAIN/VAL LOW-ARM NAMES.")
        
    if high_risk_val_name != high_risk_name:
        print("[INFO] YOU ARE USING DIFFERENT TRAIN/VAL HIGH-ARM NAMES.")

    # Generate instructions
    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    
    
    # Generate training instructions
    instructions = []
    for seed in range(args.seed, args.seed + args.train_size):
        env = TwoArmedBanditEnv(low_risk_name=low_risk_name, high_risk_name=high_risk_name, seed=seed)
        observation = env.reset(seed=seed)
        # shuffle to make prompt more robust
        names = [low_risk_name, high_risk_name]
        rng = random.Random(seed)
        rng.shuffle(names)  # shuffle in place using seeded random number generator
        instruction = intro.format(observation=observation, name_a=names[0], name_b=names[1])
        instructions.append(instruction)

    # Generate validation instructions
    for seed in range(args.seed + args.train_size, args.seed + args.train_size + args.test_size):
        env = TwoArmedBanditEnv(low_risk_name=low_risk_val_name, high_risk_name=high_risk_val_name, seed=seed)
        observation = env.reset(seed=seed)
        # shuffle to make prompt more robust
        names = [low_risk_val_name, high_risk_val_name]
        rng = random.Random(seed)
        rng.shuffle(names)  # shuffle in place using seeded random number generator
        instruction = intro.format(observation=observation, name_a=names[0], name_b=names[1])
        instructions.append(instruction)


    def _create_instance(idx, instruction):
        prompt_formatted = templates[args.prefix].format(prompt=instruction)
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "rl",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": idx}
        }

    # Create datasets
    train_dataset = Dataset.from_list([
        _create_instance(args.seed + i, instructions[i]) 
        for i in range(args.train_size)
    ])
    
    test_dataset = Dataset.from_list([
        _create_instance(args.seed + i, instructions[i]) 
        for i in range(args.train_size, args.train_size + args.test_size)
    ])

    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Save datasets
    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))

if __name__ == "__main__":
    main()