"""
Preprocess dataset for sokoban task - see rage/env/sokoban/sokoban.py for details
"""

import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets

from ragen.env.sokoban import SokobanEnv


INSTRUCTION_TEMPLATE = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Answers:
<answer> Up </answer> | <answer> Down </answer> | <answer> Left </answer> | <answer> Right </answer>

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0


[Cumulative Observations]:
{observation}
Decide the next action:\
"""




templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--env", type=str, default="sokoban", help="Environment name (default: 'sokoban').")
    parser.add_argument("--algo", type=str, default="bfs", choices=["bfs"], help="Algorithm to use (default: 'bfs').")
    parser.add_argument("--seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/sokoban", help="Output file to save the trajectories (default: 'data/sokoban').")
    parser.add_argument("--train_size", type=int, default=300, help="Number of trajectories to generate (default: 3000).")
    parser.add_argument("--test_size", type=int, default=10, help="Number of trajectories to generate (default: 100).")
    parser.add_argument("--bfs_max_nodes", type=int, default=1000, help="Maximum number of nodes to use for BFS (default: 100000).") # not using this now. This will usually give the best traj. To compare with SFT, we will try this later.
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])

    args = parser.parse_args()
    
    assert args.env == "sokoban", "Unsupported environment: {args.env}"
    assert args.algo == "bfs", "Unsupported algorithm: {args.algo}"
    data_source = args.env
    
    dim_x, dim_y, num_boxes, max_steps, search_depth = os.environ.get("DIM_X"), os.environ.get("DIM_Y"), os.environ.get("NUM_BOXES"), os.environ.get("MAX_STEPS"), os.environ.get("SEARCH_DEPTH")
    dim_x, dim_y, num_boxes, max_steps, search_depth = int(dim_x), int(dim_y), int(num_boxes), int(max_steps), int(search_depth)

    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    instructions = []
    for seed in seeds:
        env = SokobanEnv(
            dim_room=(dim_x, dim_y),
            num_boxes=num_boxes,
            max_steps=max_steps,
            search_depth=search_depth
        )
        observation = env.reset(seed=seed, mode='tiny_rgb_array')
        instruction = INSTRUCTION_TEMPLATE.format(observation=observation)
        instructions.append(instruction)
    
    
    def _create_instance(idx, instruction):
        prompt_formatted = templates[args.prefix].format(prompt=instruction)

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": idx}
        }
    train_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i]) for i in range(args.train_size)])
    test_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i]) for i in range(args.train_size, args.train_size + args.test_size)])


    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn

    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))

if __name__ == "__main__":
    main()