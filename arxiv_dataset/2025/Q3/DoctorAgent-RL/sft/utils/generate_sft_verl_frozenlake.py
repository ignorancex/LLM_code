"""
Generate SFT data for finetuning using verl framework on frozenlake environment
transform into parquet format

Conversation data:
    - LLM needs to output ground-truth action sequence for each environment
    - human needs to provide the current state of the sokoban environment after action
    - data should be formatted (qwen) like <|im_start|>user...<|im_end|>, <|im_start|>assistant...<|im_end|>, ...


NOTE seed is set as 100000 to avoid overlap with training data for RAGEN
"""

import os
import argparse
from datasets import Dataset
import copy
from multiprocessing import Pool
from tqdm import tqdm

from ragen.env import FrozenLakeEnv
from ragen.env.frozen_lake.utils import get_shortest_action_path, plot_animation


INSTRUCTION_TEMPLATE = """You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G).

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Answers:
<answer> Left </answer> | <answer> Down </answer> | <answer> Right </answer> | <answer> Up </answer>

Rewards:
Fall into hole: 0
Reach goal: +1.0


[Cumulative Observations]:
{observation}
Decide the next action:\
"""








######################################## different language model input templates ########################################

qwen_instruction_template = """\
<|im_start|>user
{prompt}
Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>
<|im_start|>assistant
<think>\
"""
qwen_observation_template = """\
<|im_start|>user
After you take this action, the observation is: 
{observation}
<|im_end|>
<|im_start|>assistant
<think>\
"""
qwen_response_template = """\
</think> <answer> {action} </answer> <|im_end|>
"""

system_message = "You are a helpful assistant."
instruction_message = "{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra test. Strictly follow this format."
observation_message = "After you take this action, the observation is: \n{observation}\n"
response_message = "<answer> {action} </answer>"
# NOTE in message format, <think> token is not included here

base_instruction_template = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.
User: {prompt}
Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>
Assistant: 
<think>\
"""
base_observation_template = """\
User: After you take this action, the observation is: {observation}
Assistant: 
<think>\
"""
base_response_template = """\
</think> <answer> {action} </answer>
"""

templates = {
    'qwen-instruct': {
        'instruction': qwen_instruction_template,
        'observation': qwen_observation_template,
        'response': qwen_response_template
    },
    'message': {
        'system': system_message,
        'instruction': instruction_message,
        'observation': observation_message,
        'response': response_message
    },
    'base': {
        'instruction': base_instruction_template,
        'observation': base_observation_template,
        'response': base_response_template
    }
}









def create_chat_str_for_env(args):
    """
    Chat is formated as pure string like <|im_start|>user...<|im_end|>, <|im_start|>assistant...<|im_end|>, ...
    """
    seed, prefix, data_source, MAX_DEPTH, SIZE, P, IS_SLIPPERY = args
    instance_template = {
        "data_source": data_source,
        "prompt": None,
        "response": None,
        "ability": "bfs",
        "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
        "extra_info": {"split": "train", "index": seed}
    }
    instances = []

    env = FrozenLakeEnv(size=SIZE, p=P, is_slippery=IS_SLIPPERY, seed=seed)
    obs = env.reset(seed=seed, reset_map=False, mode='tiny_rgb_array')
    init_prompt_formatted = templates[prefix]['instruction'].format(prompt=INSTRUCTION_TEMPLATE.format(observation=obs))

    # images = []

    # images.append(env.render('rgb_array'))
    gt_action_sequence = get_shortest_action_path(
        seed=seed,
        size=env.map_kwargs['size'],
        p=env.map_kwargs['p'],
        is_slippery=env.env_kwargs['is_slippery'],
        MAX_DEPTH=MAX_DEPTH
    )
    assert gt_action_sequence, f"No action sequence found for seed {seed}"
    user_prompt = init_prompt_formatted
    for action in gt_action_sequence:
        response_formatted = templates[prefix]['response'].format(action=action)
        instance = copy.deepcopy(instance_template)
        instance['prompt'] = user_prompt
        instance['response'] = response_formatted
        instances.append(instance)
        user_prompt += response_formatted

        obs, reward, done, _ = env.step(action)
        user_prompt += templates[prefix]['observation'].format(observation=obs)
        # images.append(env.render('rgb_array'))
    assert done and reward > 0, f"Environment did not terminate for seed {seed}"
    # ani = plot_animation(images)
    # ani.save(f'data/sft/animation_{seed}.gif')
    return instances


def create_chat_messages_for_env(args):
    """
    Chat is formated as list of messages like [{'role': 'system', 'content': 'xxx'}, {'role': 'user', 'content': 'xxx'}, {'role': 'assistant', 'content': 'xxx'}]
    """
    seed, prefix, data_source, MAX_DEPTH, SIZE, P, IS_SLIPPERY = args
    instance_template = {
        "data_source": data_source,
        "prompt": None,
        "response": None,
        "ability": "bfs",
        "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
        "extra_info": {"split": "train", "index": seed}
    }
    instances = []
    messages = [{'role': 'system', 'content': templates[prefix]['system']}]
   
    env = FrozenLakeEnv(size=SIZE, p=P, is_slippery=IS_SLIPPERY, seed=seed)
    obs = env.reset(seed=seed, reset_map=False, mode='tiny_rgb_array')
    instruction_message = templates[prefix]['instruction'].format(prompt= INSTRUCTION_TEMPLATE.format(observation=obs))
    messages.append({'role': 'user', 'content': instruction_message})

    # images = []

    # images.append(env.render('rgb_array'))
    gt_action_sequence = get_shortest_action_path(
        seed=seed,
        size=env.map_kwargs['size'],
        p=env.map_kwargs['p'],
        is_slippery=env.env_kwargs['is_slippery'],
        MAX_DEPTH=MAX_DEPTH
    )
    assert len(gt_action_sequence), f"No action sequence found for seed {seed}"
    for action in gt_action_sequence:
        action_str = env.ACTION_LOOKUP[action]
        response_message = templates[prefix]['response'].format(action=action_str)
        instance = copy.deepcopy(instance_template)
        instance['prompt'] = copy.deepcopy(messages)
        instance['response'] = response_message
        instances.append(instance)
        messages.append({'role': 'assistant', 'content': response_message})

        obs, reward, done, _ = env.step(action)
        observation_message = templates[prefix]['observation'].format(observation=obs)
        messages.append({'role': 'user', 'content': observation_message})
        # images.append(env.render('rgb_array'))
    assert done, f"Environment did not terminate for seed {seed}"
    # ani = plot_animation(images)
    # ani.save(f'data/sft/animation_{seed}.gif')
    return instances




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="frozenlake", help="Environment name (default: 'frozenlake').")
    parser.add_argument("--algo", type=str, default="bfs", choices=["bfs"], help="Algorithm to use (default: 'bfs').")
    parser.add_argument("--seed", type=int, default=100000, help="Seed for random number generation (default: 100000).")
    parser.add_argument("--output", type=str, default="data/sft/frozenlake", help="Output file to save the trajectories (default: 'data/sft/frozenlake').")
    parser.add_argument("--train_size", type=int, default=1000, help="Number of training trajectories to generate (default: 1000).")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test trajectories to generate (default: 100).")
    parser.add_argument("--bfs_max_depths", type=int, default=100, help="Maximum number of depths for BFS (default: 100).")
    parser.add_argument("--prefix", type=str, default='message', choices=['qwen-instruct', 'base', 'message'])
    # parser.add_argument("--without_thinking", action='store_true', help="Whether to exclude <think> </think> tags in the response.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes to use for parallel processing (default: 4).")
    args = parser.parse_args()

    assert args.env == "frozenlake", "Unsupported environment: {args.env}"
    assert args.algo == "bfs", "Unsupported algorithm: {args.algo}" # bfs to find shortest action path
    os.makedirs(args.output, exist_ok=True)
    data_source = args.env
    
    size, p = os.environ.get("SIZE"), os.environ.get("P")
    size, p = int(size), float(p)
    IS_SLIPPERY = os.environ.get("IS_SLIPPERY")
    IS_SLIPPERY = IS_SLIPPERY == "True"
    print(f"size: {size}, p: {p}, IS_SLIPPERY: {IS_SLIPPERY}")

    fn = create_chat_messages_for_env if args.prefix == 'message' else create_chat_str_for_env

    

    # Create process pool
    pbar = tqdm(total=args.train_size + args.test_size, desc="Generating trajectories", position=0, leave=True)
    with Pool(processes=args.num_processes) as pool:
        # Process training data
        train_args = [(args.seed + i, args.prefix, data_source, args.bfs_max_depths, size, p, IS_SLIPPERY) for i in range(args.train_size)]
        train_instances = []
        for instances in pool.imap(fn, train_args):
            train_instances.extend(instances)
            pbar.update(1)
            pbar.refresh()
        train_dataset = Dataset.from_list(train_instances)

        # Process test data 
        test_args = [(args.seed + i, args.prefix, data_source, args.bfs_max_depths, size, p, IS_SLIPPERY) for i in range(args.train_size, args.train_size + args.test_size)]
        test_instances = []
        for instances in pool.imap(fn, test_args):
            test_instances.extend(instances)
            pbar.update(1)
            pbar.refresh()
        test_dataset = Dataset.from_list(test_instances)

    pbar.close()


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




