"""
merge our image prompts:
(1) comparative prompts:
    a. ours-comp
    b. ours-comp-claude
    c. ours-comp-gemini
(2) path-based generic prompts:
    a. ours-path
    b. ours-path-claude
    c. ours-path-gemini
to gain:
    a. ours-full
    b. ours-full-claude
    c. ours-full-gemini
respectively
"""
import os
import json
import argparse
from utils.directory import load_config

def load_img_prompts(fpath):
    with open(fpath, 'r') as f:
        prompts = json.load(f)
    print(f'loading from {fpath}')
    return prompts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=[
        'food-101', 'ucf-101', 'cub-200', 'sun-324', 'imagenet'
    ])
    parser.add_argument('--prompt', type=str, default='ours-full', choices=[
        'ours-full', 'ours-full-claude', 'ours-full-gemini'
    ])

    opts = parser.parse_args()

    config = load_config('./data_paths.yml')

    if 'claude' in opts.prompt:
        ours_comp_path = config[f'{opts.dataset}-ours-comp-claude-prompts']
        ours_path_path = config[f'{opts.dataset}-ours-path-claude-prompts']
    elif 'gemini' in opts.prompt:
        ours_comp_path = config[f'{opts.dataset}-ours-comp-gemini-prompts']
        ours_path_path = config[f'{opts.dataset}-ours-path-gemini-prompts']
    else: # ours-full
        ours_comp_path = config[f'{opts.dataset}-ours-comp-prompts']
        ours_path_path = config[f'{opts.dataset}-ours-path-prompts']
    
    ours_full_dir = './image_prompts/Ours/chatgpt/ours_full'
    if not os.path.exists(ours_full_dir): os.makedirs(ours_full_dir)

    # loading comparative and path-based prompts
    ours_comp_prompts = load_img_prompts(ours_comp_path)
    ours_path_promtps = load_img_prompts(ours_path_path)

    # merge
    ours_full = {k:[] for k in ours_comp_prompts.keys()}
    for k, _ in ours_full.items():
        ours_full[k] = ours_comp_prompts[k] + ours_path_promtps[k]

    # save
    ours_full_path = config[f'{opts.dataset}-{opts.prompt}-prompts']
    with open(ours_full_path, 'w') as f:
        json.dump(ours_full, f, indent=4)
    print(f'saving at {ours_full_path}')
    
    

    






# EOF