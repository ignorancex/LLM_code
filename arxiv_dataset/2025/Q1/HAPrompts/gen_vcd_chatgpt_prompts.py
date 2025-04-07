"""
generate VCD style descriptors-based image prompts with ChatGPT
this is using openai sync api -> vcd has relatively fewer queries
"""

import os
import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI


from lang_prompts.vcd_prompts import vcd_stringtolist
from lang_prompts.vcd_prompts import generate_vcd_llm_prompt
from lang_prompts.vcd_prompts import compose_vcd_image_prompts

from trees.tree_utils import load_tree
from utils.directory import load_api_key



def generate_chatgpt_vcd_descriptors(classnames, max_tokens, temperature):

    api_key = load_api_key('./data_paths.yml', 'openai')
    client = OpenAI(api_key=api_key)

    image_prompts = dict()
    for _, name in tqdm(enumerate(classnames), total=len(classnames)):
        
        prompt = generate_vcd_llm_prompt(name)
        while (True):
            try:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens, 
                    temperature=temperature 
                )
                break
            except:
                time.sleep(3)

        text = completion.choices[0].message.content
        if text[0] != '-': text = '- ' + text
    
        # vcd post-process: descriptors -> image prompts
        descriptors = vcd_stringtolist(text)
        image_prompts[name] = compose_vcd_image_prompts(descriptors, name)
    
    return image_prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=[
        'food-101', 'ucf-101', 'cub-200', 'sun-324', 'imagenet'
    ])

    opts = parser.parse_args()

    # ---------------- #
    max_tokens = 100
    temp = 0.0
    tree = load_tree(opts.dataset, './trees')
    classnames = tree.leaves()
    # ---------------- #

    # call sync api
    image_prompts = generate_chatgpt_vcd_descriptors(classnames, max_tokens, temp)

    vcd_img_prompts_dir = './image_prompts/VCD'
    if not os.path.exists(vcd_img_prompts_dir): os.makedirs(vcd_img_prompts_dir)
    vcd_img_prompts_path = os.path.join(
        vcd_img_prompts_dir, 
        f'{opts.dataset}-vcd-{temp:.2f}-temp-{max_tokens}-mtokens.json'
    )

    with open(vcd_img_prompts_path, 'w') as f:
        json.dump(image_prompts, f, indent=4)
    print(f'saving at {vcd_img_prompts_path}')


