"""
generate HieComp prompts based 
(1) k-means clustering on the fly
(2) the given label hierarchy (our enhanced variant of HIE)
"""

import os
import time
import json
import clip
import math
import torch
import argparse
import numpy as np

from tqdm import tqdm
from openai import OpenAI
from collections import OrderedDict


from trees.tree_utils import load_tree
from utils.directory import load_config, load_api_key
from lang_prompts.hie_prompts import generate_prompt_summary
from lang_prompts.hie_prompts import generate_prompt_compare
from lang_prompts.hie_prompts import load_initial_vcd_img_prompts
from lang_prompts.hie_prompts import generate_prompt_given_overall_feature
from lang_prompts.hie_prompts import stringtolist, compose_hie_image_prompts




def generate_description_direct_compare(categories_group, descriptors, api_key, temp=1.0):
    """
        1-vs-rest direct comparison
    """

    client = OpenAI(api_key=api_key)

    for x in tqdm(categories_group, total=len(categories_group)):

        subtracted_list = [y for y in categories_group if y != x]
        string = ', '.join(subtracted_list)
        prompt = generate_prompt_compare(x, string)

        while True:
            try:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                )
                break
            except:
                time.sleep(3)
        
        text = completion.choices[0].message.content
        if text[0] != '-': text = '- ' + text
        res_descriptors = stringtolist(text)
        
        if len(res_descriptors) != 0:
            descriptors[x].append(compose_hie_image_prompts(res_descriptors, x))
        else:
            print(f'\t\t\tempty level for {x}!!!\t\t\t')
    
    return


def generate_description_summary_compare(categories_group, descriptors, api_key, temp=1.0):
    """
        1-vs-summary comparison
    """

    client = OpenAI(api_key=api_key)
    string = ', '.join(categories_group)
    prompt = generate_prompt_summary(string)

    while True:
        try:
            completion = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                        {"role": "user", "content": prompt}
                ],
                temperature=temp,
            )
            break
        except:
            time.sleep(3)
    
    summary = completion.choices[0].message.content
    print(f'summary response:\n{prompt}"{summary}"')

    prompt_list = [generate_prompt_given_overall_feature(category.replace('_', ' '), summary) for category in categories_group]
    for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
        while True:
            try:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                    {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                )
                break
            except:
                time.sleep(3)

        text = completion.choices[0].message.content
        
        # parse chatgpt response
        if text[0] != '-': text = '- ' + text
        res_descriptors = stringtolist(text)

        name = categories_group[idx]
        # update descriptors
        if len(res_descriptors) != 0:
            descriptors[name].append(compose_hie_image_prompts(res_descriptors, name))
        else:
            print(f'\t\t\tempty level for {name}!!!\t\t\t')
    
    return



def build_tree_description(descriptors, tree, api_key, threshold=3):
    """
        descriptors: preload VCD_LLM descriptors, it will be enriched by HieComp descriptors
        the label hierarchy is already known, no need for cluster
    """


    for node in tree:

        if isinstance(node, str): # a leaf node
            categories_group = [node]
        else:
            categories_group = node.leaves()

        if len(categories_group)<=1: 
            print('lonely!')
            print(categories_group)
        elif len(categories_group)<=threshold and len(categories_group)>=2: 
            print(f'direct comparison {len(categories_group)} classes')
            print(categories_group)
            generate_description_direct_compare(categories_group, descriptors, api_key)
        else: 
            print(f'summary {len(categories_group)} classes')
            generate_description_summary_compare(categories_group, descriptors, api_key)
            build_tree_description(descriptors, node, api_key, threshold)

    return


def k_means(data, k, max_iters=300):
    """
    Runs the k-means algorithm on the given data.

    Args:
        data (torch.Tensor): The data to cluster, of shape (N, D).
        k (int): The number of clusters to form.
        max_iters (int): The maximum number of iterations to run the algorithm for.

    Returns:
        A tuple containing:
        - cluster_centers (torch.Tensor): The centers of the clusters, of shape (k, D).
        - cluster_assignments (torch.Tensor): The cluster assignments for each data point, of shape (N,).
    """
    # Initialize cluster centers randomly
    np.random.seed(42)
    cluster_centers = data[np.random.choice(data.shape[0], k, replace=False)]
    cluster_assignments = None

    # Run the algorithm for a fixed number of iterations
    for _ in range(max_iters):
        # Compute distances between data and cluster centers using broadcasting
        distances = torch.norm(data[:, None, :] - cluster_centers[None, :, :], dim=-1)
        # Assign each data point to the nearest cluster center
        cluster_assignments = torch.argmin(distances, dim=1)

        # Update the cluster centers based on the mean of the assigned points
        for j in range(k):
            mask = cluster_assignments == j
            if mask.any():
                cluster_centers[j] = data[mask].mean(dim=0)

    return cluster_centers, cluster_assignments



def build_hierarchical_cluster_description(class_names, descriptors, clip_model, api_key, num_group_div=6, threshold=3):
    """
        descriptors: preload VCD_LLM descriptors, it will be enriched by HieComp descriptors
        the label tree is derived via clustering of class text encodings
    """

    description_encodings = OrderedDict()
    for k, v in descriptors.items():
        if k in class_names:
            tokens = clip.tokenize(texts=v[-1], context_length=77, truncate=True).cuda()
            description_encodings[k] = \
                torch.nn.functional.normalize(clip_model.encode_text(tokens))
    
    text_avg_emb = [None] * len(description_encodings)
    for i, (k,v) in enumerate(description_encodings.items()):
        text_avg_emb[i] = v.mean(dim=0)
    
    try:
        text_avg_emb = torch.stack(text_avg_emb, dim=0)
    except:
        import pdb
        pdb.set_trace()
    
    num_group = int(math.ceil(len(class_names) / num_group_div))

    if num_group <= 1:
        num_group = 2
    
    print('clustering')
    _, cluster_assignments = k_means(text_avg_emb, num_group)
    

    label_to_classname_np = np.array(class_names)

    for group_idx in range(num_group):
        tmp_index = torch.where(cluster_assignments == group_idx)[0]
        categories_group = label_to_classname_np[tmp_index.cpu()]
        
        if isinstance(categories_group, np.ndarray):
            categories_group = categories_group.tolist()
        if not isinstance(categories_group, list):
            categories_group = [categories_group]
        
        if len(categories_group)<=threshold and len(categories_group)>=2:
            print(f'direct comparison: {categories_group}')
            generate_description_direct_compare(categories_group, descriptors, api_key)
        elif len(categories_group) <=1:
            print(f"lonely: {categories_group}")
        else:
            print("summary")
            generate_description_summary_compare(categories_group, descriptors, api_key)
            build_hierarchical_cluster_description(
                categories_group, descriptors, clip_model, 
                api_key, num_group_div, threshold
            )
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=[
        'food-101', 'ucf-101', 'cub-200', 'sun-324', 'imagenet'
    ])
    parser.add_argument('--method', type=str, choices=['hiet', 'hiec'])
    opts = parser.parse_args()

    # ---------------------------- #
    api_key = load_api_key('./data_paths.yml', 'openai')
    config = load_config('./data_paths.yml')
    tree = load_tree(opts.dataset, './trees')
    classnames = tree.leaves()
    init_vcd_prompt_path = config[f'{opts.dataset}-vcd-prompts']
    descriptors = load_initial_vcd_img_prompts(init_vcd_prompt_path)
    # ---------------------------- #


    if opts.method == 'hiec':
        save_dir = os.path.join('./image_prompts/HIE_Cluster/')
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        descriptors_path = os.path.join(save_dir, f'{opts.dataset}-{opts.method}-1.00-temp.json')
        clip_model, _ = clip.load("ViT-L/14@336px")
        clip_model.eval()
        clip_model.requires_grad_(False)
        build_hierarchical_cluster_description(classnames, descriptors, clip_model, api_key)
    else: # opts.method = 'hiet' 
        save_dir = os.path.join('./image_prompts/HIE_Tree/')
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        descriptors_path = os.path.join(save_dir, f'{opts.dataset}-{opts.method}-1.00-temp.json')
        build_tree_description(descriptors, tree, api_key)


    with open(descriptors_path, 'w') as f:
        json.dump(descriptors, f, indent=4)
    print(f'saving at {descriptors_path}')
    



