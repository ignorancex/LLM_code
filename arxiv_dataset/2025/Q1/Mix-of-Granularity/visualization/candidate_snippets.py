'''
Scripts to plot for the experiment of number of candidate snippets

16 May Zijie
'''

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_cand_snip(res_list):
    print("Visualization starts.")
    
    # Create a figure with 5 subplots arranged in one row
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    for i in tqdm(range(len(res_list))):
        dataset = res_list[i]['dataset']
        mog = res_list[i]['MoG']
        mogg = res_list[i]['MoGG']
        cot = [res_list[i]['cot']]*len(mog)
        
        # Plot each line on a separate subplot
        x_list = [3, 6, 12, 24, 48]
        axs[i].plot(x_list, mog, label='MoG', color='#8FABDB')
        axs[i].plot(x_list, mogg, label='MoGG', color='#F4B184')
        axs[i].plot(x_list, cot, label='CoT', linestyle='--', color='#A8D08F')
        
        # Set subplot title and labels
        axs[i].set_title(dataset)
        axs[i].set_xlabel('Number of candidate snippets')
        axs[i].set_ylabel('Acc.')
        axs[i].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    save_path = 'candidate_snippets_plot.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved in {save_path}")
        
    return 0

# input
mmlu_res = {
    'dataset': 'mmlu',
    'cot': 0.3, 
    'MoG': [0.44, 0.43, 0.28, 0.11, 0.09],
    'MoGG': [0.42, 0.12, 0.2, 0.5, 0.6],
}
medqa_res = {
    'dataset': 'medqa',
    'cot': 0.4, 
    'MoG': [0.12, 0.13, 0.25, 0.35, 0.52],
    'MoGG': [0.09, 0.1, 0.28, 0.47, 0.62],
}
pubmedqa_res = {
    'dataset': 'pubmedqa',
    'cot': 0.5, 
    'MoG': [0.53, 0.42, 0.11, 0.87, 0.9],
    'MoGG': [0.49, 0.47, 0.16, 0.77, 0.79],
}
medmcqa_res = {
    'dataset': 'medmcqa',
    'cot': 0.3, 
    'MoG': [0.16, 0.26, 0.29, 0.36, 0.37],
    'MoGG': [0.14, 0.22, 0.24, 0.34, 0.35],
}
bioasq_res = {
    'dataset': 'bioasq',
    'cot': 0.4, 
    'MoG': [0.2, 0.12, 0.45, 0.36, 0.33],
    'MoGG': [0.33, 0.43, 0.47, 0.49, 0.23],
}

res_list = [mmlu_res, medqa_res, pubmedqa_res, medmcqa_res, bioasq_res]

plot_cand_snip(res_list)

