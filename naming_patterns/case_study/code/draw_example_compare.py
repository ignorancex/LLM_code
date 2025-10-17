from typing import Dict, List, Tuple
import os
import re
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 


INTERSECTION_VARS_CSV_PATH = "LLM_code/codeforces/simulation/result/case/unique/intersection_vars.csv"
OUT_DIR = "plots" 
os.makedirs(OUT_DIR, exist_ok=True)


TOTAL_PROBLEMS = 1215 


intersection_df: pd.DataFrame = pd.DataFrame()
MODEL_NAMES: List[str] = [] 

if not os.path.exists(INTERSECTION_VARS_CSV_PATH):
    exit()
else:
    try:
        intersection_df = pd.read_csv(INTERSECTION_VARS_CSV_PATH)

        if 'variable' in intersection_df.columns:

            intersection_df = intersection_df.set_index('variable')

            
            potential_model_cols = [col for col in intersection_df.columns if col.endswith('_ac_frequency')]
            
            fixed_model_order = ["GPT", "Gemini", "DeepSeek", "Llama", "Qwen", "Gemma"]
            MODEL_NAMES = [m for m in fixed_model_order if f"{m}_ac_frequency" in intersection_df.columns]
            
            if not MODEL_NAMES:
                exit()
            
            print(f"Model: {MODEL_NAMES}")

        else:
            exit() 

    except Exception as e:
        exit() 

TARGET_VARS: List[str] = [
    "max_length"
]


TARGET_VARS = [var for var in TARGET_VARS if var in intersection_df.index]
if not TARGET_VARS:
    print("Error")

colors = {                                   
    "ac":  "#c8c8c8",   # Human
    "ref": "#ffde7b",   # LLM-Revised
    "ans": "#6ad1a3",   # LLM-Generated
}
legend_labels = {"ref": "LLM-Revised", "ans": "LLM-Generated"}

def plot_single_variable_comparison(target_var: str):

    if target_var not in intersection_df.index:
        return

    row_data = intersection_df.loc[target_var]


    human_freq = 0.0
    if MODEL_NAMES:
        first_model_ac_col = f"{MODEL_NAMES[0]}_ac_frequency"
        if first_model_ac_col in row_data:
            human_freq = row_data[first_model_ac_col] / TOTAL_PROBLEMS
        else:
            print(f"set to 0.")
    
    ref_vals, ans_vals = [], []
    for model in MODEL_NAMES:
        ref_col = f"{model}_ref_frequency"
        ans_col = f"{model}_ans_frequency"
        
        current_ref_freq = row_data.get(ref_col, 0.0) / TOTAL_PROBLEMS
        current_ans_freq = row_data.get(ans_col, 0.0) / TOTAL_PROBLEMS
        
        ref_vals.append(current_ref_freq)
        ans_vals.append(current_ans_freq)

    bar_width   = 0.5     
    sub_width   = 0.3     
    x           = np.arange(len(MODEL_NAMES) + 1) 
    human_pos   = x[0]
    model_pos   = x[1:]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.bar(human_pos, human_freq,
           width=bar_width,
           color=colors["ac"],)

    for i, model_name in enumerate(MODEL_NAMES): 
        pos = model_pos[i]
        
        ax.bar(pos - sub_width/2, ref_vals[i],
               width=sub_width,
               color=colors["ref"],
               label=legend_labels["ref"] if i == 0 else "")
        ax.bar(pos + sub_width/2, ans_vals[i],
               width=sub_width,
               color=colors["ans"],
               label=legend_labels["ans"] if i == 0 else "")

    xticks       = x
    xtick_labels = ["Human", "GPT", "Gemini", "DS", "Llama", "Qw", "Gemma"] 
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=8)

    max_val = max(human_freq, *ref_vals, *ans_vals)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 0.1) 
    plt.yticks(fontsize=8)

    ax.legend(fontsize=8, loc="upper right", frameon=True, labelspacing=0.2)
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"{target_var}_compare.pdf") 
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if not TARGET_VARS:
        print("Error")
    else:
        for var_to_plot in tqdm(TARGET_VARS, desc="Generating plots for specified variables"):
            plot_single_variable_comparison(var_to_plot)