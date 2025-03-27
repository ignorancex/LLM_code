import json
import os, sys
import re
import string
import argparse
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats.stats import pearsonr
# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)



def main_bird(args):
    with open(args.model1_path, 'r') as result_file:
        model1_preds = json.load(result_file)
    with open(args.model2_path, 'r') as result_file:
        model2_preds = json.load(result_file)

    wrong_preds = []
    score2count = defaultdict(int)
    human_scores, scores = [], []
    pred1_scores, pred2_scores = [], []
    pred1_invalid, pred2_invalid = 0, 0

    for pred_idx, (model1_pred, model2_pred) in enumerate(zip(model1_preds, model2_preds)):
        model1_pred['id'] = pred_idx
        score2count[str(model1_pred['pred_score'])] += 1
        human_scores.append(model1_pred['human_score'] * 100.0)
        pred1, pred2 = -1, -1

        raw_model1_pred = model1_pred['raw_pred'][0].replace('out of 100', '')
        finds = re.findall(r"(\d+\.?\d*)", raw_model1_pred)
        if len(finds):
            pred1 = float(finds[-1])
        else:
            pred1_invalid += 1
            pass
        raw_model2_pred = model2_pred['raw_pred'][0].replace('out of 100', '')
        finds = re.findall(r"(\d+\.?\d*)", raw_model2_pred)
        if len(finds):
            pred2 = float(finds[-1])
        else:
            pred2_invalid += 1
            pass
        pred1 = -1 if pred1 > 100 else pred1
        pred2 = -1 if pred2 > 100 else pred2
        pred1_scores.append(pred1)
        pred2_scores.append(pred2)
        scores.append({
            'Human': model1_pred['human_score'] * 100.0,
            'GPT-3.5T': pred1,
            'GPT-4T': pred2,
        })
        print(pred_idx, model1_pred['human_score'] * 100.0, pred1, pred2)
    for score, count in sorted(score2count.items(), key=lambda k:k[1], reverse=True):
        print(score, count)
    cor, _ = pearsonr(pred1_scores, human_scores)
    print(args.model1_path)
    print(f"model1 cor={cor}, #invalid={pred1_invalid}")

    cor, _ = pearsonr(pred2_scores, human_scores)
    print(args.model2_path)
    print(f"model2 cor={cor}, #invalid={pred2_invalid}")

    # Plotting human_scores histogram
    # plt.hist(human_scores, bins=10, color='skyblue', edgecolor='black')
    # plt.xlabel('Score')
    # plt.ylabel('Frequency')
    # plt.title('Basic Histogram')
    # plt.show()

    # plotting three histograms on the same axis
    df = pd.json_normalize(scores)
    print(plt.style.available)
    plt.style.use('seaborn-v0_8-pastel')
    # Option 1: hard to see the normal distribution
    # plt.hist([df['Human'], df['GPT-3.5T'], df['GPT-4T']], bins=10, alpha=0.5,
    #          color=['red', 'blue', 'green'],
    #          label=['Human', 'GPT-3.5T', 'GPT-4T'], log=True)

    # Option 2: mixed together
    # plt.hist(df['Human'], bins=15, alpha=0.5, color='red')
    # plt.hist(df['GPT-3.5T'], bins=15, alpha=0.5, color='blue')
    # plt.hist(df['GPT-4T'], bins=15, alpha=0.8, color='yellow')

    # Option 3: plot separately, side by side
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(4, 2))
    df.hist('Human', bins=10, ax=axes[0])
    df.hist('GPT-3.5T', bins=10, ax=axes[1])
    df.hist('GPT-4T', bins=10, ax=axes[2])

    # fig.title("histogram with multiple variables (overlapping histogram)")
    # fig.legend(['Human', 'GPT-3.5T', 'GPT-4T'])
    fig.tight_layout()
    fig.show()
    fig.savefig('/Users/ruimeng/Desktop/score_dist.png', bbox_inches='tight')


    # header = ['id', 'query', 'candidates', 'target', 'pred', 'raw_pred', 'prompt']
    # preds = []
    # for pred in wrong_preds:
    #     pred = {k: pred[k] for k in header}
    #     pred['raw_pred'] = pred['raw_pred'][0]
    #     preds.append(pred)
    # df = pd.json_normalize(preds)
    # csv_df = df.to_csv('~/Desktop/bird-badcase.csv')
    # print(csv_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_path', type=str,  required=True, default='')
    parser.add_argument('--model2_path', type=str,  required=True, default='')
    args = parser.parse_args()
    main_bird(args)
