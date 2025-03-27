"""
This code uses embeddings across layers of Llama2 to perform lexical semantics tasks.
The tasks include:
- WiC
- RAW-C
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils import extract_sentence, extract_sentence_raw, get_batch_data, get_layer_representations, evaluation, anisotropy_removal, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Probing for the WiC task.')

    # model and data
    parser.add_argument('--root_dir', type=str, default='/YOURDIR')
    parser.add_argument('--model_path', type=str, default='/YOURDIR/llama-2-7b-hf')
    parser.add_argument('--txt_path_test', type=str, default="data/test/test.data.txt")
    parser.add_argument('--gold_path_test', type=str, default="data/test/test.gold.txt")
    parser.add_argument('--acc_save_path', type=str, default='output/test_acc.csv')
    parser.add_argument('--threshold_path', default="data/thresholds.csv")
    parser.add_argument('--raw_path', default="data/RAW/raw-c_inx.csv")

    parser.add_argument('--mark', choices=['repeat', 'repeat_prev', 'prompt1', 'prompt2'], default='repeat')
    parser.add_argument('--NO_aniso', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['WiC', 'Raw'], default='WiC')

    parser.add_argument('--pos', choices=[None, 'N', 'V'], default=None)

    return parser.parse_args()

#%%
if __name__ == "__main__":
    args = parse_args()
    logger = get_logger()
    # models
    root = args.root_dir
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = '[PAD]'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM.from_pretrained(args.model_path).half().to(device)

    # data
    txt_path = root + '/' + args.txt_path_test
    gold_path = root + '/' + args.gold_path_test
    acc_save_path = root + '/' + args.acc_save_path 
    threshold_path = root+ '/' + args.threshold_path
    if os.path.exists(threshold_path):
        thresholds = pd.read_csv(threshold_path, delimiter=" ") # obtained by a validate set
    else:
        raise FileNotFoundError(
            f'{threshold_path} does not exist! '
            'Run get_thresholds.py first.'
        )

    # Get data
    logger.info(f'load {txt_path}')
    if args.dataset == "WiC":
        sents_list_1, sents_list_2, GT_list, inx_list = extract_sentence(txt_path, gold_path, args.mark, prev=int(args.mark == "repeat_prev"), pos=args.pos)
    else:
        sents_list_1, sents_list_2, GT_list, inx_list = extract_sentence_raw(args.raw_path, args.mark)
    logger.info(f'The first example:\n{sents_list_1[0], sents_list_2[0], GT_list[0], inx_list[0]}')
    logger.info(f'Data size: {len(sents_list_1)}')
    if args.pos:
        logger.warning(f'This is only a subset of ALL data with POS: {args.pos}')
    inx_list_1 = [int(i.split("-")[0]) for i in inx_list]
    inx_list_2 = [int(i.split("-")[1]) for i in inx_list]
    batch_list_1, tar_inx_list_1 = get_batch_data(tokenizer, sents_list_1, inx_list_1, batch_size=30)
    batch_list_2, tar_inx_list_2 = get_batch_data(tokenizer, sents_list_2, inx_list_2, batch_size=30)

    # Get represetation
    logger.info(f'Extracting representations...')
    data_pred_layers = {}
    data_sim_layers = {}
    for batch_1, batch_2, tar_inx_1, tar_inx_2 in tqdm(zip(batch_list_1, batch_list_2, tar_inx_list_1, tar_inx_list_2),total=len(batch_list_1)):
        batch_rep_1 = get_layer_representations(model, batch_1.to(device), tar_inx_1, device)
        batch_rep_2 = get_layer_representations(model, batch_2.to(device), tar_inx_2, device)
        for i in range(33):
            if args.NO_aniso:
                batch_sim = torch.nn.functional.cosine_similarity( batch_rep_1[:,i], batch_rep_2[:,i] ).cpu()
            else:
                batch_sim = torch.nn.functional.cosine_similarity( anisotropy_removal(batch_rep_1[:,i]), anisotropy_removal(batch_rep_2[:,i]) ).cpu()
            batch_pred = (batch_sim > thresholds.at[i, args.mark]).int().tolist()
            if i in data_pred_layers:
                data_pred_layers[i] += batch_pred
                data_sim_layers[i] += batch_sim.tolist()
            else:
                data_pred_layers[i] = batch_pred
                data_sim_layers[i] = batch_sim.tolist()

    # Save and Plot
    logger.info("Calculating accuracy...")
    if args.dataset == "WiC":
        acc_layers = [evaluation(pred, GT_list) for _, pred in data_pred_layers.items()]
        logger.info(f"ACC with layers: ")
        logger.info(acc_layers)
        logger.info("The best layer: %d with acc: %.3f" % (np.argmax(acc_layers), np.max(acc_layers)))
        plt.title(args.mark)
        plt.plot(acc_layers)
        if os.path.exists(acc_save_path):
            df_acc = pd.read_csv(acc_save_path)
        else:
            df_acc = pd.DataFrame()
            df_acc["Index"] = range(len(acc_layers))
        df_acc[args.mark] = acc_layers
        df_acc.to_csv(acc_save_path, index=False)
    else: # RAW
        corr_layers = [evaluation(sim, GT_list, type="corr") for _, sim in data_sim_layers.items()]
        print(np.argmax([round(i*100,1) for i in corr_layers]))
        print([round(i*100,1) for i in corr_layers])
        print(corr_layers)
        plt.plot(corr_layers)

# %%
