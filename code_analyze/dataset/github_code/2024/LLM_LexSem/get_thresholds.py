'''
To get thresholds for each layer using dev data
'''
import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils import extract_sentence, get_batch_data, get_layer_representations, evaluation, anisotropy_removal, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Probing for the WiC task.')

    # model and data
    parser.add_argument('--root_dir', type=str, default='/YOURDIR')
    parser.add_argument('--model_path', type=str, default='/YOURDIR/llama-2-7b-hf')
    parser.add_argument('--txt_path_dev', type=str, default="data/dev/dev.data.txt")
    parser.add_argument('--gold_path_dev', type=str, default="data/dev/dev.gold.txt")
    parser.add_argument('--acc_save_path', type=str, default='output/test_acc.csv')
    parser.add_argument('--threshold_path', default="data/thresholds.csv")

    parser.add_argument('--mark', choices=['repeat', 'repeat_prev', 'prompt1', 'prompt2'], default='repeat')
    parser.add_argument('--NO_aniso', action='store_true')

    parser.add_argument('--pos', choices=[None, 'N', 'V'], default=None)

    return parser.parse_args()

#%%
if __name__ == "__main__":
    args = parse_args()
    logger = get_logger()
    # models
    logger.info(f'load {args.model_path}')
    root = args.root_dir
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = '[PAD]'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM.from_pretrained(args.model_path).half().to(device)

    # data

    txt_path = root + '/' + args.txt_path_dev
    gold_path = root + '/' + args.gold_path_dev
    threshold_path = root+ '/' + args.threshold_path
    if os.path.exists(threshold_path):
        thresholds = pd.read_csv(threshold_path, delimiter=" ") # obtained by a validate set
    else:
        thresholds = pd.DataFrame()

    # Get data
    logger.info(f'load {txt_path}')
    sents_list_1, sents_list_2, GT_list, inx_list = extract_sentence(txt_path, gold_path, args.mark, prev=int(args.mark == "repeat_prev"), pos=args.pos)
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
    for threshold in tqdm(range(0,100,5)):
        for batch_1, batch_2, tar_inx_1, tar_inx_2 in tqdm(zip(batch_list_1, batch_list_2, tar_inx_list_1, tar_inx_list_2),total=len(batch_list_1)):
            batch_rep_1 = get_layer_representations(model, batch_1.to(device), tar_inx_1, device)
            batch_rep_2 = get_layer_representations(model, batch_2.to(device), tar_inx_2, device)

            for i in range(33):
                if args.NO_aniso:
                    batch_sim = torch.nn.functional.cosine_similarity( batch_rep_1[:,i], batch_rep_2[:,i] ).cpu()
                else:
                    batch_sim = torch.nn.functional.cosine_similarity( anisotropy_removal(batch_rep_1[:,i]), anisotropy_removal(batch_rep_2[:,i]) ).cpu()

                batch_pred = (batch_sim > threshold/100).int().tolist()
                if threshold not in data_pred_layers:
                    data_pred_layers[threshold] = {}
                if i not in data_pred_layers[threshold]:
                    data_pred_layers[threshold][i] = []
                data_pred_layers[threshold][i] += batch_pred

    # Save
    logger.info(f'Calculate best thresholds...')
    acc_layers = [[evaluation(pred, GT_list) for _, pred in preds.items()] for _, preds in data_pred_layers.items()]
    acc_layers = np.array(acc_layers)
    best_th_layers = [list(range(0,100,5))[i]/100 for i in np.argmax(acc_layers, axis=0)]
    best_acc_layers = np.max(acc_layers, axis=0)
    for ix,(th,acc) in enumerate(zip(best_th_layers, best_acc_layers)):
        print("The best accuracy: %.2f\tthreshold: %.2f\tlayer: %d" % (acc,th,ix))
    thresholds['Index'] = range(len(best_th_layers))
    thresholds[args.mark] = best_th_layers
    thresholds.to_csv(threshold_path, sep=" ", index=False)
    logger.info(f'Thresholds file saved.')
    logger.info(f'Saved path: {threshold_path}')
