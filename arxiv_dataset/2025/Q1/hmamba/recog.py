# -*- coding: utf-8 -*-
# @Time    : 04/15/24
# @Author  : Fu-An Chao
# @Affiliation  : National Taiwan Normal University
# @Email   : fuann@ntnu.edu.tw
# @File    : recog.py

import sys
import re
import os
import time
import json
from tqdm import tqdm
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from dataset import GoPDataset

from models import *
import argparse

def load_conf(config):
    with open(config, "r", encoding="utf-8") as f:
        try:
            args = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    return args

def load_phn_dict(path):
    phn_dict = {}
    with open(path, "r") as rf:
        try:
            phn_dict = json.load(rf)
            phn_dict = {int(id):phn for phn,id in phn_dict.items()}
        except:
            for line in rf.readlines():
                phn, phn_id = line.split()[0], line.split()[1]
                phn_dict[int(phn_id)] = phn
    return phn_dict

def recog(audio_model, val_loader, args):

    # output path
    hyp_f = open(os.path.join(args.exp_dir, "hyp"), "w")
    rel_f = open(os.path.join(args.exp_dir, "rel"), "w")
    can_f = open(os.path.join(args.exp_dir, "can"), "w")
    if args.remove_sil:
        hyp_nosil_f = open(os.path.join(args.exp_dir, "hyp_nosil"), "w")
        rel_nosil_f = open(os.path.join(args.exp_dir, "rel_nosil"), "w")
        can_nosil_f = open(os.path.join(args.exp_dir, "can_nosil"), "w")

    # id2phn dict
    id2phn = load_phn_dict(args.phn_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    # load state dict
    model_state_dict = torch.load(args.exp_dir+"/models/best_audio_model.pth", map_location=torch.device('cpu'))
    audio_model.load_state_dict(model_state_dict, strict=False)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []

    with torch.no_grad():
        for i, (audio_input, audio_input2, audio_input3, canophns, realphns, bies, utt_id) in enumerate(tqdm(val_loader)):
            audio_input = audio_input.to(device)
            audio_input2 = [ input2.to(device) for input2 in audio_input2 ]
            if isinstance(audio_input3, torch.Tensor):
                audio_input3 = audio_input3.to(device)

            # uttid is tuple
            utt_id, = utt_id

            # compute output
            mask = (realphns>=0)    #     pad mask
            #mask = (phn_label>=0)  # pad+sil mask

            #u1, u2, u3, u4, u5, p, w1, w2, w3, logits
            logits = audio_model(
                audio_input, audio_input2, audio_input3, canophns, bies, mask=mask
            )[-1]

            # token ids
            hyp = torch.argmax(logits, dim=-1).to('cpu').detach()
            rel = realphns.to('cpu').detach()
            can = canophns.to('cpu').detach()

            # phones
            hyp = " ".join([ id2phn[int(id)] for id in hyp.masked_select(mask).tolist() ]).lower()
            rel = " ".join([ id2phn[int(id)] for id in rel.masked_select(mask).int().tolist() ]).lower()
            can = " ".join([ id2phn[int(id)] for id in can.masked_select(mask).int().tolist() ]).lower()

			# remove special token
            if args.remove_special_token:
                hyp = " ".join(re.sub(args.special_token, "", hyp).split())
                rel = " ".join(re.sub(args.special_token, "", rel).split())
                can = " ".join(re.sub(args.special_token, "", can).split())

            hyp_f.write(f"{utt_id} {hyp}\n")
            rel_f.write(f"{utt_id} {rel}\n")
            can_f.write(f"{utt_id} {can}\n")

            if args.remove_sil:
                # remove sil
                hyp_nosil = " ".join(re.sub(r'sil', "", hyp).split())
                rel_nosil = " ".join(re.sub(r'sil', "", rel).split())
                can_nosil = " ".join(re.sub(r'sil', "", can).split())

                hyp_nosil_f.write(f"{utt_id} {hyp_nosil}\n")
                rel_nosil_f.write(f"{utt_id} {rel_nosil}\n")
                can_nosil_f.write(f"{utt_id} {can_nosil}\n")


    hyp_f.close()
    rel_f.close()
    can_f.close()
    hyp_nosil_f.close()
    rel_nosil_f.close()
    can_nosil_f.close()

if __name__ == '__main__':
    print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--remove-special-token", action="store_true")
    parser.add_argument("--remove-sil", action="store_true")
    parser.add_argument("--special-token", type=str, default="<del>")
    parser.add_argument("--set", type=str, default="test", help="testing directory")
    parser.add_argument("--am", type=str, default='librispeech', help="name of the acoustic models")
    parser.add_argument("--phn-dict", type=str, default='prep_data/phn_dict_all.txt')
    parser.add_argument("--model", type=str, default='gopt', help="name of the model")
    parser.add_argument("--model-conf", type=str, help="model config")
    parser.add_argument("--gop-dir", type=str, default="./data/seq_data_librispeech", help="train & test data directory for experiments")
    parser.add_argument("--ssl-dir", type=str, default=None, help="extra ssl data")
    parser.add_argument("--raw-dir", type=str, default=None, help="extra raw audio data")
    parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")

    args = parser.parse_args()
    conf = load_conf(args.model_conf)

    am = args.am
    print('now using {:s} acoustic model'.format(am))

    if args.model == 'HMamba':
        print('now use a HMamba model')
        audio_mdl = HMamba(**conf)
    else:
        raise ValueError(f"Invalid model {args.model}")

    # parse test dataset
    te_dataset = GoPDataset(args.set, data_dir=args.gop_dir, data_dir2=args.ssl_dir, data_dir3=args.raw_dir, am=am, mode="mdd")
    te_dataloader = DataLoader(te_dataset, batch_size=1, shuffle=False)

    recog(audio_mdl, te_dataloader, args)
