# -*- coding: utf-8 -*-
# @Time    : 05/15/24
# @Author  : Fu-An Chao
# @Affiliation  : National Taiwan Normal University
# @Email   : fuann@ntnu.edu.tw
# @File    : train_full.py

# modifed from the https://github.com/YuanGongND/gopt

import sys
import os
import time
import yaml
import torch
import random
import json
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import GoPDataset
from sklearn.metrics import f1_score
from jiwer import wer
from thop import profile, clever_format
from scheduler import TriStageLRScheduler

from models import *
from loss import cross_entropy_lsm, decoupled_cross_entropy_lsm
import argparse
import wandb

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

# just to generate the header for the result.csv
def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    # NOTE: stress header
    stress_header_set = ['stress_train_f1', 'stress_test_f1']
    stress_header_score = ['macro', 'micro']
    stress_header = []
    for dset in stress_header_set:
        stress_header = stress_header + [dset+'_'+x for x in stress_header_score]
    
    # NOTE: per header
    per_header = ['per_train', 'per_test']
    
    header = phn_header + utt_header + word_header + stress_header + per_header
    return header

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 1
    # NOTE: modify
    total_step = len(train_loader) * args.n_epochs
    exp_dir = args.exp_dir

    # loss related
    loss_fn = nn.MSELoss()

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)

    if args.pretrain:
        # load state dict
        print('Loading pretrained model from {} ...'.format(args.pretrain))
        model_state_dict = torch.load(args.pretrain+"/models/best_audio_model.pth", map_location=torch.device(device))
        audio_model.load_state_dict(model_state_dict, strict=False)

    trainables = []
    for name, p in audio_model.named_parameters():
        if p.requires_grad:
            if "utt_mlp" in name:
                trainables += [{"params": [p], 'lr': args.lr*0.045}]
            elif "wrd_mlp" in name:
                trainables += [{"params": [p], 'lr': args.lr}]
            else:
                trainables += [{"params": [p], 'lr': args.lr}]

    optimizer = torch.optim.Adam(trainables, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = TriStageLRScheduler(optimizer, 
        init_lr_scale=args.init_lr_scale,
        peak_lr=args.lr,
        final_lr=args.final_lr,
        phase_ratio=args.phase_ratio,
        total_steps=total_step
    )

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 38])

    while epoch <= args.n_epochs:

        audio_model.train()

        for i, (audio_input, audio_input2, audio_input3, canophns, realphns, bies, phn_label, word_label, utt_label, utt_id) in enumerate(train_loader):

            audio_input = audio_input.to(device, non_blocking=True)
            audio_input2 = [ input2.to(device, non_blocking=True) for input2 in audio_input2 ]
            if isinstance(audio_input3, torch.Tensor):
                audio_input3 = audio_input3.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)
            canophns = canophns.to(device, non_blocking=True)
            realphns = realphns.to(device, non_blocking=True)
            bies = bies.to(device, non_blocking=True)

            # filter out the padded tokens, only calculate the loss based on the valid tokens
            # < 0 is a flag of padded tokens
            mask = (phn_label>=0).to(device, non_blocking=True)
            
            if global_step==0:
                macs, paras = profile(audio_model.module, inputs=(
                    audio_input, audio_input2, audio_input3, canophns, bies, mask
                ))
                macs, paras = clever_format([macs, paras], "%.3f")
                print(f'[INFO] Params(M): {paras}, MACs(G): {macs}')
                audio_model.module.to(device)

            u1, u2, u3, u4, u5, p, w1, w2, w3, logits = audio_model(
                audio_input, audio_input2, audio_input3, canophns, bies, mask=mask
            )

            # filter out the padded tokens, only calculate the loss based on the valid tokens
            # < 0 is a flag of padded tokens
            p = p.squeeze(2)
            p = p * mask
            loss_phn = loss_fn(p, phn_label * mask)

            # avoid the 0 losses of the padded tokens impacting the performance
            loss_phn = loss_phn * (mask.shape[0] * mask.shape[1]) / torch.sum(mask)

            # utterance level loss, also mse
            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            loss_utt = loss_fn(utt_preds ,utt_label)

            # word level loss
            word_label = word_label[:, :, 0:3]

            mask = (word_label>=0)
            word_pred = torch.cat((w1,w2,w3), dim=2)
            word_pred = word_pred * mask
            word_label = word_label * mask
            loss_word = loss_fn(word_pred, word_label)
            loss_word = loss_word * (mask.shape[0] * mask.shape[1] * mask.shape[2]) / torch.sum(mask)

            # NOTE: loss for regression
            loss = args.loss_w_phn * loss_phn + args.loss_w_utt * loss_utt + args.loss_w_word * loss_word

            # NOTE: loss for classification
            if args.loss_type == "xent":
                loss_xent = cross_entropy_lsm(logits, realphns.long(), lsm_prob=0.0, ignore_index=-1, training=True)
                loss = loss + args.loss_w_xent * loss_xent
            elif args.loss_type == "dexent":
                loss_xent = decoupled_cross_entropy_lsm(logits, realphns, canophns, a=args.loss_w_a, ignore_index=-1, training=True)
                loss = loss + args.loss_w_xent * loss_xent
            else:
                raise ValueError(f"only xent and dexent is available.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            # write to wandb
            wandb.log({'train/loss': loss})

        print(f"Epoch-{epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}")

        print('start validation')
        # ensemble results
        # don't save prediction for the training set
        tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr, tr_word_stress, tr_per \
            = validate(audio_model, train_loader, args)
        te_mse, te_corr, te_utt_mse, te_utt_corr, te_word_mse, te_word_corr, te_word_stress, te_per, \
            A_phn_target, valid_word_target, A_utt_target, A_recog_target, A_phn, valid_word_pred, A_utt, A_recog \
            = validate(audio_model, test_loader, args, valid=True)

        if te_mse < best_mse:

            # create the directory
            if os.path.exists(args.exp_dir + '/preds') == False:
                os.mkdir(args.exp_dir + '/preds')

            # saving the phn target, only do once
            if os.path.exists(args.exp_dir + '/preds/phn_target.npy') == False:
                np.save(args.exp_dir + '/preds/phn_target.npy', A_phn_target)
                np.save(args.exp_dir + '/preds/word_target.npy', valid_word_target)
                np.save(args.exp_dir + '/preds/utt_target.npy', A_utt_target)

            np.save(args.exp_dir + '/preds/phn_pred.npy', A_phn)
            np.save(args.exp_dir + '/preds/word_pred.npy', valid_word_pred)
            np.save(args.exp_dir + '/preds/utt_pred.npy', A_utt)

            result[epoch-1, :6] = [epoch, tr_mse, tr_corr, te_mse, te_corr, optimizer.param_groups[0]['lr']]

            result[epoch-1, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, te_utt_mse, te_utt_corr])

            result[epoch-1, 26:32] = np.concatenate([tr_word_corr, te_word_corr])

            # stress report
            result[epoch-1, 32:36] = np.concatenate([tr_word_stress, te_word_stress])

            # per report
            result[epoch-1, 36:38] = [tr_per, te_per]

            header = ','.join(gen_result_header())
            if not args.save_last_epoch:
                np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')

        if te_mse < best_mse:
            best_mse = te_mse
            best_epoch = epoch
            print(f"New Best Phone MSE(test): {best_mse:.3f}")

        if best_epoch == epoch:
            if os.path.exists("%s/models/" % (exp_dir)) == False:
                os.mkdir("%s/models" % (exp_dir))
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

        print('Phone: Test MSE: {:.3f}, CORR: {:.3f}'.format(te_mse.item(), te_corr))
        print('Utterance:, ACC: {:.3f}, COM: {:.3f}, FLU: {:.3f}, PROC: {:.3f}, Total: {:.3f}'.format(te_utt_corr[0], te_utt_corr[1], te_utt_corr[2], te_utt_corr[3], te_utt_corr[4]))
        print('Word:, ACC: {:.3f}, Stress: {:.3f}, Total: {:.3f}'.format(te_word_corr[0], te_word_corr[1], te_word_corr[2]))
        print('Phone error rate: {:.3f}'.format(te_per))
        print('-------------------validation finished-------------------')

        # write to wandb
        wandb.log({
            # train
            "train/phone (PCC)": tr_corr,
            "train/word-acc (PCC)": tr_word_corr[0],
            "train/word-stress (PCC)": tr_word_corr[1],
            "train/word-total (PCC)": tr_word_corr[2],
            "train/utt-acc (PCC)": tr_utt_corr[0],
            "train/utt-comp (PCC)": tr_utt_corr[1],
            "train/utt-flu (PCC)": tr_utt_corr[2],
            "train/utt-pro (PCC)": tr_utt_corr[3],
            "train/utt-total (PCC)": tr_utt_corr[4],
            "train/per": tr_per,
            # test
            "test/phone (PCC)": te_corr,
            "test/word-acc (PCC)": te_word_corr[0],
            "test/word-stress (PCC)": te_word_corr[1],
            "test/word-total (PCC)": te_word_corr[2],
            "test/utt-acc (PCC)": te_utt_corr[0],
            "test/utt-comp (PCC)": te_utt_corr[1],
            "test/utt-flu (PCC)": te_utt_corr[2],
            "test/utt-pro (PCC)": te_utt_corr[3],
            "test/utt-total (PCC)": te_utt_corr[4],
            "test/per": te_per,
        })

        if args.save_last_epoch:
            np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')
        epoch += 1

def validate(audio_model, val_loader, args, valid=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    # id2phn
    id2phn = load_phn_dict(args.phn_dict)

    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []
    A_recog, A_recog_target = [], []

    with torch.no_grad():
        for i, (audio_input, audio_input2, audio_input3, canophns, realphns, bies, phn_label, word_label, utt_label, utt_id) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            audio_input2 = [ input2.to(device) for input2 in audio_input2 ]
            if isinstance(audio_input3, torch.Tensor):
                audio_input3 = audio_input3.to(device)
            canophns = canophns.to(device)
            realphns = realphns.to(device)

            mask = (phn_label>=0)

            # compute output
            u1, u2, u3, u4, u5, p, w1, w2, w3, logits = audio_model(
                audio_input, audio_input2, audio_input3, canophns, bies, mask=mask
            )

            p = p.to('cpu').detach()
            u1, u2, u3, u4, u5 = u1.to('cpu').detach(), u2.to('cpu').detach(), u3.to('cpu').detach(), u4.to('cpu').detach(), u5.to('cpu').detach()
            w1, w2, w3 = w1.to('cpu').detach(), w2.to('cpu').detach(), w3.to('cpu').detach()
            recogphns = torch.argmax(logits, dim=-1).to('cpu').detach().masked_select(mask)
            realphns = realphns.to('cpu').detach().masked_select(mask).int()
            canophns = canophns.to('cpu').detach().masked_select(mask).int()

            A_phn.append(p)
            A_phn_target.append(phn_label)

            A_u1.append(u1)
            A_u2.append(u2)
            A_u3.append(u3)
            A_u4.append(u4)
            A_u5.append(u5)
            A_utt_target.append(utt_label)

            A_w1.append(w1)
            A_w2.append(w2)
            A_w3.append(w3)
            A_word_target.append(word_label)

            A_recog.append(str(recogphns.tolist()))
            A_recog_target.append(str(realphns.tolist()))

        # phone level
        A_phn, A_phn_target  = torch.cat(A_phn), torch.cat(A_phn_target)

        # utterance level
        A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = torch.cat(A_u1), torch.cat(A_u2), torch.cat(A_u3), torch.cat(A_u4), torch.cat(A_u5), torch.cat(A_utt_target)

        # word level
        A_w1, A_w2, A_w3, A_word_target = torch.cat(A_w1), torch.cat(A_w2), torch.cat(A_w3), torch.cat(A_word_target)

        # get the scores
        phn_mse, phn_corr = valid_phn(A_phn, A_phn_target)

        A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
        utt_mse, utt_corr = valid_utt(A_utt, A_utt_target)

        A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
        word_mse, word_corr, word_stress, valid_word_pred, valid_word_target = valid_word(A_word, A_word_target)

        # compute phone error rate, wer(ref, hyp)
        per = wer(A_recog_target, A_recog)

    if valid:
        return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr, word_stress, per, \
                A_phn_target, valid_word_target, A_utt_target, A_recog_target, A_phn, valid_word_pred, A_utt, A_recog
    else:
        return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr, word_stress, per

def valid_phn(audio_output, target):
    valid_token_pred = []
    valid_token_target = []
    audio_output = audio_output.squeeze(2)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            # only count valid tokens, not padded tokens (represented by negative values)
            if target[i, j] >= 0:
                valid_token_pred.append(audio_output[i, j])
                valid_token_target.append(target[i, j])
    valid_token_target = np.array(valid_token_target)
    valid_token_pred = np.array(valid_token_pred)

    valid_token_mse = np.mean((valid_token_target - valid_token_pred) ** 2)
    corr = np.corrcoef(valid_token_pred, valid_token_target)[0, 1]
    return valid_token_mse, corr

def valid_utt(audio_output, target):
    mse = []
    corr = []
    for i in range(5):
        cur_mse = np.mean(((audio_output[:, i] - target[:, i]) ** 2).numpy())
        cur_corr = np.corrcoef(audio_output[:, i], target[:, i])[0, 1]
        mse.append(cur_mse)
        corr.append(cur_corr)
    return mse, corr

def valid_word(audio_output, target):
    word_id = target[:, :, -1]
    target = target[:, :, 0:3]

    valid_token_pred = []
    valid_token_target = []

    # unique, counts = np.unique(np.array(target), return_counts=True)
    # print(dict(zip(unique, counts)))

    # for each utterance
    for i in range(target.shape[0]):
        prev_w_id = 0
        start_id = 0
        # for each token
        for j in range(target.shape[1]):
            cur_w_id = word_id[i, j].int()
            # if a new word
            if cur_w_id != prev_w_id:
                # average each phone belongs to the word
                valid_token_pred.append(np.mean(audio_output[i, start_id: j, :].numpy(), axis=0))
                valid_token_target.append(np.mean(target[i, start_id: j, :].numpy(), axis=0))

                # sanity check, if the range indeed contains a single word
                if len(torch.unique(target[i, start_id: j, 1])) != 1:
                    print(target[i, start_id: j, 0])
                # if end of the utterance
                if cur_w_id == -1:
                    break
                else:
                    prev_w_id = cur_w_id
                    start_id = j

    valid_token_pred = np.array(valid_token_pred)
    # this rounding is to solve the precision issue in the label, round(2) ?
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list, corr_list = [], []
    # for each (accuracy, stress, total) word score
    for i in range(3):
        valid_token_mse = np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2)
        corr = np.corrcoef(valid_token_pred[:, i], valid_token_target[:, i])[0, 1]
        mse_list.append(valid_token_mse)
        corr_list.append(corr)

    # NOTE: stress macro/micro f1
    stress_list = []
    N = valid_token_pred[:, 1].shape
    hyp = valid_token_pred[:, 1]
    hyp = np.around(valid_token_pred[:, 1]) # 1-5: 0.2 to 1 / 6-10: 1.2 to 2
    ref = valid_token_target[:, 1]
    stress_list.append(f1_score(ref, hyp, average='macro'))
    stress_list.append(f1_score(ref, hyp, average='micro'))

    return mse_list, corr_list, stress_list, valid_token_pred, valid_token_target


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--warmup-step", type=int, default=100, help="number of steps for warmup")
    parser.add_argument("--phase-ratio", type=list, default=[0.4, 0.4, 0.2], help="phase ratio used in tri-stage scheduler")
    parser.add_argument("--init-lr-scale", type=float, default=1e-2)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=25, help="training batch size")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of maximum training epochs")
    parser.add_argument("--loss-w-phn", type=float, default=1, help="weight for phoneme-level loss")
    parser.add_argument("--loss-w-word", type=float, default=1, help="weight for word-level loss")
    parser.add_argument("--loss-w-utt", type=float, default=1, help="weight for utterance-level loss")
    parser.add_argument("--loss-type", type=str, default="dexent", choices=['xent', 'dexent', 'none'], help="loss type for xent")
    parser.add_argument("--loss-w-a", type=float, default=0.7, help="weight for controlling mispronunciation part's magnitude")
    parser.add_argument("--loss-w-xent", type=float, default=0.003, help="weight for xent loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--save-last-epoch", action='store_true', default=False)
    parser.add_argument("--model", type=str, default='gopt', help="name of the model")
    parser.add_argument("--model-conf", type=str, help="model config")
    parser.add_argument("--am", type=str, default='librispeech', help="name of the acoustic models")
    parser.add_argument("--noise", type=float, default=0., help="the scale of random noise added on the input GoP feature")
    parser.add_argument("--gop-dir", type=str, default=None, help="train & test data directory for experiments")
    parser.add_argument("--ssl-dir", type=str, default=None, help="extra ssl data")
    parser.add_argument("--raw-dir", type=str, default=None, help="extra raw audio data")
    parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
    parser.add_argument("--phn-dict", type=str, default='prep_data/phn_dict_all.txt')

    args = parser.parse_args()

    wandb.init(project="hmamba", name=args.exp_dir)

    # NOTE: set seed
    print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
    print("setting seed %d" %(args.seed))
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    am = args.am
    print('now train with {:s} acoustic models'.format(args.am))

    conf = load_conf(args.model_conf)

    if args.model == 'HMamba':
        print('now train a HMamba model')
        audio_mdl = HMamba(**conf)
    else:
    	raise ValueError(f"Invalid model {args.model}")

    tr_dataset = GoPDataset('train', data_dir=args.gop_dir, data_dir2=args.ssl_dir, data_dir3=args.raw_dir, am=am)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    te_dataset = GoPDataset('test', data_dir=args.gop_dir, data_dir2=args.ssl_dir, data_dir3=args.raw_dir, am=am)
    te_dataloader = DataLoader(te_dataset, batch_size=2500, shuffle=False)

    train(audio_mdl, tr_dataloader, te_dataloader, args)

    wandb.finish()
