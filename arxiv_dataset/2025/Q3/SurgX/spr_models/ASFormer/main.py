import torch
 
from causal_model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='predict')
parser.add_argument('--dataset', default="cholec80")
parser.add_argument('--split', default='2')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')

args = parser.parse_args()
 
# num_epochs = 120
num_epochs = 13

lr = 0.0005
num_layers = 10
num_f_maps = 64
# features_dim = 2048
features_dim = 768
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    
if args.dataset == 'breakfast':
    lr = 0.0001


def assert_model_causal(model, C_in=64, T=128, split=None, tol=1e-7, device=device):
    """
    모델이 진짜 causal인지 검증:
    - t < split 구간의 출력이, t >= split 입력을 바꿔도 동일해야 함.
    - 또한 미래 구간 마스크를 0으로 꺼도 과거 출력이 동일해야 함.
    """
    model.eval()  # dropout 등 비활성화
    if split is None:
        split = T // 2

    x = torch.randn(1, C_in, T, device=device)
    mask = torch.ones(1, 1, T, device=device)

    with torch.no_grad():
        y_full = model(x, mask)[-1].clone()  # (B, num_classes, T)

        # 미래 절반을 크게 교란
        x2 = x.clone()
        x2[:, :, split:] = x2[:, :, split:] + torch.randn_like(x2[:, :, split:]) * 10.0
        y_perturb = model(x2, mask)[-1].clone()

        # 과거 절반이 완전히 동일해야 함
        diff_max = (y_full[:, :, :split] - y_perturb[:, :, :split]).abs().max().item()
        print(f"[Causality check] max |Δ| on past segment = {diff_max:.3e} (tol={tol})")
        if diff_max > tol:
            raise AssertionError("❌ Non-causal behaviour detected (past outputs changed when future inputs perturbed).")
        else:
            print("✅ Passed: model is causal w.r.t. input features.")

    # 마스크 변화에 대해서도 동일 검증 (미래 구간 마스크만 바꿔도 과거 출력은 동일해야 함)
    with torch.no_grad():
        y_full2 = model(x, mask)[-1].clone()

        mask2 = mask.clone()
        mask2[:, :, split:] = 0  # 미래 프레임 무효화
        y_masked = model(x, mask2)[-1].clone()

        diff_max_mask = (y_full2[:, :, :split] - y_masked[:, :, :split]).abs().max().item()
        print(f"[Mask causality] max |Δ| on past segment = {diff_max_mask:.3e} (tol={tol})")
        if diff_max_mask > tol:
            raise AssertionError("❌ Non-causal masking effect detected.")
        else:
            print("✅ Passed: masking future frames does not alter past outputs.")

 
vid_list_file = "/data2/local_datasets/kayoung_data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "/data2/local_datasets/kayoung_data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "/data2/local_datasets/kayoung_data/"+args.dataset+"/features/lovit_for_asformer/"
gt_path = "/data2/local_datasets/kayoung_data/"+args.dataset+"/groundtruth/"
 
mapping_file = "/data2/local_datasets/kayoung_data/"+args.dataset+"/mapping.txt"

model_dir = "./{}/".format(args.model_dir)+args.dataset+"/split_"+args.split

results_dir = "./{}/".format(args.result_dir)+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)
    
if args.action == "extract_activations":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.extract_activations(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

if args.action == "extract_contributions":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.extract_contributions(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)