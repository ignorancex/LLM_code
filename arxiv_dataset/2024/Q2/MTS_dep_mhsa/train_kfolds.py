import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import sys
from torch.cuda.amp import autocast as autocast, GradScaler
from models.dep_mhsa.dep_mhsa_network import resnet18_depmhsa
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import torch
import copy
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import time
import torch.nn.functional as F
from ct_data_loader import CtImages
import argparse
import pickle
from IPython.display import clear_output, display, HTML
from matplotlib import pyplot as plt
import torchvision
from tqdm import tqdm
from collections import defaultdict
from utils.early_stop import EarlyStopping
from utils.metric_evaluate import *
from utils.train_utils import *
from einops import rearrange, repeat, reduce


parser = argparse.ArgumentParser(description="BabyNet for CT disease prediction")
parser.add_argument("--data",
                    type=str,
                    default="/home/yixin/study/phd_program/changshu_files/processed_data/anonymized_ct_scans/",
                    help="Path to the data directory.")

parser.add_argument("--x_img_size",
                    type=int,
                    default=128,
                    help="Input X image size.")

parser.add_argument("--y_img_size",
                    type=int,
                    default=128,
                    help="Input Y image size")

parser.add_argument("--batch_size",
                    type=int,
                    default=32,  # 48
                    help="Number of batch size.")

parser.add_argument("--epochs",
                    type=int,
                    default=100,
                    help="Number of epochs.")

parser.add_argument("--lr",
                    type=float,
                    default=3e-4, #8e-4, 5e-4
                    help="Number of learning rate.")

parser.add_argument("--step_lr",
                    type=int,
                    default=25, #25,
                    help="Step of learning rate")

parser.add_argument("--w_decay",
                    type=float,
                    default=0.0001,
                    help="Number of weight decay.")

parser.add_argument("--GPU",
                    type=bool,
                    default=True,
                    help="Use GPU.")

parser.add_argument("--display_steps",
                    type=int,
                    default=50,
                    help="Number of display steps.")


parser.add_argument("--frames_num",
                    type=int,
                    default=12,
                    help="Number of frames in chunk")

parser.add_argument("--skip_frames",
                    type=int,
                    default=0,
                    help="Number of frames to skip")

parser.add_argument("--pixels_crop_height",
                    type=int,
                    default=260, # 260
                    help="Number of frames in chunk")

parser.add_argument("--pixels_crop_width",
                    type=int,
                    default=260, # 260
                    help="Number of frames in chunk")

parser.add_argument("--msha3D",
                    type=bool,
                    default=True,
                    help='Add MSHA to ResNet3D')

parser.add_argument("--valid_batch_size",
                    type=int,
                    default=32,
                    help='batch size for validation')

args = parser.parse_args()


seed = 114514
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True


data_folder = "/home/yixin/study/phd_program/changshu_files/"
save_folder = "/home/yixin/study/phd_program/changshu_files/tmp_results/"
args.model_name = "mts_diagnosis_ct_scans_dep_mhsa"
train_name = "train_data.csv"
test_name = "test_data.csv"

scaler = GradScaler()
train_set = CtImages(input_path=args.data,
                origin_size_x=512,
                origin_size_y=512,
                x_image_size=args.x_img_size,
                y_image_size=args.y_img_size,
                pixels_crop_height=args.pixels_crop_height,
                pixels_crop_width=args.pixels_crop_width,
                skip_frames=args.skip_frames,
                n_frames=args.frames_num,
                mode='train',
                diagnosis_labels_path = data_folder+train_name)

test_set = CtImages(input_path=args.data,
                origin_size_x=512,
                origin_size_y=512,
                x_image_size=args.x_img_size,
                y_image_size=args.y_img_size,
                pixels_crop_height=args.pixels_crop_height,
                pixels_crop_width=args.pixels_crop_width,
                skip_frames=args.skip_frames,
                n_frames=args.frames_num,
                mode='val',
                diagnosis_labels_path = data_folder+test_name)

if args.GPU and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
# Initialize the k-fold cross validation
kf = KFold(n_splits=5, shuffle=True)

test_loader = DataLoader(dataset=test_set,
                        batch_size=args.valid_batch_size, pin_memory=True)

seeds = [114514, 3407, 79486, 66141, 1408, 8818, 9915, 1103, 33453, 8777, 12]

for seed in seeds:
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_set)):
        print(f"Fold {fold + 1}")
        print("-------")
        criterion_reg = nn.CrossEntropyLoss(weight=torch.Tensor([1, 3]).to(device)) # label balance 1:3
        loss_min = np.inf
        # Start time of learning
        total_start_training = time.time()

        train_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size, pin_memory=True,
                                sampler=torch.utils.data.SubsetRandomSampler(train_idx),)
        valid_loader = DataLoader(dataset=train_set,
                                batch_size=args.valid_batch_size, pin_memory=True,
                                sampler=torch.utils.data.SubsetRandomSampler(test_idx),)
        
        model = resnet18_depmhsa(pretrained=False, num_classes=2, 
                                    input_size=(args.y_img_size, args.x_img_size), 
                                    n_frames=args.frames_num,  args=args)
        model.to(device)
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        scheduler = StepLR(optimizer=optimizer, 
                            step_size=args.step_lr, 
                            gamma=0.25, 
                            verbose=True)

        best_val_score = -np.inf
        best_val_preds = None
        best_val_acc = 0.0
        history_acc = {"train": [], "valid": [], 'test':[]}
        history_f1 = {"train": [], "valid": []}
        early_stopper = EarlyStopping(patience=30, min_delta=0.02, restore_best_weights=True)
        acc_cnt = 0

        for epoch in range(args.epochs):
            start_time_epoch = time.time()
            print(f"Starting epoch {epoch + 1}")
            model.train()
            running_loss = 0.0
            y_true_list = []
            y_pred_list = []
            patient_running_loss = []
            stream = tqdm(train_loader)
            metric_monitor_train = MetricMonitor()
            for batch_idx, (videos, y_true, patient_id, first_frame) in enumerate(stream):
                optimizer.zero_grad()
                # plot the data sample
                # if  epoch == 0 and batch_idx == 0:
                #     print(videos.size()) # H W C
                #     plot_first_frame(videos, patient_id)
                #     plot_first_person(videos)
                videos = videos.to(device=device).float()
                y_true_list.extend(y_true.flatten().cpu().tolist())
                y_true = y_true.to(device=device).long()

                with autocast():
                    reg_out = model(videos)
                    y_pred_list.extend(torch.argmax(F.softmax(reg_out, dim=1),dim=1).tolist())
                    loss = criterion_reg(reg_out, y_true)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # reg_out = model(videos)
                # y_pred_list.extend(torch.argmax(F.softmax(reg_out, dim=1),dim=1).tolist())
                # loss = criterion_reg(reg_out, y_true)
                # loss.backward()
                # optimizer.step()


                running_loss += loss.item()
                record_dict = calculate_metrics(torch.Tensor(torch.argmax(F.softmax(reg_out.cpu().float(), dim=1), dim=1).tolist()),
                                                y_true.cpu())
                cnt_metric = len(y_true)
                metric_monitor_train.update('Loss', loss.item(), cnt_metric)
                metric_monitor_train.update('Accuracy', record_dict['acc'], cnt_metric)

                stream.set_description(
                    "Model: {model_name}. Epoch: {epoch}. Trainning. {metric_monitor}".format(
                        model_name = args.model_name[17:33],
                        epoch=epoch,
                        metric_monitor=metric_monitor_train)
                )
                
            history_acc['train'].append(calculate_metrics(y_pred_list, y_true_list)['acc'])

            model.eval()
            val_loss, val_acc = validation_run(valid_loader, device, model, criterion_reg)
            test_loss, test_acc = validation_run(test_loader, device, model, criterion_reg, val_or_test="Test")

            train_loss = running_loss / len(train_loader)
            history_acc['valid'].append(val_acc)
            history_acc['test'].append(test_acc)

            
            print(f"Train Loss: {train_loss:.3f} ",
                f"\nVal Loss: {val_loss:.3f}",
                f"\nTrain Acc: {history_acc['train'][-1]:.3f}",
                f"\nValid Acc: {history_acc['valid'][-1]:.3f}",
                f"\nTest Acc: {history_acc['test'][-1]:.3f}",
                )
            
            curr_acc = history_acc['valid'][-1]
            early_stopper(model, val_loss, train_acc=history_acc['train'][-1], val_acc=curr_acc, epoch=epoch)
            best_for_test = history_acc['test'][np.argmax(history_acc['valid'])]
            # save_path = save_folder + f"{args.model_name}_seed_{seed}_fold_{fold}_with_early_stop_train_acc_{str(early_stopper.best_acc)[:5]}.pt"
            save_path = save_folder + f"{args.model_name}_seed_{seed}_fold_{fold}_with_early_stop_test_acc_{best_for_test}.pt"
            print(early_stopper.status)
            
            if early_stopper.early_stop:
                print('    ', end='')
                print(f"Train Loss: {train_loss:.3f} ",
                    f"\nVal Loss: {val_loss:.3f}",
                    f"\nTrain Acc: {history_acc['train'][-1]:.3f}",
                    f"\nValid Acc: {history_acc['valid'][-1]:.3f}",
                    f"\nTest Acc: {history_acc['test'][-1]:.3f}",
                    )
                torch.save(early_stopper.best_model_state, save_path)
                print(f"Current best val accuracy {str(early_stopper.best_acc)[:5]}. Model saved!")
                break

            if  np.round(curr_acc,3) > best_val_acc and (epoch > int(args.epochs*0.9)):
                best_val_acc = np.round(curr_acc, 3)
                best_weights = copy.deepcopy(model.state_dict())
            scheduler.step()

        print('Training finished, took {:.2f}s'.format(time.time() - total_start_training))
        last_mean_val_acc = np.round(np.mean(history_acc['test'][-10:]),2)

        # with open(save_folder + f'{args.model_name}_seed_{seed}_fold_{fold}_with_early_stop_history_train_{last_mean_val_acc}_best_{str(early_stopper.best_acc)[:5]}.pickle', 'wb') as handle:
        with open(save_folder + f'{args.model_name}_seed_{seed}_fold_{fold}_with_early_stop_history_train_{last_mean_val_acc}_best_for_test_{best_for_test}.pickle', 'wb') as handle:
            pickle.dump(history_acc, handle,  protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(early_stopper.best_model_state, save_path)

