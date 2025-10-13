# -*- coding: UTF-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm, trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark = True  # Provides a speedup
from torch.cuda.amp import autocast, GradScaler
import util
import test_FoL
import my_parser
import commons
import datasets
import network_FoL
from loss import loss_function, loss_function_local, loss_sep, loss_function_weak_supervision
from dataloaders.GSVCities import get_GSVCities
import warnings
warnings.filterwarnings("ignore")

args = my_parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")
args.features_dim = 8448

model = network_FoL.FoLNet()
model = model.to(args.device)
model = torch.nn.DataParallel(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_size_bytes = num_params * 4
total_size_mb = total_size_bytes / 1048576
print(f'Total number of trainable parameters: {num_params}')
print(f'Total size of parameters: {total_size_mb:.2f} MB')

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
elif args.optim == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=9.5e-9)

best_r5 = start_epoch_num = not_improved_num = 0

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Getting GSVCities
train_dataset = get_GSVCities(args.resize)

train_loader_config = {
    'batch_size': args.train_batch_size,
    'num_workers': args.num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': False}
    
logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")
args.resize=[504,504]
test_ds = datasets.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

#### Training loop
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000)
scaler = GradScaler()

for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_losses = []
    epoch_local_losses = []
    epoch_global_losses = []
    WS_losses=[]
    model = model.train()

    for images, place_id, img_names in tqdm(ds):
        BS, N, ch, h, w = images.shape
        images = images.view(BS * N, ch, h, w)
        labels = place_id.view(-1)
        image_names = []
        for i in range(len(place_id)):
            image_names.append(img_names[0][i])
            image_names.append(img_names[1][i])
            image_names.append(img_names[2][i])
            image_names.append(img_names[3][i])
        optimizer.zero_grad()
        with autocast():
            descriptors, local_f, loss_kl_1, loss_kl_2, seps, weak_supervision_info, mask_guide = model(images.to(args.device))
            descriptors = descriptors.cuda()
            loss = loss_function(descriptors, labels)
            epoch_global_losses.append(loss.item())
            local_f = local_f.cuda()
            loss_local = loss_function_local(descriptors, local_f, labels)
            loss_separate = loss_sep(seps, labels)
            device = loss_kl_1.device
            scale_factor = torch.tensor(5e-3, device=device, dtype=torch.float32)
            loss_kl_1 = loss_kl_1.mean()
            loss_kl_2 = loss_kl_2.mean()
            loss_weak_supervision = loss_function_weak_supervision(descriptors, weak_supervision_info, labels, mask_guide, image_names)
            logging.info(
                f'loss_global: {loss.item()}, loss_local: {loss_local.item()}, loss_kl_1: {loss_kl_1.item()}, loss_kl_2: {loss_kl_2.item()}, loss_separate: {loss_separate.item()}, loss_weak_supervision: {loss_weak_supervision}')
            sys.stdout.flush()
            loss = loss + loss_local + loss_kl_1 * scale_factor + loss_kl_2 * scale_factor + loss_separate + loss_weak_supervision
            epoch_local_losses.append(loss_local.item())
            if isinstance(loss_weak_supervision, float):
                WS_losses.append(loss_weak_supervision)
            else:
                WS_losses.append(loss_weak_supervision.item())
            del descriptors
            del local_f
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses.append(batch_loss)
        del loss

    average_global_loss = np.mean(epoch_global_losses) if epoch_global_losses else 0.0
    average_local_loss = np.mean(epoch_local_losses) if epoch_local_losses else 0.0
    average_ws_loss = np.mean(WS_losses) if WS_losses else 0.0
    # average_total_loss = average_global_loss + average_local_loss
    average_loss = np.mean(epoch_losses) if epoch_losses else 0.0
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch global loss = {average_global_loss:.4f}, average epoch local loss = {average_local_loss:.7f}, "
                 f"average epoch ws loss = {average_ws_loss:.7f}"
                 f"average epoch total loss = {average_loss:.6f}")


    # Compute recalls on test set
    recalls, recalls_str, recalls_rerank, recalls_str_rerank = test_FoL.test(args, test_ds, model)
    logging.info(f"Recalls on test set {test_ds}: {recalls_str}")
    logging.info(f"Reranking recalls on test set {test_ds}: {recalls_str_rerank}")

    is_best = recalls_rerank[1] > best_r5

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                 "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls_rerank,
                                 "best_r5": best_r5,
                                 "not_improved_num": not_improved_num
                                 }, is_best, filename="model.pth", recalls_rerank=recalls_rerank)

    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@5 = {best_r5:.2f}, current R@5 = {(recalls_rerank[1]):.2f}")
        best_r5 = (recalls_rerank[1])
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.2f}, current R@5 = {(recalls_rerank[1]):.2f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best R@5: {best_r5:.2f}")
logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
