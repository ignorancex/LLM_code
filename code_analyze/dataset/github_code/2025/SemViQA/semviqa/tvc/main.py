import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, XLMRobertaTokenizerFast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import os
import time
import multiprocessing

from data_utils import Data
from model import ClaimVerificationConfig, ClaimVerificationModel

multiprocessing.set_start_method('spawn', force=True)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    print(f'The model has {sum(p.numel() for p in model.parameters()):,} parameters')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = pd.read_csv(args.train_data).reset_index()
    dev_data = pd.read_csv(args.dev_data).reset_index()

    train_data["id"] = train_data.index
    dev_data["id"] = dev_data.index

    if args.n_classes == 2:
        train_data = train_data[train_data["verdict"] != "NEI"]
        dev_data = dev_data[dev_data["verdict"] != "NEI"]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    config = ClaimVerificationConfig(
        model_name=args.model_name,
        num_labels=args.n_classes,
        dropout=args.dropout_prob,
        loss_type=args.type_loss
    )
    model = ClaimVerificationModel(config).to(device)
    count_parameters(model)

    train_dataset = Data(train_data, tokenizer, args, max_len=args.max_len)
    dev_dataset = Data(dev_data, tokenizer, args, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0, 
                num_training_steps=len(train_loader)*args.epochs
            )

    info_epoch = {}
    logs = []
    best_acc = 0
    cnt = 0
    i = 0

    total_time = time.time()
    for epoch in range(args.epochs):
        set_seed(42)
        info_epoch[epoch] = {}
        print(f'Epoch {epoch+1}/{args.epochs}')
        print('-'*30)
        start_training_time = time.time()

        model.train()
        train_losses = []
        true_labels = []
        predicted_labels = []

        for data in tqdm(train_loader):
            y_true = data['targets'].to(device)
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_masks'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=y_true)
            loss = outputs["loss"]
            logits = outputs["logits"] 
            train_losses.append(loss.item())

            loss.backward()
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step() 
                optimizer.zero_grad()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            i += 1

            _, pred = torch.max(logits, dim=1)
            true_labels.extend(y_true.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())

        train_f1 = f1_score(true_labels, predicted_labels, average='macro')
        train_acc = accuracy_score(true_labels, predicted_labels)
        epoch_training_time = time.time() - start_training_time
        print(f'Training time: {epoch_training_time:.2f}s Train Loss: {np.mean(train_losses):.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f}')

        model.eval()
        eval_losses = []
        y_true_list = []
        y_pred_list = []

        start_eval_time = time.time()
        with torch.no_grad():
            for data in tqdm(dev_loader):
                y_true = data['targets'].to(device)
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_masks'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=y_true)
                loss = outputs["loss"]
                logits = outputs["logits"]
                eval_losses.append(loss.item())

                _, pred = torch.max(logits, dim=1)
                y_true_list.extend(y_true.cpu().numpy())
                y_pred_list.extend(pred.cpu().numpy())

        dev_f1 = f1_score(y_true_list, y_pred_list, average='macro')
        dev_acc = accuracy_score(y_true_list, y_pred_list)
        epoch_eval_time = time.time() - start_eval_time

        print(f'Dev time: {epoch_eval_time}s Dev Loss: {np.mean(eval_losses):.4f} F1: {dev_f1:.4f} Acc: {dev_acc:.4f}')

        info_epoch[epoch] = {
            "time_train": epoch_training_time,
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            "train_acc": train_acc,
            "f1-train": train_f1,
            "time_val": epoch_eval_time,
            "val_acc": dev_acc,
            "val_loss": np.mean(eval_losses),
            "f1-val": dev_f1
        }

        if dev_f1 > best_acc:
            cnt = 0
            tokenizer.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
            config.save_pretrained(output_dir)
            print(f'Saved best_model at epoch {epoch+1}')
            best_acc = dev_f1
        else:
            cnt += 1

        if cnt >= args.patience:
            print('Early stopping')
            break

        torch.cuda.empty_cache()

    print(f'Total training time: {time.time() - total_time:.2f}s')
    pd.DataFrame(info_epoch).T.to_csv(os.path.join(output_dir, 'info_epoch.csv'), index=False)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/train.csv')
    parser.add_argument('--dev_data', type=str, default='data/dev.csv')
    parser.add_argument('--model_name', type=str, default='MoritzLaurer/ernie-m-large-mnli-xnli')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--type_loss', type=str, default='cross_entropy')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--is_weighted', type=int, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
