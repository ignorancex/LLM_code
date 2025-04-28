# train_model.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["RAY_TEMP_DIR"] = ""  # 修改为您的路径

import ESM3
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import esm
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss
)
from scipy.stats import spearmanr
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import ray
from ray.tune.schedulers import ASHAScheduler
from ray import tune, train
from .model import Model, esm2_model_mapping
from ray.tune import CLIReporter

# 初始化 Ray
ray.init(ignore_reinit_error=True, _temp_dir="your path")  # 修改为您的路径
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def find_best_threshold(y_true, y_pred_proba):
    best_threshold = 0.0
    best_f1 = 0.0
    thresholds = np.linspace(y_pred_proba.min(), y_pred_proba.max(), num=100)
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def load_data(config):
    # df = pd.read_excel(filepath)
    df = pd.read_excel(config['file_path'])

    sequences = df[config['data_columns']].tolist()
    labels = df[config['label_column']].apply(lambda x: 1 if x < 0.5 else 0).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.4, random_state=2, stratify=labels
    )
    return X_train, y_train, X_test, y_test

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, batch_converter):
        self.sequences = sequences
        self.labels = labels
        self.batch_converter = batch_converter

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        data = [("sequence", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens.squeeze(0), label

    
def train_model(config, checkpoint_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = config['model_name']
    if 'esm2' in model_path:
        pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        embedding_dim, layers = esm2_model_mapping[model_path]['embedding_dim'],esm2_model_mapping[model_path]['layers']
    else:
        pretrained_model, alphabet = ESM3.from_pretrained("esm3_sm_open_v1")
    pretrained_model.eval()
    batch_converter = alphabet.get_batch_converter()
    pretrained_model = pretrained_model.to(device)

    # embedding_dim, layers = esm2_model_mapping[model_path]['embedding_dim'],esm2_model_mapping[model_path]['layers']

    model = Model(pretrained_model=pretrained_model, embedding_dim=embedding_dim, output_dim=1, self_attention_layers=True, repr_layers=layers, dropout=float(config["dropout"]))
    model = model.to(device)
    model.train()

    # sequences, labels = load_data(config)
    X_train, y_train, X_val, y_val = load_data(config)

    train_dataset = SequenceDataset(X_train, y_train, batch_converter)
    val_dataset = SequenceDataset(X_val, y_val, batch_converter)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    batch_size = int(config.get("batch_size"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    accumulation_steps = int(config.get("accumulation_steps"))  # 梯度累积步数

    # 初始化
    best_val_f1 = 0.0
    best_epoch = 0
    for epoch in range(int(config["num_epochs"])):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for i, (batch_tokens, labels_batch) in enumerate(train_loader):
            batch_tokens, labels_batch = batch_tokens.to(device), labels_batch.to(device)
            labels_batch = labels_batch.float()

            outputs, _ = model(batch_tokens, mask=batch_tokens != alphabet.padding_idx)
            loss = criterion(outputs, labels_batch)
            loss = loss / accumulation_steps  # 梯度累积

            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps  # 累积损失

        # 验证模型
        model.eval()
        val_loss = 0.0
        all_y_true = []
        all_y_pred_proba = []
        with torch.no_grad():
            for batch_tokens, labels_batch in val_loader:
                batch_tokens, labels_batch = batch_tokens.to(device), labels_batch.to(device)
                labels_batch = labels_batch.float()

                outputs, _ = model(batch_tokens, mask=batch_tokens != alphabet.padding_idx)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()

                outputs_np = torch.sigmoid(outputs).cpu().numpy()
                all_y_pred_proba.extend(outputs_np)
                all_y_true.extend(labels_batch.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        # 找到最佳阈值
        all_y_pred_proba = np.array(all_y_pred_proba)
        best_threshold = find_best_threshold(all_y_true, all_y_pred_proba)
        all_y_pred = (all_y_pred_proba >= best_threshold).astype(int)

        accuracy = accuracy_score(all_y_true, all_y_pred)
        precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        roc_auc = roc_auc_score(all_y_true, all_y_pred_proba)
        pr_auc = average_precision_score(all_y_true, all_y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = matthews_corrcoef(all_y_true, all_y_pred)
        kappa = cohen_kappa_score(all_y_true, all_y_pred)
        logloss = log_loss(all_y_true, all_y_pred_proba)
        brier = brier_score_loss(all_y_true, all_y_pred_proba)
        srcc, _ = spearmanr(all_y_true, all_y_pred_proba)

        fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(all_y_true, all_y_pred_proba)


        print(f"Epoch [{epoch+1}/{int(config['num_epochs'])}]")
        print(f"Training Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Predictions distribution: {np.bincount(all_y_pred)}")
        print(f"True labels distribution: {np.bincount(np.array(all_y_true, dtype=int))}")

        cm = confusion_matrix(all_y_true, all_y_pred)
        print('Confusion Matrix:')
        print(cm)
        print("-" * 50)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_epoch = epoch + 1
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            else:
                checkpoint_dir = './'
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       checkpoint_path)
        train.report({
            'loss': float(val_loss),
            'f1': float(f1),
            'roc_auc':float(roc_auc),
            'accuracy':float(accuracy),
            'mcc':mcc,
            'precision':precision, 
            'recall':float(recall), 
            'f1':float(f1), 
            'best_threshold':float(best_threshold),
            'y_true':all_y_true.tolist() if type(all_y_true) != list else all_y_true,
            'y_pred':all_y_pred.tolist() if type(all_y_pred) != list else all_y_pred,
            'y_pred_proba':all_y_pred_proba.astype(float).tolist() if type(all_y_pred_proba) != list else all_y_pred_proba,
            'roc_auc':float(roc_auc),
            'pr_auc':float(pr_auc),
            'specificity':specificity,
            'mcc':mcc,
            'kappa':kappa,
            'logloss':logloss,
            'brier':float(brier),
            'srcc':float(srcc),
            'fpr':fpr.astype(float).tolist(),
            'tpr':tpr.astype(float).tolist(),
            'precision_curve':precision_curve.astype(float).tolist(),
            'recall_curve':recall_curve.astype(float).tolist()
            })
    print(f"Best F1 Score: {best_val_f1:.4f} at epoch {best_epoch}")

def main_sweet(config, num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    if config['lr'] is None:
        config['lr'] = tune.loguniform(1e-6, 1e-3)
    if config['dropout'] is None:
        config['dropout'] = tune.uniform(0.001, 0.3)
    if config['num_epochs'] is None:
        config['num_epochs'] = 30

    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=2,
        grace_period=1,
        reduction_factor=2
    )

    result = tune.run(
        train_model,
        resources_per_trial={"cpu": int(config["cpu_per_trial"]), "gpu": float(config["gpus_per_trial"])},
        config=config,
        num_samples=int(config['num_samples']),
        scheduler=scheduler,
        # progress_reporter=reporter,
        # storage_path='./'  #
    )

    best_trial = result.get_best_trial("f1", "max", "last")
    print(f"Best trial found at {best_trial}")
    print(f"Best trial final validation F1 score: {best_trial.last_result['f1']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    plt.figure()
    plt.plot(best_trial.last_result['fpr'], best_trial.last_result['tpr'], color='darkorange', lw=2, label=f'ROC curve (area = {best_trial.last_result["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Epoch 0)')
    plt.legend(loc="lower right")
    plt.savefig('./roc_curve_classification_ray.png')
    plt.close()

    plt.figure()
    plt.plot(best_trial.last_result['recall_curve'] , best_trial.last_result['precision_curve'], color='red', lw=2, label=f'PR curve (area = {best_trial.last_result["pr_auc"]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Epoch 0)')
    plt.legend(loc="lower left")
    plt.savefig('./pr_curve_classification_ray.png')
    plt.close()

    with open(f'./result_automl_classification_ray_woHPO2.json', 'w') as f:
        json.dump(best_trial.last_result, f)


# if __name__ == "__main__":
# main_sweet()