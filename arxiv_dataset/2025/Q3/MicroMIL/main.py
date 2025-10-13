import argparse
from sklearn.metrics import  f1_score, accuracy_score,roc_auc_score

import numpy as np
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from dataset import *

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def train_model(args):
    args = args
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)

    
    criterion1 = nn.BCELoss()

    model = ClusterAttentionModel(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train  = Dataset_image('Image',True)
    test =  Dataset_image('Image',False)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False ) 
    
    patience = args.patience
    best_f1 = 0
    best_epoch = 0
    
    with open(f"D:/experiment_results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.model_name}_{args.seed}_{args.lr}.txt", "w") as f:
        for epoch in range(args.epoch):
            train_loss,train_loss2 = 0,0
            for imgs, label in tqdm(train_loader):
                y_prob = model(imgs.to(args.device))
                loss1 = criterion1(F.softmax(y_prob)[1], label.squeeze().float().to(args.device))
                loss = loss1 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

       
            model.eval()
            true_labels, predicted_labels, predicted_probs = [], [], []
            test_loss = 0
            with torch.no_grad():
                for imgs, label in test_loader:
                    y_prob = model(imgs.to(args.device)) 
                    test_loss += criterion1(F.softmax(y_prob)[1], label.squeeze().float().to(args.device))
                    _, predicted = torch.max(y_prob, 0)
                    true_labels.extend(label.cpu().numpy())
                    predicted_labels.append(predicted.cpu().numpy())  
                    predicted_probs.append(F.softmax(y_prob)[1].cpu().detach().numpy()) 

            auc = roc_auc_score(true_labels, predicted_probs)
            f1 = f1_score(true_labels, predicted_labels)
            acc = accuracy_score(true_labels, predicted_labels)

            f.write(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Train KD Loss: {train_loss2/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f} , F1 Score: {f1:.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}\n")
            print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Train KD Loss: {train_loss2/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f} , F1 Score: {f1:.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
            else:
                if epoch - best_epoch >= patience:
                    f.write(f'Early stopping on epoch {epoch+1}\n')
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--cluster_number', type=int, default=36, help='cluster number')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--model_name', type=str, default='resnet18', help='model_name')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--layer', type=int, default=2, help='layer')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu')
    parser.add_argument('--shuffle',default=False, help='gpu')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
    parser.add_argument('--num_classes', type=int, default=2, help='output_dim')
    parser.add_argument('--dropout_node', type=float, default=0.5, help='drop_rate')
    parser.add_argument('--type', type=str, default='graph', help='model_type')
    args = parser.parse_args()
    print(args)
    train_model(args)
