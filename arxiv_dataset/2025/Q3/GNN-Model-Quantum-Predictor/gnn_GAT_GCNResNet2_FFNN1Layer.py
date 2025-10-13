
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from model import QuantumGAT
#import matplotlib.pyplot as plt
#import seaborn as sns

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

perc_0 = 93 / (93 + 45 + 172 + 188)
perc_1 = 1 - perc_0

# Define the number of classes for the classification task
num_classes = 2  
# load the dataset
dataset = torch.load('gnn_dataset.pt', weights_only=False)
for data in dataset:
  data.y = torch.Tensor([0 if data.y == 0 else 1])




def split_dataset(dataset, train_ratio=0.8):
    """
    Splits the dataset into training and test sets while preserving class distribution (stratification).

    Parameters:
        dataset (list): The dataset (list of PyG Data objects).
        train_ratio (float): The proportion of the dataset to use for training.

    Returns:
        tuple: (train_set, test_set)
    """
    labels = [data.y.item() for data in dataset]  # Extract class labels
    train_data, test_data = train_test_split(dataset, train_size=train_ratio, stratify=labels)
    
    return train_data, test_data




def train_model(model, train_loader, epochs=30, lr=0.005):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([perc_1, perc_0]).to(device))


    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)  # Shape: [batch_size, num_classes]

            loss = loss_fn(output, batch.y.view(-1).long())  # Fix batch.y shape
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

            # Accuracy computation
            preds = output.argmax(dim=1)
            correct += (preds == batch.y.view(-1)).sum().item()
            total += batch.num_graphs

        scheduler.step()  # Step the scheduler per epoch, not per batch
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={accuracy*100:.2f}%")

def evaluate_model(model, test_loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)  # Shape: [batch_size, num_classes]
            loss = loss_fn(output, batch.y.view(-1).long())
            total_loss += loss.item()

            preds = output.argmax(dim=1)
            correct += (preds == batch.y.view(-1)).sum().item()
            total += batch.num_graphs

    avg_loss = total_loss / len(test_loader)
    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")


def getConfusion(model, test_loader, reps, hidden_dim, heads, unit1, unit2, loader='test'):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)  # Shape: [batch_size, num_classes]
            loss = loss_fn(output, batch.y.view(-1).long())
            total_loss += loss.item()

            preds = output.argmax(dim=1)  # Get class predictions
            correct += (preds == batch.y.view(-1)).sum().item()
            total += batch.num_graphs

            # Store predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    print(reps, hidden_dim, heads, unit1, unit2, loader)
    # Print classification report
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, digits=4))

    return tn, fp, fn, tp 
resVal = []
for reps in range(1, 6):
    for hidden_dim in [32, 64]:
        for heads in [4, 8]:
            for numUnit1 in [32, 64, 128, 256, 512, 1024, 2048]:
                numUnit2 = numUnit1
                while numUnit2 >= 16:

                    train_data, test_data = split_dataset(dataset)

                    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
                    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

                    # Determine the input dimension from the first graph's node features
                    input_dim = train_data[0].x[0].size().numel()

                    # Initialize the model for classification and move it to the device
                    model = QuantumGAT(
                        in_feats=input_dim,
                        hidden_dim=hidden_dim,
                        heads=heads,
                        mlp_units=[numUnit1, numUnit2],
                        num_residual=2,
                        num_classes=num_classes
                    ).to(device)

                    # Train and Evaluate
                    train_model(model, train_loader, epochs=50, lr=1e-3)
                    evaluate_model(model, test_loader)

                    torch.save(model, f'GAT_2GCN_1FFNN/model_gnn_GAT_2GCN_1FFNN_{reps}_{hidden_dim}_{heads}_{numUnit1}_{numUnit2}.pth')

                    tn_train, fp_train, fn_train, tp_train  = getConfusion(model, train_loader, reps, hidden_dim, heads, numUnit1, numUnit2, loader='train')
                    tn_test, fp_test, fn_test, tp_test = getConfusion(model, test_loader, reps, hidden_dim, heads, numUnit1, numUnit2, loader='test')
                    res = {'model' : 'GAT_2GCN_1FFNN',
                           'hidden_dim' : hidden_dim,
                           'heads' : heads,
                           'numUnit1' : numUnit1,
                           'numUnit2' : numUnit2,
                           'reps' : reps,
                           'tn_train' : int(tn_train),
                           'fp_train' : int(fp_train),
                           'fn_train' : int(fn_train),
                           'tp_train' : int(tp_train),
                           'tn_test' : int(tn_test),
                           'fp_test' : int(fp_test),
                           'fn_test' : int(fn_test),
                           'tp_test' : int(tp_test),
                           }
                    resVal.append(res)

                    numUnit2 = numUnit2 // 2

with open("model_gnn_GAT_2GCN_1FFNN.json", "w") as json_file:
    json.dump(resVal, json_file)
