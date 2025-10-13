import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    GATConv,
    GATv2Conv,
    TransformerConv,
    GINEConv,
)
from agent_predictor.convert_dataset_mvgnn import get_dataloader
from agent_predictor.predictor import AgentPredictor
import numpy as np
import random
from itertools import chain
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, loader, optimizer, device, args):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    i = 0

    if isinstance(loader, list):
        num_batches = len(loader[0]) + len(loader[1])
        loader = chain(*loader)
    else:
        num_batches = len(loader)

    for batch in loader:
        model.train()
        batch = batch.to(device)
        optimizer.zero_grad()
        num_graphs = batch.batch[-1] + 1
        batch.task_embedding = batch.task_embedding.reshape(num_graphs, -1)
        batch.instruction_text = batch.instruction_text.reshape(num_graphs, -1)
        batch.workflow_code = batch.workflow_code.reshape(num_graphs, -1)
        score = model(batch, args)
        loss = F.binary_cross_entropy(score, batch.y.to(torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = (score > 0.5).float()
        correct_predictions += (preds == batch.y).sum().item()
        total_predictions += batch.y.size(0)
        batch_acc = accuracy_score(batch.y.cpu().numpy(), preds.cpu().numpy())

        i += 1
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_graphs = batch.batch[-1] + 1
            batch.task_embedding = batch.task_embedding.reshape(num_graphs, -1)
            batch.instruction_text = batch.instruction_text.reshape(num_graphs, -1)
            batch.workflow_code = batch.workflow_code.reshape(num_graphs, -1)
            score = model(batch, args)
            loss = F.binary_cross_entropy(score, batch.y.float())
            total_loss += loss.item()
            preds = (score > 0.5).float()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct_predictions += (preds == batch.y).sum().item()
            total_predictions += batch.y.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_predictions
    model.train()
    return avg_loss, accuracy


def get_base_conv(base_conv):
    base_conv_dict = {
        "GCNConv": GCNConv,
        "GINEConv": GINEConv,
        "GATConv": GATConv,
        "GATv2Conv": GATv2Conv,
        "TransformerConv": TransformerConv,
    }
    return base_conv_dict[base_conv]


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using input dim: {args.input_dim}")

    args.domain = os.path.basename(args.data_path)

    branches = ["train", "val"]
    train_loader, val_loader = get_dataloader(args, branches)

    print(args.base_conv)
    base_conv = get_base_conv(args.base_conv)
    model = AgentPredictor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_mlplayers=args.n_mlplayers,
        dropout=args.dropout,
        base_conv=base_conv,
        views=args.encoding
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0

    best_model_dir = f"{args.data_path}/ckpt/{args.model_type}_{args.base_conv}_{args.n_layers}_{args.hidden_dim}_{args.encoding}_{args.label_ratio}_{args.dropout}"
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_path = os.path.join(best_model_dir, "best_model.pth")
    for epoch in tqdm(range(1, args.epochs + 1), leave=False, desc="Epochs"):
        train_loss, train_acc = train(model, train_loader, optimizer, device, args)
        val_loss, val_acc = validate(model, val_loader, device, args)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Val Acc: {val_acc:.4f} at epoch {epoch}\r")

    torch.cuda.empty_cache()
    return model
