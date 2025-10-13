import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
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
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validate(args, model, loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    workflow_count_dict = {}
    ground_workflow_dict = {}
    predicted_workflow_dict = {}
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
            batch.workflow_id = batch.workflow_id.cpu().tolist()
            batch.task_id = batch.task_id.cpu().tolist()
            for i, workflow_id in enumerate(batch.workflow_id):
                if workflow_id not in workflow_count_dict.keys():
                    workflow_count_dict[workflow_id] = 1
                else:
                    workflow_count_dict[workflow_id] += 1
                if workflow_id not in ground_workflow_dict.keys():
                    ground_workflow_dict[workflow_id] = 0
                if workflow_id not in predicted_workflow_dict.keys():
                    predicted_workflow_dict[workflow_id] = 0
                ground_workflow_dict[workflow_id] += batch.y[i].cpu().item()
                predicted_workflow_dict[workflow_id] += preds[i].cpu().item()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct_predictions += (preds == batch.y).sum().item()
            total_predictions += batch.y.size(0)
    ground_workflow_score_dict = {
        k: ground_workflow_dict[k] / workflow_count_dict[k]
        for k in ground_workflow_dict.keys()
    }
    predicted_workflow_score_dict = {
        k: predicted_workflow_dict[k] / workflow_count_dict[k]
        for k in predicted_workflow_dict.keys()
    }
    utility = calculate_utility(
        args, ground_workflow_score_dict, predicted_workflow_score_dict
    )
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_predictions
    return accuracy, utility


def precision_at_k(ground_workflows, predicted_workflows, k):

    ground_sorted = sorted(ground_workflows.items(), key=lambda x: x[1], reverse=True)
    ground_top_k = set([workflow for workflow, score in ground_sorted[:k]])
    predicted_sorted = sorted(
        predicted_workflows.items(), key=lambda x: x[1], reverse=True
    )
    predicted_top_k = [workflow for workflow, score in predicted_sorted[:k]]
    relevant_count = sum(1 for workflow in predicted_top_k if workflow in ground_top_k)
    precision = relevant_count / k

    return precision


def calculate_utility(args, ground_dict, predicted_dict):
    ground_workflows = sorted(ground_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_workflows = sorted(
        predicted_dict.items(), key=lambda x: x[1], reverse=True
    )
    ground_workflows = {k: v for k, v in ground_workflows}
    predicted_workflows = {k: v for k, v in predicted_workflows}
    lst = []
    num_workflow = len(ground_dict)
    x = range(1, num_workflow + 1)
    perfect_overlap = [1.0] * len(x)
    perfect_auc = calculate_auc_precise(x, perfect_overlap)
    # for k in range(1, num_workflow+1):
    #     overlap_count = 0
    #     for workflow in list(predicted_workflows.keys())[:k]:
    #         if workflow in list(ground_workflows.keys())[:k]:
    #             overlap_count += 1
    #     lst.append(overlap_count / k)
    lst = [precision_at_k(ground_workflows, predicted_workflows, k) for k in x]
    utility = calculate_auc_precise(x, lst) / perfect_auc
    # import matplotlib.pyplot as plt
    # import os
    # save_path = os.path.join(args.data_path,f'Precision@K_{args.base_conv}_{args.arch}.png')
    # plt.ylabel("precision")
    # plt.xlabel("k")
    # plt.title("Precision@K")
    # plt.xticks(np.arange(1, num_workflow+1, 10))
    # plt.plot(x,lst,label=f'Predicted (AUC score: {utility:.2f})')
    # plt.legend()

    # plt.savefig(save_path)
    return utility


def calculate_auc_precise(x, y, num_points=100):
    interp_func = interp1d(x, y, kind="linear")
    x_interp = np.linspace(min(x), max(x), num_points)
    y_interp = interp_func(x_interp)
    auc = trapezoid(y_interp, x_interp)
    return auc


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
    branches = ["test"]
    test_loader = get_dataloader(args, branches)[0]

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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    best_model_dir = f"{args.data_path}/ckpt/{args.model_type}_{args.base_conv}_{args.n_layers}_{args.hidden_dim}_{args.encoding}_{args.label_ratio}_{args.dropout}"
    assert os.path.exists(best_model_dir)

    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_path = os.path.join(best_model_dir, "best_model.pth")

    model.load_state_dict(torch.load(best_model_path))

    test_acc, utility = validate(args, model, test_loader, device)
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Utility: {utility :.4f}")
    print("-" * 50)

    torch.cuda.empty_cache()
    return test_acc, utility
