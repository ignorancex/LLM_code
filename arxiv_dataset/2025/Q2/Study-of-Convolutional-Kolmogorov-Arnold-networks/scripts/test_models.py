import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchmetrics

from convkan import ConvKAN, LayerNorm2D
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
import pandas as pd
import os
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis
# from lenetfastkan import LeNet5_KAN
from alexnetkan import AlexNetKAN
# from lenetcnn import LeNet5
import time

device = "cuda:0"

def load_data():
    # Using the same transform as before for consistency
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_ds = datasets.ImageNet(
        root="./imagenet/val/", 
        split='val', 
        transform=transform,
    )
    
    print(f"testing validation size: {len(val_ds)}")
    return val_ds

def load_model(model_type, snapshot_path=None):
    if model_type == "alexnetkan":
        model = AlexNetKAN()
        if snapshot_path:
            snapshot = torch.load(f"./model_snapshots/{snapshot_path}", 
                              map_location=device)
            new_model_state = {}
            model_state = snapshot["MODEL_STATE"]
            for key, val in model_state.items():
                if key.startswith("module."):
                    new_model_state[key[7:]] = val 
                else:
                    new_model_state[key] = val 
            model.load_state_dict(new_model_state)
            return model, snapshot
    
    elif model_type == "alexnet_pytorch":
        model = models.alexnet(weights=False)
        model.load_state_dict(torch.load(f"./model_snapshots/alexnet_pretrained.pth"))
        return model
    
    model.eval()
    model.to(device)
    return model

def prepare_data(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

# Your existing metrics functions remain the same
def top_k_metrics(output, target, num_classes, k=5):
    top_k_acc = torchmetrics.Accuracy(top_k=k, num_classes=num_classes, task='multiclass')
    precision = torchmetrics.Precision(num_classes=num_classes, average='macro', top_k=k, task='multiclass')
    recall = torchmetrics.Recall(num_classes=num_classes, average='macro', top_k=k, task='multiclass')
    f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro', top_k=k, task='multiclass')
    
    # Compute metrics
    top_k_acc_value = top_k_acc(output, target)
    precision_value = precision(output, target)
    recall_value = recall(output, target)
    f1_value = f1(output, target)

    out = {
        f"top_{k}_acc": top_k_acc_value.item(),
        f"top_{k}_precision": precision_value.item(), 
        f"top_{k}_recall": recall_value.item(),
        f"top_{k}_f1": f1_value.item()
    }

    return out 

def concatenated_classification_report(output, target,name, num_classes=1000, k=5, top_k = False):

    pred_classes = torch.argmax(output, dim=1).cpu().numpy()

    classification_report_sklearn = classification_report(
        target.cpu().numpy(),
        pred_classes,
        output_dict=True
    )
    classification_report_df = pd.DataFrame(classification_report_sklearn).transpose()

    if top_k:
        top_k_metrics_ = top_k_metrics(output, target, num_classes, k)
        top_k_metrics_df = pd.DataFrame([top_k_metrics_])

        classification_report_df = pd.concat([classification_report_df, top_k_metrics_df], axis=0, ignore_index=True)

    classification_report_df.to_csv(f"{name} report.csv", index=False)

    print("report saved")

def load_history_and_plot_graph(hist, epochs, name= "test"):
    valid_loss = []
    valid_acc = []
    train_loss = []
    for i in range(len(hist)):
        valid_loss.append(hist[i]['valid_loss'])
        valid_acc.append(hist[i]['valid_acc'])
        train_loss.append(hist[i]['train_loss'])

    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(epochs), valid_loss, label = "validation loss")
    plt.plot(np.arange(epochs), train_loss, label = "training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training vs Validation Loss {name}")
    plt.savefig(f"./graphs/loss {name}.png", dpi=300, bbox_inches="tight")
    print("printed")

    plt.figure()
    plt.plot(np.arange(epochs), valid_acc, label = "validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # plt.legend()
    plt.title(f"Top-1 Validation Accuracy over Epochs {name}")
    plt.savefig(f"./graphs/validation {name}.png", dpi=300, bbox_inches="tight")
    print("printed")

@torch.inference_mode()
def calculate_output(dataset, model):
    batch_size = 32
    dl = prepare_data(dataset, batch_size)

    print("computing outputs")
    all_y_preds = []
    all_y_inputs = []
    for x, y in dl:
        y_pred = model(x.to(device)).cpu().numpy()
        all_y_preds.append(y_pred)
        all_y_inputs.append(y)

    all_outputs = np.concatenate(all_y_preds, axis=0)
    all_targets = np.concatenate(all_y_inputs, axis=0)

    return torch.from_numpy(all_outputs), torch.from_numpy(all_targets)

def compare_models():
    models_to_compare = {
        "AlexNet KAN": ("alexnetkan", "alexnetkan"),  # (model_type, snapshot_path)
        "PyTorch AlexNet": ("alexnet_pytorch", None)  # Pretrained model doesn't need snapshot
    }

    dataset = load_data()
    
    for name, (model_type, snapshot_path) in models_to_compare.items():
        print(f"\nEvaluating {name}")
        
        # Load model
        if snapshot_path:
            model, snapshot = load_model(model_type, snapshot_path)
        else:
            model = load_model(model_type)
            snapshot = None
        
        model.eval()
        model.to(device)

        # Calculate FLOPs
        dl = prepare_data(dataset, 1)
        sample = next(iter(dl))[0].to(device)
        flops = FlopCountAnalysis(model, sample)
        print(f"FLOPs: {flops.total():,}")

        # Measure inference time
        start = time.time()
        _ = model(sample)
        end = time.time()
        print(f"Inference time: {end-start:.4f}s")

        # Print model summary
        print(summary(model, (3, 224, 224), device="cuda"))

        # Calculate metrics
        print(f"Computing metrics for {len(dataset)} validation samples")
        with torch.no_grad():
            all_outputs, all_targets = calculate_output(dataset, model)
        
        concatenated_classification_report(all_outputs, all_targets, name, top_k=True)
        
        # Plot training history if available
        if snapshot and "HIST" in snapshot:
            hist = snapshot["HIST"]
            epochs = snapshot["EPOCHS_RUN"]
            # load_history_and_plot_graph(hist, epochs, name)

        print("*" * 100)

if __name__ == "__main__":
    compare_models()
