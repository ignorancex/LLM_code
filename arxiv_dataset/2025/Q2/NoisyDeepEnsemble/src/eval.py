import os
import sys

import torch.distributed
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import copy

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from tqdm import tqdm

from models import resnet18, efficientnetb0, vgg16
from dataset import build_dataset

def get_dataloader(dataset, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    return loader

def negative_log_likelihood(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def calculate_ece(outputs, labels, bucket_size):
    n_samples = outputs.size(0)
    outputs = nn.Softmax(dim=-1)(outputs)
    preds_prob, preds = torch.max(outputs, 1)
    
    bin_boundaries = torch.linspace(0, 1, bucket_size + 1)
    ece = 0.0

    for i in range(bucket_size):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        
        bin_mask = (preds_prob > bin_lower) & (preds_prob <= bin_upper)
        bin_size = bin_mask.sum().item()

        if bin_size > 0:
            avg_conf_in_bin = preds_prob[bin_mask].mean()
            
            acc_in_bin = (preds[bin_mask] == labels[bin_mask]).float().mean()
            ece += bin_size / n_samples * torch.abs(acc_in_bin - avg_conf_in_bin)
    return ece    
    
    

def eval():
    print("Starting evaluation")
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--severity", default=-1, type=int)
    
    # Model
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--ckpt_dir", default="checkpoints", type=str)
    parser.add_argument("--ensemble_size", default=10, type=int)
    
    # Evaluation
    parser.add_argument("--train_acc", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--bucket_size", default=15, type=int)
    parser.add_argument("--calibration_single", action="store_true")
    
    # Additional
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    
    args = parser.parse_args()
    
    # Build dataset
    train_dataset = build_dataset(args, train=True) if args.train_acc else None
    test_dataset = build_dataset(args, train=False)
    
    # Build dataloaders
    train_loader = get_dataloader(train_dataset, args.batch_size, args.num_workers) if args.train_acc else None
    test_loader = get_dataloader(test_dataset, args.batch_size, args.num_workers)
    
    # Build model
    if args.model == "resnet18":
        model = resnet18(num_classes=test_dataset.num_classes)
    elif args.model == "vgg16":
        model = vgg16(num_classes=test_dataset.num_classes)
    elif args.model == "efficientnet_b0":
        model = efficientnetb0(num_classes=test_dataset.num_classes)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoints
    def load_checkpoint_ddp(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        # The model is pretrained on multiple GPUs, so we need to remove the "module" prefix
        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.items()})
        return model
    
    models = []
    for i in range(args.ensemble_size):
        dataset_name = args.dataset if args.severity == -1 else args.dataset.split("_")[0]
        model_ckpt = os.path.join(args.ckpt_dir, f"{args.model}_{dataset_name}_{i}.pth")
        assert os.path.exists(model_ckpt), f"Checkpoint {model_ckpt} not found"
        model = load_checkpoint_ddp(model, model_ckpt)
        model.eval()
        models.append(copy.deepcopy(model))
    
    # Evaluate
    if args.train_acc:
        assert args.severity == -1, "Train accuracy is not supported for corrupted dataset"
        print("Train accuracy evaluation")
        correct = 0
        total = 0
        for data in tqdm(train_loader):
            inputs, labels = data["image"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = []
            for model in models:
                model = model.to(device)
                with torch.no_grad():
                    outputs.append(model(inputs))
            outputs = torch.stack(outputs)  # [ensemble_size, batch_size, num_classes]
            outputs = torch.mean(outputs, dim=0)  # [batch_size, num_classes]
            
            _, predicted = torch.max(outputs, 1)  # [batch_size]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Train accuracy: {train_acc}")
    
    correct = 0
    total = 0
    for data in tqdm(test_loader):
        inputs, labels = data["image"], data["label"]
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = []
        for model in models:
            model = model.to(device)
            with torch.no_grad():
                outputs.append(model(inputs))
        outputs = torch.stack(outputs)  # [ensemble_size, batch_size, num_classes]
        
        outputs = torch.mean(outputs, dim=0)  # [batch_size, num_classes]
        _, predicted = torch.max(outputs, 1)  # [batch_size]        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"Test accuracy: {test_acc}")
    
    # Single model evaluation
    # test only
    if args.single:
        print("Single model evaluation")
        single_accs = []
        for i in range(args.ensemble_size):
            model = models[i]
            
            correct = 0
            total = 0
            for data in tqdm(test_loader):
                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            print(f"Model {i} test accuracy: {acc}")
            single_accs.append(acc)
        print(f"Mean test accuracy: {sum(single_accs) / len(single_accs)}")
    
    # Oracle evaluation
    # Oracle accuracy means the accuracy of the ensemble model if well-predicted model is selected at each sample.
    if args.oracle:
        print("Oracle evaluation")
        correct = 0
        total = 0
        
        for data in tqdm(test_loader):
            inputs, labels = data["image"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = []
            for model in models:
                model = model.to(device)
                with torch.no_grad():
                    outputs.append(model(inputs))
            outputs = torch.stack(outputs)  # (ensemble_size, batch_size, num_classes)
            preds = torch.argmax(outputs, dim=-1)  # (ensemble_size, batch_size)
            preds = preds.t()  # (batch_size, ensemble_size)
            for i, pred in enumerate(preds):
                correct_label = labels[i]
                for j in range(len(pred)):
                    if pred[j] == correct_label:
                        correct += 1
                        break
            total += labels.size(0)
           
        print(f"Oracle accuracy: {100 * correct / total}") 
        
    if args.calibration:
        # Calibration evaluation
        print("Calibration evaluation")
        # Calculate the nll
        print("Calculating NLL")
        def compute_nll(model_list, data_loader, device):
            output_list = []
            label_list = []
            for data in tqdm(data_loader):
                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = []
                for model in model_list:
                    model = model.to(device)
                    with torch.no_grad():
                        outputs.append(model(inputs))
                outputs = torch.stack(outputs)
                outputs = torch.mean(outputs, dim=0)
                output_list.extend(outputs)
                label_list.extend(labels)
            outputs = torch.stack(output_list)
            labels = torch.stack(label_list)
            nll = negative_log_likelihood(outputs, labels).item()
            return nll
        
        def compute_ece(model_list, data_loader, device, bucket_size):
            output_list = []
            label_list = []
            for data in tqdm(data_loader):
                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = []
                for model in model_list:
                    model = model.to(device)
                    with torch.no_grad():
                        outputs.append(model(inputs))
                outputs = torch.stack(outputs)
                outputs = torch.mean(outputs, dim=0)
                output_list.extend(outputs)
                label_list.extend(labels)
            outputs = torch.stack(output_list)
            labels = torch.stack(label_list)
            ece = calculate_ece(outputs, labels, bucket_size)
            return ece
        
        def compute_nll_single(model, data_loader, device):
            output_list = []
            label_list = []
            for data in tqdm(data_loader):
                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                output_list.extend(outputs)
                label_list.extend(labels)
            outputs = torch.stack(output_list)
            labels = torch.stack(label_list)
            nll = negative_log_likelihood(outputs, labels).item()
            return nll
        
        def compute_ece_single(model, data_loader, device, bucket_size):
            output_list = []
            label_list = []
            for data in tqdm(data_loader):
                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                output_list.extend(outputs)
                label_list.extend(labels)
            outputs = torch.stack(output_list)
            labels = torch.stack(label_list)
            ece = calculate_ece(outputs, labels, bucket_size)
            return ece
        
        nll = compute_nll(models, test_loader, device)
        print(f"NLL: {nll}")
        
        # Calculate the ece
        print("Calculating ECE")
        ece = compute_ece(models, test_loader, device, args.bucket_size)
        print(f"ECE: {ece}")
    
    if args.calibration_single:
        print("Single model calibration evaluation")
        nlls = []
        eces = []
        for i in range(args.ensemble_size):
            model = models[i]
            nll = compute_nll_single(model, test_loader, device)
            ece = compute_ece_single(model, test_loader, device, args.bucket_size)
            nlls.append(nll)
            eces.append(ece)
            print(f"Model {i} NLL: {nll}, ECE: {ece}")
        print(f"Mean NLL: {sum(nlls) / len(nlls)}, Mean ECE: {sum(eces) / len(eces)}")
        
    print("Evaluation done")
    
if __name__ == "__main__":
    eval()