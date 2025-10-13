import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import sys, os
from .mtl_trainer import MTLTrainer
sys.path.insert(0, os.path.abspath('..'))
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch.nn import functional as F


class MTLTrainerWithUncertainty(MTLTrainer):
    def __init__(self, model, train_loader, val_loader, optimizer, project_name="Kendal_MTL_Training"):
        super().__init__(model, train_loader, optimizer, None, project_name)
        self.val_loader = val_loader
        self.optimizer = optimizer  # Ensure optimizer is correctly set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        wandb.init(project=project_name)

    def compute_metrics(self, predictions, labels, task_name):
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro")
        micro_f1 = f1_score(labels, predictions, average="micro")
        weighted_f1 = f1_score(labels, predictions, average="weighted")

        return {
            f"{task_name}/accuracy": accuracy,
            f"{task_name}/macro_f1": macro_f1,
            f"{task_name}/micro_f1": micro_f1,
            f"{task_name}/weighted_f1": weighted_f1,
        }

    def train(self, num_epochs=10, model_path="model.pth"):
        for epoch in range(num_epochs):
            self.model.train()
            train_running_loss = 0.0
            train_metrics = {task: {"predictions": [], "labels": []} for task in self.model.task_outputs.keys()}

            for imgs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs, uncertainties = self.model(imgs)
                losses = []

                # Compute dynamically weighted loss for each task
                for i, task in enumerate(self.model.task_outputs.keys()):
                    task_output = outputs[task]
                    task_label = labels[:, i]
                    task_loss = F.cross_entropy(task_output, task_label)

                    # Weight loss by uncertainty
                    loss_weight = torch.exp(-2 * uncertainties[task])
                    weighted_loss = loss_weight * task_loss + uncertainties[task]
                    losses.append(weighted_loss)

                    # Collect predictions and labels for metrics
                    task_predictions = torch.argmax(task_output, dim=1).cpu().numpy()
                    train_metrics[task]["predictions"].extend(task_predictions)
                    train_metrics[task]["labels"].extend(task_label.cpu().numpy())

                # Backward pass
                total_loss = sum(losses)
                total_loss.backward()
                self.optimizer.step()

                train_running_loss += total_loss.item()

            train_loss = train_running_loss / len(self.train_loader)
            wandb.log({"train_loss": train_loss, "epoch": epoch + 1})

            # Compute and log metrics for training
            for task, data in train_metrics.items():
                metrics = self.compute_metrics(data["predictions"], data["labels"], f"train/{task}")
                wandb.log(metrics)

            print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")
            for task, data in train_metrics.items():
                print(f"Training Metrics for Task {task}: {self.compute_metrics(data['predictions'], data['labels'], f'train/{task}')}")

            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_metrics = {task: {"predictions": [], "labels": []} for task in self.model.task_outputs.keys()}

            with torch.no_grad():
                for imgs, labels in self.val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)

                    outputs, uncertainties = self.model(imgs)
                    losses = []

                    for i, task in enumerate(self.model.task_outputs.keys()):
                        task_output = outputs[task]
                        task_label = labels[:, i]
                        task_loss = F.cross_entropy(task_output, task_label)

                        loss_weight = torch.exp(-2 * uncertainties[task])
                        weighted_loss = loss_weight * task_loss + uncertainties[task]
                        losses.append(weighted_loss)

                        task_predictions = torch.argmax(task_output, dim=1).cpu().numpy()
                        val_metrics[task]["predictions"].extend(task_predictions)
                        val_metrics[task]["labels"].extend(task_label.cpu().numpy())

                    total_loss = sum(losses)
                    val_running_loss += total_loss.item()

            val_loss = val_running_loss / len(self.val_loader)
            wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

            # Compute and log metrics for validation
            for task, data in val_metrics.items():
                metrics = self.compute_metrics(data["predictions"], data["labels"], f"val/{task}")
                wandb.log(metrics)

            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
            for task, data in val_metrics.items():
                print(f"Validation Metrics for Task {task}: {self.compute_metrics(data['predictions'], data['labels'], f'val/{task}')}")

        # Save the model
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
