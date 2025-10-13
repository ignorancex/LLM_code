import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import sys, os
from .mtl_trainer import MTLTrainer
sys.path.insert(0, os.path.abspath('..'))
from sklearn.metrics import f1_score
import numpy as np


class MTLTrainerWithCrossStitch(MTLTrainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, project_name="MTL_Training"):
        super().__init__(model, train_loader, optimizer, criterion, project_name)
        self.val_loader = val_loader
        self.criterion = criterion  # Ensure this is passed correctly as a loss function
        self.optimizer = optimizer  # Ensure optimizer is correctly set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train(self, num_epochs=10, model_path="model.pth"):
        wandb.init(project=self.project_name)

        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()
            running_loss = 0.0
            task_losses = {task: 0.0 for task in self.model.task_names}
            task_correct = {task: 0 for task in self.model.task_names}
            task_total = {task: 0 for task in self.model.task_names}
            task_predictions = {task: [] for task in self.model.task_names}
            task_true_labels = {task: [] for task in self.model.task_names}

            for imgs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)

                losses = []
                for i, (output, task) in enumerate(zip(outputs, self.model.task_names)):
                    task_loss = self.criterion(output, labels[:, i])
                    losses.append(task_loss)
                    task_losses[task] += task_loss.item()

                    # Predictions and metrics
                    _, predicted = torch.max(output, 1)
                    task_predictions[task].extend(predicted.cpu().tolist())
                    task_true_labels[task].extend(labels[:, i].cpu().tolist())
                    correct = (predicted == labels[:, i]).sum().item()
                    task_correct[task] += correct
                    task_total[task] += labels.size(0)

                total_loss = sum(losses)
                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            train_accuracies = {task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0 for task in self.model.task_names}

            # Log Training Metrics
            wandb_log = {"epoch": epoch + 1, "Train Loss": avg_train_loss}
            for task in self.model.task_names:
                avg_task_loss = task_losses[task] / len(self.train_loader)
                train_accuracy = train_accuracies[task]

                wandb_log[f"Train {task} Loss"] = avg_task_loss
                wandb_log[f"Train {task} Accuracy"] = train_accuracy

                # F1-Scores
                f1_macro = f1_score(task_true_labels[task], task_predictions[task], average='macro')
                f1_micro = f1_score(task_true_labels[task], task_predictions[task], average='micro')
                f1_weighted = f1_score(task_true_labels[task], task_predictions[task], average='weighted')

                wandb_log[f"Train {task} F1 Macro"] = f1_macro
                wandb_log[f"Train {task} F1 Micro"] = f1_micro
                wandb_log[f"Train {task} F1 Weighted"] = f1_weighted

                print(f"Train - Task: {task}, Loss: {avg_task_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                      f"F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}, F1 Weighted: {f1_weighted:.4f}")

            print(f"Epoch {epoch + 1}: Total Train Loss = {avg_train_loss:.4f}")

            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            val_task_losses = {task: 0.0 for task in self.model.task_names}
            val_task_correct = {task: 0 for task in self.model.task_names}
            val_task_total = {task: 0 for task in self.model.task_names}
            val_task_predictions = {task: [] for task in self.model.task_names}
            val_task_true_labels = {task: [] for task in self.model.task_names}

            with torch.no_grad():
                for imgs, labels in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = self.model(imgs)

                    losses = []
                    for i, (output, task) in enumerate(zip(outputs, self.model.task_names)):
                        task_loss = self.criterion(output, labels[:, i])
                        losses.append(task_loss)
                        val_task_losses[task] += task_loss.item()

                        # Predictions and metrics
                        _, predicted = torch.max(output, 1)
                        val_task_predictions[task].extend(predicted.cpu().tolist())
                        val_task_true_labels[task].extend(labels[:, i].cpu().tolist())
                        correct = (predicted == labels[:, i]).sum().item()
                        val_task_correct[task] += correct
                        val_task_total[task] += labels.size(0)

                    val_loss += sum(losses).item()

            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracies = {task: val_task_correct[task] / val_task_total[task] if val_task_total[task] > 0 else 0.0 for task in self.model.task_names}

            # Log Validation Metrics
            wandb_log["Validation Loss"] = avg_val_loss
            for task in self.model.task_names:
                avg_val_task_loss = val_task_losses[task] / len(self.val_loader)
                val_accuracy = val_accuracies[task]

                wandb_log[f"Validation {task} Loss"] = avg_val_task_loss
                wandb_log[f"Validation {task} Accuracy"] = val_accuracy

                # F1-Scores
                f1_macro = f1_score(val_task_true_labels[task], val_task_predictions[task], average='macro')
                f1_micro = f1_score(val_task_true_labels[task], val_task_predictions[task], average='micro')
                f1_weighted = f1_score(val_task_true_labels[task], val_task_predictions[task], average='weighted')

                wandb_log[f"Validation {task} F1 Macro"] = f1_macro
                wandb_log[f"Validation {task} F1 Micro"] = f1_micro
                wandb_log[f"Validation {task} F1 Weighted"] = f1_weighted

                print(f"Validation - Task: {task}, Loss: {avg_val_task_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
                      f"F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}, F1 Weighted: {f1_weighted:.4f}")

            print(f"Epoch {epoch + 1}: Total Validation Loss = {avg_val_loss:.4f}")

            # Log overall metrics for both training and validation
            wandb.log(wandb_log)

            # Log cross-stitch weights
            for i, cs_unit in enumerate(self.model.cross_stitch_units):
                wandb.log({f"CrossStitch_Unit_{i}_Weights": cs_unit.alpha.detach().cpu().numpy()})

        # Save the trained model
        torch.save(self.model.state_dict(), model_path)
        #wandb.save(model_path)
        print(f"Model saved successfully at {model_path}.")
