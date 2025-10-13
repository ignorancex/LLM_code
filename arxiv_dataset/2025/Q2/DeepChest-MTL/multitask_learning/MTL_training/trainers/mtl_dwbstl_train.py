# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import wandb
# import sys, os
# from .mtl_trainer import MTLTrainer
# sys.path.insert(0, os.path.abspath('..'))
# from weighting.dwbstl import DynamicLossWeight
# from sklearn.metrics import f1_score
# import numpy as np

# class MTLTrainerWithDWBSTL(MTLTrainer):
#     def __init__(self, model, train_loader, val_loader, optimizer, criterion, initial_weights, project_name="MTL_Training"):
#         super().__init__(model, train_loader, optimizer, criterion, project_name)
#         self.val_loader = val_loader
#         self.dynamic_weights = DynamicLossWeight(initial_weights)
#         self.optimizer = optimizer
#         self.criterion = nn.CrossEntropyLoss()


#     def train(self, num_epochs=10, model_path="fashion_classifier.pth"):
#         wandb.init(project=self.project_name, config={
#             "num_epochs": num_epochs,
#             "initial_weights": self.dynamic_weights.loss_weights,
#             "learning_rate": self.optimizer.param_groups[0]['lr']  # Make sure this is from the optimizer
#         })

#         for epoch in range(num_epochs):
#             # Training phase
#             self.model.train()
#             running_loss = 0.0
#             task_losses = {task: 0.0 for task in self.model.task_names}
#             task_correct = {task: 0 for task in self.model.task_names}
#             task_total = {task: 0 for task in self.model.task_names}
#             task_predictions = {task: [] for task in self.model.task_names}
#             task_true_labels = {task: [] for task in self.model.task_names}

#             for imgs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
#                 self.optimizer.zero_grad()
#                 outputs = self.model(imgs)
#                 losses = []

#                 for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
#                     task_loss = self.dynamic_weights.loss_weights[task] * self.criterion(out, labels[:, i])
#                     losses.append(task_loss)
#                     task_losses[task] += task_loss.item()

#                     _, predicted = torch.max(out, 1)
#                     task_predictions[task].extend(predicted.cpu().tolist())
#                     task_true_labels[task].extend(labels[:, i].cpu().tolist())

#                     correct = (predicted == labels[:, i]).sum().item()
#                     task_correct[task] += correct
#                     task_total[task] += labels.size(0)

#                 loss = sum(losses)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             avg_train_loss = running_loss / len(self.train_loader)
#             train_accuracies = {task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0 for task in self.model.task_names}
            
#             wandb_log = {"epoch": epoch + 1, "Train Loss": avg_train_loss}
#             for task in self.model.task_names:
#                 avg_task_loss = task_losses[task] / len(self.train_loader)
#                 train_accuracy = train_accuracies[task]
#                 wandb_log[f"Train {task} Loss"] = avg_task_loss
#                 wandb_log[f"Train {task} Accuracy"] = train_accuracy
#                 wandb_log[f"{task} Weight"] = self.dynamic_weights.loss_weights[task]
#                 print(f"Train - Task: {task}, Loss: {avg_task_loss:.4f}, Accuracy: {train_accuracy:.4f}")

#                 # Compute F1-scores
#                 f1_macro = f1_score(task_true_labels[task], task_predictions[task], average='macro')
#                 f1_micro = f1_score(task_true_labels[task], task_predictions[task], average='micro')
#                 f1_weighted = f1_score(task_true_labels[task], task_predictions[task], average='weighted')
#                 wandb_log[f"Train {task} Macro F1"] = f1_macro
#                 wandb_log[f"Train {task} Micro F1"] = f1_micro
#                 wandb_log[f"Train {task} Weighted F1"] = f1_weighted
#                 print(f"Train - Task: {task}, Macro F1: {f1_macro:.4f}, Micro F1: {f1_micro:.4f}, Weighted F1: {f1_weighted:.4f}")

#             # Validation phase
#             self.model.eval()
#             val_loss = 0.0
#             val_task_losses = {task: 0.0 for task in self.model.task_names}
#             val_task_correct = {task: 0 for task in self.model.task_names}
#             val_task_total = {task: 0 for task in self.model.task_names}
#             val_task_predictions = {task: [] for task in self.model.task_names}
#             val_task_true_labels = {task: [] for task in self.model.task_names}

#             with torch.no_grad():
#                 for imgs, labels in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
#                     outputs = self.model(imgs)
#                     val_losses = []

#                     for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
#                         task_loss = self.criterion(out, labels[:, i])
#                         val_losses.append(task_loss)
#                         val_task_losses[task] += task_loss.item()

#                         _, predicted = torch.max(out, 1)
#                         val_task_predictions[task].extend(predicted.cpu().tolist())
#                         val_task_true_labels[task].extend(labels[:, i].cpu().tolist())

#                         correct = (predicted == labels[:, i]).sum().item()
#                         val_task_correct[task] += correct
#                         val_task_total[task] += labels.size(0)

#                     val_loss += sum(val_losses).item()

#             avg_val_loss = val_loss / len(self.val_loader)
#             val_accuracies = {task: val_task_correct[task] / val_task_total[task] if val_task_total[task] > 0 else 0.0 for task in self.model.task_names}

#             wandb_log["Validation Loss"] = avg_val_loss
#             for task in self.model.task_names:
#                 avg_val_task_loss = val_task_losses[task] / len(self.val_loader)
#                 val_accuracy = val_accuracies[task]
#                 wandb_log[f"Validation {task} Loss"] = avg_val_task_loss
#                 wandb_log[f"Validation {task} Accuracy"] = val_accuracy

#                 # Compute F1-scores
#                 f1_macro = f1_score(val_task_true_labels[task], val_task_predictions[task], average='macro')
#                 f1_micro = f1_score(val_task_true_labels[task], val_task_predictions[task], average='micro')
#                 f1_weighted = f1_score(val_task_true_labels[task], val_task_predictions[task], average='weighted')
#                 wandb_log[f"Validation {task} Macro F1"] = f1_macro
#                 wandb_log[f"Validation {task} Micro F1"] = f1_micro
#                 wandb_log[f"Validation {task} Weighted F1"] = f1_weighted
#                 print(f"Validation - Task: {task}, Macro F1: {f1_macro:.4f}, Micro F1: {f1_micro:.4f}, Weighted F1: {f1_weighted:.4f}")

#             # Log training and validation metrics to wandb
#             wandb.log(wandb_log)
#             self.dynamic_weights.update_weights(train_accuracies)

#         torch.save(self.model.state_dict(), model_path)
#         wandb.save(model_path)
#         print(f"Model saved successfully at {model_path}.")
#         wandb.finish()

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import sys, os
from .mtl_trainer import MTLTrainer
sys.path.insert(0, os.path.abspath('..'))
from weighting.dwbstl import DynamicLossWeight
from sklearn.metrics import f1_score
import numpy as np

class MTLTrainerWithDWBSTL(MTLTrainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, initial_weights, project_name="MTL_Training"):
        super().__init__(model, train_loader, optimizer, criterion, project_name)
        self.val_loader = val_loader
        self.dynamic_weights = DynamicLossWeight(initial_weights)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train(self, num_epochs=10, model_path="fashion_classifier.pth"):
        wandb.init(project=self.project_name, config={
            "num_epochs": num_epochs,
            "initial_weights": self.dynamic_weights.loss_weights,
            "learning_rate": self.optimizer.param_groups[0]['lr']  # Make sure this is from the optimizer
        })

        for epoch in range(num_epochs):
            # Training phase
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

                for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
                    task_loss = self.dynamic_weights.loss_weights[task] * self.criterion(out, labels[:, i])
                    losses.append(task_loss)
                    task_losses[task] += task_loss.item()

                    _, predicted = torch.max(out, 1)
                    task_predictions[task].extend(predicted.cpu().tolist())
                    task_true_labels[task].extend(labels[:, i].cpu().tolist())

                    correct = (predicted == labels[:, i]).sum().item()
                    task_correct[task] += correct
                    task_total[task] += labels.size(0)

                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            train_accuracies = {task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0 for task in self.model.task_names}
            
            wandb_log = {"epoch": epoch + 1, "Train Loss": avg_train_loss}
            for task in self.model.task_names:
                avg_task_loss = task_losses[task] / len(self.train_loader)
                train_accuracy = train_accuracies[task]
                wandb_log[f"Train {task} Loss"] = avg_task_loss
                wandb_log[f"Train {task} Accuracy"] = train_accuracy
                wandb_log[f"{task} Weight"] = self.dynamic_weights.loss_weights[task]
                print(f"Train - Task: {task}, Loss: {avg_task_loss:.4f}, Accuracy: {train_accuracy:.4f}")

                # Compute F1-scores
                f1_macro = f1_score(task_true_labels[task], task_predictions[task], average='macro')
                f1_micro = f1_score(task_true_labels[task], task_predictions[task], average='micro')
                f1_weighted = f1_score(task_true_labels[task], task_predictions[task], average='weighted')
                wandb_log[f"Train {task} Macro F1"] = f1_macro
                wandb_log[f"Train {task} Micro F1"] = f1_micro
                wandb_log[f"Train {task} Weighted F1"] = f1_weighted
                print(f"Train - Task: {task}, Macro F1: {f1_macro:.4f}, Micro F1: {f1_micro:.4f}, Weighted F1: {f1_weighted:.4f}")

            # Validation phase
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
                    val_losses = []

                    for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
                        task_loss = self.criterion(out, labels[:, i])
                        val_losses.append(task_loss)
                        val_task_losses[task] += task_loss.item()

                        _, predicted = torch.max(out, 1)
                        val_task_predictions[task].extend(predicted.cpu().tolist())
                        val_task_true_labels[task].extend(labels[:, i].cpu().tolist())

                        correct = (predicted == labels[:, i]).sum().item()
                        val_task_correct[task] += correct
                        val_task_total[task] += labels.size(0)

                    val_loss += sum(val_losses).item()

            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracies = {task: val_task_correct[task] / val_task_total[task] if val_task_total[task] > 0 else 0.0 for task in self.model.task_names}

            wandb_log["Validation Loss"] = avg_val_loss
            for task in self.model.task_names:
                avg_val_task_loss = val_task_losses[task] / len(self.val_loader)
                val_accuracy = val_accuracies[task]
                wandb_log[f"Validation {task} Loss"] = avg_val_task_loss
                wandb_log[f"Validation {task} Accuracy"] = val_accuracy

                # Compute F1-scores
                f1_macro = f1_score(val_task_true_labels[task], val_task_predictions[task], average='macro')
                f1_micro = f1_score(val_task_true_labels[task], val_task_predictions[task], average='micro')
                f1_weighted = f1_score(val_task_true_labels[task], val_task_predictions[task], average='weighted')
                wandb_log[f"Validation {task} Macro F1"] = f1_macro
                wandb_log[f"Validation {task} Micro F1"] = f1_micro
                wandb_log[f"Validation {task} Weighted F1"] = f1_weighted
                print(f"Validation - Task: {task}, Macro F1: {f1_macro:.4f}, Micro F1: {f1_micro:.4f}, Weighted F1: {f1_weighted:.4f}")

            # Log training and validation metrics to wandb
            wandb.log(wandb_log)
            self.dynamic_weights.update_weights(train_accuracies)

        torch.save(self.model.state_dict(), model_path)
        wandb.save(model_path)
        print(f"Model saved successfully at {model_path}.")
        wandb.finish()

# class MTLTrainerWithDWBSTL(MTLTrainer):
#     def __init__(self, model, train_loader, val_loader, optimizer, criterion, initial_weights, project_name="MTL_Training", device=None):
#         super().__init__(model, train_loader, optimizer, criterion, project_name)
#         self.val_loader = val_loader
#         self.dynamic_weights = DynamicLossWeight(initial_weights)
#         self.optimizer = optimizer
#         self.criterion = nn.CrossEntropyLoss()
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(self.device)  # Move model to device
#         print(f"Training is running on {self.device}")

#     def train(self, num_epochs=10, model_path="fashion_classifier.pth"):
#         wandb.init(project=self.project_name, config={
#             "num_epochs": num_epochs,
#             "initial_weights": self.dynamic_weights.loss_weights,
#             "learning_rate": self.optimizer.param_groups[0]['lr']  # Make sure this is from the optimizer
#         })

#         for epoch in range(num_epochs):
#             # Training phase
#             self.model.train()
#             running_loss = 0.0
#             task_losses = {task: 0.0 for task in self.model.task_names}
#             task_correct = {task: 0 for task in self.model.task_names}
#             task_total = {task: 0 for task in self.model.task_names}
#             task_predictions = {task: [] for task in self.model.task_names}
#             task_true_labels = {task: [] for task in self.model.task_names}

#             for imgs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)  # Move inputs and labels to device
#                 self.optimizer.zero_grad()
#                 outputs = self.model(imgs)
#                 losses = []

#                 for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
#                     task_loss = self.dynamic_weights.loss_weights[task] * self.criterion(out, labels[:, i])
#                     losses.append(task_loss)
#                     task_losses[task] += task_loss.item()

#                     _, predicted = torch.max(out, 1)
#                     task_predictions[task].extend(predicted.cpu().tolist())
#                     task_true_labels[task].extend(labels[:, i].cpu().tolist())

#                     correct = (predicted == labels[:, i]).sum().item()
#                     task_correct[task] += correct
#                     task_total[task] += labels.size(0)

#                 loss = sum(losses)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             avg_train_loss = running_loss / len(self.train_loader)
#             train_accuracies = {task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0 for task in self.model.task_names}
            
#             wandb_log = {"epoch": epoch + 1, "Train Loss": avg_train_loss}
#             for task in self.model.task_names:
#                 avg_task_loss = task_losses[task] / len(self.train_loader)
#                 train_accuracy = train_accuracies[task]
#                 wandb_log[f"Train {task} Loss"] = avg_task_loss
#                 wandb_log[f"Train {task} Accuracy"] = train_accuracy
#                 wandb_log[f"{task} Weight"] = self.dynamic_weights.loss_weights[task]
#                 print(f"Train - Task: {task}, Loss: {avg_task_loss:.4f}, Accuracy: {train_accuracy:.4f}")

#                 # Compute F1-scores
#                 f1_macro = f1_score(task_true_labels[task], task_predictions[task], average='macro')
#                 f1_micro = f1_score(task_true_labels[task], task_predictions[task], average='micro')
#                 f1_weighted = f1_score(task_true_labels[task], task_predictions[task], average='weighted')
#                 wandb_log[f"Train {task} Macro F1"] = f1_macro
#                 wandb_log[f"Train {task} Micro F1"] = f1_micro
#                 wandb_log[f"Train {task} Weighted F1"] = f1_weighted
#                 print(f"Train - Task: {task}, Macro F1: {f1_macro:.4f}, Micro F1: {f1_micro:.4f}, Weighted F1: {f1_weighted:.4f}")

#             # Validation phase
#             self.model.eval()
#             val_loss = 0.0
#             val_task_losses = {task: 0.0 for task in self.model.task_names}
#             val_task_correct = {task: 0 for task in self.model.task_names}
#             val_task_total = {task: 0 for task in self.model.task_names}
#             val_task_predictions = {task: [] for task in self.model.task_names}
#             val_task_true_labels = {task: [] for task in self.model.task_names}

#             with torch.no_grad():
#                 for imgs, labels in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
#                     imgs, labels = imgs.to(self.device), labels.to(self.device)  # Move inputs and labels to device
#                     outputs = self.model(imgs)
#                     val_losses = []

#                     for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
#                         task_loss = self.criterion(out, labels[:, i])
#                         val_losses.append(task_loss)
#                         val_task_losses[task] += task_loss.item()

#                         _, predicted = torch.max(out, 1)
#                         val_task_predictions[task].extend(predicted.cpu().tolist())
#                         val_task_true_labels[task].extend(labels[:, i].cpu().tolist())

#                         correct = (predicted == labels[:, i]).sum().item()
#                         val_task_correct[task] += correct
#                         val_task_total[task] += labels.size(0)

#                     val_loss += sum(val_losses).item()

#             avg_val_loss = val_loss / len(self.val_loader)
#             val_accuracies = {task: val_task_correct[task] / val_task_total[task] if val_task_total[task] > 0 else 0.0 for task in self.model.task_names}

#             wandb_log["Validation Loss"] = avg_val_loss
#             for task in self.model.task_names:
#                 avg_val_task_loss = val_task_losses[task] / len(self.val_loader)
#                 val_accuracy = val_accuracies[task]
#                 wandb_log[f"Validation {task} Loss"] = avg_val_task_loss
#                 wandb_log[f"Validation {task} Accuracy"] = val_accuracy

#                 # Compute F1-scores
#                 f1_macro = f1_score(val_task_true_labels[task], val_task_predictions[task], average='macro')
#                 f1_micro = f1_score(val_task_true_labels[task], val_task_predictions[task], average='micro')
#                 f1_weighted = f1_score(val_task_true_labels[task], val_task_predictions[task], average='weighted')
#                 wandb_log[f"Validation {task} Macro F1"] = f1_macro
#                 wandb_log[f"Validation {task} Micro F1"] = f1_micro
#                 wandb_log[f"Validation {task} Weighted F1"] = f1_weighted
#                 print(f"Validation - Task: {task}, Macro F1: {f1_macro:.4f}, Micro F1: {f1_micro:.4f}, Weighted F1: {f1_weighted:.4f}")

#             # Log training and validation metrics to wandb
#             wandb.log(wandb_log)
#             self.dynamic_weights.update_weights(train_accuracies)

#         torch.save(self.model.state_dict(), model_path)
#         wandb.save(model_path)
#         print(f"Model saved successfully at {model_path}.")
#         wandb.finish()
