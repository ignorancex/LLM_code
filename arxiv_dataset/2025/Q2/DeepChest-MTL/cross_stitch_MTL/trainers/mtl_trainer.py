import torch
from tqdm import tqdm
import wandb
import sys, os
from sklearn.metrics import f1_score


class MTLTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, project_name="MTL_Training", device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.project_name = project_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training is on {self.device}")

        # Move model to the appropriate device
        self.model = self.model.to(self.device)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training is on {self.device}")
        self.model.to(self.device)


    def train(self, num_epochs=10, model_path="fashion_classifier.pth"):
        wandb.init(project=self.project_name, config={
            "num_epochs": num_epochs,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        })

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            task_losses = {task: 0.0 for task in self.model.task_names}
            task_correct = {task: 0 for task in self.model.task_names}
            task_total = {task: 0 for task in self.model.task_names}
            all_true_labels = {task: [] for task in self.model.task_names}
            all_pred_labels = {task: [] for task in self.model.task_names}

            for imgs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):

                imgs, labels = imgs.to(self.device), labels.to(self.device)

                # Move data to the device
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)


                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                losses = []

                for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
                    task_loss = self.criterion(out, labels[:, i])
                    losses.append(task_loss)
                    task_losses[task] += task_loss.item()

                    _, predicted = torch.max(out, 1)
                    correct = (predicted == labels[:, i]).sum().item()
                    task_correct[task] += correct
                    task_total[task] += labels.size(0)

                    # Collect true and predicted labels for F1-score
                    all_true_labels[task].extend(labels[:, i].cpu().numpy())
                    all_pred_labels[task].extend(predicted.cpu().numpy())

                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            wandb_log = {"epoch": epoch + 1, "Train Loss": avg_train_loss}

            for task in self.model.task_names:
                avg_task_loss = task_losses[task] / len(self.train_loader)
                train_accuracy = task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0
                # Calculate F1-scores
                f1_macro = f1_score(all_true_labels[task], all_pred_labels[task], average='macro')
                f1_micro = f1_score(all_true_labels[task], all_pred_labels[task], average='micro')
                f1_weighted = f1_score(all_true_labels[task], all_pred_labels[task], average='weighted')

                wandb_log[f"Train {task} Loss"] = avg_task_loss
                wandb_log[f"Train {task} Accuracy"] = train_accuracy
                wandb_log[f"Train {task} F1 Macro"] = f1_macro
                wandb_log[f"Train {task} F1 Micro"] = f1_micro
                wandb_log[f"Train {task} F1 Weighted"] = f1_weighted

                print(f"Train - Task: {task}, Loss: {avg_task_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                      f"F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}, F1 Weighted: {f1_weighted:.4f}")

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_task_losses = {task: 0.0 for task in self.model.task_names}
            val_task_correct = {task: 0 for task in self.model.task_names}
            val_task_total = {task: 0 for task in self.model.task_names}
            all_val_true_labels = {task: [] for task in self.model.task_names}
            all_val_pred_labels = {task: [] for task in self.model.task_names}

            with torch.no_grad():
                for imgs, labels in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):

                    imgs, labels = imgs.to(self.device), labels.to(self.device)

                    # Move data to the device
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)


                    outputs = self.model(imgs)
                    val_losses = []

                    for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
                        task_loss = self.criterion(out, labels[:, i])
                        val_losses.append(task_loss)
                        val_task_losses[task] += task_loss.item()

                        _, predicted = torch.max(out, 1)
                        correct = (predicted == labels[:, i]).sum().item()
                        val_task_correct[task] += correct
                        val_task_total[task] += labels.size(0)

                        # Collect true and predicted labels for F1-score
                        all_val_true_labels[task].extend(labels[:, i].cpu().numpy())
                        all_val_pred_labels[task].extend(predicted.cpu().numpy())

                    val_loss += sum(val_losses).item()

            avg_val_loss = val_loss / len(self.val_loader)
            wandb_log["Validation Loss"] = avg_val_loss

            for task in self.model.task_names:
                avg_val_task_loss = val_task_losses[task] / len(self.val_loader)
                val_accuracy = val_task_correct[task] / val_task_total[task] if val_task_total[task] > 0 else 0.0
                # Calculate F1-scores
                f1_macro = f1_score(all_val_true_labels[task], all_val_pred_labels[task], average='macro')
                f1_micro = f1_score(all_val_true_labels[task], all_val_pred_labels[task], average='micro')
                f1_weighted = f1_score(all_val_true_labels[task], all_val_pred_labels[task], average='weighted')

                wandb_log[f"Validation {task} Loss"] = avg_val_task_loss
                wandb_log[f"Validation {task} Accuracy"] = val_accuracy
                wandb_log[f"Validation {task} F1 Macro"] = f1_macro
                wandb_log[f"Validation {task} F1 Micro"] = f1_micro
                wandb_log[f"Validation {task} F1 Weighted"] = f1_weighted

                print(f"Validation - Task: {task}, Loss: {avg_val_task_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
                      f"F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}, F1 Weighted: {f1_weighted:.4f}")

            # Log training and validation results to wandb
            wandb.log(wandb_log)

        torch.save(self.model.state_dict(), model_path)
        wandb.save(model_path)
        print(f"Model saved successfully at {model_path}.")
        wandb.finish()
