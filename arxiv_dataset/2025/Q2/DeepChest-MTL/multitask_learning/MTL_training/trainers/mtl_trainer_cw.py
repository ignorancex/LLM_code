import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score
import numpy as np

class MTLTrainerCW:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, task_names, class_weights=None, project_name="MTL_Training"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.task_names = task_names
        self.class_weights = class_weights if class_weights else {task: None for task in task_names}  # Class weights for each task
        self.project_name = project_name

    def compute_class_weights(self, labels):
        """Compute class weights for each task based on label frequency"""
        class_weights = {}
        for task_idx, task in enumerate(self.task_names):
            task_labels = labels[:, task_idx].cpu().numpy()  # Get the labels for the current task
            class_counts = np.bincount(task_labels)
            total = len(task_labels)
            # Compute the class weights as inverse frequency (e.g., total / class_count)
            weights = total / (len(class_counts) * (class_counts + 1e-6))  # Adding small epsilon to avoid division by zero
            class_weights[task] = torch.tensor(weights, dtype=torch.float32).to(labels.device)
        return class_weights

    def train(self, num_epochs=10, model_path="fashion_classifier.pth"):
        wandb.init(project=self.project_name, config={
            "num_epochs": num_epochs,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        })

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            task_losses = {task: 0.0 for task in self.task_names}
            task_correct = {task: 0 for task in self.task_names}
            task_total = {task: 0 for task in self.task_names}
            all_true_labels = {task: [] for task in self.task_names}
            all_pred_labels = {task: [] for task in self.task_names}

            for imgs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
                self.optimizer.zero_grad()
                
                # Compute class weights dynamically based on the current batch
                class_weights = self.compute_class_weights(labels)

                outputs = self.model(imgs)
                losses = []

                for i, (out, task) in enumerate(zip(outputs, self.task_names)):
                    # Modify the criterion to include class weights
                    task_loss = self.criterion(out, labels[:, i], weight=class_weights[task] if class_weights[task] is not None else None)
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

            for task in self.task_names:
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
            val_task_losses = {task: 0.0 for task in self.task_names}
            val_task_correct = {task: 0 for task in self.task_names}
            val_task_total = {task: 0 for task in self.task_names}
            all_val_true_labels = {task: [] for task in self.task_names}
            all_val_pred_labels = {task: [] for task in self.task_names}

            with torch.no_grad():
                for imgs, labels in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                    outputs = self.model(imgs)
                    val_losses = []

                    for i, (out, task) in enumerate(zip(outputs, self.task_names)):
                        # Modify the criterion to include class weights
                        task_loss = self.criterion(out, labels[:, i], weight=class_weights[task] if class_weights[task] is not None else None)
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

            for task in self.task_names:
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
