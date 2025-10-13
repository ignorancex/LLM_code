import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from .mtl_trainer import MTLTrainer
from tqdm import tqdm
import wandb


class MTLTrainerWithGradNorm(MTLTrainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, alpha=0.12, project_name="MTL_GradNorm"):
        super().__init__(model, train_loader, val_loader, optimizer, criterion, project_name)
        self.alpha = alpha
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.task_weights = {task: 1.0 for task in self.model.task_names}

    def compute_gradient_norms(self, losses, model):
        """
        Compute gradient norms for each task loss.
        """
        gradient_norms = {}
        for task, loss in losses.items():
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grad_norm = torch.norm(
                torch.cat([
                    p.grad.flatten()
                    for p in model.parameters()
                    if p.grad is not None
                ])
            )
            gradient_norms[task] = grad_norm.item()
            #print(f"Task {task}, Gradient Norm: {gradient_norms[task]}")  # Debugging line
        return gradient_norms

    def compute_target_gradient_norms(self, gradient_norms, task_train_rates):
        """
        Compute target gradient norms for GradNorm.
        """
        avg_gradient_norm = sum(gradient_norms.values()) / len(gradient_norms)
        target_gradient_norms = {
            task: avg_gradient_norm * (rate ** self.alpha)
            for task, rate in task_train_rates.items()
        }
        return target_gradient_norms

    def update_task_weights(self, gradient_norms, target_gradient_norms):
        """
        Update task-specific weights using GradNorm.
        """
        gradient_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task in gradient_norms.keys():
            grad_norm = torch.tensor(gradient_norms[task], device=self.device, requires_grad=True)
            target_grad_norm = torch.tensor(target_gradient_norms[task], device=self.device, requires_grad=False)
            gradient_loss = gradient_loss + torch.abs(grad_norm - target_grad_norm)

            #print(f"Task {task}, Grad Norm: {grad_norm.item()}, Target Grad Norm: {target_grad_norm.item()}")

        self.optimizer.zero_grad()
        gradient_loss.backward()

        with torch.no_grad():
            for task in self.task_weights.keys():
                grad_norm_diff = gradient_norms[task] - target_gradient_norms[task]
                self.task_weights[task] -= self.alpha * grad_norm_diff
                self.task_weights[task] = max(self.task_weights[task], 1e-6)  # Ensure weights remain positive

        #print(f"Updated Task Weights: {self.task_weights}")
        return gradient_loss.item()

    def train(self, num_epochs=10, model_path="model_with_gradnorm.pth"):
        wandb.init(project="MTL_GradNorm", config={
            "num_epochs": num_epochs,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "alpha": self.alpha,
        })

        for epoch in range(num_epochs):
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

                losses = {}
                for i, task in enumerate(self.model.task_names):
                    out = outputs[task]
                    task_loss = self.task_weights[task] * self.criterion(out, labels[:, i])
                    losses[task] = task_loss
                    task_losses[task] += task_loss.item()

                    _, predicted = torch.max(out, 1)
                    task_predictions[task].extend(predicted.cpu().tolist())
                    task_true_labels[task].extend(labels[:, i].cpu().tolist())

                    correct = (predicted == labels[:, i]).sum().item()
                    task_correct[task] += correct
                    task_total[task] += labels.size(0)

                gradient_norms = self.compute_gradient_norms(losses, self.model)
                target_gradient_norms = self.compute_target_gradient_norms(gradient_norms, task_correct)
                self.update_task_weights(gradient_norms, target_gradient_norms)

                total_loss = sum(losses.values())
                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            train_accuracies = {task: task_correct[task] / task_total[task] for task in self.model.task_names}
            train_f1_scores = {task: {
                "macro": f1_score(task_true_labels[task], task_predictions[task], average='macro'),
                "micro": f1_score(task_true_labels[task], task_predictions[task], average='micro'),
                "weighted": f1_score(task_true_labels[task], task_predictions[task], average='weighted')
            } for task in self.model.task_names}

            val_loss, val_accuracies, val_f1_scores = self.evaluate()

            wandb_log = {"epoch": epoch + 1, "Train Loss": avg_train_loss, "Validation Loss": val_loss}
            for task in self.model.task_names:
                wandb_log[f"Train {task} Accuracy"] = train_accuracies[task]
                wandb_log[f"Train {task} Macro F1"] = train_f1_scores[task]["macro"]
                wandb_log[f"Train {task} Micro F1"] = train_f1_scores[task]["micro"]
                wandb_log[f"Train {task} Weighted F1"] = train_f1_scores[task]["weighted"]

                wandb_log[f"Validation {task} Accuracy"] = val_accuracies[task]
                wandb_log[f"Validation {task} Macro F1"] = val_f1_scores[task]["macro"]
                wandb_log[f"Validation {task} Micro F1"] = val_f1_scores[task]["micro"]
                wandb_log[f"Validation {task} Weighted F1"] = val_f1_scores[task]["weighted"]

            wandb.log(wandb_log)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        torch.save(self.model.state_dict(), model_path)
        wandb.save(model_path)
        print(f"Model saved to {model_path}.")
        wandb.finish()

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        val_loss = 0.0
        task_correct = {task: 0 for task in self.model.task_names}
        task_total = {task: 0 for task in self.model.task_names}
        task_predictions = {task: [] for task in self.model.task_names}
        task_true_labels = {task: [] for task in self.model.task_names}
        task_losses = {task: 0.0 for task in self.model.task_names}

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validation"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)

                for i, task in enumerate(self.model.task_names):
                    out = outputs[task]
                    task_loss = self.criterion(out, labels[:, i])
                    task_losses[task] += task_loss.item()

                    _, predicted = torch.max(out, 1)
                    task_predictions[task].extend(predicted.cpu().tolist())
                    task_true_labels[task].extend(labels[:, i].cpu().tolist())

                    correct = (predicted == labels[:, i]).sum().item()
                    task_correct[task] += correct
                    task_total[task] += labels.size(0)

        avg_val_loss = sum(task_losses.values()) / len(task_losses)
        val_accuracies = {task: task_correct[task] / task_total[task] for task in self.model.task_names}
        val_f1_scores = {task: {
            "macro": f1_score(task_true_labels[task], task_predictions[task], average='macro'),
            "micro": f1_score(task_true_labels[task], task_predictions[task], average='micro'),
            "weighted": f1_score(task_true_labels[task], task_predictions[task], average='weighted')
        } for task in self.model.task_names}

        return avg_val_loss, val_accuracies, val_f1_scores
