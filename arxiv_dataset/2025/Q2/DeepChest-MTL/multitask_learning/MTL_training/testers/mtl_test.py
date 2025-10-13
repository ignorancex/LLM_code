# import torch
# from tqdm import tqdm
# import wandb
# from sklearn.metrics import f1_score

# class MTLTester:
#     def __init__(self, model, test_loader, criterion, run_name="MTL_Test_Run"):
#         """
#         Initializes the Tester class.

#         Args:
#             model (torch.nn.Module): The model to be tested.
#             test_loader (DataLoader): DataLoader for test data.
#             criterion (nn.Module): The loss function.
#         """
#         self.model = model
#         self.test_loader = test_loader
#         self.criterion = criterion
#         self.run_name = run_name

#     def test(self):
#         """
#         Tests the model and logs results to wandb.
#         """
#         # Initialize wandb
#         wandb.init(project="fashion-multitask-classification", name=self.run_name)

#         self.model.eval()
#         test_loss = 0.0
#         test_task_losses = {task: 0.0 for task in self.model.task_names}
#         test_task_correct = {task: 0 for task in self.model.task_names}
#         test_task_total = {task: 0 for task in self.model.task_names}

#         # Store predictions and labels for F1 score calculation
#         all_preds = {task: [] for task in self.model.task_names}
#         all_labels = {task: [] for task in self.model.task_names}

#         with torch.no_grad():
#             for imgs, labels in tqdm(self.test_loader, desc="Testing"):
#                 outputs = self.model(imgs)
#                 losses = []

#                 for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
#                     # Calculate task loss
#                     task_loss = self.criterion(out, labels[:, i])
#                     losses.append(task_loss)
#                     test_task_losses[task] += task_loss.item()

#                     # Predict and calculate accuracy
#                     _, predicted = torch.max(out, 1)
#                     correct = (predicted == labels[:, i]).sum().item()
#                     test_task_correct[task] += correct
#                     test_task_total[task] += labels.size(0)

#                     # Collect predictions and labels for F1 score
#                     all_preds[task].extend(predicted.cpu().numpy())
#                     all_labels[task].extend(labels[:, i].cpu().numpy())

#                 # Sum up all task losses
#                 test_loss += sum(losses).item()

#         # Average test loss across batches
#         avg_test_loss = test_loss / len(self.test_loader)

#         # Calculate and log metrics
#         wandb.log({"test_loss": avg_test_loss})
#         for task in self.model.task_names:
#             avg_task_loss = test_task_losses[task] / len(self.test_loader)
#             test_accuracy = (
#                 test_task_correct[task] / test_task_total[task]
#                 if test_task_total[task] > 0 else 0.0
#             )

#             # Calculate F1 scores
#             f1_micro = f1_score(all_labels[task], all_preds[task], average="micro")
#             f1_macro = f1_score(all_labels[task], all_preds[task], average="macro")
#             f1_weighted = f1_score(all_labels[task], all_preds[task], average="weighted")

#             # Log metrics to wandb
#             wandb.log({
#                 f"test_loss_{task}": avg_task_loss,
#                 f"test_accuracy_{task}": test_accuracy,
#                 f"test_f1_micro_{task}": f1_micro,
#                 f"test_f1_macro_{task}": f1_macro,
#                 f"test_f1_weighted_{task}": f1_weighted
#             })

#             print(f"Task: {task}")
#             print(f"  Loss: {avg_task_loss:.4f}")
#             print(f"  Accuracy: {test_accuracy:.4f}")
#             print(f"  F1-Score (Micro): {f1_micro:.4f}")
#             print(f"  F1-Score (Macro): {f1_macro:.4f}")
#             print(f"  F1-Score (Weighted): {f1_weighted:.4f}")

#         print(f"Total Test Loss: {avg_test_loss:.4f}")

import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score

class MTLTester:
    def __init__(self, model, test_loader, criterion, run_name="MTL_Test_Run"):
        """
        Initializes the Tester class.

        Args:
            model (torch.nn.Module): The model to be tested.
            test_loader (DataLoader): DataLoader for test data.
            criterion (nn.Module): The loss function.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)  # Move model to the device
        self.test_loader = test_loader
        self.criterion = criterion
        self.run_name = run_name

    def test(self):
        """
        Tests the model and logs results to wandb.
        """
        # Initialize wandb
        wandb.init(project="fashion-multitask-classification", name=self.run_name)

        self.model.eval()
        test_loss = 0.0
        test_task_losses = {task: 0.0 for task in self.model.task_names}
        test_task_correct = {task: 0 for task in self.model.task_names}
        test_task_total = {task: 0 for task in self.model.task_names}

        # Store predictions and labels for F1 score calculation
        all_preds = {task: [] for task in self.model.task_names}
        all_labels = {task: [] for task in self.model.task_names}

        with torch.no_grad():
            for imgs, labels in tqdm(self.test_loader, desc="Testing"):
                # Move data to the device
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(imgs)
                losses = []

                for i, (out, task) in enumerate(zip(outputs, self.model.task_names)):
                    # Calculate task loss
                    task_loss = self.criterion(out, labels[:, i])
                    losses.append(task_loss)
                    test_task_losses[task] += task_loss.item()

                    # Predict and calculate accuracy
                    _, predicted = torch.max(out, 1)
                    correct = (predicted == labels[:, i]).sum().item()
                    test_task_correct[task] += correct
                    test_task_total[task] += labels.size(0)

                    # Collect predictions and labels for F1 score
                    all_preds[task].extend(predicted.cpu().numpy())
                    all_labels[task].extend(labels[:, i].cpu().numpy())

                # Sum up all task losses
                test_loss += sum(losses).item()

        # Average test loss across batches
        avg_test_loss = test_loss / len(self.test_loader)

        # Calculate and log metrics
        wandb.log({"test_loss": avg_test_loss})
        for task in self.model.task_names:
            avg_task_loss = test_task_losses[task] / len(self.test_loader)
            test_accuracy = (
                test_task_correct[task] / test_task_total[task]
                if test_task_total[task] > 0 else 0.0
            )

            # Calculate F1 scores
            f1_micro = f1_score(all_labels[task], all_preds[task], average="micro")
            f1_macro = f1_score(all_labels[task], all_preds[task], average="macro")
            f1_weighted = f1_score(all_labels[task], all_preds[task], average="weighted")

            # Log metrics to wandb
            wandb.log({
                f"test_loss_{task}": avg_task_loss,
                f"test_accuracy_{task}": test_accuracy,
                f"test_f1_micro_{task}": f1_micro,
                f"test_f1_macro_{task}": f1_macro,
                f"test_f1_weighted_{task}": f1_weighted
            })

            print(f"Task: {task}")
            print(f"  Loss: {avg_task_loss:.4f}")
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  F1-Score (Micro): {f1_micro:.4f}")
            print(f"  F1-Score (Macro): {f1_macro:.4f}")
            print(f"  F1-Score (Weighted): {f1_weighted:.4f}")

        print(f"Total Test Loss: {avg_test_loss:.4f}")
