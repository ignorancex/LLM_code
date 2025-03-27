from tqdm import tqdm
from torchvision import transforms
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss
import numpy as np

from models import REGISTRY_MODEL, get_family
from datasets import REGISTRY_INFO


class KLDivLoss_custom(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        super(KLDivLoss_custom, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return F.kl_div(
            F.log_softmax(input, dim=-1),
            target,
            reduction=self.reduction,
            log_target=self.log_target,
        )


data_transforms = transforms.Compose([transforms.RandomResizedCrop(256),])


def perturb(x_batch, bounds=[-1, 1], device=torch.device("cuda")):
    x_batch = (x_batch - bounds[0]) / (bounds[1] - bounds[0])
    x_batch = x_batch.cpu()

    if x_batch.ndim == 3:
        x_batch = x_batch.unsqueeze(0)

    if x_batch.shape[1] == 1:
        normalize = transforms.Normalize([0.5], [0.5])
    else:
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                15, translate=(0.1, 0.1), scale=(0.9, 1.0), shear=(0.1, 0.1)
            ),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    x_batch_mod = torch.stack([data_transforms(xi) for xi in x_batch], axis=0)
    x_batch_mod = x_batch_mod.to(device)

    return x_batch_mod


def adaptive_pred(model, x, n=5, bounds=[-1, 1], mode="normal"):
    ys = []
    diff_list = []
    y_orig = 0
    hash_list = []
    for i in range(n):
        x_mod = perturb(x, bounds)
        hash_mod = np.array(model.get_hash_list(x_mod))
        hash_list.append(hash_mod)
        if mode == "normal":
            y = model(x_mod)
        elif mode == "normal_sim":
            y = model(x_mod, index=i)
            y_orig = model(x, index=i)
            y_orig = F.softmax(y_orig, dim=-1)
        elif mode == "ideal_attack":
            y = model(x, index=i)
        elif mode == "ideal_defense":
            y = model(x_mod, x_hash=x)
        else:
            raise ValueError(f"invalid mode {mode}")

        y = F.softmax(y, dim=-1)

        diff = y_orig - y
        diff_abs_sum = torch.abs(diff).sum(dim=-1)
        diff_list.append(diff_abs_sum.cpu().numpy())
        ys.append(y)
    hash_np = np.stack(hash_list, axis=-1)
    num_unique = [len(np.unique(hi)) for hi in hash_np]
    num_unique_avg = np.mean(num_unique)
    diff_mean = np.mean(np.stack(diff_list))
    ys = torch.stack(ys, dim=0)
    return torch.mean(ys, dim=0), num_unique_avg

class Knockoff:
    def __init__(self, teacher, dataset, model, batch_size=128, budget=50000, epochs=50, lr_S=0.1, pred_type='soft', disable_pbar=False, adaptive_mode="none", n_adaptive_queries=5, scheduler="multistep", steps=[0.1, 0.3, 0.5], scale=0.3, weight_decay=5e-4, momentum=0.9, device=torch.device("cuda"), **kwargs):
        
        self.num_classes = REGISTRY_INFO[dataset][0]
        self.budget = budget
        self.batch_size = batch_size
        self.pred_type = pred_type
        self.device = device

        family = get_family(dataset)
        self.student = REGISTRY_MODEL[family][model](self.num_classes).to(device)
        self.teacher = teacher

        self.epochs = epochs

        print(f"\nProxy self.budget: {budget}")
        print("Train self.epochs: ", self.epochs)

        self.opt = optim.SGD(self.student.parameters(), lr=lr_S, weight_decay=weight_decay, momentum=momentum)

        steps = sorted([int(step * self.epochs) for step in steps])
        print("Learning rate scheduling at steps: ", steps)
        print()

        if scheduler == "multistep":
            self.sch = optim.lr_scheduler.MultiStepLR(self.opt, steps, scale)
        elif scheduler == "cosine":
            self.sch = optim.lr_scheduler.CosineAnnealingLR(self.opt, self.epochs)

        self.disable_pbar = disable_pbar
        self.adaptive_mode = adaptive_mode
        self.n_adaptive_queries = n_adaptive_queries


    def run(self, test_loader, sur_loader, victim_acc):
        acc_list = []

        device = self.device
        # self.student = self.student.to(device)
        results = {"self.epochs": [], "accuracy": [], "accuracy_x": []}

        print("== Constructing Surrogate Dataset ==")
        xs = torch.tensor([])
        ys = torch.tensor([])
        self.teacher.eval()
        queries = 0
        unique_list = []
        with torch.no_grad():
            for x, _ in tqdm(sur_loader, ncols=100, leave=True, disable=self.disable_pbar):
                x = x.to(device)
                if self.adaptive_mode != "none":
                    if self.pred_type == "hard":
                        raise ValueError(
                            "adaptive attacks is only supported for self.pred_type=soft"
                        )
                    y, n_unique = adaptive_pred(
                        self.teacher, x, mode=self.adaptive_mode, n=self.n_adaptive_queries
                    )
                    unique_list.append(n_unique)
                else:
                    y = self.teacher(x)
                    if self.pred_type == "soft":
                        y = F.softmax(y, dim=-1)
                    else:
                        y = torch.argmax(y, dim=-1)

                xs = torch.cat((xs, x.cpu()), dim=0)
                ys = torch.cat((ys, y.cpu()), dim=0)
                queries += x.shape[0]
                if queries >= self.budget:
                    break

        if self.pred_type == "hard":
            ys = ys.long()

        ds_knockoff = TensorDataset(xs, ys)

        dataloader_knockoff = torch.utils.data.DataLoader(
            ds_knockoff, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

        print("\n== Training Clone Model ==")
        if self.pred_type == "soft":
            criterion = KLDivLoss_custom(reduction="batchmean")
        else:
            criterion = CrossEntropyLoss()

        for epoch in range(1, self.epochs + 1):

            loss_train, acc_train = train_epoch(
                self.student, dataloader_knockoff, self.opt, criterion, self.disable_pbar, self.device
            )
            acc_test = test(self.student, test_loader, self.device)

            print(
                "Epoch: {} Loss: {:.4f} Train Acc: {:.2f} Test Acc: {:.2f} ({:.2f}x)".format(
                    epoch, loss_train, 100 * acc_train, 100 * acc_test, acc_test / victim_acc
                )
            )

            if self.sch:
                self.sch.step()
            results["self.epochs"].append(epoch)
            results["accuracy"].append(acc_test)
            results["accuracy_x"].append(acc_test / victim_acc)

            acc_list.append(acc_test)
            # print("Epoch {}, steal acc: {:.2f}%({:.2f}x)".format(epoch, acc_test * 100, acc_test / victim_acc))
            
        return acc_list
        

def test(model, data_loader, device):
    """
    test accuracy
    """
    model = model.to(device)
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == y).sum().item()
        acc = correct / len(data_loader.dataset)
    return acc


def train_epoch(
    model: Module,
    data_loader: DataLoader,
    opt: Optimizer,
    criterion: _Loss,
    disable_pbar: bool = False,
    device = torch.device("cuda"),
):
    """
    Train for 1 epoch
    """
    model = model.to(device)
    model.train()
    running_loss = correct = 0.0
    n_batches = len(data_loader)
    for (x, y) in tqdm(data_loader, ncols=80, disable=disable_pbar, leave=False):
        # if y.shape[0] < 128:
        #    continue

        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        pred_class = torch.argmax(pred, dim=-1)
        if y.ndim == 2:
            y = torch.argmax(y, dim=-1)
        correct += (pred_class == y).sum().item()

    loss = running_loss / n_batches
    acc = correct / len(data_loader.dataset)
    return loss, acc