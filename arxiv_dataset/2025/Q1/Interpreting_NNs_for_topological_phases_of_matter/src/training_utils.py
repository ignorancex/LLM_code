import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

warnings.filterwarnings("ignore")

import copy
import os
import pathlib
import time

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from prettytable.colortable import ColorTable, Theme
from sklearn.metrics import classification_report
from termcolor import cprint
from torch.autograd import Variable
from torch.utils.data import random_split
from torchvision import datasets, transforms, utils
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from src.architecture import CNN_SSH
from src.data_loaders import Importer
from src.coral import coral


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss

    def __call__(
        self, current_valid_loss, model, verb=False, checkpoint_best="upgraded_model_x"
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            if verb:
                cprint("[INFO]", "magenta", end=" ")
                print(f"\nBest validation loss: {self.best_valid_loss}")
                cprint("[INFO]", "magenta", end=" ")
                print(f"\nSaving new best model\n")
            torch.save(model.state_dict(), f"Models/{checkpoint_best}_checkpoint.dict")


class CheckpointBest(object):
    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.checkpointed_epoch = 0
        self.checkpointed_model_parts = None

    def __call__(
        self,
        current_valid_loss,
        current_epoch,
        model,
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.checkpointed_model_parts = model
            self.checkpointed_epoch = current_epoch
            return True
        return False

    def get_checkpointed_model(self):
        return self.checkpointed_model_parts

    def get_best_valid_loss(self):
        return self.best_valid_loss

    def get_checkpointed_epoch(self):
        return self.checkpointed_epoch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, warmup=100):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.warmup = warmup

    def early_stop(self, validation_loss, epoch):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        else:
            self.counter += 1

        if self.counter >= self.patience and epoch >= self.warmup:
            return True
        return False


def count_parameters(model, device):
    th = Theme(
        default_color="36",
        vertical_color="32",
        horizontal_color="32",
        junction_color="36",
    )
    th_2 = Theme(
        default_color="92",
        vertical_color="32",
        horizontal_color="32",
        junction_color="36",
    )
    table = ColorTable(["Modules", "Parameters"], theme=th)
    table_2 = ColorTable(["Additional stats", ""], theme=th_2)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, f"{params:,}"])
        total_params += params
    print(table)

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    table_2.add_row(["Total parameters", f"{total_params:,}"])
    table_2.add_row(["Trainable parameters", f"{total_trainable_params:,}"])
    table_2.add_row(["Computation device", device])
    print(table_2)
    return total_params


def train_model(
    model,
    epoch_count,
    trainDataLoader,
    valDataLoader,
    optimizer,
    batch_size,
    classes_no,
    device,
    print_all=True,
    therm=False,
    therm_levels=100,
    checkpoint_best=False,
    early_stopping=None,
    scheduler=None,
    fname=None,
    leave=True,
    coralTargetLoader=None,
    lambda_coral=0.5,
):
    """
    CNN training function

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    epoch_count : int
        The number of epochs to train the model.
    trainDataLoader : torch.utils.data.DataLoader
        The data loader for the training set.
    valDataLoader : torch.utils.data.DataLoader
        The data loader for the validation set.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    batch_size : int
        The batch size used for training.
    classes_no : int
        The number of classes in the dataset.
    device : torch.device
        The device (CPU or GPU) to be used for training.
    print_all : bool, optional
        Whether to print the training and validation information. Default is True.
    therm : bool, optional
        Should the model use Thermometer encoding. Default is False.
    therm_levels : int, optional
        The number of bins for thermometer encoding. Default is 100.
    checkpoint_best : str, optional
        The filename of the checkpointed model. Default is None.
    early_stopping : dict, optional
        The configuration for early stopping. Default is None.
    scheduler : torch.optim.lr_scheduler, optional
        The learning rate scheduler. Default is None.
    fname : str, optional
        The filename to save the model. Default is None.
    leave : bool, optional
        Whether to keep the progress bar after training is finished. Default is True.

    Returns
    -------
    H : dict
        Training history dictionary.

    References
    ----------
    .. [1] Buckman, Jacob, Aurko Roy, Colin Raffel, and Ian Goodfellow.
    "Thermometer encoding: One hot way to resist adversarial examples."
    In International Conference on Learning Representations. 2018.

    """

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [], 
        "val_loss": [], 
        "val_acc": []
        }
    if coralTargetLoader is not None:
        H['train_classification_loss'] = []
        H['train_coral_loss'] = []


    trainSteps = len(trainDataLoader.dataset) // batch_size
    valSteps = len(valDataLoader.dataset) // batch_size

    # cprint("\n[INFO]", "magenta", end=" ")
    # print("Training the network...")
    startTime = time.time()

    # cprint("\n======================================", "green")
    # cprint("          Nerd stats section\n", "green")
    # count_parameters(model, device)
    # cprint("\n======================================\n", "green")

    if checkpoint_best:
        # initialize SaveBestModel class
        checkpointer = CheckpointBest()

    if early_stopping is not None:
        early_stopper = EarlyStopper(
            patience=early_stopping["patience"],
            min_delta=early_stopping["tolerance"],
            warmup=early_stopping["warmup"],
        )

    outer = tqdm(
        range(0, epoch_count),
        desc="Epoch",
        colour="green",
        unit="epoch",
        position=1,
        leave=leave,
    )
    time_epoch_pbar = tqdm(total=0, position=2, bar_format="{desc}", leave=leave)
    train_pbar = tqdm(total=0, position=3, bar_format="{desc}", leave=leave)
    val_pbar = tqdm(total=0, position=4, bar_format="{desc}", leave=leave)

    train_acc = 0
    val_acc = 0

    for e in range(0, epoch_count):
        # set the training mode in the model
        if type(model) is list:
            for m in model:
                m.train()
        else:
            model.train()

        # Initialize total traininig and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        if coralTargetLoader is not None:
            totalClassificationLoss = 0
            totalCoralLoss = 0

        # Initialize the number of correct predictons in the training and validation step
        trainCorrect = 0
        valCorrect = 0

        coral_involved = (train_acc > 0.95) and (val_acc > 0.95)
        # coral_involved = True

        # Loop over the training set
        for batch_idx, (x, y) in enumerate(trainDataLoader):
            y = F.one_hot(y, num_classes=classes_no)
            y = y.float()

            (x, y) = (x.to(device), y.to(device))
            # Zero out the gradients


            if coralTargetLoader is not None:
                target_data, _ = next(iter(coralTargetLoader))
                target_data = target_data.to(device)
                if x.shape[0] != target_data.shape[0]:
                    continue
                
                if type(model) is not list:
                    acts_tmp = []
                    def hook(model, input, output):
                        acts_tmp.append(output)

                    hook_model = model.cnn_layers_2.register_forward_hook(hook)

                    pred = model(x)
                    x_act = acts_tmp[0]
                    pred_coral = model(target_data)
                    x_coral_act = acts_tmp[1]
                    hook_model.remove()
                else:
                    x_act = model[0](x)
                    x_coral_act = model[0](x)
                    pred = model[1](x_act)
                coral_loss = lambda_coral * coral(x_act, x_coral_act)
                classification_loss = nn.BCELoss()(pred, y)
                    # loss = classification_loss
                if type(model) is not list:
                    if coral_involved:
                        loss = classification_loss + coral_loss
                    else:
                        loss = classification_loss
                else:
                    loss = classification_loss + coral_loss
            else:
                pred = model(x)
                coral_loss = 0
                classification_loss = nn.BCELoss()(pred, y)
                loss = classification_loss

            # Zero out the gradients, perform the backprop step, and update the weights
            # optimizer.zero_grad() is supposedly slower than this loop:
            if type(model) is not list:
                for param in model.parameters():
                    param.grad = None
                loss.backward()
                optimizer.step()
            else:
                for model_tmp in model:
                    for param in model_tmp.parameters():
                        param.grad = None
                loss.backward()
                for opt in optimizer:
                    opt.step()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss
            if coralTargetLoader is not None:
                totalClassificationLoss += classification_loss
                totalCoralLoss += coral_loss
            trainCorrect += (
                (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            )

            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         e, batch_idx * len(data['is_insulator']), len(trainloader.dataset),
            #         100. * batch_idx / len(trainloader), loss.item()))

        # Validation

        # Switch off autograd for evaluation
        with torch.no_grad():
            if type(model) is not list:
                #  Set model in evaluation mode
                model.eval()
            else:
                for m in model:
                    m.eval()

            #  Loop over the validation set
            for (x, y) in valDataLoader:
                # These three lines are essential to apply for our data!!!!!!
                y = F.one_hot(y, num_classes=classes_no)
                y = y.float()

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                if type(model) is not list:
                    # Make the predictions and calulate the validation loss
                    pred = model(x)
                else:
                    pred = model[1](model[0](x))
                totalValLoss += nn.BCELoss()(pred, y)

                # Calculate the number of correct predictions
                valCorrect += (
                    (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
                )

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        if coralTargetLoader is not None:
            avgClassLoss = totalClassificationLoss / trainSteps
            avgCoralLoss = totalCoralLoss / trainSteps
            coral_loss_last = avgCoralLoss

        # Calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        train_acc = trainCorrect
        valCorrect = valCorrect / len(valDataLoader.dataset)
        val_acc = valCorrect

        # save the best model till now if we have the least loss in the current epoch
        if checkpoint_best:
            saved_best = checkpointer(
                avgValLoss, e, model, 
            )

        # Update the training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        if coralTargetLoader is not None:
            H['train_classification_loss'].append(avgClassLoss.cpu().detach().numpy())
            if avgCoralLoss != 0:
                H['train_coral_loss'].append(avgCoralLoss.cpu().detach().numpy())
            else:
                H['train_coral_loss'].append(0)


        # TODO: Generalize this
        if early_stopping is not None:
            if early_stopper.early_stop(H["train_loss"][-1], e):
                break

        if scheduler is not None:
            scheduler.step(avgValLoss)

        # print the model training and validation information
        if print_all:
            time_epoch_pbar.set_description_str(
                f"{'Current time':^12}: {time.strftime('%H:%M:%S', time.localtime())}"
            )
            time_epoch_pbar.update(1)
            if coralTargetLoader is None:
                train_pbar.set_description_str(
                    "{:^12}: {:.6f},{:^12}: {:.4f}".format(
                        "Train loss", avgTrainLoss, "Train acc", trainCorrect
                    )
                )
            else:
                if avgCoralLoss > 1e-3:
                    train_pbar.set_description_str(
                        "{:^12}: {:.6f},{:^12}: {:.4f}, {}: {:.6f}, {}: {:.6f}".format(
                            "Train loss", avgClassLoss, "Train acc", trainCorrect, "Composite loss", avgTrainLoss, "Coral loss", avgCoralLoss
                        )
                    )
                else:
                    train_pbar.set_description_str(
                        "{:^12}: {:.6f},{:^12}: {:.4f}, {}: {:.6f}, {}: {:.6e}".format(
                            "Train loss", avgClassLoss, "Train acc", trainCorrect, "Composite loss", avgTrainLoss, "Coral loss", avgCoralLoss
                        )
                    )
            train_pbar.update(1)
            val_pbar.set_description_str(
                "{:^12}: {:.6f},{:^12}: {:.4f}".format(
                    "Val loss", avgValLoss, "Val acc", valCorrect
                )
            )
            val_pbar.update(1)
        else:
            pass

        outer.update(1)

    # finish measuring how long training took
    endTime = time.time()

    if checkpoint_best:
        return H, checkpointer
    else:
        return H


def plot_training_history(H, plots_save_path, fname):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if 'train_coral_loss' in H.keys():
        ax[0].plot(H["train_loss"], label="Full training loss")
        ax[0].plot(H["train_classification_loss"], "--", label="Training dataset")
        ax[0].plot(H["train_coral_loss"], "--", label="CORAL loss")
    else:
        ax[0].plot(H["train_loss"], label="Training dataset")
    ax[0].plot(H["val_loss"], label="Validation dataset")

    ax[1].plot(H["train_acc"], label="Training dataset")
    ax[1].plot(H["val_acc"], label="Validation dataset")

    fig.suptitle("Loss and Accuracy during model training", size=25)

    ax[0].set_xlabel("Epoch No.")
    ax[1].set_xlabel("Epoch No.")
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Accuracy")

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")

    fig.savefig(
        plots_save_path.joinpath(f"{fname}_training_history.pdf"), bbox_inches="tight"
    )
    plots_save_path.joinpath("PNG").mkdir(exist_ok=True, parents=True)
    fig.savefig(
        plots_save_path.joinpath("PNG").joinpath(f"{fname}_training_history.png"),
        bbox_inches="tight",
    )
