from typing import Any
import copy
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

class EarlyStopping():
    """
    Early Stopping ito stop the training when the loss does not decrease after certain epochs
    """

    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        """_summary_

        Args:
            patience (int, optional): _description_. Defaults to 10.
                    How many epochs to wait before stopping when loss is not improving
            min_delta (int, optional): _description_. Defaults to 0.
                    minimum difference between new loss and old loss for new loss to be
                    considerred as an improvement
            restore_best_weights (bool, optional): _description_. Defaults to True.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.restore_patients = 0
        self.best_model = None
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.status = ""
        self.early_stop = False

    def __call__(self, model, val_loss, train_acc, val_acc, epoch, train_acc_threshold=0.80, epoch_threshold = 20):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_acc = val_acc

        elif val_acc > self.best_acc and epoch >= epoch_threshold and train_acc >= train_acc_threshold:
            self.counter = 0
            self.restore_patients = 0
            self.best_acc = val_acc
            self.best_model_state = copy.deepcopy(model.state_dict())
            # self.best_model.load_status_dict(model.state_dict())

        elif val_acc < self.best_acc - self.min_delta and epoch >= epoch_threshold and train_acc >= train_acc_threshold:
            self.counter += 1
            self.restore_patients += 1
            if self.counter > self.patience:
                self.status = f"Stopped on {self.counter}"
                self.early_stop = True

            if self.restore_best_weights and self.restore_patients >= 5 : #10 for 0.82
                model.load_state_dict(self.best_model_state)
                print("best weight restored, best acc: ", self.best_acc)
                self.restore_patients = 0
                
        self.status = (f"{self.counter}/{self.patience}")

