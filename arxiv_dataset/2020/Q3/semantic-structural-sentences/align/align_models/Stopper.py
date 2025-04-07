import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, saver=False):
        """
        :param patience: How many steps to wait since improvement.
        :param verbose: If True, prints each validation loss.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.saver = saver

    def __call__(self, val_loss, model):
        """
        :param val_loss: An instance of validation loss (float).
        :param model: The model being trained.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {c} out of {p}'.format(c=self.counter, p=self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        :param val_loss: An instance of validation loss (float).
        :param model: The model being trained.
        """
        if self.verbose:
            print('Validation loss decreased ({a} --> {b}). Saving model.'.format(a=self.val_loss_min,
                                                                                  b=val_loss))
        if self.saver:
            torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
