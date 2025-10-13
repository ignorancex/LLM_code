from abc import ABCMeta, abstractmethod
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from utils.neural_sort import soft_quantile


def build_preprocessor(preprocess, model, device, num_classes=None,freeze_num = 0):

    if preprocess == 'ts':
        preprocessor = TemperatureScaling(model, device)
    elif preprocess == 'vs':
        if num_classes is None:
            raise ValueError('num_classes cannot be None')
        preprocessor = VectorScaling(model, device, num_classes=num_classes,freeze_num = freeze_num)
    elif preprocess == 'conftr':
        if num_classes is None:
            raise ValueError('num_classes cannot be None')
        preprocessor = ConfTr(model, device, num_classes=num_classes,freeze_num = freeze_num)
    elif preprocess == 'ps':
        preprocessor = PlattScaling(model, device)
    else:
        raise NotImplementedError(f"The preprocessor: {preprocess} is not supported.")
    return preprocessor


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    The input to this loss is the logits of a model, NOT the softmax scores.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, softmax=True):
        if softmax:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class BaseCalibrator(nn.Module):
    __metaclass__ = ABCMeta
    

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, model):
        return NotImplementedError

    @abstractmethod
    def scaling(self, logits, softmax=False):
        return NotImplementedError





class TemperatureScaling(BaseCalibrator):
    def __init__(self,model, device):
        super().__init__()
        self.device = device
        self.temperature = nn.Parameter(torch.tensor([1.5]).to(self.device))
        self.model = model
        

    def train(self, valid_loader):

        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.1, max_iter=200)
        def eval():
            optimizer.zero_grad()
            out = logits / self.temperature
            loss = nll_criterion(out, labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        # Calculate NLL and ECE after temperature scaling
        print(self.temperature)
        after_temperature_nll = nll_criterion(self.scaling(logits), labels).item()
        after_temperature_ece = ece_criterion(self.scaling(logits), labels).item()
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    def scaling(self, logits, softmax=False):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits / self.temperature.to(self.device))

        return logits / (self.temperature.to(self.device))


class PlattScaling(BaseCalibrator):
    def __init__(self,model,device):
        super().__init__()
        self.device = device
        self.a = nn.Parameter(torch.tensor([1.5]).to(device))
        self.b = nn.Parameter(torch.tensor([1.5]).to(device))
        self.model = model

    def train(self,  valid_loader):
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)


        ece_before = ece_criterion(logits, labels)
        print("ece_before: %.4f" % ece_before.item())

        optimizer = optim.LBFGS([self.a, self.b], lr=0.1, max_iter=100)

        def eval():
            optimizer.zero_grad()
            out = logits * self.a + self.b
            loss = nll_criterion(out, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        out = logits * self.a + self.b
        ece_after = ece_criterion(out, labels)
        print("ece_after: %.4f" % ece_after.item())


    def scaling(self, logits, softmax=False):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits * self.a + self.b)

        return logits * self.a + self.b


class VectorScaling(BaseCalibrator):
    def __init__(self,model, device, num_classes,freeze_num):
        super().__init__()
        self.device = device
        self.w = nn.Parameter((torch.ones(num_classes) * 1.5).to(self.device))
        self.frozen_indices = torch.randperm(num_classes)[:freeze_num]
        
        self.b = nn.Parameter((torch.rand(num_classes) * 2.0 - 1.0).to(self.device))
        self.model = model

    def train(self, valid_loader):
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)


        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        optimizer = optim.LBFGS([self.w, self.b], lr=0.1, max_iter=300)
        def eval():
            optimizer.zero_grad()
            out = logits * self.w + self.b
            loss = nll_criterion(out, labels)
            loss.backward()
            self.w.grad[self.frozen_indices] = 0 
            return loss
        
        optimizer.step(eval)
        after_temperature_nll = nll_criterion(self.scaling(logits), labels).item()
        after_temperature_ece = ece_criterion(self.scaling(logits), labels).item()
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    def scaling(self, logits, softmax=False):
        if softmax:
            softmax_layer = nn.Softmax(dim=1)
            return softmax_layer(logits * self.w + self.b)

        return logits * self.w + self.b


class ConfTr(BaseCalibrator):
    def __init__(self, model, device, num_classes, freeze_num):
        super().__init__()
        self.device = device
        self.num_epochs = 50
        self.projection_weight = nn.Parameter(torch.eye(num_classes, device=device))
        self.projection_bias = nn.Parameter(torch.zeros(num_classes, device=device))
        self.model = model
        self.alpha = 0.01


    def train(self, valid_loader):
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam([self.projection_weight, self.projection_bias], lr=0.001)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            for input, label in valid_loader:
                input, label = input.to(self.device), label.to(self.device)
                logits = self.model(input)
                out = torch.matmul(logits, self.projection_weight.T) + self.projection_bias
                val_tau = self.smooth_calibrate_fn(out, label)
                loss = 0.00001*self.smooth_predict_fn(out, val_tau).mean() + nll_criterion(out, label)
                loss.backward()
            optimizer.step()


    def scaling(self, logits, softmax=False):
        if softmax:
            softmax_layer = nn.Softmax(dim=1)
            return softmax_layer(torch.matmul(logits, self.projection_weight.T) + self.projection_bias)
        return torch.matmul(logits, self.projection_weight.T) + self.projection_bias
    
    def _q(self, n, alpha):
        if alpha is None:
            alpha = self.alpha
        q = math.ceil((n + 1) * (1 - alpha)) / n
        if q > 1:
            return 1.0
        else:
            return q
         
    def smooth_calibrate_fn(self, logits, labels):
        n = logits.shape[0]
        log_probabilities = F.softmax(logits,dim=1)
        conformity_scores = log_probabilities[
        torch.arange(log_probabilities.shape[0]), labels]
        threshold = soft_quantile(-conformity_scores, self._q(n, self.alpha))
        return threshold
    
    def smooth_predict_fn(self, logits, tau):
        log_probabilities = F.softmax(logits,dim=1)
        membership_logits = tau - (-log_probabilities)
        membership_scores = torch.sigmoid(membership_logits/0.1)
        return torch.clamp(membership_scores.sum(-1) - 1, min=0) + 1
    









    

