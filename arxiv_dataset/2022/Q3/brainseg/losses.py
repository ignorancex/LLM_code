import torch
import torch.nn.functional as F
import numpy as np
import pdb


class BCE(torch.nn.Module):
    def __init__(self, num_classes,device,beta=0.7):
        super(BCE, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.beta=beta
        return

    def forward(self, pred, true_mask):
        pred = pred.permute((0, 2, 3, 1))
        pred = pred.reshape((-1, self.num_classes))
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        labels = true_mask.view(-1, 1)
        label_one_hot = torch.nn.functional.one_hot(torch.squeeze(labels), self.num_classes).float().to(self.device)
        term1 = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.beta)) / self.beta

        term2 = torch.sum(torch.pow(pred, self.beta + 1), dim=1) / (self.beta + 1)
        # term2 = torch.sum(torch.log(torch.pow(pred, self.beta + 1)), dim=1) / (self.beta + 1)
        # print(term1)
        # print(term2.size())
        bce = torch.mean(term1 + term2)
        #
        return bce


class BCE_Weighted(torch.nn.Module):
    def __init__(self, num_classes,device,beta=0.7, weights=None):
        super(BCE_Weighted, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.beta=beta
        self.weights = weights
        return

    def forward(self, pred, true_mask):
        pred = pred.permute((0, 2, 3, 1))
        pred = pred.reshape((-1, self.num_classes))
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        labels = true_mask.view(-1, 1)
        label_one_hot = torch.nn.functional.one_hot(torch.squeeze(labels), self.num_classes).float().to(self.device)
        term1 = torch.zeros_like(pred[:, 0])
        term2 = term1
        for c in range(self.num_classes):
            term1 = term1 + self.weights[c] * label_one_hot[:, c] * pred[:, c]

            term2 = term2 + self.weights[c] * torch.pow(pred[:, c], self.beta + 1)

        term1 = (1. - torch.pow(term1, self.beta)) / self.beta
        term2 = term2 / (self.beta + 1)
        # term2 = torch.sum(torch.log(torch.pow(pred, self.beta + 1)), dim=1) / (self.beta + 1)
        # print(term1)
        # print(term2.size())
        bce = torch.mean(term1 + term2)
        #
        return bce


class SCE(torch.nn.Module):
    def __init__(self, num_classes,device,alpha=1, beta=1):
        super(SCE, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.beta=beta
        self.alpha=alpha
        return

    def forward(self, pred, true_mask):


        pred = pred.permute((0, 2, 3, 1))
        pred = pred.reshape((-1, self.num_classes))
        pred = F.log_softmax(pred, dim=1)
        labels=true_mask.view(-1, 1)
        label_one_hot = torch.nn.functional.one_hot(torch.squeeze(labels), self.num_classes).float().to(self.device)
        ce = -1 * torch.sum(label_one_hot * pred, dim=1)

        # RCE
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce.mean() + self.beta * rce.mean()
        return loss


class GCE(torch.nn.Module):
    def __init__(self, num_classes,device,q=0.7):
        super(GCE, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.q=q

    def forward(self, pred, true_mask):

        pred = pred.permute((0, 2, 3, 1))
        pred = pred.reshape((-1, self.num_classes))
        pred = F.log_softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        labels = true_mask.view(-1, 1)
        label_one_hot = torch.nn.functional.one_hot(torch.squeeze(labels), self.num_classes).float().to(self.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1),self.q)) / self.q
        return gce.mean()

