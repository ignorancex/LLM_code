import torch
from algs.base_alg import Base_alg
from data.tools import rotate_batch
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np


class GdScore(Base_alg):
    """
    GdScore
    """

    def evaluate(self):
        self.base_model.train()
        t = self.args["threshold"]
        score_list = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.base_model.parameters(),
                              lr=self.args['lr'],
                              momentum=0.9,
                              weight_decay=0.0)
        for batch_idx, batch_data in enumerate(self.val_loader):
            inputs, labels = batch_data[0], batch_data[1]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.base_model(inputs)
            with torch.no_grad():
                # pseudo-label by base_model
                outputs2 = nn.functional.softmax(outputs, dim=1)
                p, pseudo_labels = outputs2.max(1)
                pseudo_labels = pseudo_labels.detach()
                pseudo_labels[p<t] = torch.LongTensor(pseudo_labels[p<t].shape[0]).random_(0, self.args["num_classes"]).to(self.device)

            optimizer.zero_grad()
            loss = criterion(outputs, pseudo_labels)
            loss.backward()
            weight = self.base_model.fc.weight.grad
            score = torch.norm(weight, p=self.args["norm_type"])
            score_list.append(score)
        scores = torch.Tensor(score_list).numpy()
        return scores.mean()




