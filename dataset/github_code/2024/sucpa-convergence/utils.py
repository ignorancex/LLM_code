

import torch
from torch import nn
import numpy as np

dataset2numclasses = {
    "sst2": 2,
    "mnli": 3,
    "dogs-vs-cats": 2,
}

def load_data(dataset, prefix="", split="train"):
    rs = np.random.RandomState(92893)
    logits = np.load(f'data/{dataset}/{prefix}{split}_logits.npy')
    labels = np.load(f'data/{dataset}/{prefix}{split}_labels.npy')
    if split == "train":
        idx = rs.choice(logits.shape[0], size=1000, replace=False)
        logits = torch.from_numpy(logits[idx])
        labels = torch.from_numpy(labels[idx])
    else:
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
    return logits, labels

def compute_metric(logits, labels, metric, bootstrap=True, n_boots=100, random_state=0):
    
    def _compute_metric(logits, labels, metric):
        n = logits.shape[0]
        if metric == "cross_entropy":
            logits = torch.log_softmax(logits, dim=1)
            return -torch.mean(logits[torch.arange(n), labels])
        elif metric == "norm_cross_entropy":
            logits = torch.log_softmax(logits, dim=1)
            log_priors = torch.log(torch.bincount(labels, minlength=logits.shape[1]).float() / n)
            return torch.mean(logits[torch.arange(n), labels]) / torch.mean(log_priors[labels])
        elif metric == "accuracy":
            return (logits.argmax(dim=1) == labels).float().mean()
        else:
            raise ValueError(f"Metric {metric} not supported.")
        
    if bootstrap:
        rs = np.random.RandomState(random_state)
        n = logits.shape[0]
        metrics = []
        for _ in range(n_boots):
            idx = rs.choice(n, size=n, replace=True)
            metrics.append(_compute_metric(logits[idx], labels[idx], metric).item())
        return metrics
    
    return [_compute_metric(logits, labels, metric).item()]

class SUCPA(nn.Module):
    
    def __init__(self, num_classes, steps=10, beta_init=None):
        super().__init__()
        self.num_classes = num_classes
        self.steps = steps
        if beta_init is not None:
            if beta_init.shape != (num_classes,):
                raise ValueError(f"beta_init must be of shape ({num_classes},), but got {beta_init.shape}")
            self.beta = nn.Parameter(beta_init, requires_grad=False)
        else:
            self.beta = nn.Parameter(torch.zeros(num_classes), requires_grad=False)
        self.jacobian = torch.zeros(num_classes, num_classes)
        self.beta_history = None
        self.jacobian_history = None

    def fit(self, logits, class_samples):
        beta_history = [self.beta.data]
        jacobian_history = [self.jacobian]
        for i in range(self.steps):
            log_den = torch.logsumexp(logits + self.beta, dim=1)
            log_sum_all = torch.logsumexp(logits.T - log_den, dim=1)
            self.beta.data = -log_sum_all + torch.log(class_samples)
            beta_history.append(self.beta.data)
            
            probs = torch.softmax(logits, dim=1)
            exp_beta = torch.exp(self.beta)
            den = probs @ exp_beta.unsqueeze(1)
            probs_den = probs / den
            self.jacobian = (probs_den.T @ torch.softmax(logits + self.beta, dim=1)) / probs_den.sum(dim=0).unsqueeze(1)
            jacobian_history.append(self.jacobian)
        self.beta_history = torch.stack(beta_history)
        self.jacobian_history = torch.stack(jacobian_history)
        return self
    
    def calibrate(self, logits):
        return logits + self.beta
    
    def save_to_disk(self, results_dir):
        state_dict = self.state_dict()
        state_dict["num_classes"] = torch.tensor(self.num_classes)
        state_dict["steps"] = torch.tensor(self.steps)
        state_dict["beta_history"] = self.beta_history
        state_dict["jacobian_history"] = self.jacobian_history
        torch.save(state_dict, f'{results_dir}/model.pt')

    @classmethod
    def load_from_disk(cls, results_dir):
        state_dict = torch.load(f'{results_dir}/model.pt')
        model = cls(num_classes=state_dict["num_classes"].item(), steps=state_dict["steps"].item())
        model.load_state_dict(state_dict, strict=False)
        model.beta_history = state_dict["beta_history"]
        model.jacobian_history = state_dict["jacobian_history"]
        return model



