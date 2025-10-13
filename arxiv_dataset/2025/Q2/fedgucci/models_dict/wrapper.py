from torch import nn
import torch.nn.functional as F

class MoonWrapper(nn.Module):

    def __init__(self, base_model: nn.Module, use_proj_head = False, proj_dim = None):
        super(MoonWrapper, self).__init__()
        self.features = base_model
        self.repres_dim = base_model.classifier.in_features
        self.n_class = base_model.classifier.out_features
        self.use_proj_head = use_proj_head
        self.proj_dim = proj_dim

        if use_proj_head:
            self.l1 = nn.Linear(self.repres_dim, self.repres_dim // 2)
            self.l2 = nn.Linear(self.repres_dim // 2, self.proj_dim)
            self.classifier = nn.Linear(self.proj_dim, self.n_class)
        else:
            self.classifier = nn.Linear(self.repres_dim, self.n_class).to(self.features.classifier.weight.device)

        # remove the classifier of the original model
        self.features.classifier = nn.Sequential()

    def forward(self, x, return_features=False):
        h = self.features(x)
        if self.use_proj_head:
            h = self.l1(h)
            h = F.relu(h)
            h = self.l2(h)
        out = self.classifier(h)

        if return_features:
            return out, h
        else:
            return out
