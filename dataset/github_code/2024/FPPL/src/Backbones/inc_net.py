import copy

import timm
import torch
import torch.nn.functional as F
from Backbones.linears import CosineLinear
from Backbones.prompt import FedPrompt
from torch import nn
from torchvision import transforms


def get_backbone(args):
    name = args.backbone.lower()
    if "_fppl" in name:
        if "fppl" in args.model:
            from Backbones import vision_transformer_fppl

            model = timm.create_model(
                args.backbone,
                pretrained=args.pretrained,
                num_classes=0,
                insert_layers=args.insert_layers,
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class FedPromptVitNet(nn.Module):
    def __init__(self, args):
        super(FedPromptVitNet, self).__init__()
        self.args = args
        insert_layers = getattr(args, "insert_layers", 5)
        args.insert_layers = insert_layers
        self.backbone = get_backbone(args)
        self.feature_dim = self.backbone.embed_dim
        self.fc = nn.Linear(self.feature_dim, args.n_class)
        self.prompt = FedPrompt(
            n_task=args.n_task,
            embed_dim=self.feature_dim,
            num_layers=args.insert_layers,
        )
        self.prompt_fc = CosineLinear(self.feature_dim, args.n_task)
        self.prompt_fc_tasks = []

    def forward(self, x, cur_task=-1, train=False):
        out = dict()
        q = self.query(x)
        weights = self.qweights(q)
        fused_prompt = self.prompt(weights, cur_task, train=train)
        features = self.backbone(x, prompt=fused_prompt)
        out["weights"] = weights
        out["features"] = features
        out["logits"] = self.fc(features)
        return out

    def query(self, x):
        with torch.no_grad():
            q = self.backbone(x)
        return q

    def qweights(self, q):
        logits = self.prompt_fc(q)
        class_mask = [
            col_index
            for col_index in range(logits.shape[1])
            if col_index not in self.prompt_fc_tasks
        ]
        logits[:, class_mask] = float("-inf")

        weights = F.softmax(logits, dim=1)

        if self.args.wo_fusion:
            max_indices = torch.argmax(logits, dim=1)
            weights = torch.zeros_like(logits)
            weights[torch.arange(logits.size(0)), max_indices] = 1

        return weights

    def extract_vector(self, x):
        with torch.no_grad():
            out = self.backbone(x)
        return out
