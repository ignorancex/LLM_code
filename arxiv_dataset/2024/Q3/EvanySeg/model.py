import os.path
import torch
import warnings
from torchvision.models import resnet50, resnet101

import copy
import timm
import argparse
import numpy as np
from pathlib import Path
import torch.nn as nn
from monai.metrics import compute_hausdorff_distance
from transformers import ViTForImageClassification, ViTModel


class ViT_ImageClassification(nn.Module):
    def __init__(self, model_path, flag):
        super(ViT_ImageClassification, self).__init__()
        model = ViTForImageClassification.from_pretrained(model_path)  # google/vit-base-patch16-224

        self.vit = list(model.children())[0]
        self.flag = flag
        if "large" in model_path:
            self.classifier = nn.Linear(1024, 1)
        else:
            self.classifier = nn.Linear(768, 1)
        if self.flag == 2:
            if "large" in model_path:
                self.hd_classifier = nn.Linear(1024, 1)
            else:
                self.hd_classifier = nn.Linear(768, 1)

    def forward(self, x):

        output = self.vit(pixel_values=x)
        sequence_output = output[0]
        if self.flag == 1:
            logits_dice = self.classifier(sequence_output[:, 0, :])
            return {"logits_dice": logits_dice}
        else:
            logits_dice = self.classifier(sequence_output[:, 0, :])
            logits_hd = self.hd_classifier(sequence_output[:, 0, :])
            return {"logits_dice": logits_dice, "logits_hd": logits_hd}


class ResNet_ImageClassification(nn.Module):
    def __init__(self, model, flag):
        super(ResNet_ImageClassification, self).__init__()
        self.backbone = list(model.children())[:-1]
        self.backbone += [nn.Flatten()]
        self.backbone = nn.Sequential(*self.backbone)

        self.flag = flag
        self.dice_classifier = nn.Linear(2048, 1)
        if self.flag == 2:
            self.hd_classifier = nn.Linear(2048, 1)

    def forward(self, x):
        output = self.backbone(x)
        if self.flag == 1:
            logits_dice = self.dice_classifier(output)
            return {"logits_dice": logits_dice}
        else:
            logits_dice = self.dice_classifier(output)
            logits_hd = self.hd_classifier(output)
            return {"logits_dice": logits_dice, "logits_hd": logits_hd}