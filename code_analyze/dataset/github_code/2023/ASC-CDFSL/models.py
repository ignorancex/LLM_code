import torch.nn as nn
import torch
import torch.nn.functional as F
from methods import backbone


class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type='softmax'):
        super(BaselineTrain, self).__init__()
        self.feature = model_func()

        if loss_type == 'softmax':
            self.classifier = nn.Linear(512, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        # self.top1 = utils.AverageMeter()

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        # self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))

        return self.loss_fn(scores, y), correct

    def test_loop(self, val_loader):
        return -1  # no validation, just save model during iteration

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 6272),
        )

        self.transconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.linear(input).view(-1, 32, 14, 14)
        output = self.transconv(output)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 6272),
        )

        self.transconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.linear(input).view(-1, 32, 14, 14)
        output = self.transconv(output)
        return output


class Delta_attn(nn.Module):
    def __init__(self):
        super(Delta_attn, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(784 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, tgt, src):
        input = torch.cat([tgt, src], dim=1)
        output = self.linear(input)
        output = F.softmax(output, dim=1)
        return output