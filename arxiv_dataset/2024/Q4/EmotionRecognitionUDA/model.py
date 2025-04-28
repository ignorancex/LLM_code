import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class Model():
    def Generator(self, pixelda=False):
        return Feature()

    def Classifier(self, num_classes):
            return Predictor(num_classes)

class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * (-1.0))

#Feature Extractor
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.fc1 = nn.Linear(160, 128)
        self.bn1_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2_fc = nn.BatchNorm1d(64)

    def forward(self, x, reverse=False):
        if reverse:
          x = GradReverse.apply(x)
        x = x.view(x.size(0), x.size(1)*x.size(2))
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x

#Classifier
class Predictor(nn.Module):
    def __init__(self, num_classes, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.bn1_fc = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)
        self.bn2_fc = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, num_classes)
        self.bn_fc3 = nn.BatchNorm1d(num_classes)
        self.fc4 = nn.Softmax(dim=1)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = GradReverse.apply(x)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x_prev = self.fc3(x)
        x = self.fc4(x_prev)
        return x, x_prev
