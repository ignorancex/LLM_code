from losses import AngularPenaltySMLoss, BroadFaceArcFace, ArcFace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# This is Our model With Projection included with Loss cross entropy
class CCE_Model(nn.Module):
    def __init__(self, num_classes=10, projection=True):
        super(CCE_Model, self).__init__()
        self.projection = projection
        self.project = Projection()
        self.res50_model = models.resnet50(pretrained=True)
        self.res50_conv = nn.Sequential(
            *list(self.res50_model.children())[:-4])
        for param in self.res50_conv.parameters():
            param.requires_grad = True
        self.fatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(8192, 512)
        self.fc1 = nn.Linear(512, 256)
        if self.projection:
            self.fc2 = nn.Linear(257, num_classes, bias=False)
        else:
            self.fc2 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x, labels):
        x = self.res50_conv(x)
        x = self.fatten(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        if self.projection:
            print("Running With Projection")
            x = self.dropout(x)
            x = self.project(x)
            out = self.fc2(x)
        else:
            print("Running Without Projection")
            x = self.dropout(x)
            out = self.fc2(x)

        loss = F.cross_entropy(x, labels)
        return out, loss

# Custome Projection Class Function


class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()

    def forward(self, x):
        l = []
        for i in x:
            x_new = torch.zeros_like(torch.empty(1)).cuda()
            concated = torch.cat((i, x_new)).cuda()
            s = -torch.div((torch.tensor(1) - torch.sum(torch.square(i))),
                           (torch.tensor(1) + torch.sum(torch.square(i)))).cuda()
            basis = torch.cat((torch.zeros(len(i)), torch.ones(1))).cuda()
            proj = concated + (s * (basis - concated)).cuda()
            l.append(proj)
        finalproj = torch.stack(l).cuda()
        return finalproj

# This base Model with Projection Function For all Loss Types


class Backbone_Net(nn.Module):
    def __init__(self, projection=True):
        super(Backbone_Net, self).__init__()
        self.projection = projection
        self.project = Projection()
        self.res50_model = models.resnet50(pretrained=True)
        self.res50_conv = nn.Sequential(
            *list(self.res50_model.children())[:-4])
        for param in self.res50_conv.parameters():
            param.requires_grad = True
        self.fatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(8192, 512)
        self.fc1 = nn.Linear(512, 256)

    def forward(self, x, embed=False):
        x = self.res50_conv(x)
        x = self.fatten(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        if self.projection:
            x = self.dropout(x)
            x = self.project(x)
        return x

# This Model with Projection For loss  ['sphereface', 'cosface','broadface','arcface']


class Model(nn.Module):
    def __init__(self, num_classes=10, loss_type='sphereface', projection=True):
        super(Model, self).__init__()
        if projection:
            print("Running With Projection")
            self.backbone = Backbone_Net(projection=True)
            self.features = 257
        else:
            print("Running Without Projection")
            self.backbone = Backbone_Net(projection=False)
            self.features = 256

        if loss_type == 'sphereface':
            self.loss = AngularPenaltySMLoss(
                self.features, num_classes, loss_type='sphereface')
        elif loss_type == 'cosface':
            self.loss = AngularPenaltySMLoss(
                self.features, num_classes, loss_type='cosface')
        elif loss_type == 'broadface':
            self.loss = BroadFaceArcFace(
                self.features, num_classes, compensate=False)
        elif loss_type == 'arcface':
            self.loss = ArcFace(self.features, num_classes)
        else:
            raise ValueError(
                "Enter The Valid Loss: ['sphereface', 'cosface','broadface','arcface'] ")

    def forward(self, x, labels, embed=False):
        x = self.backbone(x)
        if embed:
            return x
        L = self.loss(x, labels)
        return L
