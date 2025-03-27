import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
# from torchvision import models

from torchvision.models import resnet34

from cdac_loss import BCE_softlabels, advbce_unlabeled, sigmoid_rampup
from evaluation import prediction

import pdb

from lightly import loss
from lightly import transforms


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


class ProtoClassifier(nn.Module):
    def __init__(self, size):
        super(ProtoClassifier, self).__init__()
        self.center = None
        self.label = None
        self.size = size

    def init(self, model, t_loader):
        t_pred, t_feat = prediction(t_loader, model)
        label = t_pred.argmax(dim=1)
        center = torch.nan_to_num(
            torch.vstack([t_feat[label == i].mean(dim=0) for i in range(self.size)])
        )
        invalid_idx = center.sum(dim=1) == 0
        if invalid_idx.any() and self.label is not None:
            old_center = torch.vstack(
                [t_feat[self.label == i].mean(dim=0) for i in range(self.size)]
            )
            center[invalid_idx] = old_center[invalid_idx]
        else:
            self.label = label
        self.center = center.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, T=1.0):
        dist = torch.cdist(x, self.center)
        return F.softmax(-dist * T, dim=1)


class ResBase(nn.Module):
    def __init__(self, backbone="resnet34", **kwargs):
        super(ResBase, self).__init__()
        # self.res = models.__dict__[backbone](**kwargs)
        self.res = resnet34(**kwargs)
        self.last_dim = self.res.fc.in_features
        self.res.fc = nn.Identity()

    def forward(self, x):
        return self.res(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_classes=65, temp=0.05):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.temp = temp

    def forward(self, x, reverse=False):
        x = self.get_features(x, reverse=reverse)
        return self.get_predictions(x)

    def get_features(self, x, reverse=False):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x)
        return F.normalize(x) / self.temp

    def get_predictions(self, x):
        return self.fc2(x)

class Classifier_vanilla(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_classes=65, temp=0.05):
        super(Classifier_vanilla, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.temp = temp

    def forward(self, x, reverse=False):
        x = self.get_features(x, reverse=reverse)
        return self.get_predictions(x)

    def get_features(self, x, reverse=False):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        if reverse:
            x = grad_reverse(x)
        return F.normalize(x) / self.temp

        # return F.normalize(x)

    def get_predictions(self, x):
        return self.fc2(x)


class ResModel(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        hidden_dim=512,
        output_dim=65,
        temp=0.05,
        pre_trained=True,
    ):
        super(ResModel, self).__init__()
        self.f = ResBase(
            backbone=backbone,
            # weights=models.__dict__[backbone][f"ResNet{backbone[6:]}_Weights"].DEFAULT
            pretrained=True
            if pre_trained
            else None,
        )
        self.c = Classifier(self.f.last_dim, hidden_dim, output_dim, temp)
        init_weights(self.c)

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.bce = BCE_softlabels()



    def forward(self, x, reverse=False):
        return self.c(self.f(x), reverse)

    def get_params(self, lr):
        params = []
        for k, v in dict(self.f.named_parameters()).items():
            if v.requires_grad:
                if "classifier" not in k:
                    params += [{"params": [v], "base_lr": lr * 0.1, "lr": lr * 0.1}]
                else:
                    params += [{"params": [v], "base_lr": lr, "lr": lr}]
        params += [{"params": self.c.parameters(), "base_lr": lr, "lr": lr}]
        return params

    def get_features(self, x, reverse=False):
        return self.c.get_features(self.f(x), reverse=reverse)

    def get_predictions(self, x):
        return self.c.get_predictions(x)

    def base_loss(self, x, y):
        return self.criterion(self.forward(x), y).mean()

    def feature_base_loss(self, f, y):
        return self.criterion(self.get_predictions(f), y).mean()

    def sla_loss(self, f, y1, y2, alpha):
        out = self.get_predictions(f)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def nl_loss(self, f, y, alpha, T):
        out = self.get_predictions(f)
        y2 = F.softmax(out.detach() * T, dim=1)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y)
        soft_loss = -(y2 * log_softmax_out).sum(dim=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def mme_loss(self, _, x, lamda=0.1):
        out = self.forward(x, reverse=True)
        out = F.softmax(out, dim=1)
        return lamda * torch.mean(torch.sum(out * (torch.log(out + 1e-10)), dim=1))

    def cdac_loss(self, step, x, x1, x2):
        w_cons = 30 * sigmoid_rampup(step, 2000)
        f = self.f(x)
        f1 = self.f(x1)
        f2 = self.f(x2)

        out = self.c(f, reverse=True)
        out1 = self.c(f1, reverse=True)

        prob, prob1 = F.softmax(out, dim=1), F.softmax(out1, dim=1)
        aac_loss = advbce_unlabeled(
            target=None, f=f, prob=prob, prob1=prob1, bce=self.bce
        )

        out = self.c(f)
        out1 = self.c(f1)
        out2 = self.c(f2)

        prob, prob1, prob2 = (
            F.softmax(out, dim=1),
            F.softmax(out1, dim=1),
            F.softmax(out2, dim=1),
        )
        mp, pl = torch.max(prob.detach(), dim=1)
        mask = mp.ge(0.95).float()

        pl_loss = (F.cross_entropy(out2, pl, reduction="none") * mask).mean()
        con_loss = F.mse_loss(prob1, prob2)

        return aac_loss + pl_loss + w_cons * con_loss


class ResDec(nn.Module):
    def __init__(
        self,
        input_dim=512,
        temp=0.05,
    ):
        super(ResDec, self).__init__()

        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2) 
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0) 

        self.fc4 = nn.Linear(input_dim, 512)
        self.fc_bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

        # self.c = Classifier(self.f.last_dim, hidden_dim, output_dim, temp)
        # init_weights(self.c)

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.bce = BCE_softlabels()

    def forward(self, x, reverse=False):
        out = self.relu(self.fc_bn4(self.fc4(x)))
        out = self.relu(self.fc_bn5(self.fc5(out))).view(-1, 64, 4, 4)
        out = self.convTrans6(out)
        out = self.convTrans7(out)
        out = self.convTrans8(out)
        out = F.interpolate(out, size=(224, 224), mode='bilinear')
        return out

        # return self.c(self.f(x), reverse)

class SENNModel(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        hidden_dim=512,
        output_dim=65,
        temp=0.05,
        pre_trained=True,
    ):
        super(SENNModel, self).__init__()
        self.f = ResBase(
            backbone=backbone,
            pretrained=True
            if pre_trained
            else None,
        )
        self.relev = ResBase(
            backbone=backbone,
            pretrained=True
            if pre_trained
            else None,
        )
        self.res_dec = ResDec()

        # self.c = Classifier(output_dim, hidden_dim, output_dim, temp)
        self.c = Classifier(hidden_dim, hidden_dim, output_dim, temp)
        init_weights(self.c)

        self.concept_map = Classifier_vanilla(hidden_dim, hidden_dim, hidden_dim, temp)
        # init_weights(self.concept_map)

        self.relev_map = nn.Sequential(nn.Linear(1,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))

        self.concept_to_pred = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim,hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim,output_dim))

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.recontruction_criterion = nn.MSELoss(reduction="none")
        self.bce = BCE_softlabels()

        self.sslcriterion = loss.NTXentLoss(temperature=0.5)

    def forward(self, x, reverse=False):
        # pdb.set_trace()
        concepts = self.concept_map(self.f(x))
        sufficient_output = self.concept_to_pred(concepts)
        relevances = self.relev_map(self.relev(x).unsqueeze(-1))
        recons = self.res_dec(concepts)
        # aggregated = torch.bmm(relevances.permute(0,2,1),concepts.unsqueeze(-1))
        aggregated = torch.mul(relevances.squeeze(),concepts)
        return self.c(aggregated.squeeze(), reverse), concepts, relevances, recons, sufficient_output
        # return aggregated.squeeze(), concepts, relevances, recons, sufficient_output

    def get_params(self, lr):
        params = []
        for k, v in dict(self.f.named_parameters()).items():
            if v.requires_grad:
                if "classifier" not in k:
                    params += [{"params": [v], "base_lr": lr * 0.1, "lr": lr * 0.1}]
                else:
                    params += [{"params": [v], "base_lr": lr, "lr": lr}]

        for k, v in dict(self.relev.named_parameters()).items():
            if v.requires_grad:
                if "classifier" not in k:
                    params += [{"params": [v], "base_lr": lr * 0.1, "lr": lr * 0.1}]
                else:
                    params += [{"params": [v], "base_lr": lr, "lr": lr}]

        for k, v in dict(self.res_dec.named_parameters()).items():
            if v.requires_grad:
                if "classifier" not in k:
                    params += [{"params": [v], "base_lr": lr * 0.1, "lr": lr * 0.1}]
                else:
                    params += [{"params": [v], "base_lr": lr, "lr": lr}]
        
        params += [{"params": self.concept_map.parameters(), "base_lr": lr, "lr": lr}]
        params += [{"params": self.relev_map.parameters(), "base_lr": lr, "lr": lr}]
        params += [{"params": self.c.parameters(), "base_lr": lr, "lr": lr}]
        return params

    def get_features(self, x, reverse=False):
        return self.c.get_features(self.f(x), reverse=reverse)

    def get_predictions(self, x):
        return self.c.get_predictions(x)

    def base_loss(self, x, y):
        aggregates, concepts, relevances, reconstructions, sufficient_output = self.forward(x)
        # pdb.set_trace()
        return self.criterion(aggregates, y).mean() + 0*self.criterion(sufficient_output, y).mean()+ 1e-5*torch.norm(concepts,p=1, dim=1).mean() +1.0*self.recontruction_criterion(x,reconstructions).mean()

    def base_source_loss(self, x, y):
        aggregates, concepts, relevances, reconstructions, sufficient_output = self.forward(x)
        # pdb.set_trace()
        return ( 1 * self.criterion(aggregates, y).mean()  
                + 1 * self.criterion(sufficient_output, y).mean()
                + 1e-5 * torch.norm(concepts,p=1, dim=1).mean() 
                + 1.0 * self.recontruction_criterion(x,reconstructions).mean())

    def base_target_loss(self, x, y):
        aggregates, concepts, relevances, reconstructions, sufficient_output = self.forward(x)
        # pdb.set_trace()
        return (self.criterion(aggregates, y).mean()  
                + 1 * self.criterion(sufficient_output, y).mean()
                + 1e-5 * torch.norm(concepts,p=1, dim=1).mean() 
                + 1.0 * self.recontruction_criterion(x,reconstructions).mean())


    def ssl_loss(self, x1, x2):
        aggregates, concepts1, relevances, reconstructions, sufficient_output = self.forward(x1)
        aggregates, concepts2, relevances, reconstructions, sufficient_output = self.forward(x2)
        return self.sslcriterion(concepts1, concepts2).mean()
    

    def feature_base_loss(self, f, y):
        return self.criterion(self.get_predictions(f), y).mean()


    def sla_loss(self, f, y1, y2, alpha):
        out = self.get_predictions(f)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def nl_loss(self, f, y, alpha, T):
        out = self.get_predictions(f)
        y2 = F.softmax(out.detach() * T, dim=1)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y)
        soft_loss = -(y2 * log_softmax_out).sum(dim=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def mme_loss(self, _, x, lamda=0.1):
        out = self.forward(x, reverse=True)
        out = F.softmax(out, dim=1)
        return lamda * torch.mean(torch.sum(out * (torch.log(out + 1e-10)), dim=1))

    def cdac_loss(self, step, x, x1, x2):
        w_cons = 30 * sigmoid_rampup(step, 2000)
        f = self.f(x)
        f1 = self.f(x1)
        f2 = self.f(x2)

        out = self.c(f, reverse=True)
        out1 = self.c(f1, reverse=True)

        prob, prob1 = F.softmax(out, dim=1), F.softmax(out1, dim=1)
        aac_loss = advbce_unlabeled(
            target=None, f=f, prob=prob, prob1=prob1, bce=self.bce
        )

        out = self.c(f)
        out1 = self.c(f1)
        out2 = self.c(f2)

        prob, prob1, prob2 = (
            F.softmax(out, dim=1),
            F.softmax(out1, dim=1),
            F.softmax(out2, dim=1),
        )
        mp, pl = torch.max(prob.detach(), dim=1)
        mask = mp.ge(0.95).float()

        pl_loss = (F.cross_entropy(out2, pl, reduction="none") * mask).mean()
        con_loss = F.mse_loss(prob1, prob2)

        return aac_loss + pl_loss + w_cons * con_loss




class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd
        return output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)
