import torch
import torch.nn as nn
import math 

class HierarchicalLoss(nn.Module):
    def __init__(self,
                 config,
                 target_real_label=1.0,
                 target_fake_label=0.0):
        super(HierarchicalLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.criterion = nn.ReLU()
        self.alpha_inter = config["alpha_inter"]
        self.alpha_global = config["alpha_global"]

    def __call__(self, outputs, divider, is_real, is_disc=None):
        inter_feat, global_feat = outputs
        if is_disc:
            if is_real:
                inter_loss = self.criterion(1 - inter_feat).mean()
                global_loss = self.criterion(1 - global_feat).mean()
                return self.alpha_inter * inter_loss + self.alpha_global * global_loss
            else:
                global_loss = self.criterion(1 + global_feat).mean()
                batch, inter_loss = inter_feat.size(0), 0
                for b in range(batch):
                    thr = math.ceil(float(inter_feat.size(-1))/divider[b])
                    inter_feat1, inter_feat2 = inter_feat[b][..., :thr], inter_feat[b][..., -thr:]
                    inter_loss += (self.criterion(1 + inter_feat1).mean() + self.criterion(1 + inter_feat2).mean()) / 2
                return self.alpha_global * global_loss + self.alpha_inter * inter_loss / batch
        else:
            return (-global_feat).mean()
        

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """
    def __init__(self,
                 type='nsgan',
                 target_real_label=1.0,
                 target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label
                      if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss
