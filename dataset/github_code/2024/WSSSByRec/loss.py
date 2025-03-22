import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconLoss(nn.Module):
    def __init__(self, L=1):
        super().__init__()
        self.L = L
    
    def forward(self, x, y_img, y_mask):
        num_classes = y_mask.shape[1]
        y_mask = y_mask.unsqueeze(2)
        loss = 0
        for _ in range(self.L):
            # reparameterization of categorical distribution
            # sm = F.gumbel_softmax(y_mask, dim=1, hard=True)
            sm = F.softmax(y_mask, dim=1)
            # compose the image from pixelets and masks
            x_hat = torch.sum(sm * y_img, dim=1)
            loss += F.mse_loss(x_hat, x)
        loss /= self.L
        return loss

class MaskRegLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
    
    def forward(self, label, y_mask):
        y_mask = F.softmax(y_mask, dim=1)
        h, w = y_mask.shape[-2:]
        y_pred = 1/(h*w) * torch.sum(y_mask, dim=(-2, -1))
        loss = torch.sum(-label*torch.log(y_pred + self.eps) - (1-label)*torch.log(1-y_pred + self.eps))
        loss /= label.shape[0]
        return loss

class MaskRegLossV2(nn.Module):
    def __init__(
        self,
        num_classes,
        alpha=0.01,
        beta=3,
        epsilon=1e-6
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
    
    def get_y_ngwp(self, y, m):
        y_ngwp = (
            torch.sum(m*y, dim=(-2, -1)) /
            torch.sum(m, dim=(-2, -1)) + self.epsilon
        )
        return y_ngwp

    def get_y_size_focal(self, m):
        h, w = m.shape[-2:]
        m_b = 1/(h*w)*torch.sum(m, dim=(-2, -1))
        y_size_focal = (
            (1 - m_b)**self.beta * torch.log(self.alpha + m_b)
        )
        return y_size_focal
    
    def forward(self, label, y_mask):
        y = y_mask
        m = F.softmax(y_mask, dim=1)
        y_ngwp = self.get_y_ngwp(y, m)
        y_size = self.get_y_size_focal(m)
        y_pred = y_ngwp + y_size
        loss = F.binary_cross_entropy(F.sigmoid(y_pred), label)
        return loss

class ClsGuideLoss(nn.Module):
    def __init__(self, classifier, L=1, eps=1e-6):
        super().__init__()
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        self.classifier = classifier
        self.L = L
        self.eps = eps
    
    def forward(self, label, y_img, y_mask):
        num_classes = y_img.shape[1]
        y_mask = y_mask.unsqueeze(2)
        loss_all = 0

        for _ in range(self.L):
            # reparameterization of categorical distribution
            # sm = F.gumbel_softmax(y_mask, dim=1, hard=True)
            sm = F.softmax(y_mask, dim=1)

            loss = 0
            for i in range(num_classes):
                # label num_classes-1 is the background class
                if i == num_classes-1:
                    x_hat = sm[:, i] * y_img[:, i]
                    y_pred = self.classifier(x_hat)
                    # y_pred = torch.clamp(y_pred, self.eps, 1-self.eps)
                    y_true = torch.zeros_like(y_pred)
                    loss += F.binary_cross_entropy(y_pred, y_true)
                else:
                    x_hat = sm[:, i] * y_img[:, i]
                    y_pred = self.classifier(x_hat)
                    # y_pred = torch.clamp(y_pred, self.eps, 1-self.eps)
                    y_true = torch.zeros_like(y_pred)
                    y_true[:, i] = label[:, i]
                    loss += F.binary_cross_entropy(y_pred, y_true)
            
            loss /= num_classes
            loss_all += loss
        
        loss_all /= self.L
        return loss_all
