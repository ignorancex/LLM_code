import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math

from config import cfg

class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(
        self,
        tau: float = 0.6,
        change_epoch: int = 1,
        margin: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits - self.margin, logits)

        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1,dtype=targets.dtype).cuda(), targets)

        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight
        #loss *= pt

        return loss.sum(), targets
    
class SPLC_WAN(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(
        self,
        tau: float = 0.6,
        change_epoch: int = 1,
        margin: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits - self.margin, logits)

        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1,dtype=targets.dtype).cuda(), targets)

        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred)**self.gamma * targets + (1/109) * (1 - targets)
        #focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        #loss *= focal_weight
        loss *= pt

        return loss.sum(), targets
    
class GRLoss(nn.Module):

    def __init__(
            self,
            beta: list = [0,2,-2,-2],
            alpha: list = [0.5,2,0.8,0.5],
            q: list = [0.01,1],
    ) -> None:
        super(GRLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.q = q

    def neg_log(self,x):
        return (- torch.log(x + 1e-7))

    def loss1(self,x,q):
        return (1 - torch.pow(x, q)) / q

    def loss2(self,x,q):
        return (1 - torch.pow(1-x, q)) / q
    
    def K_function(self,preds,epoch):
        w_0,w_max,b_0,b_max=self.beta
        w=w_0+(w_max-w_0)*epoch/(cfg.epochs)
        b=b_0+(b_max-b_0)*epoch/(cfg.epochs)
        return 1 / (1 + torch.exp(-(w * preds + b)))
    
    def V_function(self,preds,epoch):
        mu_0,sigma_0,mu_max,sigma_max=self.alpha
        mu=mu_0+(mu_max-mu_0)*epoch/(cfg.epochs)
        sigma=sigma_0+(sigma_max-sigma_0)*epoch/(cfg.epochs)
        return torch.exp(-0.5 * ((preds - mu) / sigma) ** 2)  
    
    def forward(self,preds : torch.Tensor,label : torch.Tensor, epoch):
        preds = torch.sigmoid(preds)
        K = self.K_function(preds,epoch)
        V = self.V_function(preds,epoch)
        q2,q3=self.q
        loss_mtx = torch.zeros_like(preds)
        loss_mtx[label == 1]=self.neg_log(preds[label == 1])
        loss_mtx[label == 0]=V[label == 0]*(K[label == 0]*self.loss1(preds[label == 0],q2)+(1-K[label == 0])*self.loss2(preds[label == 0],q3))
        main_loss=loss_mtx.sum()
        return main_loss, label
    

class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss. 

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2

    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        return loss.sum(), targets
    
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y, epoch):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum(), y
    
class WAN(nn.Module):

    def __init__(
        self,
        gamma: float = (1/109),
    ) -> None:
        super(WAN, self).__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """

        los_pos = targets * F.logsigmoid(logits)
        los_neg = self.gamma * (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)

        return loss.sum(), targets
    
class iWAN(nn.Module):

    def __init__(
        self,
        gamma: float = -(1/109),
        p: float = 0.5
    ) -> None:
        super(iWAN, self).__init__()
        self.gamma = gamma
        self.p = p
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        if epoch < 2:
            return self.bce(logits,targets), targets

        los_pos = targets * F.logsigmoid(logits)
        los_neg = self.gamma * (1/self.p) * (1 - targets) * (torch.sigmoid(logits)**self.p)

        loss = - (los_pos - los_neg)

        return 0.5*loss.sum(), targets
    
class G_AN(nn.Module):

    def __init__(
        self,
        gamma: float = -(1/109),
        q: float = 0.5
    ) -> None:
        super(G_AN, self).__init__()
        self.gamma = gamma
        self.q = q
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        if epoch < 2:
            return self.bce(logits,targets), targets

        los_pos = targets * F.logsigmoid(logits)
        los_neg = self.gamma * (1/self.q) * (1 - targets) * (1-(1-torch.sigmoid(logits))**self.q)
        loss = -(los_pos - los_neg)

        return 0.5*loss.sum(), targets

class VLPL_Loss(nn.Module):

    def __init__(
        self,
        theta: float = 0.3,
        delta: float = 0.1,
        alpha: float = 0.2,
        beta: float = 0.7,
        gamma : float = 0.0,
        rho1: float = 0.9,
        rho2 : float = 0.1, 
        num_classes: int = 110,
        warmup_epoch: int = 0
    ) -> None:
        super(VLPL_Loss, self).__init__()
        self.theta = theta
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho1 = rho1
        self.rho2 = rho2
        self.ncls = num_classes
        self.warmup_epoch = warmup_epoch
        # self.count = 0
    
    def neg_log(self,v):
        return - torch.log(v + 1e-7)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch):

        preds = torch.sigmoid(logits)
        pseudolabels = torch.zeros_like(preds).cuda()
        pseudolabels = torch.where(preds > self.theta,
                torch.tensor(1,dtype=pseudolabels.dtype).cuda(), pseudolabels)

        k = int(self.delta * self.ncls)
        if k > 0:
            _, lowest_k_indices = torch.topk(preds, k, largest=False)
            row_indices = torch.full(lowest_k_indices.shape,-1,dtype=pseudolabels.dtype).cuda()
            pseudolabels.scatter_(1,lowest_k_indices,row_indices)

        pseudo_pos_mask = (pseudolabels == 1).float()
        pseudo_neg_mask = (pseudolabels == -1).float()
        pseudo_unk_mask = (pseudolabels == 0).float()

        loss_positive = targets * self.neg_log(preds)
        loss_neg = - (1-targets) * self.alpha * (preds*self.neg_log(preds)+ (1-preds)*self.neg_log(1-preds))
        loss_pseudounk = - (1-targets) * pseudo_unk_mask * self.alpha * (preds*self.neg_log(preds)+ (1-preds)*self.neg_log(1-preds))
        loss_pseudopos = (1-targets) * pseudo_pos_mask * self.beta * ((1-self.rho1)*self.neg_log(1-preds)+self.rho1*self.neg_log(preds))
        loss_pseudoneg = (1-targets) * pseudo_neg_mask * self.gamma * ((1-self.rho2)*self.neg_log(1-preds)+self.rho2*self.neg_log(preds))

        if epoch > self.warmup_epoch:
            loss = loss_positive + loss_pseudounk + loss_pseudopos + loss_pseudoneg
        else:
            loss = loss_positive + loss_neg

        return loss.sum(), targets

class Modified_VLPL(nn.Module):

    def __init__(
        self,
        alpha: float = 0.2,
        num_classes: int = 110
    ) -> None:
        super(Modified_VLPL, self).__init__()
        self.alpha = alpha
        self.ncls = num_classes
    
    def neg_log(self,v):
        return - torch.log(v + 1e-7)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch):

        preds = torch.sigmoid(logits)

        loss_positive = targets * self.neg_log(preds)
        loss_neg = - (1-targets) * self.alpha * (preds*self.neg_log(preds)+ (1-preds)*self.neg_log(1-preds))

        loss = loss_positive + loss_neg

        return loss.sum(), targets
    
class LL(nn.Module):

    def __init__(
        self,
        delta_rel: float = 0.05,
        scheme: str = 'LL-Ct'
    ) -> None:
        super(LL, self).__init__()
        self.scheme = scheme
        self.delta_rel = delta_rel

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        unobserved_mask = (targets == 0)

        assert torch.min(targets) >= 0
        loss_matrix = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(targets).float(), reduction='none')

        if epoch == 0:
            final_loss_matrix = loss_matrix
        else:
            clean_rate = 1 - epoch*self.delta_rel
            k = math.ceil(batch_size * num_classes * (1-clean_rate))
        
            unobserved_loss = unobserved_mask.bool() * loss_matrix
            topk = torch.topk(unobserved_loss.flatten(), k)
            topk_lossvalue = topk.values[-1]
            if self.scheme == 'LL-Ct':
                final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
            else:
                zero_loss_matrix = torch.zeros_like(loss_matrix)
                final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)

        main_loss = final_loss_matrix.mean()
        
        return main_loss, targets
    
class Weighted_Hill(nn.Module):

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Weighted_Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        self.task_w = self.get_task_weights(16112,6960,72815)
        assert(self.task_w.shape == (110,))

    def get_task_weights(self,n1,n2,n3):
        sigma_inv_ni = 1/n1 + 1/n2 + 1/n3
        w1 = (1/n1) / sigma_inv_ni
        w2 = (1/n2) / sigma_inv_ni
        w3 = (1/n3) / sigma_inv_ni
        t1 = torch.full((7,),w1)
        t2 = torch.full((3,),w2)
        t3 = torch.full((100,),w3)
        res = torch.cat((t1,t2,t3),dim=0)
        return res

    def forward(self, logits, targets, epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        task_w = self.task_w.to(logits.device)

        # Hill loss calculation
        los_pos = task_w * targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        return loss.sum(), targets