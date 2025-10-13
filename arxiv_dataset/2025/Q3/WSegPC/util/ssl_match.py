import torch
from torch.nn import functional as F

def consistency_loss(logits, targets, mask=None, weight=None):
    loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')
    if mask is not None:
        loss = loss * mask

    return loss.mean()

class FreeMatch():

    def __init__(self, p_cutoff, classes=20, weight=False):
        super(FreeMatch, self).__init__() 
        self.p_cutoff = p_cutoff
        self.num_classes = classes
        self.weight = weight
        self.classwise_acc = torch.zeros((self.num_classes,))
        self.pseudo_counter = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def masking(self, probs_x_ulb, logits_x_ulb_w_w):
        if not self.classwise_acc.is_cuda:
            self.classwise_acc = self.classwise_acc.to(probs_x_ulb.device)
        if not self.pseudo_counter.is_cuda:
            self.pseudo_counter = self.pseudo_counter.to(probs_x_ulb.device)

        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1)
        if logits_x_ulb_w_w is not None:
            # print(logits_x_ulb_w_w.shape, max_probs.shape)
            mask = max_probs.ge(self.p_cutoff * logits_x_ulb_w_w * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx])))  # convex
        else:
            mask = max_probs.ge(self.p_cutoff * self.classwise_acc[max_idx])  # convex
        # select = max_probs.ge(self.p_cutoff)
        mask = mask.to(max_probs.dtype)
        
        # update
        hist = torch.bincount(max_idx[mask == 1].reshape(-1), minlength=self.pseudo_counter.shape[0]).to(self.pseudo_counter.dtype)
        self.pseudo_counter += hist
        self.classwise_acc = self.pseudo_counter / torch.max(self.pseudo_counter)

        return mask

    def train_step(self, logits_x_ulb_w, logits_x_ulb_s, logits_x_ulb_w_p=None, logits_x_ulb_w_w=None):

        probs_x_ulb_w = logits_x_ulb_w.detach()

        # compute mask
        mask = self.masking(probs_x_ulb_w, logits_x_ulb_w_w)
        pseudo_label = torch.argmax(probs_x_ulb_w, dim=-1)
        if logits_x_ulb_w_p is not None:
            max_probs_p, max_idx_p = torch.max(logits_x_ulb_w_p, dim=-1)
            mask[pseudo_label!=max_idx_p] = 0.0
        unsup_loss = consistency_loss(logits_x_ulb_s, pseudo_label, mask=mask)#, weight=weights)

        return unsup_loss