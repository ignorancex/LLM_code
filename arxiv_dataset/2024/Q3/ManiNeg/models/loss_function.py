# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
                  for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class HardNegative(nn.Module):
    def __init__(self, device, tau_plus, beta, temperature, estimator, trainable_temperature=False):
        super().__init__()

        self.device = device
        self.tau_plus = tau_plus
        self.beta = beta
        self.estimator = estimator
        if trainable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature, device=device), requires_grad=True)
        else:
            self.temperature = temperature

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size),device=self.device, dtype=torch.bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def forward(self, out_1, out_2, attr_uni=None):
        # neg score
        if attr_uni is not None:
            out_1 = out_1[attr_uni, :]
            out_2 = out_2[attr_uni, :]
        batch_size, _ = out_1.size()
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        # old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if self.estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        elif self.estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()

        return loss


class HardNegative_Inter(nn.Module):
    """
    For inter-modal loss, won't count intra-modal negative samples
    """

    def __init__(self, device, tau_plus, beta, temperature, estimator, trainable_temperature=False):
        super().__init__()

        self.device = device
        self.tau_plus = tau_plus
        self.beta = beta
        self.estimator = estimator
        if trainable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature, device=device), requires_grad=True)
        else:
            self.temperature = temperature

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, batch_size), device=self.device, dtype=torch.bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
        return negative_mask

    def forward(self, out_1, out_2, attr_uni=None):
        # neg score
        if attr_uni is not None:
            out_1 = out_1[attr_uni, :]
            out_2 = out_2[attr_uni, :]
        batch_size, _ = out_1.size()
        # out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out_1, out_2.t().contiguous()) / self.temperature)
        # old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size)
        neg = neg.masked_select(mask).view(batch_size, -1)
        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # negative samples similarity scoring
        if self.estimator == 'hard':
            N = batch_size - 1
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        elif self.estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()

        return loss


class ManiNeg_NT_Xent(nn.Module):
    def __init__(self, temperature, device, intra_modal_neg, label_smoothing=0.0):
        super().__init__()

        self.temperature = temperature
        self.device = device
        # for now, this is  set at __init__(), this can also be set during forward
        self.intra_modal_neg = intra_modal_neg

        self.criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)
        self.similarity_f = nn.CosineSimilarity(dim=1)

        self.labels = torch.zeros(2).long().to(device)

    def forward(self, z_i, z_j):

        # batch_size, _ = z_i.size()

        pos_i = z_i[0:1]
        pos_j = z_j[0:1]
        if self.intra_modal_neg:
            neg_i = z_i[1:]
            neg_j = z_j[1:]

            neg_cat_j = torch.cat([z_j, neg_i], dim=0)
            sim_i = self.similarity_f(pos_i, neg_cat_j) / self.temperature

            neg_cat_i = torch.cat([z_i, neg_j], dim=0)
            sim_j = self.similarity_f(pos_j, neg_cat_i) / self.temperature
        else:
            sim_i = self.similarity_f(pos_i, z_j) / self.temperature
            sim_j = self.similarity_f(pos_j, z_i) / self.temperature

        logits = torch.stack((sim_i, sim_j))
        loss = self.criterion(logits, self.labels)
        return loss


# 对比损失
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size, label_smoothing=0.0):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class NT_Xent_AttrUni(nn.Module):
    def __init__(self, temperature, device, label_smoothing=0.0, trainable_temperature=False):
        """

        Args:
            temperature: if trainable_temperature is True, it is the initial value of temperature
            device: device of parameters
            label_smoothing: label smoothing of criterion
            trainable_temperature: whether the tempterature is trainable.
        """
        super().__init__()
        if trainable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature, device=device), requires_grad=True)
        else:
            self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), device=self.device, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, attr_uni=None):
        if attr_uni is not None:
            z_i = z_i[attr_uni, :]
            z_j = z_j[attr_uni, :]
        batch_size, _ = z_i.size()

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class NT_Xent_AttrUni_Inter(nn.Module):
    """
    For inter-modal loss, won't count intra-modal negative samples
    """

    def __init__(self, temperature, device, label_smoothing=0.0, trainable_temperature=False):
        """

        Args:
            temperature: if trainable_temperature is True, it is the initial value of temperature
            device: device of parameters
            label_smoothing: label smoothing of criterion
            trainable_temperature: whether the tempterature is trainable.
        """
        super().__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)
        self.similarity_f = nn.CosineSimilarity(dim=2)
        if trainable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature, device=device), requires_grad=True)
        else:
            self.temperature = temperature

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((batch_size, batch_size), device=self.device, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        return mask

    def forward(self, z_i, z_j, attr_uni=None):
        if attr_uni is not None:
            z_i = z_i[attr_uni, :]
            z_j = z_j[attr_uni, :]
        batch_size, _ = z_i.size()

        sim = self.similarity_f(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature

        positive_samples = torch.diag(sim)[:, None]
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(batch_size, -1)

        labels = torch.zeros(batch_size).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        return loss


class Vikl_Loss(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size, ls_it, ls_ia, ls_ta, ls_ii):
        super().__init__()
        # fixme: temperature could be set seperately. for now, all loss are set with same temperature
        self.it_nt_xent = NT_Xent(batch_size, temperature, device, world_size, ls_it)
        self.ia_nt_xent = NT_Xent(batch_size, temperature, device, world_size, ls_ia)
        self.ta_nt_xent = NT_Xent(batch_size, temperature, device, world_size, ls_ta)
        self.ii_nt_xent = NT_Xent(batch_size, temperature, device, world_size, ls_ii)

    def forward(self, img_z1, img_z2, text_z, attr_z):
        # do not need "+ self.nt_xent(text_z, img_z1)", this is just a doubler
        loss_i_t = self.it_nt_xent(img_z1, text_z)  # + self.nt_xent(text_z, img_z1)
        loss_i_a = self.ia_nt_xent(img_z1, attr_z)  # + self.nt_xent(attr_z, img_z1)
        loss_t_a = self.ta_nt_xent(text_z, attr_z)  # + self.nt_xent(attr_z, text_z)

        # todo: notice the ratio here, inter is 1/3 against intra
        loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / 3
        loss_intra_modal = self.ii_nt_xent(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t.item(), 'loss_i_a': loss_i_a.item(), 'loss_t_a': loss_t_a.item(),
                    'loss_intra_modal': loss_intra_modal.item(), 'loss_sum': loss.item()}
        return loss, met_dict


class Vikl_HardNeg(nn.Module):
    def __init__(self, device, tau_plus, beta, tit, tia, tta, tii, etit, etia, etta, etii, **kwargs):
        super().__init__()
        # fixme: all params could be set seperately. for now, all loss are set with same param
        self.it = HardNegative(device, tau_plus, beta, tit, etit)
        self.ia = HardNegative(device, tau_plus, beta, tia, etia)
        self.ta = HardNegative(device, tau_plus, beta, tta, etta)
        self.ii = HardNegative(device, tau_plus, beta, tii, etii)

    def forward(self, img_z1, img_z2, text_z, attr_z, attr_uni):
        loss_i_t = self.it(img_z1, text_z)
        loss_i_a = self.ia(img_z1, attr_z, attr_uni)
        loss_t_a = self.ta(text_z, attr_z, attr_uni)

        loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / 3
        # todo: notice the ratio here, inter is 1/3 against intra

        loss_intra_modal = self.ii(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t.item(), 'loss_i_a': loss_i_a.item(), 'loss_t_a': loss_t_a.item(),
                    'loss_intra_modal': loss_intra_modal.item(), 'loss_sum': loss.item()}
        return loss, met_dict


class Vikl_ManiHardNeg(nn.Module):
    def __init__(self, device, temperature, **kwargs):
        super().__init__()
        # fixme: all params could be set seperately. for now, all loss are set with same param
        self.it = ManiNeg_NT_Xent(temperature, device, False, 0.2)  # HardNegative(device, tau_plus, beta, tit, etit)
        self.ia = ManiNeg_NT_Xent(temperature, device, False, 0.0)
        self.ta = ManiNeg_NT_Xent(temperature, device, False, 0.2)
        self.ii = ManiNeg_NT_Xent(temperature, device, True, 0.0)

    def forward(self, img_z1, img_z2, text_z, attr_z):
        loss_i_t = self.it(img_z1, text_z)
        loss_i_a = self.ia(img_z1, attr_z)
        loss_t_a = self.ta(text_z, attr_z)

        loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / 3
        # todo: notice the ratio here, inter is 1/3 against intra

        loss_intra_modal = self.ii(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t.item(), 'loss_i_a': loss_i_a.item(), 'loss_t_a': loss_t_a.item(),
                    'loss_intra_modal': loss_intra_modal.item(), 'loss_sum': loss.item()}
        return loss, met_dict


class Vikl_Loss_AttrUni(nn.Module):
    def __init__(self, temperature, device, ls_it, ls_ia, ls_ta, ls_ii, trainable_temperature=False):
        super().__init__()
        # fixme: temperature could be set seperately. for now, all loss are set with same temperature
        self.it_nt_xent = NT_Xent_AttrUni(temperature, device, ls_it, trainable_temperature)
        self.ia_nt_xent = NT_Xent_AttrUni(temperature, device, ls_ia, trainable_temperature)
        self.ta_nt_xent = NT_Xent_AttrUni(temperature, device, ls_ta, trainable_temperature)
        self.ii_nt_xent = NT_Xent_AttrUni(temperature, device, ls_ii, trainable_temperature)

    def forward(self, img_z1, img_z2, text_z, attr_z, attr_uni):
        # do not need "+ self.nt_xent(text_z, img_z1)", this is just a doubler
        loss_i_t = self.it_nt_xent(img_z1, text_z)  # + self.nt_xent(text_z, img_z1)
        loss_i_a = self.ia_nt_xent(img_z1, attr_z, attr_uni)  # + self.nt_xent(attr_z, img_z1)
        loss_t_a = self.ta_nt_xent(text_z, attr_z, attr_uni)  # + self.nt_xent(attr_z, text_z)

        # todo: notice the ratio here, inter is 1/3 against intra
        loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / 3
        loss_intra_modal = self.ii_nt_xent(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t.item(), 'loss_i_a': loss_i_a.item(), 'loss_t_a': loss_t_a.item(),
                    'loss_intra_modal': loss_intra_modal.item(), 'loss_sum': loss.item()}
        return loss, met_dict


class Vikl_Loss_Inter_AttrUni(nn.Module):
    # no intra neg for multi modal
    def __init__(self, temperature, device, ls_it, ls_ia, ls_ta, ls_ii, trainable_temperature=False):
        super().__init__()
        # fixme: temperature could be set seperately. for now, all loss are set with same temperature
        self.it_nt_xent = NT_Xent_AttrUni_Inter(temperature, device, ls_it, trainable_temperature)
        self.ia_nt_xent = NT_Xent_AttrUni_Inter(temperature, device, ls_ia, trainable_temperature)
        self.ta_nt_xent = NT_Xent_AttrUni_Inter(temperature, device, ls_ta, trainable_temperature)
        self.ii_nt_xent = NT_Xent_AttrUni(temperature, device, ls_ii, trainable_temperature)

    def forward(self, img_z1, img_z2, text_z, attr_z, attr_uni):
        # do not need "+ self.nt_xent(text_z, img_z1)", this is just a doubler
        loss_i_t = self.it_nt_xent(img_z1, text_z)  # + self.nt_xent(text_z, img_z1)
        loss_i_a = self.ia_nt_xent(img_z1, attr_z, attr_uni)  # + self.nt_xent(attr_z, img_z1)
        loss_t_a = self.ta_nt_xent(text_z, attr_z, attr_uni)  # + self.nt_xent(attr_z, text_z)

        # todo: notice the ratio here, inter is 1/3 against intra
        loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / 3
        loss_intra_modal = self.ii_nt_xent(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t.item(), 'loss_i_a': loss_i_a.item(), 'loss_t_a': loss_t_a.item(),
                    'loss_intra_modal': loss_intra_modal.item(), 'loss_sum': loss.item()}
        return loss, met_dict


class Vikl_Fusion(nn.Module):
    def __init__(self, device, **kwargs):
        # FIXME: ALL FIXED HYPER-PARAMETERS
        # parameter list:
        # hn
        # tau_plus:0.1
        # beta:1.0
        # it: easy t=0.8
        # ia: hard t=0.8
        # ta: easy t=0.8
        # ii: hard t=0.5

        # parameter list:
        # ls
        # it: easy ls=0.2
        # ia: hard ia=0.3
        # ta: easy ta=0.2
        # ii: hard ii=0.0
        super().__init__()
        self.tau_plus = 0.1
        self.beta = 1.0
        self.it = NT_Xent_AttrUni(0.7, device, 0.2)
        self.ia = HardNegative(device, self.tau_plus, self.beta, 0.7, "hard")
        self.ta = NT_Xent_AttrUni(0.7, device, 0.2)
        self.ii = HardNegative(device, self.tau_plus, self.beta, 0.7, "hard")

    def forward(self, img_z1, img_z2, text_z, attr_z, attr_uni):
        loss_i_t = self.it(img_z1, text_z)
        loss_i_a = self.ia(img_z1, attr_z, attr_uni)
        loss_t_a = self.ta(text_z, attr_z, attr_uni)

        loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / 3
        # todo: notice the ratio here, inter is 1/3 against intra

        loss_intra_modal = self.ii(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t.item(), 'loss_i_a': loss_i_a.item(), 'loss_t_a': loss_t_a.item(),
                    'loss_intra_modal': loss_intra_modal.item(), 'loss_sum': loss.item()}
        return loss, met_dict


def dummyzero(*args):
    return 0.0


class Vikl_Loss_Ablation(nn.Module):
    def __init__(self, device, ls_it, ls_ia, ls_ta, ls_ii, **kwargs):
        super().__init__()
        # fixme: temperature could be set seperately. for now, all loss are set with same temperature
        self.tau_plus = 0.1
        self.beta = 1.0
        self.it = NT_Xent_AttrUni(0.7, device, 0.2) if ls_it else dummyzero
        self.ia = HardNegative(device, self.tau_plus, self.beta, 0.8, "hard") if ls_ia else dummyzero
        self.ta = NT_Xent_AttrUni(0.7, device, 0.2) if ls_ta else dummyzero
        self.ii = HardNegative(device, self.tau_plus, self.beta, 0.8, "hard") if ls_ii else dummyzero
        # self.it_nt_xent = NT_Xent_AttrUni(temperature, device, ls_it) if ls_it > 0 else dummyzero
        # self.ia_nt_xent = NT_Xent_AttrUni(temperature, device, ls_ia) if ls_ia > 0 else dummyzero
        # self.ta_nt_xent = NT_Xent_AttrUni(temperature, device, ls_ta) if ls_ta > 0 else dummyzero
        # self.ii_nt_xent = NT_Xent_AttrUni(temperature, device, ls_ii) if ls_ii > 0 else dummyzero
        self.numerator = ls_it + ls_ia + ls_ta

    def forward(self, img_z1, img_z2, text_z, attr_z, attr_uni):
        # do not need "+ self.nt_xent(text_z, img_z1)", this is just a doubler
        loss_i_t = self.it(img_z1, text_z)  # + self.nt_xent(text_z, img_z1)
        loss_i_a = self.ia(img_z1, attr_z, attr_uni)  # + self.nt_xent(attr_z, img_z1)
        loss_t_a = self.ta(text_z, attr_z, attr_uni)  # + self.nt_xent(attr_z, text_z)

        # todo: notice the ratio here, inter is 1/3 against intra
        if self.numerator > 0:
            loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / self.numerator
        else:
            loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a)
        loss_intra_modal = self.ii(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t.item(), 'loss_i_a': loss_i_a.item(), 'loss_t_a': loss_t_a.item(),
                    'loss_intra_modal': loss_intra_modal.item(), 'loss_sum': loss.item()}
        return loss, met_dict


class Vikl_Loss_Adaptive(nn.Module):
    """
    ViKL loss with adaptive input. If some modal is None, this loss automatically exclude it.
    The included losses are standard NT_Xent loss, and the hard negative are implemented via batch sampler.
    """

    def __init__(self, device, ii_enable, im_enable, it_enable, tm_enable, temperature, trainable_temperature,
                 **kwargs):
        super().__init__()
        # fixme: temperature could be set seperately. for now, all loss are set with same temperature

        self.it = NT_Xent_AttrUni_Inter(temperature, device, 0.2, trainable_temperature) if it_enable else dummyzero
        self.ia = NT_Xent_AttrUni_Inter(temperature, device, 0.0, trainable_temperature) if im_enable else dummyzero
        self.ta = NT_Xent_AttrUni_Inter(temperature, device, 0.2, trainable_temperature) if tm_enable else dummyzero
        self.ii = NT_Xent_AttrUni(temperature, device, 0.0, trainable_temperature) if ii_enable else dummyzero
        self.numerator = im_enable + it_enable + tm_enable

    def forward(self, img_z1, img_z2, text_z, attr_z, attr_uni):

        loss_i_t = 0.5 * (self.it(img_z1, text_z) + self.it(img_z2, text_z))
        loss_i_a = 0.5 * (self.ia(img_z1, attr_z, attr_uni) + self.ia(img_z2, attr_z, attr_uni))
        loss_t_a = self.ta(text_z, attr_z, attr_uni)

        # todo: notice the ratio here, inter is 1/3 against intra
        if self.numerator > 0:
            loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / self.numerator
        else:
            loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a)
        loss_intra_modal = self.ii(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t, 'loss_i_a': loss_i_a, 'loss_t_a': loss_t_a,
                    'loss_intra_modal': loss_intra_modal, 'loss_sum': loss}
        if hasattr(self.it, "temperature"):
            met_dict['temp_i_t'] = self.it.temperature
        if hasattr(self.ia, "temperature"):
            met_dict['temp_i_a'] = self.ia.temperature
        if hasattr(self.ta, "temperature"):
            met_dict['temp_t_a'] = self.ta.temperature
        if hasattr(self.ii, "temperature"):
            met_dict['temp_i_i'] = self.ii.temperature
        for k in met_dict:
            if isinstance(met_dict[k], torch.Tensor):
                met_dict[k] = met_dict[k].item()
        return loss, met_dict


class Vikl_Loss_Adaptive_HRW(nn.Module):
    """
    ViKL loss with adaptive input. If some modal is None, this loss automatically exclude it.
    The included losses are Hard Negative loss with reweight method.
    burned-in parameters.
    """

    def __init__(self, device, ii_enable, im_enable, it_enable, tm_enable, temperature, trainable_temperature,
                 **kwargs):
        super().__init__()
        # fixme: temperature could be set seperately. for now, all loss are set with same temperature
        # note: Assert !it_enable and !im_enable. They shouldn't sample hard negative
        assert (not it_enable) and (not tm_enable), ("Assert !it_enable and !im_enable. They shouldn't sample hard "
                                                     "negative")
        self.tau_plus = 0.1
        self.beta = 1.0
        self.it = dummyzero
        self.ia = HardNegative_Inter(device, self.tau_plus, self.beta, temperature, "hard",
                                     trainable_temperature) if im_enable else dummyzero
        self.ta = dummyzero
        self.ii = HardNegative(device, self.tau_plus, self.beta, temperature, "hard",
                               trainable_temperature) if ii_enable else dummyzero
        self.numerator = im_enable + it_enable + tm_enable

    def forward(self, img_z1, img_z2, text_z, attr_z, attr_uni):

        loss_i_t = 0.5 * (self.it(img_z1, text_z) + self.it(img_z2, text_z))
        loss_i_a = 0.5 * (self.ia(img_z1, attr_z, attr_uni) + self.ia(img_z2, attr_z, attr_uni))
        loss_t_a = self.ta(text_z, attr_z, attr_uni)

        # todo: notice the ratio here, inter is 1/3 against intra
        if self.numerator > 0:
            loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a) / self.numerator
        else:
            loss_inter_modal = (loss_i_a + loss_i_t + loss_t_a)
        loss_intra_modal = self.ii(img_z1, img_z2)

        loss = loss_inter_modal + loss_intra_modal
        met_dict = {'loss_i_t': loss_i_t, 'loss_i_a': loss_i_a, 'loss_t_a': loss_t_a,
                    'loss_intra_modal': loss_intra_modal, 'loss_sum': loss}
        if hasattr(self.it, "temperature"):
            met_dict['temp_i_t'] = self.it.temperature
        if hasattr(self.ia, "temperature"):
            met_dict['temp_i_a'] = self.ia.temperature
        if hasattr(self.ta, "temperature"):
            met_dict['temp_t_a'] = self.ta.temperature
        if hasattr(self.ii, "temperature"):
            met_dict['temp_i_i'] = self.ii.temperature
        for k in met_dict:
            if isinstance(met_dict[k], torch.Tensor):
                met_dict[k] = met_dict[k].item()
        return loss, met_dict
