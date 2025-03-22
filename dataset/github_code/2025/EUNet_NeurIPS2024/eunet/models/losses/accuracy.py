from collections import defaultdict
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ACCURACY
from eunet.datasets.utils.cloth3d import GARMENT_TYPE
from mmcv.ops import QueryAndGroup
from .compare_loss import cmp_error
from eunet.utils import vertex_normal_batched_simple


def accuracy_numpy(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.
    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.
    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x)
                 if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res

def accuracy_mse(pred, label, indices, indices_type, type_names, prefix='', reduction='sum', default_key='total', overwrite_align=False, vert_mask=None, **kwargs):
    '''
        indices: bs, max(n_outfit) + 1
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    # element-wise losses
    mse_error = []
    for i in range(bs):
        m_e = F.mse_loss(pred[i], label[i], reduction='none')
        if vert_mask is not None:
            m_e *= vert_mask[i]
        mse_error.append(m_e)

    # Calculate per outfit error
    for b_error, b_ind, b_type in zip(mse_error, indices, indices_type):
        if reduction == 'sum':
            acc_dict[default_key].append(b_error[0])
            for i in range(1, b_ind.shape[0]):
                assert b_type.shape[-1] == 1, "Only support idx input instead of one hot"
                g_type_idx = b_type[i-1, 0].int()
                g_type = type_names[g_type_idx]
                g_error = b_error[0]
                acc_dict[g_type].append(g_error)
        else:
            for i in range(1, b_ind.shape[0]):
                g_type_idx = torch.argmax(b_type[i-1], dim=0)
                g_type = type_names[g_type_idx]
                start, end = b_ind[i-1], b_ind[i]
                g_error = torch.mean(b_error[start:end])
                acc_dict[g_type].append(g_error)
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    if reduction != 'sum' and not overwrite_align:
        # Align the length of keys
        for g_type in type_names:
            key = f"{prefix}.{g_type}"
            if key not in acc_dict.keys():
                acc_dict[key] = torch.zeros(1).cuda()

    return acc_dict

def accuracy_l2(pred, label, indices, indices_type, prefix='', **kwargs):
    '''
        indices: bs, max(n_outfit) + 1
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    # element-wise losses
    square_error = [
        torch.sqrt(torch.sum(
            F.mse_loss(pred[i], label[i], reduction='none'), 
            dim=-1)) 
        for i in range(bs)]

    # Calculate per outfit error
    for b_error, b_ind, b_type in zip(square_error, indices, indices_type):
        for i in range(1, b_ind.shape[0]):
            g_type_idx = torch.argmax(b_type[i-1], dim=0)
            g_type = GARMENT_TYPE[g_type_idx]
            start, end = b_ind[i-1], b_ind[i]
            g_error = torch.mean(b_error[start:end])
            acc_dict[g_type].append(g_error)
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    # Align the length of keys
    for g_type in GARMENT_TYPE:
        key = f"{prefix}.{g_type}"
        if key not in acc_dict.keys():
            acc_dict[key] = torch.zeros(1).cuda()

    return acc_dict

def accuracy_collision_count(query_mesh, anchor_mesh, anchor_normals, grouper, max_dist=0.5, **kwargs):
    '''
        indices: bs, max(n_outfit) + 1
    '''
    anchor_mesh = anchor_mesh.contiguous()
    query_mesh = query_mesh.contiguous()
    grouped_results = grouper(anchor_mesh, query_mesh, anchor_normals)
    grouped_normals, grouped_xyz = grouped_results
    grouped_diff = query_mesh.transpose(1, 2).unsqueeze(-1) - grouped_xyz  # relative offsets
    grouped_normals = grouped_normals.permute(0, 2, 3, 1)
    grouped_diff = grouped_diff.permute(0, 2, 3, 1)
    grouped_diff_l2 = torch.sqrt(torch.sum(grouped_diff**2, dim=-1))

    dot = torch.sum(grouped_diff * grouped_normals, dim=-1)
    valid_mask = grouped_diff_l2 <= max_dist
    dot *= valid_mask
    collision_mask = dot < 0
    assert collision_mask.shape[-1] == 1
    collision_mask = collision_mask.squeeze(-1)
    num_collisions = torch.sum(collision_mask, dim=-1, keepdim=True)

    return num_collisions

def accuracy_collision(garment_mesh, indices, indices_type, faces, h_state, h_faces, grouper, prefix='', eps=1e-7, **kwargs):
    acc_dict = defaultdict(list)
    garment_mesh = garment_mesh.unsqueeze(0)
    indices = indices.unsqueeze(0)
    indices_type = indices_type.unsqueeze(0)
    faces = faces.unsqueeze(0)
    h_state = h_state[:, :3].unsqueeze(0)
    h_faces = h_faces.unsqueeze(0)
    faces = faces

    human_vert_normals = vertex_normal_batched_simple(h_state, h_faces)

    # Gamrent to human
    g2h_collision_count = accuracy_collision_count(garment_mesh, h_state, human_vert_normals.transpose(-1, -2).contiguous(), grouper)
    garment_verts_num = torch.tensor([i[-1] for i in indices]).reshape(garment_mesh.shape[0], -1).cuda()
    g2h_collision_rate = g2h_collision_count / garment_verts_num
    acc_dict['garment2human'].append(g2h_collision_rate)
    # Per Garment
    for b_ind, b_type in zip(indices, indices_type):
        for i in range(1, b_ind.shape[0]):
            g_type_idx = torch.argmax(b_type[i-1], dim=0)
            g_type = GARMENT_TYPE[g_type_idx]
            start, end = b_ind[i-1], b_ind[i]
            sub_g_mesh = garment_mesh[:, start:end]
            sub_g2h_collision_count = accuracy_collision_count(sub_g_mesh, h_state, human_vert_normals.transpose(-1, -2).contiguous(), grouper)
            sub_garment_verts_num = torch.tensor([i[-1] for i in indices]).reshape(sub_g_mesh.shape[0], -1).cuda()
            sub_g2h_collision_rate = sub_g2h_collision_count / sub_garment_verts_num
            acc_dict[g_type].append(sub_g2h_collision_rate)

    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }

    return acc_dict

def accuracy_compare_energy(pred, label, indices, indices_type, prefix='', cmp_key=None, hop_mask=None, weight=None, reduction='sum', min_diff=1e-8, eps=1e-7, **kwargs):
    '''
        indices: bs, max(n_outfit) + 1
        from loss/compare_loss.py
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    # element-wise losses

    for b_pred, b_label, b_hopmask, b_ind, b_type in zip(pred, label, hop_mask, indices, indices_type):
        for i in range(1, b_ind.shape[0]):
            g_type_idx = torch.argmax(b_type[i-1], dim=0)
            g_type = GARMENT_TYPE[g_type_idx]
            start, end = b_ind[i-1], b_ind[i]
            g_pred = b_pred[start:end]
            g_label = b_label[start:end]
            if b_hopmask is not None:
                g_hopmask = b_hopmask[start:end]
            else:
                g_hopmask = None
            g_error = cmp_error(g_pred, g_label, cmp_key=cmp_key, hop_mask=g_hopmask, weight=weight, reduction=reduction, min_diff=min_diff, eps=eps)[0]
            acc_dict[g_type].append(g_error)
    
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }

    return acc_dict


@ACCURACY.register_module()
class MSEAccuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l2',
                 ratio=False):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(MSEAccuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name
        self.ratio = ratio

    def forward(self, pred, target, indices, indices_type=None, type_names=[], overwrite_align=False, vert_mask=None, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_mse(pred, target, indices, indices_type=indices_type, type_names=type_names, prefix=self.acc_name, reduction=self.reduction, overwrite_align=overwrite_align, vert_mask=vert_mask, ratio=self.ratio, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name

@ACCURACY.register_module()
class L2Accuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l2'):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(L2Accuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name

    def forward(self, pred, target, indices, indices_type=None, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_l2(pred, target, indices, indices_type=indices_type, prefix=self.acc_name, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name

@ACCURACY.register_module()
class CollisionAccuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_collision'):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(CollisionAccuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name
        self.grouper = QueryAndGroup(
            max_radius=None,
            sample_num=1,
            min_radius=0,
            use_xyz=False,
            normalize_xyz=False,
            return_grouped_xyz=True,
            return_grouped_idx=False,
            return_unique_cnt=False,
        )

    def forward(self, pred, target, indices, indices_type, faces, h_state, h_faces, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        bs = len(pred)
        rst_dict_list = [
            accuracy_collision(
                pred[i], indices[i], indices_type[i], faces[i], h_state[i], h_faces[i],
                grouper=self.grouper, prefix=self.acc_name, **kwargs) 
                for i in range(bs)]
        rst_dict = defaultdict(list)
        for r_dict in rst_dict_list:
            for key, val in r_dict.items():
                rst_dict[key].append(val)
        for key in rst_dict.keys():
            val = torch.stack(rst_dict[key])
            ## -1 is invalid
            valid_val = val[torch.where(val != -1)]
            if valid_val.size(0) > 0:
                valid_val = torch.mean(valid_val)
            else:
                valid_val = val[0] # -1 here
            rst_dict[key] = valid_val

        return rst_dict
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name
    
@ACCURACY.register_module()
class CompareAccuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_cmp'):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(CompareAccuracy, self).__init__()
        self.reduction = reduction

        self._acc_name = acc_name

    def forward(self, pred, target, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_compare_energy(
            pred, target, prefix=self.acc_name, reduction=self.reduction, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name