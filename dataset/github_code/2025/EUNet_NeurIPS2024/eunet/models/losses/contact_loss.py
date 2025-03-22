import torch
import torch.nn as nn

from mmcv.ops import QueryAndGroup, grouping_operation
from eunet.utils import face_normals_batched

from ..builder import LOSSES
from .utils import weight_reduce_loss
from eunet.core import multi_apply
from eunet.datasets.utils.hood_common import gather


def contact_error(pred, state, h_state, h_faces, frame_dim, grouper, thresh, eps=1e-7, **kwargs):
    x_t1 = pred[:, :3]
    x_t0 = state[:, :3]

    h_t1 = h_state[:, :3]
    h_t0 = h_state[:, frame_dim:frame_dim+3]

    h_face_pos = gather(h_t1, h_faces, 0, 1, 1).mean(dim=-2)
    h_face_prev_pos = gather(h_t0, h_faces, 0, 1, 1).mean(dim=-2)
    h_face_normal = face_normals_batched(h_t1[None, :], h_faces[None, :])[0]
    
    grouped_results = grouper(h_face_prev_pos.unsqueeze(0).contiguous(), x_t0.unsqueeze(0).contiguous())
    _, grouped_idx = grouped_results
    _bs, _nverts, _nsample = grouped_idx.shape
    assert _bs == 1

    grouped_pos = grouping_operation(h_face_pos[None, :].transpose(-1, -2), grouped_idx).permute(0, 2, 3, 1)[0]
    grouped_normal = grouping_operation(h_face_normal[None, :].transpose(-1, -2), grouped_idx).permute(0, 2, 3, 1)[0]
    
    xyz_diff = ((x_t1[:, None] - grouped_pos) * grouped_normal).sum(dim=-1, keepdim=True)
    interpenetration = torch.maximum(thresh - xyz_diff, torch.FloatTensor([0]).to(xyz_diff.device))
    loss = interpenetration.pow(3)
    loss = torch.sum(loss, dim=-2)
    return loss

def contact_loss(
        pred, state, h_state, h_faces, frame_dim, grouper, thresh,
        weight=None, reduction='mean', avg_factor=None, eps=1e-7, **kwargs):
    loss = contact_error(pred, state, h_state, h_faces, frame_dim, grouper, thresh, eps=eps, **kwargs)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    assert not torch.isnan(loss)

    return loss,


@LOSSES.register_module()
class ContactLoss(nn.Module):
    def __init__(self,
                 thresh=0.001,
                 radius=None,
                 min_radius=0.0,
                 sample_num=8,
                 use_xyz=True,
                 normalize_xyz=False,
                 return_grouped_xyz=False,
                 return_grouped_idx=True,
                 return_unique_cnt=False,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_contact',
                 max_loss=None,
                 weight_start=5e3, weight_max=5e5, start_rampup_iteration=50000, n_rampup_iterations=100000):
        super(ContactLoss, self).__init__()
        self.thresh = thresh
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.grouper = QueryAndGroup(
            radius,
            sample_num,
            min_radius=min_radius,
            use_xyz=use_xyz,
            normalize_xyz=normalize_xyz,
            return_grouped_xyz=return_grouped_xyz,
            return_grouped_idx=return_grouped_idx,
            return_unique_cnt=return_unique_cnt,
        )

        self.criterion = contact_loss
        self._loss_name = loss_name

        self.weight_start = weight_start
        self.weight_max = weight_max
        self.start_rampup_iteration = start_rampup_iteration
        self.n_rampup_iterations = n_rampup_iterations

        self.max_loss = max_loss

    def _get_weight(self, num_iter):
        num_iter = num_iter - self.start_rampup_iteration
        num_iter = max(num_iter, 0)
        progress = num_iter / self.n_rampup_iterations
        progress = min(progress, 1.)
        weight = self.weight_start + (self.weight_max - self.weight_start) * progress
        return weight

    def forward(self,
                cls_score,
                label,
                state, h_state, h_faces,
                frame_dim=6,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                num_iter=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = multi_apply(
            self.criterion,
            cls_score, state,
            h_state, h_faces,
            frame_dim=frame_dim,
            grouper=self.grouper, thresh=self.thresh,
            weight=weight, reduction=reduction, avg_factor=avg_factor,
            **kwargs)[0]
        loss = torch.stack(loss)
        if self.max_loss is not None:
            loss = loss.clamp_max(self.max_loss)
        loss = loss.mean()
        loss *= self.loss_weight
        if num_iter is not None:
            progress_weight = self._get_weight(num_iter)
            loss *= progress_weight
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
