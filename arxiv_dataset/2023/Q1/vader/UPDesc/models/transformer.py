import logging
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.nnutils import get_norm

log = logging.getLogger(__name__)


class Transformer(nn.Module):
    """Estimate local rotations and scales
    """

    def __init__(
            self,
            in_channels,
            num_rotations,
            dim_rotations,
            num_scales,
            max_scale,
            min_scale,
            dropout=False,
            dropout_prob=0.2,
            weight_normalize=False,
            use_bn=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_rotations = num_rotations  # If <= 0, disable rotation prediction
        self.dim_rotations = dim_rotations
        self.num_scales = num_scales  # If <= 0, disable scale prediction
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.weight_normalize = weight_normalize
        self.use_bn = use_bn

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        def mlp(ich, och, sig):
            wn = lambda m: nn.utils.weight_norm(m) if weight_normalize else m
            norm1d = get_norm('batch_norm', '1d', trainable=False)

            seq = list()
            seq.append(wn(nn.Linear(ich, ich, bias=not use_bn)))
            if use_bn:
                seq.append(norm1d(ich))
            seq.append(nn.ReLU(True))
            seq.append(wn(nn.Linear(ich, ich, bias=not use_bn)))
            if use_bn:
                seq.append(norm1d(ich))
            seq.append(nn.ReLU(True))
            seq.append(wn(nn.Linear(ich, och)))
            if sig:
                seq.append(nn.Sigmoid())
            seq = nn.Sequential(*seq)
            # seq.apply(init_weights)
            return seq

        if num_rotations > 0:
            self.fcr = nn.ModuleList()
            for _ in range(num_rotations):
                self.fcr.append(mlp(in_channels, dim_rotations, False))
        else:
            self.fcr = None

        if num_scales > 0:
            self.fcs = mlp(in_channels, num_scales, True)
        else:
            self.fcs = None

        log.info('Transformer: in_channels - {}; Rotations - {}; Scales - {}'.format(
            in_channels, num_rotations, num_scales))

    def forward(self, x):
        # x: (B, C_l)

        # r: (B, R, 4/6)
        # s: (B, S)

        if self.fcr is not None:
            r = [module(x) for module in self.fcr]
            r = torch.stack(r, dim=1)
            if r.size(-1) == 4:  # Quaternion
                r = F.normalize(r, p=2, dim=-1)
        else:
            r = None

        if self.fcs is not None:
            s = self.fcs(x)
            s = s * (self.max_scale - self.min_scale) + self.min_scale
        else:
            s = None

        return r, s
