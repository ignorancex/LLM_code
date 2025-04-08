import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from mmcv.cnn import normal_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from torch.nn import init
from config import *
FLAGS = flags.FLAGS
# Point_center  encode the segmented point cloud
# one more conv layer compared to original paper


class SizeHead(nn.Module):
    def __init__(self, in_dim):
        super(SizeHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 4
        self.feat_dim = FLAGS.feat_ts

        self.conv1 = torch.nn.Conv1d(self.in_dim, self.feat_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.feat_dim, self.out_dim, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(self.feat_dim)
        self._init_weights()

    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        # bs,1024,8,8
        x = x.flatten(2,3).max(dim=-1, keepdim=True).values
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)

        x = x.squeeze(2)
        x = x.contiguous()
        x1 = x[:, :3]
        x2 = x[:, 3]
        return x1, x2

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

def main(argv):
    feature = torch.rand(3, 3, 1000)
    obj_id = torch.randint(low=0, high=15, size=[3, 1])
    net = SizeHead()
    out = net(feature, obj_id)
    t = 1

if __name__ == "__main__":
    app.run(main)
