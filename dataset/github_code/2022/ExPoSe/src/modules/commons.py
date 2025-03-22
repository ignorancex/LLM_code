import torch as t

from .bgnn.lib.selfgnn.dyAggWe_tsp_coder import tsp_coder as tspmodel


class Conv(t.nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = t.nn.Conv2d(c1, c2, k, s, p)
        self.bn = t.nn.BatchNorm2d(c2)
        self.act = t.nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse(self):
        if hasattr(self, 'bn'):
            fusedconv = t.nn.Conv2d(self.conv.in_channels,
                                    self.conv.out_channels,
                                    kernel_size=self.conv.kernel_size,
                                    stride=self.conv.stride,
                                    padding=self.conv.padding,
                                    groups=self.conv.groups,
                                    bias=True).requires_grad_(False).to(self.conv.weight.device)

            # prepare filters
            w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
            w_bn = t.diag(self.bn.weight.div(t.sqrt(self.bn.eps + self.bn.running_var)))
            fusedconv.weight.data = t.mm(w_bn, w_conv).view(fusedconv.weight.size()).data

            # prepare spatial bias
            b_conv = t.zeros(self.conv.weight.size(0), device=self.conv.weight.device) if self.conv.bias is None else self.conv.bias
            b_bn = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(t.sqrt(self.bn.running_var + self.bn.eps))
            fusedconv.bias.data = (t.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).data

            self.conv = fusedconv

            delattr(self, 'bn')  # remove batchnorm
            self.forward = lambda x: self.act(self.conv(x))


class ResidualBlock(t.nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()

        self.cv1 = Conv(ch, ch, k=3, s=1, p=1)
        self.cv2 = Conv(ch, ch, k=3, s=1, p=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class BGNN(t.nn.Module):
    def __init__(self):
        super(BGNN, self).__init__()

        self.model = tspmodel(nodeFeature=8,
                              weightFeature=2,
                              with_global=True,
                              with_gnn_decode=True,
                              dropout=False)

    def forward(self, x):
        edges, features = x

        if len(features.shape) == 2:
            features = features.unsqueeze(0)
            edges = edges.unsqueeze(0)

        batch, n_nodes, n_features = features.shape
        features = t.transpose(features, 1, 2)
        idx = t.arange(0, n_nodes, device=features.device).reshape(1, 1, n_nodes).repeat(batch, n_nodes, 1)
        edges_weight = edges.unsqueeze(1)
        edges_weight = t.cat(((edges_weight == 1), (edges_weight == 0)), dim=1).float()

        return self.model(features, edges_weight, idx)
