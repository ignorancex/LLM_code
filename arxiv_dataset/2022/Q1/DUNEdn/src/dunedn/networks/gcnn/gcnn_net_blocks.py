"""
    This module contains the GCNN Net building blocks.
"""
import torch
from torch import nn
from dunedn.networks.gcnn.gcnn_net_utils import (
    pairwise_dist,
    batched_index_select,
    local_mask,
)


class ROI(nn.Module):
    """U-net style binary segmentation"""

    def __init__(self, kernel_size, ic, hc, getgraph_fn, model):
        super(ROI, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.pre_process_block = PreProcessBlock(
            kernel_size, ic, hc, getgraph_fn, model
        )
        self.gcs = nn.ModuleList([choose_conv(model, hc, hc) for i in range(8)])
        self.gc_final = choose_conv(model, hc, 1)
        self.activ = nn.LeakyReLU(0.05)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.pre_process_block(x)
        for i, gc in enumerate(self.gcs):
            if i % 3 == 0:
                graph = self.getgraph_fn(x)
            x = self.activ(gc(x, graph))
        return self.act(self.gc_final(x, graph))


class PreProcessBlock(nn.Module):
    def __init__(self, kernel_size, ic, oc, getgraph_fn, model):
        ks = kernel_size
        kso2 = kernel_size // 2
        super(PreProcessBlock, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.activ = nn.LeakyReLU(0.05)
        self.convs = nn.Sequential(
            nn.Conv2d(ic, oc, ks, padding=(kso2, kso2)),
            self.activ,
            nn.Conv2d(oc, oc, ks, padding=(kso2, kso2)),
            self.activ,
            nn.Conv2d(oc, oc, ks, padding=(kso2, kso2)),
            self.activ,
        )
        self.bn = nn.BatchNorm2d(oc)
        self.gc = choose_conv(model, oc, oc)

    def forward(self, x):
        x = self.convs(x)
        graph = self.getgraph_fn(x)
        return self.activ(self.gc(x, graph))


class HPF(nn.Module):
    """High Pass Filter"""

    def __init__(self, ic, oc, getgraph_fn, model):
        super(HPF, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.conv = nn.Sequential(
            nn.Conv2d(ic, ic, 3, padding=1), nn.BatchNorm2d(ic), nn.LeakyReLU(0.05)
        )
        self.gcs = nn.ModuleList(
            [
                choose_conv(model, ic, ic),
                choose_conv(model, ic, oc),
                choose_conv(model, oc, oc),
            ]
        )
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.conv(x)
        graph = self.getgraph_fn(x)
        for gc in self.gcs:
            x = self.act(gc(x, graph))
        return x


class LPF(nn.Module):
    """Low Pass Filter"""

    def __init__(self, ic, oc, getgraph_fn, model):
        super(LPF, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.conv = nn.Sequential(
            nn.Conv2d(ic, ic, 5, padding=2), nn.BatchNorm2d(ic), nn.LeakyReLU(0.05)
        )
        self.gcs = nn.ModuleList(
            [
                choose_conv(model, ic, ic),
                choose_conv(model, ic, oc),
                choose_conv(model, oc, oc),
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(ic), nn.BatchNorm2d(oc), nn.BatchNorm2d(oc)]
        )
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        y = self.conv(x)
        graph = self.getgraph_fn(y)
        for bn, gc in zip(self.bns, self.gcs):
            y = self.act(bn(gc(y, graph)))
        return x + y


class PostProcessBlock(nn.Module):
    def __init__(self, ic, hc, getgraph_fn, model):
        super(PostProcessBlock, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.gcs = nn.ModuleList(
            [
                choose_conv(model, hc * 3 + 1, hc * 2),
                choose_conv(model, hc * 2, hc),
                choose_conv(model, hc, ic),
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(hc * 2), nn.BatchNorm2d(hc), nn.Identity()]
        )
        self.acts = nn.ModuleList(
            [nn.LeakyReLU(0.05), nn.LeakyReLU(0.05), nn.Identity()]
        )

    def forward(self, x):
        for act, bn, gc in zip(self.acts, self.bns, self.gcs):
            graph = self.getgraph_fn(x)
            x = act(bn(gc(x, graph)))
        return x


class NonLocalGraph:
    """Non-local graph layer."""

    def __init__(self, k, crop_size):
        """
        Parameters
        ----------
            - k: int, nearest neighbor number.
            - crop_size: tuple, (edge_h, edge_w)
        """
        self.k = k
        self.local_mask = local_mask(crop_size)

    def __call__(self, arr):
        """
        Parameters
        ----------
            - arr: torch.Tensor, input tensor of shape=(N,C,H,W)

        Returns
        -------
            - torch.Tensor, output tensor of shape=(N,H*W*K,C)
        """
        arr = arr.data.permute(0, 2, 3, 1)
        b, h, w, f = arr.shape
        arr = arr.view(b, h * w, f)
        hw = h * w
        dists = pairwise_dist(arr, self.k, self.local_mask)
        selected = batched_index_select(arr, 1, dists.view(dists.shape[0], -1)).view(
            b, hw, self.k, f
        )
        diff = arr.unsqueeze(2) - selected
        return diff


# ==============================================================================
# functions and classes to be called within this module only


def choose_conv(model, ic, oc):
    """
    Utility function to retrieve GConv or Conv layer from its name.

    Parameters
    ----------
        - model: str, available options cnn | cnn
        - ic: int, input channel dimension size
        - oc: int, output channel dimension size

    Returns
    -------
        - torch.nn.Module, the layer instance

    Raises
    ------
        - NotImplementedError if op is not in ['gcnn', 'cnn']
    """
    if model == "gcnn":
        return GConv(ic, oc)
    elif model == "cnn":
        return Conv(ic, oc)
    else:
        raise NotImplementedError("Operation not implemented")


class GConv(nn.Module):
    """GConv layer."""

    def __init__(self, ic, oc):
        """
        Parameters
        ----------
            - ic: int, input channel dimension size
            - oc: int, output channel dimension size
        """
        super(GConv, self).__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.nla = NonLocalAggregator(ic, oc)

    def forward(self, x, graph):
        return torch.mean(torch.stack([self.conv1(x), self.nla(x, graph)]), dim=0)


class Conv(nn.Module):
    """GConv layer."""

    def __init__(self, ic, oc):
        """
        Parameters
        ----------
            - ic: int, input channel dimension size
            - oc: int, output channel dimension size
        """
        super(Conv, self).__init__()

        self.conv1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.conv2 = nn.Conv2d(ic, oc, 5, padding=2)

    def forward(self, x, graph):
        return torch.mean(torch.stack([self.conv1(x), self.conv2(x)]), dim=0)


class NonLocalAggregator(nn.Module):
    """NonLocalAggregator layer."""

    def __init__(self, input_channels, out_channels):
        """
        Parameters
        ----------
            - input_channels: int, input channel dimension size
            - output_channels: int, output channel dimension size
        """
        super(NonLocalAggregator, self).__init__()
        self.diff_fc = nn.Linear(input_channels, out_channels)
        self.w_self = nn.Linear(input_channels, out_channels)
        # self.bias = nn.Parameter(torch.randn(out_channels), requires_grad=True)

    def forward(self, x, graph):
        """
        Parameters
        ----------
            - x: torch.Tensor, of shape=(N,C,H,W)
            - graph: torch.Tensor

        Returns
        -------
            - torch.Tensor of shape=(N,C,H,W)
        """
        x = x.permute(0, 2, 3, 1)
        b, h, w, f = x.shape
        x = x.view(b, h * w, f)

        # closest_graph = get_graph(x, self.k, local_mask) #this builds the graph
        agg_weights = self.diff_fc(graph)  # look closer
        agg_self = self.w_self(x)

        x_new = torch.mean(agg_weights, dim=-2) + agg_self  # + self.bias

        return x_new.view(b, h, w, x_new.shape[-1]).permute(0, 3, 1, 2)
