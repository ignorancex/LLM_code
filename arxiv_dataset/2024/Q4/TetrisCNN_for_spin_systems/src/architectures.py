import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Taken from PyTorch documentation:
    The parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:

    * a single `int` in which case the same value is used for the height and width dimension
    * a `tuple` of two ints - in which case, the first int is used for the height dimension, and the second int for the width dimension
"""


class ConvBranch_twoLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        kernel_size,
        number_of_filters,
        dilation=1,
        padding_size=0,
        padding_mode=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.hidden_size = 32
        self.number_of_filters = number_of_filters
        self.dilation = dilation

        if padding_mode is None:
            padding_mode = "zeros"
            padding_size = 0

        self.conv1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding_size,
            padding_mode=padding_mode,
            dilation=self.dilation,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.number_of_filters,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        # compute features (kxk convolution, then ReLU, then 1x1 convolution)
        f = self.conv2(F.relu(self.conv1(x)))

        # average 'pool' over the entire image
        f = torch.mean(torch.mean(f, dim=3), dim=2)
        return f


class ShapeAdaptiveConvNet(nn.Module):
    def __init__(
        self, input_dim, shape_list, device, padding_size=0, padding_mode=None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.shape_list = shape_list
        self.branch_num = len(shape_list)
        self.device = device

        self.branches = nn.ModuleList()
        for b in range(self.branch_num):
            shape_details = shape_list[b]
            try:
                self.branches.append(
                    ConvBranch_twoLayer(
                        input_dim,
                        shape_details[0],
                        shape_details[1],
                        (shape_details[2], 1),
                        padding_size=padding_size,
                        padding_mode=padding_mode,
                    )
                )
            except:
                self.branches.append(
                    ConvBranch_twoLayer(
                        input_dim,
                        shape_details[0],
                        shape_details[1],
                        padding_size=padding_size,
                        padding_mode=padding_mode,
                    )
                )

    def forward(self, x):
        all_filters = []
        for b in range(
            self.branch_num
        ):  # can likely be sped up with block sparse approach
            h = self.branches[b](x)
            all_filters.append(h)

        return torch.cat(all_filters, dim=1)  # need to adjust later for multiclass


class SmallModel(nn.Module):
    def __init__(self, sizes, device):
        super(SmallModel, self).__init__()

        self.sizes = sizes
        self.length = len(self.sizes) - 1
        self.activation = F.relu
        self.device = device

        self.hiddens = nn.ModuleList()
        for k in range(self.length):
            self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k + 1]))

    def forward(self, x):
        h = x
        for k in range(self.length):
            h = self.hiddens[k](h)
            if k != self.length - 1:
                h = self.activation(h)
        return h

    def collectParameters(self):
        all_param_list = []
        for k in range(self.length):
            for x in self.hiddens[k].parameters():
                all_param_list.append(x.view(-1))
        return torch.cat(all_param_list)
