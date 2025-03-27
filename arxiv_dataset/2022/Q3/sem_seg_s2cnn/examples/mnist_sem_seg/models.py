from s2cnn import SO3Convolution
from s2cnn import SO3Shortcut
from s2cnn import S2Convolution
from s2cnn import SO3ToS2
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid

import torch.nn.functional as F
import torch
import torch.nn as nn

import numpy as np



class S2ConvNet_sem_seg(nn.Module):
    def __init__(self, b_in, f_in, fs, b_ls, b_out, f_out, kernel_max_beta, use_skips):
        super(S2ConvNet_sem_seg, self).__init__()

        self.use_skips = use_skips

        if isinstance(kernel_max_beta, float):
            kernel_max_beta = [kernel_max_beta] * (len(fs) + 1)

        assert len(fs) == len(b_ls) == len(kernel_max_beta) - 1

        grid_s2 = s2_near_identity_grid(max_beta=kernel_max_beta[0] * np.pi)
        grids_so3 = [
            so3_near_identity_grid(max_beta=max_beta * np.pi) for max_beta in kernel_max_beta[1:]
        ]

        self.input_layer = S2Convolution(
            nfeature_in=f_in, nfeature_out=fs[0], b_in=b_in, b_out=b_ls[0], grid=grid_s2
        )

        self.output_layer = SO3ToS2(
            nfeature_in=fs[-1], nfeature_out=f_out, b_in=b_ls[-1], b_out=b_out, grid=grids_so3[-1]
        )

        self.hidden_layers = nn.ModuleList()

        for i in range(len(fs) - 1):
            self.hidden_layers.append(
                SO3Convolution(
                    nfeature_in=fs[i],
                    nfeature_out=fs[i + 1],
                    b_in=b_ls[i],
                    b_out=b_ls[i + 1],
                    grid=grids_so3[i],
                )
            )

        if self.use_skips:
            self.skip_layers = nn.ModuleList()

            for i in range(len(fs) - 1):
                self.skip_layers.append(
                    SO3Shortcut(
                        nfeature_in=fs[i],
                        nfeature_out=fs[i + 1],
                        b_in=b_ls[i],
                        b_out=b_ls[i + 1],
                    )
                )

        feature_dims = [str(2 * b_in) + "x" + str(2 * b_in)]
        feature_dims += [str(2 * b_l) + "x" + str(2 * b_l) + "x" + str(2 * b_l) for b_l in b_ls]
        feature_dims += [str(2 * b_out) + "x" + str(2 * b_out)]

        self.add_to_h_params = {"feature_dims": str(feature_dims).replace("'", "")}

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for i, layer in enumerate(self.hidden_layers):
            if self.use_skips:
                x_skip = self.skip_layers[i](x)
                x = F.relu(layer(x) + x_skip)
            else:
                x = F.relu(layer(x))

        x = self.output_layer(x)

        return x




class ConvNet_sem_seg(nn.Module):
    def __init__(self, dim_in, f_in, fs, f_out, k_sizes, strides, use_skips):
        super(ConvNet_sem_seg, self).__init__()

        self.use_skips = use_skips

        assert len(fs) == len(k_sizes) == len(strides)
        self.dims = [dim_in]

        self.input_layer = torch.nn.Conv2d(f_in, fs[0], kernel_size=k_sizes[0], stride=strides[0])
        self.dims.append(self.dim_out(dim_in, k_sizes[0], strides[0]))

        self.downsample_layers = nn.ModuleList()

        for i in range(len(fs) - 1):
            self.downsample_layers.append(
                torch.nn.Conv2d(fs[i], fs[i + 1], kernel_size=k_sizes[i + 1], stride=strides[i + 1])
            )
            self.dims.append(self.dim_out(self.dims[-1], k_sizes[i + 1], strides[i + 1]))

        if self.use_skips:
            self.skip_max_pool_layers = nn.ModuleList()
            self.skip_down_conv_layers = nn.ModuleList()

            for i in range(len(fs) - 1):
                self.skip_max_pool_layers.append(
                    torch.nn.MaxPool2d(kernel_size=k_sizes[i + 1], stride=strides[i + 1])
                )
                self.skip_down_conv_layers.append(torch.nn.Conv2d(fs[i], fs[i + 1], kernel_size=1))

        self.upsample_layers = nn.ModuleList()

        for i in range(len(fs) - 1, 0, -1):
            output_padding = self.output_padding(
                self.dims[i + 1], self.dims[i], k_sizes[i], strides[i]
            )
            self.upsample_layers.append(
                torch.nn.ConvTranspose2d(
                    fs[i],
                    fs[i - 1],
                    kernel_size=k_sizes[i],
                    stride=strides[i],
                    output_padding=output_padding,
                )
            )

        if self.use_skips:
            self.skip_upsample_layers = nn.ModuleList()
            self.skip_up_conv_layers = nn.ModuleList()
            for i in range(len(fs) - 1, 0, -1):
                self.skip_upsample_layers.append(
                    torch.nn.Upsample(size=self.dims[i], mode="nearest")
                )
                self.skip_up_conv_layers.append(torch.nn.Conv2d(fs[i], fs[i - 1], kernel_size=1))

        output_padding = self.output_padding(self.dims[1], dim_in, k_sizes[0], strides[0])
        self.out_layer = torch.nn.ConvTranspose2d(
            fs[0], f_out, kernel_size=k_sizes[0], stride=strides[0], output_padding=output_padding
        )

        dims_h_param = [str(dim) + "x" + str(dim) for dim in self.dims]
        dims_h_param += [dims_h_param[i] for i in range(len(self.dims) - 2, -1, -1)]

        self.add_to_h_params = {"feature_dims": str(dims_h_param).replace("'", "")}

    def dim_out(self, dim_in, kernel_size, stride):
        """Computes the output dimension of input to Conv2d layer with given parameters"""
        dim_out = np.floor((dim_in - kernel_size) / stride + 1)
        return int(dim_out)

    def output_padding(self, dim_in, dim_out, kernel_size, stride):
        """Computes the output_padding necessary for ConvTranspose2d layer with given parameters
        to produce output of size dim_out, given input of size dim_in"""
        output_padding = dim_out - kernel_size - (dim_in - 1) * stride
        return output_padding

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for i, layer in enumerate(self.downsample_layers):
            if self.use_skips:
                x_skip = self.skip_down_conv_layers[i](self.skip_max_pool_layers[i](x))
                x = F.relu(layer(x) + x_skip)
            else:
                x = F.relu(layer(x))

        for i, layer in enumerate(self.upsample_layers):
            if self.use_skips:
                x_skip = self.skip_upsample_layers[i](self.skip_up_conv_layers[i](x))
                x = F.relu(layer(x) + x_skip)
            else:
                x = F.relu(layer(x))

        x = self.out_layer(x)

        return x
