import datetime

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torchvision
from torch import Tensor


class TemporalShift(nn.Module):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(
        self,
        net,
        num_segments=3,
        shift_div=8,
        unidirection: bool = False,
        num_frames=None,
        num_shift: int = 0,
    ):
        """
        Initializes the TemporalShift module.

        Parameters
        ----------
        net : nn.module
            Module to make temporal shift.
        num_segments : int, optional
            Number of frame segments. Default is 3.
        shift_div : int, optional
            Number of divisions for shift. Default is 8.
        unidirection : bool, optional
            If True, do unidirectional shift. Default is False.
        num_frames : int, optional
            Number of frames. Default is None.
        num_shift : int, optional
            Number of shifts. Default is 0.
        """
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.num_frames = num_frames
        self.unidirection = unidirection
        self.num_shift = num_shift

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.shift_packed_sequence(
            x,
            self.num_frames(),
            shift_div=self.shift_div,
            unidirection=self.unidirection,
            num_shift=self.num_shift,
        )

        return self.net(x)

    @staticmethod
    def shift_packed_sequence(
        x, num_frames, shift_div=3, unidirection: bool = True, num_shift: int = 0
    ):
        """
        Shift packed sequence.

        Args:
            x (torch.Tensor): The input data.
            num_frames (list[int]): List of frame lengths.
            shift_div (int, optional): Number of divisions for shift. Default is 3.
            unidirection (bool, optional): If True, do unidirectional shift. Default is True.
            num_shift (int, optional): Number of shifts. Default is 0.

        Returns:
            torch.Tensor: The output of the module.
        """
        n, c, h, w = x.shape
        cummulative_margin = 0
        zeros_out = torch.zeros_like(x)

        # get shift fold
        fold = c // shift_div

        for i in range(len(num_frames)):
            curr_frames = int(num_frames[i])

            temp_tensor = x[cummulative_margin : cummulative_margin + curr_frames, :, :, :]

            # unidirectional shift

            if num_shift >= curr_frames:
                # no shift
                zeros_out[
                    cummulative_margin : cummulative_margin + curr_frames, :, :, :
                ] = temp_tensor

                cummulative_margin += curr_frames

                continue

            if unidirection:
                left_split = temp_tensor[:, :fold, :, :]  # T, C, H, W
                right_split = temp_tensor[:, fold:, :, :]

                zeros = left_split - left_split
                blank = zeros[:1, :, :, :]
                left_split = left_split[:-1, :, :, :]
                left_split = torch.cat((blank, left_split), 0)

                out = torch.cat((left_split, right_split), 1)
            else:
                left_split = temp_tensor[:, :fold, :, :]  # T, C, H, W
                mid_split = temp_tensor[:, fold : 2 * fold, :]
                right_split = temp_tensor[:, 2 * fold :, :, :]

                # shift left on num_segments channel in `left_split`
                zeros = left_split - left_split
                blank = zeros[:1, :, :, :]
                left_split = left_split[1:, :, :, :]
                left_split = torch.cat((left_split, blank), 0)

                # shift right on num_segments channel in `mid_split`
                zeros = mid_split - mid_split
                blank = zeros[:1, :, :, :]
                mid_split = mid_split[:-1, :, :, :]
                mid_split = torch.cat((blank, mid_split), 0)

                out = torch.cat((left_split, mid_split, right_split), 1)

            zeros_out[cummulative_margin : cummulative_margin + curr_frames, :, :, :] = out

            cummulative_margin += curr_frames

        # [N, C, H, W]
        return zeros_out


def make_resnet(name="resnet18"):
    """
    Creates a resnet model of a given name with pre-trained weights and removes the last fully connected layer.

    Parameters
    ----------
    name : str
        The name of the resnet model to create. Should be one of ["resnet18", "resnet34", "resnet50", "resnet101"].

    Returns
    -------
    torch.nn.Module
        The resnet model with the last fully connected layer removed.
    """
    if name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
    elif name == "resnet34":
        model = torchvision.models.resnet34(pretrained=True)
    elif name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
    elif name == "resnet101":
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception("There are no supported resnet model {}.".format("resnet"))

    model.out_channel = model.fc.in_features
    model.fc = nn.Identity()
    return model


class resnet(nn.Module):
    def __init__(self, vis_encoder):
        """
        Initializes the resnet model with the given visual encoder name.

        Parameters
        ----------
        vis_encoder : str
            The name of the visual encoder to use. Should be one of ["resnet18", "resnet34", "resnet50", "resnet101"].
        """
        super().__init__()
        self.resnet = make_resnet(name=vis_encoder)

    def forward(self, x):
        """
        Performs a forward pass through the ResNet model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor representing a batch of images with dimensions [N, C, H, W].

        Returns
        -------
        torch.Tensor
            The output tensor from the ResNet model after feature extraction.
        """
        x = self.resnet(x)
        return x


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        """
        Initializes a TemporalConv module with specified input size, hidden size, and convolution type.

        Parameters
        ----------
        input_size : int
            The number of input features for the temporal convolution layers.
        hidden_size : int
            The number of output features for the temporal convolution layers.
        conv_type : int, optional
            The type of temporal convolution to apply, determining the sequence of kernels and pooling:
            - 0: Applies a single convolution with kernel size 3.
            - 1: Applies two layers of convolution with kernel size 5 followed by pooling with size 2.
            - 2: Applies four layers alternating between convolution with kernel size 5 and pooling with size 2
            (default is 2).

        Attributes
        ----------
        temporal_conv : torch.nn.Sequential
            A sequential container of convolutional, batch normalization, and activation layers based on the
            specified `conv_type`.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ["K3"]
        elif self.conv_type == 1:
            self.kernel_size = ["K5", "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ["K5", "P2", "K5", "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == "P":
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == "K":
                modules.append(
                    nn.Conv1d(
                        input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=1
                    )
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        """
        Performs a forward pass through the TemporalConv module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor representing a sequence of features with dimensions [N, T, C].

        Returns
        -------
        torch.Tensor
            The output tensor from the TemporalConv module after feature extraction with dimensions [N, T, C].
        """
        x = self.temporal_conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class FeatureExtracter(nn.Module):
    def __init__(
        self,
        resnet: str,
        frozen=False,
        num_segments: int = 8,
        temporal_pool: bool = False,
        unidirection: bool = False,
    ):
        """
        Initializes a FeatureExtracter module.

        Parameters
        ----------
        resnet : str
            The name of the ResNet variant to use as the feature extractor.
        frozen : bool, optional
            Whether to freeze the weights of the ResNet model. Default is False.
        num_segments : int, optional
            The number of segments to divide the input sequence into. Default is 8.
        temporal_pool : bool, optional
            Whether to perform temporal pooling on the output of the ResNet model. Default is False.
        unidirection : bool, optional
            Whether to use a unidirectional (forward) convolutional layer instead of a bidirectional layer. Default is False.
        """
        super().__init__()
        print(f'start FeatureExtracter {datetime.datetime.now().strftime("%H:%M:%S")}')
        self.num_segments = num_segments
        self.unidirection = unidirection
        self.temporal_pool = temporal_pool
        self.init_tvtsm(resnet, frozen)
        self.forward_func = self.forward_tvtsm
        self.relu = nn.ReLU()

        print(f'end FeatureExtracter {datetime.datetime.now().strftime("%H:%M:%S")}')

    def make_temporal_shift(self, net):
        """Code copied from mmaction.models.backbones.resnet_tsm"""
        """Make temporal shift for some layers."""
        n_round = 1
        if len(list(net.layer3.children())) >= 23:
            n_round = 2

        def make_block_temporal(stage, num_segments_f, unidirection, num_frames_f):
            """Make temporal shift on some blocks.

            Args:
                stage (nn.Module): Model layers to be shifted.
                num_segments_f (():int): Func. to get number of frame segments.

            Returns:
                nn.Module:s The shifted blocks.
            """
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    self.count_shifts += 1
                    blocks[i].conv1 = TemporalShift(
                        b.conv1,
                        num_segments=num_segments_f,
                        shift_div=self.shift_div,
                        unidirection=unidirection,
                        num_frames=num_frames_f,
                        num_shift=self.count_shifts,
                    )
            return nn.Sequential(*blocks)

        self.count_shifts = 0
        net.layer1 = make_block_temporal(
            net.layer1, lambda: self.tsm_num_segments, self.unidirection, lambda: self.num_frames
        )
        net.layer2 = make_block_temporal(
            net.layer2, lambda: self.tsm_num_segments, self.unidirection, lambda: self.num_frames
        )
        net.layer3 = make_block_temporal(
            net.layer3, lambda: self.tsm_num_segments, self.unidirection, lambda: self.num_frames
        )
        net.layer4 = make_block_temporal(
            net.layer4, lambda: self.tsm_num_segments, self.unidirection, lambda: self.num_frames
        )

    def init_tvtsm(self, resnet, frozen):
        """
        Initializes TSM-ResNet model and a 1D convolutional layer.

        Parameters
        ----------
        resnet : str
            The name of the ResNet model. Supported: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        frozen : bool
            Whether to freeze the weights of the ResNet model. Default is False.

        Attributes
        ----------
        tsm_num_segments : int
            The number of segments to split the input video into for temporal shift.
        num_frames : int
            The total number of frames in the input video.
        resnet : nn.Module
            The ResNet model.
        conv_1d : nn.Module
            The 1D convolutional layer.
        """
        self.shift_div = 8
        self.tsm_num_segments = None  # updated in forward()
        self.num_frames = None  # updated in forward()

        use_tsm = True
        self.resnet = make_resnet(name=resnet)
        if use_tsm:
            self.make_temporal_shift(self.resnet)
        self.conv_1d = TemporalConv(
            input_size=self.resnet.out_channel, hidden_size=1024, conv_type=0
        )
        if frozen:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(
        self,
        src: Tensor,
        input_lenghts=None,
    ):
        """
        Forward pass of the model.

        This function processes the input tensor `src` through the model, utilizing
        the temporal shift module and 1D convolutional layer.

        Parameters
        ----------
        src : Tensor
            The input tensor containing video frames, expected to have dimensions (Batch, Channels, Height, Width).
        input_lenghts : Optional
            The lengths of the input sequences, used to handle variable sequence lengths. Default is None.

        Returns
        -------
        Tensor
            The output tensor after processing through the model's forward function.
        """
        return self.forward_func(src, input_lenghts)

    def forward_tvtsm(self, src: Tensor, input_lenghts=None):
        """
        Forward pass of the model.

        This function processes the input tensor `src` through the model, utilizing
        the temporal shift module and 1D convolutional layer.

        Parameters
        ----------
        src : Tensor
            The input tensor containing video frames, expected to have dimensions (Batch, Channels, Height, Width).
        input_lenghts : Optional
            The lengths of the input sequences, used to handle variable sequence lengths. Default is None.

        Returns
        -------
        Tensor
            The output tensor after processing through the model's forward function.
        """
        B, C, H, W = src.shape
        self.tsm_num_segments = None
        self.num_frames = input_lenghts
        src = self.resnet(src)

        src_pad = torch.zeros(len(input_lenghts), input_lenghts.max(), 512).cuda()
        cummulative_margin = 0

        # fill in padding to restore temporal dimension
        for i in range(len(input_lenghts)):
            curr_frames = int(input_lenghts[i])
            src_pad[i, :curr_frames] = src[
                cummulative_margin : cummulative_margin + curr_frames, :
            ]
            cummulative_margin += curr_frames

        src = self.conv_1d(src_pad)

        return src
