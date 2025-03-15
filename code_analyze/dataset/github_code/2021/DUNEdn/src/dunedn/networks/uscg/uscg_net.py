"""This module contains the UscgNet model class."""
import logging
from pathlib import Path
from time import time as tm
from typing import Tuple
import torch
from torch import nn
from math import ceil
from torchvision.models import resnext50_32x4d
from ..abstract_net import AbstractNet
from .uscg_dataloading import UscgDataset
from .uscg_net_blocks import (
    SCG_Block,
    GCN_Layer,
    Pooling_Block,
    Recombination_Layer,
)
from .utils import uscg_inference_pass, time_windows
from dunedn import PACKAGE

logger = logging.getLogger(PACKAGE + ".uscg")


class UscgNet(AbstractNet):
    """U-shaped Self Constructing Graph Network."""

    def __init__(
        self,
        channel: str = "collection",
        out_channels: int = 1,
        h_induction: int = 800,
        h_collection: int = 960,
        w: int = 6000,
        stride: int = 1000,
        pretrained: bool = True,
        node_size: list[int] = [28, 28],
        dropout: float = 0.5,
        enhance_diag: bool = True,
        aux_pred: bool = True,
    ):
        """
        Parameters
        ----------
        channel: str
            Available options induction | collection.
        out_channels: int
            Output image channels number.
        h_induction: int
            Induction input image height.
        h_collection: int
            Collection input image height.
        w: int
            Input image width.
        stride: int
            Steps between time windows.
        pretrained: bool
            Wether to download weight of pretrained resnet or not.
        node_size: list
            [height, width] of the image input of SCG block.
        dropout: float
            Percentage of neurons turned off in graph layer.
        enhance_diag: bool
            SCG_block flag.
        aux_pred: bool
            SCG_block flag.
        """
        super(UscgNet, self).__init__()
        self.out_channels = out_channels
        self.channel = channel
        self.h_collection = h_collection
        self.h_induction = h_induction
        self.w = w
        self.stride = stride
        self.pretrained = pretrained
        self.node_size = node_size
        self.dropout = dropout
        self.enhance_diag = enhance_diag
        self.aux_pred = aux_pred

        self.h = self.h_induction if self.channel == "induction" else self.h_collection

        self.input_shape = (1, self.h, self.w)

        resnet = resnext50_32x4d(pretrained=self.pretrained, progress=True)
        resnet_12 = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        resnet_34 = nn.Sequential(
            resnet.layer3, resnet.layer4, nn.Conv2d(2048, 1024, 1)
        )
        self.downsamples = nn.ModuleList(
            [resnet_12, resnet_34, Pooling_Block(1024, 28, 28)]
        )
        self.upsamples = nn.ModuleList(
            [
                Pooling_Block(1, ceil(self.h / 32), ceil(self.w / 32)),
                Pooling_Block(1, ceil(self.h / 8), ceil(self.w / 8)),
                Pooling_Block(1, self.h, self.w),
            ]
        )
        self.gcns = nn.Sequential(
            GCN_Layer(
                1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=self.dropout
            ),
            GCN_Layer(128, self.out_channels, bnorm=False, activation=None),
        )
        self.scg = SCG_Block(
            in_ch=1024,
            hidden_ch=self.out_channels,
            node_size=self.node_size,
            add_diag=self.enhance_diag,
            dropout=self.dropout,
        )
        # weight_xavier_init(*self.GCNs, self.scg)
        self.adapts = nn.ModuleList(
            [
                nn.Conv2d(512, 1, 1, bias=False),
                nn.Conv2d(1024, 1, 1, bias=False),
                nn.Conv2d(1024, 1, 1, bias=False),
            ]
        )
        self.recombs = nn.ModuleList([Recombination_Layer() for i in range(3)])
        self.last_recomb = Recombination_Layer()
        # self.act = nn.Sigmoid() if task == "roi" else nn.Identity()

    def forward(self, x):
        """USCG Net Forwards pass.

        Parameters
        ----------
            - x: torch.Tensor, input tensor of shape=(N,C,H,W)

        Returns
        -------
            - torch.Tensor, output tensor of shape=(N,C,H,W)
        """
        # if self.task == "roi":
        #     x /= 3197 + 524  # normalizing according to dataset
        i = x

        # downsampling
        ys = []
        for adapt, downsample in zip(self.adapts, self.downsamples):
            x = downsample(x)
            ys.append(adapt(x))

        # Graph
        batch_size, nb_channels, _, _ = x.size()
        a, x, loss, z_hat = self.scg(x)
        x, _ = self.gcns((x.reshape(batch_size, -1, nb_channels), a))
        if self.aux_pred:
            x += z_hat
        x = x.reshape(batch_size, self.out_channels, *self.node_size)

        # upsampling
        for y, recomb, upsample in zip(
            reversed(ys), reversed(self.recombs), self.upsamples
        ):
            x = upsample(recomb(x, y))

        if self.training:
            return x * i, loss
            # return self.act(x * i), loss
        return x * i
        # return self.act(x * i)

    def predict(
        self,
        generator: UscgDataset,
        dev: str = "cpu",
        no_metrics: bool = False,
        verbose: int = 1,
    ) -> Tuple[torch.Tensor, dict]:
        """Uscg network inference.

        Parameters
        ----------
        generator: UscgDataset
            The inference dataset generator.
        device: str
            The device hosting the computation. Defaults is "cpu".
        no_metrics: bool
            Wether to skip metric computation. Defaults to False, so metrics are
            indeed computed.
        verbose: int
            Switch to log information. Defaults to 1. Available options:

            - 0: no logs.
            - 1: display progress bar.

        Returns
        -------
        y_pred: torch.Tensor
            Denoised planes, of shape=(N,1,H,W).
        logs: dict
            The computed metrics results in dictionary form.
        """
        self.check_network_is_compiled()

        # convert planes to crops
        test_loader = torch.utils.data.DataLoader(
            dataset=generator,
            batch_size=generator.batch_size,
        )

        # inference pass
        start = tm()
        y_pred = uscg_inference_pass(test_loader, self, dev, verbose)
        inference_time = tm() - start

        if no_metrics:
            return y_pred

        # compute metrics
        y_true = generator.clear
        logs = self.metrics_list.compute_metrics(y_pred, y_true)
        logs.update({"time": inference_time})
        return y_pred, logs

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        dev: str = "cpu",
    ) -> dict:
        """Trains the network for one epoch.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The training dataloader.
        validation_data: torch.utils.data.DataLoader
            The validation dataloader.
        dev: str
            The device hosting the computation.

        Returns
        -------
        epoch_logs: dict
            The dictionary of epoch logs.
        """
        self.train()

        for noisy, clear in train_loader:
            _, cwindows, _ = time_windows(clear, self.w, self.stride)
            _, nwindows, _ = time_windows(noisy, self.w, self.stride)
            for nwindow, cwindow in zip(nwindows, cwindows):
                self.callback_list.on_train_batch_begin()
                step_logs = self.train_batch(nwindow, cwindow, dev)
                self.callback_list.on_train_batch_end(step_logs)

        epoch_logs = {}
        return epoch_logs

    def train_batch(self, noisy: torch.Tensor, clear: torch.Tensor, dev: str):
        """Makes one batch update.

        Parameters
        ----------
        noisy: torch.Tensor
            Noisy inputs batch, of shape=(B,1,H,W).
        clear: torch.Tensor
            Clear target batch, of shape=(B,1,H,W)
        dev: str
            The device hosting the computation.

        Returns
        -------
        step_logs: dict
            The step logs as a dictionary.
        """
        clear = clear.to(dev)
        noisy = noisy.to(dev)
        self.optimizer.zero_grad()
        y_pred, loss0 = self.forward(noisy)
        loss1 = self.loss_fn(y_pred, clear)
        loss = loss1 + loss0
        loss.backward()
        self.optimizer.step()

        step_logs = self.metrics_list.compute_metrics(y_pred, clear)
        step_logs.update({"loss": loss.item()})
        return step_logs

    def onnx_export(self, fname: Path):
        """Export model to ONNX format.

        Parameters
        ----------
        fname: Path
            The path to save the `.onnx` network.
        """
        raise NotImplementedError(
            "Currently, UscgNet cannot be exported to `onnx` format."
        )
        # produce dummy inputs
        inputs = torch.randn(1, 1, self.h, self.w)

        # export network
        torch.onnx.export(
            self,
            inputs,
            fname,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
