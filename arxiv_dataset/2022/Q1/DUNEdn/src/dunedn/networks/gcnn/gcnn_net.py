"""
    This module contains the GcnnNet model class.

    GcnnNet implements also the CNN variant.
"""
import logging
from typing import Tuple
from pathlib import Path
from time import time as tm
import torch
from torch import nn
from ..abstract_net import AbstractNet
from ..utils import BatchProfiler
from .gcnn_dataloading import BaseGcnnDataset
from .gcnn_net_blocks import (
    PreProcessBlock,
    ROI,
    HPF,
    LPF,
    PostProcessBlock,
    NonLocalGraph,
)
from .utils import gcnn_inference_pass
from dunedn import PACKAGE

logger = logging.getLogger(PACKAGE + ".gcnn")


class GcnnNet(AbstractNet):
    """Graph Convolutional Neural Network implementation."""

    def __init__(
        self,
        model: str,
        task: str,
        crop_edge: int,
        input_channels: int,
        hidden_channels: int,
        k: int = None,
    ):
        """
        Parameters
        ----------
        model: str
            Available options cnn | gcnn.
        task: str
            Available options dn | roi.
        crop_edge: int
            Crop edge size.
        input_channels: int
            Inputh channel dimension size.
        hidden_channels: int
            Convolutions hidden filters number.
        k: int
            Nearest neighbor number. None if model is cnn..
        """
        super(GcnnNet, self).__init__()
        self.crop_size = (crop_edge,) * 2
        self.model = model
        self.task = task
        ic = input_channels
        hc = hidden_channels
        self.k = k

        self.input_shape = (1,) + self.crop_size

        self.getgraph_fn = (
            NonLocalGraph(k, self.crop_size) if self.model == "gcnn" else lambda x: None
        )
        # self.norm_fn = choose_norm(dataset_dir, channel, normalization)
        self.roi = ROI(7, ic, hc, self.getgraph_fn, self.model)
        self.pre_process_blocks = nn.ModuleList(
            [
                PreProcessBlock(5, ic, hc, self.getgraph_fn, self.model),
                PreProcessBlock(7, ic, hc, self.getgraph_fn, self.model),
                PreProcessBlock(9, ic, hc, self.getgraph_fn, self.model),
            ]
        )
        self.lpfs = nn.ModuleList(
            [
                LPF(hc * 3 + 1, hc * 3 + 1, self.getgraph_fn, self.model)
                for _ in range(4)
            ]
        )
        self.hpf = HPF(hc * 3 + 1, hc * 3 + 1, self.getgraph_fn, self.model)
        self.post_process_block = PostProcessBlock(ic, hc, self.getgraph_fn, self.model)
        self.aa = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.bb = nn.Parameter(torch.Tensor([1]), requires_grad=False)

        self.combine = lambda x, y: x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gcnn forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape=(N,C,H,W).

        Returns
        -------
        output: torch.Tensor
            Output tensor of shape=(N,C,H,W).
        """
        # x = self.norm_fn(x)
        hits = self.roi(x)
        if self.task == "roi":
            return hits
        y = torch.cat([block(x) for block in self.pre_process_blocks], dim=1)
        y = torch.cat([y, hits], 1)
        y_hpf = self.hpf(y)
        y = self.combine(y, y_hpf)
        for lpf in self.lpfs:
            y = self.combine(lpf(y), y_hpf)
        output = self.post_process_block(y) * x
        return output

    def predict(
        self,
        generator: BaseGcnnDataset,
        dev: str = "cpu",
        no_metrics: bool = False,
        verbose: int = 1,
        profiler: BatchProfiler = None,
    ) -> Tuple[torch.Tensor, list[Tuple[float, float]], float]:
        """Gcnn network inference.

        Parameters
        ----------
        generator: GcnnDataset
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

        profiler: BatchProfiler
            The profiler object to record batch inference time.

        Returns
        -------
        y_pred: torch.Tensor
            Denoised planes, of shape=(N,1,H,W).
        logs: dict
            The computed metrics results in dictionary form.
        """
        self.check_network_is_compiled()

        # convert planes to crops
        # TODO: think about a `with` statement for the test_loader object, as it
        # shouldn't be possible to call the inference without `to_crops()` and
        # `to_planes()` methods
        generator.to_crops()
        test_loader = torch.utils.data.DataLoader(
            dataset=generator,
            batch_size=generator.batch_size,
        )

        # inference pass
        start = tm()
        output = gcnn_inference_pass(test_loader, self, dev, verbose, profiler=profiler)
        inference_time = tm() - start

        # convert back to planes
        y_pred = generator.converter.tiles2planes(output)
        generator.to_planes()

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
        logger.debug("Training epoch")
        self.train()

        for clear, noisy in train_loader:
            self.callback_list.on_train_batch_begin()
            step_logs = self.train_batch(noisy, clear, dev)
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
        noisy = noisy.to(dev)
        self.optimizer.zero_grad()
        y_pred = self.forward(noisy)

        loss = self.loss_fn(y_pred, clear)
        loss.mean().backward()
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
        # produce dummy inputs
        inputs = torch.randn(1, 1, *self.crop_size)

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
