"""
Benchmark of IGC with pRF-like simulations.

Model predicting categorical image statistics from images.
1v0 : ConvNeXt and linear blocks
"""

import os

import numpy as np
import torch
from igc import IntGradCorr
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchutils import (
    AbstractModel,
    ConvNeXt,
    LinearBlock,
    freeze_network,
    set_dtld_seed,
)

from .simulation_1v0 import Dataset  # pylint: disable=W0611


# Modules


class Encoder(nn.Module):
    def __init__(
        self,
        img_size,
        n_cat,
        cvnx_sizes,
        cvnx_stem_kernel,
        lin_sizes,
        cvnx_depth_prob=0.5,
        lin_dropout=0.25,
        lin_dropout_min_size=16,
        lin_act_type="mish",
    ):
        super().__init__()
        self.n_cat = n_cat
        conv_out_hw = img_size
        layers = []
        # ConvNeXt block
        if cvnx_sizes is None:
            cvnx_sizes = tuple()
        cvnx_sizes = (1,) + cvnx_sizes
        if len(cvnx_sizes) > 1:
            cvnx = ConvNeXt(
                cvnx_sizes,
                cvnx_stem_kernel,
                stochastic_depth_prob=cvnx_depth_prob,
            )
            layers.append(cvnx)
            conv_out_hw = cvnx.get_output_spatial_size(conv_out_hw)
        # Flatten features
        layers.append(nn.Flatten(start_dim=1))
        # Linear block
        if lin_sizes is None:
            lin_sizes = tuple()
        lin_sizes = (conv_out_hw**2 * cvnx_sizes[-1],) + lin_sizes + (n_cat,)
        layers.append(
            LinearBlock(
                lin_sizes,
                lin_act_type,
                dropout=lin_dropout,
                dropout_min_size=lin_dropout_min_size,
                output_act=False,
            )
        )
        # Build network
        self.enc = nn.Sequential(*layers)
        # Misc
        self._mse_loss = nn.MSELoss(reduction="none")

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.enc(x)
        return nn.functional.log_softmax(x, dim=1)

    def get_y_r(self, log_p):
        label = torch.argmax(log_p, dim=1)
        return nn.functional.one_hot(label, self.n_cat)  # pylint: disable=E1102

    def nll_loss(self, log_p, y):
        return -1.0 * torch.sum(log_p * y)

    def ac_loss(self, log_p, y):
        return torch.sum(self.get_y_r(log_p) * y)


# Model


class Model(AbstractModel):
    # Default parameters
    cvnx_sizes = (16, 32, 64, 128, 256)
    cvnx_stem_kernel = 2
    cvnx_depth_prob = 0.5
    lin_sizes = (128, 16)
    lin_dropout = 0.25
    lin_dropout_min_size = 16
    lin_act_type = "mish"
    learning_rate = 5e-5
    scheduler_decay = 0.9
    scheduler_patience = 4
    scheduler_min_lr = 1e-6
    save_every_n_epoch = 1
    maximum_saves = 1
    seed = 100
    dtype = torch.float32
    project_path = os.path.dirname(__file__)

    def __init__(
        self,
        dataset,
        model_name="model_cat_1v0",
        trainable=False,
        device=None,
        parameters=None,
    ):
        super().__init__(model_name, device, parameters)
        # Dataset
        self.dtst = dataset
        # Network
        self.network = Encoder(
            self.dtst.img_size,
            self.dtst.n_y,
            self.cvnx_sizes,
            self.cvnx_stem_kernel,
            self.lin_sizes,
            self.cvnx_depth_prob,
            self.lin_dropout,
            self.lin_dropout_min_size,
            self.lin_act_type,
        )
        self.network.to(self.device)
        # Optimizer
        self.optimizer, self.scheduler = None, None
        if trainable:
            self.optimizer = optim.AdamW(
                self.network.parameters(), lr=self.learning_rate
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.scheduler_decay,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr,
            )

    def train(self, n_epoch=1, batch_size=64, num_workers=0):
        self.restore("last", training=True)
        writer = SummaryWriter(self.logs_path)
        train_dtld = self.dtst.train_dtld(batch_size, self.seed, num_workers)
        val_dtld = self.dtst.val_dtld(batch_size, num_workers=num_workers)
        seed_rng = self.get_seed_rng()
        # Iterate over epochs
        for _ in range(int(n_epoch)):
            self.current_epoch += 1
            current_seed = seed_rng.get_seed()
            torch.manual_seed(current_seed)
            set_dtld_seed(train_dtld, current_seed)
            # Training #########################################################
            self.network.train()
            train_nll_avg = 0.0
            for i, (x, y) in enumerate(
                tqdm(
                    train_dtld,
                    total=len(train_dtld),
                    desc=f"trn {self.current_epoch:04d}",
                    leave=False,
                )
            ):
                # Init gradients
                self.network.zero_grad(set_to_none=True)
                # Send data to the device
                x = x.to(self.device)
                y = y.to(self.device)
                # Forward and backward passes
                log_p = self.network(x)
                nll_loss = self.network.nll_loss(log_p, y)
                nll_loss.backward()
                # Optimization
                self.optimizer.step()
                # Summary
                train_nll_avg += nll_loss.item()
            train_nll_avg /= float(len(train_dtld) * batch_size)
            writer.add_scalar(
                "train/nll_loss", train_nll_avg, self.current_epoch
            )
            # Validation #######################################################
            self.network.eval()
            with torch.no_grad():
                val_nll_avg = 0.0
                for i, (x, y) in enumerate(
                    tqdm(
                        val_dtld,
                        total=len(val_dtld),
                        desc=f"val {self.current_epoch:04d}",
                        leave=False,
                    )
                ):
                    # Send data to the device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # Forward pass
                    log_p = self.network(x)
                    nll_loss = self.network.nll_loss(log_p, y)
                    # Summary
                    val_nll_avg += nll_loss.item()
            val_nll_avg /= float(self.dtst.n_val)
            writer.add_scalar(
                "validation/val_nll_avg", val_nll_avg, self.current_epoch
            )
            # Save checkpoint ##################################################
            if not self.current_epoch % self.save_every_n_epoch:
                self.save(val_nll_avg)
            # LR scheduler #####################################################
            for i, param_group in enumerate(self.optimizer.param_groups):
                writer.add_scalar(
                    f"learning_rate/lr_{i:d}",
                    float(param_group["lr"]),
                    self.current_epoch,
                )
            self.scheduler.step(val_nll_avg)
            self.clean_saves()
        writer.close()
        return self

    @torch.no_grad()
    def score(self, batch_size=64, checkpoint="best"):
        # Restore
        self.restore(checkpoint)
        # Eval
        self.network.eval()
        val_nll_avg, val_ac_avg = 0.0, 0.0
        dtld = self.dtst.val_dtld(batch_size)
        for x, y in tqdm(dtld, total=len(dtld), desc="score"):
            # Send data to the device
            x = x.to(self.device)
            y = y.to(self.device)
            # Evaluation
            log_p = self.network(x)
            nll_loss = self.network.nll_loss(log_p, y)
            ac_loss = self.network.ac_loss(log_p, y)
            # Summary
            val_nll_avg += nll_loss.item()
            val_ac_avg += ac_loss.item()
        val_nll_avg /= float(self.dtst.n_val)
        val_ac_avg /= float(self.dtst.n_val)
        return val_nll_avg, val_ac_avg

    def _get_dtld_kwargs(self, num_workers):
        return {"num_workers": num_workers, "pin_memory": True}

    @freeze_network
    def int_grad_corr(
        self,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=100,
        x_0_seed=101,
        n_x=None,
        save_results=True,
        check_error=True,
        suffix="",
        num_workers=0,
        checkpoint="best",
    ):
        # Restore
        self.restore(checkpoint)
        # Compute IGC
        attr = IntGradCorr(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        igc = attr.compute(
            x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed, n_x, check_error
        )
        # Save results
        if suffix:
            suffix = "_" + suffix
        if save_results:
            np.savez(
                self.get_result_path(f"int_grad_corr{suffix}.npz"), data=igc
            )
        # Return results
        return igc

    @torch.no_grad()
    def igc_error(
        self,
        igc_name,
        y_idx=None,
        batch_size=None,
        num_workers=0,
        checkpoint="best",
    ):
        # Restore
        self.restore(checkpoint)
        # Load IGC data
        igc = np.load(self.get_result_path(igc_name))["data"]
        # Compute IGC error
        attr = IntGradCorr(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        error = attr.error(igc, y_idx, batch_size)
        # Return results
        return error
