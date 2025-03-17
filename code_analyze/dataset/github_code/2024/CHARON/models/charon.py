# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.utils.continual_model import ContinualModel
from utils.args import (ArgumentParser, add_experiment_args,
                        add_management_args, add_rehearsal_args)
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--masking_ratio', type=float, required=True)
    parser.add_argument('--gamma', type=float, required=True, help='Reconstruction-based regularization weight.')
    parser.add_argument('--linear_probing_epochs', type=int, default=0, help='Number of linear probing epochs to perform at the end of each task.')
    parser.add_argument('--sampling_interval', type=int, default=5, help='Interval between frames.')

    return parser


class CHARON(ContinualModel):
    NAME = 'charon'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(CHARON, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, args=self.args)
        self.gamma = self.args.gamma

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        tot_loss = 0

        outputs, reconstructions = self.net(inputs)
        loss_stream = self.loss(outputs, labels)
        loss_mae_stream = self.gamma * F.mse_loss(reconstructions, inputs)
        loss = loss_stream + loss_mae_stream
        tot_loss += loss.item()
        loss.backward()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            buf_outputs, buf_reconstructions = self.net(buf_inputs)
            loss_logits = F.mse_loss(buf_outputs, buf_logits)
            loss_mae_logits = self.gamma * F.mse_loss(buf_reconstructions, buf_inputs)
            loss = self.args.alpha * (loss_logits + loss_mae_logits)
            tot_loss += loss.item()
            loss.backward()

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            buf_outputs, buf_reconstructions = self.net(buf_inputs)
            loss_labels = self.loss(buf_outputs, buf_labels)
            loss_mae_labels = self.gamma * F.mse_loss(buf_reconstructions, buf_inputs)
            loss = self.args.beta * (loss_labels + loss_mae_labels)
            tot_loss += loss.item()
            loss.backward()

        if self.args.grad_clip:
            torch.nn.utils.clip_grad_value_(self.net.parameters(), self.args.grad_clip)

        self.opt.step()

        return tot_loss

    def observe_lp(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        tot_loss = 0

        outputs = self.net(inputs)
        loss_stream_lp = self.loss(outputs, labels)
        tot_loss += loss_stream_lp.item()
        loss_stream_lp.backward()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            buf_outputs = self.net(buf_inputs)
            loss_logits_lp = F.mse_loss(buf_outputs, buf_logits)
            loss = self.args.alpha * loss_logits_lp
            tot_loss += loss.item()
            loss.backward()

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            buf_outputs = self.net(buf_inputs)
            loss_labels_lp = self.loss(buf_outputs, buf_labels)
            loss = self.args.beta * loss_labels_lp
            tot_loss += loss.item()
            loss.backward()

        if self.args.grad_clip:
            torch.nn.utils.clip_grad_value_(self.net.parameters(), self.args.grad_clip)

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)

        return tot_loss

    def end_task(self, train_loader):

        if not self.args.linear_probing_epochs:
            return

        self.net.eval()

        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.net.classifier.parameters():
            param.requires_grad = True

        for epoch in range(self.args.linear_probing_epochs):
            with tqdm(total=len(train_loader), desc=f"Linear probing on the full poses [Epoch: {epoch}]") as pbar:
                for i, data in enumerate(train_loader):
                    if self.args.debug_mode and i > 3:
                        break

                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.float().to(self.device), labels.to(self.device)
                    not_aug_inputs = not_aug_inputs.float().to(self.device)
                    loss = self.observe_lp(inputs, labels, not_aug_inputs)

                    assert not math.isnan(loss)

                    pbar.update()

        self.update_buffer_logits()

        for param in self.net.parameters():
            param.requires_grad = True

        self.net.train()

    @torch.no_grad()
    def update_buffer_logits(self):
        B = self.args.batch_size

        for i in range(0, len(self.buffer), B):
            idxs = torch.arange(i, min(i + B, len(self.buffer)))
            buf_inputs, *_ = self.buffer.get_data_by_index(idxs, transform=self.transform)
            buf_logits = self.net(buf_inputs)
            self.buffer.logits[idxs] = buf_logits.data
