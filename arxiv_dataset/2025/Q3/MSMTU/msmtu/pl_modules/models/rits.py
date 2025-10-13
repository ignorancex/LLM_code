import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(
            self,
            rnn_hid_size,
            impute_weight,
            label_weight,
            data_dim=7,
            seq_len=223,
            output_dim: int = 29,
            ancillary_dim: int = 0,
            embedding_dim: int = 50
    ):
        super(RITS, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.output_dim = output_dim
        self.data_dim = data_dim
        self.seq_len = seq_len
        self.ancillary_dim = ancillary_dim

        # Network components
        self.rnn_cell = nn.LSTMCell(self.data_dim * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size=self.data_dim, output_size=self.rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.data_dim, output_size=self.data_dim, diag=True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.data_dim)
        self.feat_reg = FeatureRegression(self.data_dim)

        self.weight_combine = nn.Linear(self.data_dim * 2, self.data_dim)

        self.dropout = nn.Dropout(p=0.25)

        if ancillary_dim:
            # Add an FC layer for geographical coordinates
            self.ancillary_embed = nn.Linear(self.ancillary_dim, embedding_dim)

            # Add concatenation layer to concatenate the hidden state with ancillary features
            self.concat = nn.Linear(self.rnn_hid_size + embedding_dim, self.rnn_hid_size)

        self.out = nn.Linear(self.rnn_hid_size, self.output_dim)

        # Define loss function
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, data, direct, stage):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        labels = data['labels'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        h = h.type_as(values)
        c = c.type_as(values)

        x_loss = 0.0

        imputations = []
        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim=1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)

        labels = labels.long()
        labels = labels.view(-1)  # labels in 1-D

        x_loss = x_loss * self.impute_weight

        if self.ancillary_dim:
            # Compute ancillary features
            ancillary_embed = torch.relu(self.ancillary_embed(data['ancillary']))

            # Concatenate hidden states with ancillary features
            h = torch.relu(self.concat(torch.cat([h, ancillary_embed], dim=1)))

        y_h = self.out(h)

        probs = data['probs']

        y_h = F.softmax(y_h, dim=1)
        y_loss = self.mse(y_h, probs)

        y_loss = torch.sum(y_loss) / (len(labels) + 1e-5)

        y_loss = y_loss * self.label_weight

        return {'x_loss': x_loss, 'y_loss': y_loss, 'loss': x_loss + y_loss,
                'predictions': y_h, 'imputations': imputations, 'labels': labels, 'probs': probs,
                'square_ids': data['square_ids']}

    def change_last_layer(self, output_dim: int):
        self.output_dim = output_dim
        self.out = nn.Linear(self.rnn_hid_size, self.output_dim)
