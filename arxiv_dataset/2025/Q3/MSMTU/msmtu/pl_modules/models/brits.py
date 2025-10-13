import torch
import torch.nn as nn
import torch.nn.functional as F

from .rits import RITS


def get_consistency_loss(pred_f, pred_b):
    loss = torch.abs(pred_f - pred_b).mean() * 1e-1
    return loss


def merge_ret(ret_f, ret_b):
    loss_x_f = ret_f['x_loss']
    loss_x_b = ret_b['x_loss']

    loss_c = get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

    imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

    ret_f['x_loss'] = (ret_f['x_loss'] + ret_b['x_loss']) / 2
    ret_f['imputations'] = imputations

    loss_y_f = ret_f['y_loss']
    loss_y_b = ret_b['y_loss']

    loss = (loss_x_f + loss_y_f) + (loss_x_b + loss_y_b) + loss_c

    predictions_f = ret_f['predictions']
    predictions_b = ret_b['predictions']

    predictions = (ret_f['predictions'] + ret_b['predictions']) / 2

    ret_f['loss'] = loss

    ret_f['y_loss'] = (ret_f['y_loss'] + ret_b['y_loss']) / 2
    ret_f['predictions_f'] = predictions_f
    ret_f['predictions_b'] = predictions_b
    ret_f['predictions'] = predictions

    return ret_f


def reverse(ret):
    def reverse_tensor(tensor_):
        if tensor_.dim() <= 1:
            return tensor_
        indices = range(tensor_.size()[1])[::-1]
        indices = torch.LongTensor(indices)

        if torch.cuda.is_available():
            indices = indices.cuda()
        elif torch.backends.mps.is_available():
            indices = indices.to(torch.device("mps"))

        return tensor_.index_select(1, indices)

    for key in ret:
        if key != 'predictions':
            ret[key] = reverse_tensor(ret[key])

    return ret


class BRITS(nn.Module):
    def __init__(
            self,
            rnn_hid_size,
            impute_weight,
            label_weight,
            data_dim: int = 7,
            seq_len: int = 223,
            output_dim: int = 29,
            ancillary_dim: int = 0,
            embedding_dim: int = 50,
    ):
        super(BRITS, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.output_dim = output_dim
        self.data_dim = data_dim
        self.seq_len = seq_len
        self.ancillary_dim = ancillary_dim
        self.embedding_dim = embedding_dim

        self.build()

    def build(self):
        self.rits_f = RITS(
            self.rnn_hid_size,
            self.impute_weight,
            self.label_weight,
            self.data_dim,
            self.seq_len,
            self.output_dim,
            self.ancillary_dim,
            self.embedding_dim
        )
        self.rits_b = RITS(
            self.rnn_hid_size,
            self.impute_weight,
            self.label_weight,
            self.data_dim,
            self.seq_len,
            self.output_dim,
            self.ancillary_dim,
            self.embedding_dim
        )

    def forward(self, data, stage='train'):
        ret_f = self.rits_f(data, 'forward', stage)
        ret_b = reverse(self.rits_b(data, 'backward', stage))

        ret = merge_ret(ret_f, ret_b)

        return ret

    def change_last_layer(self, output_dim: int):
        self.output_dim = output_dim

        self.rits_f.change_last_layer(output_dim)
        self.rits_b.change_last_layer(output_dim)



