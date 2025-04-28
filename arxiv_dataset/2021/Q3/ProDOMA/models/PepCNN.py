import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class PepCNNDeep(nn.Module):

    def __init__(self, args, num_token=21):
        super(PepCNNDeep, self).__init__()

        self.num_token = num_token
        self.seq_len = args.seq_len
        self.num_class = args.num_classes
        self.channel_in = 3
        self.kernel_nums = [args.num_filters] * len(args.filter_sizes)
        self.kernel_sizes = args.filter_sizes
        self.dropout_rate = args.dropout

        self.convs1 = nn.Conv2d(self.channel_in, 256, (args.conv1_filter, self.num_token))
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(256, self.kernel_nums[i], (kernel_size, 1)) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.kernel_nums), args.num_hidden)
        self.fc2 = nn.Linear(args.num_hidden, self.num_class)

    def forward(self, x, return_conv = False):
        x = F.relu(self.convs1(x))

        x = [F.relu(conv(x)).squeeze(3) for i, conv in enumerate(self.convs2)]

        conv = x

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        # conv = x
        logit = self.fc2(x)
        if return_conv:
            return logit, conv
        return logit


class PepCNNDeepTriple(nn.Module):

    def __init__(self, args, num_token=21):
        super(PepCNNDeepTriple, self).__init__()

        self.num_token = num_token
        self.seq_len = args.seq_len
        self.num_class = args.num_classes
        self.channel_in = 1
        self.kernel_nums = [args.num_filters] * len(args.filter_sizes)
        self.kernel_sizes = args.filter_sizes
        self.dropout_rate = args.dropout

        self.convs1 = nn.Conv2d(self.channel_in, self.kernel_nums[0], (args.conv1_filter, self.num_token))
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(self.kernel_nums[0], self.kernel_nums[i], (kernel_size, 1)) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(3 * sum(self.kernel_nums), args.num_hidden)
        self.fc2 = nn.Linear(args.num_hidden, self.num_class)

    def branch_feature(self, input):
        input = input.unsqueeze(1)    
        input = F.relu(self.convs1(input))
        input = [F.relu(conv(input)).squeeze(3) for i, conv in enumerate(self.convs2)]
        input = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in input]  # [(N, Co), ...]*len(Ks)
        input = torch.cat(input, 1)
        return input

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 0, :, :]
        x3 = x[:, 0, :, :]

        x1 = self.branch_feature(x1)
        x2 = self.branch_feature(x2)
        x3 = self.branch_feature(x3)

        x = torch.cat([x1, x2, x3], 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        logit = self.fc2(x)
        return logit

class PepCNNDeep2Layer(nn.Module):

    def __init__(self, args, num_token=21):
        super(PepCNNDeep2Layer, self).__init__()

        self.num_token = num_token
        self.seq_len = args.seq_len
        self.num_class = args.num_classes
        self.channel_in = 3
        self.kernel_nums = [args.num_filters] * len(args.filter_sizes)
        self.kernel_sizes = args.filter_sizes
        self.dropout_rate = args.dropout

        self.convs1 = nn.Conv2d(self.channel_in, 64, (args.conv1_filter, self.num_token))
        self.convs2 = nn.Conv2d(64, 64, (args.conv1_filter, 1))
        self.convs3 = nn.ModuleList(
            [nn.Conv2d(64, self.kernel_nums[i], (kernel_size, 1)) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.kernel_nums), args.num_hidden)
        self.fc2 = nn.Linear(args.num_hidden, self.num_class)

    def forward(self, x):
        x = F.relu(self.convs1(x))

        x = F.relu(self.convs2(x))

        x = [F.relu(conv(x)).squeeze(3) for i, conv in enumerate(self.convs3)]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        logit = self.fc2(x)
        return logit

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class PepCNNDeep2LayerBN(nn.Module):

    def __init__(self, args, num_token=21):
        super(PepCNNDeep2LayerBN, self).__init__()

        self.num_token = num_token
        self.seq_len = args.seq_len
        self.num_class = args.num_classes
        self.channel_in = 3
        self.kernel_nums = [args.num_filters] * len(args.filter_sizes)
        self.kernel_sizes = args.filter_sizes
        self.dropout_rate = args.dropout

        self.convs1 = BasicConv2d(self.channel_in, self.kernel_nums[0], kernel_size=(args.conv1_filter, self.num_token))
        self.convs2 = BasicConv2d(self.kernel_nums[0], self.kernel_nums[0], kernel_size=(args.conv1_filter, 1))
        self.convs3 = nn.ModuleList(
            [BasicConv2d(self.kernel_nums[0], self.kernel_nums[i], kernel_size=(kernel_size, 1)) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.kernel_nums), args.num_hidden)
        self.fc2 = nn.Linear(args.num_hidden, self.num_class)

    def forward(self, x):
        x = self.convs1(x)

        x = self.convs2(x)

        x = [conv(x).squeeze(3) for i, conv in enumerate(self.convs3)]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        logit = self.fc2(x)
        return logit

class PepCNNDeepBN(nn.Module):

    def __init__(self, args, num_token=21):
        super(PepCNNDeepBN, self).__init__()

        self.num_token = num_token
        self.seq_len = args.seq_len
        self.num_class = args.num_classes
        self.channel_in = 3
        self.kernel_nums = [args.num_filters] * len(args.filter_sizes)
        self.kernel_sizes = args.filter_sizes
        self.dropout_rate = args.dropout

        self.convs1 = BasicConv2d(self.channel_in, self.kernel_nums[0], kernel_size=(args.conv1_filter, self.num_token))
        self.convs2 = nn.ModuleList(
            [BasicConv2d(self.kernel_nums[0], self.kernel_nums[i], kernel_size=(kernel_size, 1)) for i, kernel_size in
             enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.kernel_nums), args.num_hidden)
        self.fc2 = nn.Linear(args.num_hidden, self.num_class)

    def forward(self, x):
        x = self.convs1(x)

        x = [conv(x).squeeze(3) for i, conv in enumerate(self.convs2)]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        logit = self.fc2(x)
        return logit
