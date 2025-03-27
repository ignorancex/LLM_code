import torch
from torch.autograd import Function
from torch import nn
import math


class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        # print(x)
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T)  # batchSize * N
        # print(out)

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        # print(x)
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()

        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        # print(weight_pos)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        # print(x.data)
        # print('*'*60)
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None


class LinearAverage(nn.Module):

    def __init__(self, mb_feature, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params', torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        if mb_feature.shape[0] > 0:
            self.register_buffer('memory', mb_feature)
        else:
            self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        # print(x)
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

