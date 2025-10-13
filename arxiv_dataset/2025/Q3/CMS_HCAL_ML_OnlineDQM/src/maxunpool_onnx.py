import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.pooling import _MaxUnpoolNd
from torch.nn.modules.utils import _triple


class MaxUnpool3dop(Function):
    """We warp the `torch.nn.functional.max_unpool2d`
    with an extra `symbolic` method, which is needed while exporting to ONNX.
    Users should not call this function directly.
    """

    @staticmethod
    def forward(ctx, input, indices, kernel_size, stride, padding,
                output_size):
        """Forward function of MaxUnpool3dop.
        Args:
            input (Tensor): Tensor needed to upsample.
            indices (Tensor): Indices output of the previous MaxPool.
            kernel_size (Tuple): Size of the max pooling window.
            stride (Tuple): Stride of the max pooling window.
            padding (Tuple): Padding that was added to the input.
            output_size (List or Tuple): The shape of output tensor.
        Returns:
            Tensor: Output tensor.
        """
        return F.max_unpool3d(input, indices, kernel_size, stride, padding,
                              output_size)

    @staticmethod
    def symbolic(g, input, indices, kernel_size, stride, padding, output_size):
        # get shape
        input_shape = g.op('Shape', input)
        const_0 = g.op('Constant', value_t=torch.tensor(0))
        const_1 = g.op('Constant', value_t=torch.tensor(1))
        batch_size = g.op('Gather', input_shape, const_0, axis_i=0)
        channel = g.op('Gather', input_shape, const_1, axis_i=0)
        
        # depth = (depth - 1) * stride + kernel_size
        depth = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(1)),
            axis_i=0)
        depth = g.op('Sub', depth, const_1)
        depth = g.op('Mul', depth,
                     g.op('Constant', value_t=torch.tensor(stride[2])))
        depth = g.op('Add', depth,

                     g.op('Constant', value_t=torch.tensor(kernel_size[2])))

        # height = (height - 1) * stride + kernel_size
        height = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(2)),
            axis_i=0)
        height = g.op('Sub', height, const_1)
        height = g.op('Mul', height,
                      g.op('Constant', value_t=torch.tensor(stride[1])))
        height = g.op('Add', height,
                      g.op('Constant', value_t=torch.tensor(kernel_size[1])))

        # width = (width - 1) * stride + kernel_size
        width = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(3)),
            axis_i=0)
        width = g.op('Sub', width, const_1)
        width = g.op('Mul', width,
                     g.op('Constant', value_t=torch.tensor(stride[0])))
        width = g.op('Add', width,
                     g.op('Constant', value_t=torch.tensor(kernel_size[0])))


        # step of channel
        channel_step = g.op('Mul', height, width)
        channel_step = g.op('Mul', channel_step, depth)
        
        # step of batch
        batch_step = g.op('Mul', channel_step, channel)

        # print(channel_step, batch_step)
        # channel offset
        range_channel = g.op('Range', const_0, channel, const_1)
        range_channel = g.op(
            'Reshape', range_channel,
            g.op('Constant', value_t=torch.tensor([1, -1, 1, 1])))
        range_channel = g.op('Mul', range_channel, channel_step)
        range_channel = g.op('Cast', range_channel, to_i=7)  # 7 is int64

        # print(range_channel)
        # batch offset
        range_batch = g.op('Range', const_0, batch_size, const_1)
        range_batch = g.op(
            'Reshape', range_batch,
            g.op('Constant', value_t=torch.tensor([-1, 1, 1, 1])))
        range_batch = g.op('Mul', range_batch, batch_step)
        range_batch = g.op('Cast', range_batch, to_i=7)  # 7 is int64

        # print(range_batch)

        # update indices
        indices = g.op('Add', indices, range_channel)
        indices = g.op('Add', indices, range_batch)

        return g.op(
            'MaxUnpool',
            input,
            indices,
            kernel_shape_i=kernel_size,
            strides_i=stride)

class MaxUnpool3d(_MaxUnpoolNd):
    """This module is modified from Pytorch `MaxUnpool2d` module.
    Args:
      kernel_size (int or tuple): Size of the max pooling window.
      stride (int or tuple): Stride of the max pooling window.
          Default: None (It is set to `kernel_size` by default).
      padding (int or tuple): Padding that is added to the input.
          Default: 0.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride or kernel_size)
        self.padding = _triple(padding)

    def forward(self, input, indices, output_size=None):
        """Forward function of MaxUnpool2d.
        Args:
            input (Tensor): Tensor needed to upsample.
            indices (Tensor): Indices output of the previous MaxPool.
            output_size (List or Tuple): The shape of output tensor.
                Default: None.
        Returns:
            Tensor: Output tensor.
        """
        return MaxUnpool3dop.apply(input, indices, self.kernel_size,
                                   self.stride, self.padding, output_size)


'''
# test

import maxunpool_onnx

# 3d: [BxCxd1xd2xd3]

# example
x = torch.randn(20, 16, 51, 33, 15) 
x.size()

# pooling
pool = nn.MaxPool3d(3, stride=2, return_indices=True)
output, indices = pool(x)

# unpooling
unpool_onnx = maxunpool_onnx.MaxUnpool3d(3, stride=2)
unpooled_output_onnx = unpool_onnx(output, indices)

unpooled_output_onnx.size()

'''
